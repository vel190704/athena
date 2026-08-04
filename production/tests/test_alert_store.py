"""ADR-019 (Stage 2 persistence) validation: alert_store.py -- the
SQLite-backed alert-history store.

Every test isolates itself into pytest's own `tmp_path` (monkeypatching
`DB_DIR`/`DB_PATH`) so nothing here ever touches the real
`data/app_state/alerts.db` a running server would use, matching
`test_mlflow.py`'s own isolated-tracking-URI discipline for the same
reason (real, on-disk state should not leak between test runs).
"""

import concurrent.futures
import logging

import pytest

from production.src.serving import alert_store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_store, "DB_DIR", tmp_path / "app_state")
    monkeypatch.setattr(alert_store, "DB_PATH", tmp_path / "app_state" / "alerts.db")


def _make_alert(i: int) -> dict:
    return {
        "source": "statsbomb" if i % 2 == 0 else "cv",
        "match_id": 1000 + i if i % 2 == 0 else None,
        "video_path": None if i % 2 == 0 else f"data/raw/video_{i}.mp4",
        "minute": float(i),
        "threat_before": 0.05,
        "threat_after": 0.05 + 0.01 * i,
        "explanation_text": f"alert number {i}",
        "explanation_source": "mock" if i % 2 == 0 else "gemini",
    }


def test_concurrent_writes_no_corruption_no_lost_writes():
    """The single most important test in ADR-019. This project's own
    Milestone 16 per-connection concurrency testing
    (`test_per_connection_spike_state_is_isolated`,
    `test_cv_source_per_connection_state_isolation`) already establishes
    that multiple simultaneous WebSocket connections can each
    independently fire their own spike alert at any time -- so multiple
    concurrent `log_alert()` calls hitting the SAME database file is a
    real scenario this implementation must handle correctly today, not a
    hypothetical future-scale concern. Mirrors that same
    many-simultaneous-operations pattern, applied here to writes: every
    one of N concurrent inserts must survive, unmodified, with none lost
    and none corrupting another -- exactly what WAL mode + a busy_timeout
    are for.
    """
    n = 40
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(alert_store.log_alert, **_make_alert(i)) for i in range(n)]
        for future in futures:
            future.result(timeout=30)  # re-raises here if log_alert somehow raised

    rows = alert_store.fetch_alerts(limit=n + 10)
    assert len(rows) == n, f"expected {n} rows, found {len(rows)} -- a concurrent write was lost"

    texts = {row["explanation_text"] for row in rows}
    assert texts == {f"alert number {i}" for i in range(n)}, (
        "every concurrent write's own content must be intact and distinct -- "
        "not corrupted or overwritten by another concurrent writer"
    )

    # Spot-check a few rows' non-text fields survived correctly too (not
    # just the text, which alone wouldn't catch e.g. two rows' numeric
    # fields getting swapped under a genuine corruption bug).
    by_text = {row["explanation_text"]: row for row in rows}
    assert by_text["alert number 4"]["source"] == "statsbomb"
    assert by_text["alert number 4"]["match_id"] == 1004
    assert by_text["alert number 5"]["source"] == "cv"
    assert by_text["alert number 5"]["video_path"] == "data/raw/video_5.mp4"


def test_write_failure_logs_warning_and_does_not_raise(tmp_path, monkeypatch, caplog):
    """Simulates a genuinely unwritable DB path -- a FILE sitting where
    `DB_DIR` needs to be a directory, so `DB_DIR.mkdir(parents=True,
    exist_ok=True)` raises `FileExistsError` -- and confirms `log_alert`
    swallows the failure, logs a `logging.warning`, and never raises. This
    is the guarantee ADR-019's "persistence must never block the real
    alert" requirement rests on; `api.py`'s own equivalent case (the
    real-time WebSocket alert still sending despite a logging failure) is
    covered separately in `test_api.py`.
    """
    blocking_file = tmp_path / "blocks_app_state_dir"
    blocking_file.write_text("a file, not a directory")
    monkeypatch.setattr(alert_store, "DB_DIR", blocking_file)
    monkeypatch.setattr(alert_store, "DB_PATH", blocking_file / "alerts.db")

    with caplog.at_level(logging.WARNING, logger="production.src.serving.alert_store"):
        alert_store.log_alert(**_make_alert(0))  # must NOT raise

    assert any(
        "Failed to persist alert history entry" in record.message for record in caplog.records
    ), "log_alert must log a warning on failure, not fail silently or crash"


def test_fetch_alerts_filters_correctly():
    alert_store.log_alert(
        source="statsbomb", match_id=111, video_path=None, minute=5.0,
        threat_before=0.05, threat_after=0.10, explanation_text="A", explanation_source="mock",
    )
    alert_store.log_alert(
        source="statsbomb", match_id=222, video_path=None, minute=6.0,
        threat_before=0.05, threat_after=0.10, explanation_text="B", explanation_source="gemini",
    )
    alert_store.log_alert(
        source="cv", match_id=None, video_path="data/raw/x.mp4", minute=7.0,
        threat_before=0.05, threat_after=0.10, explanation_text="C", explanation_source="mock",
    )

    by_match = alert_store.fetch_alerts(match_id=111)
    assert [row["explanation_text"] for row in by_match] == ["A"]

    by_source = alert_store.fetch_alerts(source="cv")
    assert [row["explanation_text"] for row in by_source] == ["C"]

    all_rows = alert_store.fetch_alerts()
    assert len(all_rows) == 3
    # Most-recent-first: "C" was logged last.
    assert all_rows[0]["explanation_text"] == "C"


def test_init_db_is_idempotent_and_wal_mode_is_active():
    """CREATE TABLE/INDEX IF NOT EXISTS must tolerate being called
    repeatedly (init_db is called from api.py's lifespan AND defensively
    from every log_alert/fetch_alerts call) without error, and WAL mode --
    the whole point of ADR-019's concurrency guarantee -- must actually be
    the active journal mode, not just requested."""
    alert_store.init_db()
    alert_store.init_db()  # must not raise on a second call

    conn = alert_store._get_connection()
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"
