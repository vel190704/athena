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


# ============================================================================
# SQL injection proof-of-safety (production-readiness audit, Priority 0).
# The audit's own exhaustive re-search (every `.execute(`/`sqlite3`/`SELECT`/
# `INSERT` call site in `production/`, not just this file) found `fetch_alerts`
# already parameterized correctly: `where_sql` is built only from a FIXED
# whitelist of clause strings ("match_id = ?", "source = ?", ...), never from
# a raw value, and every actual value is bound via `?`. No fix was needed --
# these tests exist to PROVE that directly, with a real triggered malicious
# payload, rather than rest on code-review confirmation alone. Exercises the
# persistence layer itself (not FastAPI's own `Literal["statsbomb", "cv"]`
# typing on `source`, which is a separate, shallower defense at the API layer
# -- see test_api.py's own endpoint-level version of this same check) so this
# proves the INNERMOST layer is safe regardless of what validation exists
# above it.
# ============================================================================


def test_fetch_alerts_sql_injection_payload_in_source_is_inert_literal_not_executed():
    """A malicious `source` value, passed directly to `fetch_alerts` (below
    FastAPI's own `Literal` type check, which would reject this before it
    ever reached here in the real request path) -- must be treated as an
    inert string to compare against the `source` column, never as executable
    SQL. Proven by: (1) no exception: a real DROP TABLE, if it executed,
    would leave the table gone and the SECOND query below would raise
    `sqlite3.OperationalError: no such table: alerts`, not return cleanly;
    (2) the seeded real rows are still all there afterward."""
    alert_store.log_alert(
        source="statsbomb", match_id=1, video_path=None, minute=1.0,
        threat_before=0.05, threat_after=0.10, explanation_text="real alert", explanation_source="mock",
    )

    payload = "cv'; DROP TABLE alerts; --"
    result = alert_store.fetch_alerts(source=payload)
    assert result == [], "the malicious value must not match the real 'statsbomb' row -- inert literal comparison"

    # The table must still exist and still contain the real row -- proves
    # the injected `DROP TABLE` was never executed, not merely that this
    # one call didn't crash.
    survivors = alert_store.fetch_alerts()
    assert len(survivors) == 1
    assert survivors[0]["explanation_text"] == "real alert"


def test_fetch_alerts_sql_injection_payload_in_date_range_is_inert_literal_not_executed():
    """Same proof, for `start_utc`/`end_utc` -- the two genuinely free-text
    fields a real caller controls via `GET /alerts/history`'s own query
    params (see test_api.py for the full HTTP-level version)."""
    alert_store.log_alert(
        source="statsbomb", match_id=2, video_path=None, minute=2.0,
        threat_before=0.05, threat_after=0.10, explanation_text="real alert 2", explanation_source="mock",
    )

    payload = "2020-01-01T00:00:00+00:00'; DROP TABLE alerts; --"
    result = alert_store.fetch_alerts(start_utc=payload, end_utc="2099-01-01T00:00:00+00:00")
    # A plain string comparison (`logged_at_utc >= payload`) -- not asserting
    # which side of a lexicographic string compare it lands on, only that it
    # executed as ONE inert comparison, not as injected SQL.
    assert isinstance(result, list)

    survivors = alert_store.fetch_alerts()
    assert len(survivors) == 1
    assert survivors[0]["explanation_text"] == "real alert 2"


def test_log_alert_sql_injection_payload_in_text_fields_is_inert_literal_not_executed():
    """The write path (`log_alert`'s own INSERT): a malicious payload in
    `explanation_text`/`video_path` must be stored verbatim as data, never
    executed, and must round-trip byte-for-byte on read -- the strongest
    possible proof it was never interpreted as SQL along the way."""
    payload = "'; DROP TABLE alerts; --"
    alert_store.log_alert(
        source="cv", match_id=None, video_path=payload, minute=3.0,
        threat_before=0.05, threat_after=0.10, explanation_text=payload, explanation_source="mock",
    )

    rows = alert_store.fetch_alerts()
    assert len(rows) == 1
    assert rows[0]["explanation_text"] == payload
    assert rows[0]["video_path"] == payload


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
