"""Production-readiness audit, Priority 3.1: `scripts/backup_alerts_db.py`.

Isolates `alert_store.py`'s own `DB_DIR`/`DB_PATH` AND `backup_alerts_db.py`'s
own `SOURCE_DB`/`BACKUP_DIR` into pytest's `tmp_path`, matching
`test_alert_store.py`'s own established isolation discipline -- nothing here
ever touches the real `data/app_state/alerts.db` or its real backups.
"""

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from production.src.serving import alert_store

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backup_alerts_db.py"
_spec = importlib.util.spec_from_file_location("backup_alerts_db", _SCRIPT_PATH)
backup_alerts_db_module = importlib.util.module_from_spec(_spec)
sys.modules["backup_alerts_db"] = backup_alerts_db_module
_spec.loader.exec_module(backup_alerts_db_module)


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_store, "DB_DIR", tmp_path / "app_state")
    monkeypatch.setattr(alert_store, "DB_PATH", tmp_path / "app_state" / "alerts.db")
    monkeypatch.setattr(backup_alerts_db_module, "SOURCE_DB", tmp_path / "app_state" / "alerts.db")
    monkeypatch.setattr(backup_alerts_db_module, "BACKUP_DIR", tmp_path / "app_state" / "backups")


def test_backup_with_no_source_db_is_a_graceful_no_op():
    result = backup_alerts_db_module.backup_alerts_db()
    assert result is None
    assert not backup_alerts_db_module.BACKUP_DIR.exists()


def test_backup_produces_a_real_independently_openable_copy_with_same_rows():
    alert_store.log_alert(
        source="statsbomb", match_id=1, video_path=None, minute=1.0,
        threat_before=0.05, threat_after=0.20, explanation_text="real alert", explanation_source="mock",
    )
    alert_store.log_alert(
        source="cv", match_id=None, video_path="data/raw/clip.mp4", minute=2.0,
        threat_before=0.05, threat_after=0.30, explanation_text="real alert 2", explanation_source="gemini",
    )

    backup_path = backup_alerts_db_module.backup_alerts_db()

    assert backup_path is not None
    assert backup_path.exists()
    assert backup_path != backup_alerts_db_module.SOURCE_DB

    # Independently open the BACKUP file (not alert_store's own connection
    # machinery) to prove it's a real, standalone, valid SQLite database --
    # not just a byte-copy that happens to exist.
    conn = sqlite3.connect(backup_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM alerts ORDER BY id").fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    assert {row["explanation_text"] for row in rows} == {"real alert", "real alert 2"}


def test_backup_is_a_real_point_in_time_snapshot_not_a_live_reference():
    alert_store.log_alert(
        source="statsbomb", match_id=1, video_path=None, minute=1.0,
        threat_before=0.05, threat_after=0.20, explanation_text="before backup", explanation_source="mock",
    )
    backup_path = backup_alerts_db_module.backup_alerts_db()

    # A write AFTER the backup must NOT appear in the already-taken snapshot --
    # proves this is a real copy, not a live view/symlink into the same file.
    alert_store.log_alert(
        source="statsbomb", match_id=2, video_path=None, minute=2.0,
        threat_before=0.05, threat_after=0.20, explanation_text="after backup", explanation_source="mock",
    )

    conn = sqlite3.connect(backup_path)
    try:
        texts = {row[0] for row in conn.execute("SELECT explanation_text FROM alerts").fetchall()}
    finally:
        conn.close()
    assert texts == {"before backup"}


def test_backup_retention_prunes_oldest_beyond_cap(monkeypatch):
    monkeypatch.setattr(backup_alerts_db_module, "MAX_BACKUPS_TO_KEEP", 3)
    alert_store.log_alert(
        source="statsbomb", match_id=1, video_path=None, minute=1.0,
        threat_before=0.05, threat_after=0.20, explanation_text="a", explanation_source="mock",
    )

    backup_paths = []
    for i in range(5):
        # Distinct filenames even at second-level timestamp resolution --
        # this test isolates _prune_old_backups's own logic directly rather
        # than depending on real wall-clock spacing between backup() calls.
        path = backup_alerts_db_module.BACKUP_DIR / f"alerts_2026010{i}T000000Z.db"
        backup_alerts_db_module.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake db content")
        backup_paths.append(path)

    backup_alerts_db_module._prune_old_backups()

    remaining = sorted(backup_alerts_db_module.BACKUP_DIR.glob("alerts_*.db"))
    assert len(remaining) == 3
    # The 3 NEWEST (highest-sorting filenames) survive; the 2 oldest are gone.
    assert remaining == backup_paths[-3:]
    for old_path in backup_paths[:-3]:
        assert not old_path.exists()
