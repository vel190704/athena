"""Production-readiness audit, Priority 3.1: a simple, periodic backup for
`data/app_state/alerts.db` -- the only genuinely stateful, hard-to-regenerate
data in this system (an alert fired live cannot be recomputed after the
fact; see ADR-019).

Proportional to this project's actual scale (a single SQLite file, single
operator, non-commercial) -- deliberately NOT enterprise-grade
infrastructure: no replication, no point-in-time recovery, no offsite/cloud
sync. Uses `sqlite3.Connection.backup()`, the stdlib's own ONLINE backup
API -- safe to run against a live database in WAL mode with the server
still running and writing, unlike a naive `cp`/`shutil.copy` (which can
copy a half-written page mid-write and produce a torn, unusable snapshot;
this project's own alert_store.py already established that concurrent
writes are a real, not hypothetical, scenario here -- see ADR-019).

Usage: run manually (`python scripts/backup_alerts_db.py`), or on a
schedule -- see README.md's "Production Readiness" section for a suggested
cron/systemd-timer schedule and the real, plainly-stated consequence of
skipping this (total loss of alert history on disk failure, with no other
copy anywhere).
"""

import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

SOURCE_DB = Path("data/app_state/alerts.db")
BACKUP_DIR = Path("data/app_state/backups")

# ~2 weeks at one backup/day -- proportional, bounded retention, not
# unbounded growth and not a single point-in-time snapshot either.
MAX_BACKUPS_TO_KEEP = 14


def backup_alerts_db() -> Path | None:
    """Returns the new backup's path, or `None` if there was nothing to
    back up yet (a fresh install, or a server that has never logged an
    alert -- not an error)."""
    if not SOURCE_DB.exists():
        print(f"No database at {SOURCE_DB} -- nothing to back up yet.")
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = BACKUP_DIR / f"alerts_{timestamp}.db"

    source_conn = sqlite3.connect(SOURCE_DB)
    try:
        dest_conn = sqlite3.connect(backup_path)
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        source_conn.close()

    print(f"Backed up {SOURCE_DB} -> {backup_path}")
    _prune_old_backups()
    return backup_path


def _prune_old_backups() -> None:
    existing = sorted(BACKUP_DIR.glob("alerts_*.db"))
    for old_backup in existing[: -MAX_BACKUPS_TO_KEEP if MAX_BACKUPS_TO_KEEP > 0 else None]:
        old_backup.unlink()
        print(f"Removed old backup (retention cap {MAX_BACKUPS_TO_KEEP}): {old_backup}")


if __name__ == "__main__":
    backup_alerts_db()
    sys.exit(0)
