"""ADR-019 (Stage 2 persistence): the alert-history store.

Plain stdlib `sqlite3`, WAL mode, `CREATE TABLE IF NOT EXISTS` -- no ORM,
no migration framework, per ADR-019's "smallest real step up" reasoning.
`log_alert`/`fetch_alerts` are synchronous, blocking functions, meant to be
called via `asyncio.to_thread` from the async serving layer (`api.py`),
the same established pattern `_predict_cumulative_incidence_sync`/
`_build_alert_prompt_sync` already use for PyTorch/Captum work.

Deliberately stored OUTSIDE `data/raw/` -- that directory is documented
throughout this project as a cache of EXTERNAL data (StatsBomb matches,
SoccerNet clips, football-data.co.uk CSVs, CV video). This database is
locally-generated application state, a categorically different kind of
thing; mixing the two would blur a boundary this project keeps clean
everywhere else. See ADR-019 for the full reasoning (WAL mode, SQLite vs.
Postgres, why no aiosqlite/ORM).
"""

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_DIR = Path("data/app_state")
DB_PATH = DB_DIR / "alerts.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at_utc TEXT NOT NULL,
    source TEXT NOT NULL,
    match_id INTEGER,
    video_path TEXT,
    minute REAL,
    threat_before REAL NOT NULL,
    threat_after REAL NOT NULL,
    delta REAL NOT NULL,
    explanation_text TEXT NOT NULL,
    explanation_source TEXT NOT NULL
)
"""
_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_alerts_match_id ON alerts(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_source ON alerts(source)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_logged_at_utc ON alerts(logged_at_utc)",
)

# Milliseconds. ADR-019: Milestone 16's own concurrent-WebSocket-connection
# tests mean concurrent writers here are a real, already-tested scenario,
# not a hypothetical -- this is what lets a second concurrent writer wait
# briefly under real contention instead of immediately raising
# `sqlite3.OperationalError: database is locked`. Generous relative to how
# short a single INSERT is, tiny relative to a human reviewing history.
_BUSY_TIMEOUT_MS = 5000


def _get_connection() -> sqlite3.Connection:
    """A FRESH connection per call -- `sqlite3.Connection` objects are not
    safe to share across threads without `check_same_thread=False`, and
    both `log_alert` and `fetch_alerts` are always invoked via
    `asyncio.to_thread`, i.e. a different worker thread on every call.
    WAL mode is requested on every connection (idempotent once the
    database file itself is in WAL mode -- a fast no-op confirming it,
    never a behavior change on repeat calls) rather than assumed already
    active, since a fresh or rotated db file would otherwise default back
    to SQLite's normal rollback-journal mode.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=_BUSY_TIMEOUT_MS / 1000.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn


def init_db() -> None:
    """Schema init -- `CREATE TABLE`/`INDEX IF NOT EXISTS` only, no
    migration framework (ADR-019: intentionally minimal). Safe to call
    repeatedly (once from `api.py`'s `lifespan`, and defensively again
    from every `log_alert`/`fetch_alerts` call -- see there) and safe
    under concurrent first-time initialization: `IF NOT EXISTS` is itself
    safe under SQLite's own locking, so a second concurrent `CREATE TABLE`
    simply finds the table already there rather than racing.
    """
    conn = _get_connection()
    try:
        conn.execute(_SCHEMA_SQL)
        for index_sql in _INDEX_SQL:
            conn.execute(index_sql)
        conn.commit()
    finally:
        conn.close()


def log_alert(
    *,
    source: str,
    match_id: int | None,
    video_path: str | None,
    minute: float | None,
    threat_before: float,
    threat_after: float,
    explanation_text: str,
    explanation_source: str,
) -> None:
    """Synchronous, blocking -- call via `asyncio.to_thread`, never
    directly on the event loop.

    NEVER raises: any failure (disk full, a lock timeout past
    `_BUSY_TIMEOUT_MS`, a corrupt db file, anything else) is caught and
    logged as a `logging.warning`, then swallowed. ADR-019: the real-time
    alert must reach the client regardless of whether this companion write
    succeeds -- persistence is strictly additive, never a dependency of
    the live alert flow.
    """
    try:
        init_db()  # idempotent; defends direct/test callers bypassing api.py's lifespan
        conn = _get_connection()
        try:
            conn.execute(
                """
                INSERT INTO alerts (
                    logged_at_utc, source, match_id, video_path, minute,
                    threat_before, threat_after, delta,
                    explanation_text, explanation_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    source,
                    match_id,
                    video_path,
                    minute,
                    threat_before,
                    threat_after,
                    threat_after - threat_before,
                    explanation_text,
                    explanation_source,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        logger.warning(f"[alert_store] Failed to persist alert history entry: {exc!r}")


def fetch_alerts(
    *,
    match_id: int | None = None,
    source: str | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """Read path for `GET /alerts/history`. Every filter is optional and
    AND-combined; `start_utc`/`end_utc` compare directly against the
    stored ISO-8601 `logged_at_utc` strings (these sort correctly as plain
    text, so no date parsing is needed here). Returns most-recent-first.
    """
    init_db()
    conn = _get_connection()
    try:
        conn.row_factory = sqlite3.Row
        clauses: list[str] = []
        params: list = []
        if match_id is not None:
            clauses.append("match_id = ?")
            params.append(match_id)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if start_utc is not None:
            clauses.append("logged_at_utc >= ?")
            params.append(start_utc)
        if end_utc is not None:
            clauses.append("logged_at_utc <= ?")
            params.append(end_utc)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            # `where_sql` is built only from the fixed clause strings above
            # (never from raw user input); every actual value is passed as
            # a bound parameter below, never interpolated into the SQL text.
            f"SELECT * FROM alerts {where_sql} ORDER BY logged_at_utc DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
