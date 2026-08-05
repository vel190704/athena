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
import time
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

# ADR-019 UPDATE (real, reproducible failure found in production use of this
# test suite, not a hypothetical): SQLite's own busy_timeout above makes a
# SINGLE blocked write wait up to 5s internally before raising `database is
# locked` -- but under a genuine burst of many simultaneous writers (this
# project's own `test_concurrent_writes_no_corruption_no_lost_writes`, 40
# threads), an individual writer can still exhaust that internal 5s budget
# purely from bad luck in SQLite's retry/backoff scheduling against 39
# competing writers, even though the lock genuinely does clear soon after.
# Measured directly (10 real runs of that test, unmodified code): 2/10
# failed with exactly this `OperationalError('database is locked')`, a real
# 20% failure rate under this project's own existing concurrency load, not
# a one-off flake. `log_alert`'s prior code caught this exact transient,
# recoverable condition with the SAME broad `except Exception` used for
# genuinely unrecoverable failures (disk full, corrupt db) and never
# retried -- silently losing the write. This is an APPLICATION-level retry
# on top of SQLite's own busy_timeout, not a duplicate of it: it gives a
# write that lost the internal busy-timeout race additional independent
# attempts, each with its own fresh busy_timeout window, before finally
# giving up and logging a warning.
_MAX_LOCK_RETRIES = 5
# Exponential backoff starting at 50ms, doubling each retry (50/100/200/
# 400/800ms, ~1.55s of explicit sleep across all retries) -- short relative
# to the 5s busy_timeout each individual attempt already gets internally,
# long enough to let a losing writer fall behind the current contention
# burst before trying again rather than immediately re-joining the same
# thundering herd. `log_alert` is invoked via `asyncio.create_task(
# asyncio.to_thread(...))` and is NEVER awaited before the real-time
# WebSocket alert send (see this module's/ADR-019's own docstring) -- a
# slower background write here cannot delay or block the alert a client
# actually receives, so being generous with retries costs nothing
# user-facing.
_RETRY_BACKOFF_BASE_SECONDS = 0.05


def _is_database_locked_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "database is locked" in str(exc).lower()


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
    `_BUSY_TIMEOUT_MS` on every retry attempt, a corrupt db file, anything
    else) is caught and logged as a `logging.warning`, then swallowed.
    ADR-019: the real-time alert must reach the client regardless of
    whether this companion write succeeds -- persistence is strictly
    additive, never a dependency of the live alert flow.

    RETRY, ADR-019 UPDATE: a `database is locked` `OperationalError` is a
    transient, recoverable condition under real concurrent-writer load
    (measured, not assumed -- see `_MAX_LOCK_RETRIES`'s own comment), not
    the same class of failure as a disk-full or corrupt-db error. This is
    retried up to `_MAX_LOCK_RETRIES` times with exponential backoff
    before being treated the same way those genuinely unrecoverable
    failures are (logged and swallowed). Any OTHER exception is still
    caught and swallowed immediately, on the first attempt, exactly as
    before this change -- only the specific, transient lock error gets
    retried.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_LOCK_RETRIES + 1):
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
            return  # success -- no retry needed
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
            last_exc = exc
            if _is_database_locked_error(exc) and attempt < _MAX_LOCK_RETRIES:
                time.sleep(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt))
                continue
            break

    logger.warning(f"[alert_store] Failed to persist alert history entry: {last_exc!r}")


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
