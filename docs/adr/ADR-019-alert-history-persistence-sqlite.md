# ADR-019: Alert History Persistence via SQLite (Stage 2)

## Status
Accepted

## Context

Milestone 16's spike-alert flow (`production/src/serving/api.py`, extended
by Milestone 33's CV source) works like this today: `_maybe_trigger_spike_alert`
detects a threat spike (a >5-percentage-point jump in `threat_15s` between
consecutive frames) and fires `_run_alert_pipeline` via `asyncio.create_task`
— decoupled from the main per-frame `threat` stream so a slow explanation
never delays it. `_run_alert_pipeline` computes Integrated Gradients
attributions, builds a prompt, calls `generate_tactical_explanation` (real
Gemini Flash-Lite or the mock, per ADR-006's Update section), and sends the
result as a single `{"type": "alert", "explanation": ...}` WebSocket
message. The moment that message is sent, the alert is gone — there is no
code path anywhere that writes it down. A user watching the dashboard sees
it once, live; nobody can come back after the match and ask "what alerts
fired, and when."

ADR-018 deliberately did NOT introduce a database, reasoning that nothing
in this project's reporting-consolidation problem was a persistence
problem — every report there is a read-only, on-demand recomputation over
data that already lives somewhere (MLflow, `data/raw/`). This is different.
An alert is generated exactly once, at a specific moment, from live
inference — if it isn't captured the instant it fires, it cannot be
recomputed later; the moment is gone. A recent engineering review
identified this as the first genuinely write-worthy need this project has
had: "alert history logging, so a match's alerts can be reviewed
afterward" — named here explicitly as the trigger ADR-018 said would be
needed before persistence made sense, not as a premature addition made
without one.

## Decision

**Persist every alert that fires, in SQLite**, via a new
`production/src/serving/alert_store.py`.

**Why SQLite, not Postgres:** the same reasoning ADR-018 already
established for consolidation applies here to storage: file-based, zero
operational overhead (no server process, no connection pooling, no schema
migration tool), and it matches this project's existing single-machine
deployment story exactly — `mlruns/` and `data/raw/` are already local
files this project depends on; one more local file is a natural fit, a new
network service is not. Postgres would solve a multi-writer PRODUCTION
contention problem this project does not have.

**Why stdlib `sqlite3`, not `aiosqlite`:** the task establishing this ADR
itself frames `log_alert()` as "a function callable via `asyncio.to_thread`
— consistent with this project's established pattern for wrapping blocking
calls in the async serving layer," exactly how `_predict_cumulative_incidence_sync`
and `_build_alert_prompt_sync` already wrap PyTorch/Captum work. `aiosqlite`
exists specifically to AVOID needing `asyncio.to_thread` (it runs a
background thread internally and exposes an async interface over it) —
combining `aiosqlite` with `asyncio.to_thread` would wrap an
already-async-wrapping library in a second thread-executor layer, solving
nothing and adding a dependency for no benefit. Plain `sqlite3`, called
synchronously and wrapped in `asyncio.to_thread` exactly like every other
blocking call in this file, is simpler and fits the established pattern
directly. Neither SQLAlchemy nor any ORM is used, per the review's own
"smallest real step up" framing — a single `CREATE TABLE IF NOT EXISTS`
and hand-written SQL is genuinely all ten columns and three query filters
need.

**WAL mode is mandatory, not a tuning knob.** This project's own
Milestone 16 concurrency testing (`test_per_connection_spike_state_is_isolated`,
`test_cv_source_per_connection_state_isolation`) already exercises and
relies on multiple simultaneous WebSocket connections, each independently
capable of firing its own spike alert at any time. That means concurrent
writes to `alerts.db` are a REAL possibility today, on a single machine,
under this project's own existing test discipline — not a hypothetical
future-scale concern. SQLite's default rollback-journal mode serializes
writers in a way that can make one connection's write block (or, under
enough contention, fail with `database is locked`) while another holds the
lock; WAL mode lets readers proceed concurrently with a writer and, paired
with a `busy_timeout`, lets concurrent writers queue safely instead of
erroring. This is exactly the scenario this project's own existing tests
already create — the implementation must handle it correctly at THIS
stage, not defer it to some later Stage 3, because concurrent writes are
already possible with the concurrency this codebase already has.

**Schema** (`alerts` table, `CREATE TABLE IF NOT EXISTS`, no migration
framework — intentionally minimal per the review's framing): logged
timestamp (UTC), source (`statsbomb`/`cv`), `match_id` (nullable — only
set for `source="statsbomb"`), `video_path` (nullable — only set for
`source="cv"`), minute, `threat_before`, `threat_after`, `delta`,
`explanation_text`, `explanation_source` (`mock`/`gemini`).

**`explanation_source` required a small, disclosed addition to
`explainer.py`.** `generate_tactical_explanation(prompt) -> str` does not
expose which executor actually produced its output — a caller cannot
correctly infer this just by checking whether `GEMINI_API_KEY` is set,
because the key being present does not mean the real call succeeded; any
failure there silently falls back to the mock. Guessing "gemini" whenever
the key happens to be configured would silently mislabel every fallback —
worse than not recording the field, given this project's own honesty
discipline elsewhere. A new `generate_tactical_explanation_with_source(prompt) -> tuple[str, str]`
was added, returning `(text, "gemini" | "mock")` accurately; the existing
`generate_tactical_explanation` is now a one-line wrapper around it that
discards the source, so every other existing call site
(`zone_explainer.py`, tests) keeps its exact prior signature and behavior
unchanged. `api.py`'s alert pipeline is the only caller that switches to
the new function.

**Persistence is strictly additive to the real-time alert flow.**
`log_alert()` is invoked via `asyncio.create_task(asyncio.to_thread(log_alert, ...))`
from inside `_run_alert_pipeline`, right alongside the existing
`websocket.send_json(...)` call — never awaited before it, so a slow or
failing disk write can never delay or block the alert a client actually
receives. The WebSocket message itself (`{"type": "alert", "explanation": ...}`)
is byte-for-byte unchanged. Any failure inside `log_alert` (disk full,
lock timeout after the busy_timeout window, a corrupt db file, anything
else) is caught, logged as a `logging.warning`, and swallowed — it never
propagates, never crashes the live alert flow, and never prevents the
real-time message from sending.

**New read endpoint**, `GET /alerts/history`, filterable by `match_id`,
`source`, and a UTC timestamp range — closes the loop the review actually
asked for. Persisting alerts nobody can query back is not meaningfully
different from not persisting them.

**Storage location: `data/app_state/alerts.db`, not `data/raw/`.**
`data/raw/` is documented throughout this project as a cache of EXTERNAL
data (StatsBomb matches, SoccerNet clips, football-data.co.uk CSVs, CV
video) — locally-generated application state (this database) is a
categorically different kind of thing, and mixing the two would blur a
boundary this project has otherwise kept clean. `data/app_state/` is added
to `.gitignore`, matching `data/raw/`'s own treatment.

## Consequences

- Every alert that fires from either the StatsBomb-replay or CV source is
  now durably recorded, queryable after the fact via `GET /alerts/history`.
- The real-time alert-sending behavior is unchanged: the WebSocket message
  shape, content, and timing are identical to before this change. This was
  verified directly (Step 4) rather than assumed.
- `_maybe_trigger_spike_alert` and `_run_alert_pipeline` now also carry
  `source`/`match_id`/`video_path`/`minute`/`previous_threat_15s` (needed
  to populate the new schema) — an internal plumbing change, not a
  behavioral one; nothing about what gets sent to a client changed.
- A new failure mode exists in principle (the disk write can fail) but it
  is fully contained: caught, logged, and never surfaced to the alert
  flow.
- `explainer.py` gained `generate_tactical_explanation_with_source` (see
  above) — a disclosed, backward-compatible addition, not a modification
  of any existing call site's behavior.
- **Stage 3 (a genuine case for Postgres) still has not arrived.** This
  remains SQLite deliberately, until a real MULTI-WRITER PRODUCTION
  scenario — multiple independent server processes writing to the same
  store, not just this project's existing single-machine concurrent
  WebSocket connections within one process — actually requires otherwise.
  That is a materially different scenario from what WAL mode + a
  busy_timeout already handle correctly here, and it has not occurred yet.

## Alternatives Considered

- **`aiosqlite`**: considered, rejected — see Decision above. It solves a
  problem (avoiding thread-blocking on the event loop) this project's own
  established `asyncio.to_thread` pattern already solves identically for
  every other blocking call in this file; adding it would mean two
  different ways of doing the same thing for no benefit.
- **SQLAlchemy / an ORM**: rejected — one table, three read filters, and
  one insert do not need a query-building/migration layer. This is
  exactly the "more machinery than genuinely needed" the review's own
  framing warned against.
- **Postgres**: rejected for the reasons ADR-018 already gave for not
  introducing a database at all, still true here: no multi-writer
  PRODUCTION workload exists. See Consequences for the explicit Stage 3
  boundary.
- **An in-memory list/buffer instead of a file**: rejected — it would not
  survive an `api.py` process restart, which defeats the entire stated
  purpose ("review a match's alert history afterward"); a restart between
  a match and a later review is a completely normal, expected case, not
  an edge case.
- **Default (rollback-journal) SQLite mode instead of WAL**: rejected —
  see Decision's WAL paragraph. This project's own existing concurrent-
  connection tests already create a real concurrent-write scenario; WAL
  plus a busy_timeout is the correct-sized fix for that, not an
  over-engineered one.

## Update: "WAL + busy_timeout Alone Is Enough" Was Not Fully True — Found, Fixed, and Verified

This ADR's original Decision claimed WAL mode "paired with a busy_timeout,
lets concurrent writers queue safely instead of erroring." This project's
own `test_concurrent_writes_no_corruption_no_lost_writes` — described in
its own docstring as "the single most important test in ADR-019" — was
built specifically to hold that claim to account. It did its job: a real
run of the full test suite surfaced a genuine failure (`expected 40 rows,
found 39`, with a logged `sqlite3.OperationalError: database is locked`),
not a hypothetical one.

**Characterization (Step 1, before any code change).** Ran the test 10
times, unmodified, in isolation: **2/10 failed (20%)** — a real,
meaningfully reproducible failure rate under this project's own existing
40-concurrent-writer test load, not a one-off fluke.

**Root cause, confirmed by reading `log_alert`'s actual implementation
before writing any fix (not assumed):** `_get_connection()` already sets
a 5000ms SQLite busy_timeout, which makes SQLite's own internal busy
handler retry a blocked write repeatedly for up to 5 seconds before
raising `OperationalError`. That part of the original claim is true. What
was missing: `log_alert` caught that error with the exact same broad
`except Exception` used for genuinely unrecoverable failures (disk full,
a corrupt db file) and immediately logged-and-swallowed it — **there was
no APPLICATION-level retry at all.** Under a genuine burst of 40
simultaneous writers, an individual writer can lose SQLite's own internal
busy-handler race against the other 39 and exhaust its 5-second budget
purely from scheduling bad luck, even though the lock clears again
shortly after — at which point the original code gave up permanently
instead of trying again. A `database is locked` error is a transient,
recoverable condition under this specific load pattern; treating it
identically to a disk-full error was the actual gap between what this ADR
claimed and what the code did.

**Fix applied (Step 2, `alert_store.py`):** `log_alert` now retries up to
`_MAX_LOCK_RETRIES` (5) additional times, with exponential backoff
(50ms/100ms/200ms/400ms/800ms), **specifically and only** for
`sqlite3.OperationalError` whose message contains "database is locked" —
confirmed via a new `_is_database_locked_error` helper, not a blanket
retry-everything change. Every other exception (the existing
`test_write_failure_logs_warning_and_does_not_raise` case: a genuinely
unwritable path) still fails immediately on the first attempt, logged and
swallowed exactly as before — this fix narrows what gets a second chance,
it does not weaken the existing "never raises, never blocks the live
alert" guarantee for anything else. This is safe to make generous with
retries specifically because `log_alert` is invoked via
`asyncio.create_task(asyncio.to_thread(...))` and is never awaited before
the real-time WebSocket alert send (see Decision above) — a slower
background write cannot delay or block what the client actually receives.

**Verification (Step 2.3), same methodology as the baseline:** re-ran the
test 30 times post-fix (10 immediately after the change, then 20 more for
statistical confidence beyond what a single 10/10 result alone would
justify, given a true 20% underlying rate has roughly an 11% chance of
producing 10/10 passes by pure luck) — **0/30 failed (0%)**. The full
`test_alert_store.py` suite (all 4 tests, including the unrecoverable-
failure and idempotency tests) still passes unchanged.

**What was explicitly NOT done, per this project's own discipline against
hiding a real gap behind a weaker test:** the test's concurrency level
(40 threads) was not reduced, no timeout was loosened to paper over the
symptom, and the test was not marked expected-to-flake. The actual
guarantee this ADR claims — no lost writes under real concurrent load —
is now genuinely, measurably true (0/30), not just less frequently false.

A secondary, non-blocking observation from this investigation, recorded
here for completeness rather than acted on: `log_alert` also calls
`init_db()` (a second connection, its own `CREATE TABLE`/`INDEX IF NOT
EXISTS` transaction) on every single invocation, not just once — under
the same 40-writer burst this doubles the number of write-lock
acquisitions actually hitting the file before the retry fix even applies.
The retry fix alone was sufficient to reach 0/30 without touching this,
so it was left as-is rather than changed speculatively; it remains a
plausible amplifying factor worth revisiting if failure-rate regressions
are ever observed again at a higher concurrency level than 40.
