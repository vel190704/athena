# ADR-022: A Single, Optional API-Key Check — Not a Full Auth System

## Status
Accepted

## Context

Two separate engineering reviews flagged the same gap and neither was
ever acted on: `production/src/serving/api.py`'s live WebSocket/REST
serving layer has no authentication or rate-limiting of any kind.
`docs/FULL_PROJECT_REPORT.md` §11 records this plainly: "Auth/rate-
limiting on the live serving layer (`production/src/serving/api.py`) —
untouched, out of scope for this engineering-hygiene pass." `README.md`'s
own "Achieved vs. Not Yet" table repeats it. Flagged twice, addressed
neither time — this ADR closes that out with an explicit decision, not
another deferral.

This project has always run as a single-user, single-machine, local
research tool (`uvicorn ...:app --reload` + `streamlit run
production/frontend/dashboard.py`, per the README's own Quick Start).
ADR-021 introduced a `PUBLIC_DEPLOYMENT` flag for a *free, non-monetized*
public site — but "publicly reachable" and "needs user accounts" are
different problems. Nothing about this project's actual use case (one
operator, no multi-tenant data, no billing, no per-user permissions)
justifies a real auth system.

## Decision

**Add a single, optional shared-secret header check — `X-API-Key`,
checked against one value from the `API_KEY` environment variable — and
nothing more.**

- **Off by default.** `API_KEY` unset (the default, in every environment
  this project has run in so far, including its own test suite): every
  protected endpoint behaves exactly as it did before this ADR, with zero
  friction. This is not a "secure by default, opt out" design —
  deliberately the opposite, matching `PUBLIC_DEPLOYMENT`'s own
  established convention in this codebase (`os.environ.get(...)`, unset
  = today's behavior, unchanged).
- **When set**, every request to a protected endpoint must carry a
  matching `X-API-Key` header, checked with a plain exact-string
  comparison (`_require_api_key`, applied via FastAPI's
  `dependencies=[Depends(...)]` on each REST route). A missing or
  mismatched key gets a `401`.
- **`GET /health` is deliberately exempted** — a common, sensible
  convention (load balancers and uptime monitors need to probe liveness
  without a credential) and the only endpoint in this file exempted this
  way.
- **The WebSocket endpoint (`/ws/tactical-stream`) gets the same
  protection via a manual check before `accept()`**, not
  `Depends()` — FastAPI's dependency-injection wiring for
  `@app.websocket` routes doesn't apply the same way it does for REST
  routes in the FastAPI version this project is pinned to
  (`requirements-lock.txt`), so this is an explicit, separate check,
  not an oversight or a gap left uncovered.

**Why a single shared-secret header, not a full auth system (user
accounts, sessions, JWTs, OAuth):** disproportionate to this project's
actual scope. There are no per-user permissions to enforce (one
operator), no user-owned data to isolate (every report is a read-only
recomputation over the same shared StatsBomb/football-data.co.uk cache),
and no login UX anywhere in `dashboard.py` to hang a session off of. A
real auth system would add a real operational dependency (a user store,
a session/token issuance flow, password or OAuth-provider handling) to
solve a problem — distinguishing WHO is calling — this project does not
currently have. It has a narrower, real problem instead: an unauthenticated
person on the same network (or, if ever `PUBLIC_DEPLOYMENT`-hosted, on the
open internet) can hit expensive endpoints (`/reports/team/{name}`
measured up to ~100s for a well-supported team, per the Team Reports
timeout-incident fix) for free. A shared secret closes exactly that gap:
"only people who have the key can call this," which is the actual,
minimal requirement — not "only Alice can see Alice's data," which does
not apply here because there is no per-user data.

**Rate-limiting itself (request throttling, not identity) remains
explicitly UNRESOLVED by this ADR** — stated plainly, not implied to be
covered by the API-key check. A shared secret controls WHO can call an
endpoint; it does not bound HOW OFTEN a holder of that key can call it.
If this project is ever deployed somewhere a single API key's traffic
volume becomes a real cost/availability concern, request throttling
(e.g. `slowapi`, or a simple in-memory token-bucket keyed by API key)
is the natural next step — named here as the deferred half of the
original "auth/rate-limiting" flag, not silently folded into this
decision.

## Consequences

- `production/src/serving/api.py` gained: an `API_KEY` module-level flag
  (same convention as `PUBLIC_DEPLOYMENT`), a `_require_api_key`
  dependency applied to every REST endpoint except `GET /health`, a
  manual equivalent check on the WebSocket endpoint, and a startup log
  line stating plainly which mode is active.
- **Local development and this project's own test suite are unaffected.**
  `API_KEY` is never set in `production/tests/`, so every existing test
  continues to exercise the no-auth default path unchanged.
- A future multi-user or genuinely public-facing deployment (anyone
  other than this project's own operator holding the key) would need
  this decision revisited — a single shared secret gives no way to
  revoke one caller without revoking everyone, and no way to attribute
  which caller did what. That is the explicit condition under which this
  ADR's scope stops being sufficient: **the moment more than one party
  needs distinguishable access**, not before.
- Real rate-limiting (request throttling) remains a genuinely open item,
  not resolved by this ADR — see Decision above.

## Alternatives Considered

- **Full auth system (user accounts, sessions, OAuth)**: rejected as
  disproportionate to this project's current, single-operator scope —
  see Decision. Revisit only under the multi-user condition stated in
  Consequences.
- **JWT-based bearer tokens instead of a static shared secret**: rejected
  — JWTs solve token expiry/rotation and claims-based authorization,
  neither of which this project needs with exactly one real credential
  holder. A static shared secret is simpler and equally sufficient for
  "is this caller allowed to use this API at all," the actual question
  being answered here.
- **Rate-limiting instead of (or in addition to) an API key, in this same
  pass**: considered — the original review flag names both together.
  Scoped out of this ADR specifically because it is a genuinely separate
  mechanism (throttling volume vs. gating identity) with its own design
  questions (per-key vs. per-IP, what window, what response on
  exceeding it) that deserve their own deliberate pass rather than being
  bolted on here for the sake of closing both review items in one
  commit. Named explicitly in Decision/Consequences as still open,
  not silently resolved.
- **API_KEY defaulting to REQUIRED unless explicitly disabled (secure-
  by-default)**: rejected — would break every existing test and every
  local development workflow the moment this ADR landed, for a threat
  model (unauthenticated local access) this project has never actually
  had until `PUBLIC_DEPLOYMENT` existed. `PUBLIC_DEPLOYMENT`'s own
  README section already tells an operator what to configure before
  going public; adding "and set API_KEY too" to that same checklist is
  a documentation problem, not a reason to change the default here.

## Update: Rate Limiting (Phase 2 — closing the item this ADR left open)

This ADR's own Decision section named rate-limiting as "explicitly
UNRESOLVED... the deferred half of the original 'auth/rate-limiting'
flag, not silently folded into this decision." This Update closes that
out, following the exact same scoping discipline as the API-key decision
above: minimal, off by default, real and enforced the moment
`PUBLIC_DEPLOYMENT` is ever turned on.

**Library/approach chosen: hand-rolled, not `slowapi`.** `slowapi` (this
ADR's own roadmap note named it as the likely candidate) was checked
directly, not assumed: it and its own dependency (`limits`) are
genuinely lightweight for the default in-memory backend — `slowapi`
itself is a 14KB wheel, `limits` a 60KB wheel with only
`deprecated`/`packaging`/`typing-extensions` as required (non-extra)
dependencies; the heavier backends (redis, memcached, mongodb) are all
optional extras this project would never install. Tonight's own repeated
OOM history was the reason to check this rather than assume it, and the
answer is genuinely "no real memory concern here."

**Rejected anyway, for a different, disqualifying reason:** `slowapi`'s
own documentation states plainly, "`websocket` endpoints are not
supported yet." This project's live tactical stream (`/ws/tactical-stream`)
is exactly the endpoint most in need of its OWN connection-rate
consideration (Step 2.2) — meaning even with `slowapi` installed, the
WebSocket half of this problem would need a hand-rolled mechanism
regardless, the exact same class of framework gap this ADR's own auth
decision already hit and solved manually (`Depends()` not applying to
`@app.websocket` routes the same way it does REST ones). Rather than run
TWO different rate-limiting implementations side by side — a
library-based decorator for REST, a hand-rolled check for WebSocket, with
no guarantee their semantics agree — ONE simple, in-memory token bucket
(`_RateLimiter`/`_TokenBucket` in `api.py`) is applied uniformly to both,
via `Depends(_rate_limit(tier))` for REST routes and a manual check
before `accept()` for the WebSocket route (mirroring `_require_api_key`'s
own REST-vs-WebSocket split exactly). This also avoids a new dependency
for only partial coverage of the actual problem.

**Correct for this project's actual deployment model:** an in-memory,
per-process dict is sufficient because this project runs as a single
uvicorn process (this ADR's own stated context: one operator, no
multi-worker/shared-state need) — the same reasoning that already
justified SQLite over Postgres for alert history (ADR-019) and a single
shared secret over a full auth system (this ADR's own Decision above). A
genuinely multi-instance deployment would need a shared backend (Redis —
`limits` itself would have needed exactly this too), explicitly out of
scope for the same reason a real auth system was.

**Rate-limit KEY (Step 0.2), two genuinely different modes, not one
hardcoded scheme:**
- `API_KEY` set (auth enabled): keyed on the API key VALUE itself
  (`f"key:{API_KEY}"`). Since this ADR supports exactly ONE shared
  secret today, this is, in practice, one shared bucket for every
  authenticated caller — stated plainly, not hidden. Keying on the
  value (rather than a hardcoded single global bucket name) is still the
  right, forward-compatible choice: this ADR's own Consequences section
  already names "more than one party needs distinguishable access" as
  the trigger for revisiting the single-shared-secret decision, and a
  per-key-value bucket needs no further change if that day comes.
- `API_KEY` unset (today's local-dev default, unaffected either way
  since rate limiting itself is off then — see below): keyed on client
  IP (`request.client.host` for REST, `websocket.client.host` for the
  WebSocket route).

**Tiered limits (Step 0.3), each independently justified, not one
blanket number — all requests/minute (token-bucket capacity) unless
stated otherwise:**

| Tier | Limit | Endpoints | Reasoning |
|---|---|---|---|
| (none — fully exempt) | unlimited | `GET /health` | Same full exemption this ADR already gives it from the API-key check — a load balancer/uptime monitor must never be throttled OR credentialed. |
| `metrics` | 300/min | `GET /metrics` | "Effectively unlimited" per this project's own Step 1 requirement — a real monitoring poller checks every 10-60s; 300/min is far above any real polling cadence. |
| `standard` | 30/min | `/simulate`, `/reports/player/{id}` (report, shot-map, match-summary, press-resistance, touch-map, timeline), `/reports/pass-network/{match_id}`, `/reports/team/{name}/pass-entropy`, `/reports/team/{name}/opposition-analysis`, `/alerts/history` | Bounded single-player/single-match linear scans over already-cached files. Real dashboard usage fires ~6-8 of these per "Generate Report" click; 30/min gives comfortable headroom for a legitimate multi-panel session while bounding a scripted loop to a modest, sustainable rate. |
| `heavy` | 6/min | `/reports/team/{name}`, `/reports/team/{name}/passing-lanes`, `/reports/team-comparison`, `/reports/team-comparison/match`, `/reports/player/{id}/similar` | This ADR's own Context section already measured `/reports/team/{name}` at "up to ~100s for a well-supported team" (real `BiomechanicalPitchControl` computation); 6/min (1 per 10s) bounds sustained hammering from stacking concurrent/queued heavy work while still letting a real user explore a handful of teams/comparisons per minute. `/reports/player/{id}/similar` is placed here per this task's own explicit scope, even though its own live query is in fact a fast precomputed-index lookup, not a slow one. |
| `similarity_rebuild` | 1 per 30 minutes | `POST /reports/player-similarity/rebuild` | Its own, uniquely tight tier — a REAL measured ~27-minute full-population operation (`player_similarity.py`'s own docstring), meant to be triggered rarely and deliberately ("once after fetching new player data"). Allowing even a handful of these per hour would be genuinely excessive resource consumption for what this endpoint is for. |
| `websocket_connect` | 10 new connections/min | `WS /ws/tactical-stream` | A CONNECTION-rate limit, deliberately NOT per-message throttling — a legitimately open stream sends many `threat`/`alert` messages by design (Milestone 17) and none of that volume is throttled. Each NEW connection does real setup work (a fresh `CVPipeline` instance for `source="cv"`, a `live_match_stream` generator) before a single message is sent, so this bounds connection-flood abuse specifically. A real interactive session opens a handful of streams per sitting (start/stop/retry while testing settings); 10/min covers that comfortably. |

**Response on exceeding a limit:** REST returns a real `429` with a
`Retry-After` header (the real number of seconds until the bucket's next
token, not a fixed/generic value) and a JSON body naming the tier. The
WebSocket route closes with code `1013` ("Try Again Later" — the
standard WebSocket close code for exactly this situation, the REST-side
analog of a 429), before `accept()`, mirroring the existing `1008`
unauthorized-close pattern exactly.

**Tied to `PUBLIC_DEPLOYMENT`, not a separate flag (Step 1):**
`PUBLIC_DEPLOYMENT` unset (today's default, and this project's own test
suite's only configuration) means rate limiting is genuinely OFF — no
bucket is even checked, not merely "set very high" — so local
development and the existing test suite's own rapid sequential requests
see zero behavior change (confirmed directly: the full existing suite,
`production/tests/test_api.py` and `production/tests/test_dashboard.py`,
passes unchanged with zero new failures). `PUBLIC_DEPLOYMENT=true`
enables it for real, with an explicit startup log line (mirroring every
other `PUBLIC_DEPLOYMENT`-gated status line this file already prints)
stating plainly that rate limiting is active, which tiers exist at what
capacity, and which keying mode is in effect — an operator turning
`PUBLIC_DEPLOYMENT` on sees this confirmed, not merely assumes it from
reading source.

**Relationship to ADR-021's content-exposure gating (Step 1.3):**
genuinely separate concerns, layered, not duplicated or in conflict.
ADR-021's `PUBLIC_DEPLOYMENT` checks control WHAT DATA a response
contains (e.g. the shot map's aggregated-vs-raw variant); this Update's
checks control HOW OFTEN a caller may ask for it at all, checked earlier
in the same dependency chain (`Depends(_require_api_key)` — WHO,
`Depends(_rate_limit(tier))` — HOW OFTEN — then the endpoint's own
`PUBLIC_DEPLOYMENT` branching decides WHAT). A request can be rejected by
either check independently; neither one changes the other's behavior.

**Validated, not merely reasoned through:** a REAL, triggered 429 was
confirmed for the `heavy` tier's own unmodified 6/min capacity, and
separately for `similarity_rebuild`'s own 1-per-30-minutes capacity
(`production/tests/test_rate_limiting.py`); a REAL, triggered WebSocket
close-code-1013 was confirmed for `websocket_connect`'s own unmodified
10/min capacity. `/health` and `/metrics` were confirmed to stay
unthrottled under real repeated requests with `PUBLIC_DEPLOYMENT=true`.
The full existing test suite was confirmed unaffected with
`PUBLIC_DEPLOYMENT` at its real default (unset).

This closes the rate-limiting half of the original "auth/rate-limiting"
review flag this ADR's own Context section named — both halves are now
resolved, this ADR's own Decision text notwithstanding (that text is left
unmodified above, per this project's own append-don't-overwrite ADR
convention; this Update supersedes only the "remains explicitly
UNRESOLVED" framing, not the API-key decision itself).
