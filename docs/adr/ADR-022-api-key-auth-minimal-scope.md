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
