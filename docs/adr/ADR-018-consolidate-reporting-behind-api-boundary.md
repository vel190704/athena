# ADR-018: Consolidate Reporting Data Access Behind the FastAPI Serving Layer

## Status
Accepted

## Context

`production/frontend/dashboard.py` is a single Streamlit process with five
tabs. Only one of them was actually architected as a client of the backend:
the "Live CV Monitor" tab's What-If Simulator and Live Tactical Threat
Monitor panels go through `production/src/serving/api.py`'s REST (`/simulate`)
and WebSocket (`/ws/tactical-stream`) endpoints, exactly as Milestones 16-19
built them.

The four reporting tabs added since (Player Reports, Team Reports, Team
Trends, Team Comparison) do not follow that pattern. `dashboard.py` imports
`generate_player_report`, `generate_team_report`, `generate_team_trend_report`,
and `compare_team_seasons` directly and calls them in-process. Concretely,
that means the Streamlit process itself talks to MLflow (via
`explainer.load_deterministic_mlp`, transitively) and reads `data/raw/`
independently of `api.py` — a second, parallel data-access path with its own
copy of the same environment assumptions.

The symptom is already visible in the dashboard's own source, in a comment
that exists ONLY because this dual-entrypoint problem is real, not
hypothetical:

> "Must be set before any import that transitively touches MLflow (the Team
> Reports tab's generate_team_report -> load_deterministic_mlp -> mlflow.
> tracking.MlflowClient() chain) -- this project's mlflow version treats the
> file-store backend as read-only "maintenance mode" otherwise.
> `production/src/serving/api.py` sets this itself at its own module top
> level for the exact same reason (the standalone-launch fix from Milestone
> 17); this dashboard is ALSO a standalone entrypoint (`streamlit run
> production/frontend/dashboard.py`) that never imports api.py, so nothing
> else in this process would set it otherwise -- found via this file's own
> Step 6 validation (a real crash, not a hypothetical), not assumed in
> advance."

`dashboard.py` has to carry its own `os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")`
at module top level, duplicating a fix `api.py` already has, purely because
the dashboard is a second, independent MLflow client. This only works today
because both processes run on the same machine against the same local
`mlruns/` and `data/raw/` directories. It would break immediately the moment
`dashboard.py` and `api.py` were ever deployed to separate machines (the
Streamlit process would have no MLflow tracking URI or `data/raw/` to read
from at all) — a real architectural gap, not a stylistic one.

## Decision

Consolidate all reporting-related MLflow and `data/raw/` access behind
`api.py`, via three new thin, unmodified-logic-wrapping endpoints:

- `GET /reports/player/{player_id}` — wraps `player_report.generate_player_report`
- `GET /reports/team/{team_name}` — wraps `team_report.generate_team_report`
- `GET /reports/team-comparison` — wraps `team_comparison.compare_team_seasons`

`dashboard.py`'s reporting tabs call these over HTTP (reusing the existing
`REST_API_BASE_URL` sidebar config already used for `/simulate`) instead of
importing the report-generation functions directly. `api.py` becomes the
ONLY code path that touches MLflow or `data/raw/` for reporting purposes
going forward; the report-generation functions themselves
(`player_report.py`, `team_report.py`, `team_comparison.py`) are not
modified — this is a consolidation of WHERE they are called from, not a
change to what they do.

**`team_trend_data.py` is explicitly EXCLUDED from this consolidation.**
That module's own docstring already states a real, pre-existing
restriction: its football-data.co.uk data source has an unresolved
licensing scope ("for the purposes of league match prediction only", no
clean redistribution license found), so it is deliberately scoped to
personal, non-distributed, LOCAL research use only — the same conservative
posture ADR-014 applies to the AGPL-derived CV pitch-keypoint model. That
module's docstring is explicit: "Nothing in this module is wired into
`production/src/serving/api.py`'s live WebSocket/REST layer or any other
network-served endpoint... If this feature is ever extended toward a
served/distributed use case, the licensing question above must be
revisited and resolved first." Adding a served endpoint for it would
directly contradict that already-made decision. The Team Trends tab
therefore continues to call `generate_team_trend_report` in-process,
unchanged — a deliberate, named exception to this ADR's consolidation, not
an oversight. This is the one remaining assumption that still ties
`dashboard.py` to running on the same machine as a `data/raw/`-adjacent
Python environment (see Consequences).

**No database is introduced.** This decision is purely about WHERE data
access happens (consolidated behind one HTTP boundary instead of two
independent processes each talking to MLflow/disk), not about adding new
persistence. There is no write-heavy workload here, no concurrent-user
contention on shared state, and no transactional requirement — every one of
these reports is a read-only, on-demand aggregation over already-existing
StatsBomb event/360 data and an already-trained, already-logged MLflow
model. A database would add a real operational dependency (a service to
run, schema to migrate, connection pooling to configure) to solve a
problem — write contention, multi-writer consistency — this project does
not have. `api.py` reading `data/raw/` and MLflow directly, exactly as it
already does for `/simulate` and the WebSocket stream, is the correct-sized
solution; only the entrypoint doing that reading needed to be
consolidated, not the storage layer replaced.

## Consequences

- `dashboard.py` no longer needs its own `MLFLOW_ALLOW_FILE_STORE`
  workaround (removed — see Step 3 of the implementing change) and no
  longer imports `generate_player_report`, `generate_team_report`, or
  `compare_team_seasons` directly. It has no MLflow or `data/raw/`
  dependency for those three tabs at all.
- The Player Reports, Team Reports, and Team Comparison tabs can now
  genuinely run with `dashboard.py` and `api.py` on separate machines,
  provided the Streamlit host can reach the API host over HTTP (the same
  requirement the Live CV Monitor tab already has).
- **The Team Trends tab remains a real, named exception.** It still
  imports and calls `generate_team_trend_report` directly. That path never
  needed the `MLFLOW_ALLOW_FILE_STORE` workaround in the first place (it
  doesn't touch MLflow at all) — but it does still need
  `data/raw/football_data_co_uk/` write access and network access to
  football-data.co.uk from wherever `dashboard.py` itself runs. This means
  full separation is not unconditionally true for the whole app: it holds
  for four of the five
  tabs; the fifth still assumes `dashboard.py` runs somewhere with its own
  disk and network access for that one data source. This is a deliberate,
  documented carve-out (per the licensing restriction above), not an
  incomplete migration.
- Every caveat/reliability field each report already computes (low-sample
  flags, `heatmap_used_uniform_fallback`, `reliability_caveat`, etc.) must
  survive the move to a JSON HTTP response unchanged — these are read-only
  passthrough wrappers, so this is a verification concern for the
  implementation, not a design question this ADR needs to resolve.

## Alternatives Considered

- **Introduce a database (e.g. Postgres/SQLite) as the shared access
  layer**: rejected. Nothing about this problem is a persistence problem —
  it is an entrypoint-consolidation problem. Every report is computed
  on-demand from data that already lives in MLflow's tracking store and
  `data/raw/`'s cached StatsBomb JSON; there is no write-heavy workload, no
  concurrent-writer conflict, and no transactional requirement a database
  would address. Adding one here would introduce a new operational
  dependency to solve a problem this project doesn't have.
- **Wire `team_trend_data.py` into `api.py` alongside the other three**:
  considered, since the task prompting this ADR initially asked for it.
  Rejected after re-reading that module's own docstring, which already
  states — as a real, prior, deliberate decision — that it must not be
  served over a network endpoint pending resolution of its data source's
  licensing scope. Overriding that silently would have re-opened a
  question this project already answered conservatively elsewhere
  (ADR-014); the exclusion is deliberate, not a gap.
- **Have `dashboard.py` keep direct MLflow/disk access for all four
  reporting tabs, and only document the limitation**: rejected — the whole
  point of this ADR is that "only works on one machine" is not an
  acceptable steady state for three of the four tabs when fixing it is a
  thin wrapper away, not a redesign.

## Update: `candidate_index.py` Is a Second, Narrower Exception to Full Separability

A later change added dynamic Player/Team Reports dropdowns
(`production/src/reporting/candidate_index.py`), replacing a small,
hand-picked preset list with a real scan of what's actually cached. A
subsequent independent verification audit checked this ADR's own claim —
*"The Player Reports, Team Reports, and Team Comparison tabs can now
genuinely run with `dashboard.py` and `api.py` on separate machines"* —
against what `candidate_index.py` actually does, and found the claim no
longer fully holds, in a way this ADR had not been updated to say.

**What's actually true, stated plainly (the same way this ADR already
names `team_trend_data.py` as a named exception above, not implied to be
covered by the "fully separable" claim)**: `candidate_index.py` reads
`data/raw/` **directly from the Streamlit process** to populate the
Player Reports and Team Reports dropdowns — team/player names, season
groupings, and (post-audit) 360-coverage-based sample-size labels are all
computed by scanning cached JSON on disk, not by calling `api.py`. This
means:

- **Report *generation* remains exactly as this ADR originally described**:
  once a candidate and season(s) are selected, `dashboard.py` still calls
  `api.py`'s `/reports/player/{id}` / `/reports/team/{name}` endpoints
  over HTTP, unchanged. Nothing about the actual reporting round-trip
  regressed.
- **Dropdown *population* for those same two tabs did not go through this
  ADR's consolidation** — it is a second, narrower re-introduction of the
  co-location assumption this ADR otherwise removed, scoped specifically
  to "what candidates can I browse," not to the reports themselves.
- Team Comparison's own tab has no such dependency (it never had a
  candidate-browsing dropdown to begin with — both team names are always
  free-text `st.text_input` fields), so it is unaffected by this
  exception.

**Why this wasn't fixed by adding a `/candidates/...` endpoint instead**:
that would be the fully-correct fix (matching this ADR's own original
`/reports/...` pattern exactly), and remains the natural next step if full
separability for these two tabs is ever required — but the task that
introduced `candidate_index.py` was explicitly scoped as a
dashboard/enumeration-layer change only, not an `api.py` change, so this
Update records the resulting gap rather than silently closing it via
scope creep.

**Consequence, restated precisely**: full multi-machine separability now
holds for report *generation* on all four reporting tabs (Team Trends
excepted, per the original decision above), but **not** for Player
Reports/Team Reports dropdown *population* specifically — that still
requires `dashboard.py` to run with its own `data/raw/` access, exactly
like the `team_trend_data.py` exception already documented, just for a
different reason (a dropdown convenience feature, not a licensing
restriction).
