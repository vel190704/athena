# Project Athena

A research-grade, end-to-end AI system that ingests football (soccer) match telemetry — from StatsBomb's open event/360 data, and now also from broadcast video via a parallel Computer Vision pipeline — and produces real-time tactical intelligence: goal-probability forecasting, spatial weak-spot detection, and a counterfactual "what-if" simulator. Two additional, entirely additive reporting capabilities sit on top of the StatsBomb track: historical player/team analysis reports (with a live Streamlit UI), and season-by-season team trend data from a second, non-StatsBomb source.

This repository has grown across numbered milestones 1 through 45, plus several further additions made after the numbering convention lapsed (see `docs/FULL_PROJECT_REPORT.md` for the exact, verified timeline and an explicit note on when/why numbering stopped). **This page exists to disambiguate the documents** — to tell you which one to open for what you're actually trying to do.

## Which document do I want?

| Document | Read this when you want... | Length |
|---|---|---|
| **This file (`README.md`)** | A map of the other documents, and a one-glance table of what's done vs. not | Short |
| [`context.md`](context.md) | Fast orientation on the whole project: what it is, how it's structured, what's been built and why | Medium |
| [`docs/FULL_PROJECT_REPORT.md`](docs/FULL_PROJECT_REPORT.md) | The complete, single-narrative history: every milestone, every ADR's real decision, every RQ's actual answer, every bug found and fixed with real numbers, the full honest gap list, and the recurring methodological lessons | Very long |
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | A short, non-technical, plain-language summary — no jargon, aimed at a non-ML reader | Very short |
| [`readme.txt`](readme.txt) | The full original technical/architecture blueprint — research questions, system assumptions, module-by-module math, engineering standards | Long |
| [`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md) | The StatsBomb/physics-ML track's actual RQ1-RQ5 findings, every number re-verified against MLflow before being cited | Long |
| [`docs/CV_PIPELINE_FINDINGS.md`](docs/CV_PIPELINE_FINDINGS.md) | The Computer Vision track's component-by-component validation status, the tactical-map/trust-radius work, and the real-data (SoccerNet NDA) blocker | Long |
| [`docs/REPORTING_FINDINGS.md`](docs/REPORTING_FINDINGS.md) | The historical player/team reporting layer, the football-data.co.uk team-trend feature (§8), and the general StatsBomb team-comparison tool (§9) | Long |
| [`docs/adr/`](docs/adr/) | *Why* a specific design decision was made, and what it cost — one file per decision, **17 total** (ADR-001 through ADR-017, no gaps) | One ADR at a time |

If you only read one thing for a fast orientation: read `context.md`. If you want the complete, ground-truth picture (every number sourced, every discrepancy between older docs and the real repo called out): read `docs/FULL_PROJECT_REPORT.md`.

## At a Glance: Achieved vs. Not Yet

| Track / Area | Achieved so far | Not yet achieved |
|---|---|---|
| Physics core (Kalman friction, pitch control) | Kalman filter validated to 0.336% error on synthetic data (5x tighter than the 2% gate, ADR-008); analytical pitch-control ODE, sparse-masked, in production | Real StatsBomb pass-trajectory validation (RQ3's literal "pass landing error" criterion) — only synthetic convergence proven |
| Survival prediction (DeepHit, RQ1) | Well-calibrated Brier Scores in absolute terms (0.0942 / 0.1588 @15s/30s, M14B) across 8,074 real samples / 55 matches | A non-physics-informed baseline ablation, to measure RQ1's literal "% improvement" criterion |
| Graph vs. handcrafted features (RQ4) | GNN wired in, trained head-to-head under matched, stabilized hyperparameters; current data point: MLP wins | A settled verdict — this comparison has already reversed direction twice; re-running it under the match-level split has not been done |
| Uncertainty quantification | 5-member Deep Ensemble (M21) with per-member disentangled loss | True Batch Ensemble (shared weights, ADR-004) — the ~5x compute cost remains unresolved |
| Explainability | Integrated Gradients + templated LLM-style summaries, fully async (M15, ADR-006); extended to per-grid-cell zone attribution (M40 Step 3) and natural-language zone explanations (M43) | A real LLM integration (currently a mock/templated executor everywhere) |
| Historical/habit memory (RQ2) | Implemented, tested, honestly reported as a null result (M22-23) | Re-running it against the ~10x larger training corpus the match-level split unlocked (4 → up to 42 eligible matches) |
| Digital Twin / counterfactuals (RQ5) | Heuristic perturbation engine (M13) + Oracle Substitution Validation against one real match (M20) | Multi-match Oracle validation — only 1 genuinely unconfounded observation exists so far |
| Live serving & dashboard | WebSocket/REST API, Streamlit dashboard now **five tabs** (Live CV Monitor, Player Reports, Team Reports, Team Trends, Team Comparison), per-connection isolation, async non-blocking CV integration | Real calibration wired through the live CV API path (currently pixel-space only, not physically meaningful yet) |
| Computer Vision pipeline (Module 4) | Full pipeline built and internally validated end-to-end on one real private clip; a real bug found & fixed (ball-detector `imgsz`); anchor-based tactical-map rendering with measured trust-radius gating (ADR-015/016/017: ~21-27% of real players fall within the ~150px reliable radius on the one available camera framing) | SoccerNet-gated ground-truth validation (detection P/R, true ID-switch rate, real-lens calibration accuracy) — **the single biggest blocker in the whole project.** A second, differently-angled clip to test whether the six-vertex exclusion list generalizes. |
| Historical reporting layer (StatsBomb-based) | Player/team profile reports + zone-level IG attribution + dashboard visualization (M40/42/43), a validation sweep across varied real profiles that found and fixed a real sample-size-transparency gap (M44), a static HTML browsing index (M45), and a general two-mode (pitch-control / location-only) team-season style comparison tool, all wired into the live Streamlit dashboard | Validating the zone-explainer's aggregate pattern against a team with a *known* real tactical asymmetry; a temporal/transition-state layer (needed for duration/movement claims in generated text) |
| Team trend reports (football-data.co.uk, new, non-StatsBomb source) | Season-by-season team results/stats (goals, cards, home/away record) for the "big five" leagues through 2025/26, with honest gap-season reporting and year-over-year deltas, wired into the dashboard | Scoped to personal, non-distributed research use only — a real, unresolved licensing ambiguity (see `docs/REPORTING_FINDINGS.md` §8), handled the same conservative way as the CV track's AGPL-derived pitch-keypoint model (ADR-014) |
| Docker packaging | `docker/backend.Dockerfile`, `docker/frontend.Dockerfile`, `docker-compose.yml` exist and represent real work (backend/frontend containerization, volume-mounted cached data, health-gated startup) | **Attempted and abandoned mid-validation** due to local disk-space constraints on this machine (CPU-only PyTorch wheels alone exhausted available space). Never validated end-to-end via `docker compose up`. The direct local-process run (`uvicorn` + `streamlit`) is the only validated way to run this project, and was re-confirmed working after the abandonment. |
| Test suite | **158 passed, 1 skipped**, 0 failed (`pytest production/tests/`, 39 test files) — the 1 skip is the SoccerNet-gated detection-accuracy test, blocked on NDA access, not a real failure. CI now runs this suite on every push/PR (`.github/workflows/test.yml`). `team_trend_data.py`, `team_comparison.py`, `player_comparison.py`, and the dashboard's Streamlit-tab integration now all have real regression coverage (13 new tests, engineering-review action item) | — |
| Engineering hygiene (external review response) | CI (GitHub Actions), pinned dependencies (`requirements-lock.txt`), `train.py` converted from `print()` to `logging`, `train_and_evaluate`/the two `_train_and_log_*` functions decomposed into 9 named sub-functions (behavior-preserving — see `docs/FULL_PROJECT_REPORT.md` §11 for the exact numerical diff), the 4-file test-coverage gap closed | Auth/rate-limiting on the live serving layer, a true Batch Ensemble implementation (ADR-004), the SoccerNet access path, RQ2/RQ4 re-runs under the match-level split — all still open, per the review's own medium/long-term roadmap; see `docs/FULL_PROJECT_REPORT.md` §11 |

## Quick start

```bash
pip install -r requirements-lock.txt            # exact versions this project was built/validated against
# or: pip install -r requirements.txt            # unpinned, human-readable, grouped-by-track list instead
cp .env.example .env                             # optional -- fill in whichever keys you actually need (all optional, see the file itself)
python -m production.src.pipeline.train        # full StatsBomb training pipeline (expensive)
pytest production/tests/                        # full test suite
mlflow ui                                        # inspect logged runs (from repo root)
uvicorn production.src.serving.api:app --reload  # live WebSocket/REST API
streamlit run production/frontend/dashboard.py   # live dashboard (5 tabs: CV monitor + 4 reporting tabs)
```

`requirements-lock.txt` (new, engineering-review action item) is a full `pip freeze` snapshot — every direct and transitive package pinned to the exact version this project's real results were produced against. `requirements.txt` stays unpinned and grouped by track for readability. See `requirements-lock.txt`'s own header for two honest caveats found while generating it (a `torch` local-version-suffix quirk, and `python-dotenv` having been an undocumented-but-real dependency until this pass — now added to `requirements.txt`).

This direct-process path is the only one actually validated end-to-end on this machine. A `docker-compose.yml` and two Dockerfiles also exist in this repo (see the Docker row above) — they represent real, reviewed work, but **were never successfully run via `docker compose up`** here due to local disk-space limits encountered mid-build. Treat them as a real starting point for a machine with more headroom, not as a validated alternative to the commands above.

## Public deployment mode

Set `PUBLIC_DEPLOYMENT=true` in the environment of **both** the `api.py` process and the `dashboard.py` process before deploying this app somewhere the general public can reach it (a free, non-monetized public site — see [ADR-021](docs/adr/ADR-021-statsbomb-free-public-deployment-scoping.md) for the licensing scoping this flag enforces). Unset, or set to anything other than `"true"` (the default): behavior is identical to local/private research use, unchanged from before this flag existed.

What the flag changes when set:

- **Shot map** (`/reports/player/{player_id}/shot-map`): serves and renders only an aggregated shot-density / mean-xG grid — no individual shot's exact location is ever computed or returned on this path. Local/private mode (flag unset) continues to show the real per-shot scatter plot exactly as before. See ADR-021's Update section for the full reasoning.
- **Team Trends tab**: disabled entirely, with a visible explanation in its place. This tab's data source (football-data.co.uk, via `team_trend_data.py`) has its own separate, already-documented restriction — personal, non-distributed research use only — that a public deployment of the dashboard would otherwise silently violate regardless of this flag's effect on the shot map. See `team_trend_data.py`'s own docstring and [ADR-018](docs/adr/ADR-018-consolidate-reporting-behind-api-boundary.md)'s Update section.

Both the shot map and Team Trends fixes were found via a dedicated compliance audit that checked every user-facing feature against ADR-021's condition 2 ("no raw StatsBomb data exposed to site visitors") — everything else audited (positional distribution, the aggregate heatmap, team reports, team comparison, the What-If simulator, the StatsBomb-sourced live tactical stream) was already compliant by construction and needed no gate.

See `context.md` §2 for the full directory layout and module table, and `docs/FULL_PROJECT_REPORT.md` for the complete verified history.
