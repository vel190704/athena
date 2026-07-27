# Project Athena

A research-grade, end-to-end AI system that ingests football (soccer) match telemetry — from StatsBomb's open event/360 data, and now also from broadcast video via a parallel Computer Vision pipeline — and produces real-time tactical intelligence: goal-probability forecasting, spatial weak-spot detection, and a counterfactual "what-if" simulator.

This repository has grown several documents over 36 milestones. **This page exists to disambiguate them** — to tell you which one to open for what you're actually trying to do.

## Which document do I want?

| Document | Read this when you want... | Length |
|---|---|---|
| **This file (`README.md`)** | A map of the other documents, and a one-glance table of what's done vs. not | Short |
| [`context.md`](context.md) | Fast orientation on the whole project: what it is, how it's structured, what's been built and why, what's currently open | Medium |
| [`readme.txt`](readme.txt) | The full original technical/architecture blueprint — research questions, system assumptions, module-by-module math, engineering standards | Long |
| [`docs/RESEARCH_FINDINGS.md`](docs/RESEARCH_FINDINGS.md) | The StatsBomb/physics-ML track's actual RQ1-RQ5 findings, every number re-verified against MLflow before being cited | Long |
| [`docs/CV_PIPELINE_FINDINGS.md`](docs/CV_PIPELINE_FINDINGS.md) | The Computer Vision track's component-by-component validation status and the real-data (SoccerNet NDA) blocker | Long |
| [`docs/adr/`](docs/adr/) | *Why* a specific design decision was made, and what it cost — one file per decision, 11 total | One ADR at a time |

If you only read one thing: read `context.md`, then follow its pointers into the ADRs or findings docs for whatever you need to go deeper on.

## At a Glance: Achieved vs. Not Yet

| Track / Area | Achieved so far | Not yet achieved |
|---|---|---|
| Physics core (Kalman friction, pitch control) | Kalman filter validated to 0.336% error on synthetic data (5x tighter than the 2% gate, ADR-008); analytical pitch-control ODE, sparse-masked, in production | Real StatsBomb pass-trajectory validation (RQ3's literal "pass landing error" criterion) — only synthetic convergence proven |
| Survival prediction (DeepHit, RQ1) | Well-calibrated Brier Scores in absolute terms (0.0942 / 0.1588 @15s/30s, M14B) across 8,074 real samples / 55 matches | A non-physics-informed baseline ablation, to measure RQ1's literal "% improvement" criterion |
| Graph vs. handcrafted features (RQ4) | GNN wired in, trained head-to-head under matched, stabilized hyperparameters; current data point: MLP wins | A settled verdict — this comparison has already reversed direction twice; re-running it under the new match-level split has not been done |
| Uncertainty quantification | 5-member Deep Ensemble (M21) with per-member disentangled loss | True Batch Ensemble (shared weights, ADR-004) — the ~5x compute cost remains unresolved |
| Explainability | Integrated Gradients + templated LLM-style summaries, fully async (M15, ADR-006) | A real LLM integration (currently a mock/templated executor) |
| Historical/habit memory (RQ2) | Implemented, tested, honestly reported as a null result (M22-23) | Re-running it against the ~10x larger training corpus the match-level split unlocked (4 → up to 42 eligible matches) |
| Digital Twin / counterfactuals (RQ5) | Heuristic perturbation engine (M13) + Oracle Substitution Validation against one real match (M20) | Multi-match Oracle validation — only 1 genuinely unconfounded observation exists so far |
| Live serving & dashboard | WebSocket/REST API, Streamlit dashboard, per-connection isolation, async non-blocking CV integration (M16-19, M33) | Real calibration wired through the live CV API path (currently pixel-space only, not physically meaningful yet) |
| Train/val split methodology | Match-level split — no match can straddle both splits, by construction (ADR-011) | *(closed — nothing further planned here)* |
| Training-stability safety net | 4-signal detector, regression-tested against real historical failure patterns; a false positive found & fixed (ADR-010) | *(considered solid pending a genuinely novel failure mode)* |
| Computer Vision pipeline (Module 4) | Full pipeline built and internally validated (detection→tracking→calibration→team ID→ball detection→shot classification→orchestration→adapter→live API, M25-34); one real private clip processed end-to-end, a real bug found & fixed (ball-detector `imgsz`), and a first real throughput number (8.57 fps vs. this clip's 28fps source) | SoccerNet-gated ground-truth validation — detection P/R, a true ID-switch rate, calibration accuracy vs. real lens distortion. **The single biggest blocker in the whole project.** |

## Quick start

```bash
pip install -r requirements.txt
python -m production.src.pipeline.train        # full StatsBomb training pipeline (expensive)
pytest production/tests/                        # full test suite
mlflow ui                                        # inspect logged runs (from repo root)
uvicorn production.src.serving.api:app --reload  # live WebSocket/REST API
streamlit run production/frontend/dashboard.py   # live dashboard
```

See `context.md` §2 for the full directory layout and module table.
