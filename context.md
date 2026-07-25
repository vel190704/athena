# Project Athena – Context Document

**Repository:** https://github.com/vel190704/athena  
**Version:** 5.1 (Research-Grade HPC Architecture Blueprint)  
**Last major update:** Milestone 34 (CV Pipeline Findings synthesis)  
**Status:** Two tracks now implemented, each validated to its own appropriate standard — StatsBomb/physics-ML track: implemented **and empirically validated** against real match data (8,074 samples, ~55 matches, 12 competitions — see `docs/RESEARCH_FINDINGS.md`); CV/Module 4 track: implemented **and internally validated** on synthetic/adversarial/mocked tests only, **not yet validated on real broadcast footage**, blocked on SoccerNet NDA access (see `docs/CV_PIPELINE_FINDINGS.md`).

---

## 1. What you're building & why

**Project Athena** is a research-grade, end-to-end AI system that ingests live (or replayed) football match telemetry and produces real-time tactical intelligence.

It models a football match as a **differential game**. Physical forces (player biomechanics, fatigue, ground friction, environmental conditions) actively shape the tactical spaces available on the pitch. The system is designed to estimate tactical states and their likely consequences — not to act as an “omniscient football brain.”

### Core real-time outputs
- **Phase-by-phase Goal Probability** — Continuous time-to-event prediction using DeepHit survival analysis, with a 5-member Deep Ensemble (Milestone 21) providing predictive uncertainty alongside the point estimate.
- **Tactical Cheat Sheets** — Real-time spatial weak-spot detection via velocity-aware pitch control, with optional Bayesian habit-memory blending (Milestone 22-23; implemented and wired in, but empirically a null result under this project's current data constraints — see Research Findings §RQ2).
- **Digital Twin & Counterfactual Simulator** — What-if engine answering questions such as “What happens to expected threat (xT) if we force the play wide / high-press / drop deep?”, now backed by two complementary evidence threads: the original heuristic perturbation engine (Milestone 13) for general tactical-alignment probes, and Oracle Substitution Validation (Milestone 20) which backtests the same machinery against real historical substitutions — a hedged, confounded-but-real result, not a clean pass.

### What's been added since the Milestone 13 blueprint
Beyond the three outputs above, the system now also includes: ensemble-based uncertainty quantification (Milestone 21); an Integrated-Gradients feature-attribution layer with a templated LLM-style natural-language explainer (Milestone 15, ADR-006); an interactive live Streamlit dashboard and a FastAPI WebSocket/REST `/simulate` serving layer (Milestones 16-19); and a full parallel Computer Vision ingestion track (Module 4, Milestones 25-34) that lets broadcast video feed the exact same tensor contract the StatsBomb path produces.

### Why this project exists
Football is a complex dynamical system. Most existing analytics tools rely on handcrafted features or black-box models that ignore the underlying physics. Athena deliberately separates **deterministic physics** from **statistical inference** so that every prediction remains interpretable, independently testable, and grounded in real biomechanical constraints.

The project is framed around five explicit Research Questions (RQ1–RQ5) with measurable success criteria (Brier Score, calibration error, AUROC, pass landing error, substitution xT shift, etc.). All five now have working, honestly-reported answers in `docs/RESEARCH_FINDINGS.md` — none treated as permanently settled (RQ4 in particular has reversed direction twice as data scale and training stability changed).

---

## 2. What's the structure & architecture

The codebase follows a strict modular design with clear separation of concerns.

```
athena/
├── docs/
│   ├── adr/                    # Architecture Decision Records (9 ADRs)
│   ├── RESEARCH_FINDINGS.md    # StatsBomb track: RQ1-RQ5 synthesis (Milestone 24)
│   └── CV_PIPELINE_FINDINGS.md # CV track: component validation status (Milestone 34)
├── production/
│   ├── src/
│   │   ├── ingestion/          # StatsBomb data loading & parsing
│   │   ├── physics/            # Causal Kalman friction filter
│   │   ├── spatial/            # Biomechanical pitch control engine
│   │   ├── models/             # DeepHit + GNN + Deep Ensemble + explainer + graph builder + evaluation
│   │   ├── pipeline/           # Feature extraction, chains, direction, training, counterfactual sim, habit memory, oracle validator
│   │   ├── cv/                 # Module 4: detector, tracker, calibration, team_classifier, ball_detector, shot_classifier, pipeline orchestrator, adapter
│   │   └── serving/            # FastAPI WebSocket + REST /simulate API, live-feed replay simulator
│   ├── frontend/                # Streamlit dashboard (live threat monitor + What-If panel)
│   └── tests/                   # Unit & integration tests
├── readme.txt                   # Master design document
└── requirements.txt
```

### Core Architectural Principles
1. **Physics ↔ ML Decoupling**  
   Classical deterministic physics (force-velocity curves, friction, time-to-intercept) produces immutable feature layers. Machine learning models only consume these features — they never rewrite the physics.

2. **Causal Correctness**  
   The Kalman filter strictly follows predict → physics uses μ → correct. No look-ahead bias is allowed.

3. **Research-First Build Order**  
   Physics must be validated on synthetic data before any ML work begins. Every major design choice is recorded in an ADR.

4. **Sparse & GPU-Friendly Computation**  
   Pitch control is evaluated only on cells within 30 m of the ball (~2 000 points instead of the full 6 800-cell grid).

5. **One Tensor Contract, Any Data Source**  
   Any data source — StatsBomb, CV, or a future one — must produce `player_pos [N,2]`, `player_vel [N,2]`, `is_teammate [N]` (relative to the *possessing* team, never an absolute team_id), and `ball_pos [2]`, all in the verified 100×68m pitch-grid space (`PITCH_LENGTH`/`PITCH_WIDTH`, `production/src/pipeline/feature_extractor.py`). Existing physics/ML/serving code is never modified to accommodate a new source — this single rule is what made the CV pivot (Module 4) possible as an additive adapter (Milestone 30) rather than a rewrite.

### Key Implemented Modules
| Module | Responsibility | Key File(s) |
|--------|----------------|-------------|
| Ingestion | StatsBomb events + 360 frames, coordinate scaling, caching | `statsbomb_io.py` |
| Physics | Latent rolling friction (causal Kalman) | `kalman_friction.py` |
| Spatial | Velocity-aware pitch control (analytical ODE + sparse mask) | `control.py` |
| Pipeline | Possession chains, feature extraction, direction handling, counterfactual simulator, habit memory, oracle validation | `chain_builder.py`, `feature_extractor.py`, `direction.py`, `simulator.py`, `habit_memory.py`, `oracle_validator.py` |
| Survival Model | Single-risk DeepHit + ranking loss + Brier + cumulative incidence | `deephit.py`, `deephit_loss.py`, `evaluation.py` |
| Graph | PyTorch Geometric graph builder + GNN (SAGEConv), trained head-to-head against the MLP (RQ4) | `graph_builder.py`, `gnn_model.py` |
| Uncertainty | 5-member Deep Ensemble, gradient-disentangled ranking loss (ADR-004) | `deep_ensemble.py` |
| Explainability | Integrated Gradients + templated LLM-style summary, async (ADR-006) | `explainer.py` |
| Training | End-to-end training + MLflow tracking | `train.py` |
| Serving | FastAPI WebSocket tactical-threat stream + REST `/simulate`, live-feed replay | `api.py`, `serving/simulator.py` |
| Frontend | Streamlit live threat monitor + What-If simulator panel | `dashboard.py` |
| CV — Detection | YOLOv8m person/ball detection + IoU matching/scoring | `detector.py`, `metrics.py`, `acquisition.py` — synthetic-math validated + real-photo smoke test; SoccerNet P/R gate still skips (NDA-blocked) |
| CV — Tracking | ByteTrack multi-object tracking, real-fps-derived pixel velocity | `tracker.py` — synthetic camera-pan demo only; real-footage test skips |
| CV — Calibration | `cv2.findHomography` pixel↔pitch-meter mapping | `calibration.py` — synthetic pinhole-camera model only; no real lens tested |
| CV — Team ID | Circular-hue extraction + masking-aware iterative KMeans | `team_classifier.py` — solid synthetic swatches only |
| CV — Ball | YOLO + shape-aware fallback | `ball_detector.py` — real-photo validated (one photo) + synthetic fallback |
| CV — Shot Classification | Green-ratio/edge-density tactical-view gate | `shot_classifier.py` — synthetic, incl. one documented, unresolved adversarial failure |
| CV — Orchestration | Frame-by-frame pipeline, skip-aware dt, staleness fallback | `pipeline.py` — mocked end-to-end only; real-footage throughput test skips |
| CV — Adapter | Pixel→meter tensor conversion into the shared tensor contract | `adapter.py` — synthetic homography only |

### Architecture Decision Records (ADRs)
All 9 ADR files exist in `docs/adr/`:
- ADR-001 · DeepHit over Cox / DeepSurv
- ADR-002 · StatsBomb coordinate scaling (120×80 → 100×68)
- ADR-003 · Attacking direction normalization — **superseded by ADR-009** (see both; ADR-003's `direction.py` module is kept, not deleted, for a future data source that genuinely needs it)
- ADR-004 · Deep Ensemble instead of true Batch Ensemble (Milestone 21; simpler and statistically valid, but ~5x parameters/compute, explicitly not solving Batch Ensemble's latency goal)
- ADR-005 · Sparse tensors for physics
- ADR-006 · Asynchronous explainability
- ADR-007 · Reaction-time / fatigue coupling
- ADR-008 · Kalman synthetic validation baseline
- ADR-009 · StatsBomb per-team coordinate convention (supersedes ADR-003 — coordinates are already actor-oriented, no direction flip needed; explicitly scoped to StatsBomb's convention, noted as NOT applicable to a future CV pixel-coordinate source)

---

## 3. What have you done so far & why

### Completed Milestones (in order)

| Milestone | What was delivered | Why it mattered |
|-----------|--------------------|-----------------|
| **1** | Causal Kalman friction filter + synthetic validation gate (≤ 2 % error) | Physics must be correct before any ML is allowed to consume it |
| **2** | BiomechanicalPitchControl – analytical ODE solver + sparse masking | Enables fast, biomechanically grounded pitch control |
| **3–4** | StatsBomb ingestion, possession-chain builder with high-quality censoring, scalar feature extraction | Creates the first usable training signal |
| **5–7** | Single-risk DeepHit model, fully vectorized ranking loss, time-dependent Brier Score, horizon censoring | Proper survival analysis that respects right-censoring and non-proportional hazards |
| **8** | End-to-end training pipeline + MLflow experiment tracking | Reproducible baseline that can be compared against future models |
| **9–10** | Direction inference experiments + discovery of StatsBomb’s true coordinate convention (ADR-009) | Removed a systematic orientation error and doubled usable data (both halves) |
| **11** | Standalone PyTorch Geometric graph builder with bidirectional edges and variable player count | Lays the foundation for RQ4 (graph vs handcrafted features) without contaminating the existing baseline |
| **12–12B** | GNN wired into a parallel training path and run head-to-head against the MLP; exploding-gradient instability found and fixed (grad clipping, weight decay, lr schedule) | First real RQ4 data point (MLP wins at single-competition scale) — and the first of two silent-collapse lessons that shaped this project's instability detector |
| **13** | Counterfactual perturbation engine + single-sample cumulative incidence prediction + end-to-end OOD test | First concrete step toward the Digital Twin / what-if capability (RQ5), using heuristic multiplicative perturbations, explicitly flagged as research probes |
| **14–14B** | Scaled training to 12 competitions/8,074 samples; discovered the MLP had *silently collapsed* (frozen softmax, `max_prob≈1.0`) — invisible to the instability detector active at the time, which only watched for single-epoch spikes; built a strengthened multi-signal detector (cumulative/windowed drift, dual-signal saturation, frozen-val-loss backstop) and a 2-seed robustness check | The second, more insidious silent-collapse lesson — proved that "no spike" does not mean "healthy," and that RQ4's apparent Milestone 14 result (GNN "wins") was an artifact of the MLP's invalid failure, not a real finding; corrected result: MLP wins again once both models are confirmed genuinely healthy |
| **15** | Integrated Gradients feature attribution + templated LLM-style explanation layer, async background worker on a fixed cadence (ADR-006) | Concrete implementation of "explain, don't predict, decoupled from the real-time path," ahead of a real LLM integration |
| **16–19** | FastAPI WebSocket tactical-threat stream + REST `/simulate` counterfactual endpoint + Streamlit dashboard (live monitor + What-If panel) | First live-serving surface for the whole pipeline; established per-connection state isolation as a first-class concern after a `previous_threat_15s` global-state bug |
| **20** | Oracle Substitution Validation — backtests the counterfactual machinery against real historical substitutions (match 3857276, Canada vs. Morocco) with verified fixed-team perspective | RQ5's literal question tested against real subs for the first time; result is hedged, not clean — 9 of 10 substitutions had overlapping ±2-minute windows, leaving only 1 genuinely unconfounded observation |
| **21** | 5-member Deep Ensemble with per-member gradient-disentangled ranking loss (ADR-004) | Adds predictive uncertainty; deliberately named `DeepEnsembleDeepHit` (not `BatchEnsembleDeepHit`) since it's a heavier, simpler alternative to the true Batch Ensemble README originally specified |
| **22–23** | Bayesian habit-memory layer — per-player historical positional heatmaps, Bayesian-blended with live position, scoped to the single acting player (360 data exposes no per-player identity for ~21 of 22 visible players); match/split-aware leakage guards | RQ2 tested for the first time — result is a null result (Brier Score got slightly *worse*), attributable largely to only 4 of ~55 matches being training-bucket-eligible under a conservative split rule (68% cold-start fallback), not necessarily to the mechanism being wrong |
| **24** | `docs/RESEARCH_FINDINGS.md` — full RQ1-RQ5 synthesis, every MLflow-logged number re-verified against the tracking store before citing it | Found zero numeric discrepancies specifically *because* every figure was re-checked rather than recalled — the project's verification discipline applied to its own reporting, not just its code |
| **25–33** | Full CV ingestion track (Module 4): YOLOv8m detection → ByteTrack tracking → homography calibration → team classification → ball detection → shot/camera-cut classification → pipeline orchestrator → adapter into the shared tensor contract → live WebSocket API integration | Built an entire parallel data source (broadcast video) that feeds the exact same tensor contract the StatsBomb-trained models already consume, with zero changes to existing physics/ML/serving code — proving Principle 5 above in practice. Every component internally validated (synthetic/adversarial/mocked); a real WebSocket close-frame protocol bug was found only by testing against the actual running server (M33) |
| **34** | `docs/CV_PIPELINE_FINDINGS.md` — full CV component validation synthesis, mirroring Milestone 24's discipline for the CV track | Every cited figure re-run or freshly regenerated before citing; states plainly that the entire CV track remains blocked on SoccerNet NDA access for real-broadcast validation — the document's central, load-bearing caveat |

### Important Design Decisions Made Along the Way
- Only relative “teammate” flags are used (a single freeze-frame has no absolute team IDs) — extended in the CV track (Milestone 30) to a **possession-based** heuristic: the nearest player to the ball, by transformed pitch-meter distance, defines the possessing team, verified to flip correctly as ball position changes, rather than any hardcoded team assignment.
- Feature normalization is performed **after** the train/validation split to prevent leakage.
- Velocity is zeroed for the StatsBomb track — a **permanent, accepted limitation** of that data source (360 frames contain no velocity field), not a to-do. The CV track's tracker (Milestone 26) *does* produce real pixel-displacement velocity, but it is explicitly labeled `vel_pixels_per_sec` and documented as camera-motion-confounded (conflates true object motion with camera pan/zoom) until real camera-motion compensation exists — it is not yet a substitute for true calibrated player velocity.
- The GNN **was** wired into the training pipeline (Milestone 12 onward) and trained head-to-head against the MLP under matched, stabilized conditions for RQ4. This comparison's result reversed direction multiple times as data scale and training-health changed (Milestone 12: MLP wins → 12B: still MLP → 14 as measured: GNN "wins" only via the MLP's invalid silent collapse → 14B corrected: MLP wins again) and is treated as the current best-evidence data point, not a permanently settled architectural verdict.
- A strengthened, multi-signal training-instability detector (single-epoch spike + multi-epoch cumulative/windowed drift + dual-signal output-saturation + frozen-validation-loss backstop) was built after two separate silent collapses (the GNN's exploding-gradient blowup, Milestone 12; the MLP's frozen-softmax collapse at scale, Milestone 14) each escaped the detector version active at the time.
- Bayesian habit-memory blending (Milestone 22) is constrained to the single acting player per event, a **real 360-data schema limitation** (no per-player identity for ~21 of 22 visible players in any freeze-frame), not a design choice — this dilutes the blended signal inside features that sum over ~11 players per side.
- Counterfactual actions (Milestone 13) are intentionally simple multiplicative heuristics — research probes, not production-calibrated effects.
- The CV pipeline (Milestones 25-34) is deliberately kept synthetic/adversarial-validated and does **not** claim real-world readiness anywhere in its own documentation — every "works" claim in `CV_PIPELINE_FINDINGS.md` is explicitly labeled by validation level (synthetic vs. real-photo vs. real-broadcast-footage), and the SoccerNet NDA blocker is stated plainly rather than softened.

### Current Test Coverage
- Synthetic Kalman convergence gate
- Spatial physics (sparse mask, fatigue effect, radial velocity clamping)
- DeepHit loss (gradient flow + ranking directionality) and full training pipeline at scale (12 competitions, 8,074 samples), including the strengthened instability detector
- Chain builder, feature extraction, direction logic, graph builder
- GNN model + head-to-head training against the MLP; 5-member Deep Ensemble (gradient disentanglement, diversity sanity checks)
- Bayesian habit memory (cold-start fallback, match/split-aware leakage guards)
- Integrated-Gradients explainer + async mock-LLM executor
- MLflow logging
- Counterfactual simulator + end-to-end prediction with a real trained model (Milestone 13); Oracle Substitution Validation against real historical substitutions (Milestone 20)
- Live FastAPI WebSocket API + REST `/simulate`, including per-connection state-isolation tests (established Milestone 16, re-proven for a second, independent subsystem at Milestone 33's CV integration)
- Full CV test suite — detector, tracker, calibration, team classifier, ball detector, shot classifier, pipeline orchestrator, adapter, and the API's CV-source path — all explicitly synthetic/adversarial/mocked, per `docs/CV_PIPELINE_FINDINGS.md`; no real-broadcast-footage tests currently pass (they skip, pending SoccerNet NDA access)

---

## 4. What's the current goal

**Both tracks are now "complete" in the sense of implemented-and-tested-at-their-appropriate-standard** — the StatsBomb/physics-ML track (RQ1-RQ5) is empirically validated against real match data; the CV track (Module 4) is internally validated against synthetic/adversarial tests. Neither is "done" in an absolute sense; two concrete, real gaps remain open on the StatsBomb side, and the CV side has one single, well-understood blocker.

1. **Recalibrate the instability detector's false-positive rate.** Flagged as a known issue since Milestone 14B: the detector's conservative "10% loss decrease" health heuristic has declined to certify a clean RQ4 verdict on every subsequent re-run (Milestones 21 and 23 both), even though direct probing confirms both models are genuinely healthy on those runs. Worth fixing before trusting this detector as a hands-off gate on future training runs, rather than continuing to rely on manual override.
2. **Move the train/val split from sample-level to match-level.** The split has been at the sample level since Milestone 7 — fine while no feature depended on cross-sample information, but Milestone 23's habit-memory work showed this is a real limitation once a feature needs match-level exclusion logic (only 4 of ~55 matches were training-bucket-eligible under the current conservative rule). Worth revisiting before any further habit-memory or cross-sample-dependent feature work.
3. **Unblock the CV track's real-data validation.** SoccerNet NDA/research-use access is still pending; no real broadcast footage has ever been processed by this pipeline. Every remaining CV validation gap (detection P/R, tracking ID-switches under real occlusion, calibration against a real lens, team classification under real jersey patterns, the shot classifier's real-world adversarial failure rate, real-hardware throughput) traces back to this one blocker — `docs/CV_PIPELINE_FINDINGS.md` §5 is the prioritized list of what to validate once it's lifted.

The following Milestone-13-era goals are now resolved and should not be re-opened as if still pending:
- ~~Wire the graph builder in for RQ4~~ — done, Milestone 12 onward; result trajectory documented in Research Findings, not treated as final.
- ~~Close the velocity gap~~ — partially done: the CV tracker (Milestone 26) produces real pixel velocity, explicitly labeled camera-motion-confounded; true calibrated player velocity still requires camera-motion compensation, not yet built.
- ~~Prepare the Bayesian habit layer~~ — done, Milestones 22-23; a real, honestly-reported null result under current data constraints, not an open task.
- ~~Full Digital Twin with substitution counterfactuals~~ — done, Milestone 20 (Oracle Substitution Validation); a hedged, confounded-but-real result on one match, not a closed/clean pass.
- ~~Eventually the computer-vision pipeline~~ — done through Milestone 34; fully built and internally validated, unvalidated on real data (see gap 3 above).

The guiding principle remains unchanged, now with a demonstrated track record behind it:  
**Validate the physics first. Keep every component independently testable. Answer the research questions with measurable evidence. Treat heuristic counterfactuals as exploratory probes, not final answers.** This discipline has repeatedly caught real, silent failures — a coordinate convention error (ADR-003→009), two separate silent training collapses (Milestone 12, Milestone 14), and a homography math trap that would have produced 4,242 m/s of fabricated player velocity had it shipped uncaught (Milestone 30) — precisely because every component was independently validated rather than assumed correct.

---

*This context document reflects the state of the repository after Milestone 34 (`docs/CV_PIPELINE_FINDINGS.md`). Any specific commit hash should be treated as an approximate pointer, not a verified reference, unless independently confirmed against `git log` at the time of reading.*
