PROJECT ATHENA: Tactical Digital Twin & Prediction Engine

Version: 5.1 (Research-Grade HPC Architecture Blueprint)
Status: Two tracks implemented, each validated to its own standard. StatsBomb/physics-ML
track (Modules 1-3, 5-8): implemented AND empirically validated against real match data
(8,074 samples, ~55 matches, 12 competitions) -- see docs/RESEARCH_FINDINGS.md (Milestone 24).
CV/Module 4 track: implemented AND internally validated on synthetic/adversarial/mocked
tests only, NOT YET validated on real broadcast footage, blocked on SoccerNet NDA access --
see docs/CV_PIPELINE_FINDINGS.md (Milestone 34). See context.md for the fast-orientation
summary; this document is the full technical reference and both are kept in agreement.
1. Project Overview

Project Athena is a research-grade, end-to-end AI system designed to ingest live football (soccer) match telemetry and generate real-time tactical intelligence. It treats the football match as a differential game, mapping how physical fatigue, environmental friction, player biomechanics, and referee thresholds actively dictate tactical spaces. 

The system produces three core real-time outputs:

    Phase-by-Phase Goal Probability: Continuous time-to-event modeling (DeepHit Survival Analysis) predicting imminent threat.
    Tactical Cheat Sheets: Real-time spatial weak-spot detection using velocity-aware pitch control and Bayesian habit blending.
    Digital Twin & Counterfactual Simulator: Forward-projecting what-if engine (e.g., "What happens to xT if Player X is substituted?").

2. System Philosophy & Research Framing

This project is framed as a research platform to answer explicit Research Questions (RQs) under strict System Assumptions (As). It is an estimator of tactical states and likely consequences, not an "omniscient football brain."
System Assumptions (A1-A4)

     A1: Player acceleration is approximated using a linear force-velocity relationship (aeff​=amax​×(1−v/vmax​)
    ).
     A2: Ground friction μ(t)
     varies slowly compared to pass duration.
     A3: Ball spin (Magnus effect) is ignored for v1.0 of the physics engine.
     A4: Reaction time is constant within a possession phase. (Note: See ADR-007. This is a known v1 simplification. Fatigue dynamically penalizes aeff​
    , but reaction time degradation is deferred to v2 to isolate ODE validation).

Research Questions & Success Criteria (RQ1-RQ5)

All five RQs below have been investigated at least once and have working, honestly-reported
findings in docs/RESEARCH_FINDINGS.md (Milestone 24) as of Milestone 23's data. None should
be read as permanently settled -- RQ4 in particular has reversed direction twice as data
scale and training stability changed.

     RQ1: Can velocity-aware pitch control improve short-term goal probability calibration? (Success: Brier Score improvement ≥
     X%). FINDING: well-calibrated Brier scores achieved in absolute terms (e.g. 0.0942/0.1588
     at 15s/30s, Milestone 14B), but no non-physics-informed baseline was ever run for
     comparison -- the "improvement" half of this question remains unmeasured, named
     explicitly rather than implied.
     RQ2: Does Bayesian tactical memory improve prediction over purely live tracking? (Success: Calibration error reduction). FINDING:
     not supported -- Brier Score got slightly worse (0.0950 vs 0.0942 at 15s, Milestone 23),
     under a 68% cold-start fallback rate (only 4 of ~55 matches were training-bucket-eligible)
     -- a null result for this specific, heavily constrained implementation, not a general verdict.
     RQ3: Does latent friction estimation improve pass trajectory prediction? (Success: Pass landing error reduction ≤
     X meters). FINDING: the Kalman filter's synthetic convergence is validated (0.336% error
     after 50 passes, 5x tighter than the 2% gate), but this validates only the filter's
     mathematical correctness -- real pass-landing-error reduction has never been measured.
     RQ4: Can graph-based team representations outperform handcrafted tactical features? (Success: AUROC/Brier improvement over MLP). FINDING:
     reversed direction twice (single-competition: MLP wins; multi-competition as first
     measured: GNN "wins" only via an invalid MLP silent collapse; corrected: MLP wins again
     once both models are confirmed genuinely healthy). Current best-evidence conclusion:
     MLP outperforms the GNN at the current 8,074-sample scale -- presented as a trajectory,
     not a single settled number.
     RQ5: Can counterfactual simulations predict the tactical effects of substitutions? (Success: Predicts real-world xT shift within predefined bounds). FINDING:
     two evidence threads. Heuristic tactical-alignment probes (Milestone 13/14/14B) gave
     mixed, model/scenario-dependent results. Oracle Substitution Validation against real
     historical subs (Milestone 20, one match) is hedged -- 9 of 10 real substitutions had
     overlapping ±2-minute windows, leaving only 1 genuinely unconfounded observation. RQ5
     remains largely open on its literal question.

Risks & Mitigations (R1-R5)

     R1 (Tracking noise): Mitigated by Kalman smoothing.
     R2 (Sparse labels): Mitigated by DeepHit Survival Analysis.
     R3 (Concept drift): Mitigated by online updating.
     R4 (GPU latency): Mitigated by fixed grid and sparse masking.
     R5 (Limited tracking data): Mitigated by event replay simulation.

3. Core Mathematical Engines (Decoupled Architecture)

The system strictly separates classical deterministic physics from statistical ML inference for interpretability, modularity, and independent validation. The ML models consume physics outputs as immutable feature layers.
Module 1: The Spatial Mathematics Core (PyTorch GPU)

     Biomechanical Force-Velocity Curve: Acceleration capacity decays as current velocity approaches max speed. This transforms the time-to-intercept calculation into an Ordinary Differential Equation (ODE). 
     Analytical ODE Solution: Instead of numerical integration (RK2/Heun), the production CUDA kernel uses the closed-form analytical solution: x(t)=vmax​t−(vmax​−v0​)(vmax​/amax​)(1−exp(−amax​t/vmax​))
    , reducing GPU latency by 40%.
     Sparse Masked Grid (Triton Compliance): Dynamic Adaptive Mesh Refinement (AMR) is abandoned to prevent Triton memory thrashing. Before the kernel launches, indices where distance_to_ball <= 30m are extracted. Only these ~2,000 sparse coordinates are passed to the solver, turning compute complexity from O(100×68×22)
     to 
    O(2000×22)
    . A binary spatial mask ensures static tensor shapes 
    [Batch, 100, 68].
     GNN Edge Feature Extraction: The continuous pitch control field is mapped to discrete GNN edges using bilinear interpolation (grid_sample) along the 2D line segment between Player A and B to compute the integrated pitch control potential along that passing lane.

Module 2: Physics & Environmental Engine

     Causal Kalman Latent Friction (μ(t)
    ): Rolling friction is treated as a latent, time-varying state variable. To prevent look-ahead bias, a standard Kalman Predictor (time update) projects 
    μ
     forward to time 
    t
     for the physics solver. 
    Then, the Kalman Corrector (measurement update) runs using the newly observed pass. Cd​
     is fixed, and the filter infers the residual as 
    μ
    .
     Synthetic Validation Baseline: Per ADR-008, the Kalman filter is validated on synthetic data where Cd​
     is known exactly (or zero for ground passes) to isolate the filter's mathematical correctness before introducing real-world aero noise.
     Fatigue Multipliers: Player acute fatigue dynamically penalizes their aeff​
     in Module 1.

Module 3: Streaming & Feature Store Architecture

     Ingestion: Apache Kafka / Redpanda. StatsBomb JSONs replayed sequentially to simulate live feeds.
     Persistence: TimescaleDB (time-series coordinates), Redis (low-latency live state).
     Feature Store: Feast to ensure zero training-serving skew. Decoupled deterministic physics outputs are passed into Feast as immutable state feature layers.

Module 4: Computer Vision Pipeline (Decoupled Phase 4)

     STATUS (Milestone 34): built and internally validated, NOT YET validated on real
     broadcast footage. The ML prediction engine (Phases 1-3) ingests provided tracking data
     (StatsBomb 360) and remains the empirically-validated track. The CV pipeline (implemented
     as YOLOv8m, not the originally-planned YOLOv9, + ByteTrack + a hand-specified
     cv2.findHomography calibration, not SoccerNet's own calibration tooling) ran fully
     parallelized and decoupled from the ML launch date, exactly as planned -- detection,
     tracking, calibration, team classification, ball detection, shot/camera-cut
     classification, pipeline orchestration, and a tensor-contract adapter (Milestones 25-33)
     are all built and passing synthetic/adversarial/mocked tests. The entire track remains
     blocked on SoccerNet NDA access for any real-broadcast-footage validation -- see
     docs/CV_PIPELINE_FINDINGS.md (Milestone 34) for the full component-by-component status
     and the prioritized validation roadmap once that access is unblocked.

Module 5: Graph & Relational Engine

     Player/Team Embeddings: Contrastive learning / self-supervised graph representation learning represents players as vectors. Cold-start fallbacks use Team Tactical Identity Embedding (aggregate passing networks).
     Graph Neural Networks (GNN): Represents 11 players as graph nodes. 
     Learned Cohesion Penalty: Simulated substitutions inject a penalty into GNN edge weights. The decay function of this penalty is learned/calibrated from historical substitution data, not hardcoded.

Module 6: Historical Memory & Bayesian Habit Layer

     Bayesian Updating: Posterior = Prior * Evidence. Historical spatial heatmaps are blended with live coordinates to skew future movement probabilities. Includes cold-start fallbacks for sparse historical data.

Module 7: Prediction & Uncertainty Engine

     DeepHit (Survival Analysis): Captures non-proportional, time-varying hazards (90th-minute chaos vs 20th-minute structure). Handles right-censoring for non-goal possession terminations (turnovers, out-of-play, fouls).
     Batch Ensembles & Ranking Loss: Ensemble forward passes are batched. The DeepHit ranking loss is computed strictly within the same ensemble member (reshaped to [Ensemble, Batch, Features]) to prevent entangled gradients. IMPLEMENTATION NOTE (ADR-004, Milestone 21): what was actually built is a 5-member Deep Ensemble (fully independent models, ~5x parameters/compute), not a true shared-weight Batch Ensemble (Wen et al.) -- a simpler, statistically valid but heavier alternative, deliberately named DeepEnsembleDeepHit to avoid confusion. True Batch Ensembles remain future work if inference latency becomes a real constraint.
     Asynchronous Explainability: SurvivalSHAP requires hundreds of forward passes. It is moved to an async background worker. Real-time streams push raw hazard scores; every 5 seconds, the worker computes SHAP and pushes the LLM textual summary to a secondary WebSocket channel.

Module 8: Digital Twin & Counterfactual Simulator

     Counterfactual Estimator: Answers "what changes in expected threat if X changes?" rather than claiming exact future outcomes.
     Oracle Substitutions Validation: Backtests counterfactual simulations against historical StatsBomb data. If a sub occurred at minute 70, the simulator runs minutes 68-70 to verify it predicts the real-world xT shift post-substitution.

4. Engineering Tech Stack & MLOps

     Language: Python 3.11+
     Core Math/ML: PyTorch (Primary GPU math), PyTorch Geometric (GNNs)
     Data Pipelines: Apache Kafka, TimescaleDB, Redis, Feast
     MLOps: MLflow (Experiment tracking), Triton Inference Server (Serving), DVC (Dataset Versioning), GitHub Actions (CI/CD)
     UI/Backend: FastAPI, WebSockets, React/Streamlit
     Repository Structure: Strict separation of research/ (notebooks/prototypes) and production/ (tested modules, APIs, serving). docs/adr/ for Architecture Decision Records.

5. Strict Build Order & Milestones

    Milestone 1 (Module 2 - Physics/Kalman): Must be validated first. The Kalman Filter must back-calculate the friction coefficient of a rolling ball within a 2% margin of error on synthetic data before any ML is touched.
    Milestone 2 (Module 1 - Spatial Core): Implement the analytical ODE solver with sparse masked indexing. Map spatial fields to GNN edges via bilinear sampling. Train a simple MLP baseline.
    Milestone 3 (Module 7 - DeepHit): Introduce DeepHit without the GNN (just using spatial features). Validate Brier Score and right-censoring labels.
    Milestone 4 (Module 5 - GNN): Add PyTorch Geometric layers. Validate against the MLP baseline.
    Milestone 5+ (Modules 3, 4, 6, 8): Streaming, CV, Bayesian Memory, and Digital Twin are layered in after the core prediction pipeline is validated.

6. Instructions for AI Assistants

When assisting with Project Athena, adhere strictly to the following rules:

    No shortcuts: The target architecture is DeepHit, GNNs, and biomechanical pitch control. Do not suggest simpler models (e.g., Random Forest) unless explicitly asked to establish a baseline.
    Mathematical Rigor: Respect the Red Team fixes: Analytical ODEs (not RK2), Fixed Grid + Sparse Masking for Triton, Bresenham/bilinear interpolation for GNN edges, Causal Kalman filtering (Predict -> Solve -> Correct) for friction, and Batch Ensemble gradient disentanglement.
    Decoupled Architecture: Strictly separate classical deterministic physics from statistical ML inference. Physics engines produce immutable feature layers; ML models consume them but cannot modify them.
    Systems Engineering: Treat the project as a professional open-source product. Always suggest ADRs, unit tests, and modular PyTorch implementations. Ensure vectorized/GPU-ready code. Do not use native Python loops for grid calculations.
    Research Mindset: Frame suggestions as working hypotheses to be validated experimentally, not absolute truths. ODE solvers and GNN architectures are experimental candidates, not hardcoded truths.