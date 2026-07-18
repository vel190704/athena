# ADR-001: DeepHit over DeepSurv/Cox for Goal Probability Modeling

## Status
Accepted

## Context
Module 7 (Prediction & Uncertainty Engine) must produce a continuous, phase-by-phase
estimate of goal probability as a function of time-to-event. Classical survival models
(Cox Proportional Hazards, DeepSurv) are the default choice for time-to-event problems,
but they share a foundational assumption: the hazard ratio between any two covariate
configurations is constant over time (the "proportional hazards" assumption).

Football possession phases violate this assumption directly. The relative danger of a
given tactical state (e.g., a central overload, a high defensive line) is not a fixed
multiplier on baseline risk — it changes shape depending on match context. A central
overload in the 20th minute, against a settled defensive structure, carries a materially
different and *differently shaped* hazard curve than the same spatial configuration in
the 90th minute, when defensive discipline degrades, teams chase the game, and risk
tolerance shifts. This is a non-proportional, time-varying hazard problem, not a
constant-multiplier one.

Cox-based and DeepSurv models also assume a continuous, fully-observed event time and
handle censoring (turnovers, fouls, out-of-play — non-goal possession terminations) less
naturally than discrete-time competing-risk formulations.

## Decision
We use DeepHit (Lee et al.) instead of DeepSurv/Cox for goal probability estimation.

DeepHit:
- Directly models the discrete-time joint distribution of first-hitting-time, without
  assuming proportional hazards.
- Learns the hazard shape from data rather than constraining it to a fixed parametric
  form, allowing it to represent "90th-minute chaos vs. 20th-minute structure" as
  distinct hazard regimes.
- Natively supports right-censoring for non-goal possession terminations via its
  competing-risks / censored likelihood formulation.
- Uses a ranking loss to preserve calibrated relative risk ordering across time steps,
  which pairs well with the batched ensemble architecture described in Module 7.

## Consequences
- DeepHit requires discretizing time into bins, which introduces a granularity
  hyperparameter that must be validated against pass/possession duration statistics
  (to be tuned in Milestone 3).
- Loss of the closed-form interpretability that Cox's hazard ratios provide; this is
  mitigated by the async SurvivalSHAP explainability layer (see ADR-006).
- The ranking loss must be computed strictly within-ensemble-member (see Module 7,
  `[Ensemble, Batch, Features]` reshape) to avoid entangling gradients across ensemble
  members — this is an implementation constraint inherited directly from this decision.

## Alternatives Considered
- **Cox Proportional Hazards**: rejected — proportional hazards assumption is violated
  by football's context-dependent risk shape.
- **DeepSurv**: rejected — still inherits the proportional hazards assumption from the
  Cox partial likelihood it optimizes, despite using a neural net for the risk function.
- **Random Forest survival models**: not considered as a production candidate per
  project instructions (Section 6); may serve only as an explicitly-requested baseline.
