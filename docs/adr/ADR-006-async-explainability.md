# ADR-006: Asynchronous SurvivalSHAP for DeepHit Explainability

## Status
Accepted

## Context
Module 7 pairs DeepHit with SurvivalSHAP to explain which features are driving a given
hazard estimate. SHAP-family methods require running hundreds of forward passes per
explanation (perturbed-input evaluations against the coalition/background sample set).
The real-time hazard stream, by contrast, must publish updated goal-probability estimates
at the cadence of the live tracking feed, on the order of the frame rate — a budget that
hundreds of forward passes per update cannot fit inside without either dropping frames
or making every consumer of the hazard stream wait on the slowest explanation pass.

Computing SHAP synchronously, inline with the hazard inference path, would couple the
latency of an interpretability feature to the latency of the core real-time prediction
path — an interpretability nice-to-have should never be able to degrade the primary
signal.

## Decision
SurvivalSHAP is moved to an asynchronous background worker, decoupled from the
real-time inference path:

- The real-time stream pushes raw hazard scores to clients as soon as they are computed,
  with no dependency on SHAP.
- The background worker independently computes SHAP values on a fixed cadence (every 5
  seconds) rather than per-frame.
- The worker pushes the resulting LLM-generated textual summary to a **secondary**
  WebSocket channel, kept separate from the primary hazard-score channel so that slow or
  failed explainability computation can never block or degrade the primary signal.

## Consequences
- Explanations are necessarily stale relative to the live hazard score they describe (up
  to a 5-second lag). This is an accepted trade-off — explainability is diagnostic and
  human-facing, not a control input to any downstream automated decision.
- The background worker needs its own failure isolation (retries, timeouts) so that a
  SHAP computation failure never propagates to the primary WebSocket channel or the
  hazard inference path.
- Consumers of the textual summary channel must be built to tolerate a
  slower-than-real-time cadence and must not assume 1:1 correspondence between a hazard
  update and an explanation update.

## Alternatives Considered
- **Synchronous inline SHAP**: rejected — couples explainability latency to real-time
  hazard latency, risking dropped frames or stream stalls.
- **Reduced-sample SHAP (fewer forward passes) computed synchronously**: rejected —
  even a reduced sample count does not reliably fit the real-time budget, and reducing
  sample count degrades explanation fidelity in a way that is hard to bound.
- **Precomputed/offline-only explanations**: rejected — would not reflect the live
  in-match feature state that SurvivalSHAP is meant to explain.
