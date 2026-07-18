# ADR-008: Synthetic Validation Baseline for the Kalman Friction Filter

## Status
Accepted

## Context
Module 2's Causal Kalman Latent Friction filter estimates the rolling friction
coefficient `μ(t)` as a latent, time-varying state, using a causal predict → observe →
correct loop to avoid look-ahead bias. Per Section 5 (Strict Build Order), Milestone 1
requires validating this filter to within a 2% margin of error before any ML work
begins.

Real-world StatsBomb tracking data conflates multiple noise sources: tracking/velocity
measurement error, aerodynamic drag (governed by the drag coefficient `Cd`), spin
effects, surface irregularities, and genuine friction variation. If the filter were
validated directly against real data, a failure to hit the 2% target would be
underdetermined — it could stem from incorrect Kalman math (wrong Q/R tuning, incorrect
predict/correct ordering, a bug in the observation model) or from unmodeled real-world
aerodynamics, and there would be no way to distinguish the two from the validation
result alone.

## Decision
The Kalman filter is first validated on synthetic data with a known, exactly-specified
drag coefficient — `Cd = 0` for ground passes in this v1.0 baseline — before any
real-world aerodynamic noise is introduced.

With `Cd` fixed at a known value, the synthetic generator can compute the exact
noiseless kinematics from a chosen `true_mu`, add only measurement noise (Gaussian,
fixed magnitude) to simulate imperfect velocity tracking, and then check whether the
filter's posterior estimate of `μ` converges back to `true_mu` within 2%. Any
convergence failure in this synthetic setting is therefore attributable only to the
filter's own mathematical correctness (Kalman gain computation, Q/R tuning, or
predict/correct causal ordering) — not to unmodeled physics.

This isolates "does the Kalman math work" from "does the physics model match reality,"
and the two questions are validated separately and in sequence, matching the project's
decoupled classical-physics-vs-statistical-inference philosophy (Section 3).

## Consequences
- The Milestone 1 gate (test_friction.py) only proves the filter is mathematically
  sound under the synthetic assumptions (`Cd = 0`, Gaussian measurement noise, slowly
  varying `μ`). It does NOT prove the filter will perform well on real tracking data,
  where aerodynamic drag is nonzero for passes struck with air time, and where
  measurement noise may not be Gaussian.
- A follow-up milestone (post-Milestone-1, before real data is used in production) must
  extend synthetic validation to nonzero, known `Cd` values to validate the filter's
  behavior when drag is present but still exactly known — before finally validating
  against real StatsBomb data where `Cd` is not directly observable.
- `process_noise_q` and `measurement_noise_r` are tuned against synthetic data for this
  milestone; they are not assumed to transfer directly to real data and will need
  empirical retuning once real tracking noise characteristics are available (see
  `kalman_friction.py` constructor docstring).

## Alternatives Considered
- **Validate directly against real StatsBomb tracking data**: rejected for Milestone
  1 — conflates filter correctness with unmodeled aerodynamic/tracking noise, making
  failures undiagnosable.
- **Validate using a fully analytical (noiseless) synthetic baseline**: rejected as
  the sole test — a noiseless test would trivially pass even with subtly incorrect
  Kalman gain computation, since there would be no observation noise for the filter to
  actually filter. Measurement noise must be present for the test to be meaningful.
