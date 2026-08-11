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

## Update: Follow-Up Milestone — Nonzero-Cd Synthetic Extension Done; Real-Data Validation Blocked by a Genuine Data Gap

This ADR's own Consequences section named two required follow-up steps: extend
synthetic validation to a nonzero-but-known `Cd`, then validate against real
StatsBomb data. Both were attempted. The first is done. The second could not be —
not from lack of effort, but because the open dataset genuinely does not contain
the signal this filter's observation model requires.

### Step 0: real-data availability, checked directly (not assumed)

Verified against ~33,000 real cached `Pass` events across 30 matches
(`data/raw/*_events.json`):

- **`pass.height`** reliably distinguishes Ground Pass (24,592) / Low Pass (3,579)
  / High Pass (4,783) — usable to scope any real-data attempt to ground passes
  only, per the filter's own `Cd=0`-for-ground-passes design (`kalman_friction.py`
  docstring, System Assumption A3). Aerial passes were correctly excluded from
  consideration throughout, since this filter was never designed or validated for
  them.
- **`pass.length`** is a real, exact geometric distance — verified byte-for-byte
  against `hypot(end_location - location)` (max discrepancy `7e-6`, pure
  floating-point noise, across 3,196 checked passes). A clean distance signal.
- **`duration`** (top-level event field) is present on every pass and, treated as
  a flight-time proxy, implies a plausible average speed (`length / duration`) for
  the large majority of ground passes: median 12.7 m/s, and 98.47% fall in a
  physically sane 3–30 m/s range for a struck, decelerating ball. But it is
  genuinely noisy in the tail, not clean ground truth — ~0.5% of ground passes
  imply speeds above 40 m/s (up to a nonsensical 2,170 m/s for a 4.46m pass
  logged at `duration=0.0021s`), almost certainly near-simultaneous event
  timestamps from deflections or rapid exchanges rather than real ball flight
  time. Any real-data use of `duration` would need explicit outlier filtering.
- **No field anywhere provides an initial or final velocity, or any speed/power
  value.** Checked exhaustively: every top-level key on a `Pass` event
  (`counterpress, duration, id, index, location, minute, off_camera, out, pass,
  period, play_pattern, player, position, possession, possession_team,
  related_events, second, team, timestamp, type, under_pressure`) and every key
  inside the `pass` sub-object (`aerial_won, angle, assisted_shot_id, body_part,
  cross, cut_back, deflected, end_location, goal_assist, height, inswinging,
  length, miscommunication, no_touch, outcome, outswinging, recipient,
  shot_assist, straight, switch, technique, through_ball, type`) — none is a
  velocity or speed. The linked `Ball Receipt*` event (via `related_events`) was
  also checked and carries only `location` and `timestamp`, no velocity. This is
  StatsBomb open **event** data, not continuous ball tracking: there is no frame
  giving the ball's speed at any point along a pass, only its start location, end
  location, and total elapsed time.

**This is the decisive finding.** `observe_mu_from_pass(v_initial, v_final,
distance)` needs two *independently measured* velocities. What the real data
provides is distance `d` and elapsed time `t`, from which only one quantity is
derivable: average velocity `d/t`. Under this filter's own constant-deceleration
model, `d = v_i*t - 0.5*mu*g*t²` has two unknowns (`v_i`, `mu`) and one equation
per pass — underdetermined, and pooling across many passes doesn't help, because
every pass has its own independently unknown `v_i` (a harder strike, a lofted
weight-of-pass, etc.), so each additional pass adds one equation *and* one new
unknown. Closing the gap would require assuming a value for `v_i` or `v_f` (e.g.
"the ball arrives at rest") — exactly the fabricated-input this task was
explicitly told not to do. **Per the task's own instruction, this stops Step 2
here rather than proceeding on an invented input.**

**What would actually close this gap:** real ball-tracking data with a genuine
velocity or multi-frame position signal (e.g. optical tracking data of the kind
StatsBomb's own commercial tracking product, or a vendor like Second
Spectrum/Tracab, provides) — this specific open dataset does not include it, and
no amount of additional feature-engineering on the event data recovers a
two-endpoint velocity measurement that was never recorded.

### Step 1: nonzero-but-known Cd synthetic extension — done, real result

Pure synthetic, independent of Step 0's finding — `Cd` expressed in the same
units as `mu` (`a_drag = Cd*g`), so the combined noiseless model stays the exact
closed form `v² = u² − 2*(mu+Cd)*g*d`, deliberately the smallest real extension of
the `Cd=0` baseline rather than a different velocity-dependent drag ODE.
`observe_mu_from_pass` (unmodified — it has no notion of drag) necessarily
returns `mu+Cd` when fed a drag-affected final velocity; the known `Cd` is
subtracted in the test, outside the filter, before each `correct()` call — the
same way a real pipeline would if `Cd` were independently known. This isolates
exactly what this follow-up milestone asks: does the filter's own Kalman math
still converge correctly once a second, known physical effect is present and
must be explicitly accounted for, rather than `Cd=0` having accidentally been the
only case that ever worked.

**Result** (`production/tests/test_friction.py::test_kalman_friction_filter_converges_within_2_percent_with_known_nonzero_drag`,
fixed seed 42, `true_mu=0.35`, `true_Cd=0.15`, wrong prior `mu_0=0.5`, 50 passes):
posterior converged to **`mu=0.35098`**, relative error **0.280%** — inside the
2% gate (looser than the original Cd=0 run's 0.336%, but still well clear).
**PASSED.**

### Step 3: honest conclusion

- Step 1 (nonzero-known-Cd synthetic extension): **done, passed**, real measured
  0.280% error — confirms the Kalman math itself is not merely correct by
  accident in the `Cd=0` special case.
- Step 2 (real StatsBomb data validation): **not achievable with this dataset**,
  a genuine data-availability gap rather than lack of effort. RQ3's literal
  "pass landing error" success criterion therefore remains unmeasured against
  real trajectories, and stays that way until a data source with an actual
  velocity/tracking signal is available — StatsBomb's open event data structurally
  cannot supply it, no matter how the existing fields are combined.
- This closes ADR-008's Consequences-section checklist item as far as it can
  currently be closed: the nonzero-Cd half is done and real; the real-data half
  is honestly reported as blocked, not silently skipped or faked.
