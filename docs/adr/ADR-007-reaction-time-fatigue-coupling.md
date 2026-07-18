# ADR-007: Reaction Time Fixed, Acceleration Capacity Fatigue-Coupled (v1.0 Asymmetry)

## Status
Accepted

## Context
System Assumption A4 fixes reaction time as constant within a possession phase for
v1.0 of the physics engine. At the same time, Module 2 specifies that fatigue
multipliers *dynamically* penalize a player's effective acceleration capacity
(`a_eff`, per the force-velocity relationship in A1) within that same module.

On the surface this looks like an inconsistency: fatigue is a real physiological
process that degrades both reaction time and acceleration capacity in reality, so why
model one as dynamic and the other as fixed in the same milestone?

## Decision
For v1.0, reaction time is treated as constant within a possession phase (deferred
dynamic modeling to v2), while acceleration capacity (`a_eff`) IS dynamically penalized
by fatigue within Module 2/Module 1's ODE solver.

This asymmetry is intentional and is acceptable specifically because it isolates ODE
validation:

- The closed-form analytical ODE solution for time-to-intercept (Module 1) is a
  function of `v_max`, `v_0`, and `a_max` (via `a_eff`). Fatigue-coupled acceleration
  enters this closed-form solution as a *continuous, differentiable* parameter — it
  changes the shape of the trajectory curve without changing the *structure* of the
  ODE or requiring a discrete-event reaction delay term.
- Reaction time, by contrast, enters the model as a discrete latency offset applied
  *before* the ODE integration window begins (a player doesn't start accelerating
  until reaction time elapses). Making reaction time dynamic within a phase would
  require re-deriving the trigger condition and re-solving the ODE's initial
  conditions mid-phase, which is a structurally different — and more complex —
  modeling problem than tuning a coefficient inside an already-closed-form solution.
- Milestone 1 and Milestone 2 exist specifically to validate the ODE/Kalman math in
  isolation before any ML is introduced (Section 5, Strict Build Order). Coupling
  fatigue into `a_eff` lets us validate that the closed-form ODE solution correctly
  responds to a continuously-varying physical input, without simultaneously
  introducing a second, structurally distinct dynamic (discrete reaction-time
  re-triggering) that would confound whether a validation failure comes from the ODE
  itself or from the reaction-time coupling logic.

In short: fatigue-on-acceleration is a same-structure parameter change to a validated
closed-form solution; fatigue-on-reaction-time would be a structural change to the ODE's
initial/boundary conditions. Only the former is in scope for isolating ODE correctness
in v1.0.

## Consequences
- v1.0 will underestimate how reaction degrades late in a possession phase or match
  (e.g., a fatigued defender reacting slower to a through ball), which may show up as
  systematic error in interception-time predictions during high-fatigue, late-match
  situations. This is an accepted, documented limitation, not an oversight.
- v2 must revisit A4 and introduce dynamic reaction-time degradation, at which point
  the ODE's initial-condition handling will need to be extended to support a
  time-varying trigger delay.
- This ADR should be referenced by any future PR that attempts to "fix" reaction time
  to be dynamic in v1 — that work belongs in v2, not as an incidental fix during
  Milestone 1/2.

## Alternatives Considered
- **Make both reaction time and acceleration dynamic in v1.0**: rejected — conflates
  two structurally different modeling changes in the same validation milestone,
  making ODE validation failures harder to diagnose.
- **Make both fixed in v1.0 (no fatigue coupling at all)**: rejected — Module 2's
  stated purpose includes validating that fatigue meaningfully penalizes physical
  capacity; deferring all fatigue effects to v2 would leave Module 2 under-scoped.
