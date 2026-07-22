# ADR-002: StatsBomb Coordinate Rescaling at the Ingestion Boundary

## Status
Accepted

## Context
StatsBomb event and 360 freeze-frame data is recorded on a fixed `[0, 120] x
[0, 80]` unit grid. This is an arbitrary StatsBomb convention, not a
real-world measurement: every match is normalized onto this same 120x80
space regardless of the actual stadium's pitch dimensions (real pitches vary,
roughly 100-110m x 64-75m under IFAB law).

The biomechanical physics engine (`BiomechanicalPitchControl` in
`production/src/spatial/control.py`) and every downstream spatial feature
(`production/src/pipeline/feature_extractor.py`) operate on a `100 x 68`
unit grid. All of this module's physical constants -- `max_speed` (m/s),
`max_accel` (m/s^2), `mask_radius` (30m), the `NEAR_BALL_RADIUS` (15m) and
`FINAL_THIRD_X` (66.0) feature thresholds -- are defined in that same
100x68 space, on the implicit assumption that one grid unit is one meter.

If raw StatsBomb coordinates (120x80) were ever passed into this pipeline
un-rescaled, or rescaled inconsistently in more than one place, the ODE
solver and every downstream feature would silently operate on the wrong
spatial scale relative to the physiological constants above -- e.g. a
30-unit mask radius would cover a very different fraction of a 120-wide
pitch than of a 100-wide one, and Newton-Raphson convergence and feature
thresholds would no longer mean what their names say.

## Decision
All raw StatsBomb coordinates MUST be rescaled at the ingestion boundary,
and nowhere else. This happens exactly once, inside
`parse_360_frame` (`production/src/ingestion/statsbomb_io.py`), using the
fixed factors already implemented there:

```python
X_SCALE = 100.0 / 120.0
Y_SCALE = 68.0 / 80.0
```

applied to every coordinate `parse_360_frame` touches: the ball position
(taken from the event's own `location` field) and every player position in
the freeze-frame. No downstream physics or ML module is allowed to read or
operate on raw StatsBomb units -- by the time a coordinate reaches
`BiomechanicalPitchControl` or `feature_extractor.py`, it is already in the
project's internal 100x68 space.

This is a rescaling to match this project's internal grid convention, NOT a
claim that StatsBomb's 120-unit axis corresponds to an exact 100-meter
real-world pitch length (nor 80 units to an exact 68m width). Different
stadiums have different real dimensions; StatsBomb normalizes all of them
into the same 120x80 space regardless. `X_SCALE`/`Y_SCALE` map that
normalized space onto this project's chosen 100x68 internal grid -- they do
not recover the true meter dimensions of any specific stadium.

## Consequences
- Prevents silent precision loss and scale mismatches in the ODE solver:
  every physiological constant (`max_speed`, `max_accel`, `mask_radius`,
  `reaction_time`) and every feature threshold (`NEAR_BALL_RADIUS`,
  `FINAL_THIRD_X`) is defined once, in the same 100x68 space that all
  ingested coordinates are guaranteed to already be in.
- Because the rescaling is an approximation of real stadium geometry (see
  Context), any feature that claims a specific meter-based physical meaning
  (e.g. "15m from the ball") inherits StatsBomb's 120x80 normalization error
  for stadiums whose true dimensions differ from the implied ~101.6m x 68m
  (120 x (100/120) x ... -- i.e. the scaling is calibrated to the ratio, not
  to a verified average pitch size). This is an accepted, bounded
  approximation for v1, not a correctness bug.
- Any future ingestion path added for a different data provider (tracking
  vendor, different event schema) must perform its own explicit rescaling
  into this same 100x68 space at its own ingestion boundary -- it must not
  assume StatsBomb's `X_SCALE`/`Y_SCALE` apply to a different provider's raw
  coordinate convention.
- Unit tests that construct synthetic tensors directly (e.g.
  `test_spatial.py`) are already written directly in the 100x68 space and
  require no rescaling; only real StatsBomb-sourced data passes through
  `parse_360_frame`'s scaling step.

## Alternatives Considered
- **Rescale downstream, at the point of use (e.g. inside
  `BiomechanicalPitchControl` or `feature_extractor.py`)**: rejected --
  would require every downstream consumer to know it's receiving raw
  StatsBomb units and rescale correctly and consistently, multiplying the
  number of places the scale factors are duplicated (and could drift out of
  sync).
- **Keep raw StatsBomb 120x80 units throughout the pipeline and rescale the
  physical constants instead**: rejected -- physical constants
  (`max_speed`, `max_accel`, `g`) are grounded in real physics (m/s, m/s^2)
  and are meant to be reusable across data providers; rescaling them per
  provider would be more error-prone than rescaling coordinates once at
  ingestion.
