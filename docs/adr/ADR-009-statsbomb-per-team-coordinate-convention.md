# ADR-009: StatsBomb Data Is Already Per-Actor Oriented -- No Direction Flip Needed

## Status
Accepted (supersedes the Period-2 coordinate-flip decision made during Milestone 10)

## Context
Milestone 10 assumed StatsBomb records a single, shared, physically-fixed
0-120 x 0-80 coordinate system for the whole match, and that which end a
team defends within that shared system is unknown and must be inferred per
team per period (a coin toss decides it, and it is not guaranteed correct
even in period 1). Under that assumption, a team's period-2 attacking
direction was forced to be the exact opposite of its period-1 direction
(teams swap ends at half-time), and `feature_extractor.py` flipped x
whenever a team's inferred direction was -1.

Empirical testing during Milestone 10 (using `Shot` and `Pass`/`Carry` event
`location` fields) found that this assumption does not hold: independently
inferring a team's direction in period 1 and period 2 disagreed with the
guaranteed-opposite rule in 37 of 40 team-periods checked across 20 real
matches. Three independent checks confirmed why: StatsBomb records event
`location` relative to the ACTING team's own attacking-left-to-right
perspective, in both halves, not a shared physically-fixed frame --

1. **Goal kicks** (always taken from the kicking team's own six-yard box)
   cluster at x ~ 6-7 for every team, in every period, across the sample
   match -- physically impossible for two teams defending opposite ends of
   one shared pitch, but exactly what a per-actor-oriented convention
   produces.
2. **Turnover coordinate mirroring**: a team's pass end-location and the
   opponent's immediately-following interception location matched exactly
   under `(x, y) -> (120 - x, 80 - y)` in 5/5 sampled turnovers -- the
   coordinate frame flips 180 degrees at the exact moment possession
   changes team, consistent with each event being recorded relative to
   whichever team acted.
3. **Own-goal event pairs**: the "Own Goal For" (benefiting team) and "Own
   Goal Against" (conceding team) records of the same physical goal were
   also exact 180-degree mirrors of each other.

This ADR's Step 0 extended the check to the 360 freeze-frame data itself --
a structurally separate part of the StatsBomb JSON from event `location`,
and the actual source of `player_pos` fed to the physics engine. Using each
frame's `keeper` flag, goalkeeper x-coordinates were aggregated across all
20 cached matches (~20,000 keeper observations): teammate-flagged keepers
(same team as the frame's acting player) clustered at mean x ~ 10.7, and
opponent-flagged keepers at mean x ~ 112, and this held near-identically in
period 1 (10.6 / 112.6) and period 2 (10.8 / 111.9). The freeze-frame data
follows the exact same per-actor convention as event `location` -- this was
verified, not assumed to transfer.

## Decision
Remove the forced period-2 coordinate flip entirely. Trust raw StatsBomb
`location` and 360 freeze-frame coordinates as already per-actor-oriented:
the acting player's own team already appears to attack toward increasing x
in both periods, with no further transformation required. The only
coordinate transformation that remains is the existing ADR-002 rescale
(120x80 raw StatsBomb units -> this project's 100x68 grid -- verified as
the literal constants in `feature_extractor.py`'s `PITCH_LENGTH`/
`PITCH_WIDTH` and `statsbomb_io.py`'s `X_SCALE`/`Y_SCALE` before writing
this ADR).

`production/src/pipeline/direction.py` is retained in the codebase, not
deleted, but is no longer called from `feature_extractor.py`. It correctly
computes what it was designed to compute (an attacking-direction estimate
from shot/touch locations); that quantity simply turned out not to answer
the real-world-attacking-direction question this project actually needed,
because the input data was already normalized to the acting team's own
frame before the heuristic ever saw it. Its near-universal
period-1-equals-period-2 result across 37/40 real team-periods is itself
part of the evidence for this ADR's decision, not a bug in the heuristic --
the heuristic was measuring a property of the data's encoding, not of the
match's real-world geometry.

## Consequences
- Period 2 frames are valid for feature extraction with zero additional
  transformation -- simpler than Milestone 10's design, and correct instead
  of systematically mis-orienting roughly half the dataset.
- `feature_extractor.py` no longer takes a `direction` parameter or calls
  `direction.py`; every frame passes straight from the ADR-002 rescale into
  the physics engine.
- This conclusion is specific to StatsBomb's event/360 data convention. A
  future computer-vision/broadcast pipeline (Module 4, README Section 3)
  extracting raw pixel coordinates from video WILL need real per-half,
  per-team direction handling -- that data will not arrive pre-normalized
  to an acting-team frame the way StatsBomb's does. This ADR's "no flip
  needed" conclusion must not be read as a general claim about all future
  data sources.
- Any future StatsBomb-derived feature that assumes a shared, physically-
  fixed coordinate frame (e.g. reconstructing true home/away end-of-pitch
  geometry, or overlaying StatsBomb coordinates on broadcast video) must
  account for this per-actor convention explicitly; it is not recoverable
  from the data without external knowledge of true kickoff-end assignment.

## Alternatives Considered
- **Keep the Milestone 10 direction-inference/flip logic, but only apply it
  when the two periods' independent estimates disagree**: rejected -- the
  37/40 disagreement rate combined with the freeze-frame keeper evidence
  shows the independent per-period estimate is not measuring real-world
  attacking direction at all in this data; conditioning on it would still
  be conditioning on a signal that doesn't mean what it was assumed to mean.
- **Delete `direction.py` entirely**: rejected -- the module is correctly
  implemented for what it computes, and keeping it (with its docstring
  updated to explain this history) preserves the evidence trail and avoids
  losing working code that may be directly reusable once a data source with
  a genuinely shared coordinate frame (e.g. Module 4's CV pipeline) needs
  exactly this kind of inference.
