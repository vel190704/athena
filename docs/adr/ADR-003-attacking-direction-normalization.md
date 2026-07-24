# ADR-003: Attacking-Direction Inference and Coordinate Normalization

## Status
Accepted

## Context
StatsBomb records raw pitch coordinates in a fixed 0-120 x 0-80 space (see
ADR-002), but does NOT normalize which end of the pitch each team is
attacking. Which end a team starts on is decided by a real-world coin toss
before kickoff, and is not recoverable from the data by assumption. Critically,
this is NOT a period-2-only problem: period 1's attacking direction is just
as unknown a priori as period 2's -- a naive "period 1 needs no correction,
only period 2 does" assumption (a common shortcut in ad-hoc StatsBomb
tutorials) is unsafe and was explicitly rejected for this milestone.

Without a consistent orientation, per-frame spatial features that depend on
absolute x-coordinate thresholds -- `attacking_control_final_third` (x >
66.0), `space_behind_defending_line` -- are meaningless when aggregated
across frames from different teams or different periods, since "which way
is forward" silently flips depending on whose turn it was to act and which
half it was. This was the reason `feature_extractor.py` was restricted to
period 1 only from Milestone 3 through Milestone 9: without solving
direction, at least period 1 vs period 2 comparability was avoided by
dropping period 2 entirely. That restriction blocked roughly half of every
match's usable data.

The one fact football's rules DO guarantee: a team's period 2 attacking
direction is the exact opposite of its period 1 direction (teams swap ends
at half-time). This is a free, reliable shortcut for period 2 -- but it must
anchor to a correctly INFERRED period 1 direction, not an assumed one.

## Decision
`production/src/pipeline/direction.py` infers each team's attacking
direction per period, in StatsBomb's native 0-120 coordinate space:

1. **Primary heuristic (period 1 only)**: mean x-coordinate of the team's
   `Shot` events in that period. A team shoots at the goal it's attacking,
   so `mean_x > 60` implies attacking toward increasing x (+1); `mean_x <
   60` implies attacking toward decreasing x (-1).
2. **Fallback** (team has zero shots in that period): mean x of the team's
   `Pass`/`Carry` events whose end-location reaches either final fifth (x >
   96 or x < 24 in raw units), using the same threshold logic. Both ends
   are checked -- restricting to only x > 96 would make the fallback
   structurally incapable of ever detecting a -1 direction, defeating its
   purpose.
3. **No signal at all**: prints a warning naming the team/period/match and
   returns `None`; that team's chains for that period are excluded rather
   than guessed at.
4. **Period 2 is derived, not independently inferred as primary**: once
   period 1's direction is established, period 2's direction is set
   directly to its guaranteed opposite (`-direction[team][1]`). As a
   cheap validation-only check, if period 2 also has enough shots/touches
   to independently estimate a direction, that estimate is computed and
   compared against the guaranteed value; a disagreement is printed as a
   warning (a genuine data anomaly worth seeing) but does NOT override the
   guaranteed value -- the guaranteed, rules-based derivation is trusted
   over a second independent measurement, precisely because the
   measurement is the thing being cross-checked, not the ground truth.

`production/src/pipeline/feature_extractor.py` applies the correction
per-event: for a frame whose acting team's resolved direction is -1, ONLY
the x-coordinate of `ball_pos` and every player in `player_pos` (both
attacking and defending side) is flipped via `x_norm = 100.0 - x_meters`
(100.0, matching this project's own 100x68 grid, not a real-world 105m
pitch length). y is never flipped -- no rule in football swaps touchlines,
only ends of the pitch. This is applied per-event using that event's own
acting team's direction, never uniformly across a whole period, since the
home and away teams attack opposite directions simultaneously within the
same period.

Under this now-consistent convention, the defending team's own goal is at
x=0 (they defend against the attacking team's advance toward x=100), which
required correcting `space_behind_defending_line`'s calculation: it
measures space between the defensive line and x=0 (the defending team's own
goal), not x=100 (which is the goal the ATTACKING team is trying to score
in, the opposite of what "space behind the defense" means tactically).

## Consequences
- Period 2 frames are no longer excluded from feature extraction, roughly
  doubling the number of chains that can contribute a trainable sample per
  match (see Milestone 10's training run for the actual observed effect).
- The shot/touch-based heuristic is an accepted v1 approximation. It can
  misfire for a team with very few attacking actions in a period (a team
  parked entirely on the defensive with almost no shots or final-fifth
  touches has a noisy or absent signal); such teams are excluded rather than
  guessed at, which trades some data loss for not silently mislabeling
  direction. The warning/fallback counts from a real training run are the
  concrete way to gauge how often this actually happens -- a heuristic that
  fires cleanly for nearly every team/period is trustworthy; one that falls
  back or excludes frequently should prompt tightening the heuristic before
  trusting downstream metric movements.
- The guaranteed-opposite shortcut assumes period 1's inference was correct.
  If period 1's heuristic misfires for a team (e.g. due to a small sample of
  unrepresentative shots), period 2's derived direction inherits that same
  error, silently, since the validation-only check only flags disagreement
  with an INDEPENDENT period-2 measurement -- it cannot detect an error in
  period 1 itself. This is an accepted limitation of anchoring to a single
  inference point per team per match.
- Velocity (`player_vel`) is not flipped by the x-correction, only
  `ball_pos`/`player_pos`. This is currently inert since `parse_360_frame`
  has no velocity source yet (Milestone 3's known v1 gap), but flipping
  velocity's x-component will become necessary once real velocity data is
  wired in, to keep it consistent with the flipped position space.

## Alternatives Considered
- **Assume period 1 is always correctly oriented, only correct period 2**:
  rejected -- explicitly identified as unsafe by this milestone's own
  premise; which end a team starts on is a coin toss, not a data guarantee.
- **Independently infer period 2's direction the same way as period 1,
  rather than deriving it from period 1**: rejected as the PRIMARY method --
  a team can easily have few or zero shots in one specific period (common
  for a team that's dominant in one half and shuts up shop in the other),
  making independent per-period inference less reliable than the
  rules-guaranteed derivation. Still computed as a validation-only
  cross-check, not silently discarded.
- **Use a fixed real-world pitch length (105m) for the flip formula**:
  rejected -- this project's internal grid is 100 units (ADR-002), and using
  a different constant for the flip than for every other spatial threshold
  in the pipeline would reintroduce exactly the kind of scale inconsistency
  ADR-002 was written to prevent.
