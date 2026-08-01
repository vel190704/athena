# ADR-016: Adaptive Outlier Rejection — Two Honest Attempts, Neither Adopted; ADR-015's Fixed List Remains the Recommended Approach

## Status
Accepted (as a record of what was tried and why the fixed list from
ADR-015 remains in effect — this ADR does NOT supersede ADR-015's
decision)

## Context

ADR-015 adopted fresh-per-frame anchor-based homography solving with six
specific vertices (19, 22, 23, 24, 25, 26) excluded from the
correspondence table, after finding them consistently mislocalized
despite high reported confidence on this project's one real camera
framing. ADR-015 itself named the load-bearing weakness of that fixed
list: it was derived from ONE camera angle, and there was no way to know
whether the same six vertices would be the unreliable ones under a
different framing.

This ADR set out to replace the fixed list with a RUNTIME, self-correcting
mechanism that would generalize to an unseen camera angle BY CONSTRUCTION
— discovering whichever keypoints are unreliable from the data itself,
rather than relying on a list that only helps if the next camera happens
to foreshorten the same landmarks. Two mechanisms were built and tested,
against the identical general 25-frame sample and high-motion window
(frames 804-864) used throughout this project's CV validation, against a
PRE-COMMITTED bar: clearly beat or match ADR-015's ~6.2m LOOCV median, or
stop and report the result honestly rather than trying further variants.

**Attempt 1 — per-frame iterative outlier rejection.** Reusing
`team_classifier.classify_teams`'s Milestone 28 masking-aware template
(fit, flag by a 2x-median-residual threshold, refit, re-evaluate every
original point), extended to trim a fixed fraction every round rather than
once (a single round was empirically insufficient — this project's real
frames run ~30-40% contaminated, and one round of 20% bootstrap or
2x-median thresholding left the fit "smoothly compromised" rather than
cleanly split). **Result: 35-39m LOOCV median — clearly worse than the
fixed list.** Root cause: with only 10-15 confidently-detected points per
frame and roughly a third of them genuinely bad, there isn't enough
within-frame sample to statistically separate "bad" from
"mediocre-but-real" residuals. The aggregate manual analysis behind
ADR-015's fixed list only found a clean signal by averaging the same six
vertices' behavior over 25 frames — something a single-frame mechanism
cannot do.

**Attempt 2 — multi-frame rolling reliability tracking.** Added an EWMA of
each vertex's reprojection residual, accumulated across frames (decay 0.9,
~10-frame effective memory; a vertex needs 5+ observations before its
rolling score is trusted, falling back to Attempt 1's per-frame-only
method otherwise), and excluded vertices whose accumulated score exceeded
2x the median rolling score among currently-trusted vertices. **Result:
32-38m LOOCV median — still clearly worse than the fixed list, and worse
in a more concerning way than Attempt 1**: on the general sample, the
MOST frequently excluded vertices were 30, 15, 16, and 29 — all four
independently verified as part of the reliable set in ADR-015's own
analysis — while several genuinely bad vertices (22, 23, 26) were almost
never flagged. Root cause: the tracker accumulates each vertex's residual
against that SAME frame's own per-frame-fit homography, which Attempt 1
already showed converges to a biased, still-contaminated result. Averaging
evidence against a noisy, self-referential reference does not cancel the
noise — it appears to compound the bias, penalizing vertices near the
fit's systematic distortion rather than the vertices that are actually
mislocalized.

Both attempts failed the pre-committed bar. Per the stopping rule agreed
before starting Attempt 2, no third variant was attempted.

## Decision

**ADR-015's fixed 6-vertex exclusion list remains the adopted,
recommended approach.** Both adaptive mechanisms are KEPT in
`production/src/cv/pitch_keypoint_detector.py`, fully documented, as an
honest record of what was tried and why it under-performed — not deleted,
per this project's established practice of not discarding a real,
negative result (ADR-013's treatment of Milestone 37's ruled-out
frame-to-frame composition is the direct precedent).

Concretely, the module now supports three ways to call
`solve_homography_from_keypoints`, and states plainly which is
recommended:
- `excluded_vertices=ADR015_KNOWN_UNRELIABLE_VERTICES` — ADR-015's fixed
  list. **The only option validated to ~6.2-7.0m median error. This is
  what real usage (e.g. Milestone 38's overlay renderer) should pass.**
- No arguments — Attempt 1's per-frame iterative rejection alone. Tested,
  35-39m median. Not recommended.
- `reliability_tracker=KeypointReliabilityTracker(...)` — Attempt 2's
  rolling mechanism (optionally combined with Attempt 1). Tested, 32-38m
  median, with the wrong-direction vertex-flagging problem described
  above. Not recommended.

`excluded_vertices` and the ADR-016 iterative-trimming loop are treated as
ALTERNATIVES, not composed by default: running the trimming loop on TOP OF
an already fixed-list-filtered candidate set was measured to make the
result WORSE (10.3m vs. 7.0m on the high-motion window), because trimming
a small, already-mostly-clean set still finds some point exceeding the 2x
threshold from ordinary sampling variance and removes it needlessly. When
`excluded_vertices` is non-empty, the function skips the trimming loop
entirely and returns the single direct fit — the same shape of result
ADR-015 was actually validated against.

## Theoretical Argument vs. Empirical Evidence — Read This Section Plainly

Neither adaptive mechanism was empirically validated to generalize to a
different camera angle — no second, differently-angled clip exists to
test that, exactly as ADR-015 already stated. This attempt was motivated
by making the GENERALIZATION ARGUMENT stronger (an adaptive mechanism
should, in principle, adapt to whatever a new camera framing does, where
a fixed list cannot). **That argument remains theoretically sound, but
this ADR did not confirm it — it found the opposite result on the ONLY
camera angle available**: both adaptive attempts performed WORSE than the
fixed list on the very clip the fixed list was derived from, meaning
there is now LESS reason, not more, to expect either adaptive mechanism
would do better on an unseen angle. A theoretically-more-general
mechanism that measurably underperforms on all available evidence is not
a safer bet than a narrower one that is measured to work — generality is
a property to want, not a substitute for the evidence a specific
implementation needs to earn it. This project's discipline (verify,
don't assume — ADR-013's own measured-limitation framing is the direct
precedent) applies here exactly as anywhere else: an elegant argument
that isn't yet backed by a working implementation does not out-rank a
narrower approach that is.

## Consequences

- `production/src/cv/pitch_keypoint_detector.py` gains
  `ADR015_KNOWN_UNRELIABLE_VERTICES`, an `excluded_vertices` parameter,
  the iterative-trimming mechanism, and `KeypointReliabilityTracker` — all
  four are documented in the module's own docstring, which states plainly
  that `excluded_vertices=ADR015_KNOWN_UNRELIABLE_VERTICES` is the
  recommended call and the other two paths are tested-and-inferior.
- All ADR-014 constraints (strictly local, non-served; must not be wired
  into `production/src/serving/api.py` or any live endpoint) remain in
  force, unchanged.
- All ADR-015 scope constraints remain in force, unchanged: this is
  validated for rough visual overlay / tactical-map rendering only, NOT
  for `BiomechanicalPitchControl`, `DeepHit`, or any quantitative
  tactical/cheat-sheet analysis.
- ADR-015's own two unresolved limitations are UNCHANGED by this ADR:
  whether the six-vertex list generalizes to a different camera angle is
  still untested (if anything, this ADR's negative result for the
  adaptive alternative makes that open question MORE load-bearing, not
  less, since there is currently no tested fallback if the fixed list
  turns out not to transfer), and what accuracy the ML pipeline actually
  requires remains undetermined.
- A future attempt at an adaptive mechanism, if ever revisited, should
  treat this ADR's two specific failure diagnoses as the starting point
  (small per-frame sample size defeats within-frame statistics; a
  self-referential rolling average compounds rather than cancels a biased
  reference) rather than re-discovering them from scratch.

## Alternatives Considered

- **A third adaptive variant** (e.g., a stricter residual multiple, or
  requiring convergence to persist across consecutive rounds before
  trusting it): explicitly NOT attempted, per the stopping rule agreed
  before Attempt 2 began. Continuing to tune parameters until a variant's
  numbers looked acceptable would risk fitting the mechanism to this one
  clip's idiosyncrasies rather than genuinely fixing the diagnosed
  structural problems (insufficient per-frame sample size; a
  self-referential, bias-compounding rolling reference) — the same
  overfitting-to-one-dataset risk this project's discipline has
  consistently guarded against elsewhere.
- **Delete the adaptive mechanisms as a dead end**: rejected — both
  attempts correctly tested a real, reasonable hypothesis and produced
  honest, informative negative results (including the specific,
  actionable root-cause diagnoses above); deleting them would discard
  that value and risk a future contributor re-attempting the same failed
  approach without the benefit of already knowing why it failed.
- **Quietly keep the fixed list without documenting the adaptive
  attempts**: rejected — this project's discipline treats a real,
  deliberately-tested negative result as worth recording explicitly
  (ADR-013's precedent), not as a silently abandoned thread.
