# ADR-015: Anchor-Based Recalibration — Qualified Adoption for Local Overlay Rendering Only

## Status
Accepted

## Context

ADR-013 ruled out frame-to-frame optical-flow composition (Milestone 37)
as a general camera-motion-compensation solution and named anchor-based
recalibration — fresh per-frame homography solving against known, fixed
pitch geometry — as the viable path forward, leaving the anchor-detection
mechanism itself unbuilt. ADR-014 then scoped a candidate mechanism, a
pretrained pitch-keypoint model (Roboflow's `football-field-detection-f07vi`,
32 keypoints), as usable ONLY as a strictly local, non-served research
prototype pending unresolved weight/training-data licensing questions.

A first evaluation of this model against the real Milestone 34B clip
produced an initial verdict of outright rejection: a naive comparison
found "static" (frame-0-anchored) homography reuse outperforming fresh
per-frame solving, and the model's keypoint confidence scores appeared to
be fundamentally decoupled from positional accuracy (errors of 20-50m on
a 100m pitch even at >0.99 confidence). Before accepting that verdict,
three follow-up checks were run specifically to rule out testing-condition
artifacts masquerading as a genuine negative result — this project's
established discipline (ADR-003, ADR-009) of not trusting a first negative
result without checking whether the test itself was sound.

**Check 1 — high-motion window.** The initial comparison's test window
(clip frames 0-57) turned out, on direct measurement, to be a near-static
camera segment. Re-running the identical fresh-vs-static comparison on a
genuinely high-motion window (frames 804-864, ~5x the clip-median
frame-to-frame displacement) reversed the result: **static homography
reuse degraded monotonically from 7.3m to 38.6m error as the window
progressed** — a real drift signature — while **fresh per-frame solving
stayed flat at 13.7-14.6m** across the same window. The original "static
wins" finding was an artifact of testing on a window with too little real
motion to expose static reuse's actual weakness.

**Check 2 — geometry table.** The 32-keypoint real-world correspondence
table (`PITCH_KEYPOINTS_METERS` in `pitch_keypoint_detector.py`) was
re-diffed, line by line, against `roboflow/sports`' `soccer.py` source, and
its rescale into this project's established 100x68m grid (ADR-002,
ADR-009) was re-verified against `calibration.py`'s and
`test_calibration.py`'s own `PITCH_CORNERS_METERS`. No mismatch was
found — this specific bug class (the project's recurring 100-vs-105
pitch-dimension confusion) is ruled out as a contributor to the observed
error.

**Check 3 — per-keypoint breakdown.** The initial verdict treated
confidence/accuracy decoupling as a property of the model as a whole. A
per-vertex breakdown across a 25-frame sample found this is false: **six
specific vertices — 19, 22, 23, 24, 25, 26 — are detected in ~100% of
frames at ~0.97-1.0 confidence with 19-55m median reprojection error**,
while the other regularly-visible vertices (15, 16, 17, 20, 21, 27, 28,
29, 30) stay in a 0.6-8m range. Every one of the six bad vertices is a
far-side/background pitch landmark under this camera's framing (the
small-y side of the pitch, e.g. vertex 25 at `(length, 0)` vs. its
near-side mirror vertex 30 at `(length, w)` — 53m error vs. 1.6m at the
same x) — consistent with a real, physically-explicable cause (extreme
foreshortening compresses the far touchline into very few image pixels,
degrading this model's positional regression there specifically) rather
than random noise. Excluding just these six vertices from the
correspondence table dropped the general-sample median residual from
4.7m to 1.5m, and the high-motion-window LOOCV median from ~14.1m to
**~6.2m**, with no drift growth reappearing.

Both of the original verdict's pillars — "static beats fresh" and "the
model is uniformly unreliable" — were therefore artifacts of the specific
test conditions (a low-motion window; an aggregate statistic that hid a
concentrated, identifiable, physically-explicable failure mode), not
properties of the underlying approach.

## Decision

**Adopt fresh-per-frame anchor-based homography solving, using this
pretrained keypoint model with vertices 19, 22, 23, 24, 25, 26 EXCLUDED
from the correspondence table, as a research-prototype-grade approach —
scoped explicitly to rough visual overlay / tactical-map rendering
purposes only**, subject to ADR-014's local-only, non-served constraint
(unchanged by this ADR).

**This is explicitly NOT adopted as sufficient positional accuracy for
`BiomechanicalPitchControl`, `DeepHit`, or any other quantitative
tactical/cheat-sheet analysis.** The StatsBomb track's physics/ML pipeline
was empirically validated against StatsBomb's own coordinate precision;
~6m median positional error under real motion does not meet that bar, and
this ADR must not be read or cited as establishing that it does. Any
future attempt to feed CV-derived positions into that pipeline requires
its own accuracy validation against that pipeline's actual requirements —
see Consequences.

## Consequences

- Milestone 38's overlay renderer, and any future local/offline demo
  video generated under ADR-014's scope, may use this qualified
  anchor-based recalibration to produce a real, non-fabricated tactical
  map overlay that tracks actual camera motion instead of relying on a
  single static calibration or the ruled-out frame-to-frame composition
  approach.
- **`production/src/cv/pitch_keypoint_detector.py`'s correspondence table
  must exclude vertices 19, 22, 23, 24, 25, 26** for any use under this
  decision. This exclusion list is a documented, load-bearing part of the
  adoption, not an incidental implementation detail.
- **UNRESOLVED, NOT assumed away — camera-angle generalization.** The
  six-vertex exclusion list was derived from the ONE real camera framing
  available throughout this project (the Milestone 34B clip). Whether the
  SAME six vertices are the unreliable ones under a different camera
  angle or elevation is UNTESTED — no second, differently-angled clip has
  been available to check. The foreshortening-based physical explanation
  makes it plausible that an analogous (but not necessarily identical)
  pattern holds generally — a different framing would plausibly make a
  different subset of vertices far-side/foreshortened — but this is
  reasoning from a physical mechanism, not a second measurement, and must
  not be treated as validated until a second clip is checked.
- **UNRESOLVED, NOT assumed away — accuracy sufficiency for the ML
  pipeline.** ~6m median error under real motion is a reasonable bar for a
  visual overlay a human watches, but no work has been done to determine
  what positional accuracy `BiomechanicalPitchControl` or `DeepHit` would
  actually tolerate before their own outputs become unreliable. Until that
  bar is established and tested against, this approach's accuracy must be
  treated as unvalidated for that purpose, not merely "probably not good
  enough" — the actual threshold is unknown, not just unmet.
- This does not change ADR-013's or ADR-014's standing conclusions:
  frame-to-frame composition remains ruled out as a standalone mechanism,
  and every ADR-014 licensing constraint (no live serving, no
  `production/src/serving/api.py` wiring) remains fully in force for this
  qualified use.

## Alternatives Considered

- **Uphold the original full-rejection verdict**: rejected — once the
  three testing-condition artifacts (an unrepresentative low-motion
  window; an aggregate statistic masking a concentrated, physically-
  explicable failure mode) are corrected for, the underlying evidence
  shows real, usable signal for an overlay-grade use case. Continuing to
  reject outright would be discarding a genuine result because the first
  test of it was flawed, not because the approach itself failed.
- **Unconditional adoption (drop the scoping to overlay-only, treat the
  vertex exclusion as fully general)**: rejected — the camera-angle
  generalization question and the ML-pipeline accuracy-sufficiency
  question are both real and currently unanswered, not merely
  theoretical; adopting without these caveats would overstate what has
  actually been validated, the same overclaiming this project's
  discipline has consistently avoided elsewhere (ADR-002's "approximation,
  not verified stadium geometry" caveat is the direct precedent).
- **Keep the six suspect vertices in the correspondence table but
  downweight rather than exclude them**: not attempted — a hard exclusion
  is simpler, cheaper, and was already sufficient to produce a meaningful
  accuracy improvement (4.7m/14.1m to 1.5m/6.2m); a downweighting scheme
  would add complexity with no demonstrated need to justify it yet.
