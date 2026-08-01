# ADR-017: Homography Accuracy Degrades With Distance From the Reliable-Keypoint Cluster — Full-Pitch Player Rendering Is Not Viable As Currently Built

## Status
Accepted

## Context

Milestone 41 built `tactical_map_renderer.py`, rendering real player
positions (from `pipeline.CVPipeline`'s tracking output) onto a top-down
pitch diagram via ADR-015/016's qualified, fixed-6-vertex-exclusion
homography solve. Its own real-clip test passed every PROCESS
requirement (staleness handling, caption visibility, frame-complete
output) at a 100% valid-homography rate, but the qualitative visual check
required by that milestone's own test plan found a real problem: rendered
player dots, inspected directly against the same frame's pixel-space
overlay, did not match the real, visibly spread-out team shape — players
clearly spread across most of the broadcast frame's width collapsed onto
a narrow band on one side of the pitch diagram.

This ADR investigates that finding directly, rather than treating it as a
rendering bug to patch. ADR-015/016's own validated ~6-7m accuracy figure
was measured via leave-one-out cross-validation on DETECTED KEYPOINTS —
i.e., accuracy INTERPOLATING within the region the correspondence points
themselves cover. Player positions span the whole visible pitch,
including regions the detected keypoints never touch. The hypothesis
tested here: does homography accuracy degrade with distance from the
region the fit's own supporting keypoints actually occupy, and if so,
where do real players actually fall relative to that region?

**Method.** Using the same real clip (25-frame whole-clip systematic
sample) and the same production code path
(`solve_homography_from_keypoints(..., excluded_vertices=
ADR015_KNOWN_UNRELIABLE_VERTICES)`), each frame's reliable
(confident, non-excluded) keypoints were put through a per-keypoint
leave-one-out test: hold one out, fit on the rest via the REAL
production function, measure the held-out point's pitch-space error
against true ground truth, and record its PIXEL-space distance from the
fitting set's own centroid — pixel distance chosen specifically because
it requires no homography output to define itself (unlike a meter-space
distance, which would need the very fit being evaluated) and is the one
quantity directly comparable to real player detections, which have no
ground-truth meter position to measure against. Real player feet
positions (396 real YOLO detections, same frames) were independently
measured for their own pixel distance from the same per-frame centroid.

## The Finding, Stated Precisely

**1. Inside a real, usable trust radius, accuracy is tight and stable.**
Within ~150px of the reliable-keypoint cluster's centroid, held-out error
was **median 3.35m, mean 3.24m, max 3.86m** (n=24) — actually tighter than
ADR-015/016's aggregate ~6-7m figure, and with a genuinely bounded worst
case.

**2. Beyond that radius, error does not merely grow — it becomes
unpredictable, confirmed statistically, not just visually.** Binned
results (n=215 held-out points total) showed error climbing through
150-300px (median 6.66m, max 25.56m), 300-450px (mean 9.66m, max
195.33m), and 450-600px (mean 26.20m, **max 933.46m**). Per-bucket medians
are noisy at these sample sizes, but the underlying relationship is real
and strong: Spearman correlation between distance and error = **0.582
(p<0.0001)**; Pearson correlation between distance and log-error =
**0.521 (p<0.0001)**. A handful of catastrophic outliers (933m, 195m)
suppress the raw-scale linear correlation (Pearson r=0.10, not
significant) — the rank-based and log-scale statistics are what actually
characterize this relationship, and both agree: extrapolating beyond the
keypoint cluster is not just less accurate, it is unreliably so, with
real risk of order-of-magnitude failures, not a smoothly degrading
number.

**3. Most real players stand OUTSIDE the trust radius, not inside it.**
Real player feet positions (396 detections, same frames) had **median
distance 233px, p90=483px** from the same per-frame cluster centroid.
Only **27.5%** of detected players fell within the one bucket with a
tightly-bounded error (0-150px). The remaining **72.5%** fell in buckets
where mean error is 2-8x worse and worst-case error reaches into the
tens-to-hundreds of meters.

## Decision

**Full-pitch player-position rendering, as currently built, is NOT
viable at uniform trust across the pitch.** This is a genuine limitation
of the underlying calibration approach on this camera framing, not a
rendering-layer defect `tactical_map_renderer.py` can be tuned to fix.

A trust-radius-based mitigation (render normally inside ~150px of the
reliable cluster; gray out or omit positions beyond it) is a real,
available, HONEST option — but it is a genuine scope reduction, not a
patch that restores full-pitch rendering: given real players are
predominantly OUTSIDE that radius (72.5% of detections), applying it
strictly would hide most of the players on a typical frame, most of the
time. **This ADR does not adopt that mitigation as sufficient** — it is
recorded as the one option investigated and found honestly available, not
as a resolution. No fix is implemented as part of this ADR.

`tactical_map_renderer.py`'s existing behavior (render whatever the
homography solve produces, without distance-based filtering) is
therefore now KNOWN to render some fraction of players at unreliable,
occasionally wildly wrong positions on every frame with a nominally
"valid" homography — Milestone 41's binary `homography_valid` flag
correctly reports whether A solve succeeded, but does not, and currently
cannot, express PER-PLAYER confidence.

## Consequences

- **Milestone 41's tactical map, as it stands, must not be represented as
  reliably showing full-team shape.** Its own accuracy caption ("~6-7m
  accuracy") is now understood to describe best-case, near-cluster
  accuracy, not a uniform bound across the rendered pitch — a real gap
  between what the caption states and what a majority of rendered dots
  actually achieve. Future work on this caption/UI is warranted but out
  of scope for this ADR (a documentation finding, not a rendering
  change).
- Any future attempt to make full-pitch rendering viable has two
  realistic directions, NEITHER attempted here: (a) a per-player
  confidence/trust-radius indicator in the rendered output (the honest
  version of the mitigation above — show a majority of players
  visibly de-emphasized rather than pretending uniform accuracy), or (b)
  improving the underlying keypoint coverage/spread itself (more
  keypoints, a wider-spread reliable set, or a differently-designed
  model less prone to the ADR-015 exclusion pattern in the first place)
  so the reliable cluster actually covers more of where players stand.
  Neither is scoped or committed to by this ADR.
- ADR-015's own stated limitation ("the six-vertex exclusion list was
  derived from one camera framing, untested on a different angle")
  remains fully in force and is now joined by this ADR's distinct
  finding: even ON the one framing this project has data for, and even
  using the validated exclusion list correctly, extrapolation beyond the
  keypoints' own footprint is unreliable. A second camera angle would not
  resolve this ADR's finding even if it resolved ADR-015's.
- This does not change ADR-014's licensing scope, ADR-015's fixed-list
  decision, or ADR-016's rejection of the two adaptive alternatives — all
  three remain accurate, unchanged findings about what they each actually
  measured.

## Alternatives Considered

- **Ship the trust-radius mitigation now, as a fix bundled with this
  ADR**: rejected for this ADR specifically — the finding here is that
  the mitigation, applied honestly, subtracts most of the rendered pitch
  most of the time; shipping it silently as "the fix" without that
  caveat front and center would repeat exactly the kind of overclaim this
  project's discipline exists to catch. Recording the finding first, and
  deciding what (if anything) to build in response as a separate,
  deliberate step, was preferred.
- **Treat the visual-plausibility finding as a one-off rendering
  artifact and leave `tactical_map_renderer.py` unexamined**: rejected —
  the statistically significant distance/error relationship (Spearman
  0.582, p<0.0001) and the measured player-distance distribution (73%
  outside the tight-error radius) show this is a structural property of
  the current calibration approach on real player positions, not a
  single bad frame or a coincidence.
- **Re-attempt ADR-016's adaptive outlier-rejection mechanisms to solve
  this instead**: not attempted — ADR-016 already found both attempted
  adaptive mechanisms underperformed the fixed list on the narrower
  keypoint-accuracy question; there is no evidence either would behave
  better on this distinct extrapolation question, and re-attempting them
  here without a new hypothesis for why they'd help would repeat the
  same overfitting-to-conclusion risk ADR-016's own stopping rule was
  built to avoid.
