# ADR-013: Camera-Motion Compensation Approach — Anchor-Based Re-Calibration, Not Frame-to-Frame Composition

## Status
Accepted

## Context

Milestone 37 built frame-to-frame optical-flow-based camera-motion
estimation (`production/src/cv/camera_motion.py`): masked sparse optical
flow (`cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`, RANSAC-fit
via `cv2.findHomography`) produces a per-frame-pair motion homography,
composed over time by `CameraMotionTracker` into a cumulative transform
since a Milestone-27-style manual anchor, with an explicit `drift_budget`
and `is_stale` flag once accumulated drift exceeds a threshold. This was
validated on synthetic data: a smooth, KNOWN 0.03m/frame pinhole-camera
pan, over which the method correctly composed estimates and correctly
flagged staleness once real measured positional error crossed ~1m
(measured, not assumed — see `CV_PIPELINE_FINDINGS.md`'s Milestone 37
entry: drift_budget=0.55px/error=0.83m at frame 30, drift_budget=0.75px/
error=1.10m at frame 40).

A follow-up investigation then measured the method's ACTUAL input on real
footage, rather than continuing to reason from the synthetic case: running
the exact same, unmodified `estimate_camera_motion` frame-to-frame across
the full real Milestone 34B clip (970 frames, real detected player boxes
and confidences from `CVPipeline`'s own tracking model, not synthetic
input) found that real camera motion runs at a **median of 0.351px/frame**
against this clip — compared to the synthetic test's 0.03m/frame target
pan rate, this is (via a rough Milestone-27-derived meters-per-pixel scale
factor, explicitly NOT scene-accurate for this real clip's different
camera/resolution/FOV, but sufficient to establish order of magnitude): a
**median ratio of ~4.52x**, a **p90 ratio of ~14.70x**, and a **max ratio
of ~28.16x** the synthetic assumption. Critically, this is not a rare
tail: only **39.3%** of real frame-pairs fall within 2x of the synthetic
rate — a **majority (60.7%)** already exceed 2x, and nearly half (47.9%)
exceed 5x. Real broadcast camera motion, on this clip, is typically
several times faster than the rate this milestone's validation was built
around, not occasionally spiking above it.

## The Finding, Stated Precisely

**This is not a calibration problem, and raising or lowering
`drift_budget_threshold` cannot fix it.** The threshold only changes WHEN
`CameraMotionTracker` admits `is_stale = True` relative to a given
accumulation rate — it has no effect on the accumulation rate itself. At
the synthetic test's rate, ~0.7-0.75px of accumulated drift (reached after
~30-40 composed frames) corresponded to ~1m of real positional error. At
real footage's MEDIAN per-frame rate (0.351px, ~16x the synthetic test's
own measured ~0.0216px/frame accumulation rate), that same ~1m-error
milestone is reached in an estimated **2-3 real frames** — and faster
still during the many frame-pairs running at or above the p90 rate. A
2-3 frame reliable window defeats the entire purpose of frame-to-frame
composition, which exists specifically to BRIDGE many frames between rare,
expensive re-anchoring events. Recalibrating the threshold to real rates
would not buy back a usable window; it would simply make the system admit
staleness almost immediately — which is exactly what was observed when the
existing (synthetic-calibrated) threshold was run against this real clip
in the prior diagnostic. The distribution-shape finding (a majority of
frames already exceeding 2x, not a rare spike) rules out "design a
threshold robust to occasional whip-pans" as a fix: the typical case IS
the problem, not an outlier case a smarter threshold could route around.

## Decision

**Frame-to-frame optical-flow composition (Milestone 37) is RULED OUT as
a general-purpose camera-motion compensation solution for real broadcast
footage.** This is treated as a real, load-bearing architectural
conclusion, not a note to revisit informally — a future contributor should
not re-attempt "just tune the threshold better" without first addressing
the underlying accumulation-rate finding above.

**The viable path forward is ANCHOR-BASED re-calibration**: periodically
detect fixed, KNOWN real-world pitch geometry — lines, the center circle,
penalty-box corners, the same landmark categories Milestone 27's manual
calibration already uses — and re-solve the homography FRESH against that
geometry each time, rather than composing uncertain frame-to-frame deltas
on top of each other. Each re-solve is anchored directly to ground truth,
so error does not accumulate across re-solves the way it does across
composed frame-to-frame estimates; whatever error exists is bounded by a
single solve's own accuracy, not by how many frames have elapsed since the
last anchor.

**Two viable sub-approaches for anchor detection are named here as
options; NEITHER is committed to or built yet:**
1. **A trained keypoint-detection model** (e.g., in the style of
   SoccerNet's own calibration approach) — plausibly the highest-fidelity
   option, but gated behind the same NDA/data-access blocker that has
   stalled ground-truth CV validation since Milestone 25 (see
   `CV_PIPELINE_FINDINGS.md` Section 3). Not buildable now.
2. **Classical computer-vision line/circle detection** (e.g., Hough
   transforms over detected pitch-line pixels, corner/intersection
   geometry from the detected lines) — lower expected fidelity than a
   trained model, but requires no ground-truth training data and is
   buildable independent of the SoccerNet blocker. **This has NOT been
   built or tested** — it is a proposed direction only, named here so it
   is not lost, not a validated result of this ADR.

**Milestone 37's existing code is explicitly NOT wasted effort and should
NOT be deleted.** It correctly ruled out a real hypothesis using measured
evidence rather than assumption — that is genuine research value in this
project's own established discipline (the same discipline that produced
ADR-003's "built a fix, validated it, found the premise wrong" sequence).
Concretely reusable pieces:
- The frame-to-frame estimator and `CameraMotionTracker`'s composition/
  drift-tracking machinery remain useful as a CHEAP INTERPOLATION
  mechanism BETWEEN anchor re-solves once anchor-based detection exists
  (e.g., assume near-zero or lightly-extrapolated motion in the short gaps
  between anchors, rather than doing nothing at all between them).
- The flicker-aware masking techniques (confidence hysteresis absorbing a
  single-frame confidence dip; bounding-box padding for margin against a
  smaller-than-usual detected box) are valuable independently of camera-
  motion estimation specifically — they solve a real, separately measured
  problem (borderline-confidence detection flicker, ~23-26% of person
  detections near the 0.5 threshold) that will recur in any future masking
  task this pipeline needs, not just this one.

## Consequences

- **`adapter.py`'s `camera_motion_correction` parameter (Milestone 37,
  additive/optional) requires NO change.** It accepts a 3x3 homography
  matrix and composes it with the base `homography_matrix` regardless of
  how that matrix was derived — a genuine payoff of the decoupled-
  interface design choice made when it was added: the adapter never needed
  to know or care whether a correction came from composed frame-to-frame
  estimation or a fresh anchor-based solve, so this architectural
  conclusion requires zero changes to already-shipped, tested code.
- **Real-time speed rendering remains excluded from Milestone 38** pending
  an anchor-based approach being built and validated. This ADR does not
  itself unblock that work — it clarifies WHAT needs to be built
  (anchor-based re-calibration, via one of the two named sub-approaches)
  before it can be unblocked.
- **Track A's remaining non-speed-dependent items are NOT blocked by this
  finding** — in-match positional priors (see ADR-012) and cheat-sheet
  packaging do not depend on continuously-accurate camera-motion
  compensation the way real-time speed rendering does, and may proceed
  independently of this decision.
- Any future re-attempt at pure frame-to-frame composition as a
  STANDALONE solution (not as the interpolation layer between anchors
  described above) should treat this ADR's measured accumulation-rate
  finding as the reason not to, rather than rediscovering it from scratch.

## Alternatives Considered

- **Recalibrate `drift_budget_threshold` to match real-footage motion
  rates**: rejected — per the Finding above, the threshold does not
  affect the underlying accumulation rate; recalibrating it would only
  make `is_stale` fire sooner (or, if raised naively to avoid that, would
  silently trust positional error well beyond the ~1m reference point this
  project has treated as meaningful), not restore a usable multi-frame
  window.
- **Keep frame-to-frame composition as the sole/primary mechanism and
  accept frequent `is_stale` events**: rejected — at an estimated 2-3
  frame reliable window, staleness would fire almost continuously, which
  provides no practical advantage over re-anchoring every single frame
  directly and makes the composition machinery pure overhead.
- **Delete Milestone 37's code as a dead end**: rejected — see Decision
  above; the estimator/tracker are reusable as a future interpolation
  layer, and the flicker-aware masking components are independently
  valuable, so keeping the code (documented honestly as ruled out for its
  originally-intended standalone use) preserves real value.
- **Wait for SoccerNet/NDA access before building any anchor-detection
  approach at all**: rejected as the ONLY path forward — the classical
  Hough-line/circle-detection route is buildable now, independent of that
  blocker, mirroring the same reasoning ADR-012 already applied to
  in-match positional priors (don't gate genuinely separable work behind
  an unrelated, indefinitely-stalled access blocker when a lower-fidelity
  but buildable alternative exists).
