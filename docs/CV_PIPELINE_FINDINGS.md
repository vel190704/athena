# Project Athena: Computer Vision Pipeline Findings

**Status as of Milestone 41** (this document's original synthesis covered Milestones 25-33; it has since been extended in place through Milestone 34B's first-real-footage update, Milestone 37's camera-motion findings, Milestones 39/41's pitch-keypoint/tactical-map work and ADR-014 through ADR-017 — see each section's own Milestone label for exactly what was added when). Module 4 (Computer Vision Pipeline) is fully
implemented end-to-end: broadcast video in, the exact tensor contract the
StatsBomb-trained models already consume out. This document mirrors
`RESEARCH_FINDINGS.md` (Milestone 24) for the CV track specifically.

Every number in this document was re-verified by actually re-running the
relevant test file (or, where a figure depended on an ephemeral manual
demonstration never captured as a persisted assertion, by regenerating that
exact demonstration fresh) immediately before this document was written —
not transcribed from memory of prior milestone summaries. Section 7 lists
every figure that could not be reconfirmed this way, and why.

**Update (Milestone 34B: First Real-Footage Validation) — first
real-footage run.** In a subsequent debugging session (after this
document was first drafted), the pipeline was run end-to-end against a
real video file for the first time:
`data/raw/test_match.mp4`, a **private local broadcast-style clip**
(1284x728, 28fps, 970 frames / ~34.6s) — explicitly NOT SoccerNet, and
carrying no ground-truth annotations. `test_cv_tracker.py` and
`test_cv_pipeline.py`'s own skip messages call this kind of clip "an
acceptable stopgap for this milestone only," not a substitute for the
NDA-gated dataset. The Executive Summary and the Milestone 26/29/32
sections below are updated in place with these newly-verified figures.
**This does not lift the real-data blocker described in Section 3** — a
single unannotated private clip cannot answer any question that requires
ground truth (detection precision/recall, a true ID-switch *rate*, real
calibration accuracy), and every such gap remains exactly as open as
before. What it DOES provide, for the first time, is real (not synthetic
or mocked) tracking and end-to-end throughput behavior — see below.

---

## 1. Executive Summary

Module 4 exists to eventually let Project Athena run on live broadcast
video instead of only StatsBomb's pre-annotated event/360 data: detect
players and the ball in a video frame, track them across frames, tell the
two teams apart, convert pixel coordinates to the pitch's real 100x68m
space (ADR-002), and package the result into the identical tensor bundle
`feature_extractor.extract_features` has consumed since Milestone 3 — so
every downstream model (the MLP, the GNN, the Deep Ensemble, the live
WebSocket/REST API) can consume CV-derived state with zero modification.

**Current state, stated plainly:** all nine CV components (Milestones
25-33) are built, internally consistent, and validated against synthetic,
adversarial, or mocked test cases — every one of those tests passes. As of
the update above (Milestone 34B), the full pipeline has also now **run
successfully end-to-end on one real broadcast-style video clip** — real
tracking, real ball detection (after a real bug fix — see Milestone 29
below), and a first real throughput measurement (Milestone 32 below). **This is one
unannotated private clip, not the SoccerNet validation this track has
always needed** — no detection precision/recall, no true ID-switch rate,
no calibration-accuracy figure can be computed without ground truth, and
none of those gaps are closed by this run. Two components (Milestone 25's
baseline detector, Milestone 29's ball detector) were separately validated
against real *photographs* earlier — a meaningfully different and much
easier condition than real match video (no motion blur, no compression
artifacts, no camera pan/zoom, no crowd/broadcast-graphics clutter). This
document keeps three tiers distinct throughout: "validated on a real
photo," "run once on a real, unannotated private clip," and "validated
against real broadcast footage with ground truth" are never treated as
equivalent claims here. **A tenth component, Milestone 37 (camera-motion
compensation), was added afterward** — see its own entry in Section 2 —
providing measured drift detection and flicker-aware masking, explicitly
NOT drift-free compensation; the "nine components" count above describes
this document's original (Milestone 34) scope and is not retroactively
rewritten.

---

## 2. Component Validation Status

### Milestone 25 — Baseline Player Detection
**Implementation:** pretrained YOLOv8m filtered to COCO class 0
("person"); custom IoU-based greedy one-to-one matching + micro-averaged
precision/recall/F1 (`production/src/cv/metrics.py`), deliberately built
to avoid double-counting overlapping candidate matches.
**Validation:** the matching/scoring math is unit-tested with hand-computed
synthetic scenarios (`test_cv_detector.py`, 3 tests, all reconfirmed
passing) — including a specific test proving the greedy dedup rejects a
lower-IoU duplicate match rather than double-counting it. The detector
itself was smoke-tested on a real photograph (ultralytics' bundled
`bus.jpg`): **4 person detections**, confidences **0.928, 0.922, 0.901,
0.752** (reconfirmed by direct re-run). The intended real-data validation
target — formal P/R/F1 against SoccerNet's tracking dataset ground truth —
has **never run**; `test_soccernet_baseline_detection_accuracy` still
skips, blocked on NDA access (see Section 3).
**Limitations/Unknowns:** real precision on broadcast footage is
completely unmeasured; Milestone 25's own scope note already predicted
precision will be poor without pitch-region restriction (crowd/staff
false positives), but this remains a prediction, not a measurement.

### Milestone 26 — Player Tracking (ByteTrack) & Pixel Velocity
**Implementation:** per-frame YOLO+ByteTrack (`model.track(..., persist=True)`)
tracking; real fps read from the video file (`cv2.CAP_PROP_FPS`, never
assumed); per-track pixel displacement divided by the ACTUAL frame-rate-
derived `dt`; an explicit, stated (not hidden) sanity ceiling
(`LIKELY_ID_SWITCH_PIXEL_VELOCITY_CEILING = 800.0` px/s) flags implausible
frame-to-frame jumps as `likely_id_switch: True` rather than silently
trusting them.
**Validation:** the camera-motion-confound property was demonstrated via a
synthetic panning-crop video over a static real photograph (ultralytics'
`bus.jpg`) and reconfirmed by regenerating it fresh for this document: at
a real, file-read fps of **24.0**, with 2 track IDs persisting across all
20 frames and `likely_id_switch` firing **0** times, the two tracked
people — who did not move at all in the underlying static image — were
reported moving at **[-171.4, -1.2] px/s** and **[-166.5, -8.6] px/s** on
the first post-pan frame (settling into a roughly -170 to -230 px/s range
as the pan continued). This is not a bug; it is the camera-motion-velocity
confound working exactly as documented (see Section 4).
**`test_bytetrack_player_tracking_on_real_footage` now runs and passes**
(previously skipped, no video available) against the private clip
described in the Update note above (Milestone 34B). Direct re-run over
the full 970-frame
clip: **152 unique track_ids observed, 146 of them persisting across more
than one frame**, and **15,090 individual nonzero-`vel_pixels_per_sec`
observations** among persistent tracks — proving real displacement is
genuinely being measured, not defaulted to zero, on real footage.
**Limitations/Unknowns:** 152 unique IDs against a real roster of ~22-25
people on the pitch is a real, honestly-reported signal of heavy ID
churn/fragmentation — but WITHOUT ground-truth identity labels, this raw
count cannot be turned into a true "ID-switch rate": some of that churn is
legitimate (people entering/leaving frame, camera cuts the shot classifier
correctly treats as separate shots), not necessarily erroneous re-assigned
identity. Future Validation Roadmap item 2 (a real switch *rate*) remains
open. The 800 px/s sanity ceiling is still an unvalidated guess and was
not directly exercised by this analysis.

### Milestone 27 — Pitch Calibration & Homography
**Implementation:** `cv2.findHomography` (4 known corner correspondences,
method=0/DLT) + `cv2.perspectiveTransform` (internal `(N,1,2)` reshape
handled for callers).
**Validation:** tested against a synthetic homography derived from an
actual pinhole-camera projection (position 55m outside the touchline, 30m
elevated, real intrinsics/extrinsics) — reconfirmed by direct re-run:
**near-touchline pixel span 810.4px vs. far-touchline 396.4px**, a
**~2.04x** foreshortening ratio (810.4/396.4), asserted to exceed a 1.5x
minimum specifically so this test can't silently degrade into a flat
affine-scale case. Calibrating from ONLY the 4 pitch corners, all 3
held-out landmarks (center spot, two penalty-line spots — never used for
calibration) recovered to their true coordinates with **error 0.0000m at
printed precision** (tolerance used: 0.05m). The in-bounds/out-of-bounds
distinction was also reconfirmed: an interior point recovers in-bounds; a
point 10m beyond a touchline recovers to `y = -10.0000` (correctly
out-of-bounds, not clipped).
**Limitations/Unknowns:** zero validation against a real lens (real lenses
have radial/tangential distortion this pure-homography model doesn't
model at all); no automatic keypoint detection (correspondences are
hand-specified); explicitly does not handle continuous re-calibration
across a panning/zooming feed — a single fixed homography only.

### Milestone 28 — Team Classification & Role Separation
**Implementation:** circular-mean hue extraction (`extract_jersey_color`)
+ masking-aware iterative-refit KMeans clustering (`classify_teams`):
fit → flag top-20%-by-distance candidates → remove → refit → re-evaluate
ALL points against the refined centroids.
**Validation** (all figures reconfirmed by direct re-run,
`test_team_classifier.py`, 5 tests): on a synthetic 5-blue/5-red/1-yellow/
1-black roster, classification correctly grouped all blues together, all
reds together (a different team than blue), and flagged both the yellow
GK and black referee as `outlier`. The refit was proven non-trivial: the
naive single-pass centroid for the "red" cluster was
**[0.929, 0.124, 0.857, 0.857]** (visibly pulled toward the outliers it
had absorbed); after removing the correctly-identified top-2
largest-distance candidates (track_ids **11, 12** — exactly the GK and
referee) and refitting, the same cluster's centroid became
**[1.0, ~0.0, 1.0, 1.0]** — the true pure-red value — a **0.2474** L2
centroid shift. The circular-hue fix was proven with a single crop built
half hue≈2/half hue≈177 pixels: circular mean **179.50** (correctly red)
vs. an independently-computed naive linear mean of **89.50** (incorrectly
green). At the clustering-feature level, hue=2 and hue=177 sit
**0.0698** and **0.1047** from pure red in `(cos, sin)` feature space,
versus a naive raw-hue-scalar distance between them of **175.0**.
**Limitations/Unknowns:** validated on solid, fully-saturated synthetic
swatches only — real jersey patterns (stripes, sponsor logos, printed
numbers), lighting/shadow variation across a pitch, and motion blur are
completely untested; the "2x median inlier distance" outlier threshold is
documented as degenerating (becoming hypersensitive) under near-zero-
variance data, a property directly observed while building this
milestone's own tests.

### Milestone 29 — Ball Detection
**Implementation:** YOLO filtered to COCO class 32 ("sports ball"),
`confidence_threshold=0.3` (deliberately lower than the person detector's
0.5); a shape-aware fallback (`detect_ball_fallback`) filtering candidate
bright contours by BOTH circularity (`4*pi*area/perimeter^2 > 0.7`) and a
pixel-area range, returning the candidate closest to the last known
position, not the largest.
**Validation:** the YOLO path was tested against a REAL photograph — not a
synthetic circle, since a `cv2.circle()`-drawn disc lacks the
lighting/texture signature a photo-trained detector learned to recognize
— specifically "Soccer Ball about to be kicked.JPG" (Wikimedia Commons, CC
BY-SA 4.0, downsampled to 1280x720; full attribution in
`production/tests/fixtures/ATTRIBUTION.md`). Reconfirmed by direct re-run:
**confidence 0.9454**. The fallback's shape filter was proven with a
deliberately adversarial synthetic case — a small circle (~314px² area)
alongside a larger (1000px² area) but non-circular rectangle distractor —
reconfirmed to correctly return the circle
(`{'ball_pos_pixels': [150.5, 150.5], 'circularity': 0.832}`), not the
larger rectangle.
**Limitations/Unknowns:** the single real-photo validation is a static,
well-lit, unobstructed shot of a stationary-looking ball — nothing here
tests motion blur, partial occlusion by a player's body, or a ball
silhouetted against a crowd/advertising-board background, all realistic
broadcast conditions.

**A real bug was found and fixed via the private real-footage clip
described in the Update note above (Milestone 34B).** `detect_ball` never passed `imgsz`
to `model.predict()`, so Ultralytics defaulted to 640 — downscaling this
clip's 1284x728 frames enough to erase the ball, whose true footprint at
this camera distance is only ~5x5px (inferred from a median real
person-detection bbox height of ~33px in this clip). Direct A/B testing on
this footage: **zero "sports ball" class candidates at any confidence down
to 0.01** at the Ultralytics default `imgsz=640`, across every sampled
frame, versus a real, moving, plausible-confidence (~0.3-0.7) detection
recovered at `imgsz=1920` on the majority of frames sampled across the
whole clip (position genuinely changes frame-to-frame, ruling out a static
broadcast-graphic false positive). Fixed by adding an explicit
`imgsz: int = DEFAULT_BALL_DETECTION_IMGSZ` (1920) parameter to
`detect_ball()`. **This value is evidence-based on exactly ONE clip at one
resolution, not a generally-validated default** — a fixed pixel value does
not scale correctly to a different source resolution (upscaling an
already-larger frame to 1920 would shrink it instead), a limitation stated
directly in the code, not silently assumed away.

### Milestone 30 — CV-to-Physics Adapter Layer
**Implementation:** `convert_frame_to_tensors` — converts current AND
previous pixel positions to meters SEPARATELY before differencing for
velocity (never transforms a displacement vector through the homography
directly); possession-aware `is_teammate` (nearest player to the ball, by
transformed distance, is the possessing team — an explicit heuristic, not
a hardcoded team); outlier/unmapped-track/out-of-bounds filtering with an
explicit 0.5m touchline tolerance.
**Validation** (reconfirmed by direct re-run, `test_adapter.py`, 9 tests,
using the SAME Milestone 27 pinhole-camera homography, not a toy
uniform-scale matrix): for a player moving 2m along X near the far
touchline (a region of real, significant projective distortion), the
correct point-then-difference method gives **[50, 0] m/s**; an explicitly
computed naive method (transforming the raw pixel displacement vector
through the homography as if it were a point) gives **[3278.3, -2752.0]
m/s** — wrong in magnitude AND fabricating a nonzero Y-component where the
true motion was purely along X. Difference between methods: **4242.13
m/s**. The adapter's actual output matches the correct method exactly. The
possession heuristic was proven to flip (not stay hardcoded) when the
ball's position moved from near one team to the other:
`is_teammate = [False, False, True]` then `[True, True, False]` for the
identical 3-player roster.
**Limitations/Unknowns:** none beyond what Milestones 27-29 already carry
forward (the adapter's own logic is fully unit-tested); its output has
never been fed real CV-derived (as opposed to synthetic ground-truth-
derived) pixel data.

### Milestone 31 — Shot/Camera-Cut Classification
**Implementation:** a 2-feature heuristic (`green_ratio > 0.25 AND
edge_density > 0.05`, both UNVALIDATED guesses) deciding whether a frame
is a usable tactical wide view before running the rest of the pipeline on
it.
**Validation** (reconfirmed by direct re-run, `test_shot_classifier.py`):
the easy cases all classify correctly — a dense-grid synthetic tactical
view (`green_ratio=0.8913, edge_density=0.0630 -> True`), a close-up
off-pitch scene (`green_ratio=0.1194, edge_density=0.0010 -> False`), and
a structured (non-random-noise) crowd pattern (`green_ratio=0.0000,
edge_density=0.1569 -> False`). **The deliberately-constructed hard
adversarial case — a mostly-green close-up with enough incidental
textured foreground to cross the edge-density threshold —
`green_ratio=0.6661, edge_density=0.0600`, was INCORRECTLY classified
`True`.** This is reported as a genuine, documented failure of this
2-feature approach at these thresholds, not a bug to hide (see Section 4).
Timing was reconfirmed at a median of **8.4ms** on a 1280x720 image in
this run (5-10ms observed across separate runs), under the **20ms**
practical bound actually used (the milestone's originally-suggested 5ms
bound was empirically too optimistic for this hardware and was revised
upward, with the real number always reported regardless).
**Limitations/Unknowns:** cannot distinguish a replay of a valid tactical
shot from the live moment it replays; cannot detect a zoom change within
an otherwise-valid view (directly connected to Milestone 27's own "no
continuous re-calibration" gap — the same underlying missing capability).
The adversarial failure rate on REAL footage (as opposed to one
constructed example) is completely unknown.

### Milestone 32 — Live Stream Ingestion & Pipeline Orchestration
**Implementation:** `CVPipeline.process_video` chains shot classification
→ tracking → ball detection → team-classification refresh → calibration →
adapter into one frame-by-frame generator; tracks each `track_id`'s last
OBSERVED FRAME INDEX (not just position) so velocity uses the TRUE elapsed
gap across skipped non-tactical frames, with an explicit staleness cutoff
(`stale_gap_frames_threshold`, default 5) falling back to `[0,0]` velocity
beyond it.
**Validation** (reconfirmed by direct re-run, `test_cv_pipeline.py`, all 7
tests, including the real-footage throughput test — see the Update note
below): at the pure-function level, a frame-10-to-frame-16 gap correctly
yields `dt = 6/25 = 0.24s`.
At the full end-to-end orchestrator level (cv2/YOLO mocked, since no real
video exists in this environment, but the REAL `process_video` code path,
not a shortcut), the same scenario produced velocity **[125.0, 0.0] px/s**
— exactly `30px / (6/25)s` — versus what a naive `1/fps` assumption would
have given, **[750.0, 0.0] px/s**, a 6x error. `skipped_non_tactical`
correctly accumulated to **15** (10 frames before the track's first
observation + 5 skipped between observations). A separate scenario
confirmed the staleness cutoff: an 11-frame gap against a threshold of 5
correctly produced `stale_velocity_fallback_count = 2` (both tracks in
that scenario shared the same gap) and `[0,0]` velocity for both.
**`test_pipeline_on_real_video_and_throughput` now runs and passes**
(previously skipped, no real footage) against the private clip described
in the Update note above (Milestone 34B) — the CV track's first-ever real
end-to-end throughput measurement. Over the first 30 raw frames of
`data/raw/test_match.mp4` (all 30 yielded — the ball-detector fix above
was required for this; before it, 0 of 30 yielded because every frame
lacked a detected ball): **median 116.73ms/yielded-frame, p95 127.23ms,
effective throughput 8.57 fps**, against this clip's real 28fps source
rate — i.e. **not real-time on this hardware**, printed by the test itself
as "well below real-time; significant optimization needed before live
use."
**Limitations/Unknowns:** this is ONE clip, 30 frames, on one machine's
hardware, with no GPU-vs-CPU breakdown recorded here — not a general
throughput characterization. The ~8.57 fps effective rate is well below
this clip's own 28fps, meaning a rolling buffer or frame-skipping/dropping
strategy would be required before any real-time deployment attempt; that
strategy itself has not been designed or implemented. Future Validation
Roadmap item 6 (broader, multi-clip, multi-hardware throughput
characterization) remains open.

### Milestone 33 — Live CV API Integration
**Implementation:** `source=cv` WebSocket path in `production/src/serving/api.py`:
path-safety validation (resolved path must lie inside `data/raw/`), a
FRESH `CVPipeline` per connection, `asyncio.to_thread(next, gen)` to avoid
blocking the event loop on each blocking generator step, explicit
real-time pacing (sleep to align when ahead of pace; honest
`real_time_lag_sec` reporting when behind, never forced).
**Validation** (reconfirmed by direct re-run, `production/tests/test_api.py`,
6 new CV-specific tests, all passing): with a mocked `CVPipeline` whose
`process_video` does a REAL, genuinely-blocking 3-second `time.sleep`
inside the generator, a concurrent `/simulate` REST call completed in
**0.06-0.10s** across separate runs (well under the 3s block), proving
`asyncio.to_thread` genuinely offloads the blocking call rather than
freezing the event loop for every connection. Per-connection isolation was
confirmed by opening two CV-source connections with different video paths
and a fake pipeline that encodes its `video_path` into the returned ball
position: exactly 2 pipeline instances were constructed (never reused),
and each connection's received data matched only its own path's marker,
never the other's (the exact marker values are not deterministic across
runs, since they derive from Python's randomized string hashing, but the
inequality between the two connections' values — and each matching its own
freshly-computed expected value — holds every run and is what the test
actually asserts). **A real integration bug was found and fixed via manual
testing against the actual running uvicorn server** (not the automated
mocked-transport test suite, which did not happen to trigger it): a
WebSocket close-frame `reason` string embedding a full resolved file path
exceeded the protocol's ~123-byte control-frame limit, producing
`websockets.exceptions.ProtocolError: control frame too long` instead of a
clean close. Fixed with an explicit truncation helper; reconfirmed both
against the real server and via the automated suite (`test_cv_source_unreadable_file_closes_cleanly_not_a_crash`,
close code 1011, now passing).
**Limitations/Unknowns:** the calibration caveat is explicit and load-bearing
— this endpoint does not yet accept a real homography, so CV-sourced
`threat_15s` values are computed from PIXEL-space positions fed into
physics math that assumes ADR-002's 100x68 METER space, and are not
physically meaningful yet. This milestone proves the async/isolation/pacing
wiring, not calibrated real-world threat numbers.

### Milestone 37 — Camera-Motion Compensation
**Implementation:** `camera_motion.py`'s `estimate_camera_motion` (sparse
optical flow: `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`,
fit via `cv2.findHomography(..., method=cv2.RANSAC)`) and
`CameraMotionTracker` (composes frame-to-frame estimates into a cumulative
transform since a Milestone-27-style manual anchor, with an explicit,
never-self-clearing `is_stale` flag). Player regions are excluded from the
background feature set via base confidence masking (≥0.5) PLUS a
confidence-hysteresis rule (a 0.35–0.5 dip is still masked if a matching
box was confidently seen two frames prior) and 17.5% bounding-box padding
— both added specifically in response to this milestone's own diagnostic
finding below, not a hypothetical concern.
**Validation:** a full synthetic pinhole-camera pan sequence (Milestone
27's approach, extended to a real rendered frame sequence so the actual
optical-flow code path is exercised, not bypassed) MEASURED the
drift-vs-frame-count curve directly rather than assuming one: composed
positional error at the pitch center reached **0.83m at frame 30** and
**1.10m at frame 40** (0.03m/frame synthetic pan rate) — monotonically
increasing, as expected for composed drift. A masking adversarial test
(6 independently-flickering synthetic players, confidence cycling
0.6↔0.4, background contrast deliberately weakened so unmasked player
edges dominate corner selection — otherwise RANSAC alone absorbs a small
contaminant regardless of masking quality) measured flicker-aware masking
(hysteresis + padding) at **0.143px** mean per-frame estimate error vs.
**0.224px** for naive (current-frame-confidence-only, no padding) masking
— a real **1.57x** improvement, isolated from unrelated cumulative-drift
noise by comparing PER-FRAME estimates against ground truth rather than
final composed position (composing 60 frames of pan alone already
accumulates >1m of unrelated error that would otherwise swamp the
comparison). `is_stale` was confirmed to fire at the expected point
(frame 38) once the drift-budget threshold — itself set FROM this
measured curve (0.55px/0.83m at frame 30, 0.75px/1.10m at frame 40), not
assumed in advance — was exceeded.
**A real-orchestrator diagnostic** (driving `CVPipeline`'s own
`_tracking_model`/`is_tactical_view` gate directly over 150 real frames of
the Milestone 34B clip — not `tracker.py`'s standalone harness) confirmed
the near-threshold detection-flicker finding replicates in practice:
**26.0% of real person detections fell in the [0.4, 0.6) confidence
band** (802/3,087), closely matching the earlier 23.1% full-clip
measurement. `estimate_camera_motion` never returned `None` across this
real segment. **A load-bearing caveat found by this same diagnostic,
not assumed:** the drift-budget threshold calibrated against the
synthetic test's slow pan (0.03m/frame) is NOT validated against real
footage's actual camera-motion rate — `is_stale` fired almost immediately
on the real clip under that threshold, confirming real camera motion is
considerably faster than the deliberately slow synthetic pan this
milestone's threshold was calibrated against. The threshold is real and
measured, but only for the synthetic scenario; it is not yet a validated
real-deployment gate.
**Limitations/Unknowns:** provides camera-motion ESTIMATION, DRIFT
DETECTION, and FLICKER-AWARE MASKING — explicitly NOT drift-free
continuous compensation. There is no way to automatically clear
`is_stale` once set (no automatic pitch-keypoint detector exists in this
codebase to re-anchor against — the same SoccerNet-adjacent gap named in
Section 3); the only reset path (`reanchor()`) requires an externally
supplied fresh homography. Hysteresis and padding measurably REDUCE, not
ELIMINATE, player-motion contamination risk. The `is_stale` threshold is
synthetic-calibrated only, not validated against real footage's actual
motion rate (see above). Wired into `adapter.py` as an optional,
additive `camera_motion_correction` parameter (regression-tested to be a
no-op when omitted) but NOT wired into the live pipeline orchestrator's
default execution path — that remains separate, deferred follow-up work.

### Milestone 39 — Pretrained Pitch-Keypoint Detection (Qualified Local-Only Adoption, ADR-015)

**Scope reminder (ADR-014): this entire milestone is a strictly LOCAL,
NON-SERVED research prototype.** `production/src/cv/pitch_keypoint_detector.py`
is not imported by, and must not be wired into, `production/src/serving/api.py`
or any other live endpoint. All figures below come from Roboflow's
**hosted** inference endpoint (`football-field-detection-f07vi`, model
version 14 — the only version with a deployed model; the latest dataset
version, 18, has none), called over a real network round trip — timing
figures include that network latency and are NOT a local-inference cost.

**First-pass verdict, and why it was revisited.** An initial evaluation
against the Milestone 34B clip found: (a) a "static" (frame-0-anchored)
homography appeared to outperform fresh per-frame solving, and (b) the
model's keypoint confidence scores appeared decoupled from positional
accuracy across the board (20-50m error on a 100m pitch at >0.99
confidence). Three follow-up checks, run specifically to rule out
testing-condition artifacts before accepting a negative result, reversed
both findings:

1. **The low-motion test window was the cause of "static wins."** The
   initial comparison's window (clip frames 0-57) measured at near-zero
   real camera motion. Re-run on a genuinely high-motion window (frames
   804-864, ~5x the clip-median frame-to-frame displacement): **static
   homography reuse degraded from 7.3m to 38.6m error as the window
   progressed** (a real drift signature), while **fresh per-frame solving
   stayed flat at 13.7-14.6m** across the same window. Anchor-based
   recalibration does what ADR-013 asked of it once real motion is
   present to test against.
2. **The pitch-geometry correspondence table was re-verified against
   ADR-002/ADR-009's established 100x68m grid** — re-diffed line-by-line
   against `roboflow/sports`' source and cross-checked against
   `calibration.py`'s `PITCH_CORNERS_METERS`. No mismatch found; this
   project's recurring 100-vs-105 pitch-dimension bug class is ruled out
   as a contributor here.
3. **Confidence/accuracy decoupling is concentrated, not uniform.** A
   per-vertex breakdown (25-frame sample) found six specific vertices —
   **19, 22, 23, 24, 25, 26** — detected at ~0.97-1.0 confidence in ~100%
   of frames with **19-55m median error**, while the other
   regularly-visible vertices (15, 16, 17, 20, 21, 27, 28, 29, 30) stay
   within 0.6-8m. Every bad vertex is a far-side/background landmark under
   this camera's framing (e.g. vertex 25 at `(length,0)`: 53m error, vs.
   its near-side mirror vertex 30 at `(length,w)`: 1.6m error) — consistent
   with real, physically-explicable perspective foreshortening, not random
   noise. **Excluding just these six vertices** dropped the general-sample
   median residual from 4.7m to 1.5m, and the high-motion-window LOOCV
   median from ~14.1m to **~6.2m**, with the drift-free flatness from
   Check 1 preserved.

**Decision (ADR-015): qualified adoption, not full rejection.** Fresh
per-frame anchor-based homography solving, with vertices 19/22/23/24/25/26
excluded from the correspondence table, is adopted **for rough visual
overlay / tactical-map rendering only** (e.g. Milestone 38's renderer),
under ADR-014's local-only constraint. **This is explicitly NOT validated
as accurate enough for `BiomechanicalPitchControl`, `DeepHit`, or any
quantitative tactical/cheat-sheet analysis** — ~6m median error under real
motion does not meet the positional fidelity the StatsBomb-track physics/ML
pipeline was empirically validated against.

**Two explicitly unresolved limitations, not assumed away:**
- The six-vertex exclusion list was derived from the ONE real camera
  framing available throughout this project. Whether the same six
  vertices are the unreliable ones under a different camera angle or
  elevation is UNTESTED — no second, differently-angled clip has been
  available. The foreshortening explanation makes an analogous pattern
  plausible elsewhere, but a different framing would plausibly implicate
  a *different* subset of vertices; this is reasoning from a physical
  mechanism, not a second measurement.
- ~6m median error is a reasonable bar for a human-watched visual overlay,
  but no work has established what accuracy the ML pipeline actually
  requires before its own outputs degrade. The real threshold is unknown,
  not merely "probably higher than 6m."

**Update (ADR-016): two adaptive alternatives tried, both rejected --
fixed list remains adopted.** The first of Milestone 39's two unresolved
limitations above (the six-vertex list is derived from one camera angle)
motivated a follow-up attempt to replace the fixed list with a RUNTIME
mechanism that would generalize to an unseen camera by construction,
against a pre-committed bar (clearly beat or match the fixed list's
~6.2m LOOCV median, or stop and report honestly):

1. **Per-frame iterative outlier rejection** (reusing
   `team_classifier.classify_teams`'s Milestone 28 masking-aware
   fit/flag/refit/re-evaluate template, extended to trim a fixed fraction
   every round rather than once). **Measured 35-39m median -- clearly
   worse.** With only 10-15 confidently-detected points per frame and
   ~30-40% of them genuinely bad, there isn't enough within-frame sample
   for statistics to cleanly separate bad from mediocre-but-real residuals
   -- the fixed list's aggregate analysis only found a clean signal by
   averaging the same six vertices' behavior over 25 frames, something a
   single-frame mechanism cannot do.
2. **Multi-frame rolling reliability tracking** (an EWMA of each vertex's
   residual accumulated across frames, decay 0.9, excluding vertices whose
   rolling score exceeded 2x the median among trusted vertices). **Measured
   32-38m median -- still worse, and in a more concerning way**: it most
   frequently flagged vertices 30, 15, 16, and 29 -- all four independently
   verified as RELIABLE in the original analysis -- while under-flagging
   several genuinely bad ones. Accumulating evidence against each frame's
   own noisy, still-biased per-frame fit compounds that bias rather than
   averaging it out.

Both attempts failed the pre-committed bar; per the agreed stopping rule,
no third variant was attempted. **ADR-016 documents both as an honest,
diagnosed negative result and keeps ADR-015's fixed 6-vertex list as the
adopted, recommended approach.** Both adaptive mechanisms remain in
`pitch_keypoint_detector.py` (not deleted -- this project's established
practice for a real, informative negative result, per ADR-013's treatment
of Milestone 37), with the module's own docstring stating plainly that
`excluded_vertices=ADR015_KNOWN_UNRELIABLE_VERTICES` is the only path
validated to ~6-7m and is what real usage should pass.

**This result makes the roadmap item below (testing the six-vertex list
against a second camera angle) MORE load-bearing, not less**: there is
now no tested adaptive fallback if the fixed list turns out not to
transfer to a different framing -- the generalization argument for an
adaptive mechanism remains theoretically sound, but this ADR found the
opposite of empirical support for it on the only camera angle available,
so it must not be treated as a safety net the fixed list's own
camera-angle caveat can lean on.

### Milestone 41 — Tactical Map Rendering (Local-Only, Visual-Only, ADR-014/015/016)

**Scope reminder, unchanged: strictly LOCAL, NON-SERVED (ADR-014); PLAYER
POSITIONS ONLY, no pitch-control/zone analysis (ADR-015/016)** — nothing
in `production/src/reporting/` or `production/src/spatial/control.py` is
imported anywhere in this milestone's code.

**A load-bearing documentation/reality mismatch was found and corrected
before any other work began.** This milestone's task description referred
to an existing "Milestone 38" `overlay_renderer.py`/`video_export.py` as
groundwork to reuse. Neither file exists anywhere in this repository or
its git history (`find`/`git log --all` both confirm this) — no
`cv2.rectangle`/`cv2.putText`/`VideoWriter`/drawing code of any kind
existed anywhere under `production/src/cv/` before this milestone, despite
`context.md`'s prose describing Milestone 38 as "the only currently-
completed, fully independent Track A deliverable." **This is now
documented plainly rather than silently reused-as-if-real or silently
patched over.** `production/src/cv/pixel_overlay_renderer.py` was built
fresh this milestone to fill the gap (bounding boxes, team-color-coded via
a NEW display-color mapping — `team_classifier.py` itself defines no
colors, only role labels — track_id labels, ball marker), clearly labeled
in its own docstring as new code, not a reuse or modification of anything
prior.

**What was built:**
- `pipeline.py`: one ADDITIVE, backward-compatible yield key
  (`render_frame_data`) exposing the raw per-frame tracks/ball/team data
  `CVPipeline.process_video` already computes internally — no existing
  key, behavior, or test changed (`test_cv_pipeline.py` passes unchanged).
- `pixel_overlay_renderer.py` (new, see above) and
  `tactical_map_renderer.py`: `transform_players_to_pitch_space` (Step 1;
  reuses `pitch_keypoint_detector.py`'s `detect_pitch_keypoints`/
  `solve_homography_from_keypoints(..., excluded_vertices=
  ADR015_KNOWN_UNRELIABLE_VERTICES)` exactly, per ADR-015's recommended
  call) and `render_tactical_map` (Step 2; top-down pitch diagram drawn
  from `pitch_keypoint_detector.PITCH_KEYPOINTS_METERS`'s own verified
  vertex table, not a re-derived pitch geometry).
- `video_export.py` (new): `export_side_by_side_video`, composing the
  pixel-space overlay and tactical map side by side, one output frame per
  SOURCE video frame (including frames `process_video` itself skips
  entirely — those render explicit "no data"/"unavailable" placeholders
  rather than being dropped, which would desync the output's frame
  count/timing from the source).
- **Staleness handling, verified by test, not just described**: a failed/
  stale homography solve (`solve_homography_from_keypoints`'s own
  `is_stale`, ADR-016) makes `transform_players_to_pitch_space` return
  `None` for that frame; `render_tactical_map` renders an explicit
  "Tactical map unavailable this frame" placeholder, never a reused prior
  frame's positions. The accuracy-caveat caption
  (`"Approximate positions (~6-7m accuracy) -- visual reference only, not
  validated for tactical/zone analysis"`) is drawn on every frame,
  confirmed via pixel-region inspection (not just code review) to
  actually render as pixels, on both the valid and unavailable paths.

**Real-clip test (`data/raw/test_match.mp4`, 20 frames, real
`CVPipeline.process_video` orchestration, real per-frame Roboflow API
calls):** **20/20 frames (100%) produced a valid tactical map** — every
sampled frame had enough confidently-detected, non-excluded keypoints to
clear ADR-016's reliability floor. Render time ~1.1s/frame end-to-end
(dominated by the hosted-API network round trip per frame, consistent
with `pitch_keypoint_detector.py`'s own documented cost caveat — not
representative of local-inference latency).

**A second, more significant finding from the qualitative visual check —
stated honestly, not glossed over.** Step 4.4 explicitly asked for a
qualitative comparison between the rendered tactical-map dots and the
real team shape visible in the pixel-space overlay on the SAME frame (no
ground truth exists for this clip, so this is a sanity check, not a
quantitative measurement). On two inspected frames, **the tactical map's
dot positions did NOT look plausible relative to the real, visibly
spread-out team shape in the broadcast frame**: players spanning roughly
the left two-thirds to full width of the broadcast image collapsed onto a
narrow band on the RIGHT side of the tactical-map pitch diagram (never
reaching the left half of the diagram at all), rather than spreading
proportionally to match the broadcast view's real spatial spread.

**Why this is not a contradiction of ADR-015/016's ~6-7m figure, but a
real gap in what that figure covers**: that figure was measured via
leave-one-out cross-validation on DETECTED KEYPOINTS — i.e., accuracy
INTERPOLATING within the region the correspondence points actually cover.
This camera's detected keypoints cluster in the right-center of the frame
(center circle through the right goal, per Milestone 39's own findings);
PLAYER positions, by contrast, span the ENTIRE visible pitch, including
regions well to the left of where any correspondence point was ever
detected. A homography fit from one region can be accurate near that
region and still EXTRAPOLATE badly outside it — a standard numerical-
methods caveat that this milestone is the first to make visible, because
it is the first time this pipeline has transformed PLAYER positions
(spanning the full frame) rather than only validating KEYPOINT positions
(concentrated in the region they were detected in).

**This does not change ADR-015/016's own conclusions** — the fixed-list
approach and its ~6-7m LOOCV figure remain accurate statements about what
they actually measured. It DOES mean this milestone's own rendered output,
while satisfying every process requirement (staleness handling, caption
visibility, frame-complete output), is NOT yet demonstrated to produce
visually trustworthy player positions across the FULL pitch — a real,
newly-surfaced limitation, not a pass. Investigating and (if possible)
correcting this extrapolation gap is the natural next item on this
milestone's own follow-up list, separate from, and prior to, any future
attempt to widen scope beyond visual rendering.

**Update — ADR-017 trust-radius gating, the final resolution of the gap
above.** ADR-017 measured HOW homography accuracy actually varies with
pixel-space distance from a frame's reliable-keypoint cluster centroid:
tight and bounded within ~150px (median 3.35m, max 3.86m, n=24), then
both worse AND statistically unpredictable beyond it (Spearman r=0.582,
p<0.0001, worst observed case 933m) — and found that **72.5%** of real
detected players, on this clip, fall OUTSIDE that 150px radius. Rather
than leave `tactical_map_renderer.py` rendering every player at equal,
false confidence, `transform_players_to_pitch_space` and
`render_tactical_map` were extended with TRUST-RADIUS GATING: each
player's pixel distance from the SAME reliable-keypoint centroid is
computed every frame; players within `TRUST_RADIUS_PX` (150px, ADR-017's
measured boundary, reused verbatim) render as the original SOLID
team-color dot, and players beyond it render as a distinct FAINT, HOLLOW
(outline-only) marker in the same team color — deliberately never a
solid dot, so the two confidence levels can never be visually confused. A
second caption ("N/M players in reliable range (ADR-017, <150px)") was
added alongside the original accuracy-caveat caption, both confirmed via
pixel-region inspection to actually render, not just described.

**Re-running the real-clip test after gating: two independent
measurements now agree.** ADR-017's original measurement (25-frame
whole-clip systematic sample) found **27.5%** of players within the trust
radius. A fresh `export_side_by_side_video` run (20 frames from the start
of the same clip — a DIFFERENT sample window, not a re-run of the same
frames) found **100/460 = 21.7%**. These two numbers are not the same
because they are not measuring the same frames — but both fall in the
same ~20-27% range, and both support the identical underlying
conclusion: **a minority of detected players — roughly a fifth to a
quarter, consistently across two independent samples — fall within
reliable homography range on this clip.** This replication, not just the
original single measurement, is what makes the trust-radius gating
decision solid rather than resting on one sample's noise.

**Visual confirmation, on real data, not just synthetic tests.** A
rendered frame from the gated output was inspected directly: five solid,
filled team-color dots clustered near the right-center of the pitch
(where the reliable-keypoint cluster actually sits), with the remaining
detected players — spread across the rest of the pitch, matching the
pixel-space overlay's real team shape on the same frame — rendered as
small, faint, hollow markers, unambiguously distinct from the solid
dots. Both captions rendered together and legibly. **The result is a
sparse-but-trustworthy tactical map (a handful of confidently-placed
dots), not the dense-but-unreliable one Milestone 41 originally
produced** — the honest resolution ADR-017 called for, not a rendering
tweak that quietly restored the appearance of full-pitch coverage.

**What remains open, unchanged by this update**: whether the 150px trust
radius, or the six-vertex exclusion list underneath it, generalizes to a
different camera angle is still untested (ADR-015's own caveat, still
load-bearing). Nothing here claims that question is resolved — only that,
ON THIS camera framing, the renderer now honestly represents which
player positions it does and does not trust, rather than presenting all
of them with equal, false confidence.

---

## 3. The Real-Data Blocker

**The entire CV track is blocked on one thing: legitimately-licensed real
broadcast footage with ground-truth annotations.** SoccerNet's tracking
dataset requires a signed NDA/research-use agreement to obtain a download
password (Milestone 25); that access has not been obtained in this
environment at any point across Milestones 25-33. Every "never validated
against real footage" limitation listed in Section 2 — detection
precision/recall, tracking ID-switch behavior under real occlusion,
calibration accuracy against a real lens, team classification under real
jersey patterns and lighting, the shot classifier's real-world adversarial
failure rate, and end-to-end throughput on real video — traces back to
this single blocker, not nine independent gaps. Two components (Milestones
25's and 29's YOLO paths) were validated against real *photographs*
specifically to avoid the weaker claim of "validated on a synthetic
idealization only," but a static photograph does not exercise motion
blur, compression artifacts, camera pan/zoom, or broadcast-graphics
clutter — real video remains a categorically different and harder test
this track has not yet faced.

**Update (Milestone 34B):** the pipeline has since run end-to-end on one
real, private, unannotated broadcast-style clip (see the Update note in
the Executive Summary) — real tracking and real end-to-end throughput
data now exist for
the first time (Milestones 26 and 32 above), and a real bug (ball
detection's `imgsz` default) was found and fixed as a direct result. This
is genuine progress, but it is **not the SoccerNet unblock**. This one
clip has no ground-truth annotations, so it cannot answer detection
precision/recall, a true ID-switch rate, or calibration accuracy against a
real lens — the three most consequential items in the Future Validation
Roadmap below remain exactly as blocked as before. Treat this clip as
useful for catching real integration bugs (which it already has), not as
a substitute for licensed, annotated validation data.

---

## 4. Known Adversarial Failures & Methodological Lessons

**(a) Camera-motion velocity confound (Milestone 26).** Raw pixel
displacement conflates player motion with camera motion. Demonstrated
concretely, not just asserted: a synthetic video panning a crop window
across a completely static photograph reported two people "moving" at
roughly -170 to -230 px/s, despite neither actually moving at all within
the source image. Every velocity this module produces is named
`vel_pixels_per_sec`, never `vel`, specifically so this cannot be mistaken
for calibrated player speed.

**(b) The KMeans masking effect (Milestone 28).** Fitting outlier
detection on data that already contains the outliers lets those outliers
pull the fitted centroids toward themselves and inflate the very spread
statistic meant to catch them — a well-known statistical failure mode.
Concretely observed here: the naive single-pass "red" centroid sat at
`[0.929, 0.124, 0.857, 0.857]`, visibly displaced from the true `[1, 0, 1,
1]` by the yellow GK and black referee it had silently absorbed. Fixed
with an iterative refit (fit → flag top-20%-by-distance candidates →
remove → refit on the remainder → re-evaluate every original point against
the REFINED centroids), which recovered the true `[1, ~0, 1, 1]` centroid
— a measured 0.2474 shift proving the refit is not a no-op.

**(c) The circular-hue wraparound bug (Milestone 28).** Hue is a circular
quantity; OpenCV's 0-179 scale wraps, with 0 and 179 both meaning red. A
naive linear mean of hue values straddling that boundary is nonsensical —
concretely, averaging a crop that was half hue≈2 and half hue≈177 (both
clearly red) gave a naive linear mean of 89.50 (green) versus the correct
circular mean (cos/sin encoding, average the vectors, `atan2` back) of
179.50 (correctly red).

**(d) The homography vector-transformation trap (Milestone 30).** A
homography's perspective divide is mathematically valid only for actual
points (with an implicit w=1), never for a free displacement vector.
Applying it to a velocity vector anyway does not merely rescale the
result incorrectly — it fabricates a wrong DIRECTION. Concretely: a true
2m, purely-X-axis movement near the far touchline produced a correct
velocity of `[50, 0]` m/s via point-then-difference, versus `[3278.3,
-2752.0]` m/s via the naive vector-transform shortcut — wrong by
thousands of m/s AND inventing a large Y-component that should have been
exactly zero.

**(e) The shot classifier's honest adversarial failure (Milestone 31).** A
2-feature heuristic (green ratio + edge density) cannot, at the thresholds
currently in use, distinguish a genuine tactical wide view from a
grass-heavy close-up with enough incidental texture to cross the edge
threshold. This was deliberately constructed and confirmed to fail
(`green_ratio=0.6661, edge_density=0.0600 -> True`, incorrectly) rather
than tuned away — it is documented as a real, currently-unresolved
limitation of this approach, not a threshold that merely needs
re-tuning on this same synthetic example.

**(f) The WebSocket close-frame protocol bug (Milestone 33).** Caught only
by testing against the ACTUAL running uvicorn server — the automated
mocked-transport (`TestClient`) suite never triggered it, because its
in-process transport did not enforce the real WebSocket protocol's
~123-byte close-frame `reason` limit the way a genuine `websockets`-backed
server connection does. This is a distinct category of lesson from (a)-(e)
above: those are perception/estimation failure modes in the CV models
themselves; this one shows that even integration/plumbing code can have
failure modes completely invisible to synthetic or mocked testing, and
that "passes every automated test" is not the same claim as "works against
the real system." Manual testing against a real running instance remains
necessary even with a thorough automated suite in place.

**(g) The detection-confidence-flicker discovery, and a standalone-harness
vs. real-orchestrator discrepancy (post-Milestone-34B investigation,
feeding directly into Milestone 37).** Investigating WHY the Milestone
34B real clip showed 152 unique `track_id`s against a real ~22-25-person
roster found the dominant cause was NOT occlusion: **~23% of all
person-class detection candidates fall within ±0.1 confidence of the 0.5
tracking threshold** (a near-even split, ~1,633 just below / ~1,820 just
above, out of 33,078 total candidates on the M34B clip), and this
borderline flicker — not two players' boxes overlapping — is what
dominates short-lived track fragmentation (32% of track-runs ≤3 frames).
Strict bounding-box overlap explains only **~3%** of fragmentation
events; a looser "nearby" proxy (within 1.5x average box diagonal)
explains **~29%** — the gap between these two numbers is itself the
finding: true occlusion is rare, proximity-without-overlap is a weak
partial explanation, and the true dominant cause is per-detection
confidence noise, not any inter-player interaction. A secondary,
independent finding: sub-threshold detections skew ~13% smaller and are
1.4x overrepresented in the far half of frame (greater camera distance)
versus confidently-tracked detections — a real, partial, non-exclusive
contributing factor. **Separately, and just as important
methodologically:** the original 152-unique-track-ID figure was traced
to `tracker.py`'s standalone `run_tracking()` harness, which tracks every
frame of a clip unconditionally — it has no import of, or call to,
`shot_classifier`/`is_tactical_view` at all, confirmed directly by
grepping the file. The REAL production orchestrator, `CVPipeline.process_video()`
in `pipeline.py`, correctly gates tracking behind `is_tactical_view()`
(confirmed directly at lines 291–306: the `continue` at line 301 executes
before the tracking call at line 306 is ever reached). The two code paths
behave differently on this exact point; the 152-ID figure describes the
harness's behavior, not necessarily what the deployed orchestrator would
produce on the same footage — a distinction that matters for anyone
extrapolating from that figure. This confidence-flicker finding directly
motivated Milestone 37's confidence-hysteresis and bounding-box-padding
masking rules (see that milestone's entry in Section 2), rather than
masking against static occlusion alone.

---

## 5. Future Validation Roadmap

Prioritized — every item below is downstream of Section 3's single
blocker (real, legitimately-licensed footage with ground truth):

1. **Detection P/R on real broadcast frames** — Milestone 25's original,
   never-completed validation gate. Nothing else in this roadmap can be
   properly prioritized against real data until this exists.
2. **Tracking ID-switch frequency on real occlusions/crossings** —
   Milestone 26's synthetic pan produced zero switches by construction
   (nothing ever occluded anything). A real private clip now shows 152
   unique track_ids against a real ~22-25-person roster over 970 frames —
   a real signal of heavy churn — but without ground-truth identity labels
   this cannot be turned into a true switch *rate*, and whether the 800
   px/s sanity ceiling is anywhere near the right threshold remains
   unknown.
3. **Calibration accuracy against real lens distortion and panning** —
   Milestone 27's homography math is proven correct against a synthetic
   pinhole model with zero lens distortion; real broadcast lenses are not
   distortion-free, and no panning/zooming feed has been tested against
   any homography at all.
4. **Team classification robustness under real jersey patterns/lighting**
   — Milestone 28 is validated on solid, fully-saturated swatches only;
   stripes, sponsor logos, printed numbers, and real lighting/shadow
   variation are all untested.
5. **Whether the shot classifier's known adversarial failure mode (4e
   above) occurs often enough on real footage to matter in practice** — a
   single constructed example proves the failure mode EXISTS; only real
   footage can establish its real-world frequency and cost.
6. **End-to-end throughput on real hardware with real video** — Milestone
   32's profiling code has now run once, on one private clip, on one
   machine: median 116.73ms/frame, 8.57 effective fps, well below this
   clip's 28fps source rate. A broader characterization (multiple clips,
   multiple hardware targets, a GPU-vs-CPU breakdown, and a designed
   frame-skipping/rolling-buffer strategy for real-time use) remains open.
7. **Test the Milestone 39 six-vertex exclusion pattern (19, 22, 23, 24,
   25, 26) against a second, differently-angled real clip** — the current
   exclusion list (ADR-015) was derived from the one camera framing
   available throughout this project. Whether the same vertices are the
   unreliable ones under a different camera elevation/angle is untested;
   the foreshortening explanation is plausible but unverified beyond this
   one clip. **Raised in priority by ADR-016**: two attempts to replace
   the fixed list with a self-adapting mechanism (which would have made
   this item moot) both measured worse than the fixed list and were
   rejected, so there is currently no tested fallback if the list doesn't
   transfer to a different angle — this item is the only way to actually
   resolve that open question, not an optional nice-to-have.
8. **Determine what positional accuracy the ML/physics pipeline
   (`BiomechanicalPitchControl`, `DeepHit`) actually requires, and test
   any future CV-derived positional estimate against that bar** —
   Milestone 39's ~6m median error was judged adequate for a visual
   overlay demo only, against no established quantitative threshold. The
   real accuracy bar for feeding CV output into the StatsBomb-track
   pipeline is currently unknown, not merely assumed to be unmet.
9. **Investigate Milestone 41's keypoint-interpolation-vs-player-
   extrapolation gap** — the ~6-7m LOOCV figure (ADR-015/016) measures
   accuracy INTERPOLATING within the detected-keypoint region; Milestone
   41's real-clip rendering test found player positions across the FULL
   frame visually implausible (collapsing onto a narrow band far from
   where the broadcast footage shows them), consistent with the
   homography EXTRAPOLATING poorly outside the keypoint region. Candidate
   directions (none attempted yet): constrain/clamp rendered positions to
   a plausibility region, weight the homography fit to reduce
   extrapolation error, or explicitly flag/exclude players detected far
   from the fitted keypoint region rather than rendering their (likely
   unreliable) transformed position at all.

---

## 6. System Capabilities

Everything below is **verified via synthetic, adversarial, or mocked
tests only** — explicitly labeled as such, not implied to be
production-proven against real footage (see Section 3):

- Processes a video file frame-by-frame via a streaming generator (no
  full-video memory accumulation), reading the real fps from the file
  itself rather than assuming one.
- Skips non-tactical frames (shot classification) BEFORE running
  detection/tracking on them — the actual compute-saving mechanism,
  confirmed via exact mocked call-count verification (YOLO/tracking
  called exactly once per tactical frame, never on a skipped one).
- Correctly computes velocity across skipped-frame gaps using the TRUE
  elapsed time, not a fixed `1/fps` assumption, with an explicit staleness
  cutoff and fallback for gaps too large to trust.
- Produces the IDENTICAL tensor contract (`player_pos`, `player_vel`,
  `is_teammate`, `ball_pos`, `fatigue_mod`) `feature_extractor.extract_features`
  has consumed since Milestone 3 — proven by actually calling that
  unmodified function on the adapter's real output, not just asserting key
  names match.
- Per-connection state isolation in the live WebSocket API: a fresh
  `CVPipeline` instance per connection, verified to never leak track/ball
  data between two simultaneous connections.
- Correct async integration: blocking CV processing is offloaded to a
  worker thread per frame, verified to keep the FastAPI event loop
  responsive to other concurrent requests during a deliberately slow
  (3-second) mocked CV step.
- Honest real-time-pacing reporting: aligns sends to real video timing
  when keeping pace; reports actual lag rather than forcing or
  misrepresenting the pace when falling behind.
- Path-safety validation, clean WebSocket closes (not crashes or hangs)
  on missing parameters, path traversal attempts, nonexistent files, and
  genuinely unreadable/corrupt video files.

**Update (Milestone 34B):** the full pipeline has now run end-to-end on
one real, private, unannotated broadcast-style clip, yielding real tracked
players, real ball detections (after the Milestone 29 `imgsz` fix), and a
real (not real-time) throughput measurement — see the Update note in the Executive
Summary and the Milestone 26/29/32 entries above. This is real progress on
integration correctness, not a validated-on-real-broadcast-video claim for
any individual component's accuracy — see Section 3.

**Explicitly NOT yet capable of:** automatic pitch-keypoint/line detection
(correspondences are hand-specified); continuous re-calibration across a
panning/zooming feed; distinguishing a replay from live play; real camera
capture (webcam) input; wiring a real homography through the live API
(CV-sourced threat values are not yet physically calibrated); real-time
throughput on the one clip measured so far (8.57 fps effective vs. 28fps
source); any ground-truth-anchored accuracy validation on real broadcast
video (still blocked on SoccerNet NDA access).

---

## 7. Figures That Could Not Be Directly Reconfirmed

One category, not a specific number: **every figure that would require
real SoccerNet tracking data or a real broadcast video file** could not be
reconfirmed, because the underlying tests (`test_soccernet_baseline_detection_accuracy`
in `test_cv_detector.py`, the real-footage test in `test_cv_tracker.py`,
and `test_pipeline_on_real_video_and_throughput` in `test_cv_pipeline.py`)
all skip in this environment for lack of that data — there is no stale or
recalled number being presented as confirmed here; these results simply do
not exist yet. All other numbers cited in this document were either
re-run directly against a persisted test file, or (Milestone 26's
camera-pan pixel-velocity figures specifically, which were an ephemeral
manual demonstration during Milestone 26 and not a hard-coded assertion in
`test_cv_tracker.py`) regenerated fresh, byte-for-byte reproducibly, using
the exact same construction, immediately before writing this document.

**Figures added in the Update (Milestone 34B, first real-footage run):**
the M26 real tracking figures (152 unique track_ids, 146 persisting,
15,090 nonzero velocity observations), the M29 `imgsz` A/B-test figures,
and the M32 throughput figures (116.73ms median, 127.23ms p95, 8.57 fps)
were all
directly measured by running the real code against `data/raw/test_match.mp4`
in the same debugging session that produced this update, not recalled from
a prior write-up — this is a first-time measurement, not a
reconfirmation of an earlier claim. These figures are specific to this
one private, unannotated clip and this one machine; they are not claimed
to generalize. Every figure requiring ground-truth annotations (SoccerNet
detection P/R, a true ID-switch rate, calibration accuracy against a real
lens) remains unmeasurable in this environment, exactly as before this
update — see Section 3.
