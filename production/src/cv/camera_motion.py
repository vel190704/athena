"""Milestone 37 (Module 4): frame-to-frame camera-motion estimation and
drift-flagged composition.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25 -- nothing in `production/src/models`, `production/src/
pipeline`, `production/src/spatial`, `production/src/physics`, or
`production/src/serving` is imported by or imports from this module. This
module DOES import `production/src/cv/calibration.py` (read-only reuse of
`transform_points`) but does not modify it.

SCOPING NOTE (read before assuming this "solves" camera motion): a fully
ground-truth-anchored fix -- automatically detecting real pitch keypoints
each frame and re-solving Milestone 27's homography fresh against them --
is NOT buildable right now. This codebase has no automatic pitch-keypoint
detector, and building one is realistically gated behind the same
SoccerNet access blocker that has stalled ground-truth CV validation since
Milestone 25 (see `CV_PIPELINE_FINDINGS.md` Section 3). This module
instead implements the buildable increment: frame-to-frame optical-flow
motion estimation, composed over time, with EXPLICIT MEASURED drift
quantification -- because composing estimates over time is known to
accumulate error silently, and this project's discipline (Milestones 12,
14, 30) has consistently been to MEASURE exactly this kind of silent
failure mode rather than assume it away. See `production/tests/
test_camera_motion.py` for the actual measured drift-vs-frame-count curve.

WHY SPARSE OPTICAL FLOW (goodFeaturesToTrack + calcOpticalFlowPyrLK), NOT
DENSE (Farneback): a homography has only 8 degrees of freedom, so a modest
number (tens) of well-distributed, high-quality background corners is
sufficient to fit it robustly -- dense per-pixel flow computes far more
information than a homography fit needs, at meaningfully higher cost per
frame. Just as importantly, masking is trivial and cheap with sparse
points: a player region is excluded by simply not SEEDING a corner
detector inside it (`goodFeaturesToTrack`'s own `mask` parameter). Dense
flow would require computing a full-frame flow field first and THEN
masking it out post-hoc, discarding compute already spent on exactly the
pixels we intend to throw away.

THE FLICKER PROBLEM THIS MODULE MUST ACCOUNT FOR (see `CV_PIPELINE_FINDINGS.md`'s
Known Adversarial Findings): a prior diagnostic on the Milestone 34B real
clip found that ~23% of person-class detection candidates fall within
+/-0.1 confidence of the 0.5 tracking threshold, and this borderline-
confidence FLICKER -- not occlusion -- is the dominant cause of short-lived
player tracks (only ~3% of track-fragmentation events correlate with true
bounding-box overlap). A player whose detection confidence dips to, say,
0.42 for a single frame has NOT stopped existing or moved off-camera -- but
a masking step that only trusts the CURRENT frame's >=0.5-confidence boxes
would un-mask that player for exactly one frame, potentially seeding
background corner points on a moving person and contaminating the camera-
motion estimate. Steps 1.3a (confidence hysteresis) and 1.3b (box padding)
below exist specifically to absorb this measured phenomenon, not a
hypothetical one.
"""

import cv2
import numpy as np

# --- Masking constants ---

# Matches pipeline.py's DEFAULT_TRACKING_CONFIDENCE_THRESHOLD -- a
# detection at or above this confidence is ALWAYS masked out of the
# background feature set, no further conditions needed.
BASE_MASK_CONFIDENCE_THRESHOLD = 0.5

# Step 1.3a: a detection between this floor and BASE_MASK_CONFIDENCE_THRESHOLD
# is STILL masked if a spatially-corresponding detection was confidently
# (>=0.5) present two frames ago -- absorbing a single-frame confidence dip
# without requiring a fix to track fragmentation elsewhere in the pipeline.
# Chosen as a band directly straddling the measured ~23% near-threshold
# concentration (roughly +/-0.15 around 0.5); an unvalidated-but-reasoned
# judgment call, same status as every other hand-tuned constant in this
# project until checked against more real footage.
HYSTERESIS_CONFIDENCE_MIN = 0.35

# IoU threshold for treating a dip-confidence box as "the same real player"
# as a box seen with high confidence two frames prior. Deliberately modest
# (not requiring near-perfect overlap) since a player can move meaningfully
# in the two-frame gap this hysteresis check spans.
HYSTERESIS_IOU_THRESHOLD = 0.3

# Step 1.3b: every masked box is padded by this fraction (both dimensions)
# before exclusion -- directly justified by the measured ~13% box-height
# gap between confidently-tracked and sub-threshold detections on the
# Milestone 34B clip: a real player's box can legitimately be smaller (or
# shifted) on any given frame than the box that triggered its masking,
# and this margin buys robustness against exactly that, at the modest
# cost of discarding a few more background pixels near each player.
BBOX_MASK_PADDING_FRACTION = 0.175

# --- Optical flow constants (standard OpenCV pyramidal LK parameters;
# reasoned defaults, not validated against real broadcast footage yet) ---
GOOD_FEATURES_MAX_CORNERS = 300
GOOD_FEATURES_QUALITY_LEVEL = 0.01
GOOD_FEATURES_MIN_DISTANCE = 8
LK_WIN_SIZE = (21, 21)
LK_MAX_PYRAMID_LEVEL = 3

# A homography needs 4 non-collinear point correspondences at the
# mathematical minimum; this project consistently prefers a considerably
# higher floor for a ROBUST fit (see ADR-002's "prefer well-spread
# landmarks" note for Milestone 27's manual calibration) -- 20 is a
# reasoned, not empirically-tuned, floor.
MIN_BACKGROUND_FEATURE_POINTS = 20

RANSAC_REPROJ_THRESHOLD_PX = 3.0


def _bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _select_boxes_to_mask(
    player_bboxes_prev: list,
    player_confidences_prev: list,
    tracked_in_prev_prev: list | None,
) -> list:
    """Step 1.3 + 1.3a: which of `player_bboxes_prev` (each `[x1,y1,x2,y2]`)
    should be excluded from the background feature set, given the measured
    detection-flicker phenomenon.
    """
    selected = []
    for bbox, conf in zip(player_bboxes_prev, player_confidences_prev):
        if conf >= BASE_MASK_CONFIDENCE_THRESHOLD:
            selected.append(bbox)
        elif HYSTERESIS_CONFIDENCE_MIN <= conf < BASE_MASK_CONFIDENCE_THRESHOLD and tracked_in_prev_prev:
            if any(_bbox_iou(bbox, prior) > HYSTERESIS_IOU_THRESHOLD for prior in tracked_in_prev_prev):
                selected.append(bbox)
        # conf < HYSTERESIS_CONFIDENCE_MIN, or no matching prior-prior box:
        # not masked -- either genuinely not a player, or a dip with no
        # supporting recent-confident evidence to justify excluding it.
    return selected


def _pad_bbox(bbox, padding_fraction: float, frame_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Step 1.3b: pads `bbox` (`[x1,y1,x2,y2]`) by `padding_fraction` of its
    own width/height in each direction, clamped to the frame bounds."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    pad_w, pad_h = width * padding_fraction, height * padding_fraction
    frame_height, frame_width = frame_shape[0], frame_shape[1]
    return (
        max(0.0, x1 - pad_w),
        max(0.0, y1 - pad_h),
        min(float(frame_width), x2 + pad_w),
        min(float(frame_height), y2 + pad_h),
    )


def estimate_camera_motion(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    player_bboxes_prev: list,
    player_confidences_prev: list,
    tracked_in_prev_prev: list | None = None,
) -> np.ndarray | None:
    """Estimates the frame-to-frame camera motion between `frame_prev` and
    `frame_curr` as a 3x3 homography `H` such that
    `pixel_curr (homogeneous) ~= H @ pixel_prev (homogeneous)` for a
    BACKGROUND (non-player) point -- i.e. this is a PIXEL-SPACE, prev->curr
    motion homography, not a pixel<->meter calibration (see `calibration.py`
    for that, and `CameraMotionTracker` below for how the two compose).

    `player_bboxes_prev`: `[[x1,y1,x2,y2], ...]` for `frame_prev`.
    `player_confidences_prev`: matching per-box detection confidences.
    `tracked_in_prev_prev`: boxes (`[x1,y1,x2,y2]`) that had confidence
    `>= BASE_MASK_CONFIDENCE_THRESHOLD` in the frame BEFORE `frame_prev` --
    used only for the Step 1.3a hysteresis check. `None`/empty means no
    hysteresis information is available (e.g. the very first frame pair in
    a sequence); this degrades gracefully to base-confidence-only masking
    for that one step, not an error.

    Returns `None` if too few reliable background points remain after
    masking, before OR after the optical-flow tracking step, or after
    RANSAC's own inlier filtering -- callers must not assume a fit was
    found and must handle `None` explicitly (see `CameraMotionTracker.update`).
    """
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

    boxes_to_mask = _select_boxes_to_mask(player_bboxes_prev, player_confidences_prev, tracked_in_prev_prev)
    padded_boxes = [_pad_bbox(b, BBOX_MASK_PADDING_FRACTION, gray_prev.shape) for b in boxes_to_mask]

    feature_mask = np.full(gray_prev.shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2 in padded_boxes:
        feature_mask[int(y1) : int(y2), int(x1) : int(x2)] = 0

    corners = cv2.goodFeaturesToTrack(
        gray_prev,
        maxCorners=GOOD_FEATURES_MAX_CORNERS,
        qualityLevel=GOOD_FEATURES_QUALITY_LEVEL,
        minDistance=GOOD_FEATURES_MIN_DISTANCE,
        mask=feature_mask,
    )
    if corners is None or len(corners) < MIN_BACKGROUND_FEATURE_POINTS:
        return None

    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, corners, None, winSize=LK_WIN_SIZE, maxLevel=LK_MAX_PYRAMID_LEVEL
    )
    status = status.reshape(-1)
    good_prev = corners.reshape(-1, 2)[status == 1]
    good_curr = next_pts.reshape(-1, 2)[status == 1]

    if len(good_prev) < MIN_BACKGROUND_FEATURE_POINTS:
        return None

    # Second, independent robustness layer (per Step 1.3): RANSAC rejects
    # any remaining outlier correspondences (e.g. a player point that leaked
    # through masking) that a plain least-squares fit would not.
    homography, inlier_mask = cv2.findHomography(
        good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD_PX
    )
    if homography is None:
        return None
    num_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if num_inliers < MIN_BACKGROUND_FEATURE_POINTS:
        return None

    return homography


def _reference_point_displacement_px(frame_to_frame_homography: np.ndarray, reference_point_px: tuple[float, float]) -> float:
    """A deliberately simple drift-magnitude proxy: how far does a single
    FIXED reference pixel (the frame center) move under this one frame-to-
    frame homography? This does NOT decompose the homography into separate
    rotation/translation/scale components (that would require assuming a
    specific camera model beyond the homography itself, out of scope here)
    -- but for a pan-or-zoom-dominated real broadcast camera motion, a
    single reference point's displacement tracks the overall apparent shift
    well, and is cheap and simple to accumulate frame over frame.
    """
    x, y = reference_point_px
    homogeneous = frame_to_frame_homography @ np.array([x, y, 1.0])
    new_x, new_y = homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]
    return float(np.hypot(new_x - x, new_y - y))


# Set from this milestone's own MEASURED drift-vs-frame-count curve (see
# test_camera_motion.py's test_drift_vs_frame_count_curve), NOT assumed in
# advance. That test's synthetic pan (0.03m/frame target drift) measured:
# frame 30 -> drift_budget=0.55px, true positional error at the pitch
# center=0.83m; frame 40 -> drift_budget=0.75px, error=1.10m. The center-
# point-displacement proxy this budget accumulates is deliberately a much
# smaller number than the resulting METER-space error (inverting a long
# chain of near-identity homographies is more error-sensitive than the
# raw per-frame center displacement suggests) -- 0.7 is chosen so
# `is_stale` fires right around where real measured error crosses 1m for
# THIS synthetic setup, not an independently guessed pixel budget.
DEFAULT_DRIFT_BUDGET_THRESHOLD_PX = 0.7


class CameraMotionTracker:
    """Composes frame-to-frame camera-motion estimates (Step 1) into a
    single cumulative transform since the last known-good anchor
    (Milestone 27's manually-calibrated homography, initially), and flags
    -- but does NOT attempt to correct -- excessive accumulated drift.

    CRITICAL, STATED PLAINLY: without an automatic pitch-keypoint detector
    to re-anchor against (see this module's docstring), THERE IS NO WAY TO
    RESET `is_stale` BACK TO `False` AUTOMATICALLY IN THIS MILESTONE. Once
    drift exceeds `drift_budget_threshold`, this class can only flag it via
    `is_stale` -- mirroring Milestone 32's existing stale-velocity-fallback
    pattern (flag, don't silently trust) rather than inventing a new
    convention. The ONLY way `is_stale` is ever cleared is `reanchor()`,
    which requires an EXTERNALLY supplied fresh homography (a human
    re-running Milestone 27's manual correspondence process, or a future
    automatic keypoint detector's output) -- this class cannot produce that
    homography itself.
    """

    def __init__(
        self,
        anchor_homography: np.ndarray,
        frame_shape: tuple[int, int],
        drift_budget_threshold: float = DEFAULT_DRIFT_BUDGET_THRESHOLD_PX,
    ):
        """`anchor_homography`: Milestone 27's pixel->meter calibration
        (i.e. `calibration.compute_homography`'s own output convention),
        valid at the moment this tracker is constructed. `frame_shape`:
        `(height, width)` of the video this tracker is following -- used to
        derive the fixed reference pixel (frame center) the drift-magnitude
        proxy measures against.
        """
        self.anchor_homography = anchor_homography
        self.frame_shape = frame_shape
        self.reference_point_px = (frame_shape[1] / 2.0, frame_shape[0] / 2.0)
        self.drift_budget_threshold = drift_budget_threshold

        self.cumulative_transform = np.eye(3)
        self.drift_budget = 0.0
        self.is_stale = False

    def update(self, frame_to_frame_homography: np.ndarray | None) -> None:
        """Composes one more frame-to-frame estimate (Step 1's output, for
        the frame pair immediately following whatever this tracker has
        already consumed) onto the cumulative transform.

        Step 2.4: a `None` estimate (Step 1.4's "too few reliable
        background points" case) is treated as an unknown gap in the
        composition chain -- itself a form of drift/uncertainty, NOT zero
        motion -- and immediately sets `is_stale = True`. The cumulative
        transform is left UNCHANGED in this case (there is nothing valid to
        compose), rather than silently advancing as if no motion occurred.
        """
        if frame_to_frame_homography is None:
            self.is_stale = True
            return

        self.cumulative_transform = frame_to_frame_homography @ self.cumulative_transform
        self.drift_budget += _reference_point_displacement_px(frame_to_frame_homography, self.reference_point_px)
        if self.drift_budget > self.drift_budget_threshold:
            self.is_stale = True

    def get_corrected_homography(self) -> np.ndarray:
        """Returns the CURRENT best-estimate pixel(now)->meter homography,
        adjusting the anchor for camera motion accumulated since it was
        established: `anchor_homography @ cumulative_transform^-1` (the
        cumulative transform maps anchor-time pixels to current-time
        pixels, so its inverse maps current-time pixels back to what they
        would have been at anchor time, where `anchor_homography` is valid).

        Callers MUST also check `is_stale` before trusting this -- a
        `False` `get_corrected_homography()` result does not exist; this
        method always returns its best current estimate regardless of
        staleness, exactly like Milestone 32's stale-velocity fallback
        still returns `[0,0]` rather than raising.
        """
        return self.anchor_homography @ np.linalg.inv(self.cumulative_transform)

    def reanchor(self, new_anchor_homography: np.ndarray) -> None:
        """The ONLY way `is_stale` is ever cleared in this milestone.
        Requires an EXTERNALLY supplied fresh homography -- this class has
        no means of producing one itself (see class docstring)."""
        self.anchor_homography = new_anchor_homography
        self.cumulative_transform = np.eye(3)
        self.drift_budget = 0.0
        self.is_stale = False
