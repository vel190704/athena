"""Milestone 37 validation: frame-to-frame camera-motion estimation, drift
composition, and flicker-aware player masking.

Reuses Milestone 27's pinhole-camera synthetic projection approach
(`test_calibration.py`'s `_build_synthetic_broadcast_camera_homography`
pattern), extended across a full sequence of RENDERED synthetic frames --
not just point correspondences -- because `estimate_camera_motion` operates
on real pixel arrays via `cv2.goodFeaturesToTrack`/`cv2.calcOpticalFlowPyrLK`,
and a test that bypassed image rendering to hand it points directly would
not exercise the actual optical-flow/masking code path at all.

HONEST SCOPE NOTE (same discipline as Milestone 28's synthetic-swatches
caveat): the synthetic "ground plane" below is per-pixel random noise plus
a few drawn pitch lines -- richly-textured enough for `goodFeaturesToTrack`
to find genuine corners (that is all this test needs), but NOT a claim
about how well this approach will perform on real broadcast grass texture,
motion blur, or lighting changes. That remains untested, exactly like every
other CV component's gap between synthetic and real validation documented
in `CV_PIPELINE_FINDINGS.md`.
"""

import math

import cv2
import numpy as np
import pytest

from production.src.cv.calibration import transform_points
from production.src.cv.camera_motion import (
    BASE_MASK_CONFIDENCE_THRESHOLD,
    BBOX_MASK_PADDING_FRACTION,
    HYSTERESIS_CONFIDENCE_MIN,
    _reference_point_displacement_px,
    HYSTERESIS_IOU_THRESHOLD,
    MIN_BACKGROUND_FEATURE_POINTS,
    CameraMotionTracker,
    _bbox_iou,
    _pad_bbox,
    _select_boxes_to_mask,
    estimate_camera_motion,
)

FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
TEXTURE_SCALE_PX_PER_METER = 20
TEXTURE_MARGIN_METERS = 30  # generous margin so a pan never runs off the rendered texture


# ============================================================================
# Synthetic scene construction (reuses Milestone 27's pinhole-camera
# approach, extended to a full frame sequence with a smooth pan)
# ============================================================================


def _build_ground_plane_texture(seed: int = 42, noise_range: tuple[int, int] = (60, 200)):
    """A richly-textured synthetic 'ground plane' image (per-pixel random
    noise + pitch line markings) in a fixed meter-scaled pixel space.
    Random noise is a simple, reproducible way to guarantee plenty of
    genuine goodFeaturesToTrack corners -- a flat/blank synthetic image
    would provide none, same spirit as Milestone 28's solid-swatch fixtures
    providing exactly the property their test needs and nothing more.

    `noise_range` controls local contrast: the default is deliberately
    rich (for the background-only drift-curve test). The masking
    adversarial test below uses a NARROWER range -- real grass has far
    less local contrast than a synthetic solid-color player rectangle with
    a sharp border, and a narrower range makes that same realistic gap
    reproduce here, so an UNMASKED player's sharp edges genuinely dominate
    goodFeaturesToTrack's corner selection instead of competing evenly
    against equally-strong background corners.
    """
    width_m = 100 + 2 * TEXTURE_MARGIN_METERS
    height_m = 68 + 2 * TEXTURE_MARGIN_METERS
    width_px = int(width_m * TEXTURE_SCALE_PX_PER_METER)
    height_px = int(height_m * TEXTURE_SCALE_PX_PER_METER)

    rng = np.random.default_rng(seed)
    texture = rng.integers(noise_range[0], noise_range[1], size=(height_px, width_px, 3), dtype=np.uint8)

    def to_px(x_m, y_m):
        return (
            int((x_m + TEXTURE_MARGIN_METERS) * TEXTURE_SCALE_PX_PER_METER),
            int((y_m + TEXTURE_MARGIN_METERS) * TEXTURE_SCALE_PX_PER_METER),
        )

    cv2.rectangle(texture, to_px(0, 0), to_px(100, 68), (255, 255, 255), 2)
    cv2.line(texture, to_px(50, 0), to_px(50, 68), (255, 255, 255), 2)
    cv2.circle(texture, to_px(50, 34), 9 * TEXTURE_SCALE_PX_PER_METER, (255, 255, 255), 2)

    return texture, width_m, height_m


def _texture_pixel_to_meters_homography() -> np.ndarray:
    """Ground-plane TEXTURE pixel coords -> this project's 100x68m pitch
    space (ADR-002), accounting for TEXTURE_MARGIN_METERS."""
    scale = 1.0 / TEXTURE_SCALE_PX_PER_METER
    return np.array(
        [
            [scale, 0.0, -TEXTURE_MARGIN_METERS],
            [0.0, scale, -TEXTURE_MARGIN_METERS],
            [0.0, 0.0, 1.0],
        ]
    )


def _camera_homography_at(t: int, pan_rate_meters_per_frame: float) -> np.ndarray:
    """Milestone 27's pinhole-camera construction (`test_calibration.py`'s
    `_build_synthetic_broadcast_camera_homography`), PANNED smoothly across
    frames: camera position fixed, aim target's x-coordinate drifts
    linearly with frame index -- a plausible slow broadcast pan following
    play, not an arbitrary hand-picked matrix. Returns H_true(t): meters ->
    camera-pixel homogeneous coords (same convention as Milestone 27's own
    test, NOT the pixel->meter convention `calibration.compute_homography`
    produces -- see this file's module docstring / `CameraMotionTracker`'s
    own docstring for why the two directions matter here).
    """
    camera_pos = np.array([50.0, -55.0, 30.0])
    target = np.array([50.0 + pan_rate_meters_per_frame * t, 34.0, 0.0])
    world_up = np.array([0.0, 0.0, 1.0])

    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ camera_pos

    focal_length = 500.0
    principal_point = (FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0)
    intrinsics = np.array(
        [
            [focal_length, 0.0, principal_point[0]],
            [0.0, focal_length, principal_point[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics_on_plane = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    return intrinsics @ extrinsics_on_plane


def _project_with_homography(matrix: np.ndarray, meter_point) -> tuple[float, float]:
    homogeneous = matrix @ np.array([meter_point[0], meter_point[1], 1.0])
    assert homogeneous[2] > 0, f"point {meter_point} projects behind the camera"
    return homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]


def _render_frame(t: int, pan_rate_meters_per_frame: float, ground_texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renders frame `t` of the pan sequence by warping the ground-plane
    texture through the composition (texture-pixel -> meters -> camera
    frame `t`). Returns `(frame, H_true_t)`.
    """
    H_true_t = _camera_homography_at(t, pan_rate_meters_per_frame)
    H_texture_to_meters = _texture_pixel_to_meters_homography()
    H_full = H_true_t @ H_texture_to_meters
    frame = cv2.warpPerspective(ground_texture, H_full, (FRAME_WIDTH, FRAME_HEIGHT))
    return frame, H_true_t


def test_camera_pan_setup_has_genuine_perspective_distortion():
    """Sanity check, mirroring Milestone 27's own equivalent test: confirm
    the synthetic camera setup this file reuses is genuinely oblique BEFORE
    using it to validate anything else."""
    H_true_0 = _camera_homography_at(0, pan_rate_meters_per_frame=0.05)
    near_left = _project_with_homography(H_true_0, (0.0, 0.0))
    near_right = _project_with_homography(H_true_0, (100.0, 0.0))
    far_left = _project_with_homography(H_true_0, (0.0, 68.0))
    far_right = _project_with_homography(H_true_0, (100.0, 68.0))

    near_span = abs(near_right[0] - near_left[0])
    far_span = abs(far_right[0] - far_left[0])
    print(f"\nNear touchline pixel span: {near_span:.1f}px, far touchline: {far_span:.1f}px")
    assert near_span > 1.5 * far_span


# ============================================================================
# Step 3.3: the measured drift-vs-frame-count curve (background-only, no
# players -- isolates camera-motion estimation/composition from masking)
# ============================================================================

PAN_RATE_METERS_PER_FRAME = 0.03
NUM_FRAMES = 200
DRIFT_REPORT_INTERVAL = 10
POSITION_ERROR_THRESHOLD_METERS = 1.0  # the "meaningful" drift level Step 3.3 asks to locate, not assume


def test_drift_vs_frame_count_curve():
    """THE central measurement this milestone exists to produce: composing
    frame-to-frame camera-motion estimates over a real (rendered, tracked
    via actual optical flow) synthetic pan, how does positional error at a
    known ground-truth point (the pitch center) actually grow with frame
    count? Reports the real curve; does not assume any particular shape.
    """
    ground_texture, _width_m, _height_m = _build_ground_plane_texture()

    frames = []
    H_true_by_frame = []
    for t in range(NUM_FRAMES):
        frame, H_true_t = _render_frame(t, PAN_RATE_METERS_PER_FRAME, ground_texture)
        frames.append(frame)
        H_true_by_frame.append(H_true_t)

    # Milestone-27-style manual anchor at frame 0: pixel(anchor)->meters,
    # i.e. the INVERSE of H_true(0)'s meters->pixels convention -- matches
    # what calibration.compute_homography would actually produce if
    # calibrated against frame 0's correspondences.
    anchor_homography = np.linalg.inv(H_true_by_frame[0])
    tracker = CameraMotionTracker(anchor_homography, frame_shape=(FRAME_HEIGHT, FRAME_WIDTH))

    print(f"\n=== Drift-vs-frame-count curve (pan_rate={PAN_RATE_METERS_PER_FRAME}m/frame, {NUM_FRAMES} frames) ===")
    print(f"{'frame':>6} {'drift_budget(px)':>18} {'center_error(m)':>16} {'is_stale':>9}")

    error_at_threshold_frame = None
    drift_curve = []
    for t in range(1, NUM_FRAMES):
        H_frame = estimate_camera_motion(frames[t - 1], frames[t], [], [])
        assert H_frame is not None, f"frame-to-frame estimate failed at t={t} on a pure-background pan"
        tracker.update(H_frame)

        if t % DRIFT_REPORT_INTERVAL == 0 or t == NUM_FRAMES - 1:
            H_corrected = tracker.get_corrected_homography()
            true_pixel_now = _project_with_homography(H_true_by_frame[t], (50.0, 34.0))
            recovered_meters = transform_points(H_corrected, [true_pixel_now])[0]
            error_m = math.hypot(recovered_meters[0] - 50.0, recovered_meters[1] - 34.0)
            drift_curve.append((t, tracker.drift_budget, error_m))
            print(f"{t:6d} {tracker.drift_budget:18.2f} {error_m:16.4f} {tracker.is_stale!s:>9}")

            if error_m > POSITION_ERROR_THRESHOLD_METERS and error_at_threshold_frame is None:
                error_at_threshold_frame = t

    if error_at_threshold_frame is not None:
        print(
            f"\nMEASURED: composed drift first exceeds {POSITION_ERROR_THRESHOLD_METERS}m of "
            f"positional error at the pitch center by frame {error_at_threshold_frame} "
            f"(checked every {DRIFT_REPORT_INTERVAL} frames)."
        )
    else:
        print(
            f"\nMEASURED: composed drift never exceeded {POSITION_ERROR_THRESHOLD_METERS}m within "
            f"{NUM_FRAMES} frames at this pan rate ({PAN_RATE_METERS_PER_FRAME}m/frame)."
        )

    # The curve must be genuinely increasing overall (composition drift is
    # expected to accumulate, not stay flat or shrink) -- a real property
    # check, not a specific-number assertion (the specific numbers are
    # reported above for a human to read, not hard-asserted, since they are
    # this test's actual FINDING, not a pre-known target).
    first_error = drift_curve[0][2]
    last_error = drift_curve[-1][2]
    assert last_error > first_error, (
        f"expected drift to accumulate over {NUM_FRAMES} frames (error should grow), but "
        f"first-checkpoint error ({first_error:.4f}m) >= last-checkpoint error ({last_error:.4f}m)"
    )


# ============================================================================
# Step 3.4: masking adversarial test -- flickering-confidence player
# ============================================================================

# Multiple simultaneously-moving, independently-flickering players --
# reconstructing the ACTUAL measured phenomenon (23% of ALL person
# detections near the 0.5 threshold, not one isolated player occasionally
# dipping) more faithfully than a single-player test would. A single
# small, isolated contaminant is easily absorbed by RANSAC's own outlier
# rejection regardless of masking quality -- several simultaneously-
# unmasked players is what actually stresses the DIFFERENCE masking
# quality makes on top of RANSAC.
NUM_SYNTHETIC_PLAYERS = 6
PLAYER_BOX_SIZE_PX = (45, 130)
PLAYER_STARTS_PX = [
    (300.0, 250.0), (700.0, 550.0), (1100.0, 300.0),
    (500.0, 750.0), (1400.0, 600.0), (900.0, 850.0),
]
PLAYER_VELOCITIES_PX_PER_FRAME = [
    (14.0, 3.0), (-10.0, 6.0), (8.0, -9.0),
    (-12.0, -4.0), (6.0, 10.0), (-8.0, 7.0),
]  # all independent of, and none matching, the background pan's own apparent motion
NUM_MASKING_TEST_FRAMES = 60


def _player_bbox_at(player_index: int, t: int) -> tuple[float, float, float, float]:
    x0, y0 = PLAYER_STARTS_PX[player_index]
    vx, vy = PLAYER_VELOCITIES_PX_PER_FRAME[player_index]
    cx, cy = x0 + vx * t, y0 + vy * t
    w, h = PLAYER_BOX_SIZE_PX
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _draw_player(frame: np.ndarray, bbox) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)  # solid, high-contrast vs. noise background
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # sharp border -> strong (contaminating) corners


def _flicker_confidence(player_index: int, t: int) -> float:
    """Confident (0.6) for 3 frames, then a single-frame dip to 0.4 -- a
    concrete, reproducible reconstruction of the measured real-clip
    phenomenon (borderline detections flipping in and out around 0.5), not
    a hypothetical pattern. Each player's dip lands on a different phase
    offset so they don't all dip (or all stay confident) on the same
    frame -- matching the real finding that borderline detections are
    spread across the roster, not synchronized.
    """
    return 0.4 if (t + player_index) % 4 == 3 else 0.6


def _naive_masked_estimate_multi(frame_prev: np.ndarray, frame_curr: np.ndarray, bboxes_prev, confidences_prev) -> np.ndarray | None:
    """Independent, test-local reimplementation of a NAIVE masking
    approach -- current-frame confidence only, `>= 0.5` cutoff, NO padding,
    NO hysteresis -- for comparison purposes only. Mirrors
    test_team_classifier.py's own pattern of computing a naive baseline
    independently in the test file rather than keeping a naive path in
    production code.
    """
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

    mask = np.full(gray_prev.shape, 255, dtype=np.uint8)
    for bbox, confidence in zip(bboxes_prev, confidences_prev):
        if confidence >= 0.5:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            mask[y1:y2, x1:x2] = 0
        # confidence < 0.5: NOT masked at all under the naive approach,
        # regardless of how recently it WAS confidently detected.

    corners = cv2.goodFeaturesToTrack(gray_prev, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask)
    if corners is None or len(corners) < 20:
        return None
    next_pts, status, _err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, corners, None, winSize=(21, 21), maxLevel=3)
    status = status.reshape(-1)
    good_prev = corners.reshape(-1, 2)[status == 1]
    good_curr = next_pts.reshape(-1, 2)[status == 1]
    if len(good_prev) < 20:
        return None
    homography, inlier_mask = cv2.findHomography(good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if homography is None or (inlier_mask is not None and int(inlier_mask.sum()) < 20):
        return None
    return homography


def test_hysteresis_and_padding_reduce_contamination_vs_naive_masking():
    """THE core Step 3.4 adversarial comparison: several synthetic players
    move independently of the background pan (and of each other), each with
    DELIBERATELY FLICKERING detection confidence (0.6 -> 0.4 -> 0.6 -> ...,
    staggered so not all dip simultaneously), reconstructing the measured
    real phenomenon (23% of ALL detections near the 0.5 threshold) rather
    than one isolated occasionally-dipping player -- a single small
    contaminant is easily absorbed by RANSAC's own outlier rejection
    regardless of masking quality, so this test needs enough simultaneous
    contamination to actually stress the DIFFERENCE masking quality makes
    on top of RANSAC. Compares the flicker-aware masking (hysteresis +
    padding, Steps 1.3a/1.3b) against a naive current-frame-confidence-only,
    no-padding baseline, over the SAME rendered frames -- reporting the
    actual numeric difference in resulting camera-motion-estimate error,
    the same adversarial-comparison pattern as Milestone 28's
    masking-effect test and Milestone 29's distractor-shape test.
    """
    # Deliberately LOW-contrast background for this specific test (see
    # _build_ground_plane_texture's docstring) -- makes the players' sharp,
    # high-contrast edges dominate goodFeaturesToTrack's corner selection
    # when unmasked, the same way real grass's low local contrast (versus
    # a player's edges) would.
    ground_texture, _w, _h = _build_ground_plane_texture(seed=7, noise_range=(110, 146))

    frames = []
    H_true_by_frame = []
    for t in range(NUM_MASKING_TEST_FRAMES):
        frame, H_true_t = _render_frame(t, PAN_RATE_METERS_PER_FRAME, ground_texture)
        for player_index in range(NUM_SYNTHETIC_PLAYERS):
            _draw_player(frame, _player_bbox_at(player_index, t))
        frames.append(frame)
        H_true_by_frame.append(H_true_t)

    # confidences[player_index][t], bboxes[player_index][t]
    confidences = [[_flicker_confidence(p, t) for t in range(NUM_MASKING_TEST_FRAMES)] for p in range(NUM_SYNTHETIC_PLAYERS)]
    bboxes = [[_player_bbox_at(p, t) for t in range(NUM_MASKING_TEST_FRAMES)] for p in range(NUM_SYNTHETIC_PLAYERS)]

    reference_point_px = (FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0)

    # PER-FRAME estimate error against ground truth, NOT cumulative composed
    # error: composing many frames together mixes the masking-quality signal
    # this test wants to isolate with unrelated compounding-drift noise from
    # every other frame (see test_drift_vs_frame_count_curve above -- 60
    # frames of pan ALONE already accumulates >1m of composed error, which
    # would otherwise swamp the comparison). Each frame's homography is
    # instead compared directly against that SAME frame's true
    # prev->curr motion.
    flicker_errors_px, naive_errors_px = [], []
    flicker_errors_on_dip_frames, naive_errors_on_dip_frames = [], []
    naive_none_count = flicker_none_count = 0

    for t in range(1, NUM_MASKING_TEST_FRAMES):
        bboxes_prev = [bboxes[p][t - 1] for p in range(NUM_SYNTHETIC_PLAYERS)]
        confidences_prev = [confidences[p][t - 1] for p in range(NUM_SYNTHETIC_PLAYERS)]
        any_dip_this_frame = any(
            HYSTERESIS_CONFIDENCE_MIN <= c < BASE_MASK_CONFIDENCE_THRESHOLD for c in confidences_prev
        )

        tracked_in_prev_prev = None
        if t >= 2:
            tracked_in_prev_prev = [
                bboxes[p][t - 2] for p in range(NUM_SYNTHETIC_PLAYERS)
                if confidences[p][t - 2] >= BASE_MASK_CONFIDENCE_THRESHOLD
            ]

        H_true_frame_to_frame = H_true_by_frame[t] @ np.linalg.inv(H_true_by_frame[t - 1])

        H_flicker = estimate_camera_motion(frames[t - 1], frames[t], bboxes_prev, confidences_prev, tracked_in_prev_prev)
        if H_flicker is None:
            flicker_none_count += 1
        else:
            diff = H_flicker @ np.linalg.inv(H_true_frame_to_frame)
            err = _reference_point_displacement_px(diff, reference_point_px)
            flicker_errors_px.append(err)
            if any_dip_this_frame:
                flicker_errors_on_dip_frames.append(err)

        H_naive = _naive_masked_estimate_multi(frames[t - 1], frames[t], bboxes_prev, confidences_prev)
        if H_naive is None:
            naive_none_count += 1
        else:
            diff = H_naive @ np.linalg.inv(H_true_frame_to_frame)
            err = _reference_point_displacement_px(diff, reference_point_px)
            naive_errors_px.append(err)
            if any_dip_this_frame:
                naive_errors_on_dip_frames.append(err)

    mean_flicker = sum(flicker_errors_px) / len(flicker_errors_px)
    mean_naive = sum(naive_errors_px) / len(naive_errors_px)
    mean_flicker_dip = sum(flicker_errors_on_dip_frames) / len(flicker_errors_on_dip_frames)
    mean_naive_dip = sum(naive_errors_on_dip_frames) / len(naive_errors_on_dip_frames)

    print(f"\n=== Per-frame masking comparison over {NUM_MASKING_TEST_FRAMES} frames, "
          f"{NUM_SYNTHETIC_PLAYERS} independently-flickering players ===")
    print(f"All frames    -- flicker-aware mean error: {mean_flicker:.3f}px, naive mean error: {mean_naive:.3f}px "
          f"({mean_naive / max(mean_flicker, 1e-9):.2f}x)")
    print(f"Dip frames only ({len(flicker_errors_on_dip_frames)}/{NUM_MASKING_TEST_FRAMES - 1} frames with >=1 "
          f"player in the 0.35-0.5 confidence band) -- flicker-aware: {mean_flicker_dip:.3f}px, "
          f"naive: {mean_naive_dip:.3f}px ({mean_naive_dip / max(mean_flicker_dip, 1e-9):.2f}x)")
    print(f"estimate_camera_motion returned None {flicker_none_count} times; naive returned None {naive_none_count} times")

    assert mean_naive_dip > mean_flicker_dip, (
        f"expected naive masking to show LARGER per-frame estimate error than flicker-aware masking, "
        f"specifically on frames where a player's confidence dipped into the hysteresis band -- got "
        f"naive={mean_naive_dip:.3f}px vs flicker-aware={mean_flicker_dip:.3f}px"
    )


# ============================================================================
# Step 3.5: is_stale fires at the measured threshold, not an assumed one
# ============================================================================


def test_is_stale_fires_once_drift_budget_exceeds_threshold():
    """Confirms `is_stale` actually fires, using the SAME pan rate/setup as
    the measured drift curve above, over enough frames that
    DEFAULT_DRIFT_BUDGET_THRESHOLD_PX (itself set from that measured curve
    -- see camera_motion.py's own comment) is exceeded."""
    ground_texture, _w, _h = _build_ground_plane_texture()
    frames_and_truth = [_render_frame(t, PAN_RATE_METERS_PER_FRAME, ground_texture) for t in range(NUM_FRAMES)]
    frames = [f for f, _ in frames_and_truth]
    H_true_by_frame = [h for _, h in frames_and_truth]

    anchor_homography = np.linalg.inv(H_true_by_frame[0])
    tracker = CameraMotionTracker(anchor_homography, frame_shape=(FRAME_HEIGHT, FRAME_WIDTH))

    stale_at_frame = None
    for t in range(1, NUM_FRAMES):
        H_frame = estimate_camera_motion(frames[t - 1], frames[t], [], [])
        tracker.update(H_frame)
        if tracker.is_stale and stale_at_frame is None:
            stale_at_frame = t

    print(f"\nis_stale first fired at frame {stale_at_frame} (drift_budget={tracker.drift_budget:.2f}px, "
          f"threshold={tracker.drift_budget_threshold}px)")
    assert stale_at_frame is not None, (
        f"is_stale never fired within {NUM_FRAMES} frames -- either the threshold is too high for "
        "this pan rate, or drift genuinely never exceeded it here"
    )


def test_reanchor_is_the_only_way_to_clear_staleness():
    """Direct unit test of Step 2.3's stated limitation: once `is_stale` is
    True, nothing except `reanchor()` (which requires an EXTERNALLY
    supplied homography) clears it."""
    tracker = CameraMotionTracker(np.eye(3), frame_shape=(FRAME_HEIGHT, FRAME_WIDTH), drift_budget_threshold=1.0)
    tracker.update(None)  # Step 2.4: a missing estimate immediately flags staleness
    assert tracker.is_stale is True

    # No amount of further legitimate updates clears it on their own.
    identity_motion = np.eye(3)
    tracker.update(identity_motion)
    assert tracker.is_stale is True, "is_stale must not clear itself without an explicit reanchor()"

    tracker.reanchor(np.eye(3) * 2)  # an arbitrary "freshly supplied" homography
    assert tracker.is_stale is False
    assert tracker.drift_budget == 0.0
    assert np.array_equal(tracker.cumulative_transform, np.eye(3))


# ============================================================================
# Direct unit tests of the masking/hysteresis logic (fast, no image
# rendering needed)
# ============================================================================


def test_base_masking_always_excludes_confident_boxes():
    boxes = [(0, 0, 10, 10), (100, 100, 120, 130)]
    confidences = [0.9, 0.55]
    selected = _select_boxes_to_mask(boxes, confidences, tracked_in_prev_prev=None)
    assert set(selected) == set(boxes)


def test_hysteresis_masks_a_confidence_dip_near_a_prior_confident_box():
    box_now = (100.0, 100.0, 130.0, 190.0)
    box_two_frames_ago = (102.0, 98.0, 132.0, 188.0)  # nearly identical -- high IoU
    selected = _select_boxes_to_mask([box_now], [0.42], tracked_in_prev_prev=[box_two_frames_ago])
    assert box_now in selected


def test_hysteresis_does_not_mask_a_dip_with_no_supporting_prior_box():
    box_now = (100.0, 100.0, 130.0, 190.0)
    unrelated_prior = (900.0, 900.0, 930.0, 990.0)
    selected = _select_boxes_to_mask([box_now], [0.42], tracked_in_prev_prev=[unrelated_prior])
    assert box_now not in selected


def test_low_confidence_below_hysteresis_floor_is_never_masked():
    box_now = (100.0, 100.0, 130.0, 190.0)
    selected = _select_boxes_to_mask([box_now], [0.10], tracked_in_prev_prev=[box_now])
    assert box_now not in selected


def test_bbox_padding_expands_in_both_dimensions_and_clamps_to_frame():
    bbox = (100.0, 100.0, 140.0, 180.0)  # 40x80
    padded = _pad_bbox(bbox, BBOX_MASK_PADDING_FRACTION, frame_shape=(1080, 1920))
    x1, y1, x2, y2 = padded
    assert x1 < 100.0 and x2 > 140.0
    assert y1 < 100.0 and y2 > 180.0

    # Clamping: a box touching the frame edge must not pad past it.
    edge_bbox = (0.0, 0.0, 20.0, 20.0)
    padded_edge = _pad_bbox(edge_bbox, BBOX_MASK_PADDING_FRACTION, frame_shape=(1080, 1920))
    assert padded_edge[0] == 0.0
    assert padded_edge[1] == 0.0


def test_estimate_camera_motion_returns_none_when_too_few_background_points_remain():
    """Step 1.4: if masking excludes almost the entire frame, there aren't
    enough background points left to fit a reliable homography -- must
    return None, not force a fit."""
    ground_texture, _w, _h = _build_ground_plane_texture()
    frame_prev, _h1 = _render_frame(0, PAN_RATE_METERS_PER_FRAME, ground_texture)
    frame_curr, _h2 = _render_frame(1, PAN_RATE_METERS_PER_FRAME, ground_texture)

    # A single "player" box covering almost the entire frame at high
    # confidence -- leaves essentially no background to sample from.
    huge_box = [(0.0, 0.0, float(FRAME_WIDTH), float(FRAME_HEIGHT))]
    result = estimate_camera_motion(frame_prev, frame_curr, huge_box, [0.9])
    assert result is None
