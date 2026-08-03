"""Milestone 27 validation: pitch calibration/homography, tested for
GENERALIZATION to held-out points under GENUINE perspective distortion --
not self-consistency on the calibration points themselves, which a trivial
(even buggy) implementation could pass.

The synthetic "ground truth" homography (`H_true`) below is built from an
actual pinhole-camera projection (position, look-at target, intrinsics) of
the 100x68m pitch plane -- NOT a hand-picked/arbitrary matrix. This
guarantees genuine perspective properties a flat top-down affine scale
could never produce: parallel pitch lines converge toward a vanishing
point, and the near touchline projects to a visibly WIDER pixel span than
the far touchline (foreshortening). `test_camera_setup_has_genuine_perspective_distortion`
below asserts this property directly, so a future accidental simplification
of the synthetic setup (e.g. back to a flat scale) would itself be caught.
"""

import math

import numpy as np

from production.src.cv.calibration import compute_homography, transform_points

# Ground-truth pitch corners (ADR-002's 100x68m space) -- WELL-CONDITIONED,
# non-collinear points. These, and ONLY these, are passed to
# compute_homography.
PITCH_CORNERS_METERS = [
    (0.0, 0.0),
    (100.0, 0.0),
    (100.0, 68.0),
    (0.0, 68.0),
]

# Held-out landmarks -- NEVER passed to compute_homography. Used only to
# validate that the recovered homography generalizes beyond the exact
# points it was fit on, which a same-point round-trip check cannot prove.
HELD_OUT_LANDMARKS_METERS = {
    "center_spot": (50.0, 34.0),
    "penalty_spot_near_end": (10.0, 34.0),
    "penalty_spot_far_end": (90.0, 34.0),
}

# A location "beyond the touchline" (Y < 0) -- simulating a crowd/
# advertising-board pixel that a real, partial broadcast camera view would
# routinely include. Used for the out-of-bounds direction of Step 2.6's
# check.
OFF_PITCH_LANDMARK_METERS = (50.0, -10.0)

HOLDOUT_TOLERANCE_METERS = 0.05  # tight: this synthetic setup has ZERO noise, so
# recovery should be near machine-precision exact. Kept much tighter than the
# 0.5m the milestone brief suggests as a reference point -- that more generous
# figure anticipates a FUTURE phase with noisy, automatically-detected
# keypoints (not yet implemented), not this exact synthetic case.


def _build_synthetic_broadcast_camera_homography() -> np.ndarray:
    """Ground-truth H_true: pixel = H_true @ [X, Y, 1] (homogeneous), for a
    pinhole camera positioned behind/outside one touchline, elevated, and
    aimed at the pitch center -- a plausible elevated "main broadcast
    camera" setup, not a flat top-down scale.
    """
    camera_pos = np.array([50.0, -55.0, 30.0])  # 55m outside the Y=0 touchline, 30m up
    target = np.array([50.0, 34.0, 0.0])  # aimed at the pitch center
    world_up = np.array([0.0, 0.0, 1.0])

    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)  # completes a right-handed (right, down, forward) basis

    rotation = np.stack([right, down, forward], axis=0)  # world -> camera rotation
    translation = -rotation @ camera_pos

    focal_length = 500.0
    principal_point = (960.0, 540.0)  # 1920x1080 image canvas
    intrinsics = np.array(
        [
            [focal_length, 0.0, principal_point[0]],
            [0.0, focal_length, principal_point[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    # On the Z=0 (pitch) plane, the rotation matrix's 3rd (Z) column
    # contributes nothing -- dropping it collapses the full 3D projection
    # into a direct 3x3 homography from (X, Y, 1) pitch-meters to pixel
    # homogeneous coordinates.
    extrinsics_on_plane = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    return intrinsics @ extrinsics_on_plane


def _project_with_homography(matrix: np.ndarray, meter_point: tuple[float, float]) -> tuple[float, float]:
    homogeneous = matrix @ np.array([meter_point[0], meter_point[1], 1.0])
    assert homogeneous[2] > 0, (
        f"point {meter_point} projects behind the camera (w={homogeneous[2]:.3f}) -- invalid "
        "synthetic camera setup"
    )
    return homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]


def test_camera_setup_has_genuine_perspective_distortion():
    """Confirms the synthetic ground-truth homography is genuinely oblique
    BEFORE using it to validate anything else -- a flat top-down affine
    scale would make the whole test suite trivially easy to pass even with
    a broken homography implementation.
    """
    H_true = _build_synthetic_broadcast_camera_homography()

    near_left = _project_with_homography(H_true, (0.0, 0.0))
    near_right = _project_with_homography(H_true, (100.0, 0.0))
    far_left = _project_with_homography(H_true, (0.0, 68.0))
    far_right = _project_with_homography(H_true, (100.0, 68.0))

    near_touchline_pixel_span = abs(near_right[0] - near_left[0])
    far_touchline_pixel_span = abs(far_right[0] - far_left[0])

    print(f"\nNear touchline (Y=0) pixel span: {near_touchline_pixel_span:.1f}px")
    print(f"Far touchline (Y=68) pixel span: {far_touchline_pixel_span:.1f}px")

    # Genuine foreshortening: the near touchline must appear MEANINGFULLY
    # wider in pixels than the far touchline (a flat top-down scale would
    # make these equal).
    assert near_touchline_pixel_span > 1.5 * far_touchline_pixel_span, (
        "synthetic camera setup does not show meaningful perspective foreshortening -- "
        f"near span {near_touchline_pixel_span:.1f}px vs far span {far_touchline_pixel_span:.1f}px"
    )


def test_homography_recovers_holdout_points_under_perspective_distortion():
    """The critical check: calibrate using ONLY the 4 pitch corners, then
    verify the recovered homography generalizes to points it was NEVER fit
    on (center spot, two penalty-line spots).
    """
    H_true = _build_synthetic_broadcast_camera_homography()

    calibration_pixels = [_project_with_homography(H_true, pt) for pt in PITCH_CORNERS_METERS]
    print("\nCalibration correspondences (4 corners only):")
    for meters, pixels in zip(PITCH_CORNERS_METERS, calibration_pixels):
        print(f"  meters={meters} -> pixels=({pixels[0]:.1f}, {pixels[1]:.1f})")

    H_est = compute_homography(calibration_pixels, PITCH_CORNERS_METERS, method=0)
    assert H_est is not None, "compute_homography failed to find a solution"

    # Trivial sanity check (self-consistency on the calibration points
    # themselves) -- necessary but NOT sufficient; kept brief since the
    # real test is the held-out generalization check below.
    recovered_corners = transform_points(H_est, calibration_pixels)
    for true_pt, recovered_pt in zip(PITCH_CORNERS_METERS, recovered_corners):
        assert math.isclose(recovered_pt[0], true_pt[0], abs_tol=1e-2)
        assert math.isclose(recovered_pt[1], true_pt[1], abs_tol=1e-2)

    # THE critical check: held-out points, never used for calibration.
    print(f"\nGeneralization check (held-out points, tolerance={HOLDOUT_TOLERANCE_METERS}m):")
    for name, true_meters in HELD_OUT_LANDMARKS_METERS.items():
        holdout_pixel = _project_with_homography(H_true, true_meters)
        recovered_meters = transform_points(H_est, [holdout_pixel])[0]

        error_x = abs(recovered_meters[0] - true_meters[0])
        error_y = abs(recovered_meters[1] - true_meters[1])
        print(
            f"  {name}: true={true_meters} pixel=({holdout_pixel[0]:.1f}, {holdout_pixel[1]:.1f}) "
            f"recovered=({recovered_meters[0]:.4f}, {recovered_meters[1]:.4f}) "
            f"error=({error_x:.4f}, {error_y:.4f})"
        )

        assert error_x < HOLDOUT_TOLERANCE_METERS, (
            f"{name}: recovered x={recovered_meters[0]:.4f} vs true x={true_meters[0]} "
            f"(error {error_x:.4f}m exceeds {HOLDOUT_TOLERANCE_METERS}m)"
        )
        assert error_y < HOLDOUT_TOLERANCE_METERS, (
            f"{name}: recovered y={recovered_meters[1]:.4f} vs true y={true_meters[1]} "
            f"(error {error_y:.4f}m exceeds {HOLDOUT_TOLERANCE_METERS}m)"
        )


def test_interior_pixel_transforms_in_bounds():
    """A pixel genuinely corresponding to a pitch-interior location (the
    center spot, under H_true) must transform to IN-BOUNDS meter
    coordinates: 0<=x<=100, 0<=y<=68."""
    H_true = _build_synthetic_broadcast_camera_homography()
    calibration_pixels = [_project_with_homography(H_true, pt) for pt in PITCH_CORNERS_METERS]
    H_est = compute_homography(calibration_pixels, PITCH_CORNERS_METERS, method=0)

    interior_pixel = _project_with_homography(H_true, (50.0, 34.0))
    recovered = transform_points(H_est, [interior_pixel])[0]

    assert 0.0 <= recovered[0] <= 100.0
    assert 0.0 <= recovered[1] <= 68.0


def test_offpitch_pixel_transforms_out_of_bounds():
    """The converse of the previous test, EXPLICITLY: a pixel corresponding
    to a location beyond the touchline (simulating a crowd/advertising-
    board pixel, which any partial real broadcast frame routinely
    includes) is EXPECTED to transform to an OUT-OF-BOUNDS meter
    coordinate. This is CORRECT behavior, not a bug -- the homography
    faithfully extrapolates the ground plane beyond the pitch markings; it
    is the caller's job (not this module's) to decide what to do with an
    out-of-bounds result (e.g. discard it as "not a player on the pitch").
    """
    H_true = _build_synthetic_broadcast_camera_homography()
    calibration_pixels = [_project_with_homography(H_true, pt) for pt in PITCH_CORNERS_METERS]
    H_est = compute_homography(calibration_pixels, PITCH_CORNERS_METERS, method=0)

    offpitch_pixel = _project_with_homography(H_true, OFF_PITCH_LANDMARK_METERS)
    recovered = transform_points(H_est, [offpitch_pixel])[0]

    print(f"\nOff-pitch point {OFF_PITCH_LANDMARK_METERS} -> pixel=({offpitch_pixel[0]:.1f}, "
          f"{offpitch_pixel[1]:.1f}) -> recovered meters=({recovered[0]:.4f}, {recovered[1]:.4f})")

    assert recovered[1] < 0.0, (
        f"expected an out-of-bounds (y < 0) recovered coordinate for a point beyond the "
        f"touchline, got y={recovered[1]:.4f} -- the homography should faithfully extrapolate, "
        "not clip, points outside the pitch markings"
    )
    assert math.isclose(recovered[1], OFF_PITCH_LANDMARK_METERS[1], abs_tol=HOLDOUT_TOLERANCE_METERS)
