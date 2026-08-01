"""Milestone 30 validation: the CV-to-physics adapter layer.

Reuses the REAL Milestone 27 pinhole-camera-derived synthetic homography
(55m/30m elevation setup, ~2.04x near/far touchline foreshortening) --
deliberately NOT a toy uniform-scale matrix, since a pure affine scale
would make vector-transformation and point-then-difference velocity
methods produce IDENTICAL results, hiding exactly the bug this milestone
most needs to catch (Step 1.3).

TWO homographies matter here, and mixing them up is the easiest way to
break this test file: `H_true` (meters -> pixels) is the GROUND-TRUTH
camera model, used ONLY to generate synthetic pixel input from known meter
positions (`_pixel`). `H_calibrated` (pixels -> meters), derived via
`compute_homography` on the 4 pitch corners exactly like `test_calibration.py`
and exactly like real usage, is what actually gets passed to
`convert_frame_to_tensors` -- the adapter never sees `H_true` directly,
same as a real deployment never would.
"""

import numpy as np
import torch

from production.src.cv.adapter import convert_frame_to_tensors
from production.src.cv.calibration import compute_homography, transform_points
from production.tests.test_calibration import (
    PITCH_CORNERS_METERS,
    _build_synthetic_broadcast_camera_homography,
    _project_with_homography,
)

FPS = 25.0


def _pixel(H_true, meters):
    """Projects a real meter-space point into pixel space using the
    GROUND-TRUTH camera homography -- used only to CONSTRUCT synthetic
    test input, never passed to the adapter itself."""
    u, v = _project_with_homography(H_true, meters)
    return [u, v]


def _calibrate(H_true):
    """Derives the CALIBRATED (pixels -> meters) homography the adapter
    actually operates on, via `compute_homography` on the 4 pitch corners
    -- exactly like `test_calibration.py` and exactly like real usage.
    Passing `H_true` itself to the adapter would apply the meters->pixels
    projection a second time to an already-pixel-space input, producing
    nonsense."""
    calibration_pixels = [_project_with_homography(H_true, pt) for pt in PITCH_CORNERS_METERS]
    return compute_homography(calibration_pixels, PITCH_CORNERS_METERS, method=0)


def _build_basic_scenario(H_true):
    """3 players (track_id 1, 2 -> team_A; track_id 3 -> team_B) across two
    consecutive frames, plus a ball positioned near team_B's player (3) --
    shared by several tests below. Returns pixel-space inputs (built via
    `H_true`) ready to pass to the adapter alongside `_calibrate(H_true)`.
    """
    prev_meters = {1: (29.0, 20.0), 2: (69.0, 50.0), 3: (49.0, 34.0)}
    curr_meters = {1: (30.0, 20.0), 2: (70.0, 50.0), 3: (50.0, 34.0)}
    ball_meters = (52.0, 34.0)  # closest to track_id 3 (team_B)

    tracks = [{"track_id": tid, "pos_pixel": _pixel(H_true, curr_meters[tid])} for tid in (1, 2, 3)]
    prev_positions_pixel = {tid: _pixel(H_true, prev_meters[tid]) for tid in (1, 2, 3)}
    ball_pixel = _pixel(H_true, ball_meters)
    team_mapping = {1: "team_A", 2: "team_A", 3: "team_B"}

    return tracks, prev_positions_pixel, ball_pixel, team_mapping


def test_output_keys_dtypes_and_bounds():
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, ball_pixel, team_mapping = _build_basic_scenario(H_true)

    result = convert_frame_to_tensors(tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel)

    assert result is not None
    for key in ("player_pos", "player_vel", "is_teammate", "ball_pos"):
        assert key in result, f"missing required key {key!r}"

    assert result["player_pos"].dtype == torch.float32
    assert result["player_vel"].dtype == torch.float32
    assert result["is_teammate"].dtype == torch.bool
    assert result["ball_pos"].dtype == torch.float32

    assert result["player_pos"].shape == (3, 2)
    assert result["player_vel"].shape == (3, 2)
    assert result["is_teammate"].shape == (3,)
    assert result["ball_pos"].shape == (2,)

    tolerance = 0.5
    positions = result["player_pos"].numpy()
    assert np.all(positions[:, 0] >= -tolerance) and np.all(positions[:, 0] <= 100.0 + tolerance)
    assert np.all(positions[:, 1] >= -tolerance) and np.all(positions[:, 1] <= 68.0 + tolerance)

    print(f"\nplayer_pos:\n{result['player_pos']}")
    print(f"player_vel:\n{result['player_vel']}")
    print(f"is_teammate: {result['is_teammate']}")
    print(f"ball_pos: {result['ball_pos']}")


def test_possession_based_is_teammate_favors_nearest_team_to_ball():
    """The ball is placed nearest to track_id 3 (team_B) -- is_teammate
    must come out True for team_B's player and False for BOTH team_A
    players, directly proving the possession heuristic (nearest player to
    the ball) drives the result, not a hardcoded team.
    """
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, ball_pixel, team_mapping = _build_basic_scenario(H_true)

    result = convert_frame_to_tensors(tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel)

    # tracks list order is [1 (team_A), 2 (team_A), 3 (team_B)] and none
    # are filtered in this scenario, so tensor index order matches.
    is_teammate = result["is_teammate"].tolist()
    assert is_teammate == [False, False, True], (
        f"expected [False, False, True] (team_B possesses), got {is_teammate}"
    )


def test_possession_flips_when_ball_moves_to_the_other_team():
    """Converse of the above, same roster: move the ball near team_A's
    players instead, and confirm is_teammate flips accordingly -- proving
    the heuristic actually recomputes per frame rather than caching a
    fixed answer.
    """
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, _old_ball_pixel, team_mapping = _build_basic_scenario(H_true)

    ball_pixel_near_team_a = _pixel(H_true, (30.0, 20.5))  # right next to track_id 1

    result = convert_frame_to_tensors(
        tracks, ball_pixel_near_team_a, team_mapping, H, FPS, prev_positions_pixel
    )

    is_teammate = result["is_teammate"].tolist()
    assert is_teammate == [True, True, False], (
        f"expected [True, True, False] (team_A possesses), got {is_teammate}"
    )


def test_velocity_correctness_vs_naive_vector_transform():
    """THE critical check: a player moves 2m along X near the FAR
    touchline (y=66), a region of significant projective distortion.
    Compares the adapter's actual output against BOTH the correct
    (point-then-difference) method and an explicitly-computed NAIVE
    method (transforming the raw pixel DISPLACEMENT VECTOR through the
    homography as if it were itself a point) -- the two methods must
    differ by a large margin under this real perspective distortion, and
    the adapter's output must match the CORRECT one.
    """
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)

    prev_meters = (40.0, 66.0)
    curr_meters = (42.0, 66.0)  # true movement: 2m along X, 0 along Y

    prev_pixel = _pixel(H_true, prev_meters)
    curr_pixel = _pixel(H_true, curr_meters)
    dt = 1.0 / FPS

    # Correct method: transform current and previous POINTS separately
    # (using the CALIBRATED homography, same as the adapter does), then
    # difference.
    prev_recovered = transform_points(H, [prev_pixel])[0]
    curr_recovered = transform_points(H, [curr_pixel])[0]
    correct_velocity = (curr_recovered - prev_recovered) / dt

    # Naive (WRONG) method: transform the raw pixel DELTA as if it were a
    # point. This is mathematically invalid (perspectiveTransform's
    # homogeneous divide has no meaning for a free vector/difference, only
    # for an actual point with an implicit w=1) but is exactly the kind of
    # shortcut Step 1.3 warns against.
    pixel_delta = [curr_pixel[0] - prev_pixel[0], curr_pixel[1] - prev_pixel[1]]
    naive_velocity = transform_points(H, [pixel_delta])[0] / dt

    velocity_method_difference = np.linalg.norm(correct_velocity - naive_velocity)
    print(f"\nCorrect (point-then-difference) velocity: {correct_velocity}")
    print(f"Naive (vector-transform) velocity:         {naive_velocity}")
    print(f"Difference between methods: {velocity_method_difference:.2f} m/s")

    assert velocity_method_difference > 100.0, (
        "expected the correct and naive velocity methods to differ substantially under real "
        f"perspective distortion; got only {velocity_method_difference:.4f} difference -- the "
        "synthetic scenario may not be exercising meaningful distortion"
    )

    # Now confirm the ADAPTER's actual output matches the CORRECT method.
    tracks = [{"track_id": 1, "pos_pixel": curr_pixel}]
    prev_positions_pixel = {1: prev_pixel}
    ball_pixel = _pixel(H_true, (50.0, 34.0))
    team_mapping = {1: "team_A"}

    result = convert_frame_to_tensors(tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel)

    adapter_velocity = result["player_vel"][0].numpy()
    print(f"Adapter's returned velocity: {adapter_velocity}")

    assert np.allclose(adapter_velocity, correct_velocity, atol=0.5), (
        f"adapter velocity {adapter_velocity} does not match the correct method's "
        f"{correct_velocity}"
    )
    assert not np.allclose(adapter_velocity, naive_velocity, atol=1.0), (
        "adapter velocity matches the NAIVE (incorrect) vector-transform method -- Step 1.3's "
        "fix does not appear to be in effect"
    )


def test_first_frame_with_no_previous_position_gives_zero_velocity():
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks = [{"track_id": 1, "pos_pixel": _pixel(H_true, (50.0, 34.0))}]
    team_mapping = {1: "team_A"}
    ball_pixel = _pixel(H_true, (52.0, 34.0))

    result = convert_frame_to_tensors(tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel=None)

    assert torch.allclose(result["player_vel"][0], torch.tensor([0.0, 0.0]))


def test_missing_ball_returns_none_for_whole_bundle():
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, _ball_pixel, team_mapping = _build_basic_scenario(H_true)

    result = convert_frame_to_tensors(tracks, None, team_mapping, H, FPS, prev_positions_pixel)

    assert result is None


def test_unmapped_track_id_excluded_not_a_keyerror():
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, ball_pixel, team_mapping = _build_basic_scenario(H_true)

    # track_id 99 has NO entry in team_mapping.
    tracks = tracks + [{"track_id": 99, "pos_pixel": _pixel(H_true, (55.0, 40.0))}]

    result = convert_frame_to_tensors(tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel)

    assert result["player_pos"].shape == (3, 2)  # unchanged -- track 99 excluded, not crashed


def test_outlier_mapped_track_is_filtered():
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, ball_pixel, team_mapping = _build_basic_scenario(H_true)

    tracks = tracks + [{"track_id": 4, "pos_pixel": _pixel(H_true, (50.0, 10.0))}]
    prev_positions_pixel = dict(prev_positions_pixel)
    prev_positions_pixel[4] = _pixel(H_true, (50.0, 10.0))
    team_mapping = dict(team_mapping)
    team_mapping[4] = "outlier"  # e.g. the referee

    result = convert_frame_to_tensors(tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel)

    assert result["player_pos"].shape == (3, 2)  # the outlier (track 4) is excluded


def test_team_labels_stable_across_frames_same_mapping():
    """Calls the adapter across 2 synthetic frames using the SAME
    team_mapping dict, with the ball held near team_B's player in both --
    is_teammate's outcome for each track_id must be IDENTICAL across both
    calls, confirming team labels are genuinely persistent (looked up from
    the same fixed mapping), not silently re-clustered per call.
    """
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)

    frame_1_meters = {1: (30.0, 20.0), 2: (70.0, 50.0), 3: (50.0, 34.0)}
    frame_2_meters = {1: (31.0, 21.0), 2: (71.0, 51.0), 3: (51.0, 35.0)}
    team_mapping = {1: "team_A", 2: "team_A", 3: "team_B"}
    ball_meters = (52.0, 34.0)  # near track_id 3 (team_B) in both frames

    tracks_1 = [{"track_id": tid, "pos_pixel": _pixel(H_true, pos)} for tid, pos in frame_1_meters.items()]
    tracks_2 = [{"track_id": tid, "pos_pixel": _pixel(H_true, pos)} for tid, pos in frame_2_meters.items()]
    ball_pixel = _pixel(H_true, ball_meters)

    result_1 = convert_frame_to_tensors(tracks_1, ball_pixel, team_mapping, H, FPS, prev_positions_pixel=None)
    result_2 = convert_frame_to_tensors(
        tracks_2, ball_pixel, team_mapping, H, FPS,
        prev_positions_pixel={tid: _pixel(H_true, pos) for tid, pos in frame_1_meters.items()},
    )

    assert result_1["is_teammate"].tolist() == result_2["is_teammate"].tolist() == [False, False, True]


def test_camera_motion_correction_omitted_reproduces_original_behavior_exactly():
    """Milestone 37's additive `camera_motion_correction` parameter: when
    omitted (the default), behavior must be byte-for-byte identical to
    Milestone 30's original signature -- explicitly regression-tested here,
    same discipline as Milestone 22's opt-in `habit_heatmaps` parameter.
    """
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, ball_pixel, team_mapping = _build_basic_scenario(H_true)

    result_without_param = convert_frame_to_tensors(tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel)
    result_with_explicit_none = convert_frame_to_tensors(
        tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel, camera_motion_correction=None
    )

    assert torch.equal(result_without_param["player_pos"], result_with_explicit_none["player_pos"])
    assert torch.equal(result_without_param["player_vel"], result_with_explicit_none["player_vel"])
    assert torch.equal(result_without_param["ball_pos"], result_with_explicit_none["ball_pos"])
    assert torch.equal(result_without_param["is_teammate"], result_with_explicit_none["is_teammate"])


def test_camera_motion_correction_actually_composes_with_base_homography():
    """When `camera_motion_correction` IS provided, confirms the adapter
    genuinely composes it (`homography_matrix @ camera_motion_correction`)
    rather than ignoring it or applying it some other way -- verified by
    independently computing the expected transformed position via the SAME
    composed matrix and asserting the adapter's output matches exactly.
    """
    H_true = _build_synthetic_broadcast_camera_homography()
    H = _calibrate(H_true)
    tracks, prev_positions_pixel, ball_pixel, team_mapping = _build_basic_scenario(H_true)

    # A deliberately non-identity correction (a small synthetic pixel-space
    # shift, analogous to what one frame of real camera pan would produce)
    # -- NOT the identity matrix, so this test cannot pass by accident.
    camera_motion_correction = np.array(
        [[1.0, 0.0, 15.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]]
    )

    result = convert_frame_to_tensors(
        tracks, ball_pixel, team_mapping, H, FPS, prev_positions_pixel,
        camera_motion_correction=camera_motion_correction,
    )

    expected_homography = H @ camera_motion_correction
    expected_ball_meters = transform_points(expected_homography, [ball_pixel])[0]

    assert np.allclose(result["ball_pos"].numpy(), expected_ball_meters, atol=1e-4)

    # And confirm it's NOT just silently reproducing the uncorrected result
    # (i.e. the correction must have actually changed something).
    uncorrected_ball_meters = transform_points(H, [ball_pixel])[0]
    assert not np.allclose(expected_ball_meters, uncorrected_ball_meters, atol=1e-4), (
        "test's own correction matrix is a no-op at the ball position -- adjust the fixture"
    )
