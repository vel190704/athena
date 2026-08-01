"""Milestone 41 validation: top-down tactical map rendering + side-by-side
video composition.

Per ADR-014, everything under test here is a strictly LOCAL, non-served
research prototype -- no test in this file touches
`production/src/serving/`.
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pytest

from production.src.cv import tactical_map_renderer
from production.src.cv.pipeline import CVPipeline
from production.src.cv.pixel_overlay_renderer import TEAM_COLORS_BGR
from production.src.cv.tactical_map_renderer import (
    ACCURACY_CAVEAT_TEXT,
    TRUST_RADIUS_PX,
    UNAVAILABLE_BACKGROUND_BGR,
    render_tactical_map,
    transform_players_to_pitch_space,
)
from production.src.cv.video_export import export_side_by_side_video

TEST_MATCH_VIDEO_PATH = "data/raw/test_match.mp4"
REAL_CLIP_MAX_FRAMES = 20  # real network cost per valid frame -- keep short (matches ADR-016's own sample sizes)


def _fake_solve_with_homography(homography):
    def fake_solve(keypoints, excluded_vertices=None, **kwargs):
        return {
            "homography": homography,
            "is_stale": homography is None,
            "inlier_vertex_numbers": [],
            "outlier_vertex_numbers": [],
            "iterations": 0,
        }
    return fake_solve


def _fake_detect(frame):
    return {"keypoints": [], "pitch_confidence": 1.0}


def _fake_detect_with_reliable_cluster(centroid_px: tuple[float, float]) -> callable:
    """A fake `detect_pitch_keypoints` whose returned keypoints, once
    filtered to confident + non-excluded (the same filter
    `_reliable_keypoint_centroid_px` applies), average out to exactly
    `centroid_px` -- two confident, non-excluded vertices straddling the
    target centroid symmetrically."""
    cx, cy = centroid_px

    def fake_detect(frame):
        return {
            "keypoints": [
                {"vertex_number": 20, "x_px": cx - 10.0, "y_px": cy, "confidence": 0.99},
                {"vertex_number": 21, "x_px": cx + 10.0, "y_px": cy, "confidence": 0.99},
            ],
            "pitch_confidence": 1.0,
        }
    return fake_detect


def test_transform_players_to_pitch_space_synthetic_known_homography(monkeypatch):
    """Step 4.1: a known pure-scale homography (pixel/10 = meter) and known
    player pixel positions -- asserts the transform (feet-position pixel
    extraction + homography application) produces the expected pitch-space
    coordinates within a stated tolerance."""
    known_homography = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(tactical_map_renderer, "detect_pitch_keypoints", _fake_detect_with_reliable_cluster((200.0, 100.0)))
    monkeypatch.setattr(
        tactical_map_renderer, "solve_homography_from_keypoints", _fake_solve_with_homography(known_homography)
    )

    # bbox [x, y, w, h] = [150, 50, 100, 50] -> feet position (bottom-center)
    # = (150 + 100/2, 50 + 50) = (200, 100) -> expected pitch space (20, 10).
    # This exactly coincides with the fake reliable-keypoint cluster
    # centroid above, so the player is expected to be TRUSTED (distance 0).
    detected_players = {
        "tracks": [{"track_id": 7, "bbox": [150.0, 50.0, 100.0, 50.0]}],
        "ball_pixel": [400.0, 200.0],  # expected pitch space (40, 20)
        "team_mapping": {7: "team_A"},
    }
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = transform_players_to_pitch_space(detected_players, dummy_frame)

    TOLERANCE_M = 1e-3
    assert result is not None
    assert result["homography_valid"] is True
    assert len(result["positions"]) == 1
    track_id, team, x, y, trusted = result["positions"][0]
    assert track_id == 7
    assert team == "team_A"
    assert abs(x - 20.0) < TOLERANCE_M
    assert abs(y - 10.0) < TOLERANCE_M
    assert trusted is True
    assert result["trusted_count"] == 1
    assert result["untrusted_count"] == 0
    assert result["ball"] is not None
    ball_x, ball_y = result["ball"]
    assert abs(ball_x - 40.0) < TOLERANCE_M
    assert abs(ball_y - 20.0) < TOLERANCE_M

    print(f"Synthetic transform: player -> ({x:.4f}, {y:.4f})m, ball -> ({ball_x:.4f}, {ball_y:.4f})m, trusted={trusted}")


def test_transform_players_to_pitch_space_trust_gating_synthetic(monkeypatch):
    """ADR-017 Step 1: a player exactly AT the reliable-keypoint cluster
    centroid must be trusted; a player far beyond `TRUST_RADIUS_PX` must
    not be, using the SAME identity homography for both so only distance
    from the cluster differs between them."""
    identity_homography = np.eye(3, dtype=np.float32)
    cluster_centroid = (500.0, 500.0)
    monkeypatch.setattr(tactical_map_renderer, "detect_pitch_keypoints", _fake_detect_with_reliable_cluster(cluster_centroid))
    monkeypatch.setattr(
        tactical_map_renderer, "solve_homography_from_keypoints", _fake_solve_with_homography(identity_homography)
    )

    near_bbox = [500.0, 500.0, 0.0, 0.0]  # feet position exactly at the centroid -- distance 0
    far_bbox = [500.0 + TRUST_RADIUS_PX * 3, 500.0, 0.0, 0.0]  # feet position far beyond the radius

    detected_players = {
        "tracks": [{"track_id": 1, "bbox": near_bbox}, {"track_id": 2, "bbox": far_bbox}],
        "ball_pixel": None,
        "team_mapping": {1: "team_A", 2: "team_B"},
    }
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = transform_players_to_pitch_space(detected_players, dummy_frame)
    positions_by_id = {p[0]: p for p in result["positions"]}

    assert positions_by_id[1][4] is True, "player at the cluster centroid must be trusted"
    assert positions_by_id[2][4] is False, "player far beyond TRUST_RADIUS_PX must not be trusted"
    assert result["trusted_count"] == 1
    assert result["untrusted_count"] == 1


def test_transform_players_to_pitch_space_none_input_returns_none():
    """`detected_players=None` (process_video produced no observation this
    frame) must short-circuit to `None` without attempting a homography
    solve at all."""
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert transform_players_to_pitch_space(None, dummy_frame) is None


def test_transform_players_to_pitch_space_stale_homography_returns_none(monkeypatch):
    """Step 1.2: a failed/stale homography solve must return `None` for
    this frame, NEVER a fallback to a stale prior homography."""
    monkeypatch.setattr(tactical_map_renderer, "detect_pitch_keypoints", _fake_detect)
    monkeypatch.setattr(tactical_map_renderer, "solve_homography_from_keypoints", _fake_solve_with_homography(None))

    detected_players = {"tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10]}], "ball_pixel": None, "team_mapping": {}}
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)

    assert transform_players_to_pitch_space(detected_players, dummy_frame) is None


def test_render_tactical_map_unavailable_placeholder_not_stale_frame():
    """Step 4.2: `pitch_space_data=None` must render the explicit
    "unavailable" placeholder (a distinct background color + caption),
    never positions from some other/previous call."""
    unavailable_canvas = render_tactical_map(None, canvas_size=(300, 200))
    # Corner pixel (far from any drawn text/dot) must match the dedicated
    # "unavailable" background, not the pitch-green background
    # `render_tactical_map` uses for a valid frame.
    corner_pixel = tuple(int(c) for c in unavailable_canvas[5, 5])
    assert corner_pixel == UNAVAILABLE_BACKGROUND_BGR

    valid_data = {
        "positions": [(1, "team_A", 50.0, 34.0, True)],
        "ball": None,
        "homography_valid": True,
        "trusted_count": 1,
        "untrusted_count": 0,
    }
    valid_canvas = render_tactical_map(valid_data, canvas_size=(300, 200))
    valid_corner_pixel = tuple(int(c) for c in valid_canvas[5, 5])
    assert valid_corner_pixel != UNAVAILABLE_BACKGROUND_BGR


def test_trusted_vs_untrusted_markers_are_visually_distinct():
    """ADR-017 Step 1.3: a trusted player must render as a SOLID team-color
    dot; an untrusted one must render as a HOLLOW (outline-only) marker --
    verified by direct pixel inspection at each marker's own center, not
    just "some pixel changed somewhere." A solid dot's center pixel must
    be filled with the team color; a hollow marker's center pixel must
    NOT be (it's empty pitch background inside the outline)."""
    team_a_color = TEAM_COLORS_BGR["team_A"]

    trusted_x_m, trusted_y_m = 20.0, 34.0
    untrusted_x_m, untrusted_y_m = 80.0, 34.0
    data = {
        "positions": [
            (1, "team_A", trusted_x_m, trusted_y_m, True),
            (2, "team_A", untrusted_x_m, untrusted_y_m, False),
        ],
        "ball": None,
        "homography_valid": True,
        "trusted_count": 1,
        "untrusted_count": 1,
    }
    canvas = render_tactical_map(data, canvas_size=(600, 400))

    trusted_px = tactical_map_renderer._to_canvas_xy(trusted_x_m, trusted_y_m, (600, 400))
    untrusted_px = tactical_map_renderer._to_canvas_xy(untrusted_x_m, untrusted_y_m, (600, 400))

    trusted_center_color = tuple(int(c) for c in canvas[trusted_px[1], trusted_px[0]])
    untrusted_center_color = tuple(int(c) for c in canvas[untrusted_px[1], untrusted_px[0]])

    assert trusted_center_color == tuple(team_a_color), (
        f"trusted marker center should be filled with the team color, got {trusted_center_color}"
    )
    assert untrusted_center_color != tuple(team_a_color), (
        f"untrusted (hollow) marker center should NOT be filled with the team color, got {untrusted_center_color}"
    )


def test_trust_ratio_caption_actually_renders_as_pixels():
    """ADR-017 Step 1.4: the new trust-ratio caption ("N/M players in
    reliable range") must actually render as pixels, distinct from and in
    addition to the original accuracy-caveat caption -- checked via
    pixel-region inspection of the second-to-last caption row."""
    data = {
        "positions": [(1, "team_A", 20.0, 34.0, True), (2, "team_B", 80.0, 34.0, False)],
        "ball": None,
        "homography_valid": True,
        "trusted_count": 1,
        "untrusted_count": 1,
    }
    canvas = render_tactical_map(data, canvas_size=(600, 400))

    height = canvas.shape[0]
    # Trust caption sits at y_from_bottom=22, one line above the accuracy
    # caption at y_from_bottom=8 -- inspect a strip around that row.
    strip = canvas[height - 30 : height - 16, :, :]
    background = strip[0, 0].astype(int)
    diff = np.abs(strip.astype(int) - background).sum(axis=-1)
    non_background_pixel_count = int((diff > 30).sum())
    assert non_background_pixel_count > 20, "trust-ratio caption not found as rendered pixels"


def test_accuracy_caption_actually_renders_as_pixels():
    """Step 4.3: confirms the accuracy-caveat caption is really drawn as
    pixels in the output image (a pixel-region inspection), not just
    present in code/comments -- checked on BOTH the valid-frame and the
    unavailable-placeholder path, since Step 2 requires it on every frame."""

    def _caption_row_has_text_pixels(canvas: np.ndarray) -> bool:
        # The caption is drawn white-on-dark/green starting at
        # (6, height-8) -- inspect a horizontal strip around that row for
        # pixels that differ from the row's own most common (background)
        # color, which is a real presence check, not an assumption.
        height = canvas.shape[0]
        strip = canvas[height - 16 : height - 2, :, :]
        background = strip[0, 0].astype(int)
        diff = np.abs(strip.astype(int) - background).sum(axis=-1)
        non_background_pixel_count = int((diff > 30).sum())
        return non_background_pixel_count > 20  # a real caption draws far more than a stray pixel

    valid_data = {"positions": [], "ball": None, "homography_valid": True}
    valid_canvas = render_tactical_map(valid_data, canvas_size=(600, 400))
    assert _caption_row_has_text_pixels(valid_canvas), "accuracy caption not found as rendered pixels on a valid frame"

    unavailable_canvas = render_tactical_map(None, canvas_size=(600, 400))
    assert _caption_row_has_text_pixels(
        unavailable_canvas
    ), "accuracy caption not found as rendered pixels on the unavailable placeholder"

    print(f"Accuracy caption text under test: {ACCURACY_CAVEAT_TEXT!r}")


def test_export_side_by_side_video_real_clip(tmp_path):
    """Step 4.4: runs `export_side_by_side_video` on a short real segment
    of the Milestone 34B clip via `CVPipeline.process_video`'s real
    orchestrated path. Reports the real valid/unavailable tactical-map
    frame ratio. NO GROUND TRUTH exists for this clip -- this is a
    real-data SMOKE/statistics test, not a quantitative accuracy
    validation (that remains ADR-015/016's ~6-7m LOOCV figures, measured
    separately against known correspondences).
    """
    if not Path(TEST_MATCH_VIDEO_PATH).exists():
        pytest.skip(f"No local test video found at {TEST_MATCH_VIDEO_PATH}.")

    from dotenv import load_dotenv

    load_dotenv(".env")
    if not os.environ.get("ROBOFLOW_API_KEY"):
        pytest.skip("ROBOFLOW_API_KEY not set -- this prototype requires it (see ADR-014); skipping real-clip test.")

    output_path = tmp_path / "side_by_side_test.mp4"
    cv_pipeline = CVPipeline()

    summary = export_side_by_side_video(
        TEST_MATCH_VIDEO_PATH, str(output_path), cv_pipeline, max_frames=REAL_CLIP_MAX_FRAMES
    )

    print(f"\n[test_export_side_by_side_video_real_clip] summary: {summary}")
    assert summary["total_frames"] == REAL_CLIP_MAX_FRAMES
    assert summary["frames_with_valid_tactical_map"] + summary["frames_tactical_map_unavailable"] == summary["total_frames"]
    assert output_path.exists() and output_path.stat().st_size > 0

    valid_ratio = summary["frames_with_valid_tactical_map"] / summary["total_frames"]
    print(
        f"valid tactical map: {summary['frames_with_valid_tactical_map']}/{summary['total_frames']} "
        f"({valid_ratio:.0%}), render time: {summary['total_render_time_sec']:.1f}s"
    )

    total_players = summary["total_trusted_players"] + summary["total_untrusted_players"]
    if total_players > 0:
        trusted_ratio = summary["total_trusted_players"] / total_players
        print(
            f"ADR-017 trust ratio (this run): {summary['total_trusted_players']}/{total_players} "
            f"({trusted_ratio:.1%}) players in reliable range -- ADR-017's own measurement found ~27.5%"
        )
