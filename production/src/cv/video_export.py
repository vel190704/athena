"""Milestone 41 (Module 4): side-by-side (annotated video + tactical map)
composition and export.

STANDALONE, part of the same isolated `production/src/cv/` tree
introduced in Milestone 25. Reuses `pipeline.CVPipeline.process_video`'s
REAL orchestrated path (not a standalone harness -- the established
lesson from this project's earlier track-churn investigation), plus this
milestone's own `pixel_overlay_renderer.render_pixel_overlay` and
`tactical_map_renderer.{transform_players_to_pitch_space,
render_tactical_map}`.

CRITICAL, PER ADR-014: this module produces LOCAL VIDEO FILES only. It
MUST NOT be imported by, or wired into, `production/src/serving/api.py`
or any other live/network-accessible surface.

REAL NETWORK COST, STATED PLAINLY (inherited from
`pitch_keypoint_detector.py`, not new to this module): every frame with
tracked-player data triggers one hosted-API call to solve that frame's
homography (~0.5-2s/call per ADR-016's measured figures). A `max_frames`
cap is provided specifically so a real-clip test run stays practical.
"""

import time

import cv2
import numpy as np

from production.src.cv.pipeline import CVPipeline
from production.src.cv.pixel_overlay_renderer import render_pixel_overlay
from production.src.cv.tactical_map_renderer import (
    render_tactical_map,
    transform_players_to_pitch_space,
)
from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH


def export_side_by_side_video(
    video_path: str,
    output_path: str,
    cv_pipeline: CVPipeline,
    max_frames: int | None = None,
) -> dict:
    """Composes, frame-by-frame, the Step 2 pixel-space overlay (left
    panel) alongside the Step 2 tactical map (right panel) into one output
    video written to `output_path`.

    Iterates the SOURCE video directly (via a second `cv2.VideoCapture`
    pass) rather than only the frames `cv_pipeline.process_video` yields,
    so the output video has one composited frame per SOURCE frame,
    including frames `process_video` skipped entirely (non-tactical
    cuts, no ball found, zero surviving players) -- those are rendered
    with `render_pixel_overlay`'s "no player data this frame" caption and
    `render_tactical_map`'s "unavailable" placeholder, never silently
    dropped from the output (which would desync the output video's frame
    count/timing from the source).

    `cv_pipeline.process_video` is run to completion FIRST (materialized
    into a `{frame_num: render_frame_data}` dict) -- its generator can only
    be consumed once, and the composition loop below needs random access
    by frame number while re-reading the video a second time.

    Returns `{"total_frames": int, "frames_with_valid_tactical_map": int,
    "frames_tactical_map_unavailable": int, "total_render_time_sec": float,
    "output_path": str, "total_trusted_players": int,
    "total_untrusted_players": int}`. The last two (ADR-017's trust
    gating) are summed across every valid frame's `trusted_count`/
    `untrusted_count` -- a real, run-wide trusted/untrusted ratio, not
    just a per-frame one.
    """
    start_time = time.perf_counter()

    render_frame_data_by_frame_num: dict[int, dict] = {}
    for result in cv_pipeline.process_video(video_path, max_frames=max_frames):
        render_frame_data_by_frame_num[result["frame_num"]] = result["render_frame_data"]

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tactical-map panel width chosen to preserve the pitch's true 100:68
    # aspect ratio at the source video's height, rather than an arbitrary
    # fixed size that would visually distort it.
    tactical_map_size = (round(frame_height * PITCH_LENGTH / PITCH_WIDTH), frame_height)
    output_size = (frame_width + tactical_map_size[0], frame_height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, output_size)

    total_frames = 0
    frames_valid = 0
    frames_unavailable = 0
    total_trusted_players = 0
    total_untrusted_players = 0

    try:
        frame_index = 0
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            read_ok, frame = capture.read()
            if not read_ok:
                break

            render_frame_data = render_frame_data_by_frame_num.get(frame_index)

            left_panel = render_pixel_overlay(frame, render_frame_data)

            pitch_space_data = transform_players_to_pitch_space(render_frame_data, frame)
            right_panel = render_tactical_map(pitch_space_data, canvas_size=tactical_map_size)

            if pitch_space_data is not None and pitch_space_data.get("homography_valid", False):
                frames_valid += 1
                total_trusted_players += pitch_space_data.get("trusted_count", 0)
                total_untrusted_players += pitch_space_data.get("untrusted_count", 0)
            else:
                frames_unavailable += 1

            composite = np.hstack([left_panel, right_panel])
            writer.write(composite)

            total_frames += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    total_render_time_sec = time.perf_counter() - start_time

    return {
        "total_frames": total_frames,
        "frames_with_valid_tactical_map": frames_valid,
        "frames_tactical_map_unavailable": frames_unavailable,
        "total_render_time_sec": total_render_time_sec,
        "output_path": str(output_path),
        "total_trusted_players": total_trusted_players,
        "total_untrusted_players": total_untrusted_players,
    }
