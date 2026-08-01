"""Milestone 41 (Module 4): pixel-space annotation overlay.

FILLS A REAL GAP FOUND WHILE BUILDING THIS MILESTONE: `context.md`'s prose
describes a "Milestone 38" pixel-space overlay renderer
(`overlay_renderer.py`) as the project's one completed, camera-motion-
independent Track A deliverable. That file does not exist anywhere in this
repository or its git history (verified via `find`/`git log --all`
before writing this module, not assumed from the documentation) -- no
`cv2.rectangle`/`cv2.putText`/`VideoWriter`/drawing code of any kind
existed anywhere under `production/src/cv/` prior to this module. This is
a genuine documentation/reality mismatch, reported plainly (Milestone 41's
own findings), not silently papered over. This module is NEW code, built
to fill that gap so Milestone 41's side-by-side video composition
(`tactical_map_renderer.render_tactical_map` + `video_export.
export_side_by_side_video`) has a real pixel-space panel to compose
against -- it does not modify, and was not adapted from, any pre-existing
file (there was nothing to adapt from).

STANDALONE, part of the same isolated `production/src/cv/` tree
introduced in Milestone 25 -- nothing in `production/src/models`,
`production/src/pipeline`, `production/src/spatial`, `production/src/
physics`, or `production/src/serving` is imported by or imports from this
module. Consumes `pipeline.CVPipeline.process_video`'s ADDITIVE
`render_frame_data` yield key (added this milestone) -- does not
duplicate any of that orchestrator's detection/tracking/team-
classification/ball-detection logic itself.

TEAM DISPLAY COLORS, STATED EXPLICITLY (also not defined anywhere in
`team_classifier.py` -- that module returns only the role labels
`"team_A"`/`"team_B"`/`"outlier"`, no display colors; this mapping is new,
invented here for rendering purposes only): BGR (OpenCV's native channel
order, matching every other `cv2.*` call in this codebase) blue for
`team_A`, red for `team_B`, yellow for `outlier` (goalkeeper/referee),
white for the ball.
"""

import cv2
import numpy as np

TEAM_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "team_A": (255, 100, 0),   # blue
    "team_B": (0, 0, 255),     # red
    "outlier": (0, 220, 255),  # yellow
}
BALL_COLOR_BGR = (255, 255, 255)  # white

BOX_THICKNESS = 2
TRACK_ID_FONT_SCALE = 0.5
BALL_MARKER_RADIUS = 6
POSSESSION_BANNER_HEIGHT = 24


def render_pixel_overlay(frame: np.ndarray, render_frame_data: dict | None) -> np.ndarray:
    """Draws bounding boxes (team-color-coded), track_id labels, and a ball
    marker onto a COPY of `frame` -- the original `frame` array is never
    mutated, matching this codebase's established "never mutate the
    caller's data in place" discipline (e.g. `feature_extractor.py`'s
    `player_pos.clone()` before any in-place write).

    `render_frame_data`: `pipeline.CVPipeline.process_video`'s
    `render_frame_data` yield value (`{"tracks": [...], "ball_pixel":
    [...] | None, "team_mapping": {track_id: role}}`), or `None` for a
    frame `process_video` produced no observation for at all (e.g. a
    non-tactical/skipped frame) -- in that case, the raw frame is returned
    with only a small "no player data this frame" caption, never fabricated
    boxes.
    """
    annotated = frame.copy()

    if render_frame_data is None:
        cv2.putText(
            annotated,
            "No player data this frame (non-tactical / skipped)",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated

    team_mapping = render_frame_data.get("team_mapping", {})
    for track in render_frame_data.get("tracks", []):
        track_id = track["track_id"]
        x, y, w, h = track["bbox"]
        role = team_mapping.get(track_id, "outlier")
        color = TEAM_COLORS_BGR.get(role, TEAM_COLORS_BGR["outlier"])

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(
            annotated,
            str(track_id),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            TRACK_ID_FONT_SCALE,
            color,
            1,
            cv2.LINE_AA,
        )

    ball_pixel = render_frame_data.get("ball_pixel")
    if ball_pixel is not None:
        bx, by = int(ball_pixel[0]), int(ball_pixel[1])
        cv2.circle(annotated, (bx, by), BALL_MARKER_RADIUS, BALL_COLOR_BGR, -1)
        cv2.circle(annotated, (bx, by), BALL_MARKER_RADIUS, (0, 0, 0), 1)

    return annotated


def player_feet_position(bbox: list[float]) -> tuple[float, float]:
    """The standard "feet position" pixel convention for mapping a
    detected player's bounding box onto a top-down pitch homography:
    bottom-center of the box (`x + w/2`, `y + h`), NOT the box center or
    top-left corner -- a player's FEET are what's actually touching the
    pitch plane the homography maps, so this is the point whose
    perspective-projected position is meaningful to transform; the box
    center sits at roughly torso height, off the pitch plane, and would
    project to a systematically wrong pitch-space point under a flat-
    ground homography.

    `bbox`: `[x, y, w, h]`, top-left origin (matching `pipeline.py`/
    `detector.py`/`tracker.py`'s convention throughout this codebase).
    """
    x, y, w, h = bbox
    return (x + w / 2.0, y + h)
