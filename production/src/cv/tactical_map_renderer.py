"""Milestone 41 (Module 4): top-down tactical map rendering.

STANDALONE, part of the same isolated `production/src/cv/` tree
introduced in Milestone 25 -- nothing in `production/src/models`,
`production/src/pipeline`, `production/src/spatial`, `production/src/
physics`, or `production/src/serving` is imported by or imports from this
module.

CRITICAL, PER ADR-014 (still fully in force, unchanged by this
milestone): this module transitively depends on
`pitch_keypoint_detector.py`'s pretrained-model, hosted-API keypoint
detection. It MUST NOT be imported by, or wired into,
`production/src/serving/api.py`, any other FastAPI endpoint, or any other
live/network-accessible surface. Output is recorded video files / local
artifacts only.

CRITICAL, PER ADR-015/016 (also fully in force, unchanged): this module
renders PLAYER POSITIONS ONLY -- dots on a top-down diagram. It does NOT
compute or render pitch control, weak-zone analysis, or any other
quantity that would require feeding these CV-derived positions into
`BiomechanicalPitchControl` or `DeepHit` -- ADR-015 validated ~6-7m
accuracy for VISUAL rendering only, explicitly not for that purpose, and
determining what accuracy those models would actually need is a separate,
still-open roadmap item (see `docs/CV_PIPELINE_FINDINGS.md` §5). Nothing
in `production/src/reporting/zone_explainer.py` or
`production/src/spatial/control.py` is imported here.

REQUIRES a `ROBOFLOW_API_KEY` available to the process (see
`pitch_keypoint_detector.py`'s own docstring) -- this module does not load
`.env` itself; callers (tests, `video_export.py`) are responsible, the
same division of responsibility `pitch_keypoint_detector.py` already
establishes.
"""

import cv2
import numpy as np

from production.src.cv.calibration import transform_points
from production.src.cv.pitch_keypoint_detector import (
    ADR015_KNOWN_UNRELIABLE_VERTICES,
    DEFAULT_MIN_CONFIDENCE,
    PITCH_KEYPOINTS_METERS,
    detect_pitch_keypoints,
    solve_homography_from_keypoints,
)
from production.src.cv.pixel_overlay_renderer import (
    BALL_COLOR_BGR,
    TEAM_COLORS_BGR,
    player_feet_position,
)
from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH

ACCURACY_CAVEAT_TEXT = (
    "Approximate positions (~6-7m accuracy) -- visual reference only, not validated for tactical/zone analysis"
)

# ADR-017: within ~150px (pixel-space) of the frame's reliable-keypoint
# cluster centroid, held-out homography accuracy measured tight and
# bounded (median 3.35m, max 3.86m, n=24). Beyond it, accuracy both
# worsens AND becomes statistically unpredictable (Spearman r=0.582,
# p<0.0001, worst observed case 933m) -- not a smooth degradation a single
# "slightly bigger dot" could honestly represent. This constant is that
# measured boundary, reused verbatim, not re-tuned here.
TRUST_RADIUS_PX = 150.0

PITCH_BACKGROUND_BGR = (60, 130, 40)   # a muted pitch-green, distinguishable from team colors
PITCH_LINE_BGR = (230, 230, 230)
UNAVAILABLE_BACKGROUND_BGR = (40, 40, 40)
CANVAS_MARGIN_PX = 20
PLAYER_DOT_RADIUS_PX = 6
UNTRUSTED_MARKER_RADIUS_PX = 5
BALL_DOT_RADIUS_PX = 4

# Left/right penalty- and goal-box corner vertex numbers, from
# `pitch_keypoint_detector.PITCH_KEYPOINTS_METERS`'s 32-vertex schema
# (reused exactly, not re-derived -- see that module's own verified
# geometry table) -- a box's bounding rectangle is the min/max over its
# 4 corner vertices, so exact corner ORDER within each list doesn't matter.
_LEFT_PENALTY_BOX_VERTICES = (2, 5, 10, 13)
_LEFT_GOAL_BOX_VERTICES = (3, 4, 7, 8)
_RIGHT_PENALTY_BOX_VERTICES = (18, 21, 26, 29)
_RIGHT_GOAL_BOX_VERTICES = (19, 20, 23, 24)
_CENTRE_CIRCLE_TOP_BOTTOM = (15, 16)  # (length/2, w/2 -+ radius) -- gives the circle's radius directly
_HALFWAY_LINE_ENDS = (14, 17)


def _reliable_keypoint_centroid_px(keypoints: list[dict]) -> np.ndarray | None:
    """The pixel-space centroid of this frame's reliable (confident,
    non-ADR015-excluded) keypoints -- the SAME set, and the SAME method,
    ADR-017's distance-vs-accuracy measurement used, reused verbatim so
    `TRUST_RADIUS_PX` means what that measurement actually measured.
    Returns `None` if no reliable keypoints exist this frame (should not
    happen whenever `solve_homography_from_keypoints` itself succeeded,
    since it fits on this same set, but not assumed impossible).
    """
    reliable = [
        kp
        for kp in keypoints
        if kp["confidence"] >= DEFAULT_MIN_CONFIDENCE and kp["vertex_number"] not in ADR015_KNOWN_UNRELIABLE_VERTICES
    ]
    if not reliable:
        return None
    return np.array([[kp["x_px"], kp["y_px"]] for kp in reliable]).mean(axis=0)


def transform_players_to_pitch_space(detected_players: dict | None, frame: np.ndarray) -> dict | None:
    """Step 1: transforms one frame's detected players (+ ball) from pixel
    space into the verified 100x68m pitch space, via a FRESH per-frame
    keypoint-anchored homography solve (ADR-015/016's qualified,
    fixed-6-vertex-exclusion approach -- reused exactly, not
    reimplemented), AND classifies each player TRUSTED/UNTRUSTED per
    ADR-017's measured trust radius (`TRUST_RADIUS_PX`).

    `detected_players`: `pipeline.CVPipeline.process_video`'s
    `render_frame_data` yield value (`{"tracks": [{"track_id", "bbox",
    ...}, ...], "ball_pixel": [x,y] | None, "team_mapping": {track_id:
    role}}`), or `None` for a frame `process_video` produced no
    observation for at all (e.g. non-tactical/skipped) -- in that case
    this function returns `None` immediately, the same "no data this
    frame" signal as a failed homography solve, per Step 1.2 below.

    Returns `None` (NOT a stale prior homography, and NOT a
    partially-filled result) if:
      - `detected_players` is `None`, or
      - the keypoint-anchored homography solve fails or falls below the
        minimum-viable-point threshold (`solve_homography_from_keypoints`'s
        own `is_stale` convention, ADR-016) -- a frame with no valid
        position data must be visibly ABSENT from the tactical map, not
        silently wrong or silently reused from a previous frame.

    On success, returns `{"positions": [(track_id, team, x, y, trusted),
    ...], "ball": (x, y) | None, "homography_valid": True,
    "trusted_count": int, "untrusted_count": int}`. Player pixel positions
    are taken at the BOTTOM-CENTER of each detected bounding box
    (`pixel_overlay_renderer.player_feet_position`) -- the standard "feet
    position" convention: a player's feet are what's actually on the pitch
    plane the homography maps; the box center sits at torso height, off
    that plane, and would project to a systematically wrong point.

    `trusted` (per ADR-017): `True` iff this player's feet-position pixel
    distance from the frame's reliable-keypoint cluster centroid is within
    `TRUST_RADIUS_PX`. A player is still ASSIGNED a transformed (x, y)
    even when untrusted -- ADR-017 found extrapolated positions become
    unreliable, not that they carry zero information, and the renderer
    (not this function) decides how to visually treat that distinction.
    If the reliable-keypoint centroid itself cannot be computed (should
    not happen whenever the homography solve above succeeded, but not
    assumed impossible), every player is conservatively marked untrusted.
    """
    if detected_players is None:
        return None

    keypoint_result = detect_pitch_keypoints(frame)
    homography_result = solve_homography_from_keypoints(
        keypoint_result["keypoints"], excluded_vertices=ADR015_KNOWN_UNRELIABLE_VERTICES
    )
    if homography_result["homography"] is None:
        return None  # is_stale -- ADR-016's fail-closed convention, not a fabricated fallback

    homography = homography_result["homography"]
    team_mapping = detected_players.get("team_mapping", {})
    cluster_centroid_px = _reliable_keypoint_centroid_px(keypoint_result["keypoints"])

    positions = []
    trusted_count = 0
    untrusted_count = 0
    for track in detected_players.get("tracks", []):
        feet_px = player_feet_position(track["bbox"])
        pitch_xy = transform_points(homography, [feet_px])[0]
        team = team_mapping.get(track["track_id"], "outlier")

        if cluster_centroid_px is None:
            trusted = False
        else:
            pixel_distance = float(np.linalg.norm(np.array(feet_px) - cluster_centroid_px))
            trusted = pixel_distance <= TRUST_RADIUS_PX
        trusted_count += int(trusted)
        untrusted_count += int(not trusted)

        positions.append((track["track_id"], team, float(pitch_xy[0]), float(pitch_xy[1]), trusted))

    ball_pixel = detected_players.get("ball_pixel")
    ball_pitch = None
    if ball_pixel is not None:
        ball_xy = transform_points(homography, [ball_pixel])[0]
        ball_pitch = (float(ball_xy[0]), float(ball_xy[1]))

    return {
        "positions": positions,
        "ball": ball_pitch,
        "homography_valid": True,
        "trusted_count": trusted_count,
        "untrusted_count": untrusted_count,
    }


def _to_canvas_xy(x_m: float, y_m: float, canvas_size: tuple[int, int]) -> tuple[int, int]:
    """Pitch-meter -> canvas-pixel, uniformly scaled (preserves the
    100:68 aspect ratio rather than stretching independently per axis) and
    centered within `canvas_size` with `CANVAS_MARGIN_PX` of margin.
    `y_m=0` maps to the TOP of the canvas -- an arbitrary but explicitly
    documented rendering choice (this is a synthetic top-down diagram, not
    tied to any camera orientation, so there is no "correct" answer to
    cross-reference; it just needs to be stated, not left implicit).
    """
    canvas_w, canvas_h = canvas_size
    avail_w = canvas_w - 2 * CANVAS_MARGIN_PX
    avail_h = canvas_h - 2 * CANVAS_MARGIN_PX
    scale = min(avail_w / PITCH_LENGTH, avail_h / PITCH_WIDTH)
    offset_x = CANVAS_MARGIN_PX + (avail_w - PITCH_LENGTH * scale) / 2.0
    offset_y = CANVAS_MARGIN_PX + (avail_h - PITCH_WIDTH * scale) / 2.0
    return round(offset_x + x_m * scale), round(offset_y + y_m * scale)


def _draw_pitch_outline(canvas: np.ndarray, canvas_size: tuple[int, int]) -> None:
    """Draws the pitch boundary, halfway line, center circle, and both
    penalty/goal boxes, using `PITCH_KEYPOINTS_METERS`'s already-verified,
    ADR-002-rescaled vertex table (`pitch_keypoint_detector.py`) -- reused
    directly rather than re-deriving pitch proportions a third time in
    this codebase (`feature_extractor.py`'s grid and that vertex table
    already established two independently-verified sources; this draws
    from the second one, not a fresh guess).
    """

    def canvas_pt(vertex_number: int) -> tuple[int, int]:
        x_m, y_m = PITCH_KEYPOINTS_METERS[vertex_number]
        return _to_canvas_xy(x_m, y_m, canvas_size)

    def box_rect(vertex_numbers: tuple[int, ...]) -> None:
        pts = [canvas_pt(v) for v in vertex_numbers]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cv2.rectangle(canvas, (min(xs), min(ys)), (max(xs), max(ys)), PITCH_LINE_BGR, 1)

    top_left = _to_canvas_xy(0.0, 0.0, canvas_size)
    bottom_right = _to_canvas_xy(PITCH_LENGTH, PITCH_WIDTH, canvas_size)
    cv2.rectangle(canvas, top_left, bottom_right, PITCH_LINE_BGR, 2)

    cv2.line(canvas, canvas_pt(_HALFWAY_LINE_ENDS[0]), canvas_pt(_HALFWAY_LINE_ENDS[1]), PITCH_LINE_BGR, 1)

    centre_top = canvas_pt(_CENTRE_CIRCLE_TOP_BOTTOM[0])
    centre_bottom = canvas_pt(_CENTRE_CIRCLE_TOP_BOTTOM[1])
    centre_x = (PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0)
    centre_canvas = _to_canvas_xy(*centre_x, canvas_size)
    radius_px = abs(centre_bottom[1] - centre_top[1]) // 2
    cv2.circle(canvas, centre_canvas, max(radius_px, 1), PITCH_LINE_BGR, 1)

    box_rect(_LEFT_PENALTY_BOX_VERTICES)
    box_rect(_LEFT_GOAL_BOX_VERTICES)
    box_rect(_RIGHT_PENALTY_BOX_VERTICES)
    box_rect(_RIGHT_GOAL_BOX_VERTICES)


def render_tactical_map(pitch_space_data: dict | None, canvas_size: tuple[int, int] = (600, 400)) -> np.ndarray:
    """Step 2: draws a top-down pitch diagram with TRUST-GATED player
    markers (+ ball), OR an explicit "unavailable" placeholder when
    `pitch_space_data` is `None`/`homography_valid=False` -- NEVER a
    frozen/stale prior frame's positions rendered without being labeled as
    such (`transform_players_to_pitch_space` never returns stale data in
    the first place, so this function has nothing stale to accidentally
    render even if called carelessly).

    ADR-017 TRUST GATING (this milestone's central change): a player
    marked `trusted=True` (within `TRUST_RADIUS_PX` of the frame's
    reliable-keypoint cluster) is drawn as a SOLID, filled, team-color
    dot -- the same marker style Milestone 41 always used. A player
    marked `trusted=False` is drawn as a HOLLOW (outline-only) marker in
    the same team color, deliberately never filled. HOLLOW was chosen
    over full omission: ADR-017 found extrapolated positions become
    unreliable, not that they carry zero information at all, so a human
    viewer is better served seeing a de-emphasized "roughly here, don't
    trust this" marker than no information at all -- but the two states
    must never be visually confusable, which a filled-vs-hollow
    distinction (rather than, say, a lighter shade of the same filled
    dot) makes unambiguous even at a glance or in a compressed video
    frame.

    TWO CAPTIONS are drawn on EVERY frame this function produces, valid or
    unavailable -- persistent, visible limitation statements in the
    output itself, per Milestone 38's (intended, if never actually built
    -- see this module's docstring) precedent of surfacing real
    limitations directly in the rendered video, not just in code
    comments: `ACCURACY_CAVEAT_TEXT` (Milestone 41's original caption,
    unchanged), and a NEW trust-ratio caption ("N/M players in reliable
    range") reporting THIS frame's real trusted/untrusted split -- the
    unavailable-placeholder path has no player data to report a ratio
    for, so only the accuracy caption is drawn there.
    """
    canvas_w, canvas_h = canvas_size

    if pitch_space_data is None or not pitch_space_data.get("homography_valid", False):
        canvas = np.full((canvas_h, canvas_w, 3), UNAVAILABLE_BACKGROUND_BGR, dtype=np.uint8)
        text = "Tactical map unavailable this frame"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            canvas,
            text,
            ((canvas_w - text_w) // 2, (canvas_h - text_h) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        _draw_caption(canvas, ACCURACY_CAVEAT_TEXT, y_from_bottom=8)
        return canvas

    canvas = np.full((canvas_h, canvas_w, 3), PITCH_BACKGROUND_BGR, dtype=np.uint8)
    _draw_pitch_outline(canvas, canvas_size)

    for _track_id, team, x_m, y_m, trusted in pitch_space_data["positions"]:
        color = TEAM_COLORS_BGR.get(team, TEAM_COLORS_BGR["outlier"])
        px, py = _to_canvas_xy(x_m, y_m, canvas_size)
        if trusted:
            cv2.circle(canvas, (px, py), PLAYER_DOT_RADIUS_PX, color, -1)
            cv2.circle(canvas, (px, py), PLAYER_DOT_RADIUS_PX, (0, 0, 0), 1)
        else:
            # Hollow (outline-only, no fill) -- see docstring for why this,
            # not omission, was chosen, and why it must not be confusable
            # with the solid trusted marker above.
            cv2.circle(canvas, (px, py), UNTRUSTED_MARKER_RADIUS_PX, color, 1)

    if pitch_space_data.get("ball") is not None:
        bx_m, by_m = pitch_space_data["ball"]
        px, py = _to_canvas_xy(bx_m, by_m, canvas_size)
        cv2.circle(canvas, (px, py), BALL_DOT_RADIUS_PX, BALL_COLOR_BGR, -1)
        cv2.circle(canvas, (px, py), BALL_DOT_RADIUS_PX, (0, 0, 0), 1)

    trusted_count = pitch_space_data.get("trusted_count", 0)
    untrusted_count = pitch_space_data.get("untrusted_count", 0)
    total = trusted_count + untrusted_count
    if total > 0:
        trust_text = f"{trusted_count}/{total} players in reliable range (ADR-017, <{int(TRUST_RADIUS_PX)}px)"
        _draw_caption(canvas, trust_text, y_from_bottom=22)

    _draw_caption(canvas, ACCURACY_CAVEAT_TEXT, y_from_bottom=8)
    return canvas


def _draw_caption(canvas: np.ndarray, text: str, y_from_bottom: int) -> None:
    """Shrinks the font until `text` fits within the canvas width (minus a
    small margin) rather than letting it clip off-screen at smaller
    `canvas_size` values -- captions are required to be VISIBLE (Step 2's
    explicit instruction), so silently truncating one at the canvas edge
    would defeat the point. `y_from_bottom` lets multiple captions stack
    without overlapping (the accuracy caption stays at the very bottom,
    matching Milestone 41's original position; the trust-ratio caption
    sits one line above it)."""
    canvas_w = canvas.shape[1]
    max_text_width = canvas_w - 12
    font_scale = 0.35
    while font_scale > 0.15:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        if text_w <= max_text_width:
            break
        font_scale -= 0.02

    cv2.putText(
        canvas,
        text,
        (6, canvas.shape[0] - y_from_bottom),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
