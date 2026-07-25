"""Milestone 26 (Module 4): ByteTrack player tracking.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25 -- nothing in `production/src/models`, `production/src/
pipeline`, `production/src/spatial`, `production/src/physics`, or
`production/src/serving` is imported by or imports from this module.

CRITICAL METHODOLOGICAL NOTE -- read before using this module's output for
anything downstream, and do not let this get lost in later phases:

This module measures APPARENT PIXEL VELOCITY, not true player velocity.
Raw pixel displacement from a broadcast camera conflates PLAYER motion with
CAMERA motion (pan/zoom/cut). The two are only equivalent if the source
footage comes from a genuinely static, fixed camera -- most broadcast
football footage is NOT that; it pans and zooms to follow play. Every
velocity value this module returns is named `vel_pixels_per_sec`, never
`vel` or `velocity`, specifically so this unit ambiguity cannot silently
propagate into later code that might otherwise assume physical units.

Getting to TRUE player velocity (m/s) requires, additionally:
  1. Pitch calibration (Phase 27, NOT YET BUILT) to convert pixel
     coordinates to real pitch meters.
  2. Camera-motion compensation (NOT YET BUILT, not scoped to any phase
     yet) if the source footage is not static -- broadcast footage
     essentially never is.

This module does NOT "fix" the `player_vel = [0, 0]` limitation carried
since Milestone 3 (StatsBomb 360 freeze-frames have no velocity field at
all). It produces a NEW, useful, but explicitly incomplete signal --
pixel velocity from a possibly-moving camera -- that later phases must
convert (calibration) and correct (motion compensation) before it could
ever feed the existing physics engine (`BiomechanicalPitchControl`, whose
velocity inputs are real m/s). Do not wire this module's output directly
into that engine.
"""

import cv2

from production.src.cv.detector import COCO_PERSON_CLASS_ID, _load_model

# Sanity ceiling for frame-to-frame pixel displacement, used to FLAG (never
# silently discard) likely track-ID switches -- a real, expected ByteTrack
# failure mode when players cross paths or are briefly occluded, not
# something to hide from the output.
#
# Rough, explicitly approximate derivation (a real value requires Phase
# 27's pitch calibration, not available yet): a sprinting human tops out
# around ~10 m/s. A common wide broadcast shot renders a ~105m pitch length
# across roughly 1200-1900px of frame width, i.e. very roughly ~12-18
# pixels per meter at that zoom level -- so a genuine 10 m/s sprint could
# plausibly show up as ~120-180 px/s in a wide shot, but considerably more
# under a tighter/zoomed shot. This ceiling is set well above that range
# deliberately, so it only fires on displacements implausible even under
# generous zoom assumptions, rather than flagging legitimate fast running
# under a tight camera shot as a false ID-switch.
LIKELY_ID_SWITCH_PIXEL_VELOCITY_CEILING = 800.0


def run_tracking(
    video_path: str,
    confidence_threshold: float = 0.5,
    model_checkpoint: str = "yolov8m.pt",
) -> list[dict]:
    """Runs YOLO + ByteTrack over `video_path`, returning per-frame
    detected-person tracks with apparent pixel velocity.

    The video's ACTUAL frame rate is read directly from the file
    (`cv2.CAP_PROP_FPS`) and used for every `dt` in the velocity
    calculation -- never a hardcoded assumption (25/30/etc.), since real
    broadcast footage varies (commonly 25/30/50/60fps depending on
    source) and a wrong assumed fps would silently scale every velocity
    value incorrectly.

    Returns a list of per-frame dicts:
    `{"frame_num": int, "tracks": [{"track_id": int, "pos": [x, y],
    "vel_pixels_per_sec": [vx, vy], "likely_id_switch": bool}, ...]}`.
    `pos` is the detected box's top-left corner in pixel coordinates (same
    `[x, y, w, h]`-style top-left-origin convention as
    `detector.run_baseline_detection`'s output; width/height are not
    carried into this module's per-track output).

    Filters to COCO class 0 ("person") only, same scope note as
    `detector.run_baseline_detection`: this matches all human ground-truth
    roles (player/goalkeeper/referee) and will also track spectators/staff
    visible in the shot, since there is no pitch-region restriction
    available yet (Phase 27).
    """
    capture = cv2.VideoCapture(str(video_path))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if not actual_fps or actual_fps <= 0:
        raise ValueError(
            f"Could not read a valid frame rate from {video_path} (got {actual_fps!r}) -- "
            "refusing to silently assume a default fps, since that would scale every "
            "velocity value incorrectly."
        )
    dt_seconds = 1.0 / actual_fps

    model = _load_model(model_checkpoint)
    # stream=True: iterate frame-by-frame rather than loading the whole
    # video into memory at once -- matters more as clip length grows.
    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        conf=confidence_threshold,
        persist=True,
        stream=True,
    )

    previous_positions: dict[int, list[float]] = {}
    output = []

    for frame_num, result in enumerate(results):
        boxes = result.boxes
        frame_tracks = []

        # ByteTrack assigns no track IDs this frame (e.g. zero detections,
        # or nothing survived the tracker's own confirmation logic yet).
        if boxes.id is None:
            output.append({"frame_num": frame_num, "tracks": []})
            continue

        for box_xyxy, cls_id, track_id_tensor in zip(boxes.xyxy, boxes.cls, boxes.id):
            if int(cls_id.item()) != COCO_PERSON_CLASS_ID:
                continue

            track_id = int(track_id_tensor.item())
            x1, y1, x2, _y2 = (v.item() for v in box_xyxy)
            pos = [x1, y1]

            if track_id in previous_positions:
                prev_pos = previous_positions[track_id]
                vx = (pos[0] - prev_pos[0]) / dt_seconds
                vy = (pos[1] - prev_pos[1]) / dt_seconds
            else:
                # First frame this track_id has been seen -- no prior
                # position to difference against.
                vx, vy = 0.0, 0.0

            speed_pixels_per_sec = (vx**2 + vy**2) ** 0.5
            likely_id_switch = speed_pixels_per_sec > LIKELY_ID_SWITCH_PIXEL_VELOCITY_CEILING

            frame_tracks.append(
                {
                    "track_id": track_id,
                    "pos": pos,
                    "vel_pixels_per_sec": [vx, vy],
                    "likely_id_switch": likely_id_switch,
                }
            )

            previous_positions[track_id] = pos

        output.append({"frame_num": frame_num, "tracks": frame_tracks})

    return output
