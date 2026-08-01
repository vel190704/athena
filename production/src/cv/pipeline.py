"""Milestone 32 (Module 4): live-stream ingestion & pipeline orchestration.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25. Chains every prior CV milestone (shot classification,
detection, tracking, ball detection, team classification, calibration, the
adapter) into one frame-by-frame generator. Does not modify
`production/src/models`, `production/src/pipeline`, `production/src/
spatial`, `production/src/physics`, or `production/src/serving`.

This module DOES extend `production/src/cv/adapter.py` (Milestone 30, part
of this same CV module family, not the original ML/physics/API codebase)
with an additive, backward-compatible `dt_seconds_per_track` parameter --
see that module's docstring. Milestone 30's own test suite
(`test_adapter.py`) passes unchanged against the extended signature.

THE CORE BUG THIS ORCHESTRATOR MUST NOT REINTRODUCE: Milestone 31's shot
classifier legitimately SKIPS whole runs of non-tactical frames (camera
cuts to replays/close-ups are NORMAL broadcast behavior, not rare). If a
track_id's "previous" observation were naively assumed to be exactly one
frame ago (Milestone 30's original single-video assumption, reasonable in
isolation), every velocity computed across a skip would be silently wrong
by a factor of however many frames were actually skipped. This module
tracks each track_id's last-observed FRAME INDEX (not just its position),
computes the TRUE elapsed frame gap, and passes a per-track `dt_seconds`
into the adapter accordingly -- see `CVPipeline.process_video` Step (d).
"""

import time
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from production.src.cv.adapter import convert_frame_to_tensors
from production.src.cv.ball_detector import DEFAULT_BALL_CONFIDENCE_THRESHOLD, detect_ball
from production.src.cv.detector import COCO_PERSON_CLASS_ID
from production.src.cv.shot_classifier import compute_shot_features, is_tactical_view
from production.src.cv.team_classifier import classify_teams, extract_jersey_color

# Milestone 30's own default person-detection confidence (Milestone 25).
DEFAULT_TRACKING_CONFIDENCE_THRESHOLD = 0.5

# If a track_id's previous observation is more than this many REAL frames
# ago, its position is treated as too stale to trust for a velocity
# computation -- a real camera cut and return plausibly reassigns
# ByteTrack IDs, and the "same identity, still moving continuously"
# assumption weakens the longer the gap. Configurable via
# `CVPipeline(stale_gap_frames_threshold=...)`; this default (5) is an
# UNVALIDATED GUESS pending real-footage tuning, same status as every
# other hand-tuned constant in this project until validated.
DEFAULT_STALE_GAP_FRAMES_THRESHOLD = 5

MIN_PLAYERS_FOR_TEAM_CLASSIFICATION = 2


def compute_track_dt_seconds(
    last_observed_frame_index: int | None,
    current_frame_index: int,
    fps: float,
    stale_gap_frames_threshold: int,
) -> tuple[float | None, bool]:
    """The core gap/staleness decision this milestone exists to get right,
    factored out as a pure function (no video I/O, no model calls) so it
    can be tested directly without depending on a real video file.

    Given a track's last-observed FRAME INDEX (`None` if this is the
    track's first-ever observation), returns `(dt_seconds, is_stale)`:
      - `(None, False)`: no previous observation at all -- the caller
        should use the "track just appeared" [0, 0] velocity convention.
      - `(None, True)`: a previous observation exists, but the REAL
        elapsed gap (`current_frame_index - last_observed_frame_index`,
        which may span several real frames if Milestone 31 skipped
        non-tactical frames in between) exceeds `stale_gap_frames_threshold`
        -- too stale to trust for velocity; same [0, 0] fallback, but
        distinguishable from the "never observed" case for diagnostics
        (`stale_velocity_fallback_count`).
      - `(dt_seconds, False)`: a trustworthy previous observation --
        `dt_seconds = gap_frames / fps` using the TRUE gap, never a
        hardcoded `1/fps`.
    """
    if last_observed_frame_index is None:
        return None, False

    gap_frames = current_frame_index - last_observed_frame_index
    if gap_frames > stale_gap_frames_threshold:
        return None, True

    return gap_frames / fps, False


def _extract_person_tracks(result) -> list[dict]:
    """Parses one Ultralytics tracking `Results` object into
    `[{"track_id": int, "pos_pixel": [x, y], "bbox": [x, y, w, h]}, ...]`,
    filtered to COCO class 0 ("person"). `pos_pixel` is the box's top-left
    corner (matching `detector.py`/`tracker.py`'s convention); `bbox` is
    kept alongside (unlike `tracker.py`'s own leaner output schema) because
    THIS module also needs box width/height for jersey-color cropping
    (`team_classifier.extract_jersey_color`).
    """
    tracks = []
    boxes = result.boxes
    if boxes.id is None:
        return tracks

    for box_xyxy, cls_id, track_id_tensor in zip(boxes.xyxy, boxes.cls, boxes.id):
        if int(cls_id.item()) != COCO_PERSON_CLASS_ID:
            continue
        track_id = int(track_id_tensor.item())
        x1, y1, x2, y2 = (v.item() for v in box_xyxy)
        tracks.append(
            {
                "track_id": track_id,
                "pos_pixel": [x1, y1],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return tracks


def _build_pixel_space_tensors(
    tracks: list[dict],
    ball_pixel: list[float] | None,
    team_mapping: dict[int, str],
    fps: float,
    prev_positions_pixel: dict[int, list[float]] | None,
    dt_seconds_per_track: dict[int, float],
) -> dict | None:
    """Milestone 32 Step 1.3.g's documented fallback: when no
    `homography_matrix` is available, positions/velocities are returned in
    RAW PIXEL SPACE rather than calibrated meters -- explicitly NOT run
    through `calibration.transform_points` or `adapter.py`'s meter-space
    pitch-bounds check (which assumes `[0,100]x[0,68]` meters and would
    reject essentially every real pixel coordinate, since a 1280x720
    frame's pixel values are nowhere near that range). This mirrors
    `convert_frame_to_tensors`'s possession heuristic and per-track
    velocity logic exactly, just without the calibration step -- a
    DOCUMENTED LIMITATION (uncalibrated output), not a silent
    simplification: callers must know these are pixels, not meters, before
    feeding them anywhere physics-related.
    """
    if ball_pixel is None:
        return None

    included_positions: list[list[float]] = []
    included_velocities: list[list[float]] = []
    included_teams: list[str] = []

    default_dt_seconds = 1.0 / fps

    for track in tracks:
        track_id = track["track_id"]
        role = team_mapping.get(track_id, "outlier")
        if role == "outlier":
            continue

        pos_pixel = track["pos_pixel"]

        if prev_positions_pixel is not None and track_id in prev_positions_pixel:
            prev_pixel = prev_positions_pixel[track_id]
            track_dt_seconds = dt_seconds_per_track.get(track_id, default_dt_seconds)
            velocity = [
                (pos_pixel[0] - prev_pixel[0]) / track_dt_seconds,
                (pos_pixel[1] - prev_pixel[1]) / track_dt_seconds,
            ]
        else:
            velocity = [0.0, 0.0]

        included_positions.append(pos_pixel)
        included_velocities.append(velocity)
        included_teams.append(role)

    num_players = len(included_positions)
    if num_players == 0:
        return {
            "player_pos": torch.zeros((0, 2), dtype=torch.float32),
            "player_vel": torch.zeros((0, 2), dtype=torch.float32),
            "is_teammate": torch.zeros((0,), dtype=torch.bool),
            "ball_pos": torch.tensor(ball_pixel, dtype=torch.float32),
            "fatigue_mod": torch.zeros((0,), dtype=torch.float32),
        }

    positions_array = np.array(included_positions)
    velocities_array = np.array(included_velocities)
    ball_array = np.array(ball_pixel)

    distances_to_ball = np.linalg.norm(positions_array - ball_array, axis=1)
    nearest_player_index = int(np.argmin(distances_to_ball))
    possessing_team = included_teams[nearest_player_index]
    is_teammate = np.array([team == possessing_team for team in included_teams], dtype=bool)

    return {
        "player_pos": torch.tensor(positions_array, dtype=torch.float32),
        "player_vel": torch.tensor(velocities_array, dtype=torch.float32),
        "is_teammate": torch.tensor(is_teammate, dtype=torch.bool),
        "ball_pos": torch.tensor(ball_array, dtype=torch.float32),
        "fatigue_mod": torch.ones((num_players,), dtype=torch.float32),
    }


class CVPipeline:
    """Chains shot classification, YOLO+ByteTrack tracking, ball
    detection, team classification, calibration, and the Milestone 30
    adapter into one frame-by-frame generator.

    A DEDICATED YOLO model instance is loaded for tracking (`self._tracking_model`),
    separate from `ball_detector.detect_ball`'s own internally-cached
    detection-mode model instance (even when both use the same
    `model_checkpoint`). This is deliberate: `model.track(..., persist=True)`
    maintains persistent tracker state on the model/predictor object, and
    interleaving tracking-mode calls with `detect_ball`'s plain
    `model.predict(...)` calls on the SAME shared instance risks disturbing
    that state in ways that aren't clearly documented by the underlying
    library. Loading the checkpoint twice costs a small, one-time amount of
    memory in exchange for that isolation guarantee.
    """

    def __init__(
        self,
        homography_matrix: np.ndarray | None = None,
        team_classify_interval: int = 30,
        model_checkpoint: str = "yolov8m.pt",
        stale_gap_frames_threshold: int = DEFAULT_STALE_GAP_FRAMES_THRESHOLD,
    ):
        self.homography_matrix = homography_matrix
        self.team_classify_interval = team_classify_interval
        self.model_checkpoint = model_checkpoint
        self.stale_gap_frames_threshold = stale_gap_frames_threshold
        self._tracking_model = YOLO(model_checkpoint)

    def process_video(self, video_path: str, max_frames: int | None = None) -> Generator[dict, None, None]:
        """Yields one dict per successfully-processed, non-empty tactical
        frame: `{"frame_num", "timestamp_sec", "tensors", "diagnostics"}`.

        MEMORY: frames are read one at a time via `cv2.VideoCapture.read()`
        and processed immediately -- no full-video accumulation in memory,
        matching the "stream=True semantics" of Milestone 26's own video
        tracker. (`model.track()` is deliberately called on ONE FRAME AT A
        TIME here, not with `stream=True` over the whole video path, since
        that would process every frame unconditionally and defeat the
        entire compute-saving purpose of Milestone 31's shot classifier --
        see the per-frame sequence below.)

        TEAM CLASSIFICATION REFRESH: re-run only once per
        `team_classify_interval` TACTICAL frames (counting only frames that
        passed the shot-classification filter, not every raw video frame),
        plus unconditionally on the first tactical frame. Per-frame
        re-clustering would be wasted compute -- team jersey colors don't
        change frame-to-frame, so re-deriving the same clusters ~25-60
        times per second buys nothing. ACCEPTED LIMITATION, stated
        explicitly rather than silently assumed away: if lighting
        conditions shift meaningfully within a clip (cloud cover,
        floodlights switching), team classification could go briefly stale
        between refreshes -- a fixed-interval refresh cannot detect or
        react to that on its own.
        """
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            capture.release()
            raise ValueError(f"Could not read a valid frame rate from {video_path} (got {fps!r}).")

        # Per-track_id state: the FRAME INDEX (not just position) of each
        # track's last successful observation -- this is what lets the
        # true elapsed gap be computed instead of assuming "always 1 frame
        # ago" (Milestone 30's original, now-generalized assumption).
        last_observed_frame_index: dict[int, int] = {}
        last_observed_pixel_pos: dict[int, list[float]] = {}

        team_mapping: dict[int, str] = {}
        tactical_frame_count = 0
        skipped_non_tactical = 0
        frame_index = 0
        is_first_tracking_call = True

        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break

                read_ok, frame = capture.read()
                if not read_ok:
                    break

                try:
                    # (b) Shot classification FIRST -- the actual compute-
                    # saving purpose of Milestone 31: YOLO/tracking is
                    # never invoked on a non-tactical frame.
                    if not is_tactical_view(frame):
                        green_ratio, edge_density = compute_shot_features(frame)
                        print(
                            f"[CVPipeline] frame {frame_index} skipped (non-tactical): "
                            f"green_ratio={green_ratio:.4f}, edge_density={edge_density:.4f}"
                        )
                        skipped_non_tactical += 1
                        frame_index += 1
                        # A skipped frame produces NO observation -- it
                        # must NOT update last_observed_* for any track.
                        continue

                    tactical_frame_count += 1

                    # (c) YOLO + ByteTrack on THIS single frame only.
                    track_results = self._tracking_model.track(
                        source=frame,
                        tracker="bytetrack.yaml",
                        conf=DEFAULT_TRACKING_CONFIDENCE_THRESHOLD,
                        persist=not is_first_tracking_call,
                        verbose=False,
                    )
                    is_first_tracking_call = False
                    tracks_this_frame = _extract_person_tracks(track_results[0])

                    # (c)/(d) Real elapsed-gap computation per track, and
                    # the staleness cutoff -- via the standalone, unit-
                    # tested compute_track_dt_seconds.
                    prev_positions_pixel: dict[int, list[float]] = {}
                    dt_seconds_per_track: dict[int, float] = {}
                    stale_velocity_fallback_count = 0
                    for track in tracks_this_frame:
                        track_id = track["track_id"]
                        dt_seconds, is_stale = compute_track_dt_seconds(
                            last_observed_frame_index.get(track_id),
                            frame_index,
                            fps,
                            self.stale_gap_frames_threshold,
                        )
                        if is_stale:
                            stale_velocity_fallback_count += 1
                        elif dt_seconds is not None:
                            prev_positions_pixel[track_id] = last_observed_pixel_pos[track_id]
                            dt_seconds_per_track[track_id] = dt_seconds
                        # else: dt_seconds is None and not stale -- track's
                        # first-ever observation, nothing to add.

                    # (e) Ball detection -- may genuinely return None.
                    ball_result = detect_ball(
                        frame,
                        confidence_threshold=DEFAULT_BALL_CONFIDENCE_THRESHOLD,
                        model_checkpoint=self.model_checkpoint,
                    )
                    ball_pixel = ball_result["ball_pos_pixels"] if ball_result is not None else None

                    # (f) Team classification refresh -- interval-based,
                    # counted in TACTICAL frames only.
                    team_mapping_refreshed = False
                    should_refresh = tactical_frame_count == 1 or (
                        tactical_frame_count % self.team_classify_interval == 0
                    )
                    if should_refresh and len(tracks_this_frame) >= MIN_PLAYERS_FOR_TEAM_CLASSIFICATION:
                        players_data = [
                            {"track_id": t["track_id"], "color": extract_jersey_color(frame, t["bbox"])}
                            for t in tracks_this_frame
                        ]
                        team_mapping = classify_teams(players_data)
                        team_mapping_refreshed = True

                    # (g)/(h) Calibration + adapter (or the documented
                    # pixel-space fallback if no homography was provided).
                    if self.homography_matrix is not None:
                        tensors = convert_frame_to_tensors(
                            tracks_this_frame,
                            ball_pixel,
                            team_mapping,
                            self.homography_matrix,
                            fps,
                            prev_positions_pixel=prev_positions_pixel,
                            dt_seconds_per_track=dt_seconds_per_track,
                        )
                    else:
                        tensors = _build_pixel_space_tensors(
                            tracks_this_frame,
                            ball_pixel,
                            team_mapping,
                            fps,
                            prev_positions_pixel,
                            dt_seconds_per_track,
                        )

                    # (i) Update last-observed state for every track seen
                    # THIS frame, regardless of whether this frame ends up
                    # yielded (a track's position is still real evidence
                    # even if, say, the ball wasn't found this frame).
                    for track in tracks_this_frame:
                        last_observed_frame_index[track["track_id"]] = frame_index
                        last_observed_pixel_pos[track["track_id"]] = track["pos_pixel"]

                    # (h) Graceful degradation: missing ball, or zero
                    # players surviving outlier/bounds filtering -> yield
                    # nothing for this frame, not a crash or malformed
                    # empty-but-present bundle.
                    if tensors is None or tensors["player_pos"].shape[0] == 0:
                        frame_index += 1
                        continue

                    yield {
                        "frame_num": frame_index,
                        "timestamp_sec": frame_index / fps,
                        "tensors": tensors,
                        "diagnostics": {
                            "skipped_non_tactical": skipped_non_tactical,
                            "tracked_players": len(tracks_this_frame),
                            "ball_detected": ball_pixel is not None,
                            "team_mapping_refreshed": team_mapping_refreshed,
                            "stale_velocity_fallback_count": stale_velocity_fallback_count,
                        },
                        # ADDITIVE (Milestone 41): raw PIXEL-SPACE per-track
                        # data this loop already computed for itself
                        # (tracking/team-classification/ball-detection),
                        # exposed here so a renderer can draw real boxes/
                        # team-color/track_id/ball/possession without a
                        # second, standalone detection pass -- no new
                        # computation, no change to any existing key above,
                        # not consumed by `tensors`/`diagnostics` at all.
                        "render_frame_data": {
                            "tracks": tracks_this_frame,
                            "ball_pixel": ball_pixel,
                            "team_mapping": dict(team_mapping),
                        },
                    }

                except Exception as exc:
                    # (Step 1.5) Error resilience: one bad frame must never
                    # crash the whole run.
                    print(f"[CVPipeline] error processing frame {frame_index}: {exc!r} -- skipping frame.")

                frame_index += 1
        finally:
            capture.release()
