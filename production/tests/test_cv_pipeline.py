"""Milestone 32 validation: the CV pipeline orchestrator.

THE core thing this file must prove: when Milestone 31's shot classifier
skips a run of non-tactical frames between two observations of the same
track_id, the orchestrator computes velocity using the TRUE elapsed gap
(`dt = gap_frames / fps`), never a hardcoded `1/fps`. This is tested at
two levels: (1) the extracted, pure `compute_track_dt_seconds` function in
isolation, and (2) a full end-to-end run of the REAL `CVPipeline.process_video`
generator with `cv2.VideoCapture`/YOLO mocked out (no real video file is
available in this environment, consistent with every prior CV milestone in
this project's history) -- proving the ORCHESTRATION wiring itself, not
just the isolated helper.
"""

import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from production.src.cv.pipeline import (
    CVPipeline,
    _build_pixel_space_tensors,
    compute_track_dt_seconds,
)

TEST_MATCH_VIDEO_PATH = Path("data/raw/test_match.mp4")
FPS = 25.0


# ============================================================================
# Layer 1: the pure dt/staleness function -- no video, no models.
# ============================================================================

def test_compute_track_dt_seconds_uses_true_gap_not_one_over_fps():
    """The frame-10 -> frame-16 case from Step 2.5: a 6-frame gap must
    produce dt = 6/fps, not 1/fps."""
    dt_seconds, is_stale = compute_track_dt_seconds(
        last_observed_frame_index=10, current_frame_index=16, fps=FPS, stale_gap_frames_threshold=8
    )
    assert is_stale is False
    assert dt_seconds == pytest.approx(6.0 / FPS)
    assert dt_seconds != pytest.approx(1.0 / FPS)


def test_compute_track_dt_seconds_flags_stale_beyond_threshold():
    """A 10+ frame gap, with a threshold of 8, must be flagged stale (dt=None)."""
    dt_seconds, is_stale = compute_track_dt_seconds(
        last_observed_frame_index=5, current_frame_index=16, fps=FPS, stale_gap_frames_threshold=8
    )
    assert is_stale is True
    assert dt_seconds is None


def test_compute_track_dt_seconds_first_observation_is_not_stale():
    dt_seconds, is_stale = compute_track_dt_seconds(
        last_observed_frame_index=None, current_frame_index=0, fps=FPS, stale_gap_frames_threshold=8
    )
    assert is_stale is False
    assert dt_seconds is None


# ============================================================================
# Layer 2: zero-players-after-filtering -> yields nothing (via the
# pixel-space tensor builder, exercising the exact condition
# process_video checks).
# ============================================================================

def test_all_outliers_produce_empty_tensors_matching_the_no_yield_condition():
    tracks = [
        {"track_id": 1, "pos_pixel": [50.0, 50.0], "bbox": [45.0, 45.0, 10.0, 10.0]},
        {"track_id": 2, "pos_pixel": [80.0, 60.0], "bbox": [75.0, 55.0, 10.0, 10.0]},
    ]
    team_mapping = {1: "outlier", 2: "outlier"}  # e.g. both misclassified/referees in this frame

    result = _build_pixel_space_tensors(
        tracks, ball_pixel=[60.0, 55.0], team_mapping=team_mapping, fps=FPS,
        prev_positions_pixel=None, dt_seconds_per_track={},
    )

    assert result["player_pos"].shape[0] == 0
    # This is EXACTLY the condition production.src.cv.pipeline.CVPipeline.
    # process_video checks (`tensors["player_pos"].shape[0] == 0`) to
    # decide not to yield for this frame.


# ============================================================================
# Layer 3: full end-to-end process_video, cv2/YOLO mocked (no real video
# needed) -- THE single most important test in this milestone.
# ============================================================================

class _FakeBoxes:
    def __init__(self, xyxy, cls, ids):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32) if xyxy else torch.zeros((0, 4))
        self.cls = torch.tensor(cls, dtype=torch.float32) if cls else torch.zeros((0,))
        self.id = torch.tensor(ids, dtype=torch.float32) if ids else None


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeTrackingModel:
    """Stands in for a real YOLO model's `.track()` -- returns
    pre-scripted per-call detections instead of running real inference."""

    def __init__(self, calls: list[list[tuple[int, float, float]]]):
        self._calls = calls  # each: [(track_id, x1, y1), ...] for that call
        self.call_count = 0

    def track(self, source, tracker, conf, persist, verbose):
        data = self._calls[self.call_count]
        self.call_count += 1
        if not data:
            boxes = _FakeBoxes([], [], [])
        else:
            xyxy = [[x1, y1, x1 + 10.0, y1 + 10.0] for (_tid, x1, y1) in data]
            cls = [0.0] * len(data)  # COCO_PERSON_CLASS_ID
            ids = [float(tid) for (tid, _x1, _y1) in data]
            boxes = _FakeBoxes(xyxy, cls, ids)
        return [_FakeResult(boxes)]


class _FakeVideoCapture:
    def __init__(self, num_frames: int, fps: float):
        self.num_frames = num_frames
        self.fps = fps
        self._next_index = 0

    def get(self, _prop):
        return self.fps

    def read(self):
        if self._next_index >= self.num_frames:
            return False, None
        self._next_index += 1
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        pass


def test_process_video_end_to_end_uses_true_gap_dt_not_naive_one_over_fps(monkeypatch):
    """Frame 10: track_id 1 observed at pixel (100, 100), track_id 2 at
    (200, 150). Frames 11-15: non-tactical (skipped -- YOLO/tracking never
    called). Frame 16: track_id 1 reappears at (130, 100) -- a KNOWN +30px
    x-displacement -- track_id 2 unchanged. The real elapsed gap is 6
    frames; the adapter must be given dt = 6/fps, NOT 1/fps.

    If this test used a naive 1/fps assumption instead, track 1's computed
    velocity would be 30 / (1/25) = 750.0 px/s; the correct value, using
    the true 6-frame gap, is 30 / (6/25) = 125.0 px/s. These are asserted
    to be clearly distinct so a regression to the naive method would fail
    loudly, not silently.
    """
    num_frames = 17  # frame_index 0..16
    tactical_flags = [False] * num_frames
    tactical_flags[10] = True
    tactical_flags[16] = True

    fake_capture = _FakeVideoCapture(num_frames=num_frames, fps=FPS)
    monkeypatch.setattr("production.src.cv.pipeline.cv2.VideoCapture", lambda path: fake_capture)
    monkeypatch.setattr(
        "production.src.cv.pipeline.is_tactical_view", Mock(side_effect=tactical_flags)
    )
    monkeypatch.setattr(
        "production.src.cv.pipeline.detect_ball",
        Mock(return_value={"ball_pos_pixels": [105.0, 105.0], "bbox": [100, 100, 10, 10], "confidence": 0.9}),
    )
    monkeypatch.setattr(
        "production.src.cv.pipeline.extract_jersey_color", lambda image, bbox: [0.0, 255.0, 255.0]
    )
    monkeypatch.setattr(
        "production.src.cv.pipeline.classify_teams",
        lambda players_data, random_state=42: {1: "team_A", 2: "team_B"},
    )

    fake_tracking_model = _FakeTrackingModel(
        calls=[
            [(1, 100.0, 100.0), (2, 200.0, 150.0)],  # frame 10
            [(1, 130.0, 100.0), (2, 200.0, 150.0)],  # frame 16
        ]
    )
    monkeypatch.setattr("production.src.cv.pipeline.YOLO", lambda checkpoint: fake_tracking_model)

    pipeline = CVPipeline(homography_matrix=None, stale_gap_frames_threshold=8)
    results = list(pipeline.process_video("fake_video.mp4", max_frames=num_frames))

    assert len(results) == 2, f"expected 2 yielded frames (frame 10, frame 16), got {len(results)}"

    frame_10_result, frame_16_result = results
    assert frame_10_result["frame_num"] == 10
    assert frame_16_result["frame_num"] == 16
    assert frame_16_result["diagnostics"]["stale_velocity_fallback_count"] == 0

    # Frame 10 is track 1/2's first-ever observation -- [0,0] velocity, per
    # the established "no velocity available yet" convention.
    assert torch.allclose(frame_10_result["tensors"]["player_vel"], torch.zeros(2, 2))

    # Frame 16: track order in the tensor matches tracks_this_frame order,
    # i.e. [track_id 1, track_id 2] (see _FakeTrackingModel's call data).
    velocities = frame_16_result["tensors"]["player_vel"]
    track_1_velocity = velocities[0].tolist()
    track_2_velocity = velocities[1].tolist()

    correct_dt = 6.0 / FPS
    naive_dt = 1.0 / FPS
    expected_track_1_velocity_correct = [30.0 / correct_dt, 0.0]
    expected_track_1_velocity_naive_WRONG = [30.0 / naive_dt, 0.0]

    print(f"\nFrame 16 track 1 velocity: {track_1_velocity}")
    print(f"Correct (dt=6/fps={correct_dt:.4f}s) expected: {expected_track_1_velocity_correct}")
    print(f"Naive (dt=1/fps={naive_dt:.4f}s) WOULD have given: {expected_track_1_velocity_naive_WRONG}")

    assert track_1_velocity == pytest.approx(expected_track_1_velocity_correct, abs=0.01)
    assert track_1_velocity != pytest.approx(expected_track_1_velocity_naive_WRONG, abs=1.0)
    assert track_2_velocity == pytest.approx([0.0, 0.0], abs=0.01)

    print(f"skipped_non_tactical after frame 16: {frame_16_result['diagnostics']['skipped_non_tactical']}")
    # Cumulative since the start of the video: frames 0-9 (before track 1/2's
    # first-ever observation at frame 10) PLUS frames 11-15 (the skip run
    # between the two observations) = 10 + 5 = 15.
    assert frame_16_result["diagnostics"]["skipped_non_tactical"] == 15


def test_process_video_stale_gap_falls_back_to_zero_velocity(monkeypatch):
    """Same shape as above, but track_id 1 reappears after an 11-frame gap
    with a low stale_gap_frames_threshold (5) -- must fall back to [0,0]
    velocity and increment stale_velocity_fallback_count, not compute a
    (still technically real, but explicitly distrusted) dt=11/fps
    velocity.
    """
    num_frames = 17
    tactical_flags = [False] * num_frames
    tactical_flags[2] = True
    tactical_flags[13] = True

    fake_capture = _FakeVideoCapture(num_frames=num_frames, fps=FPS)
    monkeypatch.setattr("production.src.cv.pipeline.cv2.VideoCapture", lambda path: fake_capture)
    monkeypatch.setattr("production.src.cv.pipeline.is_tactical_view", Mock(side_effect=tactical_flags))
    monkeypatch.setattr(
        "production.src.cv.pipeline.detect_ball",
        Mock(return_value={"ball_pos_pixels": [105.0, 105.0], "bbox": [100, 100, 10, 10], "confidence": 0.9}),
    )
    monkeypatch.setattr(
        "production.src.cv.pipeline.extract_jersey_color", lambda image, bbox: [0.0, 255.0, 255.0]
    )
    monkeypatch.setattr(
        "production.src.cv.pipeline.classify_teams",
        lambda players_data, random_state=42: {1: "team_A", 2: "team_B"},
    )

    fake_tracking_model = _FakeTrackingModel(
        calls=[
            [(1, 100.0, 100.0), (2, 200.0, 150.0)],  # frame 2
            [(1, 300.0, 100.0), (2, 200.0, 150.0)],  # frame 13 -- big jump, 11-frame gap
        ]
    )
    monkeypatch.setattr("production.src.cv.pipeline.YOLO", lambda checkpoint: fake_tracking_model)

    pipeline = CVPipeline(homography_matrix=None, stale_gap_frames_threshold=5)
    results = list(pipeline.process_video("fake_video.mp4", max_frames=num_frames))

    assert len(results) == 2
    frame_13_result = results[1]
    print(f"\nstale_velocity_fallback_count: {frame_13_result['diagnostics']['stale_velocity_fallback_count']}")

    # BOTH track_id 1 and track_id 2 were last observed at frame 2 and
    # reappear at frame 13 -- an 11-frame gap for both, exceeding
    # threshold=5 -- so both fall back, not just the one that visibly moved.
    assert frame_13_result["diagnostics"]["stale_velocity_fallback_count"] == 2
    track_1_velocity = frame_13_result["tensors"]["player_vel"][0].tolist()
    track_2_velocity = frame_13_result["tensors"]["player_vel"][1].tolist()
    assert track_1_velocity == pytest.approx([0.0, 0.0])
    assert track_2_velocity == pytest.approx([0.0, 0.0])


# ============================================================================
# Layer 4: real-video tests (skipped gracefully without data/raw/test_match.mp4).
# ============================================================================

def test_pipeline_on_real_video_and_throughput():
    if not TEST_MATCH_VIDEO_PATH.exists():
        pytest.skip(
            f"No local test video found at {TEST_MATCH_VIDEO_PATH}. See test_cv_tracker.py's "
            "skip message for how to obtain one (SoccerNet preferred, private local clip "
            "acceptable as a stopgap for milestone validation only)."
        )

    pipeline = CVPipeline(homography_matrix=None)

    frame_times_ms = []
    results = []
    start_all = time.perf_counter()
    frame_generator = pipeline.process_video(str(TEST_MATCH_VIDEO_PATH), max_frames=30)
    last_time = time.perf_counter()
    for result in frame_generator:
        now = time.perf_counter()
        frame_times_ms.append((now - last_time) * 1000.0)
        last_time = now
        results.append(result)
    total_wall_seconds = time.perf_counter() - start_all

    assert len(results) >= 1, "expected at least 1 valid yielded frame from the first 30 frames"

    first = results[0]
    tensors = first["tensors"]
    for key in ("player_pos", "player_vel", "is_teammate", "ball_pos"):
        assert key in tensors
    assert tensors["player_pos"].shape[0] >= 1
    assert tensors["player_pos"].shape[1] == 2

    if frame_times_ms:
        median_ms = float(np.median(frame_times_ms))
        p95_ms = float(np.percentile(frame_times_ms, 95))
        effective_fps = 1000.0 / median_ms if median_ms > 0 else float("inf")
    else:
        median_ms = p95_ms = effective_fps = float("nan")

    tactical_ratio = len(results) / 30.0

    print(f"\n=== Throughput (first 30 raw frames, {len(results)} yielded) ===")
    print(f"Median ms/yielded-frame: {median_ms:.2f}, p95: {p95_ms:.2f}, effective fps: {effective_fps:.2f}")
    print(f"Tactical/yield ratio: {tactical_ratio:.2f}")
    print(f"Total wall time for 30 raw frames: {total_wall_seconds:.2f}s")
    if effective_fps >= 25:
        print("Assessment: comfortably real-time-sustainable at typical broadcast fps on this hardware.")
    elif effective_fps >= 10:
        print("Assessment: sub-real-time on this hardware; usable for offline/batch processing, not live.")
    else:
        print("Assessment: well below real-time; significant optimization needed before live use.")
