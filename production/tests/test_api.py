"""Milestone 16 validation: FastAPI WebSocket live tactical stream.

TestClient drives the ASGI app in-process (no real network/uvicorn
process) with delay=0.0 for fast execution, against the same cached real
match (3857276) used throughout Milestones 13-15 -- consistent with this
project's preference for testing against real data over synthetic
fixtures.
"""

import os
import queue
import threading
import time
from pathlib import Path
from typing import ClassVar

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import production.src.serving.api as api_module
from production.src.models.explainer import (
    generate_explanation as _mock_generate_explanation,
)
from production.src.serving.api import app

MATCH_ID = 3857276


@pytest.fixture(autouse=True)
def _force_mock_explanation_executor(monkeypatch):
    """This file tests WebSocket/CV-pipeline plumbing (connection
    isolation, non-blocking pacing, path-safety) -- never real LLM output
    quality, which is `test_explainer.py`'s own dedicated, opt-in real-API
    test. Forces the mock executor unconditionally here (even if a real
    `GEMINI_API_KEY` happens to be present in this environment's `.env`),
    so these tests never depend on network access or silently consume
    real, rate-limited API quota as a side effect of `api.py`'s own
    real-vs-mock gating.

    Patches the CALLABLE `api_module.generate_tactical_explanation` refers
    to, not the `GEMINI_API_KEY` environment variable -- an earlier version
    of this fixture used `monkeypatch.delenv`, which has a real race: the
    spike-alert pipeline runs via `asyncio.create_task` (fire-and-forget),
    so its own `os.environ.get(...)` check can execute after this
    function-scoped fixture has already torn down and restored the key.
    Patching the module-level name directly is synchronous and immediate,
    with no such window.

    Found the hard way, not preemptively: this file's own spike-alert
    tests were making real Gemini calls whenever a key was present (an
    accidental consequence of Step 5's `generate_tactical_explanation`
    wiring, since these tests never mocked that call), intermittently
    exhausting shared quota right before
    `test_explainer_real_gemini_integration`'s own dedicated real-API test
    ran later in the same suite -- causing IT to time out on every retry
    attempt, even with backoff. Retrying harder in the other test was
    treating the symptom; this is the actual fix.
    """
    monkeypatch.setattr(api_module, "generate_tactical_explanation", _mock_generate_explanation)


def _try_receive_json(websocket, timeout: float = 2.0):
    """Attempts one more `receive_json()`, but gives up after `timeout`
    seconds instead of blocking forever. Starlette's WebSocketTestSession
    has no built-in receive timeout, so the blocking call is run on a
    background thread and joined with a deadline; if nothing arrives in
    time, the thread is abandoned (it's a daemon) and this returns None
    rather than hanging the suite when no further message is ever sent.
    """
    result: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            result.put(("ok", websocket.receive_json()))
        except Exception as exc:
            result.put(("error", exc))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        status, value = result.get(timeout=timeout)
    except queue.Empty:
        return None

    if status == "error":
        raise value
    return value


def _assert_valid_message(message: dict) -> None:
    assert "type" in message, f"message missing 'type' field: {message}"

    if message["type"] == "threat":
        assert "minute" in message, f"threat message missing 'minute': {message}"
        assert "threat_15s" in message, f"threat message missing 'threat_15s': {message}"
        threat = message["threat_15s"]
        assert isinstance(threat, float), f"threat_15s should be a float, got {type(threat)}"
        assert 0.0 <= threat <= 1.0, f"threat_15s out of [0, 1]: {threat}"
    elif message["type"] == "alert":
        assert "explanation" in message, f"alert message missing 'explanation': {message}"
        explanation = message["explanation"]
        assert isinstance(explanation, str) and len(explanation) > 0, (
            f"alert explanation should be a non-empty string, got {explanation!r}"
        )
    else:
        raise AssertionError(f"unexpected message type: {message['type']!r}")


def test_tactical_stream_yields_valid_threat_and_optional_alert_messages():
    with TestClient(app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0"
        ) as websocket:
            messages = [websocket.receive_json() for _ in range(3)]
            for message in messages:
                _assert_valid_message(message)

            threat_count = sum(1 for m in messages if m["type"] == "threat")
            alert_count = sum(1 for m in messages if m["type"] == "alert")
            print(f"\nReceived {len(messages)} messages: {threat_count} threat, {alert_count} alert")

            # Buffer for any in-flight background alert task to finish and
            # be received -- purely opportunistic: whether a spike occurs
            # within this window depends on the match's real data and is
            # NOT asserted either way (per Step 4.4).
            time.sleep(0.5)
            trailing = _try_receive_json(websocket, timeout=1.5)
            if trailing is not None:
                _assert_valid_message(trailing)
                print(f"Trailing message after buffer: {trailing}")
                if trailing["type"] == "alert":
                    print(f"Alert fired during normal-threshold streaming: {trailing['explanation']}")


def test_per_connection_spike_state_is_isolated():
    """Validates Step 3.3/3.4's per-connection (not global) state
    requirement.

    Connection A opens with an artificially low spike_threshold (0.0), so
    across a handful of real frames its previous_threat_15s is very likely
    driven upward by at least one genuine spike -- this is the Step 5.2
    "extra check" that reliably forces the alert path (and, with it, the
    connection_lock-guarded interleaved send) to actually execute.

    Connection B then opens FRESH with the normal threshold, and its very
    FIRST message is hard-asserted to be type="threat", never "alert": a
    brand-new connection's previous_threat_15s always starts at None, so
    no spike delta can be computed on message 1 -- REGARDLESS of whatever
    connection A's state looked like. If previous_threat_15s were
    accidentally module/global state instead of connection-local, A's
    activity could leak into B's very first comparison and incorrectly
    fire an alert here.
    """
    with TestClient(app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0&spike_threshold=0.0"
        ) as connection_a:
            a_messages = [connection_a.receive_json() for _ in range(5)]
            for message in a_messages:
                _assert_valid_message(message)
            a_alert_count = sum(1 for m in a_messages if m["type"] == "alert")
            print(f"\nConnection A (spike_threshold=0.0) alerts observed in first 5 messages: {a_alert_count}")

            # Let any in-flight alert task on connection A finish before
            # tearing the connection down, to avoid a noisy (harmless)
            # "send on closed connection" background-task warning.
            time.sleep(0.5)

        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0"
        ) as connection_b:
            first_message = connection_b.receive_json()
            _assert_valid_message(first_message)
            assert first_message["type"] == "threat", (
                "connection B's first message must be type='threat' -- its own "
                "previous_threat_15s must start at None regardless of connection A's prior "
                f"activity, so no spike alert can fire on message 1. Got: {first_message}"
            )

    print(
        f"\nPer-connection isolation confirmed: connection B's first message was "
        f"type={first_message['type']!r} (never an alert), independent of connection A's "
        "spike_threshold=0.0 activity on a separate connection."
    )


# ============================================================================
# Milestone 33: CV video-source WebSocket integration.
#
# Two real, already-cached JSON files (not videos) stand in as "video_path"
# values for tests that mock CVPipeline entirely (its process_video is never
# actually asked to open them as video); one of them is ALSO used, WITHOUT
# any mocking, to prove the genuine "file exists and is inside the allowed
# directory, but cv2 can't read it as a video" failure path end-to-end.
# ============================================================================

_STANDIN_VIDEO_PATH_A = "data/raw/3773386_events.json"
_STANDIN_VIDEO_PATH_B = "data/raw/3773386_360.json"


class _FakeCVPipelineForIsolationTest:
    """Records every instance created (class-level list) so a test can
    assert a FRESH instance was constructed per connection, and yields one
    frame whose ball position encodes `video_path`, so two connections
    given different paths are directly, exactly distinguishable in the
    JSON they receive -- proving no cross-connection state leakage.
    """

    instances: ClassVar[list["_FakeCVPipelineForIsolationTest"]] = []

    def __init__(self, homography_matrix=None, model_checkpoint="yolov8m.pt", **kwargs):
        self.video_path_seen = None
        _FakeCVPipelineForIsolationTest.instances.append(self)

    def process_video(self, video_path, max_frames=None):
        self.video_path_seen = video_path
        marker = float(abs(hash(video_path)) % 1000)
        yield {
            "frame_num": 0,
            "timestamp_sec": 0.0,
            "tensors": {
                "player_pos": torch.tensor([[30.0, 20.0], [70.0, 50.0]], dtype=torch.float32),
                "player_vel": torch.zeros((2, 2), dtype=torch.float32),
                "is_teammate": torch.tensor([True, False], dtype=torch.bool),
                "ball_pos": torch.tensor([marker, marker], dtype=torch.float32),
                "fatigue_mod": torch.ones((2,), dtype=torch.float32),
            },
            "diagnostics": {
                "skipped_non_tactical": 0,
                "tracked_players": 2,
                "ball_detected": True,
                "team_mapping_refreshed": True,
                "stale_velocity_fallback_count": 0,
            },
        }


def test_cv_source_per_connection_state_isolation(monkeypatch):
    """The load-bearing correctness test for this milestone: two CV-source
    connections, given DIFFERENT video paths, must each get a FRESH
    CVPipeline instance and must never see the other's data.
    """
    _FakeCVPipelineForIsolationTest.instances.clear()
    monkeypatch.setattr(api_module, "CVPipeline", _FakeCVPipelineForIsolationTest)

    with TestClient(app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?source=cv&video_path={_STANDIN_VIDEO_PATH_A}"
        ) as connection_a:
            message_a = connection_a.receive_json()

        with client.websocket_connect(
            f"/ws/tactical-stream?source=cv&video_path={_STANDIN_VIDEO_PATH_B}"
        ) as connection_b:
            message_b = connection_b.receive_json()

    assert len(_FakeCVPipelineForIsolationTest.instances) == 2, (
        "expected a FRESH CVPipeline instance per connection, got "
        f"{len(_FakeCVPipelineForIsolationTest.instances)}"
    )
    # video_path_seen is the RESOLVED absolute path (api.py resolves it for
    # the path-safety check before ever calling process_video), not the
    # raw relative string passed in the URL.
    assert _FakeCVPipelineForIsolationTest.instances[0].video_path_seen == str(Path(_STANDIN_VIDEO_PATH_A).resolve())
    assert _FakeCVPipelineForIsolationTest.instances[1].video_path_seen == str(Path(_STANDIN_VIDEO_PATH_B).resolve())

    # The fake pipeline's process_video hashes whatever path it's actually
    # called with -- the RESOLVED absolute path, same as video_path_seen above.
    expected_marker_a = float(abs(hash(str(Path(_STANDIN_VIDEO_PATH_A).resolve()))) % 1000)
    expected_marker_b = float(abs(hash(str(Path(_STANDIN_VIDEO_PATH_B).resolve()))) % 1000)

    print(f"\nConnection A ball.pos: {message_a['ball']['pos']} (expected marker {expected_marker_a})")
    print(f"Connection B ball.pos: {message_b['ball']['pos']} (expected marker {expected_marker_b})")

    assert message_a["ball"]["pos"] == pytest.approx([expected_marker_a, expected_marker_a])
    assert message_b["ball"]["pos"] == pytest.approx([expected_marker_b, expected_marker_b])
    assert message_a["ball"]["pos"] != message_b["ball"]["pos"], (
        "connection A and B received IDENTICAL ball positions -- state may be leaking between "
        "connections"
    )


class _SlowFakeCVPipeline:
    """Simulates slow, genuinely CPU-bound CV processing via a REAL
    blocking `time.sleep` inside the generator body -- if `_stream_cv_source`
    failed to offload `next()` calls to a worker thread, this sleep would
    block the entire FastAPI event loop, not just this one connection."""

    SLEEP_SECONDS = 3.0

    def __init__(self, homography_matrix=None, model_checkpoint="yolov8m.pt", **kwargs):
        pass

    def process_video(self, video_path, max_frames=None):
        time.sleep(self.SLEEP_SECONDS)  # a REAL blocking sleep, not asyncio.sleep
        yield {
            "frame_num": 0,
            "timestamp_sec": 0.0,
            "tensors": {
                "player_pos": torch.tensor([[30.0, 20.0], [70.0, 50.0]], dtype=torch.float32),
                "player_vel": torch.zeros((2, 2), dtype=torch.float32),
                "is_teammate": torch.tensor([True, False], dtype=torch.bool),
                "ball_pos": torch.tensor([50.0, 34.0], dtype=torch.float32),
                "fatigue_mod": torch.ones((2,), dtype=torch.float32),
            },
            "diagnostics": {
                "skipped_non_tactical": 0,
                "tracked_players": 2,
                "ball_detected": True,
                "team_mapping_refreshed": True,
                "stale_velocity_fallback_count": 0,
            },
        }


def test_cv_source_does_not_block_event_loop_for_other_requests(monkeypatch):
    """Proves the async fix (Step 1.3's `asyncio.to_thread(next, gen)`
    pattern) actually works: while a CV-source connection is mid-processing
    a deliberately slow (3s, genuinely blocking) frame, a SEPARATE,
    concurrent `/simulate` REST request must still complete quickly --
    a single-connection test alone cannot show this, since it wouldn't
    reveal whether the event loop itself was blocked.
    """
    monkeypatch.setattr(api_module, "CVPipeline", _SlowFakeCVPipeline)

    with TestClient(app) as client:
        ws_elapsed_holder = {}

        def _slow_ws_worker():
            start = time.monotonic()
            with client.websocket_connect(
                f"/ws/tactical-stream?source=cv&video_path={_STANDIN_VIDEO_PATH_A}"
            ) as ws:
                ws.receive_json()
            ws_elapsed_holder["seconds"] = time.monotonic() - start

        ws_thread = threading.Thread(target=_slow_ws_worker)
        ws_thread.start()
        time.sleep(0.5)  # let the WS connection start and enter the slow sleep

        get_start = time.monotonic()
        response = client.get("/simulate?match_id=3857276&minute=10&action=no_change")
        get_elapsed = time.monotonic() - get_start

        ws_thread.join(timeout=_SlowFakeCVPipeline.SLEEP_SECONDS + 10)

    print(f"\n/simulate GET completed in {get_elapsed:.2f}s while a CV stream was mid-sleep "
          f"({_SlowFakeCVPipeline.SLEEP_SECONDS}s blocking call)")
    print(f"CV WebSocket connection took {ws_elapsed_holder.get('seconds', float('nan')):.2f}s total")

    assert response.status_code == 200
    assert get_elapsed < _SlowFakeCVPipeline.SLEEP_SECONDS, (
        f"/simulate took {get_elapsed:.2f}s, close to or exceeding the CV stream's "
        f"{_SlowFakeCVPipeline.SLEEP_SECONDS}s blocking sleep -- the event loop may have been "
        "blocked by the CV connection rather than offloading it to a worker thread"
    )


def test_cv_source_rejects_path_traversal():
    # NOTE: `websocket.accept()` runs unconditionally before the path
    # check (see api.py), so the ACCEPT message is what
    # `websocket_connect()`'s `__enter__` observes -- the subsequent
    # `close()` this test is checking for only becomes visible on an
    # explicit `receive_json()` call inside the block, not from `__enter__`
    # itself. Same pattern as `test_cv_source_unreadable_file_closes_cleanly_not_a_crash`.
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/ws/tactical-stream?source=cv&video_path=../../etc/passwd"
            ) as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_cv_source_missing_video_path_param_rejected():
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/tactical-stream?source=cv") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_cv_source_nonexistent_file_within_allowed_dir_rejected():
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/ws/tactical-stream?source=cv&video_path=data/raw/does_not_exist_12345.mp4"
            ) as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_cv_source_unreadable_file_closes_cleanly_not_a_crash():
    """A REAL file that genuinely exists inside the allowed directory --
    but is not a valid video (a cached JSON file) -- passes the path-safety
    check, so the failure must surface from CVPipeline itself
    (`cv2.VideoCapture` producing an invalid fps, per Milestone 32) and be
    caught by `_stream_cv_source`'s top-level exception handler (Step 4),
    not crash the server or hang. No mocking -- this exercises the REAL
    CVPipeline/cv2 failure path end-to-end.
    """
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/ws/tactical-stream?source=cv&video_path={_STANDIN_VIDEO_PATH_A}"
            ) as ws:
                ws.receive_json()
        print(f"\nUnreadable-video close code: {exc_info.value.code}, reason: {exc_info.value.reason}")
        assert exc_info.value.code == 1011
