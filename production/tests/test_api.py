"""Milestone 16 validation: FastAPI WebSocket live tactical stream.

TestClient drives the ASGI app in-process (no real network/uvicorn
process) with delay=0.0 for fast execution, against the same cached real
match (3857276) used throughout Milestones 13-15 -- consistent with this
project's preference for testing against real data over synthetic
fixtures.
"""

import json
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
from production.src.pipeline.habit_memory import GRID_COLS, GRID_ROWS
from production.src.serving import alert_store
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

    Patches the CALLABLE `api_module.generate_tactical_explanation_with_source`
    refers to (ADR-019: `_run_alert_pipeline` switched to this from the
    plain `generate_tactical_explanation` so it can also learn/log which
    executor ran -- this fixture patches whichever name `api.py` actually
    calls today, kept in sync with that switch), not the `GEMINI_API_KEY`
    environment variable -- an earlier version of this fixture used
    `monkeypatch.delenv`, which has a real race: the spike-alert pipeline
    runs via `asyncio.create_task` (fire-and-forget), so its own
    `os.environ.get(...)` check can execute after this function-scoped
    fixture has already torn down and restored the key. Patching the
    module-level name directly is synchronous and immediate, with no such
    window.

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

    async def _mock_generate_explanation_with_source(prompt: str) -> tuple[str, str]:
        return await _mock_generate_explanation(prompt), "mock"

    monkeypatch.setattr(
        api_module,
        "generate_tactical_explanation_with_source",
        _mock_generate_explanation_with_source,
    )


@pytest.fixture
def isolated_alert_db(tmp_path, monkeypatch):
    """ADR-019: isolates the alert-history SQLite file into pytest's
    `tmp_path` for tests that need to inspect real persisted rows, so they
    never touch the real `data/app_state/alerts.db` a running server would
    use. `api_module.log_alert`/`api_module.fetch_alerts` are the SAME
    function objects `alert_store.py` defines (imported via `from ...
    import log_alert`), and those functions resolve `DB_DIR`/`DB_PATH` via
    their own module's globals at call time -- so patching
    `alert_store.DB_DIR`/`DB_PATH` here correctly redirects them regardless
    of which module's name is used to invoke them."""
    monkeypatch.setattr(alert_store, "DB_DIR", tmp_path / "app_state")
    monkeypatch.setattr(alert_store, "DB_PATH", tmp_path / "app_state" / "alerts.db")


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


# ============================================================================
# ADR-018 reporting endpoints: thin wrappers over player_report.py/
# team_report.py/team_comparison.py's existing, unmodified functions. Same
# real-cached-data discipline as test_reporting.py/test_team_comparison.py
# (MESSI_PLAYER_ID/ARGENTINA_MATCH_IDS/Real Madrid-Barcelona are the exact
# same values those files already use, so no new network fetch is needed
# here) -- these tests exist to confirm the HTTP boundary itself (status
# code, JSON shape, and that every caveat/reliability field survives the
# move to a JSON response unchanged), not to re-validate the underlying
# report-generation logic a second time.
# ============================================================================

MESSI_PLAYER_ID = 5503
ARGENTINA_MATCH_IDS = [3857264, 3857289, 3857300, 3869151]


def test_reports_player_endpoint_returns_real_report_with_caveat_fields():
    with TestClient(app) as client:
        response = client.get(
            f"/reports/player/{MESSI_PLAYER_ID}",
            params={"match_ids": ARGENTINA_MATCH_IDS},
        )
    assert response.status_code == 200
    report = response.json()

    assert report["player_id"] == MESSI_PLAYER_ID
    assert report["matches_requested"] == len(ARGENTINA_MATCH_IDS)
    # Caveat/transparency fields (Milestone 44) must survive the HTTP
    # boundary exactly, not get silently dropped by JSON serialization.
    assert "heatmap_used_uniform_fallback" in report
    assert "heatmap_event_count" in report
    assert "positional_distribution_event_count" in report
    assert report["heatmap_used_uniform_fallback"] is False  # Messi is well-supported


def test_reports_player_endpoint_low_sample_caveat_survives_http_real_data():
    """Yu-Min Cho (1 real tagged event) -- the exact low-sample case
    dashboard.py's own Player Reports tab test guards -- must still report
    itself as a uniform fallback after going through the HTTP boundary."""
    with TestClient(app) as client:
        response = client.get("/reports/player/99479", params={"match_ids": [3857262]})
    assert response.status_code == 200
    report = response.json()
    assert report["heatmap_used_uniform_fallback"] is True


def test_reports_team_endpoint_returns_real_report_with_shape():
    with TestClient(app) as client:
        response = client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS})
    assert response.status_code == 200
    report = response.json()

    assert report["team_name"] == "Argentina"
    assert report["matches_used"] == len(ARGENTINA_MATCH_IDS)
    assert "control_heatmap_grid" in report
    assert "threat_by_pitch_zone" in report
    assert "weakest_control_zones" in report


def test_reports_team_endpoint_zero_usable_matches_returns_clean_response_not_raw_422():
    """Real reproduction case (not synthetic): Atlético Madrid has 32
    real cached matches across every season, but ZERO with 360 coverage
    -- confirmed directly via a full candidate_index.py scan before this
    test was written. Selecting this team/season in dashboard.py could
    previously reach this endpoint with an empty match_ids selection,
    which `requests` sends as no `match_ids` param at all -- and with the
    old `Query(...)` (required), that raised FastAPI's own raw,
    caller-unfriendly 422
    (`{"detail":[{"type":"missing","loc":["query","match_ids"],...}]}`),
    reproduced directly against a real running app before this fix. Must
    now return a clean 200 with an explicit no_data/reason, not a 422 of
    any kind.
    """
    with TestClient(app) as client:
        response = client.get("/reports/team/Atlético Madrid", params={})
    assert response.status_code == 200
    report = response.json()

    assert report["no_data"] is True
    assert report["reason"]
    assert report["team_name"] == "Atlético Madrid"
    assert report["matches_used"] == 0
    assert report["matches_requested"] == 0
    # The underlying generate_team_report shape must still be present and
    # well-formed (not a stub/placeholder dict) -- the fix only adds
    # no_data/reason on top of its real, already-graceful empty-input
    # output, it does not replace that output with something different.
    assert "control_heatmap_grid" in report
    assert "threat_by_pitch_zone" in report
    assert "weakest_control_zones" in report
    assert report["weakest_control_zones"] == []


def test_reports_team_endpoint_no_data_flag_absent_when_matches_provided():
    """Control case: a normal, non-empty match_ids request must NOT carry
    no_data/reason at all -- confirms the fix is additive only for the
    empty-input case, not a change to every response's shape."""
    with TestClient(app) as client:
        response = client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS})
    assert response.status_code == 200
    report = response.json()
    assert "no_data" not in report
    assert "reason" not in report


def test_reports_team_comparison_endpoint_reliability_caveat_survives_http_real_data():
    """Real Madrid 2016 vs. Barcelona 2008 -- the exact reliability-caveat
    case dashboard.py's own Team Comparison tab test guards -- must still
    render as a populated, non-null field after going through the HTTP
    boundary, not silently dropped."""
    with TestClient(app) as client:
        response = client.get(
            "/reports/team-comparison",
            params={"team_a": "Real Madrid", "season_a": 2016, "team_b": "Barcelona", "season_b": 2008},
        )
    assert response.status_code == 200
    comparison = response.json()

    assert comparison["reliability_caveat"] is not None
    assert "Real Madrid 2016" in comparison["reliability_caveat"]
    assert "NOT equally reliable" in comparison["reliability_caveat"]


# ============================================================================
# ADR-021 condition-2 compliance fix: PUBLIC_DEPLOYMENT gates which variant
# of the shot map /reports/player/{player_id}/shot-map serves. Default
# (flag unset/False, module-level api_module.PUBLIC_DEPLOYMENT) must be
# byte-for-byte identical to this endpoint's pre-fix behavior -- real
# per-shot data. monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
# flips the module-level flag the endpoint function reads at CALL time
# (Python late-binding on a module global), the same established pattern
# this file already uses for CVPipeline/log_alert overrides above -- no
# real env var or process restart needed to exercise the True path.
# ============================================================================


def test_shot_map_endpoint_default_serves_raw_per_shot_data():
    """Flag unset (the default): unchanged from before this fix existed --
    real per-shot locations/xG, exactly what generate_player_shot_map
    always returned."""
    with TestClient(app) as client:
        response = client.get(
            f"/reports/player/{MESSI_PLAYER_ID}/shot-map",
            params={"match_ids": ARGENTINA_MATCH_IDS},
        )
    assert response.status_code == 200
    shot_map = response.json()

    assert "shots" in shot_map
    assert shot_map["total_shots"] > 0
    assert len(shot_map["shots"]) == shot_map["total_shots"]
    first_shot = shot_map["shots"][0]
    assert "location" in first_shot and len(first_shot["location"]) == 2
    assert "statsbomb_xg" in first_shot
    # The aggregated-only fields must NOT appear on the raw variant --
    # confirms this really is the pre-fix response shape, not a merged one.
    assert "shot_density_grid" not in shot_map
    assert "mean_xg_grid" not in shot_map


def test_shot_map_endpoint_public_deployment_serves_aggregated_only_no_raw_leak(monkeypatch):
    """The actual compliance guarantee, checked against the REAL raw HTTP
    response body (json.dumps of the actual response, not just the
    intended code path) -- with PUBLIC_DEPLOYMENT=True, no per-shot
    location/xG value may appear ANYWHERE in the response, under any key
    name, at any nesting depth."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)

    with TestClient(app) as client:
        response = client.get(
            f"/reports/player/{MESSI_PLAYER_ID}/shot-map",
            params={"match_ids": ARGENTINA_MATCH_IDS},
        )
    assert response.status_code == 200
    raw_body_text = response.text  # the actual bytes-over-the-wire, not a re-serialization
    shot_map = json.loads(raw_body_text)

    assert "shots" not in shot_map
    assert "shot_density_grid" in shot_map
    assert "mean_xg_grid" in shot_map
    assert shot_map["total_shots"] > 0  # aggregate scalars are still real, not stubbed

    # Grid shape sanity: same GRID_COLS x GRID_ROWS convention as
    # habit_memory's positional heatmap.
    assert len(shot_map["shot_density_grid"]) == GRID_COLS
    assert len(shot_map["shot_density_grid"][0]) == GRID_ROWS
    assert abs(sum(sum(col) for col in shot_map["shot_density_grid"]) - 1.0) < 1e-9

    # Belt-and-suspenders on the RAW TEXT itself: the literal substring
    # '"shots"' (the raw-variant field name) must not appear anywhere in
    # the actual response body, not just be absent from the parsed dict's
    # top level (guards against, e.g., a future accidental nesting of the
    # raw list under a different key).
    assert '"shots"' not in raw_body_text


def test_shot_map_endpoint_public_deployment_off_by_default_confirmed(monkeypatch):
    """Explicit control: confirms api_module.PUBLIC_DEPLOYMENT's real,
    unpatched module value is False (i.e. the default, unset
    PUBLIC_DEPLOYMENT env var genuinely resolves to the private/raw
    behavior) -- this is what every OTHER test in this file relies on
    implicitly by never patching this flag."""
    assert api_module.PUBLIC_DEPLOYMENT is False


# ============================================================================
# ADR-021 condition-2 compliance (Step 0 decision, see that ADR's own
# addendum): PUBLIC_DEPLOYMENT gates which variant of the pass network
# /reports/pass-network/{match_id} serves -- SAME pattern as the shot map
# tests above, applied here. Default (flag unset/False) must be byte-for-
# byte identical to this endpoint's real per-player location/edge data.
# ============================================================================


def test_pass_network_endpoint_default_serves_raw_network_real_data():
    """Flag unset (the default): real per-player average location and
    real pairwise completed-pass edge weights for match 3857276 (Canada
    vs. Morocco, 22 real Starting XI players -- verified directly against
    this match's own cached event data before writing this test)."""
    with TestClient(app) as client:
        response = client.get(f"/reports/pass-network/{MATCH_ID}")
    assert response.status_code == 200
    pass_network = response.json()

    assert pass_network["no_data"] is False
    assert "nodes" in pass_network and "edges" in pass_network
    assert len(pass_network["nodes"]) == 22  # 11 a side, both Starting XIs
    assert len(pass_network["edges"]) > 0
    first_node = pass_network["nodes"][0]
    assert "avg_location" in first_node and len(first_node["avg_location"]) == 2
    assert "name" in first_node and "team" in first_node
    first_edge = pass_network["edges"][0]
    assert first_edge["completed_passes"] > 0
    # The aggregated-only fields must NOT appear on the raw variant.
    assert "player_summary" not in pass_network
    assert "network_density" not in pass_network


def test_pass_network_endpoint_public_deployment_serves_aggregated_only_no_raw_leak(monkeypatch):
    """The actual compliance guarantee, checked against the REAL raw HTTP
    response body (same discipline as the shot map's own equivalent test
    above): with PUBLIC_DEPLOYMENT=True, no player's individual average
    location and no pairwise edge weight may appear ANYWHERE in the
    response, under any key name, at any nesting depth."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)

    with TestClient(app) as client:
        response = client.get(f"/reports/pass-network/{MATCH_ID}")
    assert response.status_code == 200
    raw_body_text = response.text  # the actual bytes-over-the-wire, not a re-serialization
    pass_network = json.loads(raw_body_text)

    assert "nodes" not in pass_network
    assert "edges" not in pass_network
    assert "player_summary" in pass_network
    assert pass_network["num_players"] == 22
    assert pass_network["num_edges"] > 0
    assert pass_network["total_completed_passes_in_network"] > 0

    first_summary = pass_network["player_summary"][0]
    assert "completed_passes_sent" in first_summary
    assert "avg_location" not in first_summary

    # Belt-and-suspenders on the RAW TEXT itself: neither the raw-variant
    # field names nor the literal string "avg_location" may appear
    # anywhere in the actual response body.
    assert '"nodes"' not in raw_body_text
    assert '"edges"' not in raw_body_text
    assert "avg_location" not in raw_body_text


def test_pass_network_endpoint_public_deployment_off_by_default_confirmed():
    """Explicit control, same convention as the shot map's own equivalent
    test -- confirms the real, unpatched default."""
    assert api_module.PUBLIC_DEPLOYMENT is False


def test_pass_network_endpoint_no_data_for_unfetchable_match():
    """A match_id with no fetchable event data (neither cached nor a real
    StatsBomb open-data match) must return a clean no_data response, not a
    500 or a crash -- mirrors generate_pass_network's own documented
    no_data convention."""
    with TestClient(app) as client:
        response = client.get("/reports/pass-network/1")
    assert response.status_code == 200
    pass_network = response.json()
    assert pass_network["no_data"] is True
    assert "nodes" not in pass_network


# ============================================================================
# GET /reports/match/{match_id} (new reporting track, Part A -- Automatic
# Match Report). This file's own `_force_mock_explanation_executor` autouse
# fixture (top of file) forces the mock explanation executor for every test
# here, same as every other endpoint touching `generate_tactical_explanation_
# with_source` -- the real-Gemini honesty check for THIS endpoint's narrative
# is `test_match_report.py::test_automatic_match_report_real_gemini_narrative_
# passes_honesty_check`'s own dedicated, opt-in job, not this file's.
# ============================================================================


def test_automatic_match_report_endpoint_real_match_both_sides():
    """Real match, real 360 coverage on both sides -- confirms the
    endpoint wires generate_automatic_match_report + the narrative step
    together correctly and returns 200 with the expected compiled shape."""
    with TestClient(app) as client:
        response = client.get(f"/reports/match/{MATCH_ID}")
    assert response.status_code == 200
    report = response.json()

    assert report["no_data"] is False
    assert len(report["teams"]) == 2
    assert set(report["team_reports"].keys()) == set(report["teams"])
    assert set(report["opposition_analysis"].keys()) == set(report["teams"])
    assert report["pass_network"]["match_id"] == MATCH_ID
    assert isinstance(report["alert_count"], int)
    # Narrative step: the mock executor (forced by this file's autouse
    # fixture) still returns SOME non-empty templated string -- never a
    # crash or an empty body -- even though this prompt's shape doesn't
    # match the mock's own regex structure (see match_report.py's
    # build_match_report_narrative_prompt docstring for why that's an
    # accepted, explicitly-stated tradeoff).
    assert report["narrative_source"] == "mock"
    assert isinstance(report["narrative"], str) and report["narrative"].strip()


def test_automatic_match_report_endpoint_public_deployment_uses_aggregated_pass_network(monkeypatch):
    """ADR-021 condition-2 compliance: under PUBLIC_DEPLOYMENT, the compiled
    report's pass_network sub-document must be the aggregated (no raw
    nodes/edges/avg_location) variant, mirroring the standalone
    /reports/pass-network/{match_id} endpoint's own real, already-tested
    gating behavior exactly."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    with TestClient(app) as client:
        response = client.get(f"/reports/match/{MATCH_ID}")
    assert response.status_code == 200
    raw_body_text = response.text
    report = response.json()

    assert "nodes" not in report["pass_network"]
    assert "edges" not in report["pass_network"]
    assert "num_players" in report["pass_network"]
    assert "avg_location" not in raw_body_text


def test_automatic_match_report_endpoint_no_data_for_unfetchable_match():
    """A match_id with fewer than 2 real teams found must return a clean
    no_data response (no narrative attempted), not a 500 or a crash."""
    with TestClient(app) as client:
        response = client.get("/reports/match/1")
    assert response.status_code == 200
    report = response.json()
    assert report["no_data"] is True
    assert "narrative" not in report


# ============================================================================
# Player Dashboard (additive new feature, on top of Milestone 40's Player
# Report): match-level views. Match Summary is unconditionally aggregate
# (ADR-021: NOT gated by PUBLIC_DEPLOYMENT). Touch Map and Timeline follow
# the SAME shot-map/pass-network gating pattern, applied here.
# ============================================================================

PLAYER_DASHBOARD_MATCH_ID = 3857264  # Messi, Argentina vs. Poland -- 272 real tagged events


def test_player_match_summary_endpoint_real_data_not_gated():
    """Unconditionally aggregate -- no PUBLIC_DEPLOYMENT check needed or
    applied; real per-match totals for Messi's Argentina/Poland match."""
    with TestClient(app) as client:
        response = client.get(
            f"/reports/player/{MESSI_PLAYER_ID}/match-summary",
            params={"match_ids": ARGENTINA_MATCH_IDS},
        )
    assert response.status_code == 200
    summary = response.json()
    assert summary["matches_player_appeared_in"] == len(ARGENTINA_MATCH_IDS)
    match = next(m for m in summary["matches"] if m["match_id"] == PLAYER_DASHBOARD_MATCH_ID)
    assert match["event_type_counts"]["Pass"] == 70


def test_player_match_touch_map_endpoint_default_serves_raw_real_data():
    """Flag unset (the default): real per-touch locations for match
    3857264 (Argentina vs. Poland, 272 real touches)."""
    with TestClient(app) as client:
        response = client.get(f"/reports/player/{MESSI_PLAYER_ID}/match/{PLAYER_DASHBOARD_MATCH_ID}/touch-map")
    assert response.status_code == 200
    touch_map = response.json()
    assert touch_map["total_touches"] == 272
    assert len(touch_map["touches"]) == 272
    assert "touch_density_grid" not in touch_map


def test_player_match_touch_map_endpoint_public_deployment_serves_aggregated_only_no_raw_leak(monkeypatch):
    """Same raw-HTTP-text-search discipline as the shot map's/pass
    network's own equivalent tests: with PUBLIC_DEPLOYMENT=True, no
    individual touch location may appear ANYWHERE in the response."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    with TestClient(app) as client:
        response = client.get(f"/reports/player/{MESSI_PLAYER_ID}/match/{PLAYER_DASHBOARD_MATCH_ID}/touch-map")
    assert response.status_code == 200
    raw_body_text = response.text
    touch_map = json.loads(raw_body_text)

    assert "touches" not in touch_map
    assert "touch_density_grid" in touch_map
    assert touch_map["total_touches"] == 272
    assert '"touches"' not in raw_body_text


def test_player_match_timeline_endpoint_default_serves_raw_real_data():
    with TestClient(app) as client:
        response = client.get(f"/reports/player/{MESSI_PLAYER_ID}/match/{PLAYER_DASHBOARD_MATCH_ID}/timeline")
    assert response.status_code == 200
    timeline = response.json()
    assert timeline["total_events"] == 272
    assert len(timeline["timeline"]) == 272
    assert "buckets" not in timeline


def test_player_match_timeline_endpoint_public_deployment_serves_aggregated_only_no_raw_leak(monkeypatch):
    """Same discipline as the touch-map test above: with
    PUBLIC_DEPLOYMENT=True, no individually-enumerated event (exact
    minute, outcome, body part) may appear ANYWHERE in the response --
    condition 2 bans "an interactive table of raw events" even without a
    location field, so this checks for the RAW SHAPE, not just a
    location-specific substring."""
    monkeypatch.setattr(api_module, "PUBLIC_DEPLOYMENT", True)
    with TestClient(app) as client:
        response = client.get(f"/reports/player/{MESSI_PLAYER_ID}/match/{PLAYER_DASHBOARD_MATCH_ID}/timeline")
    assert response.status_code == 200
    raw_body_text = response.text
    timeline = json.loads(raw_body_text)

    assert "timeline" not in timeline
    assert "buckets" in timeline
    assert timeline["total_events"] == 272
    assert '"timeline"' not in raw_body_text
    # No real Shot freeze_frame data either -- the aggregated path never
    # even calls _event_detail, but this is a direct, belt-and-suspenders
    # confirmation against the actual wire bytes.
    assert "freeze_frame" not in raw_body_text


def test_player_dashboard_endpoints_public_deployment_off_by_default_confirmed():
    """Explicit control, same convention as the shot map's/pass network's
    own equivalent tests."""
    assert api_module.PUBLIC_DEPLOYMENT is False


def test_player_match_touch_map_endpoint_no_data_for_unfetchable_match():
    """generate_player_match_touch_map's own no_data shape includes an
    EMPTY `touches: []` (a deliberate, consistently-typed response, unlike
    the pass network's no_data shape which omits `nodes` entirely) --
    empty, not omitted, still carries zero individual locations either
    way."""
    with TestClient(app) as client:
        response = client.get(f"/reports/player/{MESSI_PLAYER_ID}/match/999999999/touch-map")
    assert response.status_code == 200
    touch_map = response.json()
    assert touch_map["no_data"] is True
    assert touch_map["touches"] == []


# ============================================================================
# ADR-019 (Stage 2 persistence): the alert-history store, exercised through
# the REAL WebSocket alert flow (not alert_store.py directly -- that's
# test_alert_store.py's job). These tests confirm the two things ADR-019
# actually promises at the api.py integration level: persistence never
# delays/blocks the real-time alert, and a real alert that fires is
# actually recorded and retrievable via /alerts/history.
# ============================================================================


def test_real_alert_is_persisted_and_retrievable_via_history_endpoint(isolated_alert_db):
    """Forces spike_threshold=0.0 (any strict frame-to-frame increase in
    threat_15s fires an alert -- same technique
    test_per_connection_spike_state_is_isolated already uses) over enough
    real messages to reliably get at least one alert, then confirms it
    shows up, with correct field values, via GET /alerts/history."""
    with TestClient(app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0&spike_threshold=0.0"
        ) as websocket:
            messages = [websocket.receive_json() for _ in range(30)]

        # Let the fire-and-forget logging task(s) finish before checking --
        # they run via asyncio.create_task, not awaited by the send path.
        time.sleep(1.0)

        alert_messages = [m for m in messages if m["type"] == "alert"]
        assert alert_messages, "expected at least one alert across 30 messages at spike_threshold=0.0"

        response = client.get(f"/alerts/history?match_id={MATCH_ID}&source=statsbomb")
        assert response.status_code == 200
        rows = response.json()

    assert len(rows) == len(alert_messages), (
        "every alert message actually sent to the client must have exactly one matching persisted row"
    )
    persisted_texts = {row["explanation_text"] for row in rows}
    sent_texts = {m["explanation"] for m in alert_messages}
    assert persisted_texts == sent_texts, "every persisted alert's text must match one actually sent to the client"

    row = rows[0]
    assert row["source"] == "statsbomb"
    assert row["match_id"] == MATCH_ID
    assert row["video_path"] is None
    assert row["explanation_source"] == "mock"  # _force_mock_explanation_executor forces this
    assert row["delta"] == pytest.approx(row["threat_after"] - row["threat_before"])


def test_persistence_failure_does_not_block_or_delay_real_alert(monkeypatch):
    """ADR-019's central safety guarantee, proven directly: even if
    log_alert() fails on every call (simulated here by monkeypatching
    api_module.log_alert to raise), the real-time WebSocket alert must
    still reach the client, unaffected. log_alert itself already never
    raises in real use (it catches everything internally -- see
    test_alert_store.py) -- this test goes one step further and proves
    that even if something upstream of that guarantee somehow broke, the
    alert pipeline's own asyncio.create_task-based fire-and-forget call
    still isolates the real send from it, since the task is never awaited
    on that path.
    """

    def _raising_log_alert(**kwargs):
        raise RuntimeError("simulated persistence failure")

    monkeypatch.setattr(api_module, "log_alert", _raising_log_alert)

    with TestClient(app) as client:
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0&spike_threshold=0.0"
        ) as websocket:
            messages = [websocket.receive_json() for _ in range(30)]
            for message in messages:
                _assert_valid_message(message)

    alert_messages = [m for m in messages if m["type"] == "alert"]
    assert alert_messages, (
        "expected at least one alert -- and, critically, it must have arrived "
        "successfully despite log_alert raising on every call"
    )


def test_alerts_history_endpoint_filters_by_source_and_match_id(isolated_alert_db):
    """Directly exercises GET /alerts/history's filters against real
    persisted rows (via alert_store.log_alert, not the full WebSocket
    flow -- test_real_alert_is_persisted_... above already covers that
    end-to-end path; this test isolates the endpoint's own filter logic)."""
    alert_store.log_alert(
        source="statsbomb", match_id=MATCH_ID, video_path=None, minute=10.0,
        threat_before=0.05, threat_after=0.11, explanation_text="stats alert", explanation_source="mock",
    )
    alert_store.log_alert(
        source="cv", match_id=None, video_path="data/raw/some_clip.mp4", minute=3.0,
        threat_before=0.05, threat_after=0.12, explanation_text="cv alert", explanation_source="gemini",
    )

    with TestClient(app) as client:
        by_match = client.get(f"/alerts/history?match_id={MATCH_ID}").json()
        by_source = client.get("/alerts/history?source=cv").json()
        all_rows = client.get("/alerts/history").json()

    assert [r["explanation_text"] for r in by_match] == ["stats alert"]
    assert [r["explanation_text"] for r in by_source] == ["cv alert"]
    assert len(all_rows) == 2


# ============================================================================
# ADR-022: the single, optional API-key check. `api_module.API_KEY` is
# monkeypatched directly (the same established pattern this file already
# uses for CVPipeline/log_alert overrides) rather than via a real env var
# + process restart, since `API_KEY` is a plain module-level global read
# at call time by `_require_api_key`, not cached anywhere.
# ============================================================================


def test_api_key_unset_by_default_protected_endpoint_works_with_no_header():
    """The real, unmodified default this project's entire existing test
    suite already depends on: api_module.API_KEY is never set anywhere in
    production/tests/, so every existing test -- and any real local dev
    session -- must keep working with zero friction, no header at all."""
    assert api_module.API_KEY is None
    with TestClient(app) as client:
        response = client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS})
    assert response.status_code == 200


def test_api_key_set_rejects_missing_and_wrong_header_accepts_correct_one(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "test-secret-key")

    with TestClient(app) as client:
        no_header = client.get("/reports/team/Argentina", params={"match_ids": ARGENTINA_MATCH_IDS})
        wrong_header = client.get(
            "/reports/team/Argentina",
            params={"match_ids": ARGENTINA_MATCH_IDS},
            headers={"X-API-Key": "wrong-key"},
        )
        correct_header = client.get(
            "/reports/team/Argentina",
            params={"match_ids": ARGENTINA_MATCH_IDS},
            headers={"X-API-Key": "test-secret-key"},
        )

    assert no_header.status_code == 401
    assert wrong_header.status_code == 401
    assert correct_header.status_code == 200
    assert "X-API-Key" in no_header.json()["detail"] or "api" in no_header.json()["detail"].lower()


def test_api_key_set_health_endpoint_still_exempt(monkeypatch):
    """/health is the one deliberate exception -- must remain reachable
    with no header even once API_KEY is configured, so liveness probes
    never need a credential."""
    monkeypatch.setattr(api_module, "API_KEY", "test-secret-key")
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_api_key_set_websocket_rejects_missing_key_accepts_correct_one(monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "test-secret-key")

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0"):
                pass  # no X-API-Key header -- must be refused at the handshake

        # Correct key: connects and receives at least one real message.
        with client.websocket_connect(
            f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0",
            headers={"X-API-Key": "test-secret-key"},
        ) as websocket:
            message = websocket.receive_json()
            assert message["type"] == "threat"


# ============================================================================
# Fix 3: GET /health and GET /metrics.
# ============================================================================


def test_health_endpoint_returns_real_state():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["mlflow_reachable"] is True
    # A real MLflow run_id, not a placeholder -- same format already
    # confirmed real elsewhere in this file (e.g. the startup print line).
    assert isinstance(body["model_run_id"], str) and len(body["model_run_id"]) > 0
    assert isinstance(body["uptime_seconds"], (int, float)) and body["uptime_seconds"] >= 0


def test_metrics_endpoint_reflects_real_alert_count(isolated_alert_db):
    """The critical "not hardcoded" check: total_alerts_logged must match
    a real, independently-queried alert_store.count_alerts() call against
    the SAME isolated db -- not a stub value -- both before and after
    logging two more real alerts."""
    with TestClient(app) as client:
        before = client.get("/metrics").json()
    assert before["total_alerts_logged"] == alert_store.count_alerts()

    alert_store.log_alert(
        source="statsbomb", match_id=MATCH_ID, video_path=None, minute=1.0,
        threat_before=0.05, threat_after=0.11, explanation_text="metrics test 1", explanation_source="mock",
    )
    alert_store.log_alert(
        source="cv", match_id=None, video_path="data/raw/x.mp4", minute=2.0,
        threat_before=0.05, threat_after=0.12, explanation_text="metrics test 2", explanation_source="mock",
    )

    with TestClient(app) as client:
        after = client.get("/metrics").json()

    assert after["total_alerts_logged"] == before["total_alerts_logged"] + 2
    assert after["total_alerts_logged"] == alert_store.count_alerts()
    assert after["active_websocket_connections"] == 0
    assert after["total_http_requests_received"] > before["total_http_requests_received"]


def test_metrics_endpoint_active_websocket_connections_reflects_real_open_connection():
    """Confirms the counter is genuinely live server state, not a static
    0 -- opens a real WebSocket connection and checks /metrics reports it
    while still open, then confirms it drops back to 0 after close."""
    with TestClient(app) as client:
        idle = client.get("/metrics").json()
        assert idle["active_websocket_connections"] == 0

        with client.websocket_connect(f"/ws/tactical-stream?match_id={MATCH_ID}&delay=0.0") as websocket:
            websocket.receive_json()  # ensure the connection is actually established/accepted
            during = client.get("/metrics").json()
            assert during["active_websocket_connections"] == 1

        after = client.get("/metrics").json()
        assert after["active_websocket_connections"] == 0
