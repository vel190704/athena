"""Milestone 16 (Modules 3 & 9): FastAPI + WebSocket real-time serving
layer. Wraps the batch pipeline (feature extraction -> normalization ->
MLP -> cumulative incidence, Milestones 4-8) and the attribution/explainer
pipeline (Milestone 15) around the live match simulator (Milestone 16
Step 2) to stream tactical updates over a WebSocket connection.

Per README.txt Module 7's "Asynchronous Explainability" principle (ADR-006):
the real-time stream pushes raw threat scores on every frame; explanation
generation is comparatively expensive (Integrated Gradients + the mock LLM
call) and only runs in the background when a spike is detected, then pushes
its result as a separate `alert` message once ready -- never blocking the
main per-frame `threat` stream.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

# Must be set before the lifespan handler's first MlflowClient() call (this
# project's mlflow version treats the file-store backend as read-only
# "maintenance mode" otherwise). Test files set this themselves before
# importing this module, but this is the actual `uvicorn ...:app`
# entrypoint, so it needs to set it too rather than depend on a test
# harness having already done so.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

from production.src.constants import TIME_BIN
from production.src.cv.pipeline import CVPipeline
from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.models.explainer import (
    _cumulative_incidence_forward,
    build_tactical_prompt,
    compute_attributions,
    generate_tactical_explanation,
    load_deterministic_mlp,
)
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.simulator import perturb_features
from production.src.pipeline.survival_dataset import FEATURE_KEYS
from production.src.serving.simulator import live_match_stream

DEFAULT_MATCH_ID = 3857276
# TIME_BIN (15s horizon, matching Milestones 8/13/14/15) now comes from
# production.src.constants (engineering-review de-duplication -- was
# defined locally here before; value unchanged).

# Milestone 33: the CV video-source path. Only paths that resolve INSIDE
# this directory are ever opened -- a raw `video_path` query parameter is
# untrusted input, and without this check a client could request an
# arbitrary file on the server's filesystem via path traversal (e.g.
# `../../etc/passwd`). Same input-validation discipline as `/simulate`'s
# `Literal` type on `action` (Milestone 18) -- reject cleanly, don't trust.
ALLOWED_CV_VIDEO_DIRECTORY = Path("data/raw").resolve()

# The checkpoint every per-connection CVPipeline instance will load. Warmed
# once at startup (see `lifespan`) so the underlying weights file is
# already local/cached by the time any connection instantiates its own
# CVPipeline -- see that warm-up call's comment for why each connection
# still gets its OWN model object rather than sharing one.
CV_MODEL_CHECKPOINT = "yolov8m.pt"

# ABSOLUTE percentage-point threshold, not relative: cumulative incidence
# values in this project are typically small (~5-15%, per Milestone 15's
# 9.1% example), so a RELATIVE threshold (e.g. "5% bigger than before")
# would fire almost every frame. A 0.05 (5 PERCENTAGE POINT) absolute jump
# -- e.g. 0.09 -> 0.14 -- is the intended, much rarer signal.
SPIKE_THRESHOLD = 0.05

# Populated once at startup (see `lifespan` below) and reused for every
# connection -- never reloaded per request/connection.
_model: torch.nn.Module | None = None
_normalization_mean: torch.Tensor | None = None
_normalization_std: torch.Tensor | None = None
_model_run_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _normalization_mean, _normalization_std, _model_run_id
    _model, _normalization_mean, _normalization_std, _model_run_id = load_deterministic_mlp()
    print(f"[api] Live inference server ready. Loaded MLP run_id={_model_run_id}")

    # Milestone 33 Step 1.1: warm the CV YOLO checkpoint ONCE at startup --
    # heavy, stateless-once-downloaded weights should be fetched/cached
    # once, not per connection. This does NOT mean every connection shares
    # one model OBJECT for tracking: `CVPipeline` (Milestone 32) instantiates
    # its own internal YOLO model instance per construction specifically
    # because `model.track(..., persist=True)` accumulates ByteTrack
    # tracker state ON that model object across calls -- sharing one
    # instance across concurrent connections would leak track IDs between
    # them, exactly the class of bug per-connection state isolation exists
    # to prevent (see the WebSocket endpoint below). What this warm-up call
    # DOES buy: the checkpoint FILE is downloaded/deserialized once here,
    # so every later per-connection `CVPipeline()` construction is fast
    # (loading already-local weights), not a fresh cold-start each time.
    print(f"[api] Warming CV model checkpoint {CV_MODEL_CHECKPOINT} ...")
    YOLO(CV_MODEL_CHECKPOINT)
    print("[api] CV model checkpoint ready.")

    yield


app = FastAPI(lifespan=lifespan)


def _predict_cumulative_incidence_sync(features_dict: dict) -> tuple[float, torch.Tensor]:
    """Synchronous, CPU-bound: builds the normalized input tensor and runs
    the MLP forward pass. Called via `asyncio.to_thread` from the async
    endpoint so this blocking PyTorch inference never runs directly on the
    event loop.

    Returns (cumulative_incidence, normalized_input_tensor) -- the tensor
    is handed back too so a subsequent spike-triggered attribution pass
    can reuse the exact same normalized features rather than recomputing
    (and risking divergence from) them.
    """
    raw_tensor = torch.tensor(
        [[features_dict[key] for key in FEATURE_KEYS]], dtype=torch.float32
    )
    normalized_input = (raw_tensor - _normalization_mean) / _normalization_std

    with torch.no_grad():
        cumulative_incidence = _cumulative_incidence_forward(
            _model, normalized_input, time_bin=TIME_BIN
        ).item()

    return cumulative_incidence, normalized_input


def _build_alert_prompt_sync(
    features_dict: dict, normalized_input: torch.Tensor, cumulative_incidence: float
) -> str:
    """Synchronous Integrated Gradients attribution + prompt building for a
    spike alert (Milestone 15). Captum requires gradients through the
    input, so a fresh `requires_grad_(True)` copy of the normalized input
    is used here -- the copy used for the plain threat prediction above is
    run under `torch.no_grad()` and cannot be reused directly for this.
    Run via `asyncio.to_thread` (see `_run_alert_pipeline`) since IG's
    several forward/backward passes are CPU-bound work that must not block
    the event loop either.
    """
    attribution_input = normalized_input.clone().requires_grad_(True)
    baseline_tensor = torch.zeros_like(attribution_input)

    attributions = compute_attributions(
        _model, attribution_input, baseline_tensor, time_bin=TIME_BIN
    )
    return build_tactical_prompt(features_dict, attributions, cumulative_incidence, time_bin=TIME_BIN)


async def _run_alert_pipeline(
    websocket: WebSocket,
    connection_lock: asyncio.Lock,
    features_dict: dict,
    normalized_input: torch.Tensor,
    cumulative_incidence: float,
) -> None:
    """Background task (Step 3.7/3.8): computes attributions, builds the
    prompt, runs the explanation executor (Milestone 15's mock, or a real
    Gemini Flash-Lite call if GEMINI_API_KEY is set -- see ADR-006's Update
    section; the choice is entirely internal to `generate_tactical_explanation`,
    this call site does not know or need to know which one actually ran),
    then sends the alert -- guarded by the SAME connection-scoped lock the
    main loop uses for `threat` messages, so this send can never interleave
    with (and corrupt) a concurrent `threat` send on the same connection.
    """
    prompt = await asyncio.to_thread(
        _build_alert_prompt_sync, features_dict, normalized_input, cumulative_incidence
    )
    explanation = await generate_tactical_explanation(prompt)

    async with connection_lock:
        await websocket.send_json({"type": "alert", "explanation": explanation})


def _maybe_trigger_spike_alert(
    websocket: WebSocket,
    connection_lock: asyncio.Lock,
    previous_threat_15s: float | None,
    cumulative_incidence: float,
    spike_threshold: float,
    features_dict: dict,
    normalized_input: torch.Tensor,
) -> None:
    """Shared spike-detection trigger (Milestone 16), reused IDENTICALLY by
    both the StatsBomb-replay and Milestone 33 CV-video sources below --
    the detection rule and alert pipeline must not silently diverge
    between sources depending on where the `threat_15s` number came from.
    """
    spike_fired = (
        previous_threat_15s is not None
        and (cumulative_incidence - previous_threat_15s) > spike_threshold
    )
    if spike_fired:
        asyncio.create_task(
            _run_alert_pipeline(
                websocket, connection_lock, features_dict, normalized_input, cumulative_incidence
            )
        )


# WebSocket close-frame reasons are limited to ~123 UTF-8-encoded bytes by
# the protocol itself (a control-frame length limit, not a choice this
# project makes -- RFC 6455's 125-byte control-frame cap minus the 2-byte
# status code). An error message that embeds a full resolved file path (as
# this module's do) can easily exceed that, which silently turns an
# intended clean close into `websockets.exceptions.ProtocolError: control
# frame too long` instead -- found via manual testing against a real
# running server (the automated TestClient suite's in-process transport
# did not happen to trip this), not something a synthetic short-path test
# alone would have caught.
MAX_WEBSOCKET_CLOSE_REASON_BYTES = 100


def _truncate_close_reason(reason: str) -> str:
    """Encodes to bytes, truncates to `MAX_WEBSOCKET_CLOSE_REASON_BYTES`,
    and decodes back leniently (`errors="ignore"`) so a multi-byte UTF-8
    character is never split mid-sequence."""
    encoded = reason.encode("utf-8")
    if len(encoded) <= MAX_WEBSOCKET_CLOSE_REASON_BYTES:
        return reason
    return encoded[:MAX_WEBSOCKET_CLOSE_REASON_BYTES].decode("utf-8", errors="ignore") + "..."


def _resolve_and_validate_cv_video_path(video_path: str) -> Path | str:
    """Resolves `video_path` to an absolute path and verifies it lies
    within `ALLOWED_CV_VIDEO_DIRECTORY` -- guards against path traversal
    via a raw, untrusted query parameter (Milestone 33 Step 1.3). Returns
    the resolved `Path` on success, or an error message `str` on failure
    (the caller decides how to close the connection; this function never
    touches the websocket itself, keeping it independently testable).
    """
    resolved = Path(video_path).resolve()
    try:
        resolved.relative_to(ALLOWED_CV_VIDEO_DIRECTORY)
    except ValueError:
        return f"video_path must resolve inside {ALLOWED_CV_VIDEO_DIRECTORY}, got {resolved}"
    if not resolved.exists():
        return f"video_path does not exist: {resolved}"
    return resolved


async def _stream_cv_source(
    websocket: WebSocket,
    video_path: str,
    connection_lock: asyncio.Lock,
    spike_threshold: float,
) -> None:
    """Milestone 33: streams tactical updates derived from REAL CV
    processing (Milestones 25-32), instead of StatsBomb event replay.

    A FRESH `CVPipeline` is instantiated HERE, inside this one connection's
    coroutine call -- never shared across connections or module-level.
    This is mandatory: `CVPipeline` carries stateful tracking dictionaries
    (`last_observed_frame_index`, `team_mapping`) and its own internal YOLO
    model instance's ByteTrack `persist=True` state, all of which must
    never leak between two videos/connections processed concurrently --
    the exact same discipline Milestone 16 already applied to
    `previous_threat_15s`, now applied to substantially more state.

    CALIBRATION CAVEAT (read before trusting `threat_15s` from this path):
    this endpoint does not accept or wire through a real homography yet --
    automatic camera recalibration is explicitly out of scope for this
    milestone. `CVPipeline(homography_matrix=None)` therefore returns
    PIXEL-SPACE positions (Milestone 32's documented fallback), not real
    meters. `extract_features`'s physics-grid math assumes ADR-002's
    100x68 METER space, so `threat_15s` values produced via this path are
    NOT physically meaningful yet. This endpoint proves the async /
    per-connection-isolation / real-time-pacing wiring end-to-end -- this
    milestone's actual scope -- not calibrated real-world threat numbers.
    """
    pipeline = CVPipeline(homography_matrix=None, model_checkpoint=CV_MODEL_CHECKPOINT)
    frame_generator = pipeline.process_video(video_path)

    previous_threat_15s = None
    stream_start_wall_time = time.monotonic()

    while True:
        try:
            # CRITICAL: `next()` on a plain (blocking) generator is
            # offloaded to a worker thread via `asyncio.to_thread` on EVERY
            # call, not just once -- calling `pipeline.process_video(...)`
            # directly in a `for` loop here would block the ENTIRE FastAPI
            # event loop (every connection, not just this one) for the
            # whole video's processing duration.
            frame_data = await asyncio.to_thread(next, frame_generator)
        except StopIteration:
            break
        except Exception as exc:
            # Step 4: a top-level pipeline failure (video can't be opened
            # at all -- corrupt file, unsupported codec) -- distinct from
            # Milestone 32's own per-frame resilience, which already
            # handles per-frame errors internally and would never raise
            # here for those. This is a failure the generator itself could
            # not recover from.
            await websocket.close(
                code=1011, reason=_truncate_close_reason(f"CV pipeline error: {exc}")
            )
            return

        tensors = frame_data["tensors"]
        # Same validated call pattern Milestone 30 proved works: the
        # adapter's output dict feeds `extract_features` UNMODIFIED.
        features_dict = await asyncio.to_thread(extract_features, tensors)
        cumulative_incidence, normalized_input = await asyncio.to_thread(
            _predict_cumulative_incidence_sync, features_dict
        )

        # Real-time pacing (explicit, not ambiguous): compare this frame's
        # video timestamp against wall-clock time elapsed since the stream
        # started. Ahead of real-time pace -> sleep to align sends with
        # real video timing (genuinely simulating live playback). Behind
        # real-time pace (a real possibility -- Milestone 32's own
        # throughput measurements were honest that this is hardware-
        # dependent) -> do NOT force real-time pacing; stream as fast as
        # the pipeline can produce frames and report the honest lag
        # instead of silently sprinting through the match or claiming a
        # pace it can't sustain.
        elapsed_wall_seconds = time.monotonic() - stream_start_wall_time
        target_timestamp_seconds = frame_data["timestamp_sec"]
        lag_seconds = elapsed_wall_seconds - target_timestamp_seconds
        if lag_seconds < 0:
            await asyncio.sleep(-lag_seconds)
            lag_seconds = 0.0

        num_players = tensors["player_pos"].shape[0]
        players_payload = [
            {
                "pos": tensors["player_pos"][i].tolist(),
                "is_teammate": bool(tensors["is_teammate"][i].item()),
            }
            for i in range(num_players)
        ]
        ball_payload = {"pos": tensors["ball_pos"].tolist()}

        async with connection_lock:
            await websocket.send_json(
                {
                    "type": "threat",
                    "minute": int(target_timestamp_seconds // 60),
                    "threat_15s": cumulative_incidence,
                    "players": players_payload,
                    "ball": ball_payload,
                    "real_time_lag_sec": lag_seconds,
                }
            )

        # Reuse the EXACT SAME spike-detection/alert logic as the
        # StatsBomb source, regardless of where threat_15s came from.
        _maybe_trigger_spike_alert(
            websocket,
            connection_lock,
            previous_threat_15s,
            cumulative_incidence,
            spike_threshold,
            features_dict,
            normalized_input,
        )
        previous_threat_15s = cumulative_incidence


@app.websocket("/ws/tactical-stream")
async def tactical_stream(
    websocket: WebSocket,
    match_id: int = DEFAULT_MATCH_ID,
    delay: float = 1.0,
    spike_threshold: float = SPIKE_THRESHOLD,
    source: Literal["statsbomb", "cv"] = "statsbomb",
    video_path: str | None = None,
):
    """`spike_threshold` defaults to the module-level SPIKE_THRESHOLD but is
    overridable per-connection via a query param -- primarily so tests can
    force a low, reliably-triggered threshold to exercise the alert path
    (Step 5.2) without depending on a specific match happening to contain a
    genuine 5-percentage-point swing within the test window.

    `source="cv"` (Milestone 33) streams from real CV processing
    (`video_path`, required, must resolve inside `ALLOWED_CV_VIDEO_DIRECTORY`)
    instead of StatsBomb event replay. See `_stream_cv_source`'s docstring
    for the calibration caveat that currently applies to that path's
    `threat_15s` values.
    """
    await websocket.accept()

    # Connection-LOCAL state (Step 3.3, Milestone 16) -- deliberately a
    # plain local variable, not module/global state, so concurrent
    # connections each track their own previous threat value independently
    # and can never clobber each other's spike detection. The SAME
    # discipline applies to `source="cv"` below via a freshly-constructed
    # `CVPipeline` per connection, inside `_stream_cv_source`.
    connection_lock = asyncio.Lock()

    if source == "cv":
        if not video_path:
            await websocket.close(code=1008, reason="video_path is required when source=cv")
            return

        resolved = _resolve_and_validate_cv_video_path(video_path)
        if isinstance(resolved, str):  # error message, not a valid Path
            await websocket.close(code=1008, reason=_truncate_close_reason(resolved))
            return

        try:
            await _stream_cv_source(websocket, str(resolved), connection_lock, spike_threshold)
        except WebSocketDisconnect:
            pass
        return

    previous_threat_15s = None
    try:
        async for event, frame in live_match_stream(match_id, delay=delay):
            features_dict = extract_features(frame)
            cumulative_incidence, normalized_input = await asyncio.to_thread(
                _predict_cumulative_incidence_sync, features_dict
            )

            async with connection_lock:
                await websocket.send_json(
                    {
                        "type": "threat",
                        "minute": event.get("minute"),
                        "threat_15s": cumulative_incidence,
                    }
                )

            _maybe_trigger_spike_alert(
                websocket,
                connection_lock,
                previous_threat_15s,
                cumulative_incidence,
                spike_threshold,
                features_dict,
                normalized_input,
            )

            # Updated regardless of whether a spike fired (Step 3.7).
            previous_threat_15s = cumulative_incidence
    except WebSocketDisconnect:
        pass


def _find_qualifying_frame_for_minute(
    match_id: int,
    minute: int,
    *,
    period: int | None = None,
    team_id: int | None = None,
    max_minute: int | None = None,
):
    """Finds the first event, in PERIOD-AWARE order, with a `minute` value
    `>= minute` that also has an associated 360 freeze-frame.

    "Period-aware order" means: scan period 1 (in its own event order)
    first, in full, THEN scan period 2 -- never a single ordering that
    treats the raw event list as already globally minute-sorted across
    both periods. This matters because StatsBomb's raw `minute` field does
    NOT reset to 0 at half-time, but the two periods' minute RANGES still
    overlap near the interval boundary (e.g. this project's cached match
    has period 1 running 0-50, including first-half stoppage time, while
    period 2 starts back at 45) -- so a requested minute in that overlap
    (e.g. 47) is genuinely ambiguous under a naive single ordering. This
    function resolves the ambiguity by always preferring a period-1 match
    over a period-2 match at the same minute value.

    Optional keyword-only filters (added for Milestone 20's oracle
    validator; all default to the exact Milestone 18 `/simulate` behavior
    when omitted, so this remains a genuine extension, not a fork):
      - `period`: if given, ONLY that period is searched (no period-1-then-
        period-2 fallback at all). Milestone 20 always knows a
        substitution's own period up front, so there is no ambiguity left
        to resolve the way the `None` case above does -- and searching
        both periods anyway would risk a false cross-period match (a
        period-1 event satisfying `minute >= X` purely because periods 1
        and 2's raw minute ranges overlap, even though period 1 already
        ended before period 2 began in real time).
      - `team_id`: if given, only events whose acting team
        (`event["team"]["id"]`) matches are considered.
      - `max_minute`: if given, only events with `minute < max_minute` are
        considered (exclusive upper bound) -- used by the oracle validator
        to guarantee a "pre-substitution" frame search can never cross past
        the substitution's own minute and accidentally return a
        post-substitution event.

    Returns (event, frame_data), or None if no qualifying frame exists
    anywhere in the searched scope.
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    periods_to_search = (period,) if period is not None else (1, 2)

    for search_period in periods_to_search:
        for event in events:
            if event["period"] != search_period:
                continue
            if "location" not in event:
                continue
            event_minute = event.get("minute")
            if event_minute is None or event_minute < minute:
                continue
            if max_minute is not None and event_minute >= max_minute:
                continue
            if team_id is not None and event.get("team", {}).get("id") != team_id:
                continue
            frame_data = frames_by_event_uuid.get(event["id"])
            if frame_data is None:
                continue
            return event, frame_data

    return None


@app.get("/simulate")
async def simulate(
    match_id: int,
    minute: int,
    action: Literal["high_press", "drop_deep", "force_wide", "no_change"],
):
    """Counterfactual "what if" endpoint (Module 8 / RQ5), exposed as a
    REST GET so a single tactical action can be queried interactively
    rather than only through the batch simulator tests (Milestone 13/14).

    Deliberately reuses the EXISTING, already-validated pipeline functions
    unchanged -- `parse_360_frame`, `extract_features`, `perturb_features`,
    and this module's own `_predict_cumulative_incidence_sync` (the same
    helper the `/ws/tactical-stream` WebSocket endpoint uses, which itself
    wraps the Milestone 15 `_cumulative_incidence_forward` convention) --
    rather than a parallel/simplified extraction path. `extract_features`
    specifically encodes ADR-002's coordinate rescaling and ADR-009's
    no-direction-flip convention; reimplementing any part of this here
    would risk silently drifting from those already-validated conventions.

    `action` is a `Literal`, so FastAPI/pydantic itself rejects an invalid
    value with a 422 before this function body ever runs -- there is no
    manual validation to fall through incorrectly.
    """
    result = _find_qualifying_frame_for_minute(match_id, minute)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No 360 freeze-frame found at or after minute {minute} for match {match_id}",
        )
    event, frame_data = result

    parsed_frame = parse_360_frame(event, frame_data)
    baseline_features = extract_features(parsed_frame)
    simulated_features = perturb_features(baseline_features, action)

    baseline_threat_15s, _ = await asyncio.to_thread(
        _predict_cumulative_incidence_sync, baseline_features
    )
    simulated_threat_15s, _ = await asyncio.to_thread(
        _predict_cumulative_incidence_sync, simulated_features
    )

    return {
        "match_id": match_id,
        "minute": minute,
        "action": action,
        "baseline_threat_15s": baseline_threat_15s,
        "simulated_threat_15s": simulated_threat_15s,
        "delta": simulated_threat_15s - baseline_threat_15s,
    }
