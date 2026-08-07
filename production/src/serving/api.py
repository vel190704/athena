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
import logging
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
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
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
    generate_tactical_explanation_with_source,
    load_deterministic_mlp,
)
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.simulator import perturb_features
from production.src.pipeline.survival_dataset import FEATURE_KEYS
from production.src.reporting.pass_network import (
    generate_pass_network,
    generate_pass_network_aggregated,
)
from production.src.reporting.player_report import (
    generate_player_match_summary,
    generate_player_match_timeline,
    generate_player_match_timeline_aggregated,
    generate_player_match_touch_map,
    generate_player_match_touch_map_aggregated,
    generate_player_press_resistance_index,
    generate_player_report,
    generate_player_shot_map,
    generate_player_shot_map_aggregated,
)
from production.src.reporting.team_comparison import compare_team_seasons
from production.src.reporting.team_report import generate_team_report
from production.src.serving.alert_store import DB_PATH as ALERT_DB_PATH
from production.src.serving.alert_store import count_alerts, fetch_alerts, init_db, log_alert
from production.src.serving.simulator import live_match_stream

logger = logging.getLogger(__name__)

# ADR-021 condition-2 compliance fix: an explicit, visible config switch --
# checked ONCE here at import time, not re-read per-request -- gating which
# variant of the shot map `/reports/player/{player_id}/shot-map` serves.
# Unset/anything-but-"true" (the DEFAULT: local/private/research use):
# behavior is byte-for-byte unchanged from before this flag existed --
# `generate_player_shot_map`'s real per-shot data is served, same as
# always. Set to "true" (a real public deployment): only
# `generate_player_shot_map_aggregated`'s binned-grid variant is ever
# computed or returned for that endpoint -- the raw per-shot `shots` list
# is never even constructed in this mode, not merely stripped from an
# already-built response. See ADR-021's own addendum and README.md's
# "Public deployment mode" section for the full reasoning. Mirrors this
# project's established os.environ.get(...)-flag convention exactly
# (MLFLOW_ALLOW_FILE_STORE above, GEMINI_API_KEY in explainer.py).
PUBLIC_DEPLOYMENT = os.environ.get("PUBLIC_DEPLOYMENT", "false").strip().lower() == "true"

# ADR-022: a single, optional shared-secret header check -- OFF by default
# (API_KEY unset/empty), so local development and this project's own test
# suite continue to work with zero friction, exactly as before this fix.
# Set the API_KEY environment variable to require every protected request
# to carry a matching `X-API-Key` header. Same established
# os.environ.get(...)-flag convention as PUBLIC_DEPLOYMENT above.
API_KEY = os.environ.get("API_KEY") or None

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

# Fix 3 (/health, /metrics): plain in-process counters -- no external
# metrics backend/dependency, matching this project's own "smallest real
# step up" discipline (ADR-019 makes the identical call for SQLite over
# Postgres). Reset on every process restart, which is the correct
# semantics for "active connections" and matches "uptime" resetting too;
# `total_http_requests_received` is a lifetime-of-this-process counter,
# not a persisted historical total.
_startup_monotonic: float | None = None
_total_http_requests_received = 0
_active_websocket_connections = 0
# Aug 2026 OOM-incident fix companion flag -- see `lifespan`'s own comment
# on `_model`'s idempotency guard for the full reasoning; applied here to
# the YOLO checkpoint warm-up for the same reason (a real server process
# warms it exactly once; `production/tests/` re-entering `lifespan` many
# times within one process was re-deserializing the checkpoint file on
# every entry for no behavioral benefit -- the checkpoint's already local
# after the first warm-up, in a real server OR a test process alike).
_yolo_checkpoint_warmed = False


async def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """ADR-022: the single, optional API-key check. A no-op (always
    passes) whenever `API_KEY` is unset -- this is what keeps local
    development and this project's own test suite working with zero
    friction by default. Once `API_KEY` is set, every request to a
    protected endpoint must carry a matching `X-API-Key` header or gets a
    401 -- no partial-match, no case-insensitive comparison, a plain
    exact string check (a single shared secret, not a multi-key/scope
    system -- see ADR-022 for why that's the deliberately-chosen scope).
    """
    if API_KEY is None:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _normalization_mean, _normalization_std, _model_run_id, _startup_monotonic, _yolo_checkpoint_warmed
    # Idempotency guard (Aug 2026 OOM-incident fix): a REAL server process
    # only ever enters this function once, for its whole lifetime -- this
    # guard is a no-op there. It matters for `production/tests/`, where
    # `with TestClient(app) as client:` re-triggers this SAME lifespan
    # function on the SAME already-imported `app`/module globals every
    # time it's used (test_api.py alone does this 36 times; test_simulate_api.py
    # 4 more) -- `load_deterministic_mlp()` is deterministic (same run_id,
    # same weights, every call, per its own name), so a fresh reload on
    # the 2nd-36th call was always redundant work, not fresh state, adding
    # real -- if measured smaller than initially assumed, see this
    # incident's own write-up -- memory/wall-clock cost across a full
    # suite run for zero behavioral benefit. Guarded on `_model` (not a
    # separate flag): `None` really does mean "never loaded in this
    # process," and no test anywhere sets `api_module._model` back to
    # `None` to force a genuine reload (confirmed by direct search before
    # relying on this).
    if _model is None:
        _model, _normalization_mean, _normalization_std, _model_run_id = load_deterministic_mlp()
        logger.info(f"Live inference server ready. Loaded MLP run_id={_model_run_id}")
    else:
        logger.info(f"Model already loaded (run_id={_model_run_id}) -- skipping redundant reload.")
    logger.info(
        f"PUBLIC_DEPLOYMENT={PUBLIC_DEPLOYMENT} -- shot-map endpoint will serve "
        + ("the AGGREGATED (ADR-021 condition-2-compliant) variant only." if PUBLIC_DEPLOYMENT
           else "raw per-shot data (local/private mode -- unchanged default behavior).")
    )
    logger.info(
        f"PUBLIC_DEPLOYMENT={PUBLIC_DEPLOYMENT} -- pass-network endpoint will serve "
        + ("the AGGREGATED (ADR-021 condition-2-compliant) variant only (no player location, no "
           "pairwise edge weight)." if PUBLIC_DEPLOYMENT
           else "raw per-player location/pairwise edge data (local/private mode).")
    )
    logger.info(
        f"PUBLIC_DEPLOYMENT={PUBLIC_DEPLOYMENT} -- player touch-map/timeline endpoints will serve "
        + ("the AGGREGATED (ADR-021 condition-2-compliant) variants only (no individual touch "
           "location, no individually-enumerated event)." if PUBLIC_DEPLOYMENT
           else "raw per-touch location / per-event timeline data (local/private mode). Match "
           "summary is unaffected -- it is unconditionally aggregate, never gated.")
    )
    logger.info(
        f"API_KEY {'is set -- protected endpoints now require a matching X-API-Key header.' if API_KEY else 'is unset -- no auth check, local/private default behavior (see ADR-022).'}"
    )

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
    if not _yolo_checkpoint_warmed:
        logger.info(f"Warming CV model checkpoint {CV_MODEL_CHECKPOINT} ...")
        YOLO(CV_MODEL_CHECKPOINT)
        _yolo_checkpoint_warmed = True
        logger.info("CV model checkpoint ready.")
    else:
        logger.info(f"CV model checkpoint {CV_MODEL_CHECKPOINT} already warmed -- skipping redundant reload.")

    # ADR-019 (Stage 2 persistence): ensure the alert-history schema exists
    # before any connection could fire an alert. `init_db()` is also called
    # defensively inside `log_alert`/`fetch_alerts` themselves (so direct/
    # test callers that bypass this lifespan still work), but doing it here
    # too means the schema is ready before the server ever accepts a
    # connection, not lazily on the first alert.
    init_db()
    logger.info(f"Alert-history store ready at {ALERT_DB_PATH.resolve()}")

    # Fix 3: uptime measured from HERE -- when the server actually becomes
    # ready to serve traffic (model loaded, CV checkpoint warm, alert
    # store ready) -- not from process-import time, which would include
    # variable, front-loaded startup cost as if it were "serving" time.
    _startup_monotonic = time.monotonic()

    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _count_requests_middleware(request, call_next):
    """Fix 3 (/metrics): counts every HTTP request that reaches this
    server, INCLUDING ones later rejected by `_require_api_key` (a 401 is
    still a request the server received and handled, just not one it
    fulfilled) -- named `total_http_requests_received` in /metrics,
    deliberately not `_served`, to state that distinction plainly rather
    than let the field name imply something narrower than what it counts.
    """
    global _total_http_requests_received
    _total_http_requests_received += 1
    return await call_next(request)


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
    alert_context: dict,
) -> None:
    """Background task (Step 3.7/3.8): computes attributions, builds the
    prompt, runs the explanation executor (Milestone 15's mock, or a real
    Gemini Flash-Lite call if GEMINI_API_KEY is set -- see ADR-006's Update
    section), then sends the alert -- guarded by the SAME connection-scoped
    lock the main loop uses for `threat` messages, so this send can never
    interleave with (and corrupt) a concurrent `threat` send on the same
    connection.

    ADR-019 (Stage 2 persistence): `alert_context` carries the
    source/match_id/video_path/minute/previous_threat_15s needed to log
    this alert to `alert_store.py`'s SQLite history table.
    `generate_tactical_explanation_with_source` (rather than the plain
    `generate_tactical_explanation` every other call site still uses)
    additionally reports which executor actually produced the text, so
    `explanation_source` is accurate rather than guessed. The persistence
    write is fired via `asyncio.create_task` and NEVER awaited here -- it
    must never block or delay the WebSocket send below, which is byte-for-
    byte identical to before this change.
    """
    prompt = await asyncio.to_thread(
        _build_alert_prompt_sync, features_dict, normalized_input, cumulative_incidence
    )
    explanation, explanation_source = await generate_tactical_explanation_with_source(prompt)

    asyncio.create_task(
        asyncio.to_thread(
            log_alert,
            source=alert_context["source"],
            match_id=alert_context["match_id"],
            video_path=alert_context["video_path"],
            minute=alert_context["minute"],
            threat_before=alert_context["previous_threat_15s"],
            threat_after=cumulative_incidence,
            explanation_text=explanation,
            explanation_source=explanation_source,
        )
    )

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
    alert_context: dict,
) -> None:
    """Shared spike-detection trigger (Milestone 16), reused IDENTICALLY by
    both the StatsBomb-replay and Milestone 33 CV-video sources below --
    the detection rule and alert pipeline must not silently diverge
    between sources depending on where the `threat_15s` number came from.

    `alert_context`: `{"source", "match_id", "video_path", "minute"}` --
    everything ADR-019's alert-history log needs beyond what was already
    being computed here. `previous_threat_15s` is added to it below before
    being forwarded, since `_run_alert_pipeline` needs `threat_before` and
    this function is where that value is still in scope; it is guaranteed
    non-None here (spike_fired already required it).
    """
    spike_fired = (
        previous_threat_15s is not None
        and (cumulative_incidence - previous_threat_15s) > spike_threshold
    )
    if spike_fired:
        asyncio.create_task(
            _run_alert_pipeline(
                websocket,
                connection_lock,
                features_dict,
                normalized_input,
                cumulative_incidence,
                {**alert_context, "previous_threat_15s": previous_threat_15s},
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
            {
                "source": "cv",
                "match_id": None,
                "video_path": video_path,
                "minute": int(target_timestamp_seconds // 60),
            },
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

    ADR-022: the API-key check happens HERE, BEFORE `accept()` -- an
    unauthorized client's connection is refused at the WebSocket handshake
    itself (close code 1008), the same as it would be rejected before ever
    reaching a REST endpoint's body via `_require_api_key`'s dependency
    injection. `Depends()` doesn't apply to `@app.websocket` routes the
    same way it does REST ones in this FastAPI version, so this is a
    deliberate, explicit manual check here, not an oversight.
    """
    if API_KEY is not None and websocket.headers.get("x-api-key") != API_KEY:
        await websocket.close(code=1008, reason="Missing or invalid X-API-Key header")
        return

    await websocket.accept()

    # Fix 3 (/metrics): incremented only after a successful accept() (a
    # rejected/never-accepted connection was never "active"), decremented
    # in `finally` below so every exit path -- normal completion,
    # WebSocketDisconnect, or a source="cv" validation failure closing the
    # connection immediately after accept -- still correctly frees the
    # count. `global` declared once here for both this counter and the
    # rest of this function's use of it.
    global _active_websocket_connections
    _active_websocket_connections += 1
    try:
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
                    {
                        "source": "statsbomb",
                        "match_id": match_id,
                        "video_path": None,
                        "minute": event.get("minute"),
                    },
                )

                # Updated regardless of whether a spike fired (Step 3.7).
                previous_threat_15s = cumulative_incidence
        except WebSocketDisconnect:
            pass
    finally:
        _active_websocket_connections -= 1


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


@app.get("/simulate", dependencies=[Depends(_require_api_key)])
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


# ============================================================================
# Reporting endpoints (ADR-018): the ONLY code path that should call
# load_deterministic_mlp() or touch data/raw/ for reporting purposes going
# forward. `dashboard.py`'s Player Reports / Team Reports / Team Comparison
# tabs call these over HTTP instead of importing player_report.py/
# team_report.py/team_comparison.py directly -- see ADR-018 for the full
# dual-entrypoint problem this closes. Each endpoint is a THIN wrapper: the
# report-generation functions themselves are unmodified, imported and called
# exactly as they already were, and their full return dicts (including every
# caveat/reliability field -- low-sample flags, heatmap_used_uniform_fallback,
# reliability_caveat, etc.) are passed straight through as the JSON response,
# not trimmed or reshaped.
#
# `team_trend_data.py` deliberately has NO endpoint here -- that module's own
# docstring states it must never be wired into this served layer pending
# resolution of its data source's licensing scope (the same conservative
# stance ADR-014 applies to the AGPL-derived CV model). See ADR-018.
#
# Each wrapped function does real, potentially slow, blocking work (network
# fetches to StatsBomb, MLflow artifact loads, pitch-control physics) -- run
# via asyncio.to_thread, the same pattern already used for /simulate's
# _predict_cumulative_incidence_sync calls, so a slow report never blocks the
# event loop (and therefore never blocks other connections' WebSocket
# streams) while it runs.
# ============================================================================


@app.get("/reports/player/{player_id}", dependencies=[Depends(_require_api_key)])
async def get_player_report(player_id: int, match_ids: list[int] = Query(...)):
    """Wraps player_report.generate_player_report, unmodified. `match_ids`
    is a repeated query parameter, e.g. `?match_ids=1&match_ids=2`."""
    return await asyncio.to_thread(generate_player_report, player_id, match_ids)


@app.get("/reports/player/{player_id}/shot-map", dependencies=[Depends(_require_api_key)])
async def get_player_shot_map(player_id: int, match_ids: list[int] = Query(...)):
    """Wraps player_report.generate_player_shot_map (or, in PUBLIC_DEPLOYMENT
    mode, generate_player_shot_map_aggregated), unmodified. A DEDICATED
    endpoint, not a field added to /reports/player/{player_id}'s response,
    deliberately: extending that endpoint's response would mean EVERY
    existing caller pays the shot-map computation's cost (measured ~11s for
    a full 596-match career) on every call, whether or not they want
    shot-map data -- a real, if secondary, violation of "additive only"
    (performance, not just response shape) that a separate endpoint avoids
    entirely. Existing /reports/player/{player_id} callers and response
    shape are completely unaffected by this endpoint's existence.

    ADR-021 condition-2 compliance fix (public/private branch, decided ONCE
    from the module-level PUBLIC_DEPLOYMENT flag, not per-request logic
    that could be tricked): PUBLIC_DEPLOYMENT=false (default) returns
    generate_player_shot_map's real per-shot data, byte-for-byte identical
    to this endpoint's behavior before this flag existed.
    PUBLIC_DEPLOYMENT=true returns generate_player_shot_map_aggregated's
    binned-grid-only variant instead -- the raw per-shot `shots` list is
    never computed at all on this path, not merely omitted from an
    already-built response.
    """
    if PUBLIC_DEPLOYMENT:
        return await asyncio.to_thread(generate_player_shot_map_aggregated, player_id, match_ids)
    return await asyncio.to_thread(generate_player_shot_map, player_id, match_ids)


@app.get("/reports/player/{player_id}/match-summary", dependencies=[Depends(_require_api_key)])
async def get_player_match_summary(player_id: int, match_ids: list[int] = Query(...)):
    """Wraps player_report.generate_player_match_summary, unmodified.

    ADR-021 condition 2: NOT gated by PUBLIC_DEPLOYMENT -- per-match TOTALS
    only (minutes, event-type counts), the same aggregate-count class of
    data /reports/player/{player_id} already serves unconditionally, just
    broken out per match instead of summed across all of them. See
    player_report.py's own Player Dashboard section for the full Step 0
    reasoning.
    """
    return await asyncio.to_thread(generate_player_match_summary, player_id, match_ids)


@app.get("/reports/player/{player_id}/press-resistance", dependencies=[Depends(_require_api_key)])
async def get_player_press_resistance_index(player_id: int, match_ids: list[int] = Query(...)):
    """Wraps player_report.generate_player_press_resistance_index, unmodified.

    ADR-021 condition 2: NOT gated by PUBLIC_DEPLOYMENT -- pure per-event-type/
    overall COUNTS and RATES (no location, no minute, no event id, no
    individual event ever enumerated), the same aggregate-count class of
    data /reports/player/{player_id}/match-summary already serves
    unconditionally. See generate_player_press_resistance_index's own
    docstring in player_report.py for the full exemption reasoning, and
    docs/adr/ADR-021-statsbomb-free-public-deployment-scoping.md for the
    addendum recording this decision.
    """
    return await asyncio.to_thread(generate_player_press_resistance_index, player_id, match_ids)


@app.get("/reports/player/{player_id}/match/{match_id}/touch-map", dependencies=[Depends(_require_api_key)])
async def get_player_match_touch_map(player_id: int, match_id: int):
    """Wraps player_report.generate_player_match_touch_map (or, in
    PUBLIC_DEPLOYMENT mode, generate_player_match_touch_map_aggregated),
    unmodified. Single `match_id` path parameter, not a `match_ids` list --
    a touch map is inherently a per-match view (see pass_network.py's own
    precedent for the same single-match-scope reasoning).

    ADR-021 condition-2 compliance (SAME gating pattern as the shot-map/
    pass-network endpoints above): PUBLIC_DEPLOYMENT=false (default)
    returns real per-touch locations. PUBLIC_DEPLOYMENT=true returns the
    grid-binned aggregated variant instead -- the raw `touches` list is
    never computed at all on this path.
    """
    if PUBLIC_DEPLOYMENT:
        return await asyncio.to_thread(generate_player_match_touch_map_aggregated, player_id, match_id)
    return await asyncio.to_thread(generate_player_match_touch_map, player_id, match_id)


@app.get("/reports/player/{player_id}/match/{match_id}/timeline", dependencies=[Depends(_require_api_key)])
async def get_player_match_timeline(player_id: int, match_id: int):
    """Wraps player_report.generate_player_match_timeline (or, in
    PUBLIC_DEPLOYMENT mode, generate_player_match_timeline_aggregated),
    unmodified. Single `match_id` path parameter -- same per-match scope
    reasoning as the touch-map endpoint above.

    ADR-021 condition-2 compliance: PUBLIC_DEPLOYMENT=false (default)
    returns the real, individually-enumerated per-event timeline.
    PUBLIC_DEPLOYMENT=true returns only event-TYPE counts per coarse time
    bucket -- the raw `timeline` list (and every individual event's exact
    minute/outcome/body-part detail) is never computed at all on this path.
    """
    if PUBLIC_DEPLOYMENT:
        return await asyncio.to_thread(generate_player_match_timeline_aggregated, player_id, match_id)
    return await asyncio.to_thread(generate_player_match_timeline, player_id, match_id)


@app.get("/reports/pass-network/{match_id}", dependencies=[Depends(_require_api_key)])
async def get_pass_network(match_id: int):
    """Wraps pass_network.generate_pass_network (or, in PUBLIC_DEPLOYMENT
    mode, generate_pass_network_aggregated), unmodified. Single `match_id`
    path parameter, not a `match_ids` list like the other /reports/*
    endpoints -- a pass network is inherently a per-match shape (see
    pass_network.py's own module docstring for why aggregating it across
    matches the way player/team reports do would not mean anything).

    ADR-021 condition-2 compliance (SAME gating decision and pattern as
    the shot-map endpoint above, decided ONCE from the module-level
    PUBLIC_DEPLOYMENT flag, not per-request logic that could be tricked):
    PUBLIC_DEPLOYMENT=false (default) returns generate_pass_network's real
    per-player average location and real pairwise completed-pass edge
    weights. PUBLIC_DEPLOYMENT=true returns generate_pass_network_aggregated's
    per-player-totals-only variant instead -- the raw `nodes`/`edges`
    lists are never computed at all on this path, not merely omitted from
    an already-built response. See docs/adr/ADR-021's own addendum for the
    full reasoning behind treating the raw pass network as RAW,
    individually-attributable data under condition 2.
    """
    if PUBLIC_DEPLOYMENT:
        return await asyncio.to_thread(generate_pass_network_aggregated, match_id)
    return await asyncio.to_thread(generate_pass_network, match_id)


@app.get("/reports/team/{team_name}", dependencies=[Depends(_require_api_key)])
async def get_team_report(team_name: str, match_ids: list[int] = Query(default=[])):
    """Wraps team_report.generate_team_report, unmodified.

    Zero-usable-match fix, reproduced directly before this change:
    Atlético Madrid, La Liga 2020/21 (2 raw cached matches, 0 with 360
    coverage) -- selecting this in dashboard.py could reach this endpoint
    with an empty `match_ids` list, which `requests` sends as NO
    `match_ids` param at all. With the old `Query(...)` (required),
    FastAPI rejected this before `generate_team_report` ever ran, with its
    own raw, caller-unfriendly 422 body
    (`{"detail":[{"type":"missing","loc":["query","match_ids"],...}]}`).
    `match_ids` is now OPTIONAL (default empty list) -- an empty
    selection is a real, expected case (many real cached teams have zero
    360-covered matches for a given season), not caller error.
    `generate_team_report` itself already handles an empty `match_ids`
    list gracefully with no code change needed (confirmed directly:
    `matches_used=0`, an all-None `control_heatmap_grid`, no crash) -- the
    real bug was purely this endpoint's parameter validation rejecting
    the request before that graceful path ever ran. `no_data`/`reason`
    are added here, on top of that unmodified return dict, only when
    `match_ids` was empty -- an explicit signal so a caller doesn't have
    to infer "no data" indirectly from `matches_used==0` alone.
    """
    report = await asyncio.to_thread(generate_team_report, team_name, match_ids)
    if not match_ids:
        report["no_data"] = True
        report["reason"] = "No match_ids provided -- select a team/season with at least one 360-covered match."
    return report


@app.get("/reports/team-comparison", dependencies=[Depends(_require_api_key)])
async def get_team_comparison(team_a: str, season_a: int, team_b: str, season_b: int):
    """Wraps team_comparison.compare_team_seasons, unmodified."""
    return await asyncio.to_thread(compare_team_seasons, team_a, season_a, team_b, season_b)


# ============================================================================
# ADR-019 (Stage 2 persistence): the read side of the alert-history store.
# Every filter is optional; `start_utc`/`end_utc` are ISO-8601 UTC strings
# (matching what `log_alert` writes) and AND-combined with any other
# filters given. This is the endpoint that makes persisting alerts actually
# useful -- "so I can review a match's alert history afterward" -- rather
# than write-only data nobody can query back.
# ============================================================================


@app.get("/alerts/history", dependencies=[Depends(_require_api_key)])
async def get_alerts_history(
    match_id: int | None = None,
    source: Literal["statsbomb", "cv"] | None = None,
    start_utc: str | None = None,
    end_utc: str | None = None,
    limit: int = 500,
):
    return await asyncio.to_thread(
        fetch_alerts,
        match_id=match_id,
        source=source,
        start_utc=start_utc,
        end_utc=end_utc,
        limit=limit,
    )


# ============================================================================
# Fix 3: /health and /metrics -- both cheap, fast, no side effects (no
# network calls, no disk I/O beyond a single SQLite COUNT(*) for
# /metrics). Neither wraps a research/model-logic function; both report
# on THIS process's own already-in-memory state.
#
# /health is deliberately NOT behind `_require_api_key` -- a common,
# sensible convention (load balancers/uptime monitors need to probe
# liveness without a credential) and the ONLY endpoint in this file
# exempted this way; /metrics, by contrast, IS behind the API-key check
# below, since operational counters (request volume, active connections)
# are more reasonably treated as needing the same protection as this
# project's actual reporting endpoints, not as a public liveness signal.
# ============================================================================


@app.get("/health")
async def health():
    """Liveness/readiness check. `model_loaded`/`mlflow_reachable` report
    the SAME underlying signal (whether `lifespan`'s
    `load_deterministic_mlp()` call succeeded at startup) -- a live
    MLflow re-query on every health-check call would itself be a network/
    disk operation, violating the "cheap, fast, no side effects" this
    endpoint exists to guarantee. This is "was MLflow reachable when the
    server started," not "is MLflow reachable this instant" -- stated
    explicitly here so it is never misread as a live re-check.
    """
    return {
        "status": "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "mlflow_reachable": _model is not None,
        "model_run_id": _model_run_id,
        "uptime_seconds": (
            time.monotonic() - _startup_monotonic if _startup_monotonic is not None else None
        ),
    }


@app.get("/metrics", dependencies=[Depends(_require_api_key)])
async def metrics():
    """Basic operational counters -- plain JSON, not a Prometheus-format
    exporter (no `prometheus_client` dependency exists anywhere in this
    project; adding one for three counters would be disproportionate
    machinery here, the same "smallest real step up" reasoning ADR-019
    already applies to SQLite-over-Postgres). `total_http_requests_received`
    counts every request the middleware saw, including ones later
    rejected by the API-key check (see that middleware's own docstring).
    `total_alerts_logged` is a real SQLite COUNT(*) via
    `alert_store.count_alerts` (run off the event loop via
    `asyncio.to_thread`, the same pattern every other blocking call in
    this file already uses), not an in-memory counter that would silently
    diverge from the actual persisted alert history.
    """
    return {
        "total_http_requests_received": _total_http_requests_received,
        "active_websocket_connections": _active_websocket_connections,
        "total_alerts_logged": await asyncio.to_thread(count_alerts),
        "uptime_seconds": (
            time.monotonic() - _startup_monotonic if _startup_monotonic is not None else None
        ),
    }
