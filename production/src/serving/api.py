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
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from production.src.models.explainer import (
    _cumulative_incidence_forward,
    build_tactical_prompt,
    compute_attributions,
    generate_explanation,
    load_deterministic_mlp,
)
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.survival_dataset import FEATURE_KEYS
from production.src.serving.simulator import live_match_stream

DEFAULT_MATCH_ID = 3857276
TIME_BIN = 3  # 15s horizon, matching Milestones 8/13/14/15

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
    prompt, runs the Milestone-15 mock LLM executor, then sends the alert
    -- guarded by the SAME connection-scoped lock the main loop uses for
    `threat` messages, so this send can never interleave with (and
    corrupt) a concurrent `threat` send on the same connection.
    """
    prompt = await asyncio.to_thread(
        _build_alert_prompt_sync, features_dict, normalized_input, cumulative_incidence
    )
    explanation = await generate_explanation(prompt)

    async with connection_lock:
        await websocket.send_json({"type": "alert", "explanation": explanation})


@app.websocket("/ws/tactical-stream")
async def tactical_stream(
    websocket: WebSocket,
    match_id: int = DEFAULT_MATCH_ID,
    delay: float = 1.0,
    spike_threshold: float = SPIKE_THRESHOLD,
):
    """`spike_threshold` defaults to the module-level SPIKE_THRESHOLD but is
    overridable per-connection via a query param -- primarily so tests can
    force a low, reliably-triggered threshold to exercise the alert path
    (Step 5.2) without depending on a specific match happening to contain a
    genuine 5-percentage-point swing within the test window.
    """
    await websocket.accept()

    # Connection-LOCAL state (Step 3.3) -- deliberately a plain local
    # variable, not module/global state, so concurrent connections each
    # track their own previous threat value independently and can never
    # clobber each other's spike detection.
    previous_threat_15s = None
    connection_lock = asyncio.Lock()

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
                    )
                )

            # Updated regardless of whether a spike fired (Step 3.7).
            previous_threat_15s = cumulative_incidence
    except WebSocketDisconnect:
        pass
