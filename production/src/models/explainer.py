"""Tactical LLM explainer (Milestone 15, Module 9): Integrated Gradients
feature attribution on the trained MLP, formatted into a structured prompt,
and a deterministic mock LLM that turns it into a natural-language tactical
explanation.

Per ADR-006, the LLM strictly EXPLAINS an already-computed prediction; it
does not predict anything itself (the mock executor in Step 4 only ever
echoes back numbers that were already computed by the model + Captum, it
never invents a new threat estimate). Precision matters throughout this
module: attributions must explain the EXACT SAME scalar quantity that gets
reported to the user -- the inclusive-cumsum cumulative incidence at a
fixed time_bin (the Milestone 8/13 convention), not the raw per-bin PMF,
which is a narrower, different quantity. `_cumulative_incidence_forward`
exists specifically so Captum attributes THAT quantity, not the model's
raw output.
"""

import asyncio
import json
import logging
import os
import re
from functools import partial
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from captum.attr import IntegratedGradients
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from production.src.constants import BIN_SIZE_SECONDS, MLFLOW_EXPERIMENT_NAME
from production.src.pipeline.survival_dataset import FEATURE_KEYS

# Loads GEMINI_API_KEY (and anything else) from a git-ignored .env file if
# one exists, same convention as ROBOFLOW_API_KEY (ADR-014) -- called here,
# at module level, rather than left to whichever caller happens to import
# this module first (this module is imported by both api.py's live
# `uvicorn` entrypoint and by reporting/test code, and only the test files
# could otherwise be relied on to have called this themselves). A no-op if
# no .env file exists or GEMINI_API_KEY isn't in it.
load_dotenv()

# Only configured by this file's own callers if they need it; this module
# itself never calls logging.basicConfig() (would mutate global logging
# config on import -- the same discipline train.py's module-level logger
# comment establishes).
logger = logging.getLogger(__name__)

# MLFLOW_EXPERIMENT_NAME and BIN_SIZE_SECONDS now come from
# production.src.constants (engineering-review de-duplication -- both were
# previously redefined locally here; BIN_SIZE_SECONDS specifically to
# "avoid a training-pipeline import," a concern the new dependency-free
# constants module makes moot). Values unchanged.

# Verified against the real, current Gemini API docs and the google-genai
# PyPI/GitHub pages immediately before this was written (not recalled from
# memory -- model names and SDK packages both changed multiple times across
# this project's own history, e.g. the roboflow/AGPL investigation in
# ADR-014). As of this check: `google-genai` (PyPI, v2.16.0) is Google's
# current, GA-recommended SDK -- the older `google-generativeai` package is
# deprecated and specifically NOT used here. The Gemini model lineup has
# moved to a Gemini 3 generation since this project's own knowledge cutoff;
# `gemini-3.5-flash-lite` is the current fastest/most cost-effective
# Flash-Lite variant (confirmed present on both ai.google.dev's models page
# and its pricing page), not the older `gemini-2.5-flash-lite` a
# from-memory guess would have produced. Free-tier RPM/TPM/RPD numbers are
# NOT published as a static table in the current docs (unlike some older
# API generations) -- Google's own docs direct users to their account's
# live AI Studio dashboard for exact figures, so no specific quota number
# is hardcoded or assumed here; the spike-triggered (not per-frame) calling
# pattern this project already uses (ADR-006) keeps real call volume low
# regardless of the exact figure.
GEMINI_MODEL_ID = "gemini-3.5-flash-lite"
GEMINI_TIMEOUT_SECONDS = 10.0
GEMINI_MAX_OUTPUT_TOKENS = 200
GEMINI_TEMPERATURE = 0.3  # low, deliberately -- a grounded explanation task, not creative writing


def select_deterministic_mlp_run_id() -> str:
    """Resolve the MLP to load via an EXPLICIT filter, not "most recent
    run" (Milestone 14B's seed-42-vs-43 ambiguity was harmless by luck --
    this milestone doesn't rely on luck again).

    Filters MLflow runs to model_type=MLP, stabilization_bundle=True,
    saturation_check_v2=True, and among matches, selects the one with the
    LOWEST logged val_brier_15s. Prints the selected run_id explicitly.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment {MLFLOW_EXPERIMENT_NAME!r} not found -- run "
            "`python -m production.src.pipeline.train` at least once first."
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            "params.model_type = 'MLP' and params.stabilization_bundle = 'True' "
            "and params.saturation_check_v2 = 'True'"
        ),
        order_by=["metrics.val_brier_15s ASC"],
    )
    if not runs:
        raise RuntimeError(
            "No MLP run found tagged model_type=MLP, stabilization_bundle=True, "
            "saturation_check_v2=True -- run `python -m production.src.pipeline.train` first."
        )

    selected = runs[0]
    print(
        f"[explainer] Selected MLP run_id={selected.info.run_id} "
        f"(lowest val_brier_15s={selected.data.metrics.get('val_brier_15s')}, "
        f"among {len(runs)} matching run(s))"
    )
    return selected.info.run_id


def load_deterministic_mlp() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, str]:
    """Load the deterministically-selected MLP (Step 0) and the
    training-split-derived normalization stats logged alongside it.

    Returns (model, normalization_mean, normalization_std, run_id).
    """
    run_id = select_deterministic_mlp_run_id()

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/mlp_model")
    model.eval()

    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="normalization")
    json_file = next(Path(local_dir).glob("*.json"))
    with open(json_file) as f:
        stats = json.load(f)

    normalization_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    normalization_std = torch.tensor(stats["std"], dtype=torch.float32)

    return model, normalization_mean, normalization_std, run_id


def _cumulative_incidence_forward(model: torch.nn.Module, x: torch.Tensor, time_bin: int = 3) -> torch.Tensor:
    """model(x) -> PMF -> inclusive cumsum -> cumulative incidence at
    `time_bin`, as a [batch] tensor. Same convention as Milestones 8/13:
    S(t) = 1 - cumsum(PMF), so cumulative incidence = 1 - S(t) = cumsum(PMF)
    evaluated at t.

    This wrapper -- NOT the raw model -- is what Captum attributes.
    Attributing the raw PMF at a single bin (e.g. via a `target=` index
    into the softmax output) would compute a different, narrower quantity
    than "cumulative incidence," and the prompt built in Step 3 would then
    misdescribe what the attributions actually explain.
    """
    pmf = model(x)
    cumulative = torch.cumsum(pmf, dim=1)
    return cumulative[:, time_bin]


def compute_attributions(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    time_bin: int = 3,
) -> dict[str, float]:
    """Integrated Gradients attribution of `_cumulative_incidence_forward`
    (NOT the raw model) with respect to each of the 4 normalized input
    features.

    `baseline_tensor` should be a ZERO vector in NORMALIZED feature space.
    This is a meaningful baseline, not an arbitrary one: since
    normalization is `(x - mean) / std`, a zero vector corresponds EXACTLY
    to the average training match state. Attributions are therefore
    computed relative to "how does this match state differ from an
    average one," not relative to a physically meaningless zero point in
    the original (unnormalized) feature space.

    Deliberately NOT wrapped in torch.no_grad() -- unlike this project's
    other inference code (Brier calculation, the scalar/graph simulators),
    Captum requires gradients to flow through the input tensor to compute
    attributions; no_grad() would break this outright, not just slow it
    down.
    """
    forward_fn = partial(_cumulative_incidence_forward, model, time_bin=time_bin)
    integrated_gradients = IntegratedGradients(forward_fn)

    attributions = integrated_gradients.attribute(input_tensor, baselines=baseline_tensor)

    attribution_values = attributions.squeeze(0).tolist()
    return dict(zip(FEATURE_KEYS, attribution_values))


def build_tactical_prompt(
    features_dict: dict,
    attributions: dict[str, float],
    cumulative_incidence: float,
    time_bin: int = 3,
) -> str:
    """Builds the LLM prompt: system framing (explain, don't predict --
    ADR-006), the predicted cumulative incidence (computed via the SAME
    `_cumulative_incidence_forward` wrapper used for the attributions, so
    the number here and the attributions explaining it are guaranteed
    consistent), and up to 2 top positive / up to 2 top negative
    attributions.

    Does NOT assume exactly 2 positive and 2 negative attributions exist --
    with only 4 features, an uneven split (e.g. 3 positive / 1 negative) is
    entirely possible. If a bucket has fewer than 2 entries, includes
    however many exist and notes this explicitly in the prompt text.
    """
    seconds = time_bin * BIN_SIZE_SECONDS

    positive_all = sorted(
        ((name, value) for name, value in attributions.items() if value > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    negative_all = sorted(
        ((name, value) for name, value in attributions.items() if value < 0),
        key=lambda item: item[1],
    )
    top_positive = positive_all[:2]
    top_negative = negative_all[:2]

    def _format_bucket(items: list[tuple[str, float]], direction_label: str) -> str:
        if not items:
            return f"No factors {direction_label} were identified."
        lines = "\n".join(f"- {name}: {value:+.4f}" for name, value in items)
        if len(items) < 2:
            lines += f"\n(Only {len(items)} factor {direction_label} found, not 2.)"
        return lines

    positive_text = _format_bucket(top_positive, "increasing this threat")
    negative_text = _format_bucket(top_negative, "decreasing this threat")

    prompt = (
        "You are an expert football tactical analyst. Your job is to explain WHY the AI "
        "model predicted this threat level based on the provided spatial feature "
        "attributions. You explain the model's existing prediction; you do not make your "
        "own prediction.\n\n"
        f"Predicted threat: {cumulative_incidence * 100:.1f}% probability of a shot in the "
        f"next {seconds:.0f} seconds.\n\n"
        "Top factors INCREASING this threat estimate:\n"
        f"{positive_text}\n\n"
        "Top factors DECREASING this threat estimate:\n"
        f"{negative_text}\n\n"
        "Provide a concise 2-3 sentence tactical explanation of why the model predicted "
        "this threat level, referencing the spatial factors above."
    )
    return prompt


_THREAT_PATTERN = re.compile(r"Predicted threat:\s*([\d.]+)%")
_INCREASING_SECTION_PATTERN = re.compile(
    r"Top factors INCREASING this threat estimate:\n(.*?)\n\nTop factors DECREASING",
    re.DOTALL,
)
_DECREASING_SECTION_PATTERN = re.compile(
    r"Top factors DECREASING this threat estimate:\n(.*?)\n\nProvide a concise",
    re.DOTALL,
)
_FACTOR_NAME_PATTERN = re.compile(r"^- ([\w_]+):", re.MULTILINE)


def _extract_factor_names(section_text: str | None) -> list[str]:
    if not section_text:
        return []
    return _FACTOR_NAME_PATTERN.findall(section_text)


async def generate_explanation(prompt: str) -> str:
    """Deterministic mock LLM executor -- no real API call. Parses the
    structured prompt (built by `build_tactical_prompt`) and returns a
    templated string, simulating ADR-006's async background worker without
    an actual network round-trip or a real language model.
    """
    threat_match = _THREAT_PATTERN.search(prompt)
    threat_pct = threat_match.group(1) if threat_match else "unknown"

    increasing_match = _INCREASING_SECTION_PATTERN.search(prompt)
    decreasing_match = _DECREASING_SECTION_PATTERN.search(prompt)

    drivers = _extract_factor_names(increasing_match.group(1) if increasing_match else None)
    mitigators = _extract_factor_names(decreasing_match.group(1) if decreasing_match else None)

    driver_text = ", ".join(drivers) if drivers else "none identified"
    mitigator_text = ", ".join(mitigators) if mitigators else "none identified"

    return (
        f"Tactical Analysis: The threat is {threat_pct}%. "
        f"Key drivers: {driver_text}. "
        f"Mitigating factors: {mitigator_text}."
    )


# =============================================================================
# ADR-006 Update: real Gemini Flash-Lite integration (see that ADR's own
# "Update" section for the full decision). ADDITIVE ONLY -- `generate_explanation`
# above is UNCHANGED and remains the permanent, deterministic fallback and
# the default test-time executor; both existing test files call it directly
# to unit-test its own parsing logic, which must keep working identically
# regardless of whether a real API key happens to be present in this
# environment.
# =============================================================================

# A REAL LLM, unlike the deterministic mock above, can plausibly hallucinate
# exactly the kind of unsupported claim Milestone 43's zone-explainer
# (`zone_explainer.build_zone_explanation_prompt`) already proved this
# project's data cannot support -- no frame-to-frame trajectory analysis or
# player-movement computation exists anywhere in this project (see
# REPORTING_FINDINGS.md Section 4). Sent via the SDK's own `system_instruction`
# channel (not string-concatenated into the user prompt), so it applies
# uniformly to BOTH `build_tactical_prompt`'s output (Milestone 15, which has
# no such constraint built into its own text) and
# `build_zone_explanation_prompt`'s output (Milestone 43, which already
# states a version of this constraint inline, as defense in depth, not as a
# substitute for this one) -- and so neither existing, already-tested
# prompt-builder function needs to change to get this protection.
_HONESTY_SYSTEM_INSTRUCTION = (
    "You are a tactical explanation generator for a live football analytics system. "
    "You explain an ALREADY-COMPUTED model prediction; you never make your own prediction. "
    "Follow these rules strictly:\n"
    "1. Only describe what is explicitly present in the data provided in the user message "
    "(the predicted threat percentage and the named factors with their signed attribution "
    "magnitudes).\n"
    "2. Never state or imply a timing, duration, or speed claim (for example: 'for the next "
    "few seconds', 'briefly', 'quickly', 'soon') -- this system computes no such quantity "
    "anywhere, for any feature.\n"
    "3. Never state or imply a specific player movement, positioning change, or recovery "
    "direction (for example: 'the defender is recovering', 'players are shifting left', "
    "'closing down') -- this system computes no player-level trajectory or movement data "
    "anywhere.\n"
    "4. If you are not certain a specific detail is directly supported by the data provided, "
    "omit it rather than guess or embellish.\n"
    "5. Keep the response to 2-3 concise sentences."
)


def _safe_error_text(exc: BaseException) -> str:
    """`str(exc)`, with the real `GEMINI_API_KEY` value masked out if
    present. Defensive, not reactive -- no known google-genai error
    message actually echoes the key back, but this project's "never log,
    print, or write the key anywhere, including in error messages" rule
    (same discipline as `ROBOFLOW_API_KEY`/`SOCCERNET_PASSWORD`) is
    absolute, not conditional on today's SDK behavior staying the same.
    """
    text = str(exc)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and api_key in text:
        text = text.replace(api_key, "***REDACTED***")
    return text


async def generate_explanation_real(prompt: str) -> str:
    """Real Gemini Flash-Lite call (`google-genai`'s native async client --
    `client.aio.models.generate_content`, not a sync call wrapped in
    `asyncio.to_thread`, since an awaitable version already exists; the
    `asyncio.to_thread` pattern this project established at Milestone 16 is
    for genuinely synchronous SDKs, e.g. Captum in `_build_alert_prompt_sync`,
    not a substitute demanded everywhere regardless of what a library
    actually offers).

    Deliberately raises on ANY failure (timeout, API error, empty/malformed
    response) rather than catching anything itself -- see
    `generate_tactical_explanation` for the fallback-to-mock behavior. This
    split keeps this function directly, cleanly testable in isolation (e.g.
    simulating an invalid key and asserting it raises) without conflating
    "did the real call fail" with "did the fallback correctly kick in" in
    one function's control flow.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_HONESTY_SYSTEM_INSTRUCTION,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                temperature=GEMINI_TEMPERATURE,
            ),
        ),
        timeout=GEMINI_TIMEOUT_SECONDS,
    )

    text = response.text
    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty/malformed response (no text content).")
    return text.strip()


async def generate_tactical_explanation_with_source(prompt: str) -> tuple[str, str]:
    """Same executor-selection logic as `generate_tactical_explanation`
    (below, now a thin wrapper around this), but also returns WHICH
    executor actually produced the text -- `"gemini"` or `"mock"`.

    ADR-019: added for the alert-history store's `explanation_source`
    column. A caller cannot correctly infer this from `GEMINI_API_KEY`
    being set alone -- the key being present does not mean the real call
    SUCCEEDED, since any failure here falls back to the mock (see below).
    Reporting "gemini" whenever the key happens to be set, without
    checking whether the fallback actually fired, would silently persist a
    wrong `explanation_source` on every fallback -- worse than not
    recording the field at all, given this project's own honesty
    discipline elsewhere (`_HONESTY_SYSTEM_INSTRUCTION`,
    `_UNSUPPORTED_CLAIM_PATTERN` in `test_zone_explainer.py`). This
    function is therefore the only accurate source of that value; nothing
    reimplements or guesses it independently.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return await generate_explanation(prompt), "mock"

    try:
        return await generate_explanation_real(prompt), "gemini"
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: ANY real-API
        # failure mode (timeout, rate limit, auth, malformed response, a
        # transient SDK/network error not yet seen) must fall back, not
        # propagate -- narrowing this to specific exception types would
        # silently reintroduce exactly the crash risk Step 2.2 exists to
        # prevent.
        logger.warning(
            f"[explainer] Real Gemini call failed ({type(exc).__name__}) -- falling back to "
            f"the mock executor for this call. Error: {_safe_error_text(exc)}"
        )
        return await generate_explanation(prompt), "mock"


async def generate_tactical_explanation(prompt: str) -> str:
    """Public entrypoint call sites (the WebSocket spike-alert pipeline in
    `api.py`, the zone-level reporting flow in `zone_explainer.py`) should
    use going forward -- internally decides real-vs-mock so no call site
    needs to know or care which executor is active. `generate_explanation`
    (the mock) is never removed and stays directly callable on its own.

    Thin wrapper around `generate_tactical_explanation_with_source` that
    discards the source -- kept as its own function, with its EXACT prior
    signature and behavior, so every existing call site (`zone_explainer.py`,
    tests) is unaffected by ADR-019's addition above.
    """
    text, _source = await generate_tactical_explanation_with_source(prompt)
    return text
