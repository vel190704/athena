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

import json
import re
from functools import partial
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from captum.attr import IntegratedGradients

from production.src.pipeline.survival_dataset import FEATURE_KEYS

MLFLOW_EXPERIMENT_NAME = "project-athena-deephit"
BIN_SIZE_SECONDS = 5.0  # Milestone 6A convention, kept local to avoid a training-pipeline import


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
