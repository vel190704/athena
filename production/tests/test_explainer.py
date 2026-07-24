"""Milestone 15 validation: Integrated Gradients feature attribution on the
deterministically-selected MLP, the tactical prompt built from it, and the
deterministic mock LLM executor.

Deliberately end-to-end against REAL match data (match 3857276, same cached
match used by Milestones 13/14's simulator tests), not synthetic tensors --
this project's established preference for real-data verification over
memory/assumption.
"""

import math
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import torch

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.models.explainer import (
    _cumulative_incidence_forward,
    build_tactical_prompt,
    compute_attributions,
    generate_explanation,
    load_deterministic_mlp,
)
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.survival_dataset import FEATURE_KEYS

MATCH_ID = 3857276
TIME_BIN = 3  # 15s horizon, matching Milestones 8/13/14
COMPLETENESS_ATOL = 1e-2  # IG's approximation error at n_steps=50


def _fetch_baseline_features() -> dict:
    """First real period-1 Pass event with an associated 360 frame from the
    cached match, run through the existing extraction pipeline -- same
    match/event-selection logic as Milestone 13/14's simulator tests, for
    direct comparability."""
    events = fetch_match_events(MATCH_ID)
    frames = fetch_match_360(MATCH_ID)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    for event in events:
        if event["period"] != 1:
            continue
        if event["type"]["name"] != "Pass":
            continue
        if "location" not in event:
            continue
        frame_data = frames_by_event_uuid.get(event["id"])
        if frame_data is None:
            continue

        parsed = parse_360_frame(event, frame_data)
        return extract_features(parsed)

    raise RuntimeError(f"no period-1 Pass event with a 360 frame found in match {MATCH_ID}")


def test_explainer_end_to_end_on_real_match_state():
    model, normalization_mean, normalization_std, run_id = load_deterministic_mlp()
    print(f"\nLoaded deterministically-selected MLP from MLflow run_id={run_id}")

    features_dict = _fetch_baseline_features()
    scalar_features = [features_dict[key] for key in FEATURE_KEYS]

    raw_tensor = torch.tensor([scalar_features], dtype=torch.float32)  # [1, num_features]
    normalized_input = (raw_tensor - normalization_mean) / normalization_std
    normalized_input.requires_grad_(True)

    # Zero vector in NORMALIZED feature space -- the average training match
    # state (see compute_attributions' docstring), not an arbitrary zero.
    baseline_tensor = torch.zeros_like(normalized_input)

    attributions = compute_attributions(model, normalized_input, baseline_tensor, time_bin=TIME_BIN)

    assert set(attributions.keys()) == set(FEATURE_KEYS)
    for name, value in attributions.items():
        assert math.isfinite(value), f"non-finite attribution for {name!r}: {value}"

    # Completeness check: both cumulative-incidence values MUST use the same
    # wrapper Captum attributed, not the raw model output, to keep this
    # check consistent with what's actually being explained.
    with torch.no_grad():
        ci_input = _cumulative_incidence_forward(model, normalized_input, time_bin=TIME_BIN).item()
        ci_baseline = _cumulative_incidence_forward(model, baseline_tensor, time_bin=TIME_BIN).item()

    attribution_sum = sum(attributions.values())
    expected_diff = ci_input - ci_baseline
    completeness_error = abs(attribution_sum - expected_diff)

    print(f"\n=== Milestone 15 attribution report (match {MATCH_ID}, time_bin={TIME_BIN} / 15s) ===")
    print(f"Predicted cumulative incidence (input): {ci_input:.6f}")
    print(f"Predicted cumulative incidence (baseline): {ci_baseline:.6f}")
    print(f"Sum of attributions: {attribution_sum:.6f}")
    print(f"Expected (ci_input - ci_baseline): {expected_diff:.6f}")
    print(f"Completeness error: {completeness_error:.6f} (tolerance: {COMPLETENESS_ATOL})")
    for name, value in sorted(attributions.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name}: {value:+.6f}")

    assert completeness_error < COMPLETENESS_ATOL, (
        f"Integrated Gradients completeness check failed: |{attribution_sum:.6f} - "
        f"{expected_diff:.6f}| = {completeness_error:.6f} >= {COMPLETENESS_ATOL} "
        "(consider increasing n_steps if this is a genuine approximation-error issue)"
    )

    prompt = build_tactical_prompt(features_dict, attributions, ci_input, time_bin=TIME_BIN)
    print(f"\n=== Generated prompt ===\n{prompt}")

    expected_pct_substring = f"{ci_input * 100:.1f}%"
    assert expected_pct_substring in prompt, (
        f"prompt should contain the predicted incidence percentage {expected_pct_substring!r}"
    )

    positive_names = [name for name, value in attributions.items() if value > 0]
    negative_names = [name for name, value in attributions.items() if value < 0]
    top_positive_names = sorted(positive_names, key=lambda n: attributions[n], reverse=True)[:2]
    top_negative_names = sorted(negative_names, key=lambda n: attributions[n])[:2]

    for name in top_positive_names:
        assert name in prompt, f"expected top positive driver {name!r} to appear in prompt"
    for name in top_negative_names:
        assert name in prompt, f"expected top negative driver {name!r} to appear in prompt"

    if not top_positive_names:
        assert "No factors" in prompt
    if not top_negative_names:
        assert "No factors" in prompt

    import asyncio

    mock_response = asyncio.run(generate_explanation(prompt))
    print(f"\n=== Mock LLM response ===\n{mock_response}")

    assert isinstance(mock_response, str)
    assert len(mock_response) > 0
    assert "Tactical Analysis:" in mock_response
