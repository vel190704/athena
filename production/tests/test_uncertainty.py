"""Milestone 21 validation: Deep Ensemble uncertainty quantification
(ADR-004 -- a Deep Ensemble, NOT a true Batch Ensemble; see that ADR).

Reporting philosophy: the diversity sanity check is a HARD assertion (a
collapsed/non-diverse ensemble is an actual engineering failure, not a
research finding -- it would make every uncertainty number downstream
meaningless). The OOD-vs-in-distribution uncertainty comparison, by
contrast, is reported as a FINDING, not hard-asserted in either direction
-- consistent with this project's established practice for open empirical
questions (Milestones 13/20): a 5-member Deep Ensemble trained via plain
MLE has no explicit mechanism (no anchor networks, no negative-correlation
term, no bootstrap resampling here) guaranteeing well-calibrated epistemic
uncertainty on out-of-distribution inputs, so whether OOD uncertainty is
actually higher here is itself the thing being tested, not assumed.
"""

import math
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.pytorch
import torch

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.survival_dataset import FEATURE_KEYS

MLFLOW_EXPERIMENT_NAME = "project-athena-deephit"
MATCH_ID = 3857276
TIME_BIN = 3  # 15s horizon, matching every prior milestone
DIVERSITY_STD_THRESHOLD = 1e-4
OOD_Z_SCORE = 5.0  # normalized-feature value defining the constructed OOD point


def select_deterministic_deep_ensemble_run_id() -> str:
    """Same deterministic-selection PATTERN as Milestone 15/16 (tag filter,
    then pick the lowest logged val_brier_15s among matches) -- not "most
    recent run" -- applied to the new `DeepEnsemble_MLP` model_type instead
    of `MLP`.
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
            "params.model_type = 'DeepEnsemble_MLP' and params.stabilization_bundle = 'True' "
            "and params.saturation_check_v2 = 'True'"
        ),
        order_by=["metrics.val_brier_15s ASC"],
    )
    if not runs:
        raise RuntimeError(
            "No DeepEnsemble_MLP run found (tagged stabilization_bundle=True, "
            "saturation_check_v2=True) -- run `python -m production.src.pipeline.train` first."
        )

    selected = runs[0]
    print(
        f"[test_uncertainty] Selected DeepEnsemble_MLP run_id={selected.info.run_id} "
        f"(lowest val_brier_15s={selected.data.metrics.get('val_brier_15s')}, "
        f"among {len(runs)} matching run(s))"
    )
    return selected.info.run_id


def _load_deep_ensemble():
    run_id = select_deterministic_deep_ensemble_run_id()
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/deep_ensemble_model")
    model.eval()

    import json
    from pathlib import Path

    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="normalization")
    json_file = next(Path(local_dir).glob("*.json"))
    with open(json_file) as f:
        stats = json.load(f)

    normalization_mean = torch.tensor(stats["mean"], dtype=torch.float32)
    normalization_std = torch.tensor(stats["std"], dtype=torch.float32)
    return model, normalization_mean, normalization_std, run_id


def _fetch_in_distribution_batch(max_samples: int = 16) -> torch.Tensor:
    """Real, in-distribution RAW feature vectors from match 3857276 --
    the first `max_samples` period-1 Pass events with an associated 360
    frame, same event-selection convention as Milestones 13-20's tests.
    """
    events = fetch_match_events(MATCH_ID)
    frames = fetch_match_360(MATCH_ID)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    raw_rows = []
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
        features_dict = extract_features(parsed)
        raw_rows.append([features_dict[key] for key in FEATURE_KEYS])
        if len(raw_rows) >= max_samples:
            break

    if not raw_rows:
        raise RuntimeError(f"no qualifying period-1 Pass events found in match {MATCH_ID}")

    return torch.tensor(raw_rows, dtype=torch.float32)


def test_deep_ensemble_diversity_sanity_check():
    """MUST run before any OOD comparison (per Step 3.2): a collapsed
    (all-members-converged-to-near-identical-weights) ensemble is an
    engineering failure, not a legitimate research finding, and would make
    every uncertainty number downstream meaningless. Hard-asserted.
    """
    model, normalization_mean, normalization_std, run_id = _load_deep_ensemble()
    print(f"\nLoaded Deep Ensemble from MLflow run_id={run_id}")

    raw_batch = _fetch_in_distribution_batch(max_samples=16)
    normalized_batch = (raw_batch - normalization_mean) / normalization_std

    with torch.no_grad():
        mean_pmf, std_cumulative_incidence, per_member_cumulative_incidence = model.predict_with_uncertainty(
            normalized_batch, time_bin=TIME_BIN
        )

    print(f"Per-sample std of per-member cumulative incidence @15s (n={len(std_cumulative_incidence)}):")
    print(std_cumulative_incidence.tolist())

    max_std = std_cumulative_incidence.max().item()
    assert max_std > DIVERSITY_STD_THRESHOLD, (
        f"Deep Ensemble appears COLLAPSED: max per-sample cross-member std of cumulative incidence "
        f"is only {max_std:.2e} (threshold: > {DIVERSITY_STD_THRESHOLD:.0e}) across "
        f"{len(std_cumulative_incidence)} real in-distribution samples. This would make every "
        "uncertainty number from this ensemble meaningless -- members converged to near-identical "
        "behavior rather than genuinely different learned beliefs."
    )
    print(f"Diversity sanity check PASSED: max per-sample std = {max_std:.6f} > {DIVERSITY_STD_THRESHOLD:.0e}")


def test_deep_ensemble_ood_uncertainty_comparison():
    """Reports (does not hard-assert) whether an explicitly-constructed OOD
    input shows higher predictive uncertainty than a real in-distribution
    match state -- see module docstring for why this is a finding, not a
    guaranteed property.
    """
    model, normalization_mean, normalization_std, run_id = _load_deep_ensemble()

    # In-distribution: first real qualifying frame from match 3857276.
    raw_id_features = _fetch_in_distribution_batch(max_samples=1)
    normalized_id_input = (raw_id_features - normalization_mean) / normalization_std

    # OOD: all NORMALIZED features set to +5 standard deviations -- a
    # clearly extreme, explicitly out-of-training-range point (training
    # features cluster near 0 in normalized space by construction; +5 std
    # is far into either tail).
    normalized_ood_input = torch.full((1, len(FEATURE_KEYS)), OOD_Z_SCORE, dtype=torch.float32)

    with torch.no_grad():
        mean_pmf_id, std_ci_id, per_member_ci_id = model.predict_with_uncertainty(
            normalized_id_input, time_bin=TIME_BIN
        )
        mean_pmf_ood, std_ci_ood, per_member_ci_ood = model.predict_with_uncertainty(
            normalized_ood_input, time_bin=TIME_BIN
        )

    threat_id = torch.cumsum(mean_pmf_id, dim=1)[0, TIME_BIN].item()
    threat_ood = torch.cumsum(mean_pmf_ood, dim=1)[0, TIME_BIN].item()

    assert math.isfinite(threat_id) and math.isfinite(threat_ood)
    assert math.isfinite(std_ci_id.item()) and math.isfinite(std_ci_ood.item())

    print(f"\n=== OOD uncertainty comparison (Deep Ensemble run_id={run_id}) ===")
    print(f"{'':<28} {'mean threat_15s':>16} {'std (uncertainty)':>18} {'per-member CI@15s'}")
    print(
        f"{'In-distribution (real)':<28} {threat_id:>16.4f} {std_ci_id.item():>18.6f} "
        f"{[round(v, 4) for v in per_member_ci_id[:, 0].tolist()]}"
    )
    print(
        f"{'OOD (+5 std, all features)':<28} {threat_ood:>16.4f} {std_ci_ood.item():>18.6f} "
        f"{[round(v, 4) for v in per_member_ci_ood[:, 0].tolist()]}"
    )

    ood_shows_higher_uncertainty = std_ci_ood.item() > std_ci_id.item()
    print(
        f"\nFinding: OOD input shows {'HIGHER' if ood_shows_higher_uncertainty else 'NOT higher'} "
        f"uncertainty than the in-distribution input ({std_ci_ood.item():.6f} vs "
        f"{std_ci_id.item():.6f}). This is reported as an empirical finding about this specific "
        "5-member, plain-MLE-trained Deep Ensemble, not asserted as a guaranteed property of "
        "Deep Ensembles in general -- a 5-member ensemble with no explicit diversity-encouraging "
        "mechanism is a small, potentially under-powered sample for detecting epistemic "
        "uncertainty growth far outside the training distribution."
    )
