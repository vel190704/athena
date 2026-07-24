"""Milestone 13 validation: the tactical counterfactual simulator (Module
8 / RQ5), run against the ACTUAL TRAINED MLP -- not a freshly-initialized
one. Testing tactical intuition against an untrained model would just be
comparing random noise to football knowledge, so loading the real trained
weights from MLflow is a hard requirement, not a preference.

RQ5 is an open research question here, not an engineering assertion: this
test reports whether the model's predicted cumulative incidence shifts in
the direction real football intuition would expect for two tactical
actions, WITHOUT hard-asserting either direction. It only fails on genuine
engineering problems -- the trained model failing to load, shape
mismatches, or NaNs -- never on the model "disagreeing" with intuition.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import math

import mlflow
import mlflow.pytorch
import torch

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.models.evaluation import predict_cumulative_incidence
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.simulator import perturb_features

MLFLOW_EXPERIMENT_NAME = "project-athena-deephit"
MATCH_ID = 3857276  # cached real StatsBomb open-data match
TIME_BIN = 3  # 15s, matching the horizon used throughout Milestones 7-12B


def _find_latest_mlp_run_id() -> str:
    """The most recent MLP run in the project's MLflow experiment. Looked
    up dynamically (not hardcoded) so this test keeps working against
    whichever training run is actually most recent.
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
        filter_string="params.model_type = 'MLP'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No MLP run found in MLflow experiment {MLFLOW_EXPERIMENT_NAME!r} -- run "
            "`python -m production.src.pipeline.train` at least once first."
        )
    return runs[0].info.run_id


def _load_trained_mlp_and_normalization_stats():
    """Loads the ACTUAL TRAINED model weights (mlflow.pytorch.load_model)
    and the training-split-derived normalization stats logged alongside
    them. No fallback to an untrained model -- if this fails, the test
    should fail loudly, not silently substitute random weights.
    """
    run_id = _find_latest_mlp_run_id()

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/mlp_model")
    model.eval()

    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="normalization")
    json_files = list(Path(local_dir).glob("*.json"))
    assert json_files, f"no normalization JSON artifact found for run {run_id}"
    with open(json_files[0]) as f:
        normalization_stats = json.load(f)

    normalization_mean = torch.tensor(normalization_stats["mean"], dtype=torch.float32)
    normalization_std = torch.tensor(normalization_stats["std"], dtype=torch.float32)

    return model, normalization_mean, normalization_std, run_id


def _fetch_baseline_features() -> dict:
    """First real period-1 Pass event with an associated 360 frame from
    the cached match, run through the existing extraction pipeline.
    """
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


def test_counterfactual_simulator_against_trained_model():
    model, normalization_mean, normalization_std, run_id = _load_trained_mlp_and_normalization_stats()
    print(f"\nLoaded trained MLP from MLflow run_id={run_id}")

    baseline_features = _fetch_baseline_features()

    scenarios = ["no_change", "force_wide", "high_press", "drop_deep"]
    cumulative_incidence = {}
    for action in scenarios:
        perturbed = perturb_features(baseline_features, action)
        ci = predict_cumulative_incidence(
            model, perturbed, normalization_mean, normalization_std, time_bin=TIME_BIN
        )
        assert math.isfinite(ci), f"non-finite cumulative incidence for action={action!r}: {ci}"
        assert 0.0 <= ci <= 1.0, f"cumulative incidence out of [0,1] for action={action!r}: {ci}"
        cumulative_incidence[action] = ci

    baseline_ci = cumulative_incidence["no_change"]

    print(f"\n=== RQ5 counterfactual comparison (match {MATCH_ID}, time_bin={TIME_BIN} / 15s) ===")
    print(f"{'Scenario':<12} {'Cum. Incidence':>15}")
    for action in scenarios:
        print(f"{action:<12} {cumulative_incidence[action]:>15.4f}")

    # Reported as findings, not pass/fail assertions (Step 3.5): a
    # disagreement with football intuition here is a legitimate research
    # result (small dataset, out-of-distribution perturbed inputs per
    # simulator.py's caveats, or a genuine limitation of 4 handcrafted
    # features), not a bug.
    high_press_aligned = cumulative_incidence["high_press"] > baseline_ci
    drop_deep_aligned = cumulative_incidence["drop_deep"] < baseline_ci

    print(
        f"\nhigh_press cumulative incidence ({cumulative_incidence['high_press']:.4f}) > "
        f"baseline ({baseline_ci:.4f}): {high_press_aligned} "
        "(football intuition: pressing high should INCREASE near-term goal threat)"
    )
    print(
        f"drop_deep cumulative incidence ({cumulative_incidence['drop_deep']:.4f}) < "
        f"baseline ({baseline_ci:.4f}): {drop_deep_aligned} "
        "(football intuition: dropping deep should DECREASE near-term goal threat)"
    )
