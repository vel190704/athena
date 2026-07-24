"""Milestone 14 validation: GNN counterfactual re-test (RQ5), on the
newly (multi-competition) trained GNN, with graphs rebuilt from scratch
after each perturbation (move-then-rebuild, not move-and-reuse-old-edges).

Same reporting philosophy as Milestone 13's MLP test
(test_simulator.py): findings are printed, not hard-asserted. A
disagreement with football intuition here is exactly as legitimate a
result as it was for the MLP -- this test only fails on genuine
engineering problems (the trained model failing to load, shape mismatches,
NaNs), never on the model "disagreeing" with intuition.
"""

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.pytorch
import torch

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.models.evaluation import predict_cumulative_incidence_graph
from production.src.models.graph_builder import build_graph_from_frame
from production.src.pipeline.simulator import perturb_player_positions

MLFLOW_EXPERIMENT_NAME = "project-athena-deephit"
MATCH_ID = 3857276  # same cached match used in Milestone 13, for direct comparability
TIME_BIN = 3  # 15s, matching Milestone 13 and the Brier horizons used throughout


def _find_latest_gnn_run_id() -> str:
    """The most recent GNN run in the project's MLflow experiment, looked
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
        filter_string="params.model_type = 'GNN'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No GNN run found in MLflow experiment {MLFLOW_EXPERIMENT_NAME!r} -- run "
            "`python -m production.src.pipeline.train` at least once first."
        )
    return runs[0].info.run_id


def _load_trained_gnn_and_normalization_stats():
    """Loads the ACTUAL TRAINED GNN weights (mlflow.pytorch.load_model) and
    the training-split-derived graph normalization stats logged alongside
    them. No fallback to an untrained model -- if this fails, the test
    should fail loudly, not silently substitute random weights.
    """
    run_id = _find_latest_gnn_run_id()

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/gnn_model")
    model.eval()

    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="normalization")
    json_files = list(Path(local_dir).glob("*.json"))
    assert json_files, f"no normalization JSON artifact found for run {run_id}"
    with open(json_files[0]) as f:
        normalization_stats = json.load(f)

    normalization_mean = torch.tensor(normalization_stats["mean"], dtype=torch.float32)
    normalization_std = torch.tensor(normalization_stats["std"], dtype=torch.float32)

    return model, normalization_mean, normalization_std, run_id


def _fetch_baseline_frame() -> dict:
    """First real period-1 Pass event with an associated 360 frame from
    the cached match -- same match/event-selection logic as Milestone 13's
    MLP test, so the two tables are directly comparable.
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

        return parse_360_frame(event, frame_data)

    raise RuntimeError(f"no period-1 Pass event with a 360 frame found in match {MATCH_ID}")


def test_gnn_counterfactual_simulator_against_trained_model():
    model, normalization_mean, normalization_std, run_id = _load_trained_gnn_and_normalization_stats()
    print(f"\nLoaded trained GNN from MLflow run_id={run_id}")

    baseline_frame = _fetch_baseline_frame()
    player_pos = baseline_frame["player_pos"]
    player_vel = baseline_frame["player_vel"]
    is_teammate = baseline_frame["is_teammate"]
    ball_pos = baseline_frame["ball_pos"]

    scenarios = ["no_change", "high_press", "drop_deep"]
    cumulative_incidence = {}
    for action in scenarios:
        # MOVE-then-REBUILD: perturb raw positions, then rebuild the graph
        # from scratch via build_graph_from_frame -- never reuse the
        # original edge_index/edge_attr, which were derived from the
        # PRE-perturbation distances and would otherwise silently measure
        # something other than the intended counterfactual.
        perturbed_pos = perturb_player_positions(player_pos, is_teammate, ball_pos, action)
        rebuilt_graph = build_graph_from_frame(perturbed_pos, player_vel, is_teammate, ball_pos)

        ci = predict_cumulative_incidence_graph(
            model, rebuilt_graph, normalization_mean, normalization_std, time_bin=TIME_BIN
        )
        assert math.isfinite(ci), f"non-finite cumulative incidence for action={action!r}: {ci}"
        assert 0.0 <= ci <= 1.0, f"cumulative incidence out of [0,1] for action={action!r}: {ci}"
        cumulative_incidence[action] = ci

    baseline_ci = cumulative_incidence["no_change"]

    print(f"\n=== RQ5 GNN counterfactual comparison (match {MATCH_ID}, time_bin={TIME_BIN} / 15s) ===")
    print(f"{'Scenario':<12} {'Cum. Incidence':>15}")
    for action in scenarios:
        print(f"{action:<12} {cumulative_incidence[action]:>15.4f}")

    # Reported as findings, not pass/fail assertions -- a disagreement with
    # football intuition here is exactly as legitimate a result as it was
    # for the MLP (Milestone 13), not a bug. Moving nodes by a fixed offset
    # (5m/10m) is itself a heuristic, out-of-distribution perturbation --
    # see simulator.py's perturb_player_positions docstring -- so graph-
    # based perturbation is not automatically more realistic than the
    # scalar version just because it operates on raw positions.
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
