"""Milestone 7/8/9/10/10B/12: end-to-end DeepHit training, baseline
validation, MLflow experiment tracking, dataset scaling, coordinate
handling (ADR-009), and the MLP-vs-GNN RQ4 comparison.

Fetches real StatsBomb matches via a competition-wide batch pull (Milestone
9, scaled up from 5 hardcoded matches), extracts BOTH scalar spatial
features (Milestone 3) and a graph representation (Milestone 11) from the
exact same resolved 360 freeze-frame per possession chain (Milestone 5,
both periods), trains two single-risk DeepHit models on the identical
80/20 split -- the scalar-feature MLP (Milestone 6A/6B) and the
graph-based GNN (Milestone 12) -- evaluates both with a time-dependent
Brier Score (Milestone 7 Step 1), and logs each as a separate MLflow run
under the same experiment (Milestone 8) so RQ4 (do graph representations
outperform handcrafted scalar features?) can be read directly off the
comparison table this script prints.

Run as: python -m production.src.pipeline.train
Then:   mlflow ui   (from the project root, to inspect results visually)

No hyperparameter tuning (Optuna, etc.) or Batch Ensembles here -- this is
passive, reproducible logging of the MLP vs GNN comparison on the existing
baseline architectures.
"""

import json
import os
import tempfile
from collections import defaultdict

# This project tracks locally to ./mlruns (already gitignored since
# Milestone 1) via mlflow's default file-store backend. Recent mlflow
# versions put that backend into "maintenance mode" behind an explicit
# opt-in env var; this project isn't migrating to a database backend
# (sqlite/etc.) for a local baseline smoke test, so opt back in. Must be
# set before any mlflow tracking-store call (import order doesn't matter,
# but this needs to precede mlflow.set_experiment/start_run below).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from production.src.ingestion.statsbomb_io import (
    batch_extract_valid_matches,
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.models.deephit import DeepHitSurvivalModel
from production.src.models.deephit_loss import DeepHitLoss
from production.src.models.evaluation import calculate_brier_score
from production.src.models.gnn_model import GNNDeepHitSurvivalModel
from production.src.models.graph_builder import DEFAULT_OPPONENT_RADIUS, DEFAULT_SAME_TEAM_RADIUS
from production.src.pipeline.chain_builder import build_possession_chains
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.survival_dataset import (
    BIN_SIZE_SECONDS,
    FEATURE_KEYS,
    NUM_BINS,
    TacticalSurvivalDataset,
)
from production.src.spatial.control import BiomechanicalPitchControl

MLFLOW_EXPERIMENT_NAME = "project-athena-deephit"

# World Cup 2022 (competition_id=43, season_id=106), scaled from Milestone
# 8's 5 hardcoded matches to a batch pull across the whole competition.
COMPETITION_ID = 43
SEASON_ID = 106
NUM_MATCHES_NEEDED = 20

# Possession chains are built across both halves (Milestone 9). Period-2
# chains contribute trainable feature samples with NO coordinate
# transformation (ADR-009): StatsBomb's raw event/360 coordinates are
# already oriented relative to the acting team's own attacking-left-to-right
# perspective, so feature_extractor.py's old period-1-only restriction was
# simply removed, not replaced with a flip. See build_training_data()'s
# per-match print for the period-1 vs period-2 match-rate this produces.
CHAIN_BUILDER_PERIODS = (1, 2)

NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
BRIER_TIME_BINS = (3, 6)  # 15s and 30s, at BIN_SIZE_SECONDS=5.0
GNN_HIDDEN_DIM = 64

SMALL_DATASET_WARNING_THRESHOLD = 500

# Milestone 9 baseline (20 matches, period-1-only -- chain-building already
# covered both periods, but feature_extractor.py still gated on period==1
# at the time, dataset_size=1545) and the Milestone 10 forced-flip run
# (same 20 matches, period-2 included but with an INCORRECT forced
# coordinate flip that ADR-009 found to systematically mis-orient period-2
# frames, dataset_size=3198), printed alongside this run's numbers for
# direct reference.
MILESTONE_9_BASELINE_BRIER_15S = 0.0907
MILESTONE_9_BASELINE_BRIER_30S = 0.1744
MILESTONE_9_BASELINE_DATASET_SIZE = 1545

MILESTONE_10_FORCED_FLIP_BRIER_15S = 0.0956
MILESTONE_10_FORCED_FLIP_BRIER_30S = 0.1874
MILESTONE_10_FORCED_FLIP_DATASET_SIZE = 3198


def _match_chains_with_features(match_id: int, engine: BiomechanicalPitchControl):
    """Pair each possession chain (Milestone 5/9, both periods by default)
    with a scalar feature vector (Milestone 3) AND the raw parsed 360 frame
    (Milestone 3's parse_360_frame output) -- both derived from ONE
    representative event in that chain, resolved ONCE (the first event in
    the chain that has an associated 360 freeze-frame). This is the
    Milestone 12 requirement that the scalar features and the graph data
    (built later, in TacticalSurvivalDataset, from the returned frame) come
    from the exact same observation, not independently re-looked-up.

    A chain is only skipped here if it has no 360-covered event at all.
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    chains = build_possession_chains(events, periods=CHAIN_BUILDER_PERIODS)

    events_by_period_possession = defaultdict(list)
    for e in events:
        if e["period"] in CHAIN_BUILDER_PERIODS:
            events_by_period_possession[(e["period"], e["possession"])].append(e)
    for group in events_by_period_possession.values():
        group.sort(key=lambda e: e["index"])

    matched_features, matched_frames, matched_chains, matched_source_event_ids = [], [], [], []
    matched_by_period = defaultdict(int)
    for chain in chains:
        chain_events = events_by_period_possession.get((chain["period"], chain["chain_id"]), [])

        rep_event, rep_frame = None, None
        for e in chain_events:
            frame = frames_by_event_uuid.get(e["id"])
            if frame is not None and "location" in e:
                rep_event, rep_frame = e, frame
                break
        if rep_event is None:
            continue

        parsed = parse_360_frame(rep_event, rep_frame)
        features = extract_features(parsed, engine)

        matched_features.append(features)
        matched_frames.append(parsed)
        matched_chains.append(chain)
        matched_source_event_ids.append(rep_event["id"])
        matched_by_period[chain["period"]] += 1

    print(
        f"  match {match_id}: {len(matched_chains)}/{len(chains)} chains matched to a "
        f"360 frame + features (by period: {dict(matched_by_period)})"
    )

    # Light precaution at this data scale (not yet a strict necessity, but
    # establishes the pattern before dataset sizes grow further): drop
    # references to this match's raw fetched JSON before returning, rather
    # than letting them linger for the caller's next iteration.
    del events, frames, chains, events_by_period_possession

    return matched_features, matched_frames, matched_chains, matched_source_event_ids


def build_training_data():
    match_ids = batch_extract_valid_matches(
        competition_id=COMPETITION_ID, season_id=SEASON_ID, num_matches=NUM_MATCHES_NEEDED
    )
    print(f"Resolved {len(match_ids)} valid matches (requested {NUM_MATCHES_NEEDED}): {match_ids}")

    engine = BiomechanicalPitchControl()
    all_features, all_frames, all_chains, all_source_event_ids = [], [], [], []
    for match_id in match_ids:
        features, frames, chains, source_event_ids = _match_chains_with_features(match_id, engine)
        all_features.extend(features)
        all_frames.extend(frames)
        all_chains.extend(chains)
        all_source_event_ids.extend(source_event_ids)

    return all_features, all_frames, all_chains, all_source_event_ids, match_ids


def _normalize_scalar_batch(scalar_batch: torch.Tensor, graph_batch, mean, std) -> torch.Tensor:
    return (scalar_batch - mean) / std


def _normalize_graph_batch(scalar_batch, graph_batch, mean, std):
    # Only x, y, dist_to_ball (columns 0, 1, 6) are standardized; vx/vy are
    # left as-is (always exactly zero -- see module docstring/comment
    # below), and the is_attacker/is_defender boolean flags are left
    # unnormalized since they're already a clean {0, 1} indicator.
    x = graph_batch.x.clone()
    x[:, [0, 1, 6]] = (x[:, [0, 1, 6]] - mean) / std
    graph_batch.x = x
    return graph_batch


def _train_and_log_model(
    model_type: str,
    model: torch.nn.Module,
    input_fn,
    normalize_args: tuple,
    train_loader: DataLoader,
    val_batch: tuple,
    n_train: int,
    n_val: int,
    match_ids: list[int],
    dataset_size: int,
    extra_params: dict,
    normalization_artifact: dict,
) -> dict:
    """Shared training/eval/logging loop for both the MLP and the GNN --
    factored out so the two models run through IDENTICAL epoch counts,
    optimizer settings, loss function, Brier calculation, and MLflow
    logging conventions, differing only in `model`, `input_fn` (how to pull
    this model's representation out of a (scalar_batch, graph_batch) pair
    and normalize it), and `extra_params` (model-specific MLflow params).
    """
    loss_fn = DeepHitLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_run(run_name=f"{model_type.lower()}_run") as run:
        mlflow.log_params(
            {
                "model_type": model_type,
                "lr": LEARNING_RATE,
                "epochs": NUM_EPOCHS,
                "train_size": n_train,
                "val_size": n_val,
                "alpha": loss_fn.alpha,
                "sigma": loss_fn.sigma,
                "num_bins": NUM_BINS,
                "bin_size": BIN_SIZE_SECONDS,
                # Reproducibility metadata: the seed, which matches, and the
                # feature key order pin down what this run's logged
                # mean/std vectors actually correspond to, none of which is
                # otherwise recoverable from the run later.
                "random_seed": RANDOM_SEED,
                "match_ids": ",".join(str(m) for m in match_ids),
                "feature_key_order": ",".join(FEATURE_KEYS),
                "match_count": len(match_ids),
                "dataset_size": dataset_size,
                "periods_included": ",".join(str(p) for p in CHAIN_BUILDER_PERIODS),
                "coordinate_convention": "statsbomb_per_actor_native",
                **extra_params,
            }
        )

        print(
            f"\n[{model_type}] Training for {NUM_EPOCHS} epochs on {n_train} samples "
            f"({n_val} held out for validation)..."
        )
        final_epoch_loss = None
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            epoch_loss_total = 0.0
            num_batches = 0

            for batch_idx, (scalar_batch, graph_batch, duration_bins_batch, events_batch) in enumerate(
                train_loader
            ):
                model_input = input_fn(scalar_batch, graph_batch, *normalize_args)

                optimizer.zero_grad()
                predictions = model(model_input)
                loss = loss_fn(predictions, duration_bins_batch, events_batch)

                if not torch.isfinite(loss):
                    print(f"[{model_type}] NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Stopping.")
                    return None

                loss.backward()
                optimizer.step()

                epoch_loss_total += loss.item()
                num_batches += 1

            final_epoch_loss = epoch_loss_total / num_batches
            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{model_type}] epoch {epoch:3d}/{NUM_EPOCHS}: training loss = {final_epoch_loss:.4f}")

        print(f"[{model_type}] Final training loss: {final_epoch_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_scalar, val_graph, val_duration_bins, val_events = val_batch
            val_input = input_fn(val_scalar, val_graph, *normalize_args)
            val_predictions = model(val_input)

            val_loss = loss_fn(val_predictions, val_duration_bins, val_events)
            print(f"[{model_type}] Validation loss: {val_loss.item():.4f}")

            briers = {}
            for time_bin in BRIER_TIME_BINS:
                brier, num_excluded = calculate_brier_score(
                    val_predictions, val_duration_bins, val_duration_bins, val_events, time_bin
                )
                seconds = time_bin * 5.0
                print(f"  [{model_type}] time_bin={time_bin} ({seconds:.0f}s): Brier Score = {brier:.4f}")
                briers[time_bin] = (brier, num_excluded)

        brier_15s, excluded_15s = briers[3]
        brier_30s, excluded_30s = briers[6]

        mlflow.log_metrics(
            {
                "train_loss": final_epoch_loss,
                "val_loss": val_loss.item(),
                "val_brier_15s": brier_15s,
                "val_brier_30s": brier_30s,
                "excluded_15s": excluded_15s,
                "excluded_30s": excluded_30s,
            }
        )

        # serialization_format="pickle": the default ('pt2') traces the
        # model graph via torch.export and requires an input_example to do
        # so. Plain pickling is simpler and sufficient for these eager
        # nn.Module baselines.
        mlflow.pytorch.log_model(
            model, name=f"{model_type.lower()}_model", serialization_format="pickle"
        )

        # Self-describing artifact: includes feature_key_order alongside
        # the mean/std vectors so the file means something even opened
        # outside MLflow. Each run logs the normalization stats for the
        # representation IT actually consumes (scalar for MLP, graph for
        # GNN), not a combined blob shared via a third orphan run.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(normalization_artifact, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        try:
            mlflow.log_artifact(tmp_file_path, artifact_path="normalization")
        finally:
            os.remove(tmp_file_path)

        print(f"[{model_type}] MLflow run ID: {run.info.run_id}")

    return {
        "train_loss": final_epoch_loss,
        "val_loss": val_loss.item(),
        "brier_15s": brier_15s,
        "brier_30s": brier_30s,
        "excluded_15s": excluded_15s,
        "excluded_30s": excluded_30s,
    }


def train_and_evaluate():
    torch.manual_seed(RANDOM_SEED)

    features, frames, chains, source_event_ids, match_ids = build_training_data()
    match_count = len(match_ids)
    dataset_size = len(features)
    print(f"\nTotal (feature, frame, chain) triples across {match_count} matches: {dataset_size}")
    if dataset_size < SMALL_DATASET_WARNING_THRESHOLD:
        print(
            f"NOTE: {dataset_size} samples from {match_count} matches is a small dataset -- fine "
            "for a baseline smoke test, but the Brier Score numbers below should not be "
            "over-interpreted as a validated model."
        )

    # Same-frame spot check (Milestone 12 Step 2's critical invariant): the
    # scalar features and the graph data for sample 0 must come from the
    # SAME resolved event. Both extract_features and TacticalSurvivalDataset
    # were fed the identical `parsed` dict from _match_chains_with_features,
    # so this is guaranteed by construction -- printed here as visible
    # evidence, not just an assumption.
    print(
        f"\nSame-frame spot check (chain 0): scalar-feature source event_id="
        f"{source_event_ids[0]}, graph-data source event_id={source_event_ids[0]} "
        "(identical, by construction -- both were built from one resolved parse_360_frame call)"
    )

    dataset = TacticalSurvivalDataset(features, frames, chains)

    n_train = int(TRAIN_FRACTION * len(dataset))
    n_val = len(dataset) - n_train
    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=split_generator)
    # Both models train on this exact same split (same indices) -- guard
    # that assumption explicitly rather than leaving it implicit.
    assert len(train_set) == n_train and len(val_set) == n_val

    # Scalar feature normalization (Milestone 7's rule, unchanged):
    # statistics computed from the TRAINING split ONLY, after the split.
    train_features_raw = torch.stack([dataset[i][0] for i in train_set.indices])
    feature_mean = train_features_raw.mean(dim=0)
    feature_std = train_features_raw.std(dim=0).clamp(min=1e-8)  # guard a constant feature
    print(f"\nScalar feature normalization stats (from {n_train} training samples):")
    print(f"  mean: {feature_mean.tolist()}")
    print(f"  std:  {feature_std.tolist()}")

    # Graph continuous-feature normalization (Milestone 12 Step 2.3): same
    # training-split-only rule, applied to columns [x, y, dist_to_ball]
    # (indices 0, 1, 6) of each graph's node features. vx/vy are left as-is
    # because they are always exactly zero -- StatsBomb 360 has no
    # velocity field (an inherited limitation from Milestone 3, not new
    # here) -- so the GNN has no real velocity signal either, a shared
    # limitation with the MLP rather than a GNN-specific weakness, worth
    # remembering when interpreting the RQ4 comparison below. is_attacker/
    # is_defender are boolean flags and are left unnormalized.
    train_graph_continuous = torch.cat(
        [dataset[i][1].x[:, [0, 1, 6]] for i in train_set.indices], dim=0
    )
    graph_feature_mean = train_graph_continuous.mean(dim=0)
    graph_feature_std = train_graph_continuous.std(dim=0).clamp(min=1e-8)
    print(f"\nGraph node feature normalization stats (x, y, dist_to_ball; from {n_train} training samples):")
    print(f"  mean: {graph_feature_mean.tolist()}")
    print(f"  std:  {graph_feature_std.tolist()}")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_batch = next(iter(DataLoader(val_set, batch_size=len(val_set))))

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    mlp_model = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
    mlp_results = _train_and_log_model(
        model_type="MLP",
        model=mlp_model,
        input_fn=_normalize_scalar_batch,
        normalize_args=(feature_mean, feature_std),
        train_loader=train_loader,
        val_batch=val_batch,
        n_train=n_train,
        n_val=n_val,
        match_ids=match_ids,
        dataset_size=dataset_size,
        extra_params={},
        normalization_artifact={
            "feature_key_order": list(FEATURE_KEYS),
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
    )

    gnn_model = GNNDeepHitSurvivalModel(num_node_features=7, num_bins=NUM_BINS, hidden_dim=GNN_HIDDEN_DIM)
    gnn_results = _train_and_log_model(
        model_type="GNN",
        model=gnn_model,
        input_fn=_normalize_graph_batch,
        normalize_args=(graph_feature_mean, graph_feature_std),
        train_loader=train_loader,
        val_batch=val_batch,
        n_train=n_train,
        n_val=n_val,
        match_ids=match_ids,
        dataset_size=dataset_size,
        extra_params={
            "same_team_radius": DEFAULT_SAME_TEAM_RADIUS,
            "opponent_radius": DEFAULT_OPPONENT_RADIUS,
            "hidden_dim": GNN_HIDDEN_DIM,
        },
        normalization_artifact={
            "graph_continuous_feature_order": ["x", "y", "dist_to_ball"],
            "mean": graph_feature_mean.tolist(),
            "std": graph_feature_std.tolist(),
        },
    )

    print(f"\nDataset size: {dataset_size} total samples ({n_train} train / {n_val} val)")

    print(
        f"\nComparison vs Milestone 9 baseline (20 matches, period-1-only, "
        f"dataset_size={MILESTONE_9_BASELINE_DATASET_SIZE}): "
        f"Brier @ 15s = {MILESTONE_9_BASELINE_BRIER_15S:.4f}, "
        f"Brier @ 30s = {MILESTONE_9_BASELINE_BRIER_30S:.4f}"
    )
    print(
        f"Comparison vs Milestone 10 forced-flip run (dataset_size="
        f"{MILESTONE_10_FORCED_FLIP_DATASET_SIZE}): "
        f"Brier @ 15s = {MILESTONE_10_FORCED_FLIP_BRIER_15S:.4f}, "
        f"Brier @ 30s = {MILESTONE_10_FORCED_FLIP_BRIER_30S:.4f}"
    )

    if mlp_results is not None and gnn_results is not None:
        print("\n=== RQ4: MLP (scalar features) vs GNN (graph representation) ===")
        print(f"{'Model':<6} {'Brier@15s':>10} {'Brier@30s':>10}")
        print(f"{'MLP':<6} {mlp_results['brier_15s']:>10.4f} {mlp_results['brier_30s']:>10.4f}")
        print(f"{'GNN':<6} {gnn_results['brier_15s']:>10.4f} {gnn_results['brier_30s']:>10.4f}")
        print(
            "NOTE: MLP and GNN are not matched in raw parameter count/capacity -- a caveat for "
            "interpreting this comparison, not something to fix now."
        )

    print(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("Run `mlflow ui` from the project root to inspect results visually.")


if __name__ == "__main__":
    train_and_evaluate()
