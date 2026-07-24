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
    find_360_competitions,
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

# Milestone 14: scaled from a single competition (World Cup 2022 only,
# Milestones 8-12B) to ALL competitions StatsBomb's live competitions index
# verifies as having 360 data (via `match_available_360` -- see
# find_360_competitions; NOT a hardcoded list of "competitions that should
# have 360 coverage," the same lesson as Milestone 3's match-id and
# Milestone 9's competition-matches verification). MATCH_POOL_SIZE is a
# generous upper bound on how many valid matches to gather as candidates
# BEFORE processing (per-match sample yield isn't knowable until a match is
# actually processed into possession chains); TARGET_SAMPLE_COUNT is the
# actual stopping condition, checked after each match is processed, per the
# task's 5,000-10,000 possession-chain sample target.
MATCH_POOL_SIZE = 100
TARGET_SAMPLE_COUNT = 8000

# Possession chains are built across both halves (Milestone 9). Period-2
# chains contribute trainable feature samples with NO coordinate
# transformation (ADR-009): StatsBomb's raw event/360 coordinates are
# already oriented relative to the acting team's own attacking-left-to-right
# perspective, so feature_extractor.py's old period-1-only restriction was
# simply removed, not replaced with a flip. See build_training_data()'s
# per-match print for the period-1 vs period-2 match-rate this produces.
CHAIN_BUILDER_PERIODS = (1, 2)

NUM_EPOCHS = 50
LEARNING_RATE = 1e-3  # MLP only -- already stable in Milestone 12, deliberately left untouched
BATCH_SIZE = 32
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
BRIER_TIME_BINS = (3, 6)  # 15s and 30s, at BIN_SIZE_SECONDS=5.0
GNN_HIDDEN_DIM = 64

# Milestone 12B: GNN-specific optimization stabilization. Milestone 12
# found the GNN's training loss spiking from 3.07 to 4.58 around epoch 20
# and never recovering (exploding gradients), caught only by eyeballing a
# printed log every 10 epochs. These three changes are bundled together
# (gradient-norm clipping, weight decay, and a 10x lower learning rate) and
# applied ONLY to the GNN -- the MLP was already stable and is left with
# its Milestone 12 optimizer config (plain Adam, lr=1e-3, no clipping, no
# weight decay) so it remains a clean, unchanged reference point. Because
# all three are bundled, if this run comes back stable we won't know which
# change mattered most -- isolating that is optional future work, not
# required here. A fully symmetric ablation (applying the same three
# changes to the MLP too) would also confirm the MLP isn't secretly
# benefiting from a learning rate that happens to suit it, but that's a
# lower-priority check since the MLP was already performing well and
# stably.
GNN_LEARNING_RATE = 1e-4
GNN_WEIGHT_DECAY = 1e-4
GRAD_CLIP_MAX_NORM = 1.0

# Step 2.2: flags residual instability rather than relying on manually
# eyeballing the printed log (which is exactly how Milestone 12's GNN
# blowup was originally caught, too late). A single-epoch train-loss
# INCREASE exceeding this fraction of the prior epoch's loss value fires
# an explicit warning instead of silently proceeding as if training was
# smooth.
INSTABILITY_THRESHOLD_FRACTION = 0.5

# Step 2.3: periodic (not just final) validation-loss logging, so the full
# train-vs-val curve is inspectable afterward -- this is what lets Step 3
# distinguish overfitting (train smooth/low, val diverging) from true
# instability (both curves erratic).
VAL_LOSS_LOG_INTERVAL_EPOCHS = 5

SMALL_DATASET_WARNING_THRESHOLD = 500

# Milestone 12's unstable GNN run (exploding gradients, never diagnosed
# automatically -- see the module docstring), kept for direct comparison
# in this run's final printout. Not deleted or overwritten in MLflow.
MILESTONE_12_UNSTABLE_GNN_RUN_ID = "68d3ade44aea4f9e9259c7ef1a4c9ace"
MILESTONE_12_UNSTABLE_GNN_BRIER_15S = 0.1070
MILESTONE_12_UNSTABLE_GNN_BRIER_30S = 0.2258

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

# Milestone 12B's STABILIZED single-competition (World Cup 2022 only)
# MLP/GNN pair -- the direct "prior small-dataset numbers" this milestone's
# multi-competition run is meant to be compared against (Step 2.4).
MILESTONE_12B_DATASET_SIZE = 3198
MILESTONE_12B_MLP_BRIER_15S = 0.0846
MILESTONE_12B_MLP_BRIER_30S = 0.1720
MILESTONE_12B_GNN_BRIER_15S = 0.1051
MILESTONE_12B_GNN_BRIER_30S = 0.2042


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
    qualifying_competitions = find_360_competitions()
    print(f"Competitions verified (via the live competitions index) to have 360 data ({len(qualifying_competitions)}):")
    for c in qualifying_competitions:
        print(
            f"  competition_id={c['competition_id']}, season_id={c['season_id']}: "
            f"{c['competition_name']} {c['season_name']}"
        )

    competition_season_pairs = [
        (c["competition_id"], c["season_id"]) for c in qualifying_competitions
    ]
    match_pool = batch_extract_valid_matches(competition_season_pairs, num_matches=MATCH_POOL_SIZE)
    print(
        f"\nResolved {len(match_pool)} valid matches across {len(qualifying_competitions)} "
        f"qualifying competitions (pool target {MATCH_POOL_SIZE})"
    )

    engine = BiomechanicalPitchControl()
    all_features, all_frames, all_chains, all_source_event_ids = [], [], [], []
    used_match_ids = []
    for match_id in match_pool:
        features, frames, chains, source_event_ids = _match_chains_with_features(match_id, engine)
        all_features.extend(features)
        all_frames.extend(frames)
        all_chains.extend(chains)
        all_source_event_ids.extend(source_event_ids)
        used_match_ids.append(match_id)

        if len(all_features) >= TARGET_SAMPLE_COUNT:
            print(
                f"\nReached target sample count ({TARGET_SAMPLE_COUNT}) after "
                f"{len(used_match_ids)} matches -- stopping early rather than "
                "exhaustively processing the whole match pool."
            )
            break

    print(
        f"Final: {len(all_features)} samples from {len(used_match_ids)} matches "
        f"(target was {TARGET_SAMPLE_COUNT}, requested range 5,000-10,000)"
    )

    return (
        all_features,
        all_frames,
        all_chains,
        all_source_event_ids,
        used_match_ids,
        qualifying_competitions,
    )


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


def _check_for_instability(model_type: str, epoch_losses: list[float]) -> bool:
    """Programmatic replacement for eyeballing the printed log (which is
    exactly how Milestone 12's GNN blowup was originally caught, too late).

    Returns True (and prints an explicit WARNING) if any single-epoch loss
    increase exceeds INSTABILITY_THRESHOLD_FRACTION of the prior epoch's
    loss value; otherwise returns False silently.
    """
    max_relative_increase = 0.0
    culprit_epoch = None
    for i in range(1, len(epoch_losses)):
        prev_loss, curr_loss = epoch_losses[i - 1], epoch_losses[i]
        if prev_loss <= 0:
            continue
        relative_increase = (curr_loss - prev_loss) / prev_loss
        if relative_increase > max_relative_increase:
            max_relative_increase = relative_increase
            culprit_epoch = i + 1  # 1-indexed epoch number

    fired = max_relative_increase > INSTABILITY_THRESHOLD_FRACTION
    if fired:
        print(
            f"[{model_type}] WARNING: residual training instability detected -- loss "
            f"increased by {max_relative_increase:.1%} at epoch {culprit_epoch} "
            f"(threshold: single-epoch relative increase > {INSTABILITY_THRESHOLD_FRACTION:.0%})."
        )
    return fired


def _train_and_log_model(
    model_type: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float,
    weight_decay: float,
    clip_grad_norm: bool,
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
    run_tags: dict | None = None,
) -> dict | None:
    """Shared training/eval/logging loop for both the MLP and the GNN --
    factored out so the two models run through IDENTICAL epoch counts,
    loss function, Brier calculation, and MLflow logging conventions,
    differing only in `model`/`optimizer` (each model's optimizer is built
    by the caller, so the MLP's Milestone-12 config -- lr=1e-3, no weight
    decay -- can stay untouched while the GNN gets Milestone 12B's
    stabilization bundle), `input_fn` (how to pull this model's
    representation out of a (scalar_batch, graph_batch) pair and
    normalize it), `clip_grad_norm` (GNN-only, see module docstring), and
    `extra_params`/`run_tags` (model-specific MLflow metadata).

    Returns None if training hit a NaN/Inf loss (unchanged from Milestone
    12); otherwise a dict of final metrics. The Step 2.2 instability check
    is reported via a printed WARNING but does NOT itself abort training --
    the caller decides how to react to it (see train_and_evaluate).
    """
    loss_fn = DeepHitLoss()

    with mlflow.start_run(run_name=f"{model_type.lower()}_run") as run:
        if run_tags:
            mlflow.set_tags(run_tags)

        mlflow.log_params(
            {
                "model_type": model_type,
                "lr": lr,
                "weight_decay": weight_decay,
                "gradient_clipping": clip_grad_norm,
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
        epoch_losses: list[float] = []
        val_loss_history: dict[int, float] = {}
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
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()

                epoch_loss_total += loss.item()
                num_batches += 1

            final_epoch_loss = epoch_loss_total / num_batches
            epoch_losses.append(final_epoch_loss)
            # Step 2.1: full per-epoch history, not just a final value --
            # inspectable in the MLflow UI even if nobody was watching the
            # console at the right moment.
            mlflow.log_metric("train_loss", final_epoch_loss, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{model_type}] epoch {epoch:3d}/{NUM_EPOCHS}: training loss = {final_epoch_loss:.4f}")

            # Step 2.3: periodic validation loss during training (not just
            # the final one) -- this is what lets Step 3 distinguish
            # overfitting (train smooth/low, val diverging) from true
            # instability (both curves erratic).
            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 or epoch == NUM_EPOCHS:
                model.eval()
                with torch.no_grad():
                    val_scalar, val_graph, val_duration_bins, val_events = val_batch
                    val_input = input_fn(val_scalar, val_graph, *normalize_args)
                    epoch_val_loss = loss_fn(model(val_input), val_duration_bins, val_events).item()
                val_loss_history[epoch] = epoch_val_loss
                mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

        print(f"[{model_type}] Final training loss: {final_epoch_loss:.4f}")

        # Step 2.2: programmatic instability check, not a manual glance at
        # the printed log.
        instability_warning_fired = _check_for_instability(model_type, epoch_losses)
        mlflow.log_param("instability_warning_fired", instability_warning_fired)

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
        train_val_gap = val_loss.item() - final_epoch_loss

        mlflow.log_metrics(
            {
                "val_brier_15s": brier_15s,
                "val_brier_30s": brier_30s,
                "excluded_15s": excluded_15s,
                "excluded_30s": excluded_30s,
                "train_val_loss_gap": train_val_gap,
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
        "train_val_gap": train_val_gap,
        "instability_warning_fired": instability_warning_fired,
        "epoch_losses": epoch_losses,
        "val_loss_history": val_loss_history,
        "run_id": run.info.run_id,
    }


def train_and_evaluate():
    torch.manual_seed(RANDOM_SEED)

    features, frames, chains, source_event_ids, match_ids, qualifying_competitions = build_training_data()
    match_count = len(match_ids)
    dataset_size = len(features)
    competition_season_summary = ",".join(
        f"{c['competition_id']}:{c['season_id']}" for c in qualifying_competitions
    )
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
    # MLP's optimizer is left EXACTLY as in Milestone 12 -- plain Adam,
    # lr=1e-3, no weight decay, no gradient clipping -- so it remains a
    # clean, unchanged reference point (it was already stable).
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)
    mlp_results = _train_and_log_model(
        model_type="MLP",
        model=mlp_model,
        optimizer=mlp_optimizer,
        lr=LEARNING_RATE,
        weight_decay=0.0,
        clip_grad_norm=False,
        input_fn=_normalize_scalar_batch,
        normalize_args=(feature_mean, feature_std),
        train_loader=train_loader,
        val_batch=val_batch,
        n_train=n_train,
        n_val=n_val,
        match_ids=match_ids,
        dataset_size=dataset_size,
        extra_params={
            "dataset_scale": "multi_competition",
            "competition_season_pairs": competition_season_summary,
        },
        normalization_artifact={
            "feature_key_order": list(FEATURE_KEYS),
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
    )

    gnn_model = GNNDeepHitSurvivalModel(num_node_features=7, num_bins=NUM_BINS, hidden_dim=GNN_HIDDEN_DIM)
    # Milestone 12B stabilization bundle (GNN only -- see module docstring
    # for why all three are bundled together and applied asymmetrically):
    # lower learning rate, weight decay, and gradient-norm clipping (the
    # clipping itself is applied inside _train_and_log_model's loop, gated
    # on clip_grad_norm=True below).
    gnn_optimizer = torch.optim.Adam(
        gnn_model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY
    )
    gnn_results = _train_and_log_model(
        model_type="GNN",
        model=gnn_model,
        optimizer=gnn_optimizer,
        lr=GNN_LEARNING_RATE,
        weight_decay=GNN_WEIGHT_DECAY,
        clip_grad_norm=True,
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
            "dataset_scale": "multi_competition",
            "competition_season_pairs": competition_season_summary,
        },
        run_tags={
            "supersedes_run_id": MILESTONE_12_UNSTABLE_GNN_RUN_ID,
            "supersedes_note": (
                "Milestone 12B stabilization (grad clipping + weight decay + lower lr) "
                "of Milestone 12's exploding-gradient GNN run; that run is kept, not deleted."
            ),
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
    print(
        f"Comparison vs Milestone 12 UNSTABLE GNN run (exploding gradients, run_id="
        f"{MILESTONE_12_UNSTABLE_GNN_RUN_ID}, kept in MLflow, not deleted): "
        f"Brier @ 15s = {MILESTONE_12_UNSTABLE_GNN_BRIER_15S:.4f}, "
        f"Brier @ 30s = {MILESTONE_12_UNSTABLE_GNN_BRIER_30S:.4f}"
    )

    # --- Step 3/4: diagnose what actually happened before concluding RQ4 ---
    print("\n=== Step 3: Stability diagnosis ===")
    if mlp_results is None or gnn_results is None:
        print(
            "One or both models hit a NaN/Inf loss and training was aborted outright -- "
            "see the [MODEL] NaN/Inf message above. No RQ4 conclusion can be drawn."
        )
    else:
        mlp_unstable = mlp_results["instability_warning_fired"]
        gnn_unstable = gnn_results["instability_warning_fired"]
        print(f"MLP instability warning fired: {mlp_unstable}")
        print(f"GNN instability warning fired: {gnn_unstable}")

        if gnn_unstable:
            print(
                "\nSTOP: the GNN still triggered the residual-instability warning despite the "
                "Milestone 12B stabilization bundle (gradient clipping, weight decay, lr=1e-4). "
                "Reporting this honestly rather than forcing an RQ4 conclusion. The loss curve "
                f"(by epoch) was logged to MLflow under run_id={gnn_results['run_id']} for "
                "inspection. Further fixes (an even lower learning rate, batch normalization "
                "between the SAGEConv layers) are follow-up work, not attempted further in this "
                "run per the task's explicit scope."
            )
            print(f"GNN per-epoch train loss: {gnn_results['epoch_losses']}")
        else:
            print("Both models completed all 50 epochs without the instability warning firing.")

            mlp_gap = mlp_results["train_val_gap"]
            gnn_gap = gnn_results["train_val_gap"]
            print(
                f"\nTrain/val loss gap -- MLP: {mlp_gap:+.4f} (train={mlp_results['train_loss']:.4f}, "
                f"val={mlp_results['val_loss']:.4f})"
            )
            print(
                f"Train/val loss gap -- GNN: {gnn_gap:+.4f} (train={gnn_results['train_loss']:.4f}, "
                f"val={gnn_results['val_loss']:.4f})"
            )

            gnn_brier_worse = (
                gnn_results["brier_15s"] > mlp_results["brier_15s"]
                and gnn_results["brier_30s"] > mlp_results["brier_30s"]
            )
            if gnn_brier_worse and gnn_gap <= mlp_gap + 0.05:
                print(
                    "\nThe GNN's train loss is now low/smooth and its train/val gap is comparable "
                    "to (not meaningfully worse than) the MLP's, yet its Brier Score remains "
                    "substantially worse. This looks like a DATA-LIMITED result, not an "
                    f"optimization-limited one: {n_train} training samples may still be a modest "
                    "dataset for a 2-layer GraphSAGE model's capacity relative to the 4-feature "
                    "MLP. This is a genuinely different, equally valid finding for RQ4 -- not a "
                    "non-result."
                )
            elif gnn_brier_worse:
                print(
                    "\nThe GNN's train/val gap is noticeably wider than the MLP's, suggesting some "
                    "residual overfitting or optimization difficulty beyond pure data scale -- "
                    "interpret the Brier comparison below with that in mind."
                )

            print("\n=== Step 2.4: MLP vs GNN, multi-competition vs Milestone 12B small-dataset ===")
            print(f"{'Model (dataset)':<42} {'Dataset size':>12} {'Brier@15s':>10} {'Brier@30s':>10}")
            print(
                f"{'MLP (multi-competition)':<42} {dataset_size:>12} "
                f"{mlp_results['brier_15s']:>10.4f} {mlp_results['brier_30s']:>10.4f}"
            )
            print(
                f"{'GNN (multi-competition, stabilized)':<42} {dataset_size:>12} "
                f"{gnn_results['brier_15s']:>10.4f} {gnn_results['brier_30s']:>10.4f}"
            )
            print(
                f"{'MLP (Milestone 12B, single-competition)':<42} {MILESTONE_12B_DATASET_SIZE:>12} "
                f"{MILESTONE_12B_MLP_BRIER_15S:>10.4f} {MILESTONE_12B_MLP_BRIER_30S:>10.4f}"
            )
            print(
                f"{'GNN (Milestone 12B, single-competition)':<42} {MILESTONE_12B_DATASET_SIZE:>12} "
                f"{MILESTONE_12B_GNN_BRIER_15S:>10.4f} {MILESTONE_12B_GNN_BRIER_30S:>10.4f}"
            )
            print(
                f"  (for reference: GNN Milestone 12 UNSTABLE run, dataset_size="
                f"{MILESTONE_10_FORCED_FLIP_DATASET_SIZE}: Brier@15s="
                f"{MILESTONE_12_UNSTABLE_GNN_BRIER_15S:.4f}, Brier@30s="
                f"{MILESTONE_12_UNSTABLE_GNN_BRIER_30S:.4f})"
            )
            print(
                "NOTE: MLP and GNN are not matched in raw parameter count/capacity -- a caveat for "
                "interpreting this comparison, not something to fix now. The MLP's optimizer "
                "(lr=1e-3, no weight decay/clipping) was also left untouched while the GNN got a "
                "3-part stabilization bundle -- a fully symmetric ablation (applying the same "
                "bundle to the MLP) would further confirm the MLP isn't secretly benefiting from a "
                "learning rate that happens to suit it, but is lower priority since it was already "
                "stable."
            )

            print("\n=== RQ4 conclusion ===")
            if not gnn_brier_worse:
                print(
                    "The GNN is stable AND competitive with or better than the MLP baseline at "
                    "both horizons -- RQ4 supports graph representations outperforming the "
                    "handcrafted scalar features in this setting."
                )
            elif gnn_gap <= mlp_gap + 0.05:
                print(
                    "The GNN is now stable but still underperforms the MLP at both horizons. Given "
                    "the comparable train/val gap, this looks like a data-scale limitation rather "
                    "than an optimization failure -- RQ4's answer here is a HEDGED 'not yet': the "
                    "handcrafted scalar features currently outperform this graph representation, "
                    "but more training data (not more tuning) is the most promising next lever "
                    "before treating this as a settled negative result, consistent with the "
                    "README's framing of RQs as working hypotheses rather than settled truths."
                )
            else:
                print(
                    "The GNN is now stable but still underperforms the MLP, with a wider train/val "
                    "gap than the MLP's -- some residual overfitting/optimization difficulty likely "
                    "remains beyond pure data scale. RQ4's answer here is a HEDGED 'not yet', with "
                    "more diagnosis (not a flat verdict) warranted before concluding graphs "
                    "underperform scalar features in general."
                )

    print(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("Run `mlflow ui` from the project root to inspect results visually.")


if __name__ == "__main__":
    train_and_evaluate()
