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
from production.src.models.deep_ensemble import DeepEnsembleDeepHit, compute_disentangled_ensemble_loss
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

# Milestone 14B: strengthened instability detector. Milestone 14's own MLP
# run is direct proof the single-epoch spike check (>50%) has a real blind
# spot: that run's train loss climbed from 3.30 to 5.07 over epochs 5-50 --
# NO single epoch-to-epoch jump ever exceeded 50% -- while val_loss went
# bit-for-bit frozen at 4.843820095062256 for the final 25 epochs. Directly
# probing the trained model afterward confirmed its softmax had saturated
# to ~1.0 on the last time bin regardless of input. These three additional
# signals are designed specifically to catch that failure mode next time,
# not a hypothetical one.
CUMULATIVE_DRIFT_WINDOW_EPOCHS = 20
CUMULATIVE_DRIFT_THRESHOLD_FRACTION = 0.30  # >30% increase over the window fires a warning

# Output-saturation check: TWO independent signals, since either alone can
# miss a real collapse. Batch variance catches "every sample gets the same
# output"; entropy catches the subtler case where outputs still differ
# slightly across samples but each individual prediction has collapsed to
# a near one-hot spike.
SATURATION_VARIANCE_THRESHOLD = 1e-6  # near-0 batch variance -> same output regardless of input
SATURATION_ENTROPY_THRESHOLD = 0.1  # near-0 mean entropy (max possible is ln(12)=2.485) -> near one-hot

SMALL_DATASET_WARNING_THRESHOLD = 500

# Milestone 12B's STABILIZED single-competition (World Cup 2022 only) MLP
# Brier Scores -- used below as the sanity-floor ceiling for judging
# whether the Milestone 14B stabilized MLP is genuinely learning well.
MILESTONE_12B_MLP_BRIER_15S = 0.0846
MILESTONE_12B_MLP_BRIER_30S = 0.1720

# Milestone 14's multi-competition run (8,074 samples): the MLP SILENTLY
# COLLAPSED (softmax saturated to the last time bin regardless of input --
# see the CUMULATIVE_DRIFT/SATURATION comment above) while the GNN, using
# the Milestone 12B stabilization bundle, trained normally. Kept here as
# historical reference rows, NOT retrained/overwritten -- that MLflow run
# stays as the record of this discovery.
MILESTONE_14_DATASET_SIZE = 8074
MILESTONE_14_MLP_COLLAPSED_RUN_ID = "07f3ef56e13d434aa02a5b832de610c4"
MILESTONE_14_MLP_COLLAPSED_BRIER_15S = 0.1263
MILESTONE_14_MLP_COLLAPSED_BRIER_30S = 0.2571
MILESTONE_14_GNN_STABLE_RUN_ID = "8267bc29a3e54d9d92e146de9b4de145"
MILESTONE_14_GNN_STABLE_BRIER_15S = 0.1127
MILESTONE_14_GNN_STABLE_BRIER_30S = 0.1905

# Milestone 14B Step 2: apply the SAME stabilization bundle used for the
# GNN to the MLP, as a first attempt -- NOT assumed to be correct just
# because it avoids collapse symptoms; train_and_evaluate() explicitly
# checks the MLP is actually learning well (meaningful loss decrease, sane
# Brier Scores), not merely "not collapsed."
MLP_STABILIZED_LR = 1e-4
MLP_STABILIZED_WEIGHT_DECAY = 1e-4

# Step 2.3: robustness check -- a second MLP weight-init seed, to
# distinguish "one-off bad initialization" from "systematic issue with the
# larger, more heterogeneous dataset." The train/val SPLIT seed (RANDOM_SEED,
# via split_generator) stays fixed at 42 for both -- only each run's model
# weight initialization differs, isolating exactly one variable.
MLP_ROBUSTNESS_CHECK_SEEDS = (42, 43)

# A stabilized-MLP run whose final Brier is drastically worse than this
# floor is undertrained, not merely "a bit worse" -- flagged explicitly in
# Step 2.2's health check rather than silently accepted.
MLP_SANITY_BRIER_15S_CEILING = MILESTONE_12B_MLP_BRIER_15S * 2.5
MLP_SANITY_BRIER_30S_CEILING = MILESTONE_12B_MLP_BRIER_30S * 2.5

# Milestone 14B's canonical (seed=42) stabilized single-MLP Brier Scores,
# on this same multi-competition dataset -- the baseline Milestone 21's
# Deep Ensemble is compared against. NOT re-derived here (that run is not
# retrained/overwritten); kept as a literal reference constant, same
# pattern as MILESTONE_14_*/MILESTONE_12B_* above.
MILESTONE_14B_MLP_RUN_ID = "e2c42aeed7374c398643298a1580a08c"
MILESTONE_14B_MLP_BRIER_15S = 0.09422525763511658
MILESTONE_14B_MLP_BRIER_30S = 0.1588379144668579

# Milestone 21: Deep Ensemble uncertainty quantification (ADR-004 -- a Deep
# Ensemble, NOT a true Batch Ensemble; see that ADR for why). Uses the SAME
# stabilization bundle as the Milestone 14B MLP/GNN (lr, weight decay,
# gradient clipping) -- a new architecture gets this safety net applied,
# not skipped, per this project's history of silent training failures at
# this exact data scale (Milestone 14).
DEEP_ENSEMBLE_M = 5


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
        # Warning flags: latched True the first time any signal fires,
        # across the whole run (not just the last epoch checked).
        cumulative_drift_fired = False
        saturation_fired = False
        frozen_val_loss_fired = False
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

            # Signal 2 (Milestone 14B): cumulative/windowed drift check.
            # Compares against the loss CUMULATIVE_DRIFT_WINDOW_EPOCHS
            # epochs prior, not just the immediately preceding epoch --
            # this is what would have caught Milestone 14's actual failure
            # (a steady, sub-spike-threshold climb across many epochs).
            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 and epoch > CUMULATIVE_DRIFT_WINDOW_EPOCHS:
                prior_loss = epoch_losses[epoch - CUMULATIVE_DRIFT_WINDOW_EPOCHS - 1]
                drift_fraction = (
                    (final_epoch_loss - prior_loss) / prior_loss if prior_loss > 0 else 0.0
                )
                mlflow.log_metric("cumulative_drift_fraction", drift_fraction, step=epoch)
                if drift_fraction > CUMULATIVE_DRIFT_THRESHOLD_FRACTION:
                    cumulative_drift_fired = True
                    print(
                        f"[{model_type}] CUMULATIVE DRIFT WARNING at epoch {epoch}: loss increased "
                        f"by {drift_fraction:.1%} over the last {CUMULATIVE_DRIFT_WINDOW_EPOCHS} "
                        f"epochs (from {prior_loss:.4f} at epoch {epoch - CUMULATIVE_DRIFT_WINDOW_EPOCHS} "
                        f"to {final_epoch_loss:.4f} now) -- exceeds the "
                        f"{CUMULATIVE_DRIFT_THRESHOLD_FRACTION:.0%} threshold. This is exactly the "
                        "failure mode a single-epoch spike check misses."
                    )

            # Step 2.3 / Signal 3: periodic validation loss AND output-
            # saturation check (two independent sub-signals: batch variance
            # and mean per-sample entropy -- see module-level comment).
            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 or epoch == NUM_EPOCHS:
                model.eval()
                with torch.no_grad():
                    val_scalar, val_graph, val_duration_bins, val_events = val_batch
                    val_input = input_fn(val_scalar, val_graph, *normalize_args)
                    epoch_val_predictions = model(val_input)
                    epoch_val_loss = loss_fn(
                        epoch_val_predictions, val_duration_bins, val_events
                    ).item()

                    batch_variance = epoch_val_predictions.var(dim=0).mean().item()
                    per_sample_entropy = -(
                        epoch_val_predictions * torch.log(epoch_val_predictions.clamp(min=1e-8))
                    ).sum(dim=1)
                    mean_entropy = per_sample_entropy.mean().item()

                mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
                mlflow.log_metric("output_batch_variance", batch_variance, step=epoch)
                mlflow.log_metric("output_mean_entropy", mean_entropy, step=epoch)

                if (
                    batch_variance < SATURATION_VARIANCE_THRESHOLD
                    or mean_entropy < SATURATION_ENTROPY_THRESHOLD
                ):
                    saturation_fired = True
                    print(
                        f"[{model_type}] SATURATION WARNING at epoch {epoch}: output batch "
                        f"variance={batch_variance:.2e} (threshold <{SATURATION_VARIANCE_THRESHOLD:.0e}), "
                        f"mean entropy={mean_entropy:.4f} (threshold <{SATURATION_ENTROPY_THRESHOLD}) -- "
                        "predictions have collapsed to a near-constant or near-one-hot output "
                        "regardless of input."
                    )

                # Signal 4 (backstop, weakest of the four -- see module
                # docstring comment): bit-for-bit frozen val_loss. Only
                # fires after the model has ALREADY fully saturated; exact
                # floating-point equality across resumed computation is
                # what confirmed Milestone 14's collapse, well after the
                # drift had already started.
                if val_loss_history:
                    previous_check_epoch = max(val_loss_history.keys())
                    if val_loss_history[previous_check_epoch] == epoch_val_loss:
                        frozen_val_loss_fired = True
                        print(
                            f"[{model_type}] FROZEN VAL LOSS WARNING at epoch {epoch}: val_loss is "
                            f"bit-for-bit identical to epoch {previous_check_epoch}'s value "
                            f"({epoch_val_loss}) -- the weakest of these signals, since it only "
                            "fires after the model has already fully saturated."
                        )
                val_loss_history[epoch] = epoch_val_loss

        print(f"[{model_type}] Final training loss: {final_epoch_loss:.4f}")

        # Signal 1: single-epoch spike check (Milestone 12B, retained).
        spike_fired = _check_for_instability(model_type, epoch_losses)

        instability_warning_fired = (
            spike_fired or cumulative_drift_fired or saturation_fired or frozen_val_loss_fired
        )
        mlflow.log_param("spike_warning_fired", spike_fired)
        mlflow.log_param("cumulative_drift_warning_fired", cumulative_drift_fired)
        mlflow.log_param("saturation_warning_fired", saturation_fired)
        mlflow.log_param("frozen_val_loss_warning_fired", frozen_val_loss_fired)
        mlflow.log_param("instability_warning_fired", instability_warning_fired)
        print(
            f"[{model_type}] Warning summary -- spike: {spike_fired}, cumulative_drift: "
            f"{cumulative_drift_fired}, saturation: {saturation_fired}, frozen_val_loss: "
            f"{frozen_val_loss_fired}"
        )

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
        "spike_fired": spike_fired,
        "cumulative_drift_fired": cumulative_drift_fired,
        "saturation_fired": saturation_fired,
        "frozen_val_loss_fired": frozen_val_loss_fired,
        "epoch_losses": epoch_losses,
        "val_loss_history": val_loss_history,
        "run_id": run.info.run_id,
    }


def _train_and_log_deep_ensemble(
    model: DeepEnsembleDeepHit,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_batch: tuple,
    n_train: int,
    n_val: int,
    match_ids: list[int],
    dataset_size: int,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    normalization_artifact: dict,
    run_tags: dict | None = None,
) -> dict | None:
    """Deep Ensemble analog of `_train_and_log_model`. Kept as a separate
    function rather than folded into the shared loop above: that loop's
    body assumes a single `[B, num_bins]` prediction tensor goes straight
    into one `DeepHitLoss` call, but the ensemble's per-member gradient
    disentanglement (Step 2.3/ADR-004) requires a genuinely different loop
    body -- a [M, B, num_bins] forward pass and a Python loop over M
    independent loss computations (see `compute_disentangled_ensemble_loss`)
    -- so forcing it through the same `input_fn`/single-loss shape would
    obscure, not clarify, that difference.

    Per Step 2.3: the strengthened Milestone 14B instability detector
    (spike, cumulative drift, saturation/entropy, frozen-val-loss) is
    applied to the ensemble's MEAN prediction's train/val loss trajectory
    -- i.e., the same `compute_disentangled_ensemble_loss` value already
    used for backprop (itself an average across members), and `mean_pmf`
    (the members' averaged PMF) for the saturation/entropy check. Ensemble
    DIVERSITY is a separate, dedicated metric (Step 2.4), not something
    this reused detector is expected to catch on its own.
    """
    loss_fn = DeepHitLoss()
    M = model.M

    with mlflow.start_run(run_name="deep_ensemble_run") as run:
        if run_tags:
            mlflow.set_tags(run_tags)

        mlflow.log_params(
            {
                "model_type": "DeepEnsemble_MLP",
                "M": M,
                "lr": MLP_STABILIZED_LR,
                "weight_decay": MLP_STABILIZED_WEIGHT_DECAY,
                "gradient_clipping": True,
                "epochs": NUM_EPOCHS,
                "train_size": n_train,
                "val_size": n_val,
                "alpha": loss_fn.alpha,
                "sigma": loss_fn.sigma,
                "num_bins": NUM_BINS,
                "bin_size": BIN_SIZE_SECONDS,
                "random_seed": RANDOM_SEED,
                "match_ids": ",".join(str(m) for m in match_ids),
                "feature_key_order": ",".join(FEATURE_KEYS),
                "match_count": len(match_ids),
                "dataset_size": dataset_size,
                "periods_included": ",".join(str(p) for p in CHAIN_BUILDER_PERIODS),
                "coordinate_convention": "statsbomb_per_actor_native",
                "stabilization_bundle": True,
                "saturation_check_v2": True,
                "ensemble_kind": "deep_ensemble_not_batch_ensemble",  # see ADR-004
            }
        )

        print(
            f"\n[DeepEnsemble] Training M={M} independent members for {NUM_EPOCHS} epochs on "
            f"{n_train} samples ({n_val} held out for validation)..."
        )

        epoch_losses: list[float] = []
        val_loss_history: dict[int, float] = {}
        final_epoch_loss = None
        cumulative_drift_fired = False
        saturation_fired = False
        frozen_val_loss_fired = False

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            epoch_loss_total = 0.0
            num_batches = 0

            for batch_idx, (scalar_batch, graph_batch, duration_bins_batch, events_batch) in enumerate(
                train_loader
            ):
                normalized_input = (scalar_batch - normalization_mean) / normalization_std

                optimizer.zero_grad()
                pmf_per_member = model(normalized_input)  # [M, B, num_bins] -- BROADCAST, see model docstring
                loss = compute_disentangled_ensemble_loss(
                    pmf_per_member, duration_bins_batch, events_batch, loss_fn
                )

                if not torch.isfinite(loss):
                    print(f"[DeepEnsemble] NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Stopping.")
                    return None

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                optimizer.step()

                epoch_loss_total += loss.item()
                num_batches += 1

            final_epoch_loss = epoch_loss_total / num_batches
            epoch_losses.append(final_epoch_loss)
            mlflow.log_metric("train_loss", final_epoch_loss, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  [DeepEnsemble] epoch {epoch:3d}/{NUM_EPOCHS}: mean-member training loss = {final_epoch_loss:.4f}")

            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 and epoch > CUMULATIVE_DRIFT_WINDOW_EPOCHS:
                prior_loss = epoch_losses[epoch - CUMULATIVE_DRIFT_WINDOW_EPOCHS - 1]
                drift_fraction = (
                    (final_epoch_loss - prior_loss) / prior_loss if prior_loss > 0 else 0.0
                )
                mlflow.log_metric("cumulative_drift_fraction", drift_fraction, step=epoch)
                if drift_fraction > CUMULATIVE_DRIFT_THRESHOLD_FRACTION:
                    cumulative_drift_fired = True
                    print(
                        f"[DeepEnsemble] CUMULATIVE DRIFT WARNING at epoch {epoch}: mean-member loss "
                        f"increased by {drift_fraction:.1%} over the last "
                        f"{CUMULATIVE_DRIFT_WINDOW_EPOCHS} epochs."
                    )

            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 or epoch == NUM_EPOCHS:
                model.eval()
                with torch.no_grad():
                    val_scalar, val_graph, val_duration_bins, val_events = val_batch
                    val_input = (val_scalar - normalization_mean) / normalization_std
                    val_pmf_per_member = model(val_input)
                    epoch_val_loss = compute_disentangled_ensemble_loss(
                        val_pmf_per_member, val_duration_bins, val_events, loss_fn
                    ).item()

                    val_mean_pmf = val_pmf_per_member.mean(dim=0)  # [B, num_bins]
                    batch_variance = val_mean_pmf.var(dim=0).mean().item()
                    per_sample_entropy = -(
                        val_mean_pmf * torch.log(val_mean_pmf.clamp(min=1e-8))
                    ).sum(dim=1)
                    mean_entropy = per_sample_entropy.mean().item()

                mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
                mlflow.log_metric("output_batch_variance", batch_variance, step=epoch)
                mlflow.log_metric("output_mean_entropy", mean_entropy, step=epoch)

                if (
                    batch_variance < SATURATION_VARIANCE_THRESHOLD
                    or mean_entropy < SATURATION_ENTROPY_THRESHOLD
                ):
                    saturation_fired = True
                    print(
                        f"[DeepEnsemble] SATURATION WARNING at epoch {epoch}: mean-prediction batch "
                        f"variance={batch_variance:.2e}, mean entropy={mean_entropy:.4f} -- the "
                        "ensemble's AVERAGED output has collapsed to a near-constant or near-one-hot "
                        "distribution regardless of input."
                    )

                if val_loss_history:
                    previous_check_epoch = max(val_loss_history.keys())
                    if val_loss_history[previous_check_epoch] == epoch_val_loss:
                        frozen_val_loss_fired = True
                        print(
                            f"[DeepEnsemble] FROZEN VAL LOSS WARNING at epoch {epoch}: val_loss is "
                            f"bit-for-bit identical to epoch {previous_check_epoch}'s value "
                            f"({epoch_val_loss})."
                        )
                val_loss_history[epoch] = epoch_val_loss

        print(f"[DeepEnsemble] Final mean-member training loss: {final_epoch_loss:.4f}")

        spike_fired = _check_for_instability("DeepEnsemble", epoch_losses)
        instability_warning_fired = (
            spike_fired or cumulative_drift_fired or saturation_fired or frozen_val_loss_fired
        )
        mlflow.log_param("spike_warning_fired", spike_fired)
        mlflow.log_param("cumulative_drift_warning_fired", cumulative_drift_fired)
        mlflow.log_param("saturation_warning_fired", saturation_fired)
        mlflow.log_param("frozen_val_loss_warning_fired", frozen_val_loss_fired)
        mlflow.log_param("instability_warning_fired", instability_warning_fired)
        print(
            f"[DeepEnsemble] Warning summary -- spike: {spike_fired}, cumulative_drift: "
            f"{cumulative_drift_fired}, saturation: {saturation_fired}, frozen_val_loss: "
            f"{frozen_val_loss_fired}"
        )

        model.eval()
        with torch.no_grad():
            val_scalar, val_graph, val_duration_bins, val_events = val_batch
            val_input = (val_scalar - normalization_mean) / normalization_std

            mean_pmf, std_cumulative_incidence, per_member_cumulative_incidence = model.predict_with_uncertainty(
                val_input, time_bin=3
            )
            val_loss = compute_disentangled_ensemble_loss(
                model(val_input), val_duration_bins, val_events, loss_fn
            )
            print(f"[DeepEnsemble] Validation loss (mean-member): {val_loss.item():.4f}")

            briers = {}
            for time_bin in BRIER_TIME_BINS:
                brier, num_excluded = calculate_brier_score(
                    mean_pmf, val_duration_bins, val_duration_bins, val_events, time_bin
                )
                seconds = time_bin * 5.0
                print(f"  [DeepEnsemble] time_bin={time_bin} ({seconds:.0f}s): Brier Score (mean PMF) = {brier:.4f}")
                briers[time_bin] = (brier, num_excluded)

            # Step 2.4: diversity metric -- mean, across the validation set,
            # of each sample's cross-member standard deviation of
            # cumulative incidence at time_bin=3. A collapsed (non-diverse)
            # ensemble would show this near zero; logged explicitly so that
            # would be visible in MLflow, not just assumed away.
            diversity_std_ci_15s = std_cumulative_incidence.mean().item()
            print(f"[DeepEnsemble] Diversity metric (mean std of per-member CI@15s across val set): {diversity_std_ci_15s:.6f}")

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
                "ensemble_diversity_std_ci_15s": diversity_std_ci_15s,
            }
        )

        mlflow.pytorch.log_model(model, name="deep_ensemble_model", serialization_format="pickle")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(normalization_artifact, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        try:
            mlflow.log_artifact(tmp_file_path, artifact_path="normalization")
        finally:
            os.remove(tmp_file_path)

        print(f"[DeepEnsemble] MLflow run ID: {run.info.run_id}")

    return {
        "train_loss": final_epoch_loss,
        "val_loss": val_loss.item(),
        "brier_15s": brier_15s,
        "brier_30s": brier_30s,
        "excluded_15s": excluded_15s,
        "excluded_30s": excluded_30s,
        "train_val_gap": train_val_gap,
        "instability_warning_fired": instability_warning_fired,
        "spike_fired": spike_fired,
        "cumulative_drift_fired": cumulative_drift_fired,
        "saturation_fired": saturation_fired,
        "frozen_val_loss_fired": frozen_val_loss_fired,
        "diversity_std_ci_15s": diversity_std_ci_15s,
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

    # === Milestone 14B Step 2: MLP stabilization (same bundle as the GNN),
    # PLUS a second weight-init seed as a robustness check. train_loader is
    # the SAME shared object across every call below (as in every prior
    # milestone's MLP-vs-GNN comparison) -- its shuffle order naturally
    # differs call-to-call as its internal generator advances, but the
    # train/val SPLIT itself (train_set/val_set, from split_generator) is
    # held fixed at RANDOM_SEED=42 for every run in this function. Only
    # each MLP run's WEIGHT INITIALIZATION seed differs (42 vs 43),
    # isolating exactly the one variable Step 2.3 asks about.
    print("\n=== Milestone 14B Step 2: MLP stabilization + robustness check ===")
    mlp_seed_results: dict[int, dict | None] = {}
    for seed in MLP_ROBUSTNESS_CHECK_SEEDS:
        torch.manual_seed(seed)
        mlp_model_seeded = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
        mlp_optimizer_seeded = torch.optim.Adam(
            mlp_model_seeded.parameters(),
            lr=MLP_STABILIZED_LR,
            weight_decay=MLP_STABILIZED_WEIGHT_DECAY,
        )
        mlp_seed_results[seed] = _train_and_log_model(
            model_type="MLP",
            model=mlp_model_seeded,
            optimizer=mlp_optimizer_seeded,
            lr=MLP_STABILIZED_LR,
            weight_decay=MLP_STABILIZED_WEIGHT_DECAY,
            clip_grad_norm=True,
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
                "stabilization_bundle": True,
                "saturation_check_v2": True,
                "init_seed": seed,
            },
            run_tags={
                "supersedes_run_id": MILESTONE_14_MLP_COLLAPSED_RUN_ID,
                "supersedes_note": (
                    "Milestone 14B stabilization (grad clipping + weight decay + lower lr) of "
                    "Milestone 14's softmax-saturated MLP run; that run is kept, not deleted."
                ),
                "init_seed": str(seed),
            },
            normalization_artifact={
                "feature_key_order": list(FEATURE_KEYS),
                "mean": feature_mean.tolist(),
                "std": feature_std.tolist(),
            },
        )

    # Reset the global RNG state before constructing the GNN, so its own
    # initialization isn't accidentally coupled to wherever the MLP seed
    # loop left the global generator.
    torch.manual_seed(RANDOM_SEED)

    # === Milestone 14B Step 3: GNN retrain, NOW with the exact same
    # hyperparameters as the MLP (both use the stabilization bundle) --
    # addresses the asymmetry noted (but not fixed) in Milestones 12B/14.
    gnn_model = GNNDeepHitSurvivalModel(num_node_features=7, num_bins=NUM_BINS, hidden_dim=GNN_HIDDEN_DIM)
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
            "stabilization_bundle": True,
            "saturation_check_v2": True,
            "init_seed": RANDOM_SEED,
        },
        run_tags={
            "supersedes_run_id": MILESTONE_14_GNN_STABLE_RUN_ID,
            "supersedes_note": (
                "Milestone 14B retrain (identical hyperparameters to the stabilized MLP, "
                "strengthened instability detector) of Milestone 14's already-stable GNN run, "
                "for a properly symmetric comparison."
            ),
        },
        normalization_artifact={
            "graph_continuous_feature_order": ["x", "y", "dist_to_ball"],
            "mean": graph_feature_mean.tolist(),
            "std": graph_feature_std.tolist(),
        },
    )

    # Reset the global RNG state before constructing the ensemble, same
    # reasoning as before the GNN above: its M independent members'
    # initialization shouldn't be accidentally coupled to wherever the GNN
    # training loop left the global generator.
    torch.manual_seed(RANDOM_SEED)

    # === Milestone 21: Deep Ensemble uncertainty quantification (ADR-004).
    # Same split, same normalization, same stabilization bundle as the
    # Milestone 14B MLP/GNN -- only the model and its per-member-
    # disentangled loss loop differ (see _train_and_log_deep_ensemble).
    print(f"\n=== Milestone 21: Deep Ensemble (M={DEEP_ENSEMBLE_M}) training ===")
    deep_ensemble_model = DeepEnsembleDeepHit(
        num_features=len(FEATURE_KEYS), num_bins=NUM_BINS, M=DEEP_ENSEMBLE_M
    )
    deep_ensemble_optimizer = torch.optim.Adam(
        deep_ensemble_model.parameters(),
        lr=MLP_STABILIZED_LR,
        weight_decay=MLP_STABILIZED_WEIGHT_DECAY,
    )
    deep_ensemble_results = _train_and_log_deep_ensemble(
        model=deep_ensemble_model,
        optimizer=deep_ensemble_optimizer,
        train_loader=train_loader,
        val_batch=val_batch,
        n_train=n_train,
        n_val=n_val,
        match_ids=match_ids,
        dataset_size=dataset_size,
        normalization_mean=feature_mean,
        normalization_std=feature_std,
        normalization_artifact={
            "feature_key_order": list(FEATURE_KEYS),
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
        run_tags={
            "baseline_single_mlp_run_id": MILESTONE_14B_MLP_RUN_ID,
            "baseline_note": (
                "Deep Ensemble (ADR-004 -- NOT a true Batch Ensemble), compared against the "
                "Milestone 14B single-MLP baseline. ~5x parameters/compute vs. that baseline, "
                "so this is not an equal-capacity comparison (same caveat as the Milestone 12 "
                "MLP-vs-GNN comparison)."
            ),
        },
    )

    print("\n=== Milestone 21: Deep Ensemble warning summary + baseline comparison ===")
    if deep_ensemble_results is None:
        print("DeepEnsemble: training ABORTED (NaN/Inf loss).")
    else:
        print(
            f"DeepEnsemble: spike={deep_ensemble_results['spike_fired']}, "
            f"cumulative_drift={deep_ensemble_results['cumulative_drift_fired']}, "
            f"saturation={deep_ensemble_results['saturation_fired']}, "
            f"frozen_val_loss={deep_ensemble_results['frozen_val_loss_fired']}"
        )
        print(
            f"DeepEnsemble diversity metric (mean std of per-member CI@15s): "
            f"{deep_ensemble_results['diversity_std_ci_15s']:.6f}"
        )
        print(
            f"\n{'Model':<40} {'Params (approx)':>16} {'Brier@15s':>10} {'Brier@30s':>10}"
        )
        print(
            f"{'Single MLP (Milestone 14B, seed=42)':<40} {'1x':>16} "
            f"{MILESTONE_14B_MLP_BRIER_15S:>10.4f} {MILESTONE_14B_MLP_BRIER_30S:>10.4f}"
        )
        print(
            f"{f'Deep Ensemble (M={DEEP_ENSEMBLE_M}, mean PMF)':<40} {f'~{DEEP_ENSEMBLE_M}x':>16} "
            f"{deep_ensemble_results['brier_15s']:>10.4f} {deep_ensemble_results['brier_30s']:>10.4f}"
        )
        print(
            "NOTE: this is NOT an equal-capacity comparison -- the Deep Ensemble has ~"
            f"{DEEP_ENSEMBLE_M}x the parameters and training/inference compute of the single MLP "
            "(same caveat already established for the Milestone 12 MLP-vs-GNN comparison). A "
            "lower ensemble Brier Score is evidence the mean-PMF prediction is at least as good, "
            "not evidence the ensembling technique itself is more parameter-efficient."
        )

    print(f"\nDataset size: {dataset_size} total samples ({n_train} train / {n_val} val)")

    # === Step 3.2: did ANY of the four warning signals fire, for either
    # model, across BOTH MLP seeds? ===
    print("\n=== Step 3: warning summary across all runs (strengthened detector) ===")
    any_warnings_anywhere = False
    for seed, results in mlp_seed_results.items():
        if results is None:
            print(f"MLP (seed={seed}): training ABORTED (NaN/Inf loss).")
            any_warnings_anywhere = True
            continue
        print(
            f"MLP (seed={seed}): spike={results['spike_fired']}, "
            f"cumulative_drift={results['cumulative_drift_fired']}, "
            f"saturation={results['saturation_fired']}, "
            f"frozen_val_loss={results['frozen_val_loss_fired']}"
        )
        any_warnings_anywhere = any_warnings_anywhere or results["instability_warning_fired"]

    if gnn_results is None:
        print("GNN: training ABORTED (NaN/Inf loss).")
        any_warnings_anywhere = True
    else:
        print(
            f"GNN: spike={gnn_results['spike_fired']}, "
            f"cumulative_drift={gnn_results['cumulative_drift_fired']}, "
            f"saturation={gnn_results['saturation_fired']}, "
            f"frozen_val_loss={gnn_results['frozen_val_loss_fired']}"
        )
        any_warnings_anywhere = any_warnings_anywhere or gnn_results["instability_warning_fired"]

    # seed=RANDOM_SEED (42) is the canonical/headline stabilized MLP result
    # (matching the split/GNN seed, for direct comparability); the other
    # seed is the Step 2.3 robustness cross-check, reported alongside but
    # not gating the RQ4 comparison itself.
    primary_mlp_results = mlp_seed_results[RANDOM_SEED]
    robustness_seed = next(s for s in MLP_ROBUSTNESS_CHECK_SEEDS if s != RANDOM_SEED)
    robustness_mlp_results = mlp_seed_results[robustness_seed]

    # === Step 2.2: is the stabilized MLP genuinely HEALTHY, not merely
    # "not collapsed"? Three explicit criteria. ===
    mlp_healthy = False
    if primary_mlp_results is not None and not primary_mlp_results["instability_warning_fired"]:
        first_loss = primary_mlp_results["epoch_losses"][0]
        last_loss = primary_mlp_results["epoch_losses"][-1]
        loss_decreased_meaningfully = (first_loss - last_loss) > 0.1 * first_loss
        brier_in_sane_range = (
            primary_mlp_results["brier_15s"] <= MLP_SANITY_BRIER_15S_CEILING
            and primary_mlp_results["brier_30s"] <= MLP_SANITY_BRIER_30S_CEILING
        )
        mlp_healthy = loss_decreased_meaningfully and brier_in_sane_range
        print(
            f"\nMLP (seed={RANDOM_SEED}) health check: no instability warnings=True, loss "
            f"decreased meaningfully={loss_decreased_meaningfully} ({first_loss:.4f} -> "
            f"{last_loss:.4f}), Brier in sane range (<= {MLP_SANITY_BRIER_15S_CEILING:.4f} / "
            f"{MLP_SANITY_BRIER_30S_CEILING:.4f})={brier_in_sane_range} (actual: "
            f"{primary_mlp_results['brier_15s']:.4f} / {primary_mlp_results['brier_30s']:.4f})"
        )
        if not brier_in_sane_range:
            print(
                "WARNING: the 'same hyperparameters as the GNN' recipe produced a STABLE but "
                "apparently UNDERTRAINED MLP (Brier far worse than the Milestone 12B sanity "
                "floor) -- lr=1e-4 was tuned for SAGEConv's specific instability, not validated "
                "as appropriate for this MLP. The 'purely architectural, hyperparameter-neutral' "
                "comparison assumption does NOT hold cleanly here."
            )
    elif primary_mlp_results is not None:
        print(f"\nMLP (seed={RANDOM_SEED}) still triggered an instability warning -- see summary above.")
    else:
        print(f"\nMLP (seed={RANDOM_SEED}) training was aborted (NaN/Inf loss).")

    print(f"\nRobustness check (seed={robustness_seed}):")
    if robustness_mlp_results is not None:
        print(
            f"  final train loss: {robustness_mlp_results['train_loss']:.4f}, any warning fired: "
            f"{robustness_mlp_results['instability_warning_fired']}, Brier@15s/30s: "
            f"{robustness_mlp_results['brier_15s']:.4f} / {robustness_mlp_results['brier_30s']:.4f}"
        )
        print(
            "  (systematic-vs-one-off read: both seeds " +
            ("avoided every warning" if not any(
                mlp_seed_results[s]["instability_warning_fired"] for s in MLP_ROBUSTNESS_CHECK_SEEDS
                if mlp_seed_results[s] is not None
            ) else "did NOT both avoid every warning") +
            " -- see per-seed detail above.)"
        )
    else:
        print("  training ABORTED (NaN/Inf loss).")

    # === Step 3.4: four-row comparison table ===
    print("\n=== Step 3.4: four-way comparison (Milestone 14 vs Milestone 14B) ===")
    print(f"{'Model (run)':<48} {'Dataset':>8} {'Brier@15s':>10} {'Brier@30s':>10}")
    print(
        f"{'MLP (Milestone 14, COLLAPSED, for the record)':<48} {MILESTONE_14_DATASET_SIZE:>8} "
        f"{MILESTONE_14_MLP_COLLAPSED_BRIER_15S:>10.4f} {MILESTONE_14_MLP_COLLAPSED_BRIER_30S:>10.4f}"
    )
    print(
        f"{'GNN (Milestone 14, stable)':<48} {MILESTONE_14_DATASET_SIZE:>8} "
        f"{MILESTONE_14_GNN_STABLE_BRIER_15S:>10.4f} {MILESTONE_14_GNN_STABLE_BRIER_30S:>10.4f}"
    )
    if primary_mlp_results is not None:
        print(
            f"{'MLP (Milestone 14B, stabilized)':<48} {dataset_size:>8} "
            f"{primary_mlp_results['brier_15s']:>10.4f} {primary_mlp_results['brier_30s']:>10.4f}"
        )
    else:
        print(f"{'MLP (Milestone 14B, stabilized)':<48} {dataset_size:>8} {'ABORTED':>10} {'ABORTED':>10}")
    if gnn_results is not None:
        print(
            f"{'GNN (Milestone 14B, same hyperparams as MLP)':<48} {dataset_size:>8} "
            f"{gnn_results['brier_15s']:>10.4f} {gnn_results['brier_30s']:>10.4f}"
        )
    else:
        print(f"{'GNN (Milestone 14B, same hyperparams as MLP)':<48} {dataset_size:>8} {'ABORTED':>10} {'ABORTED':>10}")

    # === Step 5: conditional RQ4 conclusion -- ONLY if evidence quality supports one ===
    print("\n=== Step 5: RQ4 conclusion (conditional on evidence quality) ===")
    if any_warnings_anywhere:
        print(
            "At least one run (MLP seed 42, MLP seed 43, or GNN) triggered an instability "
            "warning under the strengthened detector, or aborted outright. Per Milestone 12B's "
            "precedent: NOT issuing an RQ4 conclusion this run. Report the blocker, fix it, "
            "re-run -- do not force a verdict."
        )
    elif not mlp_healthy:
        print(
            "No instability warnings fired, but the stabilized MLP does not pass the 'genuinely "
            "learning well' health check above -- it may be stable-but-undertrained rather than a "
            "fair comparison point. NOT issuing an RQ4 conclusion this run. The 'same "
            "hyperparameters for both models' approach needs its own dedicated tuning pass for "
            "the MLP before this comparison is trustworthy."
        )
    else:
        gnn_better_or_comparable = (
            gnn_results["brier_15s"] <= primary_mlp_results["brier_15s"] * 1.1
            and gnn_results["brier_30s"] <= primary_mlp_results["brier_30s"] * 1.1
        )
        print(
            f"Both models are confirmed genuinely healthy: no warnings fired (spike, cumulative "
            f"drift, saturation, or frozen-val-loss) across both MLP seeds and the GNN, and the "
            f"MLP's loss decreased meaningfully with a sane Brier Score. MLP Brier@15s/30s = "
            f"{primary_mlp_results['brier_15s']:.4f} / {primary_mlp_results['brier_30s']:.4f}; "
            f"GNN = {gnn_results['brier_15s']:.4f} / {gnn_results['brier_30s']:.4f}."
        )
        if gnn_better_or_comparable:
            print(
                "The GNN is competitive with or better than the (now genuinely healthy) MLP at "
                "this scale, using IDENTICAL hyperparameters for both -- real, if still "
                "single-run, evidence in favor of graph representations for RQ4. Given this "
                "project's history of surprises at exactly this comparison step (Milestones 12, "
                "14), this should be read as one data point, not a settled verdict -- consistent "
                "with the README's framing of RQs as working hypotheses."
            )
        else:
            print(
                "The MLP outperforms the GNN even with both confirmed healthy and using identical "
                "hyperparameters -- RQ4's answer here leans toward the handcrafted scalar "
                "features, though still hedged given how often this exact comparison has moved "
                "across milestones as data scale and stabilization changed."
            )

    print(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print("Run `mlflow ui` from the project root to inspect results visually.")


if __name__ == "__main__":
    train_and_evaluate()
