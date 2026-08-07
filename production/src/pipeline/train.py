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
import logging
import os
import sys
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
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from production.src.constants import MLFLOW_EXPERIMENT_NAME
from production.src.ingestion.statsbomb_io import (
    batch_extract_valid_matches,
    fetch_match_360,
    fetch_match_events,
    find_360_competitions,
    parse_360_frame,
)
from production.src.models.batch_ensemble import BatchEnsembleDeepHit
from production.src.models.deep_ensemble import (
    DeepEnsembleDeepHit,
    compute_disentangled_ensemble_loss,
)
from production.src.models.deephit import DeepHitSurvivalModel
from production.src.models.deephit_loss import DeepHitLoss
from production.src.models.evaluation import calculate_brier_score
from production.src.models.gnn_model import GNNDeepHitSurvivalModel
from production.src.models.graph_builder import (
    DEFAULT_OPPONENT_RADIUS,
    DEFAULT_SAME_TEAM_RADIUS,
)
from production.src.pipeline.chain_builder import build_possession_chains
from production.src.pipeline.data_split import match_level_split
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.naive_baseline_features import (
    BASELINE_FEATURE_KEYS,
    extract_naive_baseline_features,
)
from production.src.pipeline.habit_memory import (
    MIN_HISTORICAL_EVENTS,
    build_player_match_buckets,
    heatmap_from_buckets,
)
from production.src.pipeline.survival_dataset import (
    BIN_SIZE_SECONDS,
    FEATURE_KEYS,
    NUM_BINS,
    TacticalSurvivalDataset,
)
from production.src.spatial.control import BiomechanicalPitchControl

# Engineering-review action item: module-level logger, replacing this
# file's former print()-based diagnostic output. A plain `getLogger(__name__)`
# with NO handler/basicConfig configured here -- this module is imported by
# tests and other modules, and a library module must never call
# `logging.basicConfig` (or otherwise mutate global logging config) at
# import time, since that would silently override whatever logging setup
# the IMPORTING application already has. `basicConfig` is called once,
# below, only inside the `if __name__ == "__main__":` guard, so it only
# takes effect for this file's own standalone `python -m ...` entrypoint.
logger = logging.getLogger(__name__)

# MLFLOW_EXPERIMENT_NAME now comes from production.src.constants
# (engineering-review de-duplication -- was defined locally here before,
# and independently in explainer.py; value unchanged).

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

# Milestone 35 (ADR-011): as of this milestone, every training run in this
# file uses `data_split.match_level_split` (MATCH-level train/val split),
# tagged `split_type="match_level"` in its MLflow params. EVERY MLflow run
# logged before this milestone (M8's baseline through M23's habit-blended
# MLP) used `torch.utils.data.random_split` (SAMPLE-level splitting) and
# is therefore implicitly `split_type="sample_level"` -- those historical
# runs are NOT retroactively re-tagged (MLflow params are immutable after
# logging, and rewriting history here would misrepresent what actually
# produced those numbers). A run's `split_type` param is the authoritative
# way to tell which methodology produced it; do not assume from date or
# run name alone. See ADR-011 for why this changed and why the two
# methodologies' numbers are not directly comparable.
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

    logger.info(
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
    logger.info(f"Competitions verified (via the live competitions index) to have 360 data ({len(qualifying_competitions)}):")
    for c in qualifying_competitions:
        logger.info(
            f"  competition_id={c['competition_id']}, season_id={c['season_id']}: "
            f"{c['competition_name']} {c['season_name']}"
        )

    competition_season_pairs = [
        (c["competition_id"], c["season_id"]) for c in qualifying_competitions
    ]
    match_pool = batch_extract_valid_matches(competition_season_pairs, num_matches=MATCH_POOL_SIZE)
    logger.info(
        f"\nResolved {len(match_pool)} valid matches across {len(qualifying_competitions)} "
        f"qualifying competitions (pool target {MATCH_POOL_SIZE})"
    )

    engine = BiomechanicalPitchControl()
    all_features, all_frames, all_chains, all_source_event_ids = [], [], [], []
    all_sample_match_ids = []  # Milestone 23: per-SAMPLE match_id (same order/length as all_features)
    used_match_ids = []
    for match_id in match_pool:
        features, frames, chains, source_event_ids = _match_chains_with_features(match_id, engine)
        all_features.extend(features)
        all_frames.extend(frames)
        all_chains.extend(chains)
        all_source_event_ids.extend(source_event_ids)
        all_sample_match_ids.extend([match_id] * len(features))
        used_match_ids.append(match_id)

        if len(all_features) >= TARGET_SAMPLE_COUNT:
            logger.info(
                f"\nReached target sample count ({TARGET_SAMPLE_COUNT}) after "
                f"{len(used_match_ids)} matches -- stopping early rather than "
                "exhaustively processing the whole match pool."
            )
            break

    logger.info(
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
        all_sample_match_ids,
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
        logger.warning(
            f"[{model_type}] WARNING: residual training instability detected -- loss "
            f"increased by {max_relative_increase:.1%} at epoch {culprit_epoch} "
            f"(threshold: single-epoch relative increase > {INSTABILITY_THRESHOLD_FRACTION:.0%})."
        )
    return fired


def _check_cumulative_drift(
    model_type: str,
    epoch_losses: list[float],
    epoch: int,
    window_epochs: int = CUMULATIVE_DRIFT_WINDOW_EPOCHS,
    threshold_fraction: float = CUMULATIVE_DRIFT_THRESHOLD_FRACTION,
) -> tuple[bool, float]:
    """Milestone 14B Signal 2, factored out of the epoch loop as a pure
    function (ADR-010) so it can be regression-tested against synthetic
    epoch-loss sequences without a real training run. Compares the current
    epoch's loss against the loss `window_epochs` epochs prior -- this is
    what would have caught Milestone 14's actual failure (a steady,
    sub-spike-threshold climb across many epochs), unlike a single-epoch
    spike check.

    `epoch` is the 1-indexed epoch number matching `epoch_losses`'
    construction (`epoch_losses[epoch - 1]` is that epoch's loss). Returns
    `(fired, drift_fraction)`.
    """
    current_loss = epoch_losses[epoch - 1]
    prior_loss = epoch_losses[epoch - window_epochs - 1]
    drift_fraction = (current_loss - prior_loss) / prior_loss if prior_loss > 0 else 0.0
    fired = drift_fraction > threshold_fraction
    if fired:
        logger.warning(
            f"[{model_type}] CUMULATIVE DRIFT WARNING at epoch {epoch}: loss increased "
            f"by {drift_fraction:.1%} over the last {window_epochs} epochs (from "
            f"{prior_loss:.4f} at epoch {epoch - window_epochs} to {current_loss:.4f} now) -- "
            f"exceeds the {threshold_fraction:.0%} threshold. This is exactly the failure mode "
            "a single-epoch spike check misses."
        )
    return fired, drift_fraction


def _check_saturation(
    model_type: str,
    epoch: int,
    batch_variance: float,
    mean_entropy: float,
    variance_threshold: float = SATURATION_VARIANCE_THRESHOLD,
    entropy_threshold: float = SATURATION_ENTROPY_THRESHOLD,
) -> bool:
    """Milestone 14B Signal 3, factored out of the epoch loop as a pure
    function (ADR-010) for the same regression-testability reason as
    `_check_cumulative_drift`. TWO independent signals -- batch variance
    and mean entropy -- since either alone can miss a real collapse:
    variance catches "every sample gets the same output"; entropy catches
    the subtler case where outputs still differ slightly across samples
    but each individual prediction has collapsed to a near one-hot spike.
    """
    fired = batch_variance < variance_threshold or mean_entropy < entropy_threshold
    if fired:
        logger.warning(
            f"[{model_type}] SATURATION WARNING at epoch {epoch}: output batch "
            f"variance={batch_variance:.2e} (threshold <{variance_threshold:.0e}), "
            f"mean entropy={mean_entropy:.4f} (threshold <{entropy_threshold}) -- "
            "predictions have collapsed to a near-constant or near-one-hot output "
            "regardless of input."
        )
    return fired


def _check_frozen_val_loss(
    model_type: str,
    epoch: int,
    val_loss_history: dict[int, float],
    epoch_val_loss: float,
) -> bool:
    """Milestone 14B Signal 4 (the weakest of the four -- only fires after
    a model has already fully saturated), factored out of the epoch loop
    as a pure function (ADR-010) for the same regression-testability
    reason as the two checks above.
    """
    if not val_loss_history:
        return False
    previous_check_epoch = max(val_loss_history.keys())
    fired = val_loss_history[previous_check_epoch] == epoch_val_loss
    if fired:
        logger.warning(
            f"[{model_type}] FROZEN VAL LOSS WARNING at epoch {epoch}: val_loss is "
            f"bit-for-bit identical to epoch {previous_check_epoch}'s value "
            f"({epoch_val_loss}) -- the weakest of these signals, since it only "
            "fires after the model has already fully saturated."
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

        logger.info(
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
                    logger.warning(f"[{model_type}] NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Stopping.")
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
                logger.info(f"  [{model_type}] epoch {epoch:3d}/{NUM_EPOCHS}: training loss = {final_epoch_loss:.4f}")

            # Signal 2 (Milestone 14B): cumulative/windowed drift check.
            # Compares against the loss CUMULATIVE_DRIFT_WINDOW_EPOCHS
            # epochs prior, not just the immediately preceding epoch --
            # this is what would have caught Milestone 14's actual failure
            # (a steady, sub-spike-threshold climb across many epochs).
            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 and epoch > CUMULATIVE_DRIFT_WINDOW_EPOCHS:
                drift_fired_this_epoch, drift_fraction = _check_cumulative_drift(
                    model_type, epoch_losses, epoch
                )
                mlflow.log_metric("cumulative_drift_fraction", drift_fraction, step=epoch)
                cumulative_drift_fired = cumulative_drift_fired or drift_fired_this_epoch

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

                saturation_fired = saturation_fired or _check_saturation(
                    model_type, epoch, batch_variance, mean_entropy
                )

                # Signal 4 (backstop, weakest of the four -- see module
                # docstring comment): bit-for-bit frozen val_loss. Only
                # fires after the model has ALREADY fully saturated; exact
                # floating-point equality across resumed computation is
                # what confirmed Milestone 14's collapse, well after the
                # drift had already started.
                frozen_val_loss_fired = frozen_val_loss_fired or _check_frozen_val_loss(
                    model_type, epoch, val_loss_history, epoch_val_loss
                )
                val_loss_history[epoch] = epoch_val_loss

        logger.info(f"[{model_type}] Final training loss: {final_epoch_loss:.4f}")

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
        logger.info(
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
            logger.info(f"[{model_type}] Validation loss: {val_loss.item():.4f}")

            briers = {}
            for time_bin in BRIER_TIME_BINS:
                brier, num_excluded = calculate_brier_score(
                    val_predictions, val_duration_bins, val_duration_bins, val_events, time_bin
                )
                seconds = time_bin * 5.0
                logger.info(f"  [{model_type}] time_bin={time_bin} ({seconds:.0f}s): Brier Score = {brier:.4f}")
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

        logger.info(f"[{model_type}] MLflow run ID: {run.info.run_id}")

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
                "split_type": "match_level",  # Milestone 35 / ADR-011
            }
        )

        logger.info(
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
                    logger.warning(f"[DeepEnsemble] NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Stopping.")
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
                logger.info(f"  [DeepEnsemble] epoch {epoch:3d}/{NUM_EPOCHS}: mean-member training loss = {final_epoch_loss:.4f}")

            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 and epoch > CUMULATIVE_DRIFT_WINDOW_EPOCHS:
                drift_fired_this_epoch, drift_fraction = _check_cumulative_drift(
                    "DeepEnsemble", epoch_losses, epoch
                )
                mlflow.log_metric("cumulative_drift_fraction", drift_fraction, step=epoch)
                cumulative_drift_fired = cumulative_drift_fired or drift_fired_this_epoch

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

                saturation_fired = saturation_fired or _check_saturation(
                    "DeepEnsemble", epoch, batch_variance, mean_entropy
                )
                frozen_val_loss_fired = frozen_val_loss_fired or _check_frozen_val_loss(
                    "DeepEnsemble", epoch, val_loss_history, epoch_val_loss
                )
                val_loss_history[epoch] = epoch_val_loss

        logger.info(f"[DeepEnsemble] Final mean-member training loss: {final_epoch_loss:.4f}")

        spike_fired = _check_for_instability("DeepEnsemble", epoch_losses)
        instability_warning_fired = (
            spike_fired or cumulative_drift_fired or saturation_fired or frozen_val_loss_fired
        )
        mlflow.log_param("spike_warning_fired", spike_fired)
        mlflow.log_param("cumulative_drift_warning_fired", cumulative_drift_fired)
        mlflow.log_param("saturation_warning_fired", saturation_fired)
        mlflow.log_param("frozen_val_loss_warning_fired", frozen_val_loss_fired)
        mlflow.log_param("instability_warning_fired", instability_warning_fired)
        logger.info(
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
            logger.info(f"[DeepEnsemble] Validation loss (mean-member): {val_loss.item():.4f}")

            briers = {}
            for time_bin in BRIER_TIME_BINS:
                brier, num_excluded = calculate_brier_score(
                    mean_pmf, val_duration_bins, val_duration_bins, val_events, time_bin
                )
                seconds = time_bin * 5.0
                logger.info(f"  [DeepEnsemble] time_bin={time_bin} ({seconds:.0f}s): Brier Score (mean PMF) = {brier:.4f}")
                briers[time_bin] = (brier, num_excluded)

            # Step 2.4: diversity metric -- mean, across the validation set,
            # of each sample's cross-member standard deviation of
            # cumulative incidence at time_bin=3. A collapsed (non-diverse)
            # ensemble would show this near zero; logged explicitly so that
            # would be visible in MLflow, not just assumed away.
            diversity_std_ci_15s = std_cumulative_incidence.mean().item()
            logger.info(f"[DeepEnsemble] Diversity metric (mean std of per-member CI@15s across val set): {diversity_std_ci_15s:.6f}")

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

        logger.info(f"[DeepEnsemble] MLflow run ID: {run.info.run_id}")

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


def _train_and_log_batch_ensemble(
    model: BatchEnsembleDeepHit,
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
    """True Batch Ensemble (ADR-004 Update) analog of
    `_train_and_log_deep_ensemble` immediately above -- a DELIBERATE,
    close mirror of that function (same hyperparameters, same disentangled
    per-member loss via the SAME `compute_disentangled_ensemble_loss`
    reused unchanged from `deep_ensemble.py`, same ADR-010 four-signal
    health gate, same MLflow logging conventions), not a generalization of
    it. Kept as a genuinely SEPARATE function -- not a shared helper
    parameterized by model type -- for the same reason
    `_train_and_log_deep_ensemble` itself was kept separate from the
    single-MLP `_train_and_log_model` loop: forcing two meaningfully
    different training paths through one shared function whose behavior
    silently branches on model type would obscure, not clarify, the real
    difference between them, and would risk the Deep Ensemble's own
    already-validated path being touched to accommodate this new one.
    `BatchEnsembleDeepHit`'s SAME `[M, B, num_bins]` output shape and SAME
    `predict_with_uncertainty` interface as `DeepEnsembleDeepHit` (see
    that class's own docstring) is exactly what makes this mirror possible
    without any model-specific special-casing inside the loop body itself.
    """
    loss_fn = DeepHitLoss()
    M = model.M

    with mlflow.start_run(run_name="batch_ensemble_run") as run:
        if run_tags:
            mlflow.set_tags(run_tags)

        mlflow.log_params(
            {
                "model_type": "BatchEnsemble_MLP",
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
                "ensemble_kind": "true_batch_ensemble_shared_trunk",  # see ADR-004 Update
                "split_type": "match_level",  # Milestone 35 / ADR-011
            }
        )

        logger.info(
            f"\n[BatchEnsemble] Training M={M} members (shared trunk + per-member rank-1 fast "
            f"weights) for {NUM_EPOCHS} epochs on {n_train} samples ({n_val} held out for validation)..."
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
                    logger.warning(f"[BatchEnsemble] NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Stopping.")
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
                logger.info(f"  [BatchEnsemble] epoch {epoch:3d}/{NUM_EPOCHS}: mean-member training loss = {final_epoch_loss:.4f}")

            if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 and epoch > CUMULATIVE_DRIFT_WINDOW_EPOCHS:
                drift_fired_this_epoch, drift_fraction = _check_cumulative_drift(
                    "BatchEnsemble", epoch_losses, epoch
                )
                mlflow.log_metric("cumulative_drift_fraction", drift_fraction, step=epoch)
                cumulative_drift_fired = cumulative_drift_fired or drift_fired_this_epoch

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

                saturation_fired = saturation_fired or _check_saturation(
                    "BatchEnsemble", epoch, batch_variance, mean_entropy
                )
                frozen_val_loss_fired = frozen_val_loss_fired or _check_frozen_val_loss(
                    "BatchEnsemble", epoch, val_loss_history, epoch_val_loss
                )
                val_loss_history[epoch] = epoch_val_loss

        logger.info(f"[BatchEnsemble] Final mean-member training loss: {final_epoch_loss:.4f}")

        spike_fired = _check_for_instability("BatchEnsemble", epoch_losses)
        instability_warning_fired = (
            spike_fired or cumulative_drift_fired or saturation_fired or frozen_val_loss_fired
        )
        mlflow.log_param("spike_warning_fired", spike_fired)
        mlflow.log_param("cumulative_drift_warning_fired", cumulative_drift_fired)
        mlflow.log_param("saturation_warning_fired", saturation_fired)
        mlflow.log_param("frozen_val_loss_warning_fired", frozen_val_loss_fired)
        mlflow.log_param("instability_warning_fired", instability_warning_fired)
        logger.info(
            f"[BatchEnsemble] Warning summary -- spike: {spike_fired}, cumulative_drift: "
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
            logger.info(f"[BatchEnsemble] Validation loss (mean-member): {val_loss.item():.4f}")

            briers = {}
            for time_bin in BRIER_TIME_BINS:
                brier, num_excluded = calculate_brier_score(
                    mean_pmf, val_duration_bins, val_duration_bins, val_events, time_bin
                )
                seconds = time_bin * 5.0
                logger.info(f"  [BatchEnsemble] time_bin={time_bin} ({seconds:.0f}s): Brier Score (mean PMF) = {brier:.4f}")
                briers[time_bin] = (brier, num_excluded)

            diversity_std_ci_15s = std_cumulative_incidence.mean().item()
            logger.info(f"[BatchEnsemble] Diversity metric (mean std of per-member CI@15s across val set): {diversity_std_ci_15s:.6f}")

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

        mlflow.pytorch.log_model(model, name="batch_ensemble_model", serialization_format="pickle")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(normalization_artifact, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        try:
            mlflow.log_artifact(tmp_file_path, artifact_path="normalization")
        finally:
            os.remove(tmp_file_path)

        logger.info(f"[BatchEnsemble] MLflow run ID: {run.info.run_id}")

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


def _build_habit_blended_features(
    frames: list[dict],
    sample_match_ids: list[int],
    used_match_ids: list[int],
    train_indices: list[int],
    val_indices: list[int],
    engine: BiomechanicalPitchControl,
) -> tuple[list[dict], dict]:
    """Milestone 23 Step 1 (SIMPLIFIED per Milestone 35 / ADR-011):
    re-extracts scalar features for EVERY sample with habit blending
    enabled, using a match-aware AND split-aware heatmap for each sample's
    acting player.

    Leakage discipline (both rules hold simultaneously):
      (a) the current sample's own match is always excluded from ITS OWN
          heatmap (Milestone 22's base `exclude_match_id` rule), and
      (b) every match in the VALIDATION group is excluded from the
          training-bucket corpus entirely.

    Rule (b) used to require an explicit CONSERVATIVE workaround: under
    the sample-level split used through Milestone 34, most non-trivial
    matches contributed samples to BOTH splits under random per-sample
    assignment, so "any match with at least one validation-split sample"
    had to be excluded outright, shrinking the usable historical corpus to
    just 4 of ~55 matches (Milestone 23's finding). As of Milestone 35's
    MATCH-level split (`data_split.match_level_split`), a match can no
    longer straddle both groups AT ALL -- `training_match_ids` below is
    now an exact, complete partition (every non-validation match, full
    stop), not a conservative subset working around a straddling problem
    that no longer exists. The computation below is unchanged (a plain set
    difference); what changed is that it's no longer compensating for
    anything.

    SCOPE LIMITATION (still open, NOT fixed here): heatmaps use ANY OTHER
    training-split match regardless of its real-world chronological date
    relative to the match being predicted -- true "only past matches"
    chronological ordering is NOT enforced. This tests whether the
    Bayesian blending MECHANISM helps prediction, not a fully faithful
    simulation of live deployment (where only genuinely past matches would
    be available at inference time). Match-level splitting does not
    address this; it is an independent limitation.

    Returns (blended_features, diagnostics) -- blended_features has the
    exact same length/order as `frames` (only feature VALUES differ from
    the unblended pass); diagnostics reports the counts Step 1.5 asks for.
    """
    val_match_ids = {sample_match_ids[i] for i in val_indices}
    all_match_ids_set = set(used_match_ids)
    training_match_ids = sorted(all_match_ids_set - val_match_ids)
    validation_match_ids = sorted(val_match_ids)

    logger.info(
        f"\n[Milestone 23/35] {len(training_match_ids)} training-split matches, "
        f"{len(validation_match_ids)} validation-split matches -- an exact, complete partition "
        "under Milestone 35's match-level split (no match straddles both groups, so none is held "
        "back beyond the validation matches themselves; see ADR-011)."
    )

    # Step 1.2: precompute per-player-per-match buckets ONCE, from training
    # matches' FULL event logs (every action that player took in that
    # match, not just the handful of chain-representative sample events).
    events_by_training_match = {
        match_id: fetch_match_events(match_id) for match_id in training_match_ids
    }
    buckets = build_player_match_buckets(events_by_training_match)
    del events_by_training_match  # same "don't linger on raw fetched JSON" precaution as build_training_data

    blended_features = []
    actor_ids_seen = set()
    cold_start_count = 0
    samples_with_known_actor = 0

    for i, frame in enumerate(frames):
        actor_player_id = frame.get("actor_player_id")
        habit_heatmaps = None
        if actor_player_id is not None:
            heatmap, _num_events, is_cold_start = heatmap_from_buckets(
                actor_player_id,
                buckets,
                training_match_ids,
                exclude_match_id=sample_match_ids[i],
            )
            habit_heatmaps = {actor_player_id: heatmap}
            actor_ids_seen.add(actor_player_id)
            samples_with_known_actor += 1
            if is_cold_start:
                cold_start_count += 1

        blended_features.append(extract_features(frame, engine, habit_heatmaps=habit_heatmaps))

    diagnostics = {
        "training_match_count": len(training_match_ids),
        "validation_match_count": len(validation_match_ids),
        "unique_actor_count": len(actor_ids_seen),
        "cold_start_count": cold_start_count,
        "samples_with_known_actor": samples_with_known_actor,
        "total_samples": len(frames),
    }
    logger.info(
        f"[Milestone 23] {diagnostics['unique_actor_count']} unique actors found across "
        f"{diagnostics['samples_with_known_actor']}/{diagnostics['total_samples']} samples with a "
        f"known actor; {diagnostics['cold_start_count']} of those fell back to the uniform "
        f"cold-start prior (< {MIN_HISTORICAL_EVENTS} historical events)."
    )
    return blended_features, diagnostics


def _load_and_split_dataset() -> dict:
    """Data loading + Milestone 35 match-level split (ADR-011), extracted
    from `train_and_evaluate()` (engineering-review action item: the
    ADR-010 pure-named-sub-step extraction pattern, applied to the
    OTHER large function in this file). Consumes no global RNG state
    itself (`match_level_split` takes its own explicit `seed` argument,
    not the global torch generator), so extracting it cannot change the
    RNG sequence any later, seeded stage depends on.
    """
    features, frames, chains, source_event_ids, match_ids, qualifying_competitions, sample_match_ids = (
        build_training_data()
    )
    match_count = len(match_ids)
    dataset_size = len(features)
    competition_season_summary = ",".join(
        f"{c['competition_id']}:{c['season_id']}" for c in qualifying_competitions
    )
    logger.info(f"\nTotal (feature, frame, chain) triples across {match_count} matches: {dataset_size}")
    if dataset_size < SMALL_DATASET_WARNING_THRESHOLD:
        logger.info(
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
    logger.info(
        f"\nSame-frame spot check (chain 0): scalar-feature source event_id="
        f"{source_event_ids[0]}, graph-data source event_id={source_event_ids[0]} "
        "(identical, by construction -- both were built from one resolved parse_360_frame call)"
    )

    dataset = TacticalSurvivalDataset(features, frames, chains)

    # Milestone 35 (ADR-011): MATCH-level split, replacing the SAMPLE-level
    # random_split used through Milestone 34. A match can no longer
    # contribute samples to both train and val -- see ADR-011 for why
    # sample-level splitting became a real limitation once Milestone 23's
    # habit-memory heatmaps needed match-level exclusion (only 4 of ~55
    # matches ended up training-bucket-eligible under the old approach).
    train_indices, val_indices = match_level_split(
        sample_match_ids, val_fraction=1.0 - TRAIN_FRACTION, seed=RANDOM_SEED
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    n_train, n_val = len(train_set), len(val_set)
    logger.info(
        f"\n[Milestone 35] resulting sample-level ratio: {n_train} train / {n_val} val "
        f"({n_train / (n_train + n_val):.1%} / {n_val / (n_train + n_val):.1%}) -- NOT forced to "
        f"exactly {TRAIN_FRACTION:.0%}/{1 - TRAIN_FRACTION:.0%} since matches contribute different "
        "sample counts; see match_level_split's own report above for the match-count split and "
        "any single-match imbalance warning."
    )
    # Both models train on this exact same split (same indices) -- guard
    # that assumption explicitly rather than leaving it implicit.
    assert len(train_set) == n_train and len(val_set) == n_val

    return {
        "features": features,
        "frames": frames,
        "chains": chains,
        "source_event_ids": source_event_ids,
        "match_ids": match_ids,
        "qualifying_competitions": qualifying_competitions,
        "sample_match_ids": sample_match_ids,
        "competition_season_summary": competition_season_summary,
        "dataset": dataset,
        "train_set": train_set,
        "val_set": val_set,
        "n_train": n_train,
        "n_val": n_val,
        "dataset_size": dataset_size,
    }


def _compute_normalization_stats(
    dataset: TacticalSurvivalDataset, train_set: Subset
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar (Milestone 7) AND graph (Milestone 12 Step 2.3) feature
    normalization, both training-split-only (never the full dataset) --
    a pure, independently-testable computation (feed it a small synthetic
    dataset + index list, check the returned mean/std), extracted per the
    review's own suggested 'data loading/normalization' category. Kept as
    ONE function, not two, since both stats are always computed together,
    in sequence, with no branching between them.

    Returns `(feature_mean, feature_std, graph_feature_mean, graph_feature_std)`.
    """
    n_train = len(train_set)

    # Scalar feature normalization (Milestone 7's rule, unchanged):
    # statistics computed from the TRAINING split ONLY, after the split.
    train_features_raw = torch.stack([dataset[i][0] for i in train_set.indices])
    feature_mean = train_features_raw.mean(dim=0)
    feature_std = train_features_raw.std(dim=0).clamp(min=1e-8)  # guard a constant feature
    logger.info(f"\nScalar feature normalization stats (from {n_train} training samples):")
    logger.info(f"  mean: {feature_mean.tolist()}")
    logger.info(f"  std:  {feature_std.tolist()}")

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
    logger.info(f"\nGraph node feature normalization stats (x, y, dist_to_ball; from {n_train} training samples):")
    logger.info(f"  mean: {graph_feature_mean.tolist()}")
    logger.info(f"  std:  {graph_feature_std.tolist()}")

    return feature_mean, feature_std, graph_feature_mean, graph_feature_std


def _run_mlp_stabilization_and_robustness_check(
    train_loader: DataLoader,
    val_batch: tuple,
    n_train: int,
    n_val: int,
    match_ids: list[int],
    dataset_size: int,
    competition_season_summary: str,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[int, dict | None]:
    """Milestone 14B Step 2: MLP stabilization (same bundle as the GNN),
    PLUS a second weight-init seed as a robustness check. `train_loader`
    is the SAME shared object across every call below (as in every prior
    milestone's MLP-vs-GNN comparison) -- its shuffle order naturally
    differs call-to-call as its internal generator advances, but the
    train/val SPLIT itself (train_set/val_set, from split_generator) is
    held fixed at RANDOM_SEED=42 for every run in this function. Only
    each MLP run's WEIGHT INITIALIZATION seed differs (42 vs 43),
    isolating exactly the one variable Step 2.3 asks about.
    """
    logger.info("\n=== Milestone 14B Step 2: MLP stabilization + robustness check ===")
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
                "split_type": "match_level",  # Milestone 35 / ADR-011
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

    return mlp_seed_results


def _run_gnn_stage(
    train_loader: DataLoader,
    val_batch: tuple,
    n_train: int,
    n_val: int,
    match_ids: list[int],
    dataset_size: int,
    competition_season_summary: str,
    graph_feature_mean: torch.Tensor,
    graph_feature_std: torch.Tensor,
) -> dict | None:
    """Milestone 14B Step 3: GNN retrain, with the exact same
    hyperparameters as the MLP (both use the stabilization bundle) --
    addresses the asymmetry noted (but not fixed) in Milestones 12B/14.
    """
    # Reset the global RNG state before constructing the GNN, so its own
    # initialization isn't accidentally coupled to wherever the MLP seed
    # loop left the global generator. MUST remain the first statement in
    # this function -- the caller invokes this immediately after the MLP
    # stage returns, preserving the exact same RNG-consumption sequence
    # the pre-decomposition inline code had.
    torch.manual_seed(RANDOM_SEED)

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
            "split_type": "match_level",  # Milestone 35 / ADR-011
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

    return gnn_results


def _run_deep_ensemble_stage(
    train_loader: DataLoader,
    val_batch: tuple,
    n_train: int,
    n_val: int,
    match_ids: list[int],
    dataset_size: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict | None:
    """Milestone 21: Deep Ensemble uncertainty quantification (ADR-004).
    Same split, same normalization, same stabilization bundle as the
    Milestone 14B MLP/GNN -- only the model and its per-member-
    disentangled loss loop differ (see `_train_and_log_deep_ensemble`).
    Includes Milestone 21's own warning-summary + baseline-comparison
    report, since that report is this stage's own self-contained output,
    not shared state other stages need.
    """
    # Reset the global RNG state before constructing the ensemble, same
    # reasoning as before the GNN above: its M independent members'
    # initialization shouldn't be accidentally coupled to wherever the GNN
    # training loop left the global generator. MUST remain the first
    # statement -- see `_run_gnn_stage`'s identical comment.
    torch.manual_seed(RANDOM_SEED)

    logger.info(f"\n=== Milestone 21: Deep Ensemble (M={DEEP_ENSEMBLE_M}) training ===")
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

    logger.info("\n=== Milestone 21: Deep Ensemble warning summary + baseline comparison ===")
    if deep_ensemble_results is None:
        logger.warning("DeepEnsemble: training ABORTED (NaN/Inf loss).")
    else:
        logger.info(
            f"DeepEnsemble: spike={deep_ensemble_results['spike_fired']}, "
            f"cumulative_drift={deep_ensemble_results['cumulative_drift_fired']}, "
            f"saturation={deep_ensemble_results['saturation_fired']}, "
            f"frozen_val_loss={deep_ensemble_results['frozen_val_loss_fired']}"
        )
        logger.info(
            f"DeepEnsemble diversity metric (mean std of per-member CI@15s): "
            f"{deep_ensemble_results['diversity_std_ci_15s']:.6f}"
        )
        logger.info(
            f"\n{'Model':<40} {'Params (approx)':>16} {'Brier@15s':>10} {'Brier@30s':>10}"
        )
        logger.info(
            f"{'Single MLP (Milestone 14B, seed=42)':<40} {'1x':>16} "
            f"{MILESTONE_14B_MLP_BRIER_15S:>10.4f} {MILESTONE_14B_MLP_BRIER_30S:>10.4f}"
        )
        logger.info(
            f"{f'Deep Ensemble (M={DEEP_ENSEMBLE_M}, mean PMF)':<40} {f'~{DEEP_ENSEMBLE_M}x':>16} "
            f"{deep_ensemble_results['brier_15s']:>10.4f} {deep_ensemble_results['brier_30s']:>10.4f}"
        )
        logger.info(
            "NOTE: this is NOT an equal-capacity comparison -- the Deep Ensemble has ~"
            f"{DEEP_ENSEMBLE_M}x the parameters and training/inference compute of the single MLP "
            "(same caveat already established for the Milestone 12 MLP-vs-GNN comparison). A "
            "lower ensemble Brier Score is evidence the mean-PMF prediction is at least as good, "
            "not evidence the ensembling technique itself is more parameter-efficient."
        )

    return deep_ensemble_results


def _run_batch_ensemble_stage(
    train_loader: DataLoader,
    val_batch: tuple,
    n_train: int,
    n_val: int,
    match_ids: list[int],
    dataset_size: int,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict | None:
    """ADR-004 Update: true Batch Ensemble uncertainty quantification --
    the technique README.txt's Module 7 always specified, implemented for
    real here for the first time (see `batch_ensemble.py`'s own module
    docstring). SAME split, SAME normalization, SAME stabilization bundle
    as `_run_deep_ensemble_stage` immediately above, deliberately, so a
    caller running both back to back gets a genuinely apples-to-apples
    A/B comparison -- only the model and its parameter-sharing strategy
    differ.
    """
    # Same RNG-reset discipline as `_run_deep_ensemble_stage` and
    # `_run_gnn_stage` before it -- MUST remain the first statement.
    torch.manual_seed(RANDOM_SEED)

    logger.info(f"\n=== ADR-004 Update: Batch Ensemble (M={DEEP_ENSEMBLE_M}) training ===")
    batch_ensemble_model = BatchEnsembleDeepHit(
        num_features=len(FEATURE_KEYS), num_bins=NUM_BINS, M=DEEP_ENSEMBLE_M
    )
    batch_ensemble_optimizer = torch.optim.Adam(
        batch_ensemble_model.parameters(),
        lr=MLP_STABILIZED_LR,
        weight_decay=MLP_STABILIZED_WEIGHT_DECAY,
    )
    batch_ensemble_results = _train_and_log_batch_ensemble(
        model=batch_ensemble_model,
        optimizer=batch_ensemble_optimizer,
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
                "True Batch Ensemble (ADR-004 Update -- a shared trunk + per-member rank-1 "
                "fast weights, NOT M fully independent models), compared against the "
                "Milestone 14B single-MLP baseline AND the existing Deep Ensemble (M=5, "
                "fully independent) for a genuine A/B."
            ),
        },
    )

    logger.info("\n=== ADR-004 Update: Batch Ensemble warning summary + baseline comparison ===")
    if batch_ensemble_results is None:
        logger.warning("BatchEnsemble: training ABORTED (NaN/Inf loss).")
    else:
        logger.info(
            f"BatchEnsemble: spike={batch_ensemble_results['spike_fired']}, "
            f"cumulative_drift={batch_ensemble_results['cumulative_drift_fired']}, "
            f"saturation={batch_ensemble_results['saturation_fired']}, "
            f"frozen_val_loss={batch_ensemble_results['frozen_val_loss_fired']}"
        )
        logger.info(
            f"BatchEnsemble diversity metric (mean std of per-member CI@15s): "
            f"{batch_ensemble_results['diversity_std_ci_15s']:.6f}"
        )
        logger.info(
            f"\n{'Model':<40} {'Params (approx)':>16} {'Brier@15s':>10} {'Brier@30s':>10}"
        )
        logger.info(
            f"{'Single MLP (Milestone 14B, seed=42)':<40} {'1x':>16} "
            f"{MILESTONE_14B_MLP_BRIER_15S:>10.4f} {MILESTONE_14B_MLP_BRIER_30S:>10.4f}"
        )
        logger.info(
            f"{f'Batch Ensemble (M={DEEP_ENSEMBLE_M}, mean PMF)':<40} {'~1.64x':>16} "
            f"{batch_ensemble_results['brier_15s']:>10.4f} {batch_ensemble_results['brier_30s']:>10.4f}"
        )
        logger.info(
            "NOTE: ~1.64x params (this project's tiny 4->32->32->12 architecture -- see "
            "batch_ensemble.py's own docstring for the exact parameter accounting), vs. the "
            f"Deep Ensemble's ~{DEEP_ENSEMBLE_M}x -- a real, measured reduction, but see "
            "ADR-004's own Update section for whether that reduction is the thing that "
            "actually mattered here."
        )

    return batch_ensemble_results


def _run_habit_blended_stage(
    frames: list[dict],
    chains: list[dict],
    sample_match_ids: list[int],
    match_ids: list[int],
    train_set: Subset,
    val_set: Subset,
    n_train: int,
    n_val: int,
    dataset_size: int,
    competition_season_summary: str,
    deep_ensemble_results: dict | None,
) -> tuple[dict | None, dict]:
    """Milestone 23 (RQ2): MLP with Bayesian habit blending enabled. Same
    hyperparameters, split, and instability detector as the Milestone 14B
    baseline -- the ONLY difference is that features are re-extracted
    with habit_heatmaps populated per Step 1's match-aware/split-aware
    discipline. Includes the RQ2 conclusion report, since it's this
    stage's own self-contained output (it needs `deep_ensemble_results`
    only to print one extra comparison row, not as a real dependency).

    Returns `(habit_mlp_results, habit_diagnostics)`.
    """
    logger.info("\n=== Milestone 23 (RQ2): building habit-blended features ===")
    habit_blend_engine = BiomechanicalPitchControl()
    habit_blended_features, habit_diagnostics = _build_habit_blended_features(
        frames=frames,
        sample_match_ids=sample_match_ids,
        used_match_ids=match_ids,
        train_indices=train_set.indices,
        val_indices=val_set.indices,
        engine=habit_blend_engine,
    )

    habit_dataset = TacticalSurvivalDataset(habit_blended_features, frames, chains)
    # Reuse the EXACT SAME sample indices already determined above (not a
    # fresh random_split call) -- guarantees the habit-blended run trains
    # on the identical partition the leakage guard itself was computed
    # against, rather than relying on RNG determinism to reproduce it.
    habit_train_subset = Subset(habit_dataset, train_set.indices)
    habit_val_subset = Subset(habit_dataset, val_set.indices)

    # Normalization stats recomputed from the BLENDED training split only
    # (Milestone 7's rule) -- blending shifts feature values, so reusing
    # the unblended mean/std here would normalize against the wrong
    # reference distribution.
    habit_train_features_raw = torch.stack([habit_dataset[i][0] for i in train_set.indices])
    habit_feature_mean = habit_train_features_raw.mean(dim=0)
    habit_feature_std = habit_train_features_raw.std(dim=0).clamp(min=1e-8)

    torch.manual_seed(RANDOM_SEED)
    habit_train_loader = DataLoader(
        habit_train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    habit_val_batch = next(iter(DataLoader(habit_val_subset, batch_size=len(habit_val_subset))))

    habit_mlp_model = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
    habit_mlp_optimizer = torch.optim.Adam(
        habit_mlp_model.parameters(),
        lr=MLP_STABILIZED_LR,
        weight_decay=MLP_STABILIZED_WEIGHT_DECAY,
    )
    habit_mlp_results = _train_and_log_model(
        model_type="MLP",
        model=habit_mlp_model,
        optimizer=habit_mlp_optimizer,
        lr=MLP_STABILIZED_LR,
        weight_decay=MLP_STABILIZED_WEIGHT_DECAY,
        clip_grad_norm=True,
        input_fn=_normalize_scalar_batch,
        normalize_args=(habit_feature_mean, habit_feature_std),
        train_loader=habit_train_loader,
        val_batch=habit_val_batch,
        n_train=n_train,
        n_val=n_val,
        match_ids=match_ids,
        dataset_size=dataset_size,
        extra_params={
            "dataset_scale": "multi_competition",
            "competition_season_pairs": competition_season_summary,
            "stabilization_bundle": True,
            "saturation_check_v2": True,
            "init_seed": RANDOM_SEED,
            "split_type": "match_level",  # Milestone 35 / ADR-011
            "habit_blending": True,
            "habit_unique_actor_count": habit_diagnostics["unique_actor_count"],
            "habit_cold_start_count": habit_diagnostics["cold_start_count"],
            "habit_training_match_count": habit_diagnostics["training_match_count"],
            "habit_validation_match_count": habit_diagnostics["validation_match_count"],
        },
        run_tags={
            "baseline_single_mlp_run_id": MILESTONE_14B_MLP_RUN_ID,
            "description": (
                "Milestone 23 (RQ2): MLP trained with Bayesian habit blending enabled for the "
                "acting player only (Milestone 22's data-availability constraint). Heatmaps built "
                "train-split-only, current-match-excluded, per Step 1's match-aware/split-aware "
                "discipline -- ANY match with a validation-split sample is excluded from the "
                "training bucket corpus entirely. Non-chronological: any other training-split "
                "match may contribute to a heatmap regardless of real-world date order."
            ),
        },
        normalization_artifact={
            "feature_key_order": list(FEATURE_KEYS),
            "mean": habit_feature_mean.tolist(),
            "std": habit_feature_std.tolist(),
        },
    )

    logger.info("\n=== Milestone 23 (RQ2): habit-blended MLP vs. non-blended baselines ===")
    if habit_mlp_results is None:
        logger.warning("Habit-blended MLP: training ABORTED (NaN/Inf loss).")
    else:
        logger.info(
            f"Habit-blended MLP: spike={habit_mlp_results['spike_fired']}, "
            f"cumulative_drift={habit_mlp_results['cumulative_drift_fired']}, "
            f"saturation={habit_mlp_results['saturation_fired']}, "
            f"frozen_val_loss={habit_mlp_results['frozen_val_loss_fired']}"
        )
        logger.info(f"\n{'Model':<40} {'Brier@15s':>10} {'Brier@30s':>10}")
        logger.info(f"{'MLP, no blending (Milestone 14B)':<40} {MILESTONE_14B_MLP_BRIER_15S:>10.4f} {MILESTONE_14B_MLP_BRIER_30S:>10.4f}")
        logger.info(f"{'MLP, habit blending (Milestone 23)':<40} {habit_mlp_results['brier_15s']:>10.4f} {habit_mlp_results['brier_30s']:>10.4f}")
        if deep_ensemble_results is not None:
            logger.info(f"{f'Deep Ensemble, no blending (M={DEEP_ENSEMBLE_M}, Milestone 21)':<40} {deep_ensemble_results['brier_15s']:>10.4f} {deep_ensemble_results['brier_30s']:>10.4f}")

        brier_15s_delta = habit_mlp_results["brier_15s"] - MILESTONE_14B_MLP_BRIER_15S
        brier_30s_delta = habit_mlp_results["brier_30s"] - MILESTONE_14B_MLP_BRIER_30S
        habit_healthy = not habit_mlp_results["instability_warning_fired"]

        logger.info(
            f"\nBrier delta vs. non-blended baseline: {brier_15s_delta:+.4f} @15s, "
            f"{brier_30s_delta:+.4f} @30s (negative = habit blending improved the score)."
        )

        logger.info("\n=== Step 3: RQ2 conclusion (conditional on evidence quality) ===")
        if not habit_healthy:
            logger.info(
                "The habit-blended MLP triggered an instability warning -- NOT issuing an RQ2 "
                "conclusion this run. Report the blocker, fix it, re-run -- do not force a verdict."
            )
        elif brier_15s_delta < 0 and brier_30s_delta < 0:
            logger.info(
                f"RQ2 SUPPORTED (hedged): habit blending improved Brier Score at both horizons "
                f"({brier_15s_delta:+.4f} @15s, {brier_30s_delta:+.4f} @30s). This is a notable "
                "result given how diluted the blended signal could be: only 1 of ~22 visible "
                "players' positions is ever blended per sample (the acting player, per Milestone "
                "22's data-availability constraint), inside features that are themselves sums "
                "over many players' pitch-control contributions. Scope limitations: (1) blending "
                "only ever affects the acting player, never the other ~21 visible players; "
                "(2) the historical corpus is NON-CHRONOLOGICAL -- any other training-split match "
                "may contribute to a heatmap regardless of real-world date order relative to the "
                "match being predicted, so this tests the blending MECHANISM, not a faithful "
                "live-deployment simulation."
            )
        else:
            logger.info(
                f"RQ2 NOT supported by this run: Brier Score did not improve at both horizons "
                f"({brier_15s_delta:+.4f} @15s, {brier_30s_delta:+.4f} @30s). Two distinct, "
                "non-exclusive candidate explanations, not one: (a) the actor-only constraint "
                "means the blended signal is heavily diluted within aggregate features summed "
                "over ~11 players per side -- a single player's Bayesian-adjusted position is a "
                "small perturbation within that sum; and/or (b) the chain's representative actor "
                "is not consistently on the attacking side (sometimes a defensive action's actor, "
                "per Milestone 7's frame-resolution logic), so blending doesn't consistently "
                "affect the same feature direction across samples, adding noise rather than "
                "signal. Scope limitations to name alongside this finding: the actor-only scope "
                "(Milestone 22) and the non-chronological historical corpus (Step 1.6) both apply "
                "regardless of direction -- this result should not be read as a definitive verdict "
                "on Bayesian habit blending in general, only on this specific, constrained "
                "implementation of it."
            )

    return habit_mlp_results, habit_diagnostics


def _evaluate_mlp_health(primary_mlp_results: dict | None) -> bool:
    """Step 2.2 (revised per ADR-010): is the stabilized MLP genuinely
    HEALTHY, not merely "not collapsed"? GATED on exactly two criteria:
    (1) none of the four principled signals fired (spike, cumulative
    drift, saturation/entropy, frozen val loss -- the entropy/variance
    probing that has resolved every real ambiguous case in this
    project's history, Milestones 12/14/23) and (2) Brier Score isn't
    catastrophically bad. A THIRD criterion this health check used to
    require -- "did total loss decrease by more than 10% from epoch 1 to
    the final epoch" -- was REMOVED from the gate per ADR-010: it
    produced a real false positive in Milestone 23 (a genuinely healthy,
    fast-converging-then-plateauing MLP, confirmed healthy only by
    manual entropy/variance probing), because a model that converges
    quickly within its first epoch and then correctly holds near its
    optimum for the rest of training shows exactly the same small
    further decrease this check would misread as "not learning." It is
    still computed and logged below as a non-blocking diagnostic --
    useful context for a human glance, never a gate. Extracted as its own
    function per the review's request, matching the SAME
    compute-and-log-then-return-a-bool shape as the ADR-010 `_check_*`
    signal functions above.
    """
    mlp_healthy = False
    if primary_mlp_results is not None and not primary_mlp_results["instability_warning_fired"]:
        first_loss = primary_mlp_results["epoch_losses"][0]
        last_loss = primary_mlp_results["epoch_losses"][-1]
        loss_decreased_meaningfully = (first_loss - last_loss) > 0.1 * first_loss
        brier_in_sane_range = (
            primary_mlp_results["brier_15s"] <= MLP_SANITY_BRIER_15S_CEILING
            and primary_mlp_results["brier_30s"] <= MLP_SANITY_BRIER_30S_CEILING
        )
        mlp_healthy = brier_in_sane_range
        logger.info(
            f"\nMLP (seed={RANDOM_SEED}) health check: no instability warnings=True, Brier in "
            f"sane range (<= {MLP_SANITY_BRIER_15S_CEILING:.4f} / "
            f"{MLP_SANITY_BRIER_30S_CEILING:.4f})={brier_in_sane_range} (actual: "
            f"{primary_mlp_results['brier_15s']:.4f} / {primary_mlp_results['brier_30s']:.4f})"
        )
        logger.info(
            f"  [diagnostic only, per ADR-010 NOT part of the health gate] total loss change "
            f"epoch 1 -> {NUM_EPOCHS}: {first_loss:.4f} -> {last_loss:.4f} "
            f"({'>' if loss_decreased_meaningfully else '<='} 10% of epoch-1 loss). A small or "
            "absent late-training decrease is EXPECTED and HEALTHY for a model that converged "
            "quickly and is now correctly holding near its optimum -- this line is informational "
            "only and never blocks a conclusion."
        )
        if not brier_in_sane_range:
            logger.warning(
                "WARNING: the 'same hyperparameters as the GNN' recipe produced a STABLE but "
                "apparently UNDERTRAINED MLP (Brier far worse than the Milestone 12B sanity "
                "floor) -- lr=1e-4 was tuned for SAGEConv's specific instability, not validated "
                "as appropriate for this MLP. The 'purely architectural, hyperparameter-neutral' "
                "comparison assumption does NOT hold cleanly here."
            )
    elif primary_mlp_results is not None:
        logger.info(f"\nMLP (seed={RANDOM_SEED}) still triggered an instability warning -- see summary above.")
    else:
        logger.info(f"\nMLP (seed={RANDOM_SEED}) training was aborted (NaN/Inf loss).")

    return mlp_healthy


def _report_run_summary_and_rq4_conclusion(
    mlp_seed_results: dict[int, dict | None],
    gnn_results: dict | None,
    dataset_size: int,
    n_train: int,
    n_val: int,
) -> None:
    """Step 3.2/3.4/5: warning summary across all runs, the ADR-010 MLP
    health gate, robustness-check reporting, the four-way comparison
    table, and the conditional RQ4 conclusion -- kept as ONE cohesive
    reporting function (not fragmented further, per the review's own
    "extract where it helps, don't atomize every line" guidance) since
    every piece depends on state (`any_warnings_anywhere`, `mlp_healthy`,
    primary/robustness results) derived earlier in this SAME sequence,
    not on independent inputs a caller would ever want to vary alone.
    """
    logger.info(f"\nDataset size: {dataset_size} total samples ({n_train} train / {n_val} val)")

    # === Step 3.2: did ANY of the four warning signals fire, for either
    # model, across BOTH MLP seeds? ===
    logger.info("\n=== Step 3: warning summary across all runs (strengthened detector) ===")
    any_warnings_anywhere = False
    for seed, results in mlp_seed_results.items():
        if results is None:
            logger.warning(f"MLP (seed={seed}): training ABORTED (NaN/Inf loss).")
            any_warnings_anywhere = True
            continue
        logger.info(
            f"MLP (seed={seed}): spike={results['spike_fired']}, "
            f"cumulative_drift={results['cumulative_drift_fired']}, "
            f"saturation={results['saturation_fired']}, "
            f"frozen_val_loss={results['frozen_val_loss_fired']}"
        )
        any_warnings_anywhere = any_warnings_anywhere or results["instability_warning_fired"]

    if gnn_results is None:
        logger.warning("GNN: training ABORTED (NaN/Inf loss).")
        any_warnings_anywhere = True
    else:
        logger.info(
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

    mlp_healthy = _evaluate_mlp_health(primary_mlp_results)

    logger.info(f"\nRobustness check (seed={robustness_seed}):")
    if robustness_mlp_results is not None:
        logger.info(
            f"  final train loss: {robustness_mlp_results['train_loss']:.4f}, any warning fired: "
            f"{robustness_mlp_results['instability_warning_fired']}, Brier@15s/30s: "
            f"{robustness_mlp_results['brier_15s']:.4f} / {robustness_mlp_results['brier_30s']:.4f}"
        )
        logger.info(
            "  (systematic-vs-one-off read: both seeds " +
            ("avoided every warning" if not any(
                mlp_seed_results[s]["instability_warning_fired"] for s in MLP_ROBUSTNESS_CHECK_SEEDS
                if mlp_seed_results[s] is not None
            ) else "did NOT both avoid every warning") +
            " -- see per-seed detail above.)"
        )
    else:
        logger.warning("  training ABORTED (NaN/Inf loss).")

    # === Step 3.4: four-row comparison table ===
    logger.info("\n=== Step 3.4: four-way comparison (Milestone 14 vs Milestone 14B) ===")
    logger.info(f"{'Model (run)':<48} {'Dataset':>8} {'Brier@15s':>10} {'Brier@30s':>10}")
    logger.info(
        f"{'MLP (Milestone 14, COLLAPSED, for the record)':<48} {MILESTONE_14_DATASET_SIZE:>8} "
        f"{MILESTONE_14_MLP_COLLAPSED_BRIER_15S:>10.4f} {MILESTONE_14_MLP_COLLAPSED_BRIER_30S:>10.4f}"
    )
    logger.info(
        f"{'GNN (Milestone 14, stable)':<48} {MILESTONE_14_DATASET_SIZE:>8} "
        f"{MILESTONE_14_GNN_STABLE_BRIER_15S:>10.4f} {MILESTONE_14_GNN_STABLE_BRIER_30S:>10.4f}"
    )
    if primary_mlp_results is not None:
        logger.info(
            f"{'MLP (Milestone 14B, stabilized)':<48} {dataset_size:>8} "
            f"{primary_mlp_results['brier_15s']:>10.4f} {primary_mlp_results['brier_30s']:>10.4f}"
        )
    else:
        logger.info(f"{'MLP (Milestone 14B, stabilized)':<48} {dataset_size:>8} {'ABORTED':>10} {'ABORTED':>10}")
    if gnn_results is not None:
        logger.info(
            f"{'GNN (Milestone 14B, same hyperparams as MLP)':<48} {dataset_size:>8} "
            f"{gnn_results['brier_15s']:>10.4f} {gnn_results['brier_30s']:>10.4f}"
        )
    else:
        logger.info(f"{'GNN (Milestone 14B, same hyperparams as MLP)':<48} {dataset_size:>8} {'ABORTED':>10} {'ABORTED':>10}")

    # === Step 5: conditional RQ4 conclusion -- ONLY if evidence quality supports one ===
    logger.info("\n=== Step 5: RQ4 conclusion (conditional on evidence quality) ===")
    if any_warnings_anywhere:
        logger.info(
            "At least one run (MLP seed 42, MLP seed 43, or GNN) triggered an instability "
            "warning under the strengthened detector, or aborted outright. Per Milestone 12B's "
            "precedent: NOT issuing an RQ4 conclusion this run. Report the blocker, fix it, "
            "re-run -- do not force a verdict."
        )
    elif not mlp_healthy:
        logger.info(
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
        logger.info(
            f"Both models are confirmed genuinely healthy: no warnings fired (spike, cumulative "
            f"drift, saturation, or frozen-val-loss) across both MLP seeds and the GNN, and the "
            f"MLP's loss decreased meaningfully with a sane Brier Score. MLP Brier@15s/30s = "
            f"{primary_mlp_results['brier_15s']:.4f} / {primary_mlp_results['brier_30s']:.4f}; "
            f"GNN = {gnn_results['brier_15s']:.4f} / {gnn_results['brier_30s']:.4f}."
        )
        if gnn_better_or_comparable:
            logger.info(
                "The GNN is competitive with or better than the (now genuinely healthy) MLP at "
                "this scale, using IDENTICAL hyperparameters for both -- real, if still "
                "single-run, evidence in favor of graph representations for RQ4. Given this "
                "project's history of surprises at exactly this comparison step (Milestones 12, "
                "14), this should be read as one data point, not a settled verdict -- consistent "
                "with the README's framing of RQs as working hypotheses."
            )
        else:
            logger.info(
                "The MLP outperforms the GNN even with both confirmed healthy and using identical "
                "hyperparameters -- RQ4's answer here leans toward the handcrafted scalar "
                "features, though still hedged given how often this exact comparison has moved "
                "across milestones as data scale and stabilization changed."
            )

    logger.info(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    logger.info("Run `mlflow ui` from the project root to inspect results visually.")


def train_and_evaluate():
    """Milestone 7/8/9/10/10B/12/14/14B/21/23/35 end-to-end orchestrator --
    decomposed (engineering-review action item, elevated above its
    originally-suggested medium-term priority) into the named sub-steps
    above, following the SAME pure-function extraction pattern ADR-010
    already established for the four instability-detector signals. This
    function's own job is now just SEQUENCING those steps in the exact
    order the pre-decomposition inline version executed them -- critical
    for determinism, since several steps consume global torch RNG state
    (each explicitly RESETS it via `torch.manual_seed` immediately before
    its own model construction, rather than relying on accumulated state
    from whatever ran before it), and Python function calls execute in
    the order they're CALLED, not defined -- so preserving call order
    here is what keeps every result bit-for-bit identical to before this
    decomposition, not just "the same code" in isolation.
    """
    torch.manual_seed(RANDOM_SEED)

    loaded = _load_and_split_dataset()
    dataset = loaded["dataset"]
    train_set = loaded["train_set"]
    val_set = loaded["val_set"]
    n_train = loaded["n_train"]
    n_val = loaded["n_val"]
    match_ids = loaded["match_ids"]
    dataset_size = loaded["dataset_size"]
    competition_season_summary = loaded["competition_season_summary"]

    feature_mean, feature_std, graph_feature_mean, graph_feature_std = _compute_normalization_stats(
        dataset, train_set
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_batch = next(iter(DataLoader(val_set, batch_size=len(val_set))))

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    mlp_seed_results = _run_mlp_stabilization_and_robustness_check(
        train_loader, val_batch, n_train, n_val, match_ids, dataset_size,
        competition_season_summary, feature_mean, feature_std,
    )

    gnn_results = _run_gnn_stage(
        train_loader, val_batch, n_train, n_val, match_ids, dataset_size,
        competition_season_summary, graph_feature_mean, graph_feature_std,
    )

    deep_ensemble_results = _run_deep_ensemble_stage(
        train_loader, val_batch, n_train, n_val, match_ids, dataset_size,
        feature_mean, feature_std,
    )

    habit_mlp_results, habit_diagnostics = _run_habit_blended_stage(
        loaded["frames"], loaded["chains"], loaded["sample_match_ids"], match_ids,
        train_set, val_set, n_train, n_val, dataset_size, competition_season_summary,
        deep_ensemble_results,
    )

    _report_run_summary_and_rq4_conclusion(
        mlp_seed_results, gnn_results, dataset_size, n_train, n_val,
    )


def run_match_level_split_mlp_smoke_test() -> dict | None:
    """Milestone 35 (ADR-011) validation smoke test -- NOT a replacement
    for `train_and_evaluate()`, and NOT a re-validation campaign.

    Trains ONLY the single (seed=RANDOM_SEED) stabilized MLP, with the
    EXACT same hyperparameters as Milestone 14B, under the NEW match-level
    split (`data_split.match_level_split`) instead of the sample-level
    split every prior MLflow run in this project used. Deliberately does
    NOT retrain the GNN, Deep Ensemble, or habit-blended MLP under the new
    split -- re-running the full comparison suite this way is legitimate
    future work, not required by this milestone (Step 3.4).

    CRITICAL (see ADR-011): this run's Brier Scores are NOT directly
    comparable to any pre-existing MLflow run (all of which used
    sample-level splitting) -- different samples land in the validation
    set entirely because of a METHODOLOGY change, not because anything
    about the model or features changed. This run is tagged
    `split_type="match_level"` explicitly so this is never ambiguous when
    reading MLflow later; every pre-existing run is implicitly
    `split_type="sample_level"` (not retroactively tagged -- see the
    module-level comment near RANDOM_SEED above and ADR-011).

    Returns `_train_and_log_model`'s result dict (or `None` if training
    aborted on a NaN/Inf loss).
    """
    torch.manual_seed(RANDOM_SEED)

    features, frames, chains, source_event_ids, match_ids, qualifying_competitions, sample_match_ids = (
        build_training_data()
    )
    dataset_size = len(features)
    competition_season_summary = ",".join(
        f"{c['competition_id']}:{c['season_id']}" for c in qualifying_competitions
    )
    logger.info(
        f"\n[Milestone 35 smoke test] {dataset_size} samples across {len(match_ids)} matches "
        "(same data-fetch path as train_and_evaluate(); only the split mechanism and which "
        "model gets trained differ)."
    )

    dataset = TacticalSurvivalDataset(features, frames, chains)

    train_indices, val_indices = match_level_split(
        sample_match_ids, val_fraction=1.0 - TRAIN_FRACTION, seed=RANDOM_SEED
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    n_train, n_val = len(train_set), len(val_set)
    logger.info(
        f"[Milestone 35 smoke test] resulting sample-level ratio: {n_train} train / {n_val} val "
        f"({n_train / (n_train + n_val):.1%} / {n_val / (n_train + n_val):.1%})"
    )

    # Scalar feature normalization: training-split-only, unchanged rule
    # (Milestone 7), computed from the NEW match-level training indices.
    train_features_raw = torch.stack([dataset[i][0] for i in train_set.indices])
    feature_mean = train_features_raw.mean(dim=0)
    feature_std = train_features_raw.std(dim=0).clamp(min=1e-8)

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
    mlp_optimizer = torch.optim.Adam(
        mlp_model.parameters(), lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY
    )
    results = _train_and_log_model(
        model_type="MLP",
        model=mlp_model,
        optimizer=mlp_optimizer,
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
            "init_seed": RANDOM_SEED,
            "split_type": "match_level",  # Milestone 35 / ADR-011
        },
        run_tags={
            "milestone": "35",
            "split_type": "match_level",
            "comparability_note": (
                "NOT directly comparable to Milestone 14B's sample-level MLP baseline "
                f"(run_id={MILESTONE_14B_MLP_RUN_ID}) -- different train/val split methodology, "
                "not a corrected or superseding result. See ADR-011."
            ),
        },
        normalization_artifact={
            "feature_key_order": list(FEATURE_KEYS),
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
    )

    logger.info("\n=== Milestone 35 (ADR-011): match-level-split MLP result ===")
    if results is None:
        logger.warning("MLP (match-level split): training ABORTED (NaN/Inf loss).")
    else:
        logger.info(
            "MLP (match-level split) -- INFORMATIONAL ONLY, NOT directly comparable to Milestone "
            f"14B's sample-level baseline (Brier@15s/30s = {MILESTONE_14B_MLP_BRIER_15S:.4f} / "
            f"{MILESTONE_14B_MLP_BRIER_30S:.4f}, run_id={MILESTONE_14B_MLP_RUN_ID}):"
        )
        logger.info(f"  Brier@15s = {results['brier_15s']:.4f}, Brier@30s = {results['brier_30s']:.4f}")
        logger.info(
            f"  instability warnings -- spike: {results['spike_fired']}, cumulative_drift: "
            f"{results['cumulative_drift_fired']}, saturation: {results['saturation_fired']}, "
            f"frozen_val_loss: {results['frozen_val_loss_fired']}"
        )
    return results


# Engineering-review follow-up (real CI failure, reproduced locally by moving
# mlruns/ aside and re-running the full suite): `mlruns/` is gitignored, same
# as `data/raw/` -- a completely fresh checkout (every GitHub Actions run,
# starting from nothing) has no trained model for the ~10 test files that
# load one via `explainer.load_deterministic_mlp()` / this module's own
# `DeepEnsemble_MLP`-tagged lookup (`test_uncertainty.py`) -- directly, or
# transitively through `team_report.generate_team_report()` or the live
# FastAPI app's `lifespan` startup handler. The reproduction found 25
# failures across 10 files, all the identical
# `RuntimeError: MLflow experiment 'project-athena-deephit' not found`.
CI_BOOTSTRAP_MATCH_IDS = [3857264, 3857289, 3857300, 3869151]  # Argentina, World Cup
# 2022 -- the SAME 4 real, 360-covered matches already used throughout
# production/tests/test_reporting.py, test_zone_explainer.py, and
# test_report_visualizer.py's own real-data validation. Reused verbatim here
# (not a new, separately-judged choice) specifically so this bootstrap's data
# provenance is already proven real, fetchable, and 360-covered.


def run_ci_bootstrap_training() -> None:
    """CI-only setup step -- NOT a research run, NOT a re-validation
    campaign, and NOT a replacement for a real contributor's own, much
    larger, locally-built `mlruns/` history (which is never touched or
    overwritten by this function outside of a fresh checkout that has no
    `mlruns/` at all).

    Trains ONE real MLP and ONE real (M=5) Deep Ensemble, tagged EXACTLY
    the way `select_deterministic_mlp_run_id()` / `test_uncertainty.py`'s
    own `DeepEnsemble_MLP` tag-filtered lookup expect, so those lookups
    (and everything that calls them) succeed on a fresh CI runner. This
    deliberately reuses this module's own real training code
    (`_train_and_log_model`, `_train_and_log_deep_ensemble`, the exact
    `MLP_STABILIZED_LR`/`MLP_STABILIZED_WEIGHT_DECAY` bundle every
    production run uses) against real StatsBomb data -- NOT a synthetic
    fixture, NOT a separately-maintained CI-only training recipe --
    specifically because several of the tests this unblocks
    (`test_oracle.py`'s real-substitution validation,
    `test_reporting.py`'s `attacking_third > defensive_third` threat-
    direction sanity check) assert genuine, football-plausible properties
    of the loaded model's real behavior; a model trained on synthetic data
    would make those tests pass without them meaning what they currently
    mean.

    Deliberately scoped to `CI_BOOTSTRAP_MATCH_IDS` (4 real matches, on
    the order of a few hundred samples) rather than `build_training_data`'s
    full 12-competition/~8,000-sample search -- the full search is correct
    for a real research run but would add real minutes of wall-clock and
    dozens of network fetches to every single push, which is not
    appropriate for a job meant to run on every push/PR. This is a
    genuinely fast, genuinely real bootstrap, not a scaled-down
    approximation pretending to be the full pipeline.
    """
    torch.manual_seed(RANDOM_SEED)

    engine = BiomechanicalPitchControl()
    all_features, all_frames, all_chains = [], [], []
    all_sample_match_ids: list[int] = []
    for match_id in CI_BOOTSTRAP_MATCH_IDS:
        features, frames, chains, _ = _match_chains_with_features(match_id, engine)
        all_features.extend(features)
        all_frames.extend(frames)
        all_chains.extend(chains)
        all_sample_match_ids.extend([match_id] * len(features))

    dataset_size = len(all_features)
    logger.info(
        f"\n[CI bootstrap] {dataset_size} samples across {len(CI_BOOTSTRAP_MATCH_IDS)} matches"
    )
    if dataset_size == 0:
        raise RuntimeError(
            "[CI bootstrap] Zero samples resolved from CI_BOOTSTRAP_MATCH_IDS -- cannot "
            "train. Check StatsBomb open-data connectivity, or whether these match IDs "
            "still resolve to real, 360-covered matches."
        )

    dataset = TacticalSurvivalDataset(all_features, all_frames, all_chains)

    train_indices, val_indices = match_level_split(
        all_sample_match_ids, val_fraction=1.0 - TRAIN_FRACTION, seed=RANDOM_SEED
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    n_train, n_val = len(train_set), len(val_set)
    logger.info(f"[CI bootstrap] {n_train} train / {n_val} val samples")

    train_features_raw = torch.stack([dataset[i][0] for i in train_set.indices])
    feature_mean = train_features_raw.mean(dim=0)
    feature_std = train_features_raw.std(dim=0).clamp(min=1e-8)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_batch = next(iter(DataLoader(val_set, batch_size=len(val_set))))

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    normalization_artifact = {
        "feature_key_order": list(FEATURE_KEYS),
        "mean": feature_mean.tolist(),
        "std": feature_std.tolist(),
    }
    run_tags = {
        "milestone": "ci_bootstrap",
        "split_type": "match_level",
        "comparability_note": (
            "CI-only bootstrap run on a deliberately small, fixed 4-match dataset -- NOT "
            "comparable to any research-scale run (Milestone 14B, 21, 23, 35). Exists only "
            "so tests that require a real, tagged MLflow run can load one on a fresh "
            "checkout. See run_ci_bootstrap_training()'s docstring in train.py."
        ),
    }

    mlp_model = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
    mlp_optimizer = torch.optim.Adam(
        mlp_model.parameters(), lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY
    )
    mlp_results = _train_and_log_model(
        model_type="MLP",
        model=mlp_model,
        optimizer=mlp_optimizer,
        lr=MLP_STABILIZED_LR,
        weight_decay=MLP_STABILIZED_WEIGHT_DECAY,
        clip_grad_norm=True,
        input_fn=_normalize_scalar_batch,
        normalize_args=(feature_mean, feature_std),
        train_loader=train_loader,
        val_batch=val_batch,
        n_train=n_train,
        n_val=n_val,
        match_ids=CI_BOOTSTRAP_MATCH_IDS,
        dataset_size=dataset_size,
        extra_params={
            "dataset_scale": "ci_bootstrap",
            "stabilization_bundle": True,
            "saturation_check_v2": True,
            "init_seed": RANDOM_SEED,
            "split_type": "match_level",
        },
        run_tags=run_tags,
        normalization_artifact=normalization_artifact,
    )
    if mlp_results is None:
        raise RuntimeError("[CI bootstrap] MLP training aborted (NaN/Inf loss) -- cannot proceed.")
    logger.info(
        f"[CI bootstrap] MLP done: brier_15s={mlp_results['brier_15s']:.4f}, "
        f"brier_30s={mlp_results['brier_30s']:.4f}"
    )

    # test_gnn_simulator.py's own deterministic-selection lookup filters on
    # `params.model_type = 'GNN'` alone (no stabilization_bundle/
    # saturation_check_v2 requirement, unlike the MLP/DeepEnsemble
    # selectors) -- still trained with the same stabilized hyperparameter
    # bundle for consistency with every other model this function trains.
    train_graph_continuous = torch.cat(
        [dataset[i][1].x[:, [0, 1, 6]] for i in train_set.indices], dim=0
    )
    graph_feature_mean = train_graph_continuous.mean(dim=0)
    graph_feature_std = train_graph_continuous.std(dim=0).clamp(min=1e-8)

    torch.manual_seed(RANDOM_SEED)
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
        match_ids=CI_BOOTSTRAP_MATCH_IDS,
        dataset_size=dataset_size,
        extra_params={
            "same_team_radius": DEFAULT_SAME_TEAM_RADIUS,
            "opponent_radius": DEFAULT_OPPONENT_RADIUS,
            "hidden_dim": GNN_HIDDEN_DIM,
            "dataset_scale": "ci_bootstrap",
            "stabilization_bundle": True,
            "saturation_check_v2": True,
            "init_seed": RANDOM_SEED,
            "split_type": "match_level",
        },
        run_tags=run_tags,
        normalization_artifact={
            "graph_continuous_feature_order": ["x", "y", "dist_to_ball"],
            "mean": graph_feature_mean.tolist(),
            "std": graph_feature_std.tolist(),
        },
    )
    if gnn_results is None:
        raise RuntimeError("[CI bootstrap] GNN training aborted (NaN/Inf loss) -- cannot proceed.")
    logger.info(
        f"[CI bootstrap] GNN done: brier_15s={gnn_results['brier_15s']:.4f}, "
        f"brier_30s={gnn_results['brier_30s']:.4f}"
    )

    torch.manual_seed(RANDOM_SEED)
    ensemble_model = DeepEnsembleDeepHit(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
    ensemble_optimizer = torch.optim.Adam(
        ensemble_model.parameters(), lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY
    )
    ensemble_results = _train_and_log_deep_ensemble(
        model=ensemble_model,
        optimizer=ensemble_optimizer,
        train_loader=train_loader,
        val_batch=val_batch,
        n_train=n_train,
        n_val=n_val,
        match_ids=CI_BOOTSTRAP_MATCH_IDS,
        dataset_size=dataset_size,
        normalization_mean=feature_mean,
        normalization_std=feature_std,
        normalization_artifact=normalization_artifact,
        run_tags=run_tags,
    )
    if ensemble_results is None:
        raise RuntimeError(
            "[CI bootstrap] Deep Ensemble training aborted (NaN/Inf loss) -- cannot proceed."
        )
    logger.info(
        f"[CI bootstrap] Deep Ensemble done: brier_15s={ensemble_results['brier_15s']:.4f}, "
        f"brier_30s={ensemble_results['brier_30s']:.4f}"
    )
    logger.info("[CI bootstrap] Complete -- mlruns/ now has a real, tagged MLP and Deep Ensemble run.")


# Genuine research re-run (NOT CI infrastructure -- do not confuse with
# run_ci_bootstrap_training() above, which trains on a tiny fixed 4-match
# slice purely so tests have a real MLflow run to load). ADR-011 (Milestone
# 35) switched _load_and_split_dataset() to match_level_split
# UNCONDITIONALLY, but explicitly deferred actually re-running the RQ2/RQ4
# comparisons under it: "this milestone deliberately does NOT re-run the
# full comparison suite under the new split... legitimate, separate future
# work, not performed here." This function is that future work, finally
# executed, at the SAME full multi-competition scale Milestone 14B used.
#
# Reference numbers below are Milestone 14B's SAMPLE-LEVEL-split results
# (RESEARCH_FINDINGS.md's RQ4 table / Milestone 23's RQ2 section) -- kept
# here as literal constants (not previously named in this file for the
# GNN/habit-blended cases) purely as REFERENCE POINTS for this run's
# comparison tables. They are NOT overwritten, recomputed, or treated as
# invalidated by this run -- see this function's own docstring.
MILESTONE_14B_GNN_RUN_ID = "b8565e5b4b2c4512b998bccbb39d64db"
MILESTONE_14B_GNN_BRIER_15S = 0.1141
MILESTONE_14B_GNN_BRIER_30S = 0.1932
MILESTONE_23_HABIT_MLP_RUN_ID = "671fb22ed0334f34a6a74059d1a17a4e"
MILESTONE_23_HABIT_MLP_BRIER_15S = 0.0950
MILESTONE_23_HABIT_MLP_BRIER_30S = 0.1601
MILESTONE_23_TRAINING_MATCH_COUNT = 4
MILESTONE_23_COLD_START_FALLBACK_RATE = 0.68  # 5,495 of 8,070 samples with a known actor


def run_match_level_rq2_rq4_full_revalidation() -> dict:
    """RQ2 (habit-blended MLP) and RQ4 (GNN vs. MLP) re-validation under
    Milestone 35/ADR-011's match-level split, at FULL research scale (the
    same ~55-match, 12-competition dataset Milestone 14B/23 used) -- not
    the single MLP-only smoke test Milestone 35 itself ran, and not the
    tiny CI bootstrap.

    Composes ALREADY-VALIDATED sub-functions in the SAME sequence
    `train_and_evaluate()` uses for its own MLP/GNN/habit-blended stages --
    none of `_load_and_split_dataset`, `_compute_normalization_stats`,
    `_run_mlp_stabilization_and_robustness_check`, `_run_gnn_stage`,
    `_evaluate_mlp_health`, `_build_habit_blended_features` (called inside
    `_run_habit_blended_stage`), or `_run_habit_blended_stage` itself is
    modified here. The one real, deliberate difference from
    `train_and_evaluate()`: THIS function gates Step 3 (the habit-blended
    re-run) behind BOTH the MLP's `_evaluate_mlp_health` result AND the
    GNN's own four-signal detector -- `train_and_evaluate()` runs its Deep
    Ensemble and habit-blended stages unconditionally and only evaluates
    health in its final summary, which is the wrong order for a task that
    explicitly must not build a habit-memory re-run on an unconfirmed
    foundation. Also deliberately skips the Deep Ensemble stage entirely
    (not requested here, and `_run_habit_blended_stage`'s
    `deep_ensemble_results` parameter is only used to print one extra
    comparison row -- `None` is a safe, real, supported input, not a
    workaround).

    Returns a dict of every real result (never silently dropped, even on
    a failed health gate) for the caller to build its own final report
    from -- this function prints its own tables/conclusions to the log as
    it goes, but the caller is expected to do its own synthesis on top of
    the returned dict, not parse log output.
    """
    torch.manual_seed(RANDOM_SEED)

    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: full-scale dataset + match-level split (ADR-011)")
    logger.info("=" * 80)
    loaded = _load_and_split_dataset()
    dataset = loaded["dataset"]
    train_set = loaded["train_set"]
    val_set = loaded["val_set"]
    n_train = loaded["n_train"]
    n_val = loaded["n_val"]
    match_ids = loaded["match_ids"]
    sample_match_ids = loaded["sample_match_ids"]
    dataset_size = loaded["dataset_size"]
    competition_season_summary = loaded["competition_season_summary"]

    # Early, Step-1-scoped corpus-size report -- a plain set computation,
    # the SAME one _build_habit_blended_features performs internally
    # (Step 3 re-confirms it from the real habit-diagnostics dict below;
    # this is not a second, independently-fallible computation, just an
    # earlier look at the same partition already fixed by the split above).
    val_match_ids_preview = {sample_match_ids[i] for i in val_set.indices}
    training_match_ids_preview = sorted(set(match_ids) - val_match_ids_preview)
    logger.info(
        f"\n[Step 1] {len(match_ids)} total matches, {dataset_size} total samples. Match-level "
        f"split: {len(training_match_ids_preview)} training-bucket-eligible matches / "
        f"{len(val_match_ids_preview)} validation matches (Milestone 23's original sample-level "
        f"corpus, for reference: {MILESTONE_23_TRAINING_MATCH_COUNT} matches)."
    )
    logger.info(
        f"[Step 1] Sample-level split from the SAME match partition: {n_train} train / {n_val} val "
        f"({n_train / dataset_size:.1%} / {n_val / dataset_size:.1%} of {dataset_size} total -- "
        "NOT forced to a round percentage, see match_level_split's own report above)."
    )

    feature_mean, feature_std, graph_feature_mean, graph_feature_std = _compute_normalization_stats(
        dataset, train_set
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_batch = next(iter(DataLoader(val_set, batch_size=len(val_set))))

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: retrain MLP (no habit blending) + GNN, matched Milestone 14B hyperparameters")
    logger.info("=" * 80)
    mlp_seed_results = _run_mlp_stabilization_and_robustness_check(
        train_loader, val_batch, n_train, n_val, match_ids, dataset_size,
        competition_season_summary, feature_mean, feature_std,
    )
    gnn_results = _run_gnn_stage(
        train_loader, val_batch, n_train, n_val, match_ids, dataset_size,
        competition_season_summary, graph_feature_mean, graph_feature_std,
    )

    # Reuse train_and_evaluate()'s own summary/RQ4-narrative function
    # unmodified -- gives the standard warning breakdown and its own
    # four-way (Milestone 14 / 14B) table for free, in addition to this
    # function's own explicit, split_type-labeled table below.
    _report_run_summary_and_rq4_conclusion(mlp_seed_results, gnn_results, dataset_size, n_train, n_val)

    primary_mlp_results = mlp_seed_results[RANDOM_SEED]
    mlp_healthy = _evaluate_mlp_health(primary_mlp_results)
    gnn_healthy = gnn_results is not None and not gnn_results["instability_warning_fired"]

    logger.info("\n" + "=" * 80)
    logger.info("HEALTH GATE (must pass before Step 3 -- the RQ2 habit-memory re-run)")
    logger.info("=" * 80)
    if primary_mlp_results is not None:
        logger.info(
            f"MLP (seed={RANDOM_SEED}) four-signal detector: spike={primary_mlp_results['spike_fired']}, "
            f"cumulative_drift={primary_mlp_results['cumulative_drift_fired']}, "
            f"saturation={primary_mlp_results['saturation_fired']} (entropy/variance-based -- "
            f"ADR-010's primary trusted signal for ambiguous cases), "
            f"frozen_val_loss={primary_mlp_results['frozen_val_loss_fired']} -> mlp_healthy={mlp_healthy}"
        )
    else:
        logger.warning(f"MLP (seed={RANDOM_SEED}): training ABORTED (NaN/Inf loss) -> mlp_healthy=False")
    if gnn_results is not None:
        logger.info(
            f"GNN four-signal detector: spike={gnn_results['spike_fired']}, "
            f"cumulative_drift={gnn_results['cumulative_drift_fired']}, "
            f"saturation={gnn_results['saturation_fired']} (entropy/variance-based), "
            f"frozen_val_loss={gnn_results['frozen_val_loss_fired']} -> gnn_healthy={gnn_healthy}"
        )
    else:
        logger.warning("GNN: training ABORTED (NaN/Inf loss) -> gnn_healthy=False")

    result = {
        "dataset_size": dataset_size,
        "n_train": n_train,
        "n_val": n_val,
        "match_count": len(match_ids),
        "training_match_count_preview": len(training_match_ids_preview),
        "val_match_count_preview": len(val_match_ids_preview),
        "mlp_seed_results": mlp_seed_results,
        "gnn_results": gnn_results,
        "mlp_healthy": mlp_healthy,
        "gnn_healthy": gnn_healthy,
        "gate_passed": mlp_healthy and gnn_healthy,
        "habit_mlp_results": None,
        "habit_diagnostics": None,
    }

    logger.info(
        "\n" + "=" * 80 + "\nSTEP 2 THREE-WAY COMPARISON -- this run (match_level) vs. Milestone "
        "14B (sample_level, different methodology, NOT a directly equivalent baseline)\n" + "=" * 80
    )
    logger.info(f"{'Model':<50} {'split_type':>13} {'Brier@15s':>10} {'Brier@30s':>10}")
    logger.info(
        f"{'MLP (Milestone 14B reference)':<50} {'sample_level':>13} "
        f"{MILESTONE_14B_MLP_BRIER_15S:>10.4f} {MILESTONE_14B_MLP_BRIER_30S:>10.4f}"
    )
    logger.info(
        f"{'GNN (Milestone 14B reference)':<50} {'sample_level':>13} "
        f"{MILESTONE_14B_GNN_BRIER_15S:>10.4f} {MILESTONE_14B_GNN_BRIER_30S:>10.4f}"
    )
    if primary_mlp_results is not None:
        logger.info(
            f"{'MLP (this run, NEW)':<50} {'match_level':>13} "
            f"{primary_mlp_results['brier_15s']:>10.4f} {primary_mlp_results['brier_30s']:>10.4f}"
        )
    else:
        logger.info(f"{'MLP (this run, NEW)':<50} {'match_level':>13} {'ABORTED':>10} {'ABORTED':>10}")
    if gnn_results is not None:
        logger.info(
            f"{'GNN (this run, NEW)':<50} {'match_level':>13} "
            f"{gnn_results['brier_15s']:>10.4f} {gnn_results['brier_30s']:>10.4f}"
        )
    else:
        logger.info(f"{'GNN (this run, NEW)':<50} {'match_level':>13} {'ABORTED':>10} {'ABORTED':>10}")

    if not result["gate_passed"]:
        logger.warning(
            "\nHEALTH GATE FAILED -- NOT proceeding to Step 3 (habit-memory RQ2 re-run). "
            f"mlp_healthy={mlp_healthy}, gnn_healthy={gnn_healthy}. Per this task's explicit "
            "instruction, stopping here rather than building a habit-memory re-run on top of an "
            "unconfirmed MLP/GNN foundation."
        )
        return result

    logger.info("\nHealth gate PASSED (mlp_healthy=True, gnn_healthy=True) -- proceeding to Step 3.")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: habit-blended MLP re-run (RQ2) under match-level split, enlarged corpus")
    logger.info("=" * 80)
    habit_mlp_results, habit_diagnostics = _run_habit_blended_stage(
        loaded["frames"], loaded["chains"], sample_match_ids, match_ids,
        train_set, val_set, n_train, n_val, dataset_size, competition_season_summary,
        None,  # deep_ensemble_results -- this re-run deliberately does not train a Deep Ensemble
    )
    result["habit_mlp_results"] = habit_mlp_results
    result["habit_diagnostics"] = habit_diagnostics

    if habit_diagnostics is not None:
        known_actor = habit_diagnostics["samples_with_known_actor"]
        cold_start_rate = habit_diagnostics["cold_start_count"] / known_actor if known_actor > 0 else float("nan")
        logger.info(
            f"\n[Step 3] Training-bucket corpus: {habit_diagnostics['training_match_count']} matches "
            f"(Milestone 23's original sample-level corpus: {MILESTONE_23_TRAINING_MATCH_COUNT} matches)."
        )
        logger.info(
            f"[Step 3] Cold-start fallback rate: {habit_diagnostics['cold_start_count']}/{known_actor} "
            f"samples with a known actor ({cold_start_rate:.1%}) (Milestone 23's original rate: "
            f"{MILESTONE_23_COLD_START_FALLBACK_RATE:.0%})."
        )

    logger.info(
        "\n" + "=" * 80 + "\nSTEP 3 COMPARISON -- habit-blended MLP (this run) vs. this run's own "
        "non-blended baseline (apples-to-apples) vs. Milestone 23 (sample_level, different "
        "methodology)\n" + "=" * 80
    )
    logger.info(f"{'Model':<50} {'split_type':>13} {'corpus':>7} {'Brier@15s':>10} {'Brier@30s':>10}")
    if primary_mlp_results is not None:
        logger.info(
            f"{'MLP, no blending (this run, same split)':<50} {'match_level':>13} {'n/a':>7} "
            f"{primary_mlp_results['brier_15s']:>10.4f} {primary_mlp_results['brier_30s']:>10.4f}"
        )
    if habit_mlp_results is not None:
        logger.info(
            f"{'MLP, habit blending (this run, NEW)':<50} {'match_level':>13} "
            f"{habit_diagnostics['training_match_count']:>7} "
            f"{habit_mlp_results['brier_15s']:>10.4f} {habit_mlp_results['brier_30s']:>10.4f}"
        )
    else:
        logger.info(f"{'MLP, habit blending (this run, NEW)':<50} {'match_level':>13} {'--':>7} {'ABORTED':>10} {'ABORTED':>10}")
    logger.info(
        f"{'MLP, habit blending (Milestone 23 reference)':<50} {'sample_level':>13} "
        f"{MILESTONE_23_TRAINING_MATCH_COUNT:>7} "
        f"{MILESTONE_23_HABIT_MLP_BRIER_15S:>10.4f} {MILESTONE_23_HABIT_MLP_BRIER_30S:>10.4f}"
    )

    return result


# Repeated-measurement investigation: is the GNN's disproportionately larger
# 30s-horizon degradation (relative to the MLP) under match-level splitting
# (see RESEARCH_FINDINGS.md's RQ4 "Stage 3" update: MLP 0.1009/0.1873 vs.
# GNN 0.1198/0.2437, a 0.0564 gap @30s) a real, repeatable property, or a
# single-run artifact? Investigates via (1) model-init-seed variation at the
# SAME match-level split, (2) split-seed variation (genuinely different
# match partitions), and (3) per-time-bin Brier decomposition for
# confirmed-gap runs. Deliberately REUSES the three already-existing
# match_level, split_seed=42 runs from run_match_level_rq2_rq4_full_revalidation()
# (MLP init_seed=42 `b77fdf76b79b4fc3a19035914a098091`, MLP init_seed=43
# `50fe80da239e4213bba5909cd72cdc5c`, GNN init_seed=42
# `c5fecf26a1d343c38cbcbdbeb8ebd73d`) rather than retraining what is already
# on record -- only genuinely missing (model, init_seed, split_seed)
# combinations are trained fresh. No model architecture, split function, or
# hyperparameter is modified anywhere in this section.
GNN_HORIZON_INVESTIGATION_TAG = "gnn_horizon_degradation_check"
EXISTING_MATCH_LEVEL_SPLIT42_RUNS = {
    ("MLP", 42, 42): "b77fdf76b79b4fc3a19035914a098091",
    ("MLP", 43, 42): "50fe80da239e4213bba5909cd72cdc5c",
    ("GNN", 42, 42): "c5fecf26a1d343c38cbcbdbeb8ebd73d",
}


def _train_at_seed_and_split(model_type: str, init_seed: int, split_seed: int, loaded_full: dict) -> dict:
    """Trains ONE MLP or GNN at an explicit (init_seed, split_seed) pair,
    reusing `_train_and_log_model` unmodified -- the same shared loop every
    other stage in this file uses, just called directly here instead of via
    `_run_mlp_stabilization_and_robustness_check`/`_run_gnn_stage` (both of
    which hardcode split_seed=RANDOM_SEED and, for the GNN, hardcode
    init_seed=RANDOM_SEED too, with no parameter to vary either).

    `loaded_full` is `_load_and_split_dataset()`'s return dict (or an
    equivalent one built the same way) -- only its SPLIT-INDEPENDENT fields
    (features/frames/chains/sample_match_ids/match_ids/dataset_size/
    competition_season_summary) are used; the split itself is redone here
    with the requested `split_seed`, since Step 2 of this investigation
    needs genuinely different match partitions from the one
    `_load_and_split_dataset()` itself already computed.

    Returns a dict with the real `_train_and_log_model` results (or None if
    aborted) plus everything a later per-bin re-analysis needs to
    reconstruct the exact same split/normalization deterministically.
    """
    features = loaded_full["features"]
    frames = loaded_full["frames"]
    chains = loaded_full["chains"]
    sample_match_ids = loaded_full["sample_match_ids"]
    match_ids = loaded_full["match_ids"]
    dataset_size = loaded_full["dataset_size"]
    competition_season_summary = loaded_full["competition_season_summary"]

    dataset = TacticalSurvivalDataset(features, frames, chains)
    train_indices, val_indices = match_level_split(
        sample_match_ids, val_fraction=1.0 - TRAIN_FRACTION, seed=split_seed
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    n_train, n_val = len(train_set), len(val_set)

    feature_mean, feature_std, graph_feature_mean, graph_feature_std = _compute_normalization_stats(
        dataset, train_set
    )

    torch.manual_seed(init_seed)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(init_seed),
    )
    val_batch = next(iter(DataLoader(val_set, batch_size=len(val_set))))

    run_tags = {
        "investigation": GNN_HORIZON_INVESTIGATION_TAG,
        "init_seed": str(init_seed),
        "split_seed": str(split_seed),
    }
    shared_extra_params = {
        "dataset_scale": "multi_competition",
        "competition_season_pairs": competition_season_summary,
        "stabilization_bundle": True,
        "saturation_check_v2": True,
        "init_seed": init_seed,
        "split_type": "match_level",
        "split_seed": split_seed,
    }

    torch.manual_seed(init_seed)
    if model_type == "MLP":
        model = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
        optimizer = torch.optim.Adam(model.parameters(), lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY)
        results = _train_and_log_model(
            model_type="MLP", model=model, optimizer=optimizer,
            lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY, clip_grad_norm=True,
            input_fn=_normalize_scalar_batch, normalize_args=(feature_mean, feature_std),
            train_loader=train_loader, val_batch=val_batch, n_train=n_train, n_val=n_val,
            match_ids=match_ids, dataset_size=dataset_size,
            extra_params=shared_extra_params, run_tags=run_tags,
            normalization_artifact={
                "feature_key_order": list(FEATURE_KEYS),
                "mean": feature_mean.tolist(), "std": feature_std.tolist(),
            },
        )
    else:
        model = GNNDeepHitSurvivalModel(num_node_features=7, num_bins=NUM_BINS, hidden_dim=GNN_HIDDEN_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY)
        results = _train_and_log_model(
            model_type="GNN", model=model, optimizer=optimizer,
            lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY, clip_grad_norm=True,
            input_fn=_normalize_graph_batch, normalize_args=(graph_feature_mean, graph_feature_std),
            train_loader=train_loader, val_batch=val_batch, n_train=n_train, n_val=n_val,
            match_ids=match_ids, dataset_size=dataset_size,
            extra_params={
                **shared_extra_params,
                "same_team_radius": DEFAULT_SAME_TEAM_RADIUS,
                "opponent_radius": DEFAULT_OPPONENT_RADIUS,
                "hidden_dim": GNN_HIDDEN_DIM,
            },
            run_tags=run_tags,
            normalization_artifact={
                "graph_continuous_feature_order": ["x", "y", "dist_to_ball"],
                "mean": graph_feature_mean.tolist(), "std": graph_feature_std.tolist(),
            },
        )

    return {
        "model_type": model_type,
        "init_seed": init_seed,
        "split_seed": split_seed,
        "results": results,
        "run_id": results["run_id"] if results is not None else None,
        "healthy": (
            _evaluate_mlp_health(results) if model_type == "MLP"
            else (results is not None and not results["instability_warning_fired"])
        ),
    }


def run_gnn_horizon_seed_split_sensitivity_check() -> dict:
    """Steps 1+2: does the MLP/GNN 30s-horizon gap hold across model-
    initialization seeds (same match-level split) and across genuinely
    different split seeds (different match partitions)? Reuses the three
    already-existing split_seed=42 runs (`EXISTING_MATCH_LEVEL_SPLIT42_RUNS`)
    directly from MLflow rather than retraining them; trains only the
    genuinely missing (model, init_seed, split_seed) combinations.
    """
    torch.manual_seed(RANDOM_SEED)
    loaded_full = _load_and_split_dataset()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    combinations = [
        # Step 1: same split (42), vary init seed.
        ("MLP", 42, 42), ("MLP", 43, 42), ("MLP", 44, 42),
        ("GNN", 42, 42), ("GNN", 43, 42), ("GNN", 44, 42),
        # Step 2: same init seed (42), vary split seed.
        ("MLP", 42, 43), ("GNN", 42, 43),
        ("MLP", 42, 44), ("GNN", 42, 44),
    ]

    all_results: dict[tuple[str, int, int], dict] = {}
    for model_type, init_seed, split_seed in combinations:
        existing_run_id = EXISTING_MATCH_LEVEL_SPLIT42_RUNS.get((model_type, init_seed, split_seed))
        if existing_run_id is not None:
            run = client.get_run(existing_run_id)
            healthy = not (run.data.params.get("instability_warning_fired") == "True")
            if model_type == "MLP":
                # Reconstruct the same two-criterion health check
                # _evaluate_mlp_health applies, from the already-logged
                # metrics -- this run was never re-evaluated by that
                # function directly since it predates this investigation.
                brier_in_range = (
                    run.data.metrics.get("val_brier_15s", 1e9) <= MLP_SANITY_BRIER_15S_CEILING
                    and run.data.metrics.get("val_brier_30s", 1e9) <= MLP_SANITY_BRIER_30S_CEILING
                )
                healthy = healthy and brier_in_range
            logger.info(
                f"\n[{model_type} init_seed={init_seed} split_seed={split_seed}] REUSING existing "
                f"run_id={existing_run_id} (not retrained)."
            )
            all_results[(model_type, init_seed, split_seed)] = {
                "model_type": model_type, "init_seed": init_seed, "split_seed": split_seed,
                "run_id": existing_run_id,
                "brier_15s": run.data.metrics.get("val_brier_15s"),
                "brier_30s": run.data.metrics.get("val_brier_30s"),
                "healthy": healthy,
                "reused": True,
            }
            continue

        logger.info(f"\n[{model_type} init_seed={init_seed} split_seed={split_seed}] training fresh...")
        outcome = _train_at_seed_and_split(model_type, init_seed, split_seed, loaded_full)
        r = outcome["results"]
        all_results[(model_type, init_seed, split_seed)] = {
            "model_type": model_type, "init_seed": init_seed, "split_seed": split_seed,
            "run_id": outcome["run_id"],
            "brier_15s": r["brier_15s"] if r is not None else None,
            "brier_30s": r["brier_30s"] if r is not None else None,
            "healthy": outcome["healthy"],
            "reused": False,
        }

    logger.info("\n" + "=" * 90)
    logger.info("STEP 1: init-seed sensitivity (split_seed=42 fixed) -- 30s-horizon gap per seed")
    logger.info("=" * 90)
    logger.info(f"{'init_seed':>10} {'MLP Brier@15s':>15} {'MLP Brier@30s':>15} {'GNN Brier@15s':>15} {'GNN Brier@30s':>15} {'30s gap (GNN-MLP)':>20} {'both healthy':>13}")
    for seed in (42, 43, 44):
        mlp = all_results[("MLP", seed, 42)]
        gnn = all_results[("GNN", seed, 42)]
        both_healthy = mlp["healthy"] and gnn["healthy"]
        gap_30s = (
            gnn["brier_30s"] - mlp["brier_30s"]
            if mlp["brier_30s"] is not None and gnn["brier_30s"] is not None else None
        )
        logger.info(
            f"{seed:>10} {mlp['brier_15s'] if mlp['brier_15s'] is not None else float('nan'):>15.4f} "
            f"{mlp['brier_30s'] if mlp['brier_30s'] is not None else float('nan'):>15.4f} "
            f"{gnn['brier_15s'] if gnn['brier_15s'] is not None else float('nan'):>15.4f} "
            f"{gnn['brier_30s'] if gnn['brier_30s'] is not None else float('nan'):>15.4f} "
            f"{gap_30s if gap_30s is not None else float('nan'):>20.4f} {both_healthy!s:>13}"
        )
        if not both_healthy:
            logger.warning(
                f"  init_seed={seed}: NOT both healthy (MLP healthy={mlp['healthy']}, "
                f"GNN healthy={gnn['healthy']}) -- excluded from any conclusion, per this "
                "investigation's own health-gate requirement."
            )

    logger.info("\n" + "=" * 90)
    logger.info("STEP 2: split-seed sensitivity (init_seed=42 fixed) -- 30s-horizon gap per split")
    logger.info("=" * 90)
    logger.info(f"{'split_seed':>10} {'MLP Brier@15s':>15} {'MLP Brier@30s':>15} {'GNN Brier@15s':>15} {'GNN Brier@30s':>15} {'30s gap (GNN-MLP)':>20} {'both healthy':>13}")
    for split_seed in (42, 43, 44):
        mlp = all_results[("MLP", 42, split_seed)]
        gnn = all_results[("GNN", 42, split_seed)]
        both_healthy = mlp["healthy"] and gnn["healthy"]
        gap_30s = (
            gnn["brier_30s"] - mlp["brier_30s"]
            if mlp["brier_30s"] is not None and gnn["brier_30s"] is not None else None
        )
        logger.info(
            f"{split_seed:>10} {mlp['brier_15s'] if mlp['brier_15s'] is not None else float('nan'):>15.4f} "
            f"{mlp['brier_30s'] if mlp['brier_30s'] is not None else float('nan'):>15.4f} "
            f"{gnn['brier_15s'] if gnn['brier_15s'] is not None else float('nan'):>15.4f} "
            f"{gnn['brier_30s'] if gnn['brier_30s'] is not None else float('nan'):>15.4f} "
            f"{gap_30s if gap_30s is not None else float('nan'):>20.4f} {both_healthy!s:>13}"
        )
        if not both_healthy:
            logger.warning(
                f"  split_seed={split_seed}: NOT both healthy (MLP healthy={mlp['healthy']}, "
                f"GNN healthy={gnn['healthy']}) -- excluded from any conclusion."
            )

    return all_results


def run_gnn_horizon_per_bin_investigation(seed_split_pairs: list[tuple[int, int]]) -> dict:
    """Step 3: for each (init_seed, split_seed) pair, retrains a fresh
    MLP+GNN (same hyperparameters as every other stage in this file,
    reusing `_train_and_log_model` unmodified -- both still logged to
    MLflow for the record) and computes Brier Score at EVERY time bin
    (0-11, i.e. 0-55s) on BOTH the training and validation sets --
    not just the two headline horizons (15s/30s) -- IMMEDIATELY, using
    the LIVE in-memory model, never a reloaded one.

    This deliberately does NOT reload already-trained models from MLflow
    for this analysis (an earlier version of this function did, and was
    found to give WRONG GNN predictions on reload for the full-size
    validation batch -- confirmed via extensive isolation testing to be a
    real, reproducible MLflow/PyTorch save-load discrepancy specific to
    this GNN architecture at this batch scale, not a data/split/
    normalization reconstruction error; every one of those was separately
    verified byte-identical to the original. Retraining fresh and
    evaluating in-process sidesteps the reload path entirely, so this
    function's own numbers are trustworthy regardless of that discrepancy's
    exact root cause, which remains unresolved and is reported honestly in
    this investigation's final conclusion rather than papered over.
    """
    torch.manual_seed(RANDOM_SEED)
    loaded_full = _load_and_split_dataset()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_bin_results = {}

    for init_seed, split_seed in seed_split_pairs:
        logger.info("\n" + "=" * 90)
        logger.info(f"STEP 3: per-bin Brier decomposition -- init_seed={init_seed}, split_seed={split_seed}")
        logger.info("=" * 90)

        dataset = TacticalSurvivalDataset(loaded_full["features"], loaded_full["frames"], loaded_full["chains"])
        train_indices, val_indices = match_level_split(
            loaded_full["sample_match_ids"], val_fraction=1.0 - TRAIN_FRACTION, seed=split_seed
        )
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        feature_mean, feature_std, graph_feature_mean, graph_feature_std = _compute_normalization_stats(
            dataset, train_set
        )

        torch.manual_seed(init_seed)
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(init_seed),
        )
        # FRESH batches, drawn once and never re-normalized -- each of
        # _normalize_scalar_batch/_normalize_graph_batch is called exactly
        # once per (model, split) below.
        train_full_batch_mlp = next(iter(DataLoader(train_set, batch_size=len(train_set))))
        val_full_batch_mlp = next(iter(DataLoader(val_set, batch_size=len(val_set))))
        train_full_batch_gnn = next(iter(DataLoader(train_set, batch_size=len(train_set))))
        val_full_batch_gnn = next(iter(DataLoader(val_set, batch_size=len(val_set))))

        loss_fn = DeepHitLoss()

        torch.manual_seed(init_seed)
        mlp_model = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
        mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY)
        for _epoch in range(NUM_EPOCHS):
            mlp_model.train()
            for scalar_b, graph_b, dur_b, ev_b in train_loader:
                mlp_in = _normalize_scalar_batch(scalar_b, graph_b, feature_mean, feature_std)
                mlp_optimizer.zero_grad()
                pred = mlp_model(mlp_in)
                loss = loss_fn(pred, dur_b, ev_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                mlp_optimizer.step()
        mlp_model.eval()

        torch.manual_seed(init_seed)
        gnn_model = GNNDeepHitSurvivalModel(num_node_features=7, num_bins=NUM_BINS, hidden_dim=GNN_HIDDEN_DIM)
        gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY)
        # Re-seed train_loader's own shuffle generator identically so the
        # GNN sees the SAME epoch-by-epoch batch order the MLP saw --
        # matches _train_and_log_model's own convention (one shared
        # train_loader object, reused across model stages).
        for _epoch in range(NUM_EPOCHS):
            gnn_model.train()
            for scalar_b, graph_b, dur_b, ev_b in train_loader:
                gnn_in = _normalize_graph_batch(scalar_b, graph_b, graph_feature_mean, graph_feature_std)
                gnn_optimizer.zero_grad()
                pred = gnn_model(gnn_in)
                loss = loss_fn(pred, dur_b, ev_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                gnn_optimizer.step()
        gnn_model.eval()

        bin_rows = []
        with torch.no_grad():
            for split_name, (mlp_batch, gnn_batch) in (
                ("train", (train_full_batch_mlp, train_full_batch_gnn)),
                ("val", (val_full_batch_mlp, val_full_batch_gnn)),
            ):
                mlp_scalar_b, mlp_graph_b, mlp_dur_b, mlp_ev_b = mlp_batch
                gnn_scalar_b, gnn_graph_b, gnn_dur_b, gnn_ev_b = gnn_batch
                mlp_input = _normalize_scalar_batch(mlp_scalar_b, mlp_graph_b, feature_mean, feature_std)
                gnn_input = _normalize_graph_batch(gnn_scalar_b, gnn_graph_b, graph_feature_mean, graph_feature_std)
                mlp_pred = mlp_model(mlp_input)
                gnn_pred = gnn_model(gnn_input)

                for time_bin in range(NUM_BINS):
                    mlp_brier, _ = calculate_brier_score(mlp_pred, mlp_dur_b, mlp_dur_b, mlp_ev_b, time_bin)
                    gnn_brier, _ = calculate_brier_score(gnn_pred, gnn_dur_b, gnn_dur_b, gnn_ev_b, time_bin)
                    bin_rows.append({
                        "split": split_name, "time_bin": time_bin, "seconds": time_bin * BIN_SIZE_SECONDS,
                        "mlp_brier": mlp_brier, "gnn_brier": gnn_brier, "gap": gnn_brier - mlp_brier,
                    })

        # Log both models to MLflow too, for the record -- not relied on
        # for this function's own analysis (computed above, in-process).
        with mlflow.start_run(run_name="mlp_run"):
            mlflow.set_tags({"investigation": GNN_HORIZON_INVESTIGATION_TAG + "_per_bin", "init_seed": str(init_seed), "split_seed": str(split_seed)})
            mlflow.log_params({"model_type": "MLP", "init_seed": init_seed, "split_seed": split_seed})
            mlp_val_brier_15s, _ = calculate_brier_score(mlp_pred, mlp_dur_b, mlp_dur_b, mlp_ev_b, 3)
            mlp_val_brier_30s, _ = calculate_brier_score(mlp_pred, mlp_dur_b, mlp_dur_b, mlp_ev_b, 6)
            mlflow.log_metrics({"val_brier_15s": mlp_val_brier_15s, "val_brier_30s": mlp_val_brier_30s})
        with mlflow.start_run(run_name="gnn_run"):
            mlflow.set_tags({"investigation": GNN_HORIZON_INVESTIGATION_TAG + "_per_bin", "init_seed": str(init_seed), "split_seed": str(split_seed)})
            mlflow.log_params({"model_type": "GNN", "init_seed": init_seed, "split_seed": split_seed})
            gnn_val_brier_15s, _ = calculate_brier_score(gnn_pred, gnn_dur_b, gnn_dur_b, gnn_ev_b, 3)
            gnn_val_brier_30s, _ = calculate_brier_score(gnn_pred, gnn_dur_b, gnn_dur_b, gnn_ev_b, 6)
            mlflow.log_metrics({"val_brier_15s": gnn_val_brier_15s, "val_brier_30s": gnn_val_brier_30s})

        logger.info(f"{'bin':>4} {'sec':>5} {'MLP train':>10} {'MLP val':>10} {'MLP gap':>9} {'GNN train':>10} {'GNN val':>10} {'GNN gap':>9} {'val gap (GNN-MLP)':>18}")
        train_by_bin = {r["time_bin"]: r for r in bin_rows if r["split"] == "train"}
        val_by_bin = {r["time_bin"]: r for r in bin_rows if r["split"] == "val"}
        for time_bin in range(NUM_BINS):
            t, v = train_by_bin[time_bin], val_by_bin[time_bin]
            mlp_train_val_gap = v["mlp_brier"] - t["mlp_brier"]
            gnn_train_val_gap = v["gnn_brier"] - t["gnn_brier"]
            val_gap_gnn_minus_mlp = v["gnn_brier"] - v["mlp_brier"]
            logger.info(
                f"{time_bin:>4} {time_bin * BIN_SIZE_SECONDS:>5.0f} {t['mlp_brier']:>10.4f} {v['mlp_brier']:>10.4f} "
                f"{mlp_train_val_gap:>9.4f} {t['gnn_brier']:>10.4f} {v['gnn_brier']:>10.4f} {gnn_train_val_gap:>9.4f} "
                f"{val_gap_gnn_minus_mlp:>18.4f}"
            )

        all_bin_results[(init_seed, split_seed)] = bin_rows

    return all_bin_results


def run_rq1_non_physics_baseline_ablation() -> dict:
    """RQ1 ablation: the missing non-physics-baseline comparison.

    RQ1's stated success criterion (README: "Brier Score improvement >= X%")
    implies a comparison against a non-physics-informed baseline, but per
    RESEARCH_FINDINGS.md's RQ1 "Caveats" section, no such ablation (DeepHit
    trained on non-physics-derived features) had ever been run in this
    project's history. This function builds and trains that missing side of
    the comparison.

    Additive only -- does NOT modify feature_extractor.py,
    BiomechanicalPitchControl, or any existing model architecture. Reuses
    `_load_and_split_dataset`, `_compute_normalization_stats`,
    `_train_and_log_model`, and `_evaluate_mlp_health` completely
    unmodified -- the SAME already-validated sub-functions
    `run_match_level_rq2_rq4_full_revalidation` uses for its own match-level
    MLP run -- so this ablation goes through the identical ADR-011
    match-level-split and ADR-010 four-signal health-gate machinery every
    other current-generation result in this project used. The non-physics
    feature values themselves come from
    `naive_baseline_features.extract_naive_baseline_features` (a new,
    separate module -- see its docstring for exactly what raw signal it
    uses and why).

    Isolates exactly one variable: `_load_and_split_dataset()` is called
    once and its `frames`/`chains`/`match_ids`/`sample_match_ids` (and
    therefore its match-level train/val split) are reused BYTE-FOR-BYTE --
    same matches, same chains, same split -- only the scalar feature VALUES
    fed to the model are swapped from the physics-derived ones to the
    non-physics baseline's. Model architecture (DeepHitSurvivalModel),
    hyperparameters (MLP_STABILIZED_LR/WEIGHT_DECAY, gradient clipping,
    NUM_EPOCHS), and the training/health-gate/logging loop
    (`_train_and_log_model`) are identical to the current match-level
    physics-informed MLP reference.

    Returns a result dict; never silently drops the result even if the
    health gate fails.
    """
    torch.manual_seed(RANDOM_SEED)

    logger.info("\n" + "=" * 80)
    logger.info("RQ1 ABLATION STEP 1: reuse the SAME dataset + match-level split (ADR-011) as the physics-informed reference run")
    logger.info("=" * 80)
    loaded = _load_and_split_dataset()
    dataset = loaded["dataset"]
    frames = loaded["frames"]
    match_ids = loaded["match_ids"]
    sample_match_ids = loaded["sample_match_ids"]
    dataset_size = loaded["dataset_size"]
    competition_season_summary = loaded["competition_season_summary"]

    # Explicit confirmation, per this task's own instruction: do not assume
    # the split function is inherited correctly just because this is "one
    # new run." Independently re-invoke match_level_split (ADR-011) with
    # the SAME sample_match_ids/seed/val_fraction _load_and_split_dataset
    # used internally, and assert the resulting indices are IDENTICAL to
    # what it returned, rather than trusting that by construction alone.
    confirm_train_indices, confirm_val_indices = match_level_split(
        sample_match_ids, val_fraction=1.0 - TRAIN_FRACTION, seed=RANDOM_SEED
    )
    assert confirm_train_indices == loaded["train_set"].indices, (
        "match_level_split (ADR-011) did NOT reproduce the same train indices as "
        "_load_and_split_dataset() -- aborting rather than silently training on a mismatched split."
    )
    assert confirm_val_indices == loaded["val_set"].indices, (
        "match_level_split (ADR-011) did NOT reproduce the same val indices as "
        "_load_and_split_dataset() -- aborting rather than silently training on a mismatched split."
    )
    logger.info(
        "[RQ1 ablation] CONFIRMED: match_level_split (ADR-011, production.src.pipeline.data_split) "
        f"is the split function actually in effect for this run -- independently re-invoked with "
        f"the same sample_match_ids, seed={RANDOM_SEED}, val_fraction={1.0 - TRAIN_FRACTION}, "
        f"producing indices identical to _load_and_split_dataset()'s. {loaded['n_train']} train / "
        f"{loaded['n_val']} val samples, {len(match_ids)} matches."
    )

    logger.info("\n" + "=" * 80)
    logger.info("RQ1 ABLATION STEP 2: build the non-physics baseline feature set (raw pre-physics signal only)")
    logger.info("=" * 80)
    logger.info(
        f"[RQ1 ablation] Baseline feature set (naive_baseline_features.extract_naive_baseline_features): "
        f"{BASELINE_FEATURE_KEYS} -- simple counts/distances over RAW player_pos/ball_pos/is_teammate, "
        "computed with NO BiomechanicalPitchControl call, no ODE, no pitch-grid integration. "
        "player_vel and fatigue_mod are deliberately excluded: statsbomb_io.parse_360_frame sets "
        "them to an all-zero tensor and a constant 1.0 respectively for EVERY sample project-wide "
        "(StatsBomb's public 360 data has no velocity field) -- they carry no information to "
        "aggregate, project-wide, not just here."
    )
    baseline_features = [extract_naive_baseline_features(frame) for frame in frames]

    class _ScalarFeatureOverrideDataset(Dataset):
        """Wraps the ALREADY-BUILT physics-informed TacticalSurvivalDataset,
        replacing ONLY the scalar feature tensor (index 0 of each item)
        with this ablation's non-physics baseline features. Graph data,
        duration bins, and event flags (indices 1-3) are untouched --
        byte-for-byte identical to the physics-informed reference run,
        since both come from the exact same underlying frames/chains.
        """

        def __init__(self, base_dataset, override_feature_dicts, feature_key_order):
            self.base_dataset = base_dataset
            self.override_feature_dicts = override_feature_dicts
            self.feature_key_order = feature_key_order

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            _, graph_data, duration_bin_tensor, event_tensor = self.base_dataset[idx]
            feature_dict = self.override_feature_dicts[idx]
            features_tensor = torch.tensor(
                [feature_dict[key] for key in self.feature_key_order], dtype=torch.float32
            )
            return features_tensor, graph_data, duration_bin_tensor, event_tensor

    baseline_dataset = _ScalarFeatureOverrideDataset(dataset, baseline_features, BASELINE_FEATURE_KEYS)
    train_set = Subset(baseline_dataset, confirm_train_indices)
    val_set = Subset(baseline_dataset, confirm_val_indices)
    n_train, n_val = len(train_set), len(val_set)

    feature_mean, feature_std, _graph_feature_mean, _graph_feature_std = _compute_normalization_stats(
        baseline_dataset, train_set
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_batch = next(iter(DataLoader(val_set, batch_size=len(val_set))))

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info("\n" + "=" * 80)
    logger.info(
        "RQ1 ABLATION STEP 2 (cont.): train DeepHitSurvivalModel MLP, SAME architecture + "
        "hyperparameters as the match-level physics-informed reference run"
    )
    logger.info("=" * 80)
    torch.manual_seed(RANDOM_SEED)
    baseline_model = DeepHitSurvivalModel(num_features=len(BASELINE_FEATURE_KEYS), num_bins=NUM_BINS)
    baseline_optimizer = torch.optim.Adam(
        baseline_model.parameters(), lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY
    )
    baseline_results = _train_and_log_model(
        model_type="MLP",
        model=baseline_model,
        optimizer=baseline_optimizer,
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
            "split_type": "match_level",  # Milestone 35 / ADR-011, independently confirmed above
            "ablation": "rq1_non_physics_baseline",
            # `_train_and_log_model` unconditionally logs a GLOBAL
            # "feature_key_order" param (the physics FEATURE_KEYS names) --
            # pre-existing, unmodified behavior of that shared function,
            # left as-is rather than editing it for this one ablation call.
            # That logged value does NOT describe this run; the true
            # feature names actually used are recorded here instead, under
            # a distinct key.
            "true_feature_key_order": ",".join(BASELINE_FEATURE_KEYS),
        },
        run_tags={
            "ablation_purpose": (
                "RQ1 non-physics baseline: raw player/ball positions + is_teammate only, "
                "no BiomechanicalPitchControl"
            ),
        },
        normalization_artifact={
            "feature_key_order": list(BASELINE_FEATURE_KEYS),
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
    )

    baseline_healthy = _evaluate_mlp_health(baseline_results)

    logger.info("\n" + "=" * 80)
    logger.info("RQ1 ABLATION -- HEALTH GATE (ADR-010, full four-signal check)")
    logger.info("=" * 80)
    if baseline_results is not None:
        logger.info(
            f"Non-physics baseline MLP four-signal detector: spike={baseline_results['spike_fired']}, "
            f"cumulative_drift={baseline_results['cumulative_drift_fired']}, "
            f"saturation={baseline_results['saturation_fired']} (entropy/variance-based -- ADR-010's "
            f"primary trusted signal for ambiguous cases), "
            f"frozen_val_loss={baseline_results['frozen_val_loss_fired']} -> "
            f"baseline_healthy={baseline_healthy}"
        )
    else:
        logger.warning(
            "Non-physics baseline MLP: training ABORTED (NaN/Inf loss) -> baseline_healthy=False"
        )

    # Milestone 35 (ADR-011) match-level physics-informed MLP reference,
    # from RESEARCH_FINDINGS.md's RQ4 "Update -- Stage 3" table
    # (run_match_level_rq2_rq4_full_revalidation's own Step 2 MLP result --
    # not re-run here, since re-running it would not change the recorded
    # number and this ablation must not retrain the existing baseline).
    physics_match_level_brier_15s = 0.1009
    physics_match_level_brier_30s = 0.1873

    logger.info("\n" + "=" * 80)
    logger.info(
        "RQ1 ABLATION STEP 3: non-physics baseline vs. the current physics-informed match-level "
        "reference (both match_level split, ADR-011)"
    )
    logger.info("=" * 80)
    logger.info(f"{'Model':<58} {'split_type':>13} {'Brier@15s':>10} {'Brier@30s':>10}")
    logger.info(
        f"{'MLP, physics-informed (match_level, RESEARCH_FINDINGS.md RQ4)':<58} {'match_level':>13} "
        f"{physics_match_level_brier_15s:>10.4f} {physics_match_level_brier_30s:>10.4f}"
    )
    if baseline_results is not None:
        logger.info(
            f"{'MLP, non-physics baseline (this run, NEW)':<58} {'match_level':>13} "
            f"{baseline_results['brier_15s']:>10.4f} {baseline_results['brier_30s']:>10.4f}"
        )
    else:
        logger.info(f"{'MLP, non-physics baseline (this run, NEW)':<58} {'match_level':>13} {'ABORTED':>10} {'ABORTED':>10}")

    return {
        "dataset_size": dataset_size,
        "n_train": n_train,
        "n_val": n_val,
        "match_count": len(match_ids),
        "baseline_results": baseline_results,
        "baseline_healthy": baseline_healthy,
        "physics_match_level_brier_15s": physics_match_level_brier_15s,
        "physics_match_level_brier_30s": physics_match_level_brier_30s,
        "baseline_feature_keys": BASELINE_FEATURE_KEYS,
    }


# Repeated-measurement investigation of the RQ1 non-physics-baseline gap
# (run_rq1_non_physics_baseline_ablation's single-seed, single-split result:
# physics-informed MLP 0.1009/0.1873 vs. non-physics baseline MLP
# 0.0940/0.1840, match_level split seed=42) -- is that ~0.007/0.003 gap real
# and repeatable, or a single-run artifact? Same methodology as the GNN
# horizon-degradation investigation (`run_gnn_horizon_seed_split_sensitivity_check`
# above): (1) model-init-seed variation at the SAME match-level split, (2)
# split-seed variation (genuinely different match partitions). The
# physics-informed MLP side of every (init_seed, split_seed) combination this
# check needs is ALREADY on record from that exact investigation (same
# architecture, same hyperparameters, same match_level_split call) --
# reused directly rather than retrained, per that investigation's own
# established "don't retrain what's already on record" convention. Only the
# non-physics baseline (genuinely new to this ablation) is trained fresh at
# each combination. No model architecture, split function, or hyperparameter
# is modified anywhere in this section; feature_extractor.py and
# naive_baseline_features.py's core extraction logic are untouched.
RQ1_BASELINE_INVESTIGATION_TAG = "rq1_non_physics_baseline_seed_split_check"
RQ1_PHYSICS_MLP_SEED_SPLIT_RUNS = {
    (42, 42): "b77fdf76b79b4fc3a19035914a098091",
    (43, 42): "50fe80da239e4213bba5909cd72cdc5c",
    (44, 42): "cbef43bed56f4970a10724d99f86796d",
    (42, 43): "fe9fcf4722d6499eba6bf5b844c2cd12",
    (42, 44): "4b4170c94b014f95861de4a5adf32530",
}
# The non-physics baseline's own seed=42/split=42 run
# (run_rq1_non_physics_baseline_ablation's result) -- reused, not retrained.
RQ1_BASELINE_SEED42_SPLIT42_RUN_ID = "5ea3b1868c3a4f6ebb951ff3583c132d"


def _train_rq1_baseline_at_seed_and_split(
    init_seed: int, split_seed: int, loaded_full: dict, baseline_features: list[dict]
) -> dict:
    """Same shape as `_train_at_seed_and_split`, for the RQ1 non-physics
    baseline instead of the physics-informed MLP/GNN. `baseline_features` is
    precomputed ONCE by the caller and passed in -- `extract_naive_baseline_features`
    is a pure function of `frames` alone, independent of seed/split, so
    recomputing it per combination would be wasted, identical work.
    """
    frames = loaded_full["frames"]
    chains = loaded_full["chains"]
    sample_match_ids = loaded_full["sample_match_ids"]
    match_ids = loaded_full["match_ids"]
    dataset_size = loaded_full["dataset_size"]
    competition_season_summary = loaded_full["competition_season_summary"]

    # A fresh TacticalSurvivalDataset (its physics-derived `features` are
    # built but then immediately overridden below, never consumed) -- the
    # SAME pattern `run_rq1_non_physics_baseline_ablation` uses, just redone
    # per combination since the graph_data it wraps is independent of
    # init_seed/split_seed and cheap to rebuild from already-in-memory frames.
    dataset = TacticalSurvivalDataset(loaded_full["features"], frames, chains)

    class _ScalarFeatureOverrideDataset(Dataset):
        def __init__(self, base_dataset, override_feature_dicts, feature_key_order):
            self.base_dataset = base_dataset
            self.override_feature_dicts = override_feature_dicts
            self.feature_key_order = feature_key_order

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            _, graph_data, duration_bin_tensor, event_tensor = self.base_dataset[idx]
            feature_dict = self.override_feature_dicts[idx]
            features_tensor = torch.tensor(
                [feature_dict[key] for key in self.feature_key_order], dtype=torch.float32
            )
            return features_tensor, graph_data, duration_bin_tensor, event_tensor

    baseline_dataset = _ScalarFeatureOverrideDataset(dataset, baseline_features, BASELINE_FEATURE_KEYS)

    train_indices, val_indices = match_level_split(
        sample_match_ids, val_fraction=1.0 - TRAIN_FRACTION, seed=split_seed
    )
    train_set = Subset(baseline_dataset, train_indices)
    val_set = Subset(baseline_dataset, val_indices)
    n_train, n_val = len(train_set), len(val_set)

    feature_mean, feature_std, _graph_mean, _graph_std = _compute_normalization_stats(
        baseline_dataset, train_set
    )

    torch.manual_seed(init_seed)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(init_seed),
    )
    val_batch = next(iter(DataLoader(val_set, batch_size=len(val_set))))

    torch.manual_seed(init_seed)
    model = DeepHitSurvivalModel(num_features=len(BASELINE_FEATURE_KEYS), num_bins=NUM_BINS)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_STABILIZED_LR, weight_decay=MLP_STABILIZED_WEIGHT_DECAY)

    results = _train_and_log_model(
        model_type="MLP",
        model=model,
        optimizer=optimizer,
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
            "split_type": "match_level",
            "init_seed": init_seed,
            "split_seed": split_seed,
            "ablation": "rq1_non_physics_baseline",
            "true_feature_key_order": ",".join(BASELINE_FEATURE_KEYS),
        },
        run_tags={
            "investigation": RQ1_BASELINE_INVESTIGATION_TAG,
            "init_seed": str(init_seed),
            "split_seed": str(split_seed),
            "ablation_purpose": "RQ1 non-physics baseline seed/split robustness check",
        },
        normalization_artifact={
            "feature_key_order": list(BASELINE_FEATURE_KEYS),
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
    )
    healthy = _evaluate_mlp_health(results)
    return {
        "init_seed": init_seed,
        "split_seed": split_seed,
        "results": results,
        "run_id": results["run_id"] if results is not None else None,
        "healthy": healthy,
    }


def _reconstruct_mlp_health_from_run(client, run_id: str) -> tuple[float | None, float | None, bool]:
    """Re-derive (brier_15s, brier_30s, healthy) for an ALREADY-LOGGED MLflow
    run without re-running `_evaluate_mlp_health` (which needs the full
    in-process results dict, not just what MLflow persisted) -- same
    reconstruction `run_gnn_horizon_seed_split_sensitivity_check` performs
    for its own reused runs, applied here to this check's reused runs.
    """
    run = client.get_run(run_id)
    p, m = run.data.params, run.data.metrics
    brier_15s = m.get("val_brier_15s")
    brier_30s = m.get("val_brier_30s")
    instability = p.get("instability_warning_fired") == "True"
    brier_in_range = (
        brier_15s is not None and brier_30s is not None
        and brier_15s <= MLP_SANITY_BRIER_15S_CEILING
        and brier_30s <= MLP_SANITY_BRIER_30S_CEILING
    )
    healthy = (not instability) and brier_in_range
    return brier_15s, brier_30s, healthy


def run_rq1_baseline_seed_split_check() -> dict:
    """Steps 1+2 of the RQ1 non-physics-baseline repeated-measurement check:
    does the single-run 0.0069 @15s / 0.0033 @30s gap (physics-informed
    MINUS non-physics baseline; positive = physics worse) hold across
    model-initialization seeds (same match-level split) and across genuinely
    different split seeds (different match partitions)? Reuses the
    physics-informed MLP's already-on-record runs
    (`RQ1_PHYSICS_MLP_SEED_SPLIT_RUNS`) and the baseline's own seed=42/
    split=42 run (`RQ1_BASELINE_SEED42_SPLIT42_RUN_ID`) directly from
    MLflow; trains only the genuinely missing non-physics baseline
    (init_seed, split_seed) combinations.
    """
    torch.manual_seed(RANDOM_SEED)
    loaded_full = _load_and_split_dataset()
    baseline_features = [extract_naive_baseline_features(frame) for frame in loaded_full["frames"]]

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    combinations = [
        # Step 1: same split (42), vary init seed.
        (42, 42), (43, 42), (44, 42),
        # Step 2: same init seed (42), vary split seed.
        (42, 43), (42, 44),
    ]

    baseline_results: dict[tuple[int, int], dict] = {}
    for init_seed, split_seed in combinations:
        if (init_seed, split_seed) == (42, 42):
            b15, b30, healthy = _reconstruct_mlp_health_from_run(client, RQ1_BASELINE_SEED42_SPLIT42_RUN_ID)
            logger.info(
                f"\n[baseline init_seed={init_seed} split_seed={split_seed}] REUSING existing "
                f"run_id={RQ1_BASELINE_SEED42_SPLIT42_RUN_ID} (not retrained)."
            )
            baseline_results[(init_seed, split_seed)] = {
                "run_id": RQ1_BASELINE_SEED42_SPLIT42_RUN_ID,
                "brier_15s": b15, "brier_30s": b30, "healthy": healthy, "reused": True,
            }
            continue

        logger.info(f"\n[baseline init_seed={init_seed} split_seed={split_seed}] training fresh...")
        outcome = _train_rq1_baseline_at_seed_and_split(init_seed, split_seed, loaded_full, baseline_features)
        r = outcome["results"]
        baseline_results[(init_seed, split_seed)] = {
            "run_id": outcome["run_id"],
            "brier_15s": r["brier_15s"] if r is not None else None,
            "brier_30s": r["brier_30s"] if r is not None else None,
            "healthy": outcome["healthy"], "reused": False,
        }

    physics_results: dict[tuple[int, int], dict] = {}
    for key, run_id in RQ1_PHYSICS_MLP_SEED_SPLIT_RUNS.items():
        b15, b30, healthy = _reconstruct_mlp_health_from_run(client, run_id)
        physics_results[key] = {"run_id": run_id, "brier_15s": b15, "brier_30s": b30, "healthy": healthy}
        logger.info(
            f"[physics-informed init_seed={key[0]} split_seed={key[1]}] REUSING existing "
            f"run_id={run_id} (already on record, not retrained)."
        )

    logger.info("\n" + "=" * 100)
    logger.info("RQ1 BASELINE CHECK STEP 1: init-seed sensitivity (split_seed=42 fixed)")
    logger.info("=" * 100)
    logger.info(
        f"{'init_seed':>10} {'physics@15s':>12} {'physics@30s':>12} {'baseline@15s':>13} "
        f"{'baseline@30s':>13} {'gap@15s (phys-base)':>21} {'gap@30s (phys-base)':>21} {'both healthy':>13}"
    )
    for seed in (42, 43, 44):
        phys = physics_results[(seed, 42)]
        base = baseline_results[(seed, 42)]
        both_healthy = phys["healthy"] and base["healthy"]
        gap_15s = phys["brier_15s"] - base["brier_15s"] if phys["brier_15s"] is not None and base["brier_15s"] is not None else None
        gap_30s = phys["brier_30s"] - base["brier_30s"] if phys["brier_30s"] is not None and base["brier_30s"] is not None else None
        logger.info(
            f"{seed:>10} {phys['brier_15s'] if phys['brier_15s'] is not None else float('nan'):>12.4f} "
            f"{phys['brier_30s'] if phys['brier_30s'] is not None else float('nan'):>12.4f} "
            f"{base['brier_15s'] if base['brier_15s'] is not None else float('nan'):>13.4f} "
            f"{base['brier_30s'] if base['brier_30s'] is not None else float('nan'):>13.4f} "
            f"{gap_15s if gap_15s is not None else float('nan'):>21.4f} "
            f"{gap_30s if gap_30s is not None else float('nan'):>21.4f} {both_healthy!s:>13}"
        )
        if not both_healthy:
            logger.warning(
                f"  init_seed={seed}: NOT both healthy (physics healthy={phys['healthy']}, "
                f"baseline healthy={base['healthy']}) -- excluded from any conclusion."
            )

    logger.info("\n" + "=" * 100)
    logger.info("RQ1 BASELINE CHECK STEP 2: split-seed sensitivity (init_seed=42 fixed)")
    logger.info("=" * 100)
    logger.info(
        f"{'split_seed':>10} {'physics@15s':>12} {'physics@30s':>12} {'baseline@15s':>13} "
        f"{'baseline@30s':>13} {'gap@15s (phys-base)':>21} {'gap@30s (phys-base)':>21} {'both healthy':>13}"
    )
    for split_seed in (42, 43, 44):
        phys = physics_results[(42, split_seed)]
        base = baseline_results[(42, split_seed)]
        both_healthy = phys["healthy"] and base["healthy"]
        gap_15s = phys["brier_15s"] - base["brier_15s"] if phys["brier_15s"] is not None and base["brier_15s"] is not None else None
        gap_30s = phys["brier_30s"] - base["brier_30s"] if phys["brier_30s"] is not None and base["brier_30s"] is not None else None
        logger.info(
            f"{split_seed:>10} {phys['brier_15s'] if phys['brier_15s'] is not None else float('nan'):>12.4f} "
            f"{phys['brier_30s'] if phys['brier_30s'] is not None else float('nan'):>12.4f} "
            f"{base['brier_15s'] if base['brier_15s'] is not None else float('nan'):>13.4f} "
            f"{base['brier_30s'] if base['brier_30s'] is not None else float('nan'):>13.4f} "
            f"{gap_15s if gap_15s is not None else float('nan'):>21.4f} "
            f"{gap_30s if gap_30s is not None else float('nan'):>21.4f} {both_healthy!s:>13}"
        )
        if not both_healthy:
            logger.warning(
                f"  split_seed={split_seed}: NOT both healthy (physics healthy={phys['healthy']}, "
                f"baseline healthy={base['healthy']}) -- excluded from any conclusion."
            )

    return {"physics_results": physics_results, "baseline_results": baseline_results}


if __name__ == "__main__":
    # Only configured here, for this file's own standalone entrypoint --
    # see the module-level `logger` declaration's comment for why this must
    # never run at import time. Plain single-line format, stdout -- matches
    # this file's previous print()-based output closely enough that
    # existing habits (piping to a file, grepping for "WARNING") still work.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--ci-bootstrap" in sys.argv:
        run_ci_bootstrap_training()
    elif "--rq2-rq4-revalidation" in sys.argv:
        run_match_level_rq2_rq4_full_revalidation()
    elif "--gnn-horizon-check" in sys.argv:
        run_gnn_horizon_seed_split_sensitivity_check()
    elif "--rq1-non-physics-baseline" in sys.argv:
        run_rq1_non_physics_baseline_ablation()
    elif "--rq1-baseline-seed-split-check" in sys.argv:
        run_rq1_baseline_seed_split_check()
    else:
        train_and_evaluate()
