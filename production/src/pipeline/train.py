"""Milestone 7/8: end-to-end DeepHit training, baseline validation, and
MLflow experiment tracking.

Fetches multiple real StatsBomb matches, extracts spatial features
(Milestone 3) and possession-chain survival labels (Milestone 5), trains
the single-risk DeepHit model (Milestone 6A/6B) on the aggregate, evaluates
with a time-dependent Brier Score (Milestone 7 Step 1), and logs the whole
run -- params, metrics, model, and normalization stats -- to MLflow
(Milestone 8) so a future run can be meaningfully compared against this
baseline.

Run as: python -m production.src.pipeline.train
Then:   mlflow ui   (from the project root, to inspect results visually)

No hyperparameter tuning (Optuna, etc.) here -- this is passive, reproducible
logging of the current baseline.
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
from torch.utils.data import DataLoader, random_split

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.models.deephit import DeepHitSurvivalModel
from production.src.models.deephit_loss import DeepHitLoss
from production.src.models.evaluation import calculate_brier_score
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

# Real StatsBomb open-data World Cup 2022 match IDs, tried in order until
# NUM_MATCHES_NEEDED valid ones (both events AND 360 data present) are
# collected -- mirrors Milestone 3's find_valid_match_id verification
# rather than assuming any fixed subset works.
CANDIDATE_MATCH_IDS = [
    3857276,  # Canada vs Morocco
    3857271,  # England vs Iran
    3857296,  # Croatia vs Belgium
    3857274,  # Netherlands vs Ecuador
    3857255,  # Japan vs Spain
    3857272,  # England vs United States
    3857278,  # Iran vs United States
    3857277,  # Morocco vs Croatia
]
NUM_MATCHES_NEEDED = 5

NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
BRIER_TIME_BINS = (3, 6)  # 15s and 30s, at BIN_SIZE_SECONDS=5.0

SMALL_DATASET_WARNING_THRESHOLD = 500


def _resolve_valid_match_ids(candidate_ids: list[int], count: int) -> list[int]:
    """Collect the first `count` candidates with BOTH events and 360 data.

    Rate-limiting is already handled inside statsbomb_io._fetch_json (a
    time.sleep(0.5) after every real network fetch, never after a cache
    hit) -- no need to duplicate that delay here.
    """
    valid_ids = []
    for match_id in candidate_ids:
        events = fetch_match_events(match_id)
        if events is None:
            continue
        frames = fetch_match_360(match_id)
        if frames is None:
            continue
        valid_ids.append(match_id)
        if len(valid_ids) == count:
            break

    if len(valid_ids) < count:
        raise ValueError(
            f"only found {len(valid_ids)} valid matches among {len(candidate_ids)} "
            f"candidates; needed {count}"
        )
    return valid_ids


def _match_chains_with_features(match_id: int, engine: BiomechanicalPitchControl):
    """Pair each period-1 possession chain (Milestone 5) with a spatial
    feature vector (Milestone 3) extracted from ONE representative event in
    that chain -- the first event in the chain that has an associated 360
    freeze-frame (chains built purely from raw events have no guaranteed
    360 coverage for every single event; a chain with no 360-covered event
    at all is skipped, since there's no tactical snapshot to featurize).
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    chains = build_possession_chains(events)

    events_by_possession = defaultdict(list)
    for e in events:
        if e["period"] == 1:
            events_by_possession[e["possession"]].append(e)
    for group in events_by_possession.values():
        group.sort(key=lambda e: e["index"])

    matched_features, matched_chains = [], []
    for chain in chains:
        chain_events = events_by_possession.get(chain["chain_id"], [])

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
        if features is None:  # only possible if period != 1, which can't happen here
            continue

        matched_features.append(features)
        matched_chains.append(chain)

    print(f"  match {match_id}: {len(matched_chains)}/{len(chains)} chains matched to a 360 frame")
    return matched_features, matched_chains


def build_training_data():
    match_ids = _resolve_valid_match_ids(CANDIDATE_MATCH_IDS, NUM_MATCHES_NEEDED)
    print(f"Resolved {len(match_ids)} valid matches: {match_ids}")

    engine = BiomechanicalPitchControl()
    all_features, all_chains = [], []
    for match_id in match_ids:
        features, chains = _match_chains_with_features(match_id, engine)
        all_features.extend(features)
        all_chains.extend(chains)

    return all_features, all_chains, match_ids


def train_and_evaluate():
    torch.manual_seed(RANDOM_SEED)

    features, chains, match_ids = build_training_data()
    dataset_size = len(features)
    print(f"\nTotal (feature, chain) pairs across {NUM_MATCHES_NEEDED} matches: {dataset_size}")
    if dataset_size < SMALL_DATASET_WARNING_THRESHOLD:
        print(
            f"NOTE: {dataset_size} samples from {NUM_MATCHES_NEEDED} matches' first-half "
            "possession chains is a small dataset -- fine for a baseline smoke test, but the "
            "Brier Score numbers below should not be over-interpreted as a validated model."
        )

    dataset = TacticalSurvivalDataset(features, chains)

    n_train = int(TRAIN_FRACTION * len(dataset))
    n_val = len(dataset) - n_train
    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=split_generator)

    # Feature normalization: statistics computed from the TRAINING split
    # ONLY, after the split -- normalizing before splitting would leak
    # validation-set statistics into training.
    train_features_raw = torch.stack([dataset[i][0] for i in train_set.indices])
    feature_mean = train_features_raw.mean(dim=0)
    feature_std = train_features_raw.std(dim=0).clamp(min=1e-8)  # guard a constant feature
    print(f"\nFeature normalization stats (from {n_train} training samples):")
    print(f"  mean: {feature_mean.tolist()}")
    print(f"  std:  {feature_std.tolist()}")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    model = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
    loss_fn = DeepHitLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Explicit local file-store tracking URI (./mlruns, already gitignored
    # since Milestone 1) rather than relying on mlflow's zero-config
    # default, which this installed version resolves to a local sqlite db
    # instead of the classic ./mlruns file store.
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
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
            }
        )

        print(
            f"\nTraining for {NUM_EPOCHS} epochs on {n_train} samples "
            f"({n_val} held out for validation)..."
        )
        final_epoch_loss = None
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            epoch_loss_total = 0.0
            num_batches = 0

            for batch_idx, (features_batch, duration_bins_batch, events_batch) in enumerate(
                train_loader
            ):
                normalized_features = (features_batch - feature_mean) / feature_std

                optimizer.zero_grad()
                predictions = model(normalized_features)
                loss = loss_fn(predictions, duration_bins_batch, events_batch)

                if not torch.isfinite(loss):
                    print(f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Stopping training.")
                    return

                loss.backward()
                optimizer.step()

                epoch_loss_total += loss.item()
                num_batches += 1

            final_epoch_loss = epoch_loss_total / num_batches
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{NUM_EPOCHS}: training loss = {final_epoch_loss:.4f}")

        print(f"\nFinal training loss: {final_epoch_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_features, val_duration_bins, val_events = next(
                iter(DataLoader(val_set, batch_size=len(val_set)))
            )
            normalized_val_features = (val_features - feature_mean) / feature_std
            val_predictions = model(normalized_val_features)

            val_loss = loss_fn(val_predictions, val_duration_bins, val_events)
            print(f"Validation loss: {val_loss.item():.4f}")

            print("\nValidation Brier Scores:")
            briers = {}
            for time_bin in BRIER_TIME_BINS:
                brier, num_excluded = calculate_brier_score(
                    val_predictions, val_duration_bins, val_duration_bins, val_events, time_bin
                )
                seconds = time_bin * 5.0
                print(f"  time_bin={time_bin} ({seconds:.0f}s): Brier Score = {brier:.4f}")
                briers[time_bin] = (brier, num_excluded)

        print(f"\nDataset size: {dataset_size} total samples ({n_train} train / {n_val} val)")

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
        # so. Plain pickling is simpler and sufficient for this eager
        # nn.Module baseline.
        mlflow.pytorch.log_model(model, name="deephit_model", serialization_format="pickle")

        # Self-describing artifact: includes feature_key_order alongside the
        # mean/std vectors so the file means something even opened outside
        # MLflow, without needing to cross-reference the logged params.
        normalization_stats = {
            "feature_key_order": list(FEATURE_KEYS),
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(normalization_stats, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        try:
            mlflow.log_artifact(tmp_file_path, artifact_path="normalization")
        finally:
            os.remove(tmp_file_path)

        print(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"MLflow run ID: {run.info.run_id}")
        print("Run `mlflow ui` from the project root to inspect results visually.")


if __name__ == "__main__":
    train_and_evaluate()
