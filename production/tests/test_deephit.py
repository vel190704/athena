"""Milestone 6A smoke test: DeepHit dataset wiring + single-risk network.

Only exercises tensor plumbing and output shape/validity -- the DeepHit
loss function is a later milestone (6B) and is not exercised here.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from production.src.models.deephit import DeepHitSurvivalModel
from production.src.pipeline.survival_dataset import (
    FEATURE_KEYS,
    MAX_DURATION_SECONDS,
    NUM_BINS,
    TacticalSurvivalDataset,
)

SEED = 42
NUM_POSSESSIONS = 50
BATCH_SIZE = 16


def _build_synthetic_inputs():
    rng = np.random.default_rng(SEED)

    features = [{key: float(rng.normal()) for key in FEATURE_KEYS} for _ in range(NUM_POSSESSIONS)]
    # Deliberately spans above the 60s horizon (up to 90s) to exercise the
    # horizon-censoring rule from survival_dataset.py Step 1.4.
    durations = rng.uniform(1.0, 90.0, size=NUM_POSSESSIONS).tolist()
    event_flags = rng.integers(0, 2, size=NUM_POSSESSIONS).tolist()

    chains = [
        {"chain_id": i, "duration_seconds": durations[i], "event_flag": event_flags[i]}
        for i in range(NUM_POSSESSIONS)
    ]
    return features, chains


def test_deephit_model_outputs_valid_pmf():
    features, chains = _build_synthetic_inputs()
    dataset = TacticalSurvivalDataset(features, chains)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)

    features_batch, duration_bins_batch, event_batch = next(iter(loader))
    output = model(features_batch)

    assert output.shape == (BATCH_SIZE, NUM_BINS)
    assert torch.allclose(output.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-4)


def test_horizon_censoring_rule_fires_on_synthetic_data():
    """Separate from the model smoke test: confirms Step 1.4's
    horizon-censoring rule actually overrides event_flag, not merely that
    censored-at-horizon samples happen to exist.
    """
    features, chains = _build_synthetic_inputs()

    # Samples where the rule has something to actually override: a real
    # shot (event_flag == 1) beyond the modeled horizon.
    eligible = [
        i
        for i, c in enumerate(chains)
        if c["duration_seconds"] >= MAX_DURATION_SECONDS and c["event_flag"] == 1
    ]
    assert len(eligible) > 0, "synthetic data didn't produce a horizon-censoring case to test"

    dataset = TacticalSurvivalDataset(features, chains)
    assert any(dataset.events[i] == 0 for i in eligible)
