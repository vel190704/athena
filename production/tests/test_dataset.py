"""Milestone 6A (data plumbing) validation: TacticalSurvivalDataset with
discrete time-bin labels for DeepHit.

This replaces the Milestone 4 interface (parallel `durations`/`events`
lists) with the Milestone 6A interface (`chains` dicts, matching
chain_builder.py's output) -- a breaking change, so this file is updated
in place rather than left pointing at the old constructor signature.

Only exercises the tensor plumbing for an already-computed (features,
chain) possession-level dataset -- possession-chain duration/event
derivation from raw StatsBomb events lives in chain_builder.py and is not
re-tested here.
"""

import numpy as np
import pytest
import torch

from production.src.pipeline.survival_dataset import (
    FEATURE_KEYS,
    MAX_DURATION_SECONDS,
    NUM_BINS,
    TacticalSurvivalDataset,
)

SEED = 42
NUM_POSSESSIONS = 100


def _build_synthetic_dataset_inputs():
    rng = np.random.default_rng(SEED)

    features = [{key: float(rng.normal()) for key in FEATURE_KEYS} for _ in range(NUM_POSSESSIONS)]
    durations = rng.uniform(10.0, 60.0, size=NUM_POSSESSIONS).tolist()
    event_flags = rng.integers(0, 2, size=NUM_POSSESSIONS).tolist()

    chains = [
        {"chain_id": i, "duration_seconds": durations[i], "event_flag": event_flags[i]}
        for i in range(NUM_POSSESSIONS)
    ]

    return features, chains


def test_dataset_length():
    features, chains = _build_synthetic_dataset_inputs()
    dataset = TacticalSurvivalDataset(features, chains)
    assert len(dataset) == NUM_POSSESSIONS


def test_getitem_shapes_and_dtypes():
    features, chains = _build_synthetic_dataset_inputs()
    dataset = TacticalSurvivalDataset(features, chains)

    features_tensor, duration_bin_tensor, event_tensor = dataset[0]

    assert features_tensor.shape == (len(FEATURE_KEYS),)
    assert features_tensor.dtype == torch.float32

    assert duration_bin_tensor.shape == ()
    assert duration_bin_tensor.dtype == torch.long
    assert 0 <= duration_bin_tensor.item() < NUM_BINS

    assert event_tensor.shape == ()
    assert event_tensor.dtype == torch.float32


def test_mismatched_lengths_raises_value_error():
    features, chains = _build_synthetic_dataset_inputs()
    with pytest.raises(ValueError):
        TacticalSurvivalDataset(features, chains[:-1])


def test_zero_duration_chain_is_floored_not_rejected():
    """Milestone 5 found real chains with duration_seconds == 0 (1-second
    timestamp granularity, not a bug). The dataset must floor these to a
    usable positive duration rather than raising ValueError.
    """
    features, chains = _build_synthetic_dataset_inputs()
    chains[0]["duration_seconds"] = 0.0
    dataset = TacticalSurvivalDataset(features, chains)  # must not raise
    assert dataset.durations[0] == 1.0


def test_horizon_censoring_forces_event_flag_to_zero():
    features, chains = _build_synthetic_dataset_inputs()
    chains[0]["duration_seconds"] = 90.0
    chains[0]["event_flag"] = 1
    dataset = TacticalSurvivalDataset(features, chains)
    assert dataset.events[0] == 0
