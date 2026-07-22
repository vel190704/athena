"""Milestone 4 (data plumbing) validation: TacticalSurvivalDataset.

Only exercises the tensor plumbing for an already-computed (features,
duration, event) tabular dataset -- possession-chain duration/event
derivation from raw StatsBomb events is out of scope and not exercised here.
"""

import numpy as np
import pytest
import torch

from production.src.pipeline.survival_dataset import FEATURE_KEYS, TacticalSurvivalDataset

SEED = 42
NUM_POSSESSIONS = 100


def _build_synthetic_dataset_inputs():
    rng = np.random.default_rng(SEED)

    features = [{key: float(rng.normal()) for key in FEATURE_KEYS} for _ in range(NUM_POSSESSIONS)]
    durations = rng.uniform(10.0, 60.0, size=NUM_POSSESSIONS).tolist()
    events = rng.integers(0, 2, size=NUM_POSSESSIONS).tolist()

    return features, durations, events


def test_dataset_length():
    features, durations, events = _build_synthetic_dataset_inputs()
    dataset = TacticalSurvivalDataset(features, durations, events)
    assert len(dataset) == NUM_POSSESSIONS


def test_getitem_shapes_and_dtypes():
    features, durations, events = _build_synthetic_dataset_inputs()
    dataset = TacticalSurvivalDataset(features, durations, events)

    features_tensor, duration_tensor, event_tensor = dataset[0]

    assert features_tensor.shape == (len(FEATURE_KEYS),)
    assert features_tensor.dtype == torch.float32

    assert duration_tensor.shape == ()  # 0-dimensional scalar
    assert duration_tensor.dtype == torch.float32

    assert event_tensor.shape == ()  # 0-dimensional scalar
    assert event_tensor.dtype == torch.float32


def test_mismatched_lengths_raises_value_error():
    features, durations, events = _build_synthetic_dataset_inputs()
    with pytest.raises(ValueError):
        TacticalSurvivalDataset(features, durations[:-1], events)


def test_nonpositive_duration_raises_value_error():
    features, durations, events = _build_synthetic_dataset_inputs()
    durations[0] = 0.0
    with pytest.raises(ValueError):
        TacticalSurvivalDataset(features, durations, events)
