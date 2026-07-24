"""Milestone 12 (data plumbing) validation: TacticalSurvivalDataset serving
both scalar features (MLP) and graph data (GNN) from the same underlying
frame per sample, with discrete time-bin labels for DeepHit.

This replaces the Milestone 6A interface (`features`, `chains`) with the
Milestone 12 interface (`features`, `frames`, `chains`) -- a breaking
change, so this file is updated in place rather than left pointing at the
old constructor signature.

Only exercises the tensor plumbing for an already-computed (features,
frame, chain) possession-level dataset -- possession-chain duration/event
derivation lives in chain_builder.py, and graph construction itself lives
in graph_builder.py (Milestone 11) -- neither is re-tested here.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from production.src.pipeline.survival_dataset import (
    FEATURE_KEYS,
    MAX_DURATION_SECONDS,
    NUM_BINS,
    TacticalSurvivalDataset,
)

SEED = 42
NUM_POSSESSIONS = 100
PLAYERS_PER_FRAME = 4  # 2 attackers + 2 defenders -- small and deterministic enough for a unit test


def _build_synthetic_dataset_inputs():
    rng = np.random.default_rng(SEED)

    features = [{key: float(rng.normal()) for key in FEATURE_KEYS} for _ in range(NUM_POSSESSIONS)]

    frames = []
    for _ in range(NUM_POSSESSIONS):
        player_pos = torch.tensor(rng.uniform(0, 100, size=(PLAYERS_PER_FRAME, 2)), dtype=torch.float32)
        player_pos[:, 1] = torch.tensor(rng.uniform(0, 68, size=PLAYERS_PER_FRAME), dtype=torch.float32)
        frames.append(
            {
                "ball_pos": torch.tensor([50.0, 34.0]),
                "player_pos": player_pos,
                "player_vel": torch.zeros(PLAYERS_PER_FRAME, 2),
                "is_teammate": torch.tensor([True, True, False, False]),
                "event_type": "Pass",
                "period": 1,
                "team": "TestTeam",
            }
        )

    durations = rng.uniform(10.0, 60.0, size=NUM_POSSESSIONS).tolist()
    event_flags = rng.integers(0, 2, size=NUM_POSSESSIONS).tolist()

    chains = [
        {"chain_id": i, "duration_seconds": durations[i], "event_flag": event_flags[i]}
        for i in range(NUM_POSSESSIONS)
    ]

    return features, frames, chains


def test_dataset_length():
    features, frames, chains = _build_synthetic_dataset_inputs()
    dataset = TacticalSurvivalDataset(features, frames, chains)
    assert len(dataset) == NUM_POSSESSIONS


def test_getitem_shapes_and_dtypes():
    features, frames, chains = _build_synthetic_dataset_inputs()
    dataset = TacticalSurvivalDataset(features, frames, chains)

    features_tensor, graph_data, duration_bin_tensor, event_tensor = dataset[0]

    assert features_tensor.shape == (len(FEATURE_KEYS),)
    assert features_tensor.dtype == torch.float32

    assert isinstance(graph_data, Data)
    assert graph_data.x.shape == (PLAYERS_PER_FRAME, 7)

    assert duration_bin_tensor.shape == ()
    assert duration_bin_tensor.dtype == torch.long
    assert 0 <= duration_bin_tensor.item() < NUM_BINS

    assert event_tensor.shape == ()
    assert event_tensor.dtype == torch.float32


def test_mismatched_lengths_raises_value_error():
    features, frames, chains = _build_synthetic_dataset_inputs()
    with pytest.raises(ValueError):
        TacticalSurvivalDataset(features, frames, chains[:-1])


def test_zero_duration_chain_is_floored_not_rejected():
    """Milestone 5 found real chains with duration_seconds == 0 (1-second
    timestamp granularity, not a bug). The dataset must floor these to a
    usable positive duration rather than raising ValueError.
    """
    features, frames, chains = _build_synthetic_dataset_inputs()
    chains[0]["duration_seconds"] = 0.0
    dataset = TacticalSurvivalDataset(features, frames, chains)  # must not raise
    assert dataset.durations[0] == 1.0


def test_horizon_censoring_forces_event_flag_to_zero():
    features, frames, chains = _build_synthetic_dataset_inputs()
    chains[0]["duration_seconds"] = 90.0
    chains[0]["event_flag"] = 1
    dataset = TacticalSurvivalDataset(features, frames, chains)
    assert dataset.events[0] == 0
