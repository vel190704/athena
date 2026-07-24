"""Milestone 12 Step 3: integration smoke test for TacticalSurvivalDataset's
dual (scalar, graph) representation through a real PyG DataLoader batch,
feeding BOTH DeepHitSurvivalModel (MLP) and GNNDeepHitSurvivalModel (GNN).

Deliberately run BEFORE the full two-model, 50-epoch training run (Step 4)
-- this is meant to catch PyG batching/collation issues cheaply, rather
than discovering them midway through two full training runs.
"""

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from production.src.models.deephit import DeepHitSurvivalModel
from production.src.models.gnn_model import GNNDeepHitSurvivalModel
from production.src.pipeline.survival_dataset import (
    FEATURE_KEYS,
    NUM_BINS,
    TacticalSurvivalDataset,
)

SEED = 42
NUM_POSSESSIONS = 10
BATCH_SIZE = 4
PLAYERS_PER_FRAME = 4


def _build_synthetic_dataset():
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

    return TacticalSurvivalDataset(features, frames, chains)


def test_pyg_dataloader_batches_scalar_and_graph_together():
    dataset = _build_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    scalar_batch, graph_batch, duration_batch, event_batch = next(iter(loader))

    assert scalar_batch.shape == (BATCH_SIZE, len(FEATURE_KEYS))
    assert isinstance(graph_batch, Batch)
    assert hasattr(graph_batch, "batch")  # node -> graph-index mapping
    assert graph_batch.batch is not None
    assert duration_batch.shape == (BATCH_SIZE,)
    assert event_batch.shape == (BATCH_SIZE,)


def test_both_models_consume_the_same_batch_with_no_errors_or_nans():
    dataset = _build_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    scalar_batch, graph_batch, duration_batch, event_batch = next(iter(loader))

    mlp = DeepHitSurvivalModel(num_features=len(FEATURE_KEYS), num_bins=NUM_BINS)
    gnn = GNNDeepHitSurvivalModel(num_node_features=7, num_bins=NUM_BINS)

    mlp_output = mlp(scalar_batch)
    gnn_output = gnn(graph_batch)

    assert mlp_output.shape == (BATCH_SIZE, NUM_BINS)
    assert gnn_output.shape == (BATCH_SIZE, NUM_BINS)

    assert not torch.isnan(mlp_output).any()
    assert not torch.isnan(gnn_output).any()
