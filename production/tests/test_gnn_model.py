"""Milestone 12 validation: GNNDeepHitSurvivalModel."""

import torch
from torch_geometric.data import Data

from production.src.models.gnn_model import NUM_NODE_FEATURES, GNNDeepHitSurvivalModel

NUM_BINS = 12


def test_isolated_node_produces_finite_valid_pmf():
    """A graph with at least one fully isolated node (zero edges) must
    still produce a finite, non-NaN, valid PMF -- this directly validates
    the isolated-node handling Milestone 11 explicitly deferred to this
    milestone (SAGEConv's own-feature self-transformation, not a
    self-loop workaround).
    """
    # 4 nodes: 0 and 1 are connected to each other; 2 and 3 are fully
    # isolated -- neither appears anywhere in edge_index.
    x = torch.randn(4, NUM_NODE_FEATURES)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0], [1.0]])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.batch = torch.zeros(4, dtype=torch.long)  # single graph

    model = GNNDeepHitSurvivalModel(num_node_features=NUM_NODE_FEATURES, num_bins=NUM_BINS)
    output = model(data)

    assert output.shape == (1, NUM_BINS)
    assert not torch.isnan(output).any()
    assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-4)


def test_fully_disconnected_graph_produces_finite_valid_pmf():
    """Every node isolated (zero edges at all) -- the extreme case."""
    x = torch.randn(5, NUM_NODE_FEATURES)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(5, dtype=torch.long)

    model = GNNDeepHitSurvivalModel(num_node_features=NUM_NODE_FEATURES, num_bins=NUM_BINS)
    output = model(data)

    assert output.shape == (1, NUM_BINS)
    assert not torch.isnan(output).any()
    assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-4)
