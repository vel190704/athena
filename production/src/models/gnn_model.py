"""Single-risk GNN DeepHit model (Milestone 12) -- the RQ4 candidate
architecture, compared against the scalar-feature MLP baseline
(DeepHitSurvivalModel, Milestone 6A) on the exact same train/val split.

Same single-risk PMF output convention as DeepHitSurvivalModel: output
shape is `[batch_size, num_bins]`, softmax-normalized, with NO separate
censoring-risk channel. See deephit.py's docstring for the full rationale
(one real risk -- a shot -- plus administrative censoring handled at the
loss level, not as a second predicted channel); it applies identically
here.

SAGEConv (not GCNConv) is used specifically because its neighborhood
aggregation always includes a separate linear transform of the node's OWN
features alongside the (mean-)aggregated neighbor features. A node with
zero edges (an isolated node -- explicitly possible per Milestone 11's
graph_builder.py, which deferred handling it to this milestone) still gets
a well-defined embedding purely from its own feature transform, with no
division-by-degree term that breaks down at degree zero the way GCNConv's
symmetric normalization does.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

NUM_NODE_FEATURES = 7  # [x, y, vx, vy, is_attacker, is_defender, dist_to_ball] -- graph_builder.py


class GNNDeepHitSurvivalModel(nn.Module):
    """2-layer GraphSAGE encoder -> global mean pool -> linear -> softmax
    PMF over discrete time bins, for a SINGLE risk (shot).

    Accepts PyG `Batch` objects directly in forward (also works with a
    single ungrouped `Data` object, for standalone testing -- see the
    `batch` fallback below).
    """

    def __init__(
        self, num_node_features: int = NUM_NODE_FEATURES, num_bins: int = 12, hidden_dim: int = 64
    ):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, num_bins)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        # A single (non-batched) Data object has no `.batch` attribute --
        # treat it as one graph so this also works for standalone testing.
        batch = data.batch
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        graph_embedding = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        logits = self.output_layer(graph_embedding)
        return self.softmax(logits)
