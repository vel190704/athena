"""Graph construction for a future GNN module (Milestone 11 -- standalone).

Converts a single 360 freeze-frame into a PyTorch Geometric `Data` object.
This is a fully standalone, independently-tested module: it is NOT wired
into `TacticalSurvivalDataset` or `train.py` yet. RQ4 (README) asks whether
graph representations OUTPERFORM the handcrafted scalar features validated
in Milestones 8-10B; answering that requires keeping that scalar-feature +
MLP baseline fully intact and comparable in the meantime. Wiring this
builder into the training pipeline is deferred to the milestone where an
actual GNN model (GCN/GAT conv layers) exists to consume it.
"""

import torch
from torch_geometric.data import Data

DEFAULT_SAME_TEAM_RADIUS = 30.0
DEFAULT_OPPONENT_RADIUS = 5.0


def build_graph_from_frame(
    player_pos: torch.Tensor,
    player_vel: torch.Tensor,
    is_teammate: torch.Tensor,
    ball_pos: torch.Tensor,
    same_team_radius: float = DEFAULT_SAME_TEAM_RADIUS,
    opponent_radius: float = DEFAULT_OPPONENT_RADIUS,
) -> Data:
    """Build a PyG graph from one parsed 360 freeze-frame.

    Inputs match the ACTUAL fields statsbomb_io.parse_360_frame (Milestone
    3) produces: `player_pos` [N, 2], `player_vel` [N, 2] (always zero
    today -- StatsBomb 360 has no velocity field; this is an inherited
    limitation from parse_360_frame, not new information introduced here),
    `is_teammate` [N] bool (relative to the event's acting player, NOT an
    absolute team_id -- a single freeze-frame cannot derive that), and
    `ball_pos` [2]. N is variable per frame -- no fixed 22-player count is
    assumed anywhere, consistent with Milestone 3's no-padding decision.

    Coordinates are assumed already in the final ADR-002/ADR-009-compliant
    scaled frame (100x68) by the time they reach this function -- no
    further transformation (rescaling or flipping) is applied here.

    Nodes: one per visible player. Features are
    `[x, y, vx, vy, is_attacker, is_defender, dist_to_ball]` --
    `is_attacker = is_teammate` (the possessing/acting team is the
    attacking team under this pipeline's established convention),
    `is_defender = ~is_teammate`, `dist_to_ball` is the Euclidean distance
    to `ball_pos` in the same scaled units.

    Edges are built in BOTH directions explicitly: for every connected pair
    (i, j), both (i, j) and (j, i) appear in `edge_index`. Most PyG conv
    layers (GCN, GAT) expect this explicit bidirectional representation for
    undirected graphs; omitting one direction won't error but will
    silently halve information flow once a real GNN layer consumes this
    graph in a future milestone.
      - Same-team ("passing options"): pairs sharing `is_teammate` status
        within `same_team_radius` meters.
      - Opponent ("marking/pressure"): cross-team pairs within
        `opponent_radius` meters.
    Edge weight (`edge_attr`) is `1.0 / (distance + 1.0)`, the same value
    for both directions of a pair.

    Both radii are tunable parameters, not hardcoded magic numbers --
    consistent with this project's established pattern (Kalman's Q/R,
    pitch control's `mask_radius`) of treating such distances as working
    hypotheses to validate, not fixed truths.

    A player with zero edges under both thresholds (an isolated node) is
    possible and is left as a degree-zero node here. This is a known
    consideration for whichever future milestone adds actual GNN conv
    layers (self-loops or an isolated-node strategy may be needed then) --
    not something this standalone builder attempts to fix, since there is
    no consumer yet to validate a fix against.
    """
    n = player_pos.shape[0]

    dist_to_ball = torch.linalg.norm(player_pos - ball_pos.unsqueeze(0), dim=-1)  # [N]

    is_attacker = is_teammate.float()
    is_defender = (~is_teammate).float()

    x = torch.stack(
        [
            player_pos[:, 0],
            player_pos[:, 1],
            player_vel[:, 0],
            player_vel[:, 1],
            is_attacker,
            is_defender,
            dist_to_ball,
        ],
        dim=-1,
    )  # [N, 7]

    pairwise_dist = torch.cdist(player_pos, player_pos)  # [N, N], symmetric

    same_team_mask = is_teammate.unsqueeze(1) == is_teammate.unsqueeze(0)  # [N, N], symmetric
    opponent_mask = ~same_team_mask

    within_same_team_radius = pairwise_dist <= same_team_radius
    within_opponent_radius = pairwise_dist <= opponent_radius

    # "Player pairs" means two distinct players -- exclude self-pairs (a
    # player is trivially both "same team" and distance 0 from itself,
    # which would otherwise produce a spurious same-team self-loop).
    not_self = ~torch.eye(n, dtype=torch.bool)

    same_team_edge_mask = same_team_mask & within_same_team_radius & not_self
    opponent_edge_mask = opponent_mask & within_opponent_radius & not_self

    edge_mask = same_team_edge_mask | opponent_edge_mask  # [N, N]

    # edge_mask is symmetric by construction (same_team_mask, opponent_mask,
    # and pairwise_dist are all symmetric), so nonzero() below naturally
    # yields BOTH (i, j) and (j, i) for every connected pair -- this is what
    # makes edge_index bidirectional, not a separate manual mirroring step.
    edge_index = edge_mask.nonzero(as_tuple=False).t().contiguous()  # [2, num_edges]

    # Boolean-mask indexing and .nonzero() both traverse in the same
    # row-major order, so edge_weight[k] lines up with edge_index[:, k].
    edge_weight = 1.0 / (pairwise_dist[edge_mask] + 1.0)
    edge_attr = edge_weight.unsqueeze(-1)  # [num_edges, 1]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
