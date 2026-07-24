"""Milestone 11 validation: standalone graph-construction module.

Only tests build_graph_from_frame in isolation -- NOT wired into
TacticalSurvivalDataset or train.py (deferred to the future GNN-model
milestone).
"""

import torch
from torch_geometric.data import Data

from production.src.models.graph_builder import build_graph_from_frame

OPPONENT_RADIUS = 5.0
SAME_TEAM_RADIUS = 30.0


def _synthetic_6_player_frame():
    """3 attackers clustered near (10, 34), 3 defenders -- two of them
    clustered near (90, 34) (far from everyone, guaranteeing no accidental
    edges to them), and one (index 3) placed within 3m of the attacking
    cluster to guarantee at least one opponent edge by construction.
    """
    player_pos = torch.tensor(
        [
            [10.0, 34.0],  # 0: attacker
            [12.0, 35.0],  # 1: attacker
            [11.0, 33.0],  # 2: attacker
            [11.5, 34.5],  # 3: defender, deliberately close to the attackers
            [90.0, 34.0],  # 4: defender
            [92.0, 35.0],  # 5: defender
        ]
    )
    is_teammate = torch.tensor([True, True, True, False, False, False])
    player_vel = torch.zeros_like(player_pos)
    ball_pos = torch.tensor([11.0, 34.0])
    return player_pos, player_vel, is_teammate, ball_pos


def test_graph_structure_and_types():
    player_pos, player_vel, is_teammate, ball_pos = _synthetic_6_player_frame()
    n = player_pos.shape[0]

    data = build_graph_from_frame(player_pos, player_vel, is_teammate, ball_pos)

    assert isinstance(data, Data)
    assert data.x.shape == (n, 7)
    assert data.edge_index.shape[0] == 2
    assert not torch.isnan(data.x).any()
    assert not torch.isnan(data.edge_attr).any()


def test_same_team_and_opponent_edges_both_present():
    player_pos, player_vel, is_teammate, ball_pos = _synthetic_6_player_frame()
    data = build_graph_from_frame(player_pos, player_vel, is_teammate, ball_pos)

    is_attacker = is_teammate  # index 4 in node features == is_attacker, but the raw bool is simpler here
    src, dst = data.edge_index[0], data.edge_index[1]

    same_team_pairs = (is_attacker[src] == is_attacker[dst])
    opponent_pairs = ~same_team_pairs

    assert same_team_pairs.any(), "expected at least one same-team edge"
    assert opponent_pairs.any(), "expected at least one opponent edge"


def test_edges_are_bidirectional():
    player_pos, player_vel, is_teammate, ball_pos = _synthetic_6_player_frame()
    data = build_graph_from_frame(player_pos, player_vel, is_teammate, ball_pos)

    edge_pairs = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    assert len(edge_pairs) > 0

    for i, j in edge_pairs:
        assert (j, i) in edge_pairs, f"edge ({i},{j}) present without its reverse ({j},{i})"


def test_opponent_edge_respects_tight_radius_default():
    """Sanity check on the synthetic fixture itself: the far-away defender
    pair (indices 4, 5) must NOT produce opponent edges to the attacking
    cluster under the default 5.0m opponent_radius (they're ~78m away),
    confirming the fixture's "guaranteed by construction" claim isn't
    accidentally satisfied by a too-generous default.
    """
    player_pos, player_vel, is_teammate, ball_pos = _synthetic_6_player_frame()
    data = build_graph_from_frame(player_pos, player_vel, is_teammate, ball_pos)

    src, dst = data.edge_index[0].tolist(), data.edge_index[1].tolist()
    far_defender_indices = {4, 5}
    attacker_indices = {0, 1, 2}

    for s, d in zip(src, dst):
        if s in far_defender_indices and d in attacker_indices:
            raise AssertionError(f"unexpected opponent edge between far defender {s} and attacker {d}")


def test_no_fixed_22_player_count_assumed():
    """8-player frame (simulating a partially-visible freeze-frame):
    confirms the function does not assume a fixed 22-node count anywhere.
    """
    player_pos = torch.tensor(
        [
            [10.0, 34.0],
            [12.0, 35.0],
            [11.0, 33.0],
            [13.0, 34.0],
            [11.5, 34.5],
            [90.0, 34.0],
            [92.0, 35.0],
            [88.0, 33.0],
        ]
    )
    is_teammate = torch.tensor([True, True, True, True, False, False, False, False])
    player_vel = torch.zeros_like(player_pos)
    ball_pos = torch.tensor([11.0, 34.0])

    n = player_pos.shape[0]
    assert n == 8

    data = build_graph_from_frame(player_pos, player_vel, is_teammate, ball_pos)

    assert data.x.shape == (8, 7)
    assert not torch.isnan(data.x).any()
    assert not torch.isnan(data.edge_attr).any()
