"""Tactical action counterfactual perturbation engine (Milestone 13,
Module 8 / RQ5).

Perturbs a real baseline feature vector to approximate the spatial effect
of a hypothetical tactical action (forcing play wide, pressing high,
dropping deep), so the resulting shift can be fed through the trained
DeepHit MLP and compared against real football intuition (RQ5) -- an open
research question this module reports evidence toward, not an engineering
assertion it enforces.

IMPORTANT CAVEATS (read before trusting simulator output):
- The percentages below are hand-chosen heuristic approximations of
  tactical shifts, NOT empirically calibrated from real pressing/
  dropping-deep event data. If this simulator becomes a load-bearing part
  of the project (rather than an exploratory tool), calibrating these
  numbers against real tactical-change event data is a candidate for a
  future ADR.
- Each action perturbs only 1-2 of the 4 features independently. Real
  tactical shifts likely move several features together in correlated ways
  not modeled here (e.g. a high press plausibly also compresses
  attacking_control_final_third, not just attacking_control_near_ball and
  space_behind_defending_line). A perturbed feature vector may therefore
  represent a feature combination the model never saw during training --
  an OUT-OF-DISTRIBUTION input, not a realistic match state. This matters
  for how much weight to put on the resulting predictions.
"""

import torch

FEATURE_KEYS = (
    "attacking_control_near_ball",
    "defending_control_near_ball",
    "attacking_control_final_third",
    "space_behind_defending_line",
)

SUPPORTED_ACTIONS = ("no_change", "force_wide", "high_press", "drop_deep")


def perturb_features(base_features: dict, action: str) -> dict:
    """Return a NEW feature dict representing `base_features` after a
    hypothetical tactical action. `base_features` is never mutated.

    Supported actions:
      - 'no_change': baseline, unperturbed.
      - 'force_wide': attacking_control_near_ball x0.80,
        attacking_control_final_third x0.90, space_behind_defending_line x1.15.
      - 'high_press': attacking_control_near_ball x1.20,
        space_behind_defending_line x0.70.
      - 'drop_deep': space_behind_defending_line x1.40,
        defending_control_near_ball x1.20.
    """
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unknown action {action!r}; supported: {SUPPORTED_ACTIONS}")

    features = dict(base_features)

    if action == "no_change":
        return features

    if action == "force_wide":
        features["attacking_control_near_ball"] *= 0.80
        features["attacking_control_final_third"] *= 0.90
        features["space_behind_defending_line"] *= 1.15
    elif action == "high_press":
        features["attacking_control_near_ball"] *= 1.20
        features["space_behind_defending_line"] *= 0.70
    elif action == "drop_deep":
        features["space_behind_defending_line"] *= 1.40
        features["defending_control_near_ball"] *= 1.20

    return features


GRAPH_SUPPORTED_ACTIONS = ("no_change", "high_press", "drop_deep")


def perturb_player_positions(
    player_pos: torch.Tensor, is_teammate: torch.Tensor, ball_pos: torch.Tensor, action: str
) -> torch.Tensor:
    """Move DEFENDING-team (is_teammate == False) players' positions
    according to a hypothetical tactical action -- the graph-representation
    counterpart to `perturb_features`, for the GNN (Milestone 14).

    Returns a NEW player_pos tensor. Does NOT touch any graph edges --
    callers MUST rebuild the graph from scratch (`build_graph_from_frame`)
    on the returned positions. Edges are distance-threshold-derived
    (same_team_radius, opponent_radius); moving nodes without recomputing
    edges would leave stale connectivity that no longer reflects the
    perturbed spatial arrangement, silently measuring something other than
    the intended counterfactual.

    Supported actions:
      - 'no_change': positions returned unchanged.
      - 'high_press': defenders move 5m toward the ball's actual (x, y)
        position, capped so a defender already within 5m snaps exactly to
        the ball rather than overshooting past it.
      - 'drop_deep': defenders move 10m toward their own goal (x=0, per
        ADR-009's per-actor attacking-direction convention), y unchanged,
        clamped so x cannot go negative.

    Same OOD caveat as `perturb_features` above: moving nodes by a fixed
    offset can also produce spatial arrangements outside the training
    distribution. Graph-based perturbation is not automatically more
    realistic than the scalar version just because it operates on raw
    positions instead of aggregate features.
    """
    if action not in GRAPH_SUPPORTED_ACTIONS:
        raise ValueError(f"unknown graph action {action!r}; supported: {GRAPH_SUPPORTED_ACTIONS}")

    new_pos = player_pos.clone()
    if action == "no_change":
        return new_pos

    defender_mask = ~is_teammate
    defender_pos = new_pos[defender_mask]

    if action == "high_press":
        to_ball = ball_pos.unsqueeze(0) - defender_pos  # [num_defenders, 2]
        dist = torch.linalg.norm(to_ball, dim=-1, keepdim=True).clamp(min=1e-8)
        step = torch.clamp(dist, max=5.0)  # don't overshoot past the ball
        direction = to_ball / dist
        new_defender_pos = defender_pos + direction * step
    else:  # drop_deep
        new_x = torch.clamp(defender_pos[:, 0] - 10.0, min=0.0)
        new_defender_pos = torch.stack([new_x, defender_pos[:, 1]], dim=-1)

    new_pos[defender_mask] = new_defender_pos
    return new_pos
