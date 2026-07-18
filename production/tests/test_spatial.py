"""Milestone 2 validation: BiomechanicalPitchControl (Module 1) physics.

Validates the vectorized analytical ODE solver and sparse masking (ADR-005)
without any GUI/rendering dependency.
"""

import math

import torch

from production.src.spatial.control import BiomechanicalPitchControl


def _build_pitch_grid() -> torch.Tensor:
    """Standard 100x68 grid (1m spacing), flattened to [6800, 2]."""
    xs = torch.arange(0, 100, dtype=torch.float32)
    ys = torch.arange(0, 68, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [6800, 2]


def test_sparse_masking_reduces_active_cells():
    engine = BiomechanicalPitchControl()

    pitch_grid = _build_pitch_grid()  # 6800 cells
    ball_pos = torch.tensor([50.0, 34.0])  # center circle

    player_pos = torch.tensor([[50.0, 34.0]])
    player_vel = torch.tensor([[0.0, 0.0]])
    fatigue_mod = torch.tensor([1.0])

    active_coords, control_probabilities, time_to_intercepts = engine(
        player_pos, player_vel, fatigue_mod, pitch_grid, ball_pos
    )

    # A 30m-radius circle fully inside these pitch bounds has area
    # pi * 30^2 ~= 2827 cells at 1m spacing, so the expected ballpark is
    # ~2700-2900 active cells. We assert the general masking property
    # (strictly reduced, but non-empty) rather than an exact, brittle count.
    assert len(active_coords) > 0
    assert len(active_coords) < pitch_grid.shape[0]
    assert control_probabilities.shape == (1, len(active_coords))
    assert time_to_intercepts.shape == (1, len(active_coords))


def test_moving_player_is_faster_than_stationary():
    engine = BiomechanicalPitchControl()

    target = torch.tensor([10.0, 0.0])
    pitch_grid = target.unsqueeze(0)  # single active cell
    ball_pos = target  # ensure the cell survives the 30m mask

    player_pos = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # A, B at same spot
    player_vel = torch.tensor([[0.0, 0.0], [5.0, 0.0]])  # A stationary, B sprinting toward target
    fatigue_mod = torch.tensor([1.0, 1.0])

    _, _, time_to_intercepts = engine(player_pos, player_vel, fatigue_mod, pitch_grid, ball_pos)

    t_a = time_to_intercepts[0, 0]
    t_b = time_to_intercepts[1, 0]
    assert t_b < t_a


def test_fatigue_penalty_slows_player():
    engine = BiomechanicalPitchControl()

    target = torch.tensor([10.0, 0.0])
    pitch_grid = target.unsqueeze(0)
    ball_pos = target

    player_pos = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    player_vel = torch.tensor([[3.0, 0.0], [3.0, 0.0]])  # identical velocity
    fatigue_mod = torch.tensor([1.0, 0.7])  # A fresh, B fatigued

    _, _, time_to_intercepts = engine(player_pos, player_vel, fatigue_mod, pitch_grid, ball_pos)

    t_a = time_to_intercepts[0, 0]
    t_b = time_to_intercepts[1, 0]
    assert t_b > t_a


def test_moving_away_clamp_matches_stationary():
    engine = BiomechanicalPitchControl()

    target = torch.tensor([10.0, 0.0])
    pitch_grid = target.unsqueeze(0)
    ball_pos = target

    player_pos = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    # Both players have the SAME total speed magnitude (5.0) so the Step-1
    # force-velocity decay curve (which depends on ||v||, not the radial
    # component) gives them identical a_eff. This isolates the Step-3
    # radial-velocity clamp as the only remaining variable: A moves
    # perpendicular to the target direction (v_radial == 0, untouched by the
    # clamp), B moves directly away (v_radial == -5, clamped to 0). A
    # literal vel=[0,0] "stationary" player would also have zero speed
    # magnitude, which additionally raises its a_eff relative to a 5 m/s
    # mover and would conflate the two effects instead of isolating the
    # clamp.
    player_vel = torch.tensor([[0.0, 5.0], [-5.0, 0.0]])  # A perpendicular, B moving away
    fatigue_mod = torch.tensor([1.0, 1.0])

    _, _, time_to_intercepts = engine(player_pos, player_vel, fatigue_mod, pitch_grid, ball_pos)

    t_a = time_to_intercepts[0, 0]
    t_b = time_to_intercepts[1, 0]
    assert math.isclose(t_a.item(), t_b.item(), rel_tol=1e-4, abs_tol=1e-4)
