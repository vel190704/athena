"""Spatial feature extraction: BiomechanicalPitchControl outputs -> scalar
features for the survival-analysis modeling stage (Module 7, later
milestone).

Scope restriction: only period == 1 (first half) frames are processed.
StatsBomb does not normalize attacking direction across halves -- a team
attacking left-to-right in the first half attacks right-to-left in the
second -- so raw x-coordinate thresholds like "final third" (x > 66.0) or
"defending line" are only meaningful while direction of play is held
constant within a frame. Normalizing attacking direction across periods so
second-half frames can be included too is deferred; flag as a candidate for
ADR-002.
"""

import torch

from production.src.spatial.control import BiomechanicalPitchControl

PITCH_LENGTH = 100.0
PITCH_WIDTH = 68.0
NEAR_BALL_RADIUS = 15.0
FINAL_THIRD_X = 66.0


def _build_pitch_grid() -> torch.Tensor:
    """Full 100x68 grid (1m spacing) in the same scaled unit space used by
    parse_360_frame. Sparse masking down to the active cells near the ball
    happens inside BiomechanicalPitchControl itself (ADR-005) -- this
    function always builds the full dense grid as the engine's input.
    """
    xs = torch.arange(0, PITCH_LENGTH, dtype=torch.float32)
    ys = torch.arange(0, PITCH_WIDTH, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)


def _team_control(engine, pos, vel, fatigue, pitch_grid, ball_pos):
    """Run the physics engine for one team's sub-batch and collapse the
    per-player control probabilities down to a single per-cell team control
    probability.

    No explicit team-aggregation rule is specified for this milestone, so we
    take the max control probability across a team's players at each cell:
    the player most likely to reach a cell dominates that team's claim on
    it. This is a simplification -- a full joint/probabilistic combination
    across teammates (e.g. "probability at least one teammate wins the
    race") is a candidate refinement for a future ADR -- but is sufficient
    for this milestone's scalar feature extraction.

    Returns (active_grid_coords, team_control), or (None, None) if the team
    has zero visible players in this frame (the engine is never called with
    an empty player batch; callers must treat a None team_control as "zero
    control everywhere").
    """
    if pos.shape[0] == 0:
        return None, None
    active_coords, control_probabilities, _ = engine(pos, vel, fatigue, pitch_grid, ball_pos)
    team_control = control_probabilities.max(dim=0).values  # [N_active]
    return active_coords, team_control


def extract_features(frame: dict, engine: BiomechanicalPitchControl | None = None) -> dict | None:
    """Extract scalar pitch-control features from one parsed 360 frame.

    `frame` is the dict returned by statsbomb_io.parse_360_frame. Returns
    None for any frame outside period 1 (see module docstring) so callers
    can filter a stream of frames with `if features is None: continue`.
    """
    if frame["period"] != 1:
        return None

    if engine is None:
        engine = BiomechanicalPitchControl()

    ball_pos = frame["ball_pos"]
    player_pos = frame["player_pos"]
    player_vel = frame["player_vel"]
    fatigue_mod = frame["fatigue_mod"]
    is_teammate = frame["is_teammate"]

    attacking_mask = is_teammate
    defending_mask = ~is_teammate

    pitch_grid = _build_pitch_grid()

    # Two separate sub-batches (attacking / defending), split by
    # is_teammate, are passed to the physics engine independently. No
    # ghost/padded players are ever included -- each sub-batch only
    # contains the players actually visible for that side in this frame.
    att_coords, att_control = _team_control(
        engine,
        player_pos[attacking_mask],
        player_vel[attacking_mask],
        fatigue_mod[attacking_mask],
        pitch_grid,
        ball_pos,
    )
    def_coords, def_control = _team_control(
        engine,
        player_pos[defending_mask],
        player_vel[defending_mask],
        fatigue_mod[defending_mask],
        pitch_grid,
        ball_pos,
    )

    # active_grid_coords is identical between the two calls whenever both
    # ran, since both are fed the same pitch_grid/ball_pos/mask_radius --
    # either can define the shared active-cell coordinate space.
    active_coords = att_coords if att_coords is not None else def_coords

    if active_coords is None:
        # Neither team has a single visible player in this frame -- there is
        # no control to compute. Every feature is 0 by definition.
        return {
            "attacking_control_near_ball": 0.0,
            "defending_control_near_ball": 0.0,
            "attacking_control_final_third": 0.0,
            "space_behind_defending_line": 0.0,
        }

    n_active = active_coords.shape[0]
    if att_control is None:
        att_control = torch.zeros(n_active)
    if def_control is None:
        def_control = torch.zeros(n_active)

    dist_to_ball = torch.linalg.norm(active_coords - ball_pos.unsqueeze(0), dim=-1)
    near_ball_mask = dist_to_ball <= NEAR_BALL_RADIUS
    final_third_mask = active_coords[:, 0] > FINAL_THIRD_X

    attacking_control_near_ball = att_control[near_ball_mask].sum()
    defending_control_near_ball = def_control[near_ball_mask].sum()
    attacking_control_final_third = att_control[final_third_mask].sum()

    defending_positions = player_pos[defending_mask]
    if defending_positions.shape[0] > 0:
        highest_defending_x = defending_positions[:, 0].max()
    else:
        # No visible defenders in this frame: there is no defender to
        # establish a line, so treat it as sitting at the very back (x=0),
        # meaning "behind the line" spans essentially the whole pitch.
        # def_control is already all-zero in this branch, so the result
        # stays a large-but-finite number rather than being undefined.
        highest_defending_x = torch.tensor(0.0)

    behind_line_mask = active_coords[:, 0] > highest_defending_x
    space_behind_defending_line = (1.0 - def_control[behind_line_mask]).sum()

    return {
        "attacking_control_near_ball": attacking_control_near_ball.item(),
        "defending_control_near_ball": defending_control_near_ball.item(),
        "attacking_control_final_third": attacking_control_final_third.item(),
        "space_behind_defending_line": space_behind_defending_line.item(),
    }
