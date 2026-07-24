"""Spatial feature extraction: BiomechanicalPitchControl outputs -> scalar
features for the survival-analysis modeling stage (Module 7, later
milestone).

Convention (ADR-009, superseding Milestone 10 / ADR-003): NO direction
correction is applied here. StatsBomb's raw event `location` and 360
freeze-frame coordinates are already recorded relative to the ACTING
team's own attacking-left-to-right perspective, in both halves -- verified
empirically via goal-kick clustering, turnover-coordinate mirroring,
own-goal event pairs, and (for freeze-frame data specifically) goalkeeper
position clustering by teammate flag (see ADR-009). The ATTACKING team
(the team whose action/chain this frame represents -- `is_teammate` in the
parsed frame) already moves toward INCREASING x by construction of the raw
data itself, so the DEFENDING team's own goal is at x=0 (they are defending
against the attacking team's advance toward x=100) with no flip needed.
Every x-coordinate threshold below (final third, defending line) relies on
this already-consistent orientation. Both periods are included; neither is
excluded or transformed.
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


def extract_features(
    frame: dict,
    engine: BiomechanicalPitchControl | None = None,
    habit_heatmaps: dict | None = None,
) -> dict:
    """Extract scalar pitch-control features from one parsed 360 frame.

    `frame` is the dict returned by statsbomb_io.parse_360_frame. Per
    ADR-009, no direction inference or coordinate flip is applied here --
    StatsBomb's raw coordinates are already oriented relative to the
    acting team's own attacking-left-to-right perspective in both periods,
    so every frame (either period) is used as-is after the ADR-002 rescale
    that already happened in parse_360_frame.

    `habit_heatmaps` (Milestone 22, Module 6 -- OPT-IN, defaults to None/
    off): a dict `{player_id: heatmap_grid}` for whatever player identity
    is ACTUALLY available in this data -- in practice, at most one entry,
    for the frame's own acting player (StatsBomb's public 360 data exposes
    no identity for the other ~21 visible players; see
    `statsbomb_io.parse_360_frame`'s docstring). When None (the default),
    this function behaves EXACTLY as it did before Milestone 22 -- see
    `test_habit_memory.py`'s regression test, which confirms
    byte-identical output to the pre-Milestone-22 extraction path. Every
    previously trained model (MLP, GNN, Deep Ensemble) and every logged
    MLflow baseline was trained WITHOUT habit blending; silently changing
    the default here would invalidate every prior comparison in this
    project's history -- the same risk Milestone 11 avoided by keeping the
    graph builder standalone rather than wiring it into the dataset
    prematurely.

    When provided and the frame's `actor_player_id` has a matching
    heatmap, the actor's OWN position (identified via `frame["is_actor"]`)
    is replaced with its Bayesian-blended expected position
    (`habit_memory.bayesian_blend_habit`) before running
    `BiomechanicalPitchControl`. No other player's position is touched,
    since no other player's identity is known.
    """
    if engine is None:
        engine = BiomechanicalPitchControl()

    ball_pos = frame["ball_pos"]
    player_pos = frame["player_pos"]
    player_vel = frame["player_vel"]
    fatigue_mod = frame["fatigue_mod"]
    is_teammate = frame["is_teammate"]

    if habit_heatmaps is not None:
        actor_player_id = frame.get("actor_player_id")
        if actor_player_id is not None and actor_player_id in habit_heatmaps:
            # Local import: habit_memory.py imports PITCH_LENGTH/PITCH_WIDTH
            # from THIS module, so a module-level import here would be
            # circular. By the time extract_features is actually called,
            # both modules are already fully loaded.
            from production.src.pipeline.habit_memory import bayesian_blend_habit

            is_actor = frame["is_actor"]
            actor_x, actor_y = player_pos[is_actor][0].tolist()
            blended_x, blended_y = bayesian_blend_habit(
                (actor_x, actor_y), habit_heatmaps[actor_player_id]
            )
            player_pos = player_pos.clone()  # never mutate the caller's frame in place
            player_pos[is_actor] = torch.tensor(
                [blended_x, blended_y], dtype=player_pos.dtype
            )

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

    # Under the "attacking team moves toward increasing x" convention, the
    # defending team's own goal is at x=0 (not x=100, which is the goal the
    # ATTACKING team is trying to score in). "The defensive line" is the
    # most advanced (highest-x) defender -- how far up the pitch, away from
    # their own goal, the defense has pushed -- and "space behind the
    # line" is the tactically exploitable gap between that line and their
    # own goal at x=0, i.e. cells with x < highest_defending_x. (A HIGHER
    # defensive line means MORE exploitable space behind it, matching the
    # standard football-tactics reading of a "high line" as riskier.)
    defending_positions = player_pos[defending_mask]
    if defending_positions.shape[0] > 0:
        highest_defending_x = defending_positions[:, 0].max()
    else:
        # No visible defenders in this frame: there is no line at all, so
        # treat the entire pitch up to the attacking end as exploitable
        # (maximal vulnerability) rather than reporting zero space behind a
        # nonexistent line. def_control is already all-zero in this branch.
        highest_defending_x = torch.tensor(PITCH_LENGTH)

    behind_line_mask = active_coords[:, 0] < highest_defending_x
    space_behind_defending_line = (1.0 - def_control[behind_line_mask]).sum()

    return {
        "attacking_control_near_ball": attacking_control_near_ball.item(),
        "defending_control_near_ball": defending_control_near_ball.item(),
        "attacking_control_final_third": attacking_control_final_third.item(),
        "space_behind_defending_line": space_behind_defending_line.item(),
    }
