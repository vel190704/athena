"""Non-physics baseline feature extraction for the RQ1 ablation.

RQ1's stated success criterion ("Brier Score improvement >= X% [over a
non-physics baseline]") implied a comparison against a non-physics-informed
feature set, but no such ablation was ever built or trained anywhere in this
project's history (see RESEARCH_FINDINGS.md's RQ1 "Caveats" section). This
module builds that missing "dumb baseline" feature path: additive only, and
deliberately kept separate from feature_extractor.py rather than folded into
it, so the physics-informed pipeline used by every existing model is
untouched.

Produces a comparable-shaped (4-scalar) feature vector using simple
aggregate statistics of RAW pre-physics signal only -- no
BiomechanicalPitchControl, no control-probability field, no pitch-grid
integration. Where feature_extractor.py's extract_features() aggregates
control PROBABILITY over active pitch-grid CELLS (near-ball, final-third,
behind-the-line masks applied to `active_coords`, the grid), this module
aggregates simple PLAYER COUNTS/DISTANCES over the same three geometric
regions applied directly to raw player positions -- deliberately simpler,
and never invokes the physics engine.

Known, disclosed limitation of the "raw" signal available here:
statsbomb_io.parse_360_frame always sets `player_vel` to all-zero and
`fatigue_mod` to a constant 1.0 (StatsBomb's public 360 data carries no
velocity field at all -- see that function's own comment). Both are
zero-information constants PROJECT-WIDE, not just for this baseline, so
using them here would not make this baseline meaningfully richer, just
add noise-free constant columns. The only genuinely informative raw signal
before BiomechanicalPitchControl's ODE runs is `player_pos`, `ball_pos`,
and `is_teammate` -- this module uses exactly those three and nothing else.
"""

import torch

from production.src.pipeline.feature_extractor import FINAL_THIRD_X, NEAR_BALL_RADIUS

# Fixed key order for flattening this module's feature dict into a tensor,
# mirroring survival_dataset.py's FEATURE_KEYS pattern. Deliberately NOT the
# same names as the physics FEATURE_KEYS -- these are different quantities
# (raw counts/distances, not control probabilities) and naming them
# identically would misrepresent what was actually measured.
BASELINE_FEATURE_KEYS = (
    "teammates_near_ball_count",
    "opponents_near_ball_count",
    "teammates_in_final_third_count",
    "raw_space_behind_defending_line",
)


def extract_naive_baseline_features(frame: dict) -> dict:
    """Simple aggregate statistics of RAW player/ball positions for one
    parsed 360 frame (statsbomb_io.parse_360_frame's output). Mirrors
    feature_extractor.extract_features() in shape (4 scalars) and in which
    three geometric regions it looks at (near-ball, final third, behind the
    defensive line) -- using feature_extractor.py's own NEAR_BALL_RADIUS/
    FINAL_THIRD_X constants so both feature sets are judged against the
    same pitch-geometry thresholds -- so the comparison isolates "physics-
    informed vs. not," not "looks at a different part of the pitch."

    No BiomechanicalPitchControl call, no ODE, no pitch grid: every value
    below is a raw count or raw Euclidean distance over the visible
    players' actual positions.
    """
    ball_pos = frame["ball_pos"]
    player_pos = frame["player_pos"]
    is_teammate = frame["is_teammate"]

    dist_to_ball = torch.linalg.norm(player_pos - ball_pos.unsqueeze(0), dim=-1)
    near_ball_mask = dist_to_ball <= NEAR_BALL_RADIUS
    final_third_mask = player_pos[:, 0] > FINAL_THIRD_X

    teammates_near_ball_count = (near_ball_mask & is_teammate).sum()
    opponents_near_ball_count = (near_ball_mask & ~is_teammate).sum()
    teammates_in_final_third_count = (final_third_mask & is_teammate).sum()

    # Same "defensive line" convention as feature_extractor.py: the
    # defending team's own goal is at x=0, the line is the highest-x
    # (most advanced) defender, and "space behind it" is the exploitable
    # gap between that line and x=0. Here, "space" is a raw geometric
    # proxy -- summed nearest-defender distance for every player behind
    # the line -- NOT the physics feature's 1-minus-control-probability
    # integral over grid cells.
    defending_positions = player_pos[~is_teammate]
    if defending_positions.shape[0] > 0:
        highest_defending_x = defending_positions[:, 0].max()
        behind_line_mask = player_pos[:, 0] < highest_defending_x
        behind_positions = player_pos[behind_line_mask]
        if behind_positions.shape[0] > 0:
            pairwise_dist = torch.cdist(behind_positions, defending_positions)
            raw_space_behind_defending_line = pairwise_dist.min(dim=-1).values.sum()
        else:
            raw_space_behind_defending_line = torch.tensor(0.0)
    else:
        # No visible defenders: same "maximal vulnerability" convention as
        # feature_extractor.py, expressed here as zero (no defender to
        # measure distance from at all) rather than an arbitrary constant.
        raw_space_behind_defending_line = torch.tensor(0.0)

    return {
        "teammates_near_ball_count": teammates_near_ball_count.item(),
        "opponents_near_ball_count": opponents_near_ball_count.item(),
        "teammates_in_final_third_count": teammates_in_final_third_count.item(),
        "raw_space_behind_defending_line": raw_space_behind_defending_line.item(),
    }
