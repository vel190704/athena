"""Milestone 30 (Module 4): the CV-to-physics adapter layer.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25. This module IMPORTS `PITCH_LENGTH`/`PITCH_WIDTH` from
`production/src/pipeline/feature_extractor.py` (read-only reuse of the
already-established 100x68m pitch-space constants, ADR-002) but does not
modify that file or anything else in `production/src/models`,
`production/src/pipeline`, `production/src/spatial`, `production/src/
physics`, or `production/src/serving`.

PURPOSE: bridges Milestones 25-29's CV outputs (tracked pixel positions,
team labels, calibrated homography) into the EXACT tensor contract
`production/src/pipeline/feature_extractor.py`'s `extract_features` has
consumed since Milestone 3: `{"ball_pos", "player_pos", "player_vel",
"fatigue_mod", "is_teammate"}`. `fatigue_mod` is included even though it
is not itself a CV output -- `extract_features` reads it unconditionally
(`frame["fatigue_mod"]`) and would raise `KeyError` without it. CV data
carries no fatigue signal, exactly like StatsBomb 360 freeze-frames carry
none (`parse_360_frame`'s own `fatigue_mod = torch.ones(...)` default) --
this module uses the SAME "no signal available, neutral multiplier"
convention, for the same reason.
"""

import numpy as np
import torch

from production.src.cv.calibration import transform_points
from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH

# Legitimate players near the touchline (e.g. taking a throw-in) can
# transform marginally outside strict [0,100]x[0,68] bounds due to
# calibration imprecision -- this tolerance keeps them, while still
# filtering out points genuinely far outside (crowd, coaching staff).
PITCH_BOUNDS_TOLERANCE_METERS = 0.5


def convert_frame_to_tensors(
    tracks: list[dict],
    ball_pixel: list[float] | None,
    team_mapping: dict[int, str],
    homography_matrix: np.ndarray,
    fps: float,
    prev_positions_pixel: dict[int, list[float]] | None = None,
    dt_seconds_per_track: dict[int, float] | None = None,
) -> dict | None:
    """Converts one frame's CV outputs into the tensor bundle
    `feature_extractor.extract_features` expects.

    `tracks`: `[{"track_id": int, "pos_pixel": [u, v]}, ...]` for the
    CURRENT frame.
    `ball_pixel`: `[u, v]` or `None` -- per Milestone 29, the ball can
    genuinely go undetected in a frame. If `None`, this function returns
    `None` for the WHOLE bundle (see below) rather than silently reusing a
    stale previous ball position.
    `team_mapping`: `{track_id: "team_A" | "team_B" | "outlier"}`,
    established ONCE per clip (Milestone 28) and looked up here by
    persistent track_id -- NOT recomputed per frame. A `track_id` present
    in `tracks` but absent from `team_mapping` is treated as `"outlier"`
    (excluded), not a `KeyError`.
    `homography_matrix`: from `calibration.compute_homography` (Milestone
    27).
    `fps`: real frame rate (Milestone 26 -- read from the source video,
    never assumed).
    `prev_positions_pixel`: `{track_id: [u, v]}` from the PREVIOUS
    OBSERVATION of that track (not necessarily the immediately-preceding
    frame -- see `dt_seconds_per_track`), or `None` on a clip's first
    frame. Needed for correct velocity -- see below.
    `dt_seconds_per_track` (Milestone 32, optional, additive -- omitting it
    reproduces Milestone 30's original behavior exactly): `{track_id:
    seconds}` overriding the DEFAULT `1/fps` elapsed-time assumption on a
    PER-TRACK basis. This exists because a real orchestrator (the
    Milestone 32 pipeline) may skip whole runs of non-tactical frames
    between two observations of the same track_id -- the true elapsed time
    since a track's previous observation can be several real frames (and
    real seconds), not always exactly one frame. If a track_id has no
    entry here (or this argument is omitted entirely), the default
    `1/fps` is used, i.e. the original Milestone 30 assumption that
    `prev_positions_pixel` always reflects exactly one frame ago.

    CRITICAL -- velocity is computed by transforming CURRENT and PREVIOUS
    pixel positions SEPARATELY into meter space and then differencing,
    NEVER by transforming a pixel-space displacement VECTOR through the
    homography directly. A homography is a PROJECTIVE (nonlinear)
    transform -- the effective pixel-to-meter scale varies across the
    image (Milestone 27 measured a ~2.04x foreshortening ratio between the
    near and far touchlines under a realistic broadcast camera angle).
    Vector-transformation is only correct for a pure affine/linear map; for
    a real homography it silently produces WRONG velocities almost
    everywhere in the frame. If a track has no previous pixel position
    (first frame of a clip, or the track just appeared), its velocity is
    `[0.0, 0.0]` -- an expected "no velocity available yet" case, not an
    error.

    CRITICAL -- `is_teammate` is POSSESSION-AWARE, not a hardcoded team.
    Since Milestone 3, `is_teammate` means "same team as whoever currently
    possesses the ball," because possession changes constantly and a fixed
    team label would silently invert attacking/defending features whenever
    the "wrong" team has the ball. There is no direct possession-event
    signal available from CV alone, so this function uses an explicit
    HEURISTIC: the team whose NEAREST player (by transformed meter-space
    distance) is closest to the ball is treated as the possessing/
    attacking team for this frame. This is an APPROXIMATION of real
    possession -- no CV signal can distinguish "controlling the ball" from
    "merely nearest to it" (e.g. during a loose ball or a tackle in
    progress) -- stated plainly here, not hidden.

    Filtering (in order): `"outlier"`-mapped or unmapped tracks are
    excluded; tracks whose transformed position falls outside
    `[0, 100] x [0, 68]` meters (extended by `PITCH_BOUNDS_TOLERANCE_METERS`
    in each direction) are excluded.

    Returns `{"player_pos": FloatTensor[N,2], "player_vel":
    FloatTensor[N,2], "is_teammate": BoolTensor[N], "ball_pos":
    FloatTensor[2], "fatigue_mod": FloatTensor[N]}` (N = number of
    in-bounds, non-outlier tracks -- possibly 0), or `None` if
    `ball_pixel` is `None`.
    """
    if ball_pixel is None:
        return None

    ball_pos_meters = transform_points(homography_matrix, [ball_pixel])[0]
    default_dt_seconds = 1.0 / fps

    lower_x = -PITCH_BOUNDS_TOLERANCE_METERS
    upper_x = PITCH_LENGTH + PITCH_BOUNDS_TOLERANCE_METERS
    lower_y = -PITCH_BOUNDS_TOLERANCE_METERS
    upper_y = PITCH_WIDTH + PITCH_BOUNDS_TOLERANCE_METERS

    included_positions: list[np.ndarray] = []
    included_velocities: list[np.ndarray] = []
    included_teams: list[str] = []

    for track in tracks:
        track_id = track["track_id"]
        role = team_mapping.get(track_id, "outlier")
        if role == "outlier":
            continue

        pos_meters = transform_points(homography_matrix, [track["pos_pixel"]])[0]
        x, y = pos_meters
        if not (lower_x <= x <= upper_x and lower_y <= y <= upper_y):
            continue

        if prev_positions_pixel is not None and track_id in prev_positions_pixel:
            prev_pos_meters = transform_points(homography_matrix, [prev_positions_pixel[track_id]])[0]
            if dt_seconds_per_track is not None and track_id in dt_seconds_per_track:
                track_dt_seconds = dt_seconds_per_track[track_id]
            else:
                track_dt_seconds = default_dt_seconds
            velocity_meters = (pos_meters - prev_pos_meters) / track_dt_seconds
        else:
            # No previous position for this track (clip's first frame, or
            # the track just appeared) -- "no velocity available yet",
            # not an error.
            velocity_meters = np.array([0.0, 0.0])

        included_positions.append(pos_meters)
        included_velocities.append(velocity_meters)
        included_teams.append(role)

    num_players = len(included_positions)

    if num_players == 0:
        return {
            "player_pos": torch.zeros((0, 2), dtype=torch.float32),
            "player_vel": torch.zeros((0, 2), dtype=torch.float32),
            "is_teammate": torch.zeros((0,), dtype=torch.bool),
            "ball_pos": torch.tensor(ball_pos_meters, dtype=torch.float32),
            "fatigue_mod": torch.zeros((0,), dtype=torch.float32),
        }

    positions_array = np.stack(included_positions)  # [N, 2]
    velocities_array = np.stack(included_velocities)  # [N, 2]

    # Possession heuristic (see docstring): the team of the single nearest
    # player to the ball, in meter space, is treated as possessing/
    # attacking this frame.
    distances_to_ball = np.linalg.norm(positions_array - ball_pos_meters, axis=1)
    nearest_player_index = int(np.argmin(distances_to_ball))
    possessing_team = included_teams[nearest_player_index]

    is_teammate = np.array([team == possessing_team for team in included_teams], dtype=bool)

    return {
        "player_pos": torch.tensor(positions_array, dtype=torch.float32),
        "player_vel": torch.tensor(velocities_array, dtype=torch.float32),
        "is_teammate": torch.tensor(is_teammate, dtype=torch.bool),
        "ball_pos": torch.tensor(ball_pos_meters, dtype=torch.float32),
        "fatigue_mod": torch.ones((num_players,), dtype=torch.float32),
    }
