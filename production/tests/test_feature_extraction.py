"""Milestone 3/9 validation: real StatsBomb 360 data through the pitch-control
feature extraction pipeline.

Per ADR-009, no direction inference or coordinate flip is applied -- both
periods are used as-is, since StatsBomb's raw coordinates are already
oriented relative to the acting team's own attacking-left-to-right
perspective.

Requires network access to raw.githubusercontent.com on first run; every
subsequent run reads from the data/raw/ cache (see statsbomb_io.py).
"""

import math

import torch

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    find_valid_match_id,
    parse_360_frame,
)
from production.src.pipeline.feature_extractor import extract_features

# 999999999 is a deliberately invalid match_id so find_valid_match_id has to
# actually skip a candidate rather than trivially succeeding on the first
# try. The rest are real StatsBomb open-data World Cup 2022 match IDs known
# to carry both events and 360 data as of this writing.
CANDIDATE_MATCH_IDS = [999999999, 3857276, 3857271, 3857296]

EXPECTED_FEATURE_KEYS = {
    "attacking_control_near_ball",
    "defending_control_near_ball",
    "attacking_control_final_third",
    "space_behind_defending_line",
}


def _first_n_passes_with_360(match_id: int, n: int = 5, period: int | None = None):
    """First `n` Pass events with an associated 360 frame. `period=None`
    does not restrict to a specific period -- both periods are valid per
    ADR-009.
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    matched = []
    for event in events:
        if period is not None and event["period"] != period:
            continue
        if event["type"]["name"] != "Pass":
            continue
        if "location" not in event:
            continue
        frame = frames_by_event_uuid.get(event["id"])
        if frame is None:
            continue
        matched.append((event, frame))
        if len(matched) == n:
            break
    return matched


def test_feature_extraction_on_real_statsbomb_data():
    match_id = find_valid_match_id(CANDIDATE_MATCH_IDS)

    pairs = _first_n_passes_with_360(match_id, n=5)
    assert len(pairs) == 5, "expected at least 5 Pass events with 360 frames"

    printed_one = False
    for event, frame_data in pairs:
        frame = parse_360_frame(event, frame_data)

        features = extract_features(frame)

        assert set(features.keys()) == EXPECTED_FEATURE_KEYS
        for key, value in features.items():
            assert math.isfinite(value), f"{key} was not finite: {value}"

        if not printed_one:
            print(f"\nSample feature dict (match_id={match_id}):\n{features}")
            printed_one = True


def test_feature_extraction_covers_period_2():
    """Period 2 is not excluded (ADR-009: no direction correction needed,
    so there's no exclusion path left for it to hit). Finds a real
    period-2 Pass with a 360 frame and confirms it produces a full, finite
    feature dict.
    """
    match_id = find_valid_match_id(CANDIDATE_MATCH_IDS)

    period2_pairs = _first_n_passes_with_360(match_id, n=1, period=2)
    assert len(period2_pairs) == 1, "expected at least one period-2 Pass event with a 360 frame"

    event, frame_data = period2_pairs[0]
    frame = parse_360_frame(event, frame_data)
    assert frame["period"] == 2

    features = extract_features(frame)

    assert set(features.keys()) == EXPECTED_FEATURE_KEYS
    for key, value in features.items():
        assert math.isfinite(value), f"{key} was not finite: {value}"

    print(f"\nPeriod-2 sample feature dict (match_id={match_id}, team={frame['team']}): {features}")


def test_space_behind_defending_line_uses_x0_as_defending_own_goal():
    """Under the 'attacking team moves toward increasing x' convention, the
    defending team's own goal is at x=0, so a HIGHER (more advanced,
    bigger-x) defensive line should mean MORE exploitable space behind it,
    not less. Two synthetic frames, identical except for the defending
    line's depth, confirm this: the high-line scenario must report
    strictly more space_behind_defending_line than the low-line one.
    """
    # Ball near midfield so BOTH candidate defender-line positions below
    # fall within BiomechanicalPitchControl's 30m active-cell mask (ADR-005)
    # -- a defender far outside that radius would leave behind_line_mask
    # matching zero active cells regardless of its true position, making
    # the comparison vacuous (0.0 == 0.0) rather than a real test.
    # space_behind_defending_line depends only on the DEFENDING sub-batch
    # and ball_pos, so the attacking player's exact position doesn't affect
    # it -- it only needs to exist so the frame has a valid attacking side.
    attacking_player = torch.tensor([[50.0, 34.0]])
    attacking_vel = torch.zeros(1, 2)

    def _make_frame(defender_x: float) -> dict:
        return {
            "ball_pos": torch.tensor([50.0, 34.0]),
            "player_pos": torch.cat([attacking_player, torch.tensor([[defender_x, 34.0]])]),
            "player_vel": torch.cat([attacking_vel, torch.zeros(1, 2)]),
            "fatigue_mod": torch.ones(2),
            "is_teammate": torch.tensor([True, False]),
            "event_type": "Pass",
            "period": 1,
            "team": "TestTeam",
        }

    low_line_features = extract_features(_make_frame(defender_x=30.0))
    high_line_features = extract_features(_make_frame(defender_x=45.0))

    assert (
        high_line_features["space_behind_defending_line"]
        > low_line_features["space_behind_defending_line"]
    )
