"""Milestone 3 validation: real StatsBomb 360 data through the pitch-control
feature extraction pipeline.

Requires network access to raw.githubusercontent.com on first run; every
subsequent run reads from the data/raw/ cache (see statsbomb_io.py).
"""

import math

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


def _first_n_period1_passes_with_360(match_id: int, n: int = 5):
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    matched = []
    for event in events:
        if event["period"] != 1:
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

    pairs = _first_n_period1_passes_with_360(match_id, n=5)
    assert len(pairs) == 5, "expected at least 5 period-1 Pass events with 360 frames"

    printed_one = False
    for event, frame_data in pairs:
        frame = parse_360_frame(event, frame_data)
        assert frame["period"] == 1

        features = extract_features(frame)

        assert set(features.keys()) == EXPECTED_FEATURE_KEYS
        for key, value in features.items():
            assert math.isfinite(value), f"{key} was not finite: {value}"

        if not printed_one:
            print(f"\nSample feature dict (match_id={match_id}):\n{features}")
            printed_one = True
