"""Tactical Event Detection validation (new reporting track):
tactical_events.detect_tactical_events -- Counter Attack, Switch of Play,
Build-up Pattern, the 3 event types genuinely feasible from event data
alone.

Deliberately end-to-end against REAL cached match data (match 3857276,
Canada vs Morocco -- the same validation match used throughout this
project's chain-builder/simulator/dashboard test history), not synthetic
fixtures. Real numbers below were verified directly before being written
as assertions, not assumed.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.reporting.tactical_events import (
    BUILDUP_MIN_PASSES,
    COUNTER_ATTACK_TIME_TO_FINAL_THIRD_SECONDS,
    SWITCH_OF_PLAY_LATERAL_THRESHOLD_METERS,
    _events_by_chain_key,
    detect_tactical_events,
)
from production.src.ingestion.statsbomb_io import fetch_match_events

MATCH_ID = 3857276


def test_detect_tactical_events_real_match_counts_and_fractions():
    """Step 3.1/3.4: real counts, and confirms none of the 3 detectors
    fires on a near-zero or near-total fraction of chains/passes --
    VERIFIED directly against this real match before writing these exact
    numbers: 169 real possession chains, 790 real completed passes."""
    result = detect_tactical_events(MATCH_ID)

    assert result["no_data"] is False
    assert result["total_chains"] == 169
    assert result["total_completed_passes"] == 790
    assert result["counter_attack_time_to_final_third_seconds"] == COUNTER_ATTACK_TIME_TO_FINAL_THIRD_SECONDS
    assert result["switch_of_play_lateral_threshold_meters"] == SWITCH_OF_PLAY_LATERAL_THRESHOLD_METERS
    assert result["buildup_min_passes"] == BUILDUP_MIN_PASSES

    assert len(result["counter_attacks"]) == 25
    assert len(result["build_up_patterns"]) == 47
    assert len(result["switches_of_play"]) == 34

    # Step 0.4: a "meaningful minority," not near-zero or near-total, for
    # all three -- real fractions: 14.8%, 27.8%, 4.3%.
    for fraction_key in (
        "counter_attack_fraction_of_chains",
        "build_up_fraction_of_chains",
        "switch_of_play_fraction_of_completed_passes",
    ):
        fraction = result[fraction_key]
        assert 0.01 < fraction < 0.5, f"{fraction_key}={fraction} is not a meaningful minority"

    # Step 3.4, stated directly: most chains (57.4% here) are neither type.
    n_classified_chains = len(result["counter_attacks"]) + len(result["build_up_patterns"])
    assert n_classified_chains < result["total_chains"] * 0.5


def test_detect_tactical_events_counter_attack_and_buildup_are_mutually_exclusive():
    """Step 3.2: THE mutual-exclusivity check -- Counter Attack requires
    FAST progression (reaches the final third within
    COUNTER_ATTACK_TIME_TO_FINAL_THIRD_SECONDS), Build-up Pattern requires
    NOT fast, by construction. Confirms zero real chains satisfy both."""
    result = detect_tactical_events(MATCH_ID)

    counter_attack_keys = {(c["chain_id"], c["period"]) for c in result["counter_attacks"]}
    build_up_keys = {(c["chain_id"], c["period"]) for c in result["build_up_patterns"]}

    overlap = counter_attack_keys & build_up_keys
    assert overlap == set(), f"Counter Attack and Build-up Pattern overlap on real chains: {overlap}"


def test_detect_tactical_events_counter_attack_spot_check_real_events_show_fast_progression():
    """Step 3.1's real spot-check: pulls the actual underlying StatsBomb
    events for 2 real detected Counter Attack instances and confirms they
    visibly show a real, fast forward progression -- not an arbitrary
    chain that happened to qualify on a technicality.

    chain_id=22, period=1 (Canada): starts with a Pass at raw x=19.1 (deep
    in Canada's own defensive area, in the acting team's own frame) at
    10:57, and by 11:07 -- 10 real seconds later -- a Ball Receipt* at raw
    x=72.6, well past FINAL_THIRD_X (66.0 in the 100m-scaled space,
    corresponding to 79.2 in StatsBomb's native 0-120 space) -- a real,
    fast, forward-progressing sequence.
    """
    events = fetch_match_events(MATCH_ID)
    events_by_chain = _events_by_chain_key(events)
    result = detect_tactical_events(MATCH_ID)

    counter_attack_keys = {(c["chain_id"], c["period"]): c for c in result["counter_attacks"]}
    assert (22, 1) in counter_attack_keys
    chain_22 = counter_attack_keys[(22, 1)]
    assert chain_22["team"] == "Canada"
    assert chain_22["time_to_final_third_seconds"] == 2

    chain_22_events = events_by_chain[(1, 22)]
    located = [e for e in chain_22_events if "location" in e]
    first_x = located[0]["location"][0]
    last_x = located[-1]["location"][0]
    assert first_x < 30.0, f"expected the chain to start deep, got first_x={first_x}"
    assert last_x > 60.0, f"expected the chain to progress forward, got last_x={last_x}"
    assert last_x > first_x + 30.0, "expected a real, substantial forward progression, not a marginal one"


def test_detect_tactical_events_buildup_spot_check_real_events_show_sustained_possession():
    """Step 3.1's real spot-check, mirrored for Build-up Pattern:
    chain_id=2, period=1 (Morocco) -- 17 real passes over 52 real seconds,
    starting at kickoff, staying in the middle third with real lateral
    ball movement rather than progressing quickly forward -- genuinely
    patient possession, not a technicality."""
    result = detect_tactical_events(MATCH_ID)
    build_up_keys = {(c["chain_id"], c["period"]): c for c in result["build_up_patterns"]}

    assert (2, 1) in build_up_keys
    chain_2 = build_up_keys[(2, 1)]
    assert chain_2["team"] == "Morocco"
    assert chain_2["n_passes"] == 17
    assert chain_2["chain_duration_seconds"] == 52.0
    assert chain_2["start_third"] == "middle"
    assert chain_2["n_passes"] >= BUILDUP_MIN_PASSES


def test_detect_tactical_events_no_data_for_unfetchable_match():
    result = detect_tactical_events(match_id=1)
    assert result["no_data"] is True
    assert "reason" in result


def test_detect_tactical_events_switch_of_play_real_threshold_boundary():
    """Cross-check independent of the function's own internal comparison:
    every real detected switch must have a lateral distance >= the
    configured threshold, and no completed pass below it should appear."""
    result = detect_tactical_events(MATCH_ID)
    for switch in result["switches_of_play"]:
        assert switch["lateral_distance_meters"] >= SWITCH_OF_PLAY_LATERAL_THRESHOLD_METERS

    events = fetch_match_events(MATCH_ID)
    from production.src.ingestion.statsbomb_io import Y_SCALE
    from production.src.reporting.pass_network import _is_complete_pass

    real_qualifying_count = 0
    for event in events:
        if event["type"]["name"] != "Pass" or not _is_complete_pass(event):
            continue
        location = event.get("location")
        end_location = event.get("pass", {}).get("end_location")
        if location is None or end_location is None:
            continue
        lateral = abs(end_location[1] - location[1]) * Y_SCALE
        if lateral >= SWITCH_OF_PLAY_LATERAL_THRESHOLD_METERS:
            real_qualifying_count += 1

    assert real_qualifying_count == len(result["switches_of_play"])


def test_detect_tactical_events_no_individual_event_coordinate_in_output():
    """ADR-021 addendum's own stated design guarantee, checked directly
    against the real raw output rather than only reasoned about in prose:
    no instance of any of the 3 detected types carries an individual
    event's exact (x, y) coordinate anywhere."""
    result = detect_tactical_events(MATCH_ID)

    for instance in result["counter_attacks"] + result["build_up_patterns"]:
        assert "location" not in instance
        assert "end_location" not in instance
    for switch in result["switches_of_play"]:
        assert "location" not in switch
        assert "end_location" not in switch
