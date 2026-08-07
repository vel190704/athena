"""Press Resistance Index validation (additive new feature): the new
`statsbomb_io.event_is_under_pressure` field-access helper and
`player_report.generate_player_press_resistance_index` aggregation.

Deliberately end-to-end against REAL, already-cached match data, same
discipline as test_player_dashboard.py -- reuses the SAME MESSI_PLAYER_ID
and MESSI_ARGENTINA_POLAND_MATCH_ID those files already use, so no new
network fetch is needed here.

ADR-021 condition 2: this feature is UNCONDITIONALLY served (not gated by
PUBLIC_DEPLOYMENT) -- see player_report.generate_player_press_resistance_index's
own docstring and the ADR-021 addendum for the full exemption reasoning.
There is therefore no raw/aggregated split to test here, unlike
test_player_dashboard.py's touch-map/timeline pairs.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import json

from production.src.ingestion.statsbomb_io import event_is_under_pressure, fetch_match_events
from production.src.reporting.player_report import (
    MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI,
    generate_player_press_resistance_index,
)

MESSI_PLAYER_ID = 5503
MESSI_ARGENTINA_POLAND_MATCH_ID = 3857264  # real, verified: 4,052 total events

# Real, verified low-sample case: Gonzalo Castro Irizábal (Málaga), a
# single real cached match, only 9 real under-pressure Pass/Dribble/Shot
# events total -- well under MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI.
LOW_SAMPLE_PLAYER_ID = 6740
LOW_SAMPLE_MATCH_ID = 265894


# --- event_is_under_pressure (statsbomb_io.py) ----------------------------


def test_event_is_under_pressure_real_data():
    """VERIFIED against match 3857264's real 4,052 cached events: the
    `under_pressure` key is present on 468 of them, and every single one
    of those 468 carries the value True -- zero explicit False observed.
    Key absence (the other 3,584 events) means "not under pressure," not
    an explicit False -- this is the real structural finding this helper's
    `event.get("under_pressure", False)` default depends on."""
    events = fetch_match_events(MESSI_ARGENTINA_POLAND_MATCH_ID)
    assert len(events) == 4052

    flagged = [e for e in events if event_is_under_pressure(e)]
    assert len(flagged) == 468

    has_key_true = sum(1 for e in events if e.get("under_pressure") is True)
    has_key_false = sum(1 for e in events if e.get("under_pressure") is False)
    assert has_key_true == 468
    assert has_key_false == 0


def test_event_is_under_pressure_absent_key_is_false():
    assert event_is_under_pressure({"type": {"name": "Pass"}}) is False


# --- generate_player_press_resistance_index (player_report.py) -----------


def test_generate_player_press_resistance_index_real_data():
    """Messi, real match 3857264: real per-event-type under-pressure
    attempt/success counts -- not placeholders. Cross-checked directly
    against the real cached event JSON below before trusting these
    numbers (same discipline test_player_dashboard.py's own real-count
    assertions already established)."""
    events = fetch_match_events(MESSI_ARGENTINA_POLAND_MATCH_ID)
    messi_under_pressure = [
        e for e in events
        if event_is_under_pressure(e) and e.get("player", {}).get("id") == MESSI_PLAYER_ID
    ]
    by_type = {}
    for e in messi_under_pressure:
        by_type.setdefault(e["type"]["name"], []).append(e)
    assert len(by_type.get("Pass", [])) == 6
    assert len(by_type.get("Dribble", [])) == 5
    assert len(by_type.get("Shot", [])) == 1

    pri = generate_player_press_resistance_index(MESSI_PLAYER_ID, [MESSI_ARGENTINA_POLAND_MATCH_ID])

    assert pri["player_id"] == MESSI_PLAYER_ID
    assert pri["matches_requested"] == 1
    assert pri["matches_with_data"] == 1

    assert pri["event_types"]["pass"]["under_pressure_attempts"] == 6
    assert pri["event_types"]["pass"]["successful_under_pressure"] == 5
    assert pri["event_types"]["dribble"]["under_pressure_attempts"] == 5
    assert pri["event_types"]["dribble"]["successful_under_pressure"] == 5
    assert pri["event_types"]["shot"]["under_pressure_attempts"] == 1
    assert pri["event_types"]["shot"]["successful_under_pressure"] == 0

    assert pri["overall"]["under_pressure_attempts"] == 12
    assert pri["overall"]["successful_under_pressure"] == 10
    assert abs(pri["overall"]["success_rate"] - (10 / 12)) < 1e-9

    # 12 real attempts < MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI (20):
    # correctly flagged even though every number above is real, not fake.
    assert pri["press_resistance_index_used_low_sample_flag"] is True


def test_generate_player_press_resistance_index_dribble_success_convention_differs_from_pass():
    """Real-data confirmation of Step 0's core finding: Dribble's outcome
    key is present on every real Dribble event (unlike Pass's key-absence-
    means-complete convention), so a Dribble-under-pressure success check
    must test `outcome.name == "Complete"` explicitly. Directly verifies
    every real under-pressure Dribble event in this match carries an
    explicit outcome, and that the aggregation's dribble success count
    matches counting `outcome.name == "Complete"` by hand."""
    events = fetch_match_events(MESSI_ARGENTINA_POLAND_MATCH_ID)
    messi_dribbles_under_pressure = [
        e for e in events
        if event_is_under_pressure(e)
        and e.get("player", {}).get("id") == MESSI_PLAYER_ID
        and e["type"]["name"] == "Dribble"
    ]
    assert len(messi_dribbles_under_pressure) == 5
    for e in messi_dribbles_under_pressure:
        assert "outcome" in e.get("dribble", {}), "every real Dribble event carries an outcome key"

    hand_counted_successes = sum(
        1 for e in messi_dribbles_under_pressure if e["dribble"]["outcome"]["name"] == "Complete"
    )
    pri = generate_player_press_resistance_index(MESSI_PLAYER_ID, [MESSI_ARGENTINA_POLAND_MATCH_ID])
    assert pri["event_types"]["dribble"]["successful_under_pressure"] == hand_counted_successes


def test_generate_player_press_resistance_index_no_data_match_skipped_not_fabricated():
    pri = generate_player_press_resistance_index(MESSI_PLAYER_ID, [999999999])
    assert pri["matches_requested"] == 1
    assert pri["matches_with_data"] == 0
    assert pri["overall"]["under_pressure_attempts"] == 0
    assert pri["overall"]["success_rate"] is None
    assert pri["press_resistance_index_used_low_sample_flag"] is True


def test_generate_player_press_resistance_index_low_sample_flag_real_low_event_player():
    """Real low-sample case (Gonzalo Castro Irizábal, Malaga vs Barcelona,
    match 265894): 9 real under-pressure Pass/Dribble/Shot events total,
    correctly flagged -- same false-positive/false-negative rigor as the
    original Milestone 44 audit, applied to a real player rather than a
    synthetic one."""
    pri = generate_player_press_resistance_index(LOW_SAMPLE_PLAYER_ID, [LOW_SAMPLE_MATCH_ID])

    assert pri["overall"]["under_pressure_attempts"] == 9
    assert pri["overall"]["under_pressure_attempts"] < MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI
    assert pri["press_resistance_index_used_low_sample_flag"] is True


def test_generate_player_press_resistance_index_low_sample_flag_false_for_well_sampled_real_player():
    """Negative case for the same flag (false-positive check): Messi's
    real cached matches carry far more than
    MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI under-pressure actions
    even across a small multi-match slice, so the flag must NOT fire."""
    messi_matches = [3773386, 3857264, 3857289, 3857300, 3869151, 3869321, 3869519, 3869685]
    pri = generate_player_press_resistance_index(MESSI_PLAYER_ID, messi_matches)

    assert pri["overall"]["under_pressure_attempts"] >= MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI
    assert pri["press_resistance_index_used_low_sample_flag"] is False


def test_generate_player_press_resistance_index_no_location_or_minute_leaked():
    """The actual ADR-021 exemption guarantee (see the ADR-021 addendum):
    no per-event field (location, minute, event id) survives anywhere in
    this function's return value -- only per-event-type/overall counts
    and rates."""
    pri = generate_player_press_resistance_index(MESSI_PLAYER_ID, [MESSI_ARGENTINA_POLAND_MATCH_ID])
    serialized = json.dumps(pri)
    assert "location" not in serialized
    assert "\"minute\"" not in serialized
