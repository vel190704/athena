"""Tactical Timeline UI validation (new reporting track, capstone):
match_timeline.generate_match_timeline -- unifies Weak-Spot Lifetime
Analysis and Tactical Event Detection onto one shared, chronologically-
sorted, period-boundary-corrected time axis.

Deliberately end-to-end against REAL cached match data (match 3857276,
Canada vs Morocco -- the same validation match used throughout this
project's own reporting-track test history). Real numbers below were
verified directly before being written as assertions, not assumed.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.reporting.match_timeline import (
    _PERIOD_2_NOMINAL_START_MINUTE,
    _display_minute,
    _period_1_max_minute,
    generate_match_timeline,
)

MATCH_ID = 3857276


def test_generate_match_timeline_real_match_shape_and_scope_note():
    """Step 0's scope decision, checked directly against the real
    response, not just documented in prose: momentum/segmentation are
    explicitly marked out of scope, and every timeline entry belongs to
    one of the 2 in-scope signal families."""
    result = generate_match_timeline(MATCH_ID)

    assert result["no_data"] is False
    assert result["teams"] == sorted(result["teams"])
    assert result["momentum_segmentation_in_scope"] is False
    assert "momentum_segmentation_note" in result and result["momentum_segmentation_note"]

    signals_present = {e["signal"] for e in result["timeline_entries"]}
    assert signals_present <= {"weak_spot", "counter_attack", "build_up", "switch_of_play"}
    assert "momentum" not in signals_present and "segmentation" not in signals_present


def test_generate_match_timeline_real_match_counts_match_source_functions():
    """The timeline's own per-signal counts must equal what
    generate_weak_spot_lifetime_analysis/detect_tactical_events -- called
    directly, independently -- actually return for the same real match,
    confirming the assembly step doesn't drop, duplicate, or silently
    filter anything from its own two source functions."""
    from production.src.reporting.tactical_events import detect_tactical_events
    from production.src.reporting.team_report import generate_weak_spot_lifetime_analysis

    result = generate_match_timeline(MATCH_ID)
    teams = result["teams"]

    expected_weak_spots = sum(
        len(generate_weak_spot_lifetime_analysis(team, MATCH_ID)["weak_spot_instances"]) for team in teams
    )
    tactical = detect_tactical_events(MATCH_ID)

    assert result["weak_spot_instance_count"] == expected_weak_spots
    assert result["counter_attack_count"] == len(tactical["counter_attacks"])
    assert result["build_up_count"] == len(tactical["build_up_patterns"])
    assert result["switch_of_play_count"] == len(tactical["switches_of_play"])

    n_by_signal = {"weak_spot": 0, "counter_attack": 0, "build_up": 0, "switch_of_play": 0}
    for entry in result["timeline_entries"]:
        n_by_signal[entry["signal"]] += 1
    assert n_by_signal["weak_spot"] == expected_weak_spots
    assert n_by_signal["counter_attack"] == len(tactical["counter_attacks"])
    assert n_by_signal["build_up"] == len(tactical["build_up_patterns"])
    assert n_by_signal["switch_of_play"] == len(tactical["switches_of_play"])


def test_generate_match_timeline_entries_are_chronologically_sorted():
    result = generate_match_timeline(MATCH_ID)
    display_starts = [e["display_start_minute"] for e in result["timeline_entries"]]
    assert display_starts == sorted(display_starts)


def test_generate_match_timeline_no_entry_has_end_before_start():
    """A real, direct sanity check on every single real entry -- not just
    a sample -- since a period-boundary-offset bug would most plausibly
    show up as an inverted span for SOME entries, not all."""
    result = generate_match_timeline(MATCH_ID)
    for entry in result["timeline_entries"]:
        assert entry["display_end_minute"] >= entry["display_start_minute"], entry


def test_generate_match_timeline_period_boundary_real_offset_and_no_overlap():
    """Step 3.2: the real period-boundary handling. VERIFIED directly
    against this real match before writing these numbers: period 1's
    real observed max minute is 50.0 (includes real first-half stoppage
    time), so the computed period_2_display_offset must be 50.0 - 45.0 =
    5.0 -- NOT assumed to be 0 (which would silently overlap the two
    periods on the display axis, since period 2's raw minute values
    start back at 45, inside period 1's own real 0-50 range)."""
    result = generate_match_timeline(MATCH_ID)

    assert result["period_1_max_minute"] == 50.0
    assert result["period_2_display_offset"] == 5.0

    period_1_entries = [e for e in result["timeline_entries"] if e["period"] == 1]
    period_2_entries = [e for e in result["timeline_entries"] if e["period"] == 2]
    assert period_1_entries and period_2_entries

    max_period_1_display = max(e["display_end_minute"] for e in period_1_entries)
    min_period_2_display = min(e["display_start_minute"] for e in period_2_entries)
    assert min_period_2_display >= max_period_1_display - 1e-6, (
        "a period-2 entry displays BEFORE a period-1 entry ends -- the half-time boundary is not "
        "correctly handled"
    )

    # Cross-check the raw StatsBomb minute-doesn't-reset gotcha is REAL for
    # this match, not hypothetical: period 2's own raw minute values
    # genuinely fall inside period 1's own raw range.
    period_2_raw_starts = [e["start_minute"] for e in period_2_entries]
    assert min(period_2_raw_starts) < result["period_1_max_minute"], (
        "expected a real period-2 raw minute value to fall inside period 1's own raw range -- "
        "if this no longer holds, the test match's own data changed"
    )


def test_display_minute_helper_period_1_unchanged_period_2_offset():
    assert _display_minute(10.0, period=1, period_2_display_offset=5.0) == 10.0
    assert _display_minute(45.0, period=2, period_2_display_offset=5.0) == 50.0
    assert _display_minute(93.0, period=2, period_2_display_offset=5.0) == 98.0


def test_period_1_max_minute_computed_not_assumed():
    """Direct unit test of the real-data-computed offset anchor, using a
    small constructed event list -- confirms the function reports the
    REAL observed maximum, not the nominal 45.0, when period 1 genuinely
    ran longer (real stoppage time)."""
    events = [
        {"period": 1, "minute": 10},
        {"period": 1, "minute": 47},  # real stoppage time
        {"period": 2, "minute": 45},
        {"period": 2, "minute": 90},
    ]
    assert _period_1_max_minute(events) == 47.0
    assert _PERIOD_2_NOMINAL_START_MINUTE == 45.0


def test_generate_match_timeline_real_overlap_between_counter_attack_and_weak_spot():
    """Step 3.1's real spot-check: confirms a real, football-plausible
    temporal overlap actually exists between a Counter Attack instance
    and opponent weak-spot instances -- not merely that the data
    STRUCTURALLY could overlap, but that it REALLY does for this match.

    Canada's real Counter Attack (chain_id=22, period 1, display minutes
    10.0-11.0 -- the same real fast transition already spot-checked in
    test_tactical_events.py) temporally overlaps 6 real Morocco weak-spot
    instances in that same window, a genuinely coherent picture: Canada's
    fast transition coincided with real gaps in Morocco's own defensive
    shape.
    """
    result = generate_match_timeline(MATCH_ID)
    entries = result["timeline_entries"]

    canada_ca = next(
        e for e in entries
        if e["signal"] == "counter_attack" and e["team"] == "Canada" and e["detail"]["chain_id"] == 22
    )
    assert canada_ca["display_start_minute"] == 10.0
    assert canada_ca["display_end_minute"] == 11.0

    overlapping_opponent_weak_spots = [
        e for e in entries
        if e["signal"] == "weak_spot"
        and e["team"] != canada_ca["team"]
        and e["display_end_minute"] > e["display_start_minute"]  # a real, non-instantaneous span
        and e["display_start_minute"] <= canada_ca["display_end_minute"]
        and e["display_end_minute"] >= canada_ca["display_start_minute"]
    ]
    assert len(overlapping_opponent_weak_spots) == 6
    assert all(e["team"] == "Morocco" for e in overlapping_opponent_weak_spots)


def test_generate_match_timeline_no_data_for_unfetchable_match():
    result = generate_match_timeline(match_id=1)
    assert result["no_data"] is True
    assert "reason" in result
