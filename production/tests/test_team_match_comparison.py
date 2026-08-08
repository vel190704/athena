"""Session/Match Comparison validation (additive extension of
team_comparison.py's existing compare_team_seasons): compare_team_matches,
the same general-purpose style-comparison tool at match granularity --
one team, two SPECIFIC matches, not two seasons.

Deliberately end-to-end against REAL, already-cached StatsBomb data, same
discipline as test_team_comparison.py -- reuses real Barcelona match_ids
already fetched elsewhere in this project's own test history, so these
tests run offline. MLFLOW_ALLOW_FILE_STORE is required for the 360-mode
test (it calls team_report.generate_team_report, which lazily loads the
deterministic MLP).
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.reporting.team_comparison import (
    MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON,
    compare_team_matches,
)

# Real Barcelona matches (La Liga 2020/21), verified during this feature's
# own real-data validation: 3773386 (vs Deportivo Alaves, 360-covered,
# 2760 located Barcelona events), 3773497 (also 360-covered), 68314 (vs
# Malaga, NOT 360-covered, 2172 located Barcelona events).
BARCA_MATCH_A = 3773386
BARCA_MATCH_B_NO_360 = 68314
BARCA_MATCH_B_360 = 3773497

# A real match Barcelona did NOT play in (Argentina vs Mexico) -- the
# genuine, real low-sample case this feature's own Step 1 validation
# found: 0 located events for "Barcelona" in this match, not because the
# feature is broken, but because the team genuinely didn't play in it.
ARGENTINA_MEXICO_MATCH_ID = 3857289


def test_compare_team_matches_event_location_mode_real_data():
    """Two real Barcelona matches, one WITHOUT 360 coverage -- must fall
    back to event_location_activity_map for BOTH sides (never mixed
    footing), same discipline compare_team_seasons already applies."""
    result = compare_team_matches("Barcelona", BARCA_MATCH_A, BARCA_MATCH_B_NO_360)

    assert result["team_name"] == "Barcelona"
    assert result["match_id_a"] == BARCA_MATCH_A
    assert result["match_id_b"] == BARCA_MATCH_B_NO_360
    assert result["analysis_mode"] == "event_location_activity_map"

    ra = result["data_richness"]["team_a"]
    rb = result["data_richness"]["team_b"]
    assert ra["located_events"] == 2760
    assert rb["located_events"] == 2172
    assert ra["flag"] == "well-supported"
    assert rb["flag"] == "well-supported"
    assert ra["located_events"] >= MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON
    assert result["reliability_caveat"] is None

    assert result["team_a_total_located_events"] == 2760
    assert result["team_b_total_located_events"] == 2172
    zones_a = result["zone_shares"]["team_a"]
    zones_b = result["zone_shares"]["team_b"]
    assert abs(sum(v for k, v in zones_a.items() if "half" not in k) - 1.0) < 1e-9
    assert abs(sum(v for k, v in zones_b.items() if "half" not in k) - 1.0) < 1e-9

    # Match-qualified labels (not bare "Barcelona" on both sides) --
    # same disambiguation discipline compare_team_seasons's season-
    # qualified labels already established, applied at match granularity.
    assert str(BARCA_MATCH_B_NO_360) in result["summary"] or str(BARCA_MATCH_A) in result["summary"]


def test_compare_team_matches_pitch_control_360_mode_real_data():
    """Two real Barcelona matches, BOTH 360-covered -- must use
    pitch_control_360 mode (team_report.generate_team_report reused
    unmodified for both sides)."""
    result = compare_team_matches("Barcelona", BARCA_MATCH_A, BARCA_MATCH_B_360)

    assert result["analysis_mode"] == "pitch_control_360"
    assert "360 freeze-frame coverage" in result["mode_reason"]

    assert result["team_a_report"]["matches_used"] == 1
    assert result["team_b_report"]["matches_used"] == 1
    assert result["reliability_caveat"] is None
    assert result["data_richness"]["team_a"]["flag"] == "well-supported"
    assert result["data_richness"]["team_b"]["flag"] == "well-supported"


def test_compare_team_matches_low_sample_flag_fires_real_data():
    """Real low-sample case: Barcelona genuinely did not play in the
    Argentina-vs-Mexico match -- 0 real located events, correctly
    flagged, not silently treated as a valid zero-activity comparison."""
    result = compare_team_matches("Barcelona", BARCA_MATCH_A, ARGENTINA_MEXICO_MATCH_ID)

    ra = result["data_richness"]["team_a"]
    rb = result["data_richness"]["team_b"]
    assert ra["flag"] == "well-supported"
    assert rb["located_events"] == 0
    assert rb["flag"].startswith("LOW SAMPLE")
    assert rb["located_events"] < MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON

    assert result["reliability_caveat"] is not None
    assert f"match {ARGENTINA_MEXICO_MATCH_ID}" in result["reliability_caveat"]
    assert "NOT equally reliable" in result["reliability_caveat"]


def test_compare_team_matches_low_sample_flag_false_for_well_supported_real_matches():
    """Negative case for the same flag (false-positive check): every real
    match in this project's cache clears
    MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON (500) comfortably --
    verified directly (see this feature's own real-data audit: 1,868 real
    (team, match) pairs, minimum 896 located events) before picking that
    threshold rather than assumed to be a safe margin."""
    result = compare_team_matches("Barcelona", BARCA_MATCH_A, BARCA_MATCH_B_NO_360)
    assert result["data_richness"]["team_a"]["located_events"] >= MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON
    assert result["data_richness"]["team_b"]["located_events"] >= MIN_LOCATED_EVENTS_FOR_CONFIDENT_MATCH_COMPARISON
    assert result["reliability_caveat"] is None


def test_compare_team_matches_unfetchable_match_is_low_sample_not_fabricated():
    result = compare_team_matches("Barcelona", BARCA_MATCH_A, 999999999)
    rb = result["data_richness"]["team_b"]
    assert rb["located_events"] == 0
    assert rb["flag"].startswith("LOW SAMPLE")
    assert result["reliability_caveat"] is not None
