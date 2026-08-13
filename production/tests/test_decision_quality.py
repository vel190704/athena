"""Decision Quality validation (Phase 4, final item):
decision_quality.generate_decision_quality_analysis /
generate_decision_quality_analysis_aggregated -- was a player's pass under
pressure the RIGHT choice, given the REAL best available alternative at
that moment?

Deliberately end-to-end against REAL cached match/360 data (match
3857276, Canada vs Morocco -- the same validation match used throughout
this project's own reporting-track test history). Real numbers below were
verified directly before being written as assertions, not assumed.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.reporting.decision_quality import (
    GOOD_DECISION_OPENNESS_TOLERANCE,
    generate_decision_quality_analysis,
    generate_decision_quality_analysis_aggregated,
)

MATCH_ID = 3857276
TEAM_NAME = "Canada"


def test_generate_decision_quality_analysis_real_match_shape():
    """Real match: confirms the compiled document's shape and a
    non-trivial real decision count -- VERIFIED directly before writing
    these numbers: 50 real under-pressure Pass decisions found for Canada
    in this match with a usable 360 frame and at least one real
    alternative teammate target."""
    result = generate_decision_quality_analysis(TEAM_NAME, MATCH_ID)

    assert result["no_data"] is False
    assert result["team_name"] == TEAM_NAME
    assert result["match_id"] == MATCH_ID
    assert result["good_decision_openness_tolerance"] == GOOD_DECISION_OPENNESS_TOLERANCE
    assert result["total_decisions"] == 50
    assert len(result["decisions"]) == 50

    for decision in result["decisions"]:
        assert 0.0 <= decision["chosen_lane_openness"] <= 1.0
        assert 0.0 <= decision["best_alternative_lane_openness"] <= 1.0
        assert decision["n_alternatives_considered"] >= 1
        assert abs(
            decision["openness_gap"]
            - (decision["chosen_lane_openness"] - decision["best_alternative_lane_openness"])
        ) < 1e-9
        assert decision["good_decision"] == (decision["openness_gap"] >= -GOOD_DECISION_OPENNESS_TOLERANCE)

    assert result["good_decision_count"] == sum(1 for d in result["decisions"] if d["good_decision"])
    assert result["successful_count"] == sum(1 for d in result["decisions"] if d["successful"])
    assert abs(result["good_decision_share"] - result["good_decision_count"] / 50) < 1e-9


def test_generate_decision_quality_analysis_real_data_is_not_trivially_all_good_or_all_bad():
    """Step 0.4-style sanity check (same discipline as every other
    detector tonight): confirms the metric genuinely discriminates on
    real data -- not every decision is "good" (100%) and not every
    decision is "bad" (0%), which would indicate a badly-calibrated or
    broken tolerance rather than a real signal. VERIFIED: 46/50 good
    (92%), 4 real "bad" decisions found."""
    result = generate_decision_quality_analysis(TEAM_NAME, MATCH_ID)

    assert 0 < result["good_decision_count"] < result["total_decisions"]
    assert result["good_decision_count"] == 46


def test_generate_decision_quality_analysis_real_spot_check_bad_decisions_are_football_plausible():
    """Step 3's real spot-check: every one of the 4 real "bad" decisions
    found for Canada in this match has a real, meaningfully negative
    openness_gap (not a near-zero rounding artifact) -- confirming the
    metric is catching genuine cases where a real, better alternative
    existed, not noise around the tolerance boundary. Also confirms the
    real, honest finding that all 4 were nonetheless completed
    successfully (skill/execution overcoming a real spatial disadvantage)
    -- a genuine, disclosed nuance, not assumed to correlate with failure."""
    result = generate_decision_quality_analysis(TEAM_NAME, MATCH_ID)
    bad_decisions = [d for d in result["decisions"] if not d["good_decision"]]

    assert len(bad_decisions) == 4
    for decision in bad_decisions:
        assert decision["openness_gap"] < -GOOD_DECISION_OPENNESS_TOLERANCE
        assert decision["openness_gap"] < -0.05  # meaningfully negative, not a boundary artifact

    assert all(d["successful"] for d in bad_decisions)  # the real, honest finding


def test_generate_decision_quality_analysis_no_data_for_unfetchable_match():
    result = generate_decision_quality_analysis(TEAM_NAME, match_id=1)
    assert result["no_data"] is True
    assert "reason" in result


def test_generate_decision_quality_analysis_aggregated_no_location_or_individual_decisions():
    """ADR-021's own core guarantee, checked directly against the real
    output: the aggregated variant must carry no `decisions` key and no
    individual player's location anywhere -- only per-player rate/count
    summaries, the same shape Press Resistance Index's own already-exempt
    output uses."""
    result = generate_decision_quality_analysis_aggregated(TEAM_NAME, MATCH_ID)

    assert result["no_data"] is False
    assert "decisions" not in result
    assert "player_summary" in result

    for player in result["player_summary"]:
        assert "location" not in player
        assert "chosen_lane_openness" not in player
        assert set(player.keys()) == {
            "player_id", "player_name", "total_decisions", "good_decision_count",
            "successful_count", "good_decision_share", "successful_share",
        }

    # Cross-check: the aggregated per-player totals must sum to the same
    # real total the raw function independently computes.
    raw_result = generate_decision_quality_analysis(TEAM_NAME, MATCH_ID)
    assert sum(p["total_decisions"] for p in result["player_summary"]) == raw_result["total_decisions"]
    assert sum(p["good_decision_count"] for p in result["player_summary"]) == raw_result["good_decision_count"]
