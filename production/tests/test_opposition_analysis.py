"""Opposition Analysis validation (additive new feature):
team_report.generate_team_opposition_analysis -- 3 specific
opposition-scouting metrics (weak-zone pitch control, reused unmodified
from generate_team_report; build-up length tendency and set-piece shot
reliance, both new).

Deliberately end-to-end against REAL, already-cached StatsBomb event
data -- match 3773386 (Barcelona vs Deportivo Alaves), already used
elsewhere in this project's own test history, so this test runs
offline. Event-data only -- no MLflow/360 dependency (unlike Passing
Lanes).
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.reporting.team_report import (
    BUILDUP_LONG_PASS_THRESHOLD_METERS,
    MIN_BUILDUP_PASSES_FOR_CONFIDENT_LENGTH_TENDENCY,
    MIN_SHOTS_FOR_CONFIDENT_SET_PIECE_RELIANCE,
    SET_PIECE_PLAY_PATTERNS,
    generate_team_opposition_analysis,
)

BARCA_MATCH = 3773386


def test_generate_team_opposition_analysis_real_data():
    """Barcelona, real match 3773386: real build-up/set-piece counts --
    not placeholders. Cross-checked directly against real cached event
    data before trusting these numbers (532 real defensive/middle-third
    passes, 52 exceeding the 25.0m long-pass threshold; 25 real shots,
    13 from a real "From Corner"/"From Free Kick" play_pattern)."""
    result = generate_team_opposition_analysis("Barcelona", [BARCA_MATCH])

    assert result["team_name"] == "Barcelona"
    assert result["matches_used"] == 1

    bt = result["build_up_tendency"]
    assert bt["total_buildup_passes"] == 532
    assert bt["long_passes"] == 52
    assert abs(bt["long_pass_share"] - 52 / 532) < 1e-9
    assert bt["long_pass_threshold_meters"] == BUILDUP_LONG_PASS_THRESHOLD_METERS
    assert bt["build_up_tendency_used_low_sample_flag"] is False

    sp = result["set_piece_reliance"]
    assert sp["total_shots"] == 25
    assert sp["set_piece_shots"] == 13
    assert abs(sp["set_piece_shot_share"] - 0.52) < 1e-9
    assert sp["set_piece_play_patterns"] == sorted(SET_PIECE_PLAY_PATTERNS)
    assert sp["set_piece_reliance_used_low_sample_flag"] is False


def test_generate_team_opposition_analysis_no_data_match_skipped_not_fabricated():
    result = generate_team_opposition_analysis("Barcelona", [999999999])
    assert result["matches_used"] == 0
    bt, sp = result["build_up_tendency"], result["set_piece_reliance"]
    assert bt["total_buildup_passes"] == 0
    assert bt["long_pass_share"] is None
    assert bt["build_up_tendency_used_low_sample_flag"] is True
    assert sp["total_shots"] == 0
    assert sp["set_piece_shot_share"] is None
    assert sp["set_piece_reliance_used_low_sample_flag"] is True


def test_generate_team_opposition_analysis_team_not_in_match_skipped():
    result = generate_team_opposition_analysis("Manchester United", [BARCA_MATCH])
    assert result["matches_used"] == 0
    assert result["build_up_tendency"]["total_buildup_passes"] == 0


def test_generate_team_opposition_analysis_low_sample_flag_false_for_well_supported_real_match():
    """Negative case for both flags (false-positive check): a real full
    match's real build-up/shot counts comfortably clear
    MIN_BUILDUP_PASSES_FOR_CONFIDENT_LENGTH_TENDENCY (20) /
    MIN_SHOTS_FOR_CONFIDENT_SET_PIECE_RELIANCE (20)."""
    result = generate_team_opposition_analysis("Barcelona", [BARCA_MATCH])
    assert result["build_up_tendency"]["total_buildup_passes"] >= MIN_BUILDUP_PASSES_FOR_CONFIDENT_LENGTH_TENDENCY
    assert result["build_up_tendency"]["build_up_tendency_used_low_sample_flag"] is False
    assert result["set_piece_reliance"]["total_shots"] >= MIN_SHOTS_FOR_CONFIDENT_SET_PIECE_RELIANCE
    assert result["set_piece_reliance"]["set_piece_reliance_used_low_sample_flag"] is False


def test_set_piece_play_patterns_excludes_throw_in_and_kickoff():
    """Step 0.3's explicit scoping: throw-ins/kick-offs/goal-kicks are
    real StatsBomb play_pattern restarts too, but are deliberately NOT
    counted as "set pieces" for this scouting metric."""
    assert "From Throw In" not in SET_PIECE_PLAY_PATTERNS
    assert "From Kick Off" not in SET_PIECE_PLAY_PATTERNS
    assert "From Goal Kick" not in SET_PIECE_PLAY_PATTERNS
    assert SET_PIECE_PLAY_PATTERNS == frozenset({"From Corner", "From Free Kick"})
