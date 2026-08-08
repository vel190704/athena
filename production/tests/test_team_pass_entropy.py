"""Tactical Entropy validation (additive new feature): Shannon CONDITIONAL
(bigram) entropy over a team's pass-DIRECTION transition sequence, in
`team_report.generate_team_pass_entropy`.

Deliberately end-to-end against REAL, already-cached match data, same
discipline as test_press_resistance.py -- reuses match 3857264 (Argentina
vs Poland), this session's own canonical real-data test match, so no new
network fetch is needed here. The Step 0.4 zero-probability test is the
one deliberate exception (see that test's own docstring for why a
controlled, non-real input is used there instead).
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import math

import production.src.reporting.team_report as team_report_module
from production.src.reporting.team_report import (
    MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY,
    PASS_DIRECTION_THRESHOLD_METERS,
    PASS_TYPE_CATEGORIES,
    _pass_direction_category,
    generate_team_pass_entropy,
)

ARGENTINA_POLAND_MATCH_ID = 3857264  # real, verified: 4,052 total events, 886 real Argentina pass attempts


# --- _pass_direction_category (Step 0.1 category-boundary edge cases) ----


def test_pass_direction_category_boundary_is_sideways_not_forward():
    """Exactly AT the threshold (5.0m) is Sideways -- the check is a
    strict `>`, not `>=` (see team_report.py's own docstring). location=
    [10.0, 40.0], end_location=[16.0, 40.0]: raw delta = 6.0 units,
    * X_SCALE (0.8333...) = exactly 5.0m."""
    event = {"location": [10.0, 40.0], "pass": {"end_location": [16.0, 40.0]}}
    assert _pass_direction_category(event) == "Sideways"


def test_pass_direction_category_just_past_boundary_is_forward():
    event = {"location": [10.0, 40.0], "pass": {"end_location": [16.1, 40.0]}}
    assert _pass_direction_category(event) == "Forward"


def test_pass_direction_category_just_past_boundary_is_backward():
    event = {"location": [10.0, 40.0], "pass": {"end_location": [3.9, 40.0]}}
    assert _pass_direction_category(event) == "Backward"


def test_pass_direction_category_missing_end_location_returns_none():
    assert _pass_direction_category({"location": [10.0, 40.0], "pass": {}}) is None
    assert _pass_direction_category({"pass": {"end_location": [16.0, 40.0]}}) is None


# --- generate_team_pass_entropy (real data) -------------------------------


def test_generate_team_pass_entropy_real_data():
    """Argentina, real match 3857264: real transition counts -- not
    placeholders. Cross-checked directly via the module's own real
    per-event category assignment before trusting these numbers."""
    result = generate_team_pass_entropy("Argentina", [ARGENTINA_POLAND_MATCH_ID])

    assert result["team_name"] == "Argentina"
    assert result["matches_requested"] == 1
    assert result["matches_used"] == 1
    assert result["pass_type_categories"] == list(PASS_TYPE_CATEGORIES)
    assert result["pass_direction_threshold_meters"] == PASS_DIRECTION_THRESHOLD_METERS

    assert result["total_pass_attempts_considered"] == 886
    assert result["completed_pass_attempts_considered"] == 822
    assert result["total_transitions"] == 825

    # Real transition-count matrix, verified directly.
    assert result["transition_counts"]["Forward"] == {"Forward": 86, "Backward": 106, "Sideways": 90}
    assert result["transition_counts"]["Backward"] == {"Forward": 96, "Backward": 52, "Sideways": 104}
    assert result["transition_counts"]["Sideways"] == {"Forward": 114, "Backward": 84, "Sideways": 93}
    assert sum(sum(row.values()) for row in result["transition_counts"].values()) == result["total_transitions"]

    # Each row of the probability matrix sums to 1.0 (a real, well-formed
    # conditional distribution per "from" category).
    for category in PASS_TYPE_CATEGORIES:
        row_sum = sum(result["transition_probabilities"][category].values())
        assert abs(row_sum - 1.0) < 1e-9

    assert 0.0 < result["conditional_entropy_bits"] < result["max_possible_entropy_bits"]
    assert result["max_possible_entropy_bits"] == math.log2(len(PASS_TYPE_CATEGORIES))
    assert abs(
        result["normalized_entropy"] - result["conditional_entropy_bits"] / result["max_possible_entropy_bits"]
    ) < 1e-9

    # 825 real transitions >= MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY (20).
    assert result["pass_entropy_used_low_sample_flag"] is False


def test_generate_team_pass_entropy_unfetchable_match_is_low_sample_not_fabricated():
    result = generate_team_pass_entropy("Argentina", [999999999])
    assert result["matches_used"] == 0
    assert result["total_transitions"] == 0
    assert result["conditional_entropy_bits"] is None
    assert result["normalized_entropy"] is None
    assert result["pass_entropy_used_low_sample_flag"] is True


def test_generate_team_pass_entropy_team_not_in_match_is_low_sample_not_fabricated():
    """A real match this team genuinely did not play in -- correctly
    yields zero data, not a crash or a fabricated entropy value."""
    result = generate_team_pass_entropy("Manchester United", [ARGENTINA_POLAND_MATCH_ID])
    assert result["matches_used"] == 0
    assert result["total_transitions"] == 0
    assert result["pass_entropy_used_low_sample_flag"] is True


def test_generate_team_pass_entropy_empty_match_ids_is_low_sample_not_fabricated():
    result = generate_team_pass_entropy("Argentina", [])
    assert result["matches_requested"] == 0
    assert result["matches_used"] == 0
    assert result["pass_entropy_used_low_sample_flag"] is True


def test_generate_team_pass_entropy_low_sample_flag_false_for_well_sampled_real_team():
    """Negative case for the same flag (false-positive check): a real,
    single full match's real transitions comfortably clear
    MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY (20) -- confirmed directly
    against real cached data rather than assumed always true (see this
    project's own finding, reported alongside this test, that a real full
    match's pass volume makes this flag's false-positive case genuinely
    hard to construct from real data at all -- the practically relevant
    false cases are the zero-data ones tested above instead)."""
    result = generate_team_pass_entropy("Argentina", [ARGENTINA_POLAND_MATCH_ID])
    assert result["total_transitions"] >= MIN_TRANSITIONS_FOR_CONFIDENT_PASS_ENTROPY
    assert result["pass_entropy_used_low_sample_flag"] is False


# --- Step 0.4: zero-probability (0*log2(0) = 0) handling ------------------


def test_generate_team_pass_entropy_zero_probability_handled_not_nan(monkeypatch):
    """Step 0.4's specific numerical edge case: a team with a narrow real
    repertoire (here, a team that ONLY ever passes Forward) must yield an
    exact entropy of 0.0 (perfectly predictable), not NaN or a crash.

    Deliberately a CONTROLLED, non-real input, not a real match: checked
    directly above (see this project's own real-data audit alongside this
    test) that no real cached match/team combination has few enough real
    transitions -- let alone a genuinely single-category repertoire -- to
    exercise this exact numerical edge naturally; every real team in every
    real cached match has well over 60 real Pass events. This test
    isolates Step 0.4's entropy MATH specifically (0*log2(0) treated as
    0, and a category never observed as a "from" state contributing zero
    weight rather than a NaN term) via `monkeypatch`, the same way a unit
    test for a pure numerical edge case normally would -- it is not a
    claim that any real team behaves this way. The real end-to-end
    pipeline (fetch -> chain-group -> categorize -> count -> entropy) is
    already covered by the real-data tests above.
    """
    monkeypatch.setattr(team_report_module, "_teams_in_match", lambda match_id: {"Test Team"})
    monkeypatch.setattr(
        team_report_module,
        "_team_pass_category_sequences",
        lambda match_id, team_name: [[("Forward", True)] * 5],
    )
    result = generate_team_pass_entropy("Test Team", [1])

    assert result["transition_counts"]["Forward"] == {"Forward": 4, "Backward": 0, "Sideways": 0}
    assert result["transition_counts"]["Backward"] == {"Forward": 0, "Backward": 0, "Sideways": 0}
    assert result["transition_counts"]["Sideways"] == {"Forward": 0, "Backward": 0, "Sideways": 0}

    # Backward/Sideways never observed as a "from" state -- None
    # probabilities (row_total == 0), not a ZeroDivisionError/NaN.
    assert result["transition_probabilities"]["Backward"] == {"Forward": None, "Backward": None, "Sideways": None}
    assert result["transition_probabilities"]["Sideways"] == {"Forward": None, "Backward": None, "Sideways": None}
    # Forward->Forward is the ONLY observed transition -- p=1.0, and
    # 1.0 * log2(1.0) == 0.0 exactly (not the 0*log2(0) case, but the
    # OTHER real edge -- a deterministic, zero-entropy row).
    assert result["transition_probabilities"]["Forward"]["Forward"] == 1.0

    assert result["conditional_entropy_bits"] == 0.0
    assert result["normalized_entropy"] == 0.0
    assert not math.isnan(result["conditional_entropy_bits"])
