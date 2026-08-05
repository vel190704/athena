"""Engineering-review action item: regression coverage for
`team_trend_data.py` (football-data.co.uk season-by-season team trend
reports) -- a real, real-data-tested module that had no test file at all
until this pass, per the review's own finding. Same rigor as the rest of
`production/tests/`: real, already-cached CSVs (no synthetic fixtures),
explicit assertions on the honest gap-season behavior this module was
specifically built to surface, not just "it ran."

Every case here reproduces a real validation run already documented in
`docs/REPORTING_FINDINGS.md` §8 -- the exact same team/season ranges, so
this test suite is checking the SAME real findings that document reports,
not a new, weaker standard for a newer file.
"""

from production.src.reporting.team_trend_data import (
    compare_team_trend_seasons,
    generate_team_trend_report,
)


def test_generate_team_trend_report_well_supported_no_gaps_real_data():
    """Man City, 2019/20-2025/26: REPORTING_FINDINGS.md §8's documented
    "zero gaps" case. Real, known football history: 93 points (2021/22
    title), 89 (2022/23 treble season), 91 (2023/24 title), a real drop to
    71 (2024/25), partial recovery to 78 (2025/26)."""
    report = generate_team_trend_report("Man City", 2019, 2025)

    assert report["seasons_requested"] == 7
    assert report["seasons_found"] == 7
    assert report["gap_seasons"] == []

    season_stats = report["season_stats"]
    assert season_stats["2021-22"]["points"] == 93
    assert season_stats["2022-23"]["points"] == 89
    assert season_stats["2023-24"]["points"] == 91
    assert season_stats["2024-25"]["points"] == 71
    assert season_stats["2025-26"]["points"] == 78

    # 6 consecutive seasons -> 6 year-over-year deltas, every one flagged
    # consecutive=True (no gap seasons in this range to break that).
    deltas = report["year_over_year_deltas"]
    assert len(deltas) == 6
    assert all(d["consecutive"] for d in deltas)


def test_generate_team_trend_report_honest_gap_seasons_real_data():
    """Norwich, 2018/19-2025/26: REPORTING_FINDINGS.md §8's documented
    relegation-gap case. Only 2 of 8 requested seasons exist in the
    top-flight data (2019-20: 21 points, relegated; 2021-22: 22 points,
    relegated again) -- the other 6 must be reported as explicit gaps,
    never silently dropped, and the one computed delta must be flagged
    non-consecutive since a Championship season sits between the two
    found seasons."""
    report = generate_team_trend_report("Norwich", 2018, 2025)

    assert report["seasons_requested"] == 8
    assert report["seasons_found"] == 2
    assert set(report["gap_seasons"]) == {
        "2018-19", "2020-21", "2022-23", "2023-24", "2024-25", "2025-26",
    }

    season_stats = report["season_stats"]
    assert season_stats["2019-20"]["points"] == 21
    assert season_stats["2021-22"]["points"] == 22

    deltas = report["year_over_year_deltas"]
    assert len(deltas) == 1
    assert deltas[0]["from_season"] == "2019-20"
    assert deltas[0]["to_season"] == "2021-22"
    assert deltas[0]["consecutive"] is False


def test_generate_team_trend_report_unknown_team_returns_all_gaps_not_a_crash():
    """A team name that exists nowhere in the covered leagues (never
    silently fabricated data, never a crash) -- every requested season
    must land in gap_seasons."""
    report = generate_team_trend_report("Definitely Not A Real Club FC", 2020, 2021)

    assert report["seasons_requested"] == 2
    assert report["seasons_found"] == 0
    assert set(report["gap_seasons"]) == {"2020-21", "2021-22"}
    assert report["season_stats"] == {}
    assert report["year_over_year_deltas"] == []


# ============================================================================
# Feature 3: compare_team_trend_seasons -- the team-vs-itself two-season
# comparison, football-data.co.uk track. Reuses the SAME real Man City
# 2019/20 and 2025/26 season points (81 and 78 respectively) confirmed
# directly against the live data before writing this test, consistent
# with this file's own real-data-only discipline.
# ============================================================================


def test_compare_team_trend_seasons_real_data_both_found():
    comparison = compare_team_trend_seasons("Man City", 2019, 2025)

    assert comparison["team_name"] == "Man City"
    assert comparison["season_a"] == "2019-20"
    assert comparison["season_b"] == "2025-26"
    assert comparison["season_a_found"] is True
    assert comparison["season_b_found"] is True

    stats_a, stats_b = comparison["season_a_stats"], comparison["season_b_stats"]
    assert stats_a["points"] == 81
    assert stats_b["points"] == 78

    diff = comparison["diff_b_minus_a"]
    assert diff is not None
    # season_b MINUS season_a, literally: 78 - 81 = -3, a real decrease.
    assert diff["points_delta"] == -3
    assert diff["points_delta"] == stats_b["points"] - stats_a["points"]
    assert diff["goals_scored_delta"] == stats_b["goals_scored"] - stats_a["goals_scored"]
    assert diff["red_cards_delta"] == stats_b["red_cards"] - stats_a["red_cards"]

    assert "declined" in comparison["summary"] or "improved" in comparison["summary"] or "stayed level" in comparison["summary"]
    assert "-3" in comparison["summary"] or "declined" in comparison["summary"]
    assert "NEGATIVE" in comparison["diff_convention"]


def test_compare_team_trend_seasons_reversed_order_flips_delta_sign():
    """Comparing backwards in time (season_b chronologically BEFORE
    season_a) is a real, valid use case -- the diff must be the literal
    season_b-minus-season_a subtraction, sign-flipped from the forward
    comparison of the same two seasons, not silently reordered."""
    forward = compare_team_trend_seasons("Man City", 2019, 2025)
    backward = compare_team_trend_seasons("Man City", 2025, 2019)

    assert backward["season_a"] == "2025-26"
    assert backward["season_b"] == "2019-20"
    assert backward["diff_b_minus_a"]["points_delta"] == -forward["diff_b_minus_a"]["points_delta"]


def test_compare_team_trend_seasons_gap_season_honest_no_data():
    """A season before football-data.co.uk's archive starts (real,
    verified 404 -- see this module's own docstring) must report
    season_a_found=False and a None diff, never a fabricated comparison."""
    comparison = compare_team_trend_seasons("Man City", 1990, 2025)

    assert comparison["season_a_found"] is False
    assert comparison["season_b_found"] is True
    assert comparison["diff_b_minus_a"] is None
    assert "1990-91" in comparison["summary"]
    assert comparison["season_a_stats"] is None
    assert comparison["season_b_stats"] is not None
