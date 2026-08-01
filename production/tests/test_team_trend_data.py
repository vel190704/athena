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

from production.src.reporting.team_trend_data import generate_team_trend_report


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
