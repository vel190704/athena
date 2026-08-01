"""Engineering-review action item: regression coverage for
`team_comparison.py` (general StatsBomb team-season style comparison) --
a real, real-data-tested module that had no test file at all until this
pass, per the review's own finding.

Both cases here reproduce the exact validation runs documented in
`docs/REPORTING_FINDINGS.md` §9 -- real, already-cached StatsBomb data,
not synthetic fixtures. `data/raw/`'s cache already holds every match
these two comparisons touch (Barcelona 2008/2015, Real Madrid 2016),
fetched during that module's own original validation, so these tests run
offline.
"""

from production.src.reporting.team_comparison import compare_team_seasons, LOW_SAMPLE_MATCH_THRESHOLD


def test_compare_team_seasons_well_supported_both_sides_real_data():
    """Barcelona 2008 vs. Barcelona 2015: REPORTING_FINDINGS.md §9.3's
    documented well-supported case. Neither season has 360 data, so this
    must run in location-only mode. Same team name on both sides is a
    real, expected case (comparing one club across eras) -- the summary
    must season-qualify its labels rather than read as a self-comparison
    (the real bug this module's own Step-2 validation found and fixed)."""
    result = compare_team_seasons("Barcelona", 2008, "Barcelona", 2015)

    assert result["analysis_mode"] == "event_location_activity_map"
    assert result["data_richness"]["team_a"]["matches"] >= LOW_SAMPLE_MATCH_THRESHOLD
    assert result["data_richness"]["team_b"]["matches"] >= LOW_SAMPLE_MATCH_THRESHOLD
    assert result["data_richness"]["team_a"]["flag"] == "well-supported"
    assert result["data_richness"]["team_b"]["flag"] == "well-supported"
    assert result["reliability_caveat"] is None

    zones_a = result["zone_shares"]["team_a"]
    zones_b = result["zone_shares"]["team_b"]
    # Real numbers from REPORTING_FINDINGS.md §9.3 (allow float drift from
    # cache/recompute rounding, not exact-equality brittle).
    assert abs(zones_a["middle_third"] - 0.614) < 0.01
    assert abs(zones_b["middle_third"] - 0.568) < 0.01

    # Season-qualified labels, not bare "Barcelona" on both sides --
    # confirms the real ambiguity bug found during this module's own
    # validation stays fixed.
    assert "2008" in result["summary"]
    assert "2015" in result["summary"]


def test_compare_team_seasons_low_sample_reliability_caveat_fires_real_data():
    """Real Madrid 2016 vs. Barcelona 2008: REPORTING_FINDINGS.md §9.4's
    documented low-sample case. Real Madrid's 2016 side is built from only
    3 real matches (2 La Liga + the 2016/17 Champions League final) --
    the reliability caveat must fire and must be a visible, top-level
    field, not silently absent just because the comparison itself still
    "succeeds"."""
    result = compare_team_seasons("Real Madrid", 2016, "Barcelona", 2008)

    real_madrid_info = result["data_richness"]["team_a"]
    barcelona_info = result["data_richness"]["team_b"]
    assert real_madrid_info["matches"] < LOW_SAMPLE_MATCH_THRESHOLD
    assert real_madrid_info["flag"].startswith("LOW SAMPLE")
    assert barcelona_info["flag"] == "well-supported"

    assert result["reliability_caveat"] is not None
    assert "Real Madrid 2016" in result["reliability_caveat"]
    assert "NOT equally reliable" in result["reliability_caveat"]
