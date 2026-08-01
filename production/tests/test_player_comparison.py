"""Engineering-review action item: regression coverage for
`player_comparison.py` (general player-era style comparison, built on
top of the data-fallback coverage expansion) -- had no test file at all
from the moment it was built, widening the exact gap the review flagged
for `team_comparison.py`/`team_trend_data.py`/the dashboard.

Reproduces the real, documented Messi era-comparison findings from
`docs/REPORTING_FINDINGS.md` §10.3 -- real, already-cached StatsBomb
data (his full career was fetched via the data-fallback coverage
expansion), not synthetic fixtures, so these run offline.
"""

from production.src.reporting.player_comparison import (
    LOW_SAMPLE_EVENT_COUNT_THRESHOLD,
    compare_player_across_eras,
    compare_player_seasons,
)

MESSI_PLAYER_ID = 5503
MESSI_CANDIDATE_TEAMS = ["Barcelona", "Argentina", "Paris Saint-Germain"]


def test_compare_player_seasons_early_vs_peak_barcelona_real_data():
    """2006-07 (early Barcelona) vs. 2014-15 (peak Barcelona):
    REPORTING_FINDINGS.md §10.3's documented first comparison. Both
    seasons well-supported; the largest positional-role shift is Center
    Forward involvement nearly quadrupling, NOT a full winger-to-#9
    switch (Right Wing stays the largest tagged role in both eras) --
    asserted directly against the real numbers, not assumed in advance."""
    result = compare_player_seasons(
        MESSI_PLAYER_ID, 2006, 2014, candidate_team_names=MESSI_CANDIDATE_TEAMS
    )

    richness_a = result["data_richness"]["season_a"]
    richness_b = result["data_richness"]["season_b"]
    assert richness_a["heatmap_event_count"] >= LOW_SAMPLE_EVENT_COUNT_THRESHOLD
    assert richness_b["heatmap_event_count"] >= LOW_SAMPLE_EVENT_COUNT_THRESHOLD
    assert richness_a["flag"] == "well-supported"
    assert richness_b["flag"] == "well-supported"
    assert result["reliability_caveat"] is None

    dist_a = result["positional_role"]["season_a_distribution"]
    dist_b = result["positional_role"]["season_b_distribution"]
    assert abs(dist_a["Right Wing"] - 0.737) < 0.02
    assert abs(dist_b["Right Wing"] - 0.687) < 0.02
    assert abs(dist_a["Center Forward"] - 0.064) < 0.02
    assert abs(dist_b["Center Forward"] - 0.287) < 0.02

    # The largest role shift is Center Forward (+~22.3pp), not Right Wing
    # collapsing -- Right Wing remains each season's largest tagged role.
    role_diff = result["positional_role"]["diff_b_minus_a"]
    largest_shift_position = max(role_diff, key=lambda pos: abs(role_diff[pos]))
    assert largest_shift_position == "Center Forward"
    assert max(dist_a, key=dist_a.get) == "Right Wing"
    assert max(dist_b, key=dist_b.get) == "Right Wing"


def test_compare_player_seasons_peak_barcelona_vs_psg_real_data():
    """2014-15 (peak Barcelona) vs. 2022-23 (PSG):
    REPORTING_FINDINGS.md §10.3's documented second comparison -- the
    largest shift across either comparison (Right Wing collapsing,
    Right Attacking Midfield appearing with zero presence in 2014-15),
    with the independent spatial heatmap diff telling the same real
    story (a deeper, more central role at PSG) as a genuine cross-check,
    not a repeated signal."""
    result = compare_player_seasons(
        MESSI_PLAYER_ID, 2014, 2022, candidate_team_names=MESSI_CANDIDATE_TEAMS
    )

    assert result["reliability_caveat"] is None  # both seasons well-supported

    dist_a = result["positional_role"]["season_a_distribution"]
    dist_b = result["positional_role"]["season_b_distribution"]
    assert abs(dist_a["Right Wing"] - 0.687) < 0.02
    assert abs(dist_b["Right Wing"] - 0.108) < 0.02
    assert "Right Attacking Midfield" not in dist_a
    assert dist_b["Right Attacking Midfield"] > 0.15

    role_diff = result["positional_role"]["diff_b_minus_a"]
    largest_shift_position = max(role_diff, key=lambda pos: abs(role_diff[pos]))
    assert largest_shift_position == "Right Wing"
    assert role_diff["Right Wing"] < -0.5  # a real, large collapse, not noise

    # Independent cross-check: the spatial (raw-coordinate) diff agrees
    # with the positional-tag diff -- middle-third share rises, attacking-
    # third share falls, consistent with a deeper, more central PSG role.
    zone_diff = result["spatial_activity"]["diff_b_minus_a"]
    assert zone_diff["middle_third"] > 0
    assert zone_diff["attacking_third"] < 0

    # Informational-only diagnostic: 2022-23 genuinely has full 360
    # coverage (Ligue 1 + World Cup 2022), but the overall diagnostic
    # must still report False, since 2014-15 has none and both sides are
    # required -- and it must never change which analysis actually ran.
    diag = result["pitch_control_diagnostic"]
    assert diag["season_a_360_matches"] == 0
    assert diag["season_b_360_matches"] > 0
    assert diag["pitch_control_possible_in_principle"] is False


def test_compare_player_across_eras_pairwise_real_data():
    """compare_player_across_eras([2006, 2014, 2022]) must produce exactly
    the two pairwise comparisons above, via the SAME underlying function
    -- not a separate N-way implementation."""
    results = compare_player_across_eras(
        MESSI_PLAYER_ID, [2006, 2014, 2022], candidate_team_names=MESSI_CANDIDATE_TEAMS
    )

    assert len(results) == 2
    assert (results[0]["season_a"], results[0]["season_b"]) == (2006, 2014)
    assert (results[1]["season_a"], results[1]["season_b"]) == (2014, 2022)
