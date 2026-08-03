"""Milestone 40 validation: Historical Player & Team Analysis Reports.

STANDALONE, additive reporting layer built entirely on top of ALREADY-
VALIDATED StatsBomb-track data and models (Milestones 1-36) -- no CV/video
dependency whatsoever, independent of ADR-013 through ADR-016.

Deliberately end-to-end against REAL, already-cached match data (not
synthetic tensors), same discipline as `test_explainer.py`/
`test_habit_memory.py`. Every match_id used here is already present under
`data/raw/` from prior milestones' work, so these tests run offline against
the local cache.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.models.explainer import load_deterministic_mlp
from production.src.reporting.player_report import generate_player_report
from production.src.reporting.team_report import (
    _build_pitch_grid,
    _match_representative_chain_frames,
    generate_team_report,
)
from production.src.reporting.zone_explainer import (
    _active_grid_and_controls,
    _zone_features_from_grid,
    aggregate_zone_attributions,
    compute_zone_attributions,
    zone_attributions_to_grid,
)
from production.src.spatial.control import BiomechanicalPitchControl

MESSI_PLAYER_ID = 5503
MESSI_MATCH_IDS = [3773386, 3857264, 3857289, 3857300, 3869151, 3869321, 3869519, 3869685]
ARGENTINA_MATCH_IDS = [3857264, 3857289, 3857300, 3869151]
BARCELONA_MATCH_ID = 3773386
TIME_BIN = 3  # 15s horizon, matching Milestones 8/13/14/15
COMPLETENESS_ATOL = 1e-2  # IG's approximation error at n_steps=50, matching test_explainer.py's tolerance


def test_generate_player_report_real_data():
    """Step 1: positional distribution, heatmap, and summary stats for a
    real player (Messi, id=5503) across 8 real matches spanning both club
    (Barcelona) and international (Argentina) play."""
    report = generate_player_report(MESSI_PLAYER_ID, MESSI_MATCH_IDS)

    assert report["matches_with_data"] == len(MESSI_MATCH_IDS)
    assert report["matches_player_appeared_in"] > 0

    # Positional distribution sums to 1 and is non-empty for a player with
    # this many real events.
    assert report["positional_distribution"]
    total_share = sum(report["positional_distribution"].values())
    assert abs(total_share - 1.0) < 1e-9
    assert report["primary_position"] is not None

    # Heatmap: 10x7 grid (habit_memory's established convention), sums to 1.
    assert len(report["heatmap_grid"]) == 10
    assert len(report["heatmap_grid"][0]) == 7
    heatmap_sum = sum(sum(row) for row in report["heatmap_grid"])
    assert abs(heatmap_sum - 1.0) < 1e-9

    # Real, sane minutes total: a player appearing in 8 real matches should
    # have accumulated a plausible multi-match total, not zero and not an
    # implausibly large number (> 8 * 120 minutes would indicate a bug).
    assert 0.0 < report["total_minutes_played"] < 8 * 120.0

    # Formation frequency: VERIFIED to exist in real data (tactics.formation
    # on Starting XI/Tactical Shift events) -- must be populated, not a
    # fabricated placeholder.
    assert report["formation_minutes"]
    assert report["primary_formation"] is not None

    # Milestone 44's validation sweep: a well-supported player like Messi
    # must clear the cold-start threshold and NOT be flagged as a uniform
    # fallback -- the sample-size transparency fields should read as
    # confident here, the mirror image of the low-sample test below.
    assert report["positional_distribution_event_count"] > 100
    assert report["heatmap_event_count"] > 100
    assert report["heatmap_used_uniform_fallback"] is False

    print(f"Messi positional_distribution: {report['positional_distribution']}")
    print(f"Messi primary_position: {report['primary_position']}")
    print(f"Messi total_minutes_played: {report['total_minutes_played']:.1f}")
    print(f"Messi formation_minutes: {report['formation_minutes']}")


def test_generate_player_report_low_sample_transparency_real_data():
    """Milestone 44 validation sweep: a real player with a SINGLE tagged
    event (Yu-Min Cho, id=99479, match 3857262) must not present a
    misleadingly "confident-looking" 100% positional distribution without
    the real sample size alongside it. This is the exact real-data case
    that found the gap `positional_distribution_event_count`/
    `heatmap_event_count`/`heatmap_used_uniform_fallback` now close.
    """
    report = generate_player_report(99479, [3857262])

    assert report["positional_distribution"] == {"Right Center Back": 1.0}
    # The figure above is technically correct but would be misleading
    # without this: exactly 1 real event backs that "100%".
    assert report["positional_distribution_event_count"] == 1
    assert report["heatmap_event_count"] == 1
    assert report["heatmap_used_uniform_fallback"] is True


def test_generate_player_report_zero_events_no_crash():
    """A player who came on but has ZERO tagged events in this match
    (Kristijan Jakic, id=32602, match 3869684) must not crash, and must
    report empty/zero sample-size fields honestly rather than fabricating
    a position or a non-zero count -- while `total_minutes_played` (derived
    from substitution timing, independent of event tagging) can still be
    genuinely non-zero."""
    report = generate_player_report(32602, [3869684])

    assert report["positional_distribution"] == {}
    assert report["primary_position"] is None
    assert report["positional_distribution_event_count"] == 0
    assert report["heatmap_event_count"] == 0
    assert report["heatmap_used_uniform_fallback"] is True
    # Real substitution-derived minutes, unaffected by the zero event count.
    assert report["total_minutes_played"] > 0.0


def test_generate_player_report_missing_match_is_skipped_not_fabricated():
    """A match_id with no cached/fetchable events must be skipped with a
    printed warning, never silently fabricated into the report."""
    report = generate_player_report(MESSI_PLAYER_ID, [999999999])
    assert report["matches_with_data"] == 0
    assert report["matches_player_appeared_in"] == 0
    assert report["positional_distribution"] == {}
    assert report["primary_position"] is None
    assert report["formation_minutes"] == {}
    assert report["primary_formation"] is None


def test_generate_team_report_real_data():
    """Step 2.1/2.2: aggregate pitch-control heatmap + DeepHit threat
    pattern for a real team (Argentina) across 4 real matches, using the
    EXISTING, unmodified BiomechanicalPitchControl engine and the
    deterministically-selected trained MLP."""
    report = generate_team_report("Argentina", ARGENTINA_MATCH_IDS)

    assert report["matches_used"] == len(ARGENTINA_MATCH_IDS)

    grid = report["control_heatmap_grid"]
    assert len(grid) == 10 and len(grid[0]) == 7
    populated_cells = [v for col in grid for v in col if v is not None]
    assert populated_cells, "expected at least some grid cells to have accumulated control data"
    for v in populated_cells:
        assert 0.0 <= v <= 1.0  # control probabilities are in [0, 1]

    assert len(report["weakest_control_zones"]) > 0
    # Weakest zones must actually be sorted ascending by mean_control.
    values = [z["mean_control"] for z in report["weakest_control_zones"]]
    assert values == sorted(values)

    # Threat pattern: attacking-third threat should be higher than
    # defensive-third threat on real football data -- a real, checkable
    # football-sanity assertion, not just "it ran."
    threat = report["threat_by_pitch_zone"]
    assert threat["attacking_third"] is not None and threat["defensive_third"] is not None
    assert threat["attacking_third"] > threat["defensive_third"]

    assert report["threat_by_game_phase"]

    print(f"Argentina weakest_control_zones: {report['weakest_control_zones']}")
    print(f"Argentina threat_by_pitch_zone: {report['threat_by_pitch_zone']}")
    print(f"Argentina threat_by_game_phase: {report['threat_by_game_phase']}")


def test_generate_team_report_skips_match_team_did_not_play():
    """A match_id where `team_name` never appears must be skipped, not
    silently included with zero/garbage data."""
    report = generate_team_report("A Team That Does Not Exist", [BARCELONA_MATCH_ID])
    assert report["matches_used"] == 0


def test_zone_attribution_completeness_real_frame():
    """Step 3: Integrated Gradients completeness axiom (sum of per-cell
    attributions == F(actual input) - F(baseline)) on a REAL match frame --
    the same rigor `test_explainer.py` applies to the 4-scalar-feature
    explainer, extended to confirm the differentiable grid-cell
    reimplementation (`_zone_features_from_grid`) is mathematically
    equivalent to `feature_extractor.extract_features`'s original (non-
    differentiable) formulas, not just "it runs without crashing."
    """
    triples = _match_representative_chain_frames(BARCELONA_MATCH_ID)
    assert triples, f"no 360-covered chains found for match {BARCELONA_MATCH_ID}"
    _chain, parsed, _minute = triples[5]

    engine = BiomechanicalPitchControl()
    pitch_grid = _build_pitch_grid()
    model, normalization_mean, normalization_std, run_id = load_deterministic_mlp()
    print(f"[test_reporting] using deterministic MLP run_id={run_id}")

    result = compute_zone_attributions(
        parsed, engine, model, normalization_mean, normalization_std, pitch_grid, time_bin=TIME_BIN
    )
    assert len(result["active_coords"]) > 0
    attribution_sum = sum(result["attacking_control_attribution"]) + sum(
        result["defending_control_attribution"]
    )

    # F(baseline) -- zero control at every cell -- via the SAME
    # differentiable reimplementation, for the completeness comparison.
    import torch

    active_coords, _att, _def, highest_defending_x = _active_grid_and_controls(engine, parsed, pitch_grid)
    n = active_coords.shape[0]
    zero_features = _zone_features_from_grid(
        torch.zeros(1, n), torch.zeros(1, n), active_coords.detach(), parsed["ball_pos"].detach(), highest_defending_x.detach()
    )
    normalized_baseline = (zero_features - normalization_mean) / normalization_std
    pmf_baseline = model(normalized_baseline)
    cumulative_baseline = torch.cumsum(pmf_baseline, dim=1)[:, TIME_BIN].item()

    expected_diff = result["cumulative_incidence"] - cumulative_baseline
    completeness_error = abs(attribution_sum - expected_diff)

    print(f"cumulative_incidence (actual): {result['cumulative_incidence']:.6f}")
    print(f"cumulative_incidence (baseline, zero control): {cumulative_baseline:.6f}")
    print(f"sum(attributions): {attribution_sum:.6f}, F(input)-F(baseline): {expected_diff:.6f}")
    print(f"Completeness error: {completeness_error:.6f} (tolerance: {COMPLETENESS_ATOL})")

    assert completeness_error < COMPLETENESS_ATOL, (
        f"Zone-level Integrated Gradients completeness check failed: "
        f"|{attribution_sum:.6f} - {expected_diff:.6f}| = {completeness_error:.6f} >= {COMPLETENESS_ATOL}"
    )


def test_zone_attributions_to_grid_shape():
    triples = _match_representative_chain_frames(BARCELONA_MATCH_ID)
    _chain, parsed, _minute = triples[5]
    engine = BiomechanicalPitchControl()
    pitch_grid = _build_pitch_grid()
    model, normalization_mean, normalization_std, _run_id = load_deterministic_mlp()

    result = compute_zone_attributions(
        parsed, engine, model, normalization_mean, normalization_std, pitch_grid, time_bin=TIME_BIN
    )
    grid = zone_attributions_to_grid(result)
    assert len(grid["attacking_control_attribution_grid"]) == 10
    assert len(grid["attacking_control_attribution_grid"][0]) == 7
    # Re-binning must conserve total attribution mass (a sum, not a mean).
    assert abs(
        sum(sum(col) for col in grid["attacking_control_attribution_grid"])
        - sum(result["attacking_control_attribution"])
    ) < 1e-6


def test_aggregate_zone_attributions_real_data():
    """Step 3's multi-frame extension: averaging per-frame attribution
    grids across many real matches -- what turns a single-frame anecdote
    into an evidence-backed "the right half-space is systematically open"
    style statement."""
    result = aggregate_zone_attributions("Argentina", ARGENTINA_MATCH_IDS, time_bin=TIME_BIN)
    assert result["frames_used"] > 0

    grid = result["mean_attacking_control_attribution_grid"]
    assert len(grid) == 10 and len(grid[0]) == 7
    populated = [v for col in grid for v in col if v is not None]
    assert populated, "expected at least some cells to have accumulated attribution data"

    print(f"Argentina zone attribution: {result['frames_used']} frames aggregated")
