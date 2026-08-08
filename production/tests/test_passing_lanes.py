"""Passing Lane Visualizer validation (additive new feature):
team_report.generate_team_passing_lanes[_aggregated] and
passing_lane_visualizer.render_passing_lanes[_aggregated].

Deliberately end-to-end against REAL, already-cached StatsBomb 360 data
-- match 3773386 (Barcelona vs Deportivo Alaves, 360-covered, already
used elsewhere in this project's own test history for team_report.py/
team_comparison.py), so this test runs offline. No MLflow dependency --
unlike generate_team_report, this feature only uses
BiomechanicalPitchControl (a plain, deterministic PyTorch module), never
the trained DeepHit MLP.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from production.src.reporting.team_report import (
    LANE_MIN_COVERED_SAMPLES,
    LANE_SAMPLE_POINTS,
    MIN_PASS_SAMPLES_FOR_CONFIDENT_LANE_OPENNESS,
    generate_team_passing_lanes,
    generate_team_passing_lanes_aggregated,
)
from production.src.reporting.passing_lane_visualizer import (
    render_passing_lanes,
    render_passing_lanes_aggregated,
)

BARCA_MATCH_360 = 3773386


def _assert_non_trivial_image(path) -> None:
    import numpy as np
    from PIL import Image

    assert path.exists() and path.stat().st_size > 0
    image = np.array(Image.open(path).convert("RGB"))
    assert image.std() > 5.0, f"{path} looks blank/near-uniform (std={image.std():.2f})"


def test_generate_team_passing_lanes_real_data():
    """Barcelona, real 360-covered match 3773386: real per-pair
    lane-openness -- not placeholders. Cross-checked directly (802 of
    985 real Barcelona Pass events with a named recipient AND a matched
    360 frame produced a usable lane-openness sample; the remainder
    lacked a named recipient or a matched frame, verified separately)."""
    result = generate_team_passing_lanes("Barcelona", [BARCA_MATCH_360])

    assert result["team_name"] == "Barcelona"
    assert result["matches_used"] == 1
    assert result["total_pass_samples_used"] == 802
    assert len(result["nodes"]) == 16
    assert len(result["lanes"]) == 150

    # Real, well-supported pair (n >= MIN_PASS_SAMPLES_FOR_CONFIDENT_LANE_OPENNESS).
    pique_lenglet = next(
        lane for lane in result["lanes"]
        if lane["passer_name"] == "Gerard Piqué Bernabéu" and lane["recipient_name"] == "Clément Lenglet"
    )
    assert pique_lenglet["n_pass_samples"] == 22
    assert abs(pique_lenglet["mean_lane_openness"] - 0.845387401567264) < 1e-9
    assert pique_lenglet["passing_lane_used_low_sample_flag"] is False

    # Every real openness score is a valid probability complement.
    for lane in result["lanes"]:
        assert 0.0 <= lane["mean_lane_openness"] <= 1.0
        assert lane["n_pass_samples"] >= 1

    # nodes carry real average locations (ADR-002 100x68m space).
    for node in result["nodes"]:
        x, y = node["avg_location"]
        assert 0.0 <= x <= 100.0
        assert 0.0 <= y <= 68.0

    # Sorted descending by openness (the dashboard/renderer both rely on this).
    scores = [lane["mean_lane_openness"] for lane in result["lanes"]]
    assert scores == sorted(scores, reverse=True)


def test_generate_team_passing_lanes_low_sample_flag_real_data():
    """Real low-sample case: most of this match's 150 distinct real
    (passer, recipient) pairs have fewer than
    MIN_PASS_SAMPLES_FOR_CONFIDENT_LANE_OPENNESS (20) real pass samples
    -- correctly flagged. A handful of frequent pairs (center-backs,
    the goalkeeper-to-center-back axis) clear it -- correctly NOT
    flagged -- confirming the threshold discriminates real pairs rather
    than firing uniformly."""
    result = generate_team_passing_lanes("Barcelona", [BARCA_MATCH_360])

    low_sample = [lane for lane in result["lanes"] if lane["passing_lane_used_low_sample_flag"]]
    well_supported = [lane for lane in result["lanes"] if not lane["passing_lane_used_low_sample_flag"]]
    assert len(low_sample) == 147
    assert len(well_supported) == 3
    for lane in low_sample:
        assert lane["n_pass_samples"] < MIN_PASS_SAMPLES_FOR_CONFIDENT_LANE_OPENNESS
    for lane in well_supported:
        assert lane["n_pass_samples"] >= MIN_PASS_SAMPLES_FOR_CONFIDENT_LANE_OPENNESS


def test_generate_team_passing_lanes_no_data_match_skipped_not_fabricated():
    result = generate_team_passing_lanes("Barcelona", [999999999])
    assert result["matches_used"] == 0
    assert result["total_pass_samples_used"] == 0
    assert result["nodes"] == []
    assert result["lanes"] == []


def test_generate_team_passing_lanes_team_not_in_match_skipped():
    result = generate_team_passing_lanes("Manchester United", [BARCA_MATCH_360])
    assert result["matches_used"] == 0
    assert result["total_pass_samples_used"] == 0


def test_generate_team_passing_lanes_aggregated_no_raw_leak_real_data():
    """The actual ADR-021 compliance guarantee: the aggregated variant
    must NEVER include `nodes` (the only field carrying a real per-player
    average LOCATION) -- popped, local-only, before the return dict is
    built (same discipline generate_pass_network_aggregated already
    established). `lanes` (named pairs + scores, no location) passes
    through UNCHANGED, per this feature's own ADR-021 addendum."""
    full = generate_team_passing_lanes("Barcelona", [BARCA_MATCH_360])
    aggregated = generate_team_passing_lanes_aggregated("Barcelona", [BARCA_MATCH_360])

    assert "nodes" not in aggregated
    assert aggregated["lanes"] == full["lanes"]
    assert aggregated["total_pass_samples_used"] == full["total_pass_samples_used"]


def test_lane_sample_and_coverage_constants_are_sane():
    """Step 0.1's own constants: LANE_MIN_COVERED_SAMPLES must be a
    strict majority of LANE_SAMPLE_POINTS (the roadmap's own stated
    reasoning -- "more than half")."""
    assert LANE_MIN_COVERED_SAMPLES > LANE_SAMPLE_POINTS / 2
    assert LANE_MIN_COVERED_SAMPLES <= LANE_SAMPLE_POINTS


def test_render_passing_lanes_real_data(tmp_path):
    result = generate_team_passing_lanes("Barcelona", [BARCA_MATCH_360])
    output_path = tmp_path / "barcelona_lanes.png"
    render_passing_lanes(result, str(output_path))
    _assert_non_trivial_image(output_path)


def test_render_passing_lanes_aggregated_real_data(tmp_path):
    aggregated = generate_team_passing_lanes_aggregated("Barcelona", [BARCA_MATCH_360])
    output_path = tmp_path / "barcelona_lanes_aggregated.png"
    render_passing_lanes_aggregated(aggregated, str(output_path))
    _assert_non_trivial_image(output_path)
