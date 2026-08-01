"""Milestone 42 validation: Report Visualization (Step 3).

Renders both dashboards against the SAME real Messi/Argentina data already
validated in `test_reporting.py` (Milestone 40), and checks the output
images are real, non-trivial renders -- not just "the file exists."
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
from PIL import Image

from production.src.reporting.player_report import generate_player_report
from production.src.reporting.player_visualizer import render_player_dashboard
from production.src.reporting.team_report import generate_team_report
from production.src.reporting.team_visualizer import render_team_dashboard

MESSI_PLAYER_ID = 5503
MESSI_MATCH_IDS = [3773386, 3857264, 3857289, 3857300, 3869151, 3869321, 3869519, 3869685]
ARGENTINA_MATCH_IDS = [3857264, 3857289, 3857300, 3869151]


def _assert_non_trivial_image(path) -> np.ndarray:
    assert path.exists() and path.stat().st_size > 0
    image = np.array(Image.open(path).convert("RGB"))
    # A blank/all-one-color render would have zero variance; a real
    # dashboard (pitch outline, text, colored markers/heatmap) should not.
    assert image.std() > 5.0, f"{path} looks blank/near-uniform (std={image.std():.2f})"
    return image


def test_render_player_dashboard_real_data(tmp_path):
    report = generate_player_report(MESSI_PLAYER_ID, MESSI_MATCH_IDS)
    output_path = tmp_path / "messi_dashboard.png"

    render_player_dashboard(report, str(output_path))

    image = _assert_non_trivial_image(output_path)
    print(f"player dashboard: {output_path}, shape={image.shape}, std={image.std():.2f}")
    print(f"report primary_position={report['primary_position']!r}, "
          f"total_minutes_played={report['total_minutes_played']:.1f}, "
          f"primary_formation={report['primary_formation']!r}")


def test_render_team_dashboard_real_data(tmp_path):
    report = generate_team_report("Argentina", ARGENTINA_MATCH_IDS)
    output_path = tmp_path / "argentina_dashboard.png"

    render_team_dashboard(report, str(output_path))

    image = _assert_non_trivial_image(output_path)
    print(f"team dashboard: {output_path}, shape={image.shape}, std={image.std():.2f}")
    print(f"report matches_used={report['matches_used']}, "
          f"threat_by_pitch_zone={report['threat_by_pitch_zone']}")


def test_render_player_dashboard_handles_missing_position_locations(tmp_path):
    """A report with an empty/degenerate positional_distribution must not
    crash the renderer (e.g. a player with zero tagged position events)."""
    fake_report = {
        "player_id": 999999,
        "matches_requested": 1,
        "matches_with_data": 0,
        "matches_player_appeared_in": 0,
        "positional_distribution": {},
        "primary_position": None,
        "heatmap_grid": [[0.0] * 7 for _ in range(10)],
        "heatmap_grid_shape": "10x7",
        "total_minutes_played": 0.0,
        "formation_minutes": {},
        "primary_formation": None,
    }
    output_path = tmp_path / "empty_dashboard.png"
    render_player_dashboard(fake_report, str(output_path))
    _assert_non_trivial_image(output_path)
