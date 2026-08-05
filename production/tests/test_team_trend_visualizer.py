"""Feature 3 validation: Team Trend two-season comparison visualization
(`team_trend_visualizer.py`). Same real-data, non-trivial-image discipline
as `test_report_visualizer.py` -- renders against the SAME real Man City
data already validated in `test_team_trend_data.py`, checks the output
image is a real render, not just "the file exists."
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
from PIL import Image

from production.src.reporting.team_trend_data import compare_team_trend_seasons
from production.src.reporting.team_trend_visualizer import render_team_trend_comparison


def _assert_non_trivial_image(path) -> np.ndarray:
    assert path.exists() and path.stat().st_size > 0
    image = np.array(Image.open(path).convert("RGB"))
    assert image.std() > 5.0, f"{path} looks blank/near-uniform (std={image.std():.2f})"
    return image


def test_render_team_trend_comparison_real_data(tmp_path):
    comparison = compare_team_trend_seasons("Man City", 2019, 2025)
    output_path = tmp_path / "mancity_compare.png"

    render_team_trend_comparison(comparison, str(output_path))

    image = _assert_non_trivial_image(output_path)
    print(f"team trend comparison render: {output_path}, shape={image.shape}, std={image.std():.2f}")


def test_render_team_trend_comparison_gap_season_shows_no_data_not_crash(tmp_path):
    """A season outside football-data.co.uk's archive must render a real
    "no data" message image instead of crashing -- same graceful-
    degradation discipline every other renderer in this project's
    reporting track already applies."""
    comparison = compare_team_trend_seasons("Man City", 1990, 2025)
    output_path = tmp_path / "mancity_gap.png"

    render_team_trend_comparison(comparison, str(output_path))

    _assert_non_trivial_image(output_path)
