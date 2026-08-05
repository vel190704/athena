"""Shot Map validation (additive new feature, on top of Milestone 40's
Player Report / Milestone 42's Player Visualizer): `generate_player_shot_map`
(player_report.py) and `render_shot_map` (player_visualizer.py).

Deliberately end-to-end against REAL, already-cached match data, same
discipline as `test_reporting.py`/`test_report_visualizer.py` -- reuses
the EXACT SAME MESSI_PLAYER_ID/MESSI_MATCH_IDS those files already use, so
no new network fetch is needed here.

Does NOT modify, re-test, or duplicate coverage of `generate_player_report`/
`render_player_dashboard` themselves -- those are unchanged and already
covered by `test_reporting.py`/`test_report_visualizer.py`.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
from PIL import Image

from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH
from production.src.pipeline.habit_memory import GRID_COLS, GRID_ROWS
from production.src.reporting.player_report import (
    MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP,
    generate_player_shot_map,
    generate_player_shot_map_aggregated,
)
from production.src.reporting.player_visualizer import render_shot_map, render_shot_map_aggregated

MESSI_PLAYER_ID = 5503
MESSI_MATCH_IDS = [3773386, 3857264, 3857289, 3857300, 3869151, 3869321, 3869519, 3869685]

# Hiroki Sakai (id=3530): a real player confirmed, via a full scan of this
# project's cached data, to have exactly 1 real tagged Shot event across
# his 2 cached matches -- the exact real low-shot-count case this
# feature's own low-sample flag exists to catch, mirroring
# test_reporting.py's own Yu-Min Cho / Kristijan Jakic pattern for
# generate_player_report.
SAKAI_PLAYER_ID = 3530
SAKAI_MATCH_IDS = [3857284, 3869219]


def _assert_non_trivial_image(path) -> np.ndarray:
    assert path.exists() and path.stat().st_size > 0
    image = np.array(Image.open(path).convert("RGB"))
    assert image.std() > 5.0, f"{path} looks blank/near-uniform (std={image.std():.2f})"
    return image


def test_generate_player_shot_map_real_data():
    """Messi, 8 real matches: real shots, real goals, real StatsBomb xG --
    not placeholders. Body-part breakdown must sum to total_shots exactly
    (every real shot counted in exactly one bucket)."""
    shot_map = generate_player_shot_map(MESSI_PLAYER_ID, MESSI_MATCH_IDS)

    assert shot_map["player_id"] == MESSI_PLAYER_ID
    assert shot_map["matches_requested"] == len(MESSI_MATCH_IDS)
    assert shot_map["total_shots"] == 44
    assert shot_map["goals"] == 9
    assert shot_map["goals"] <= shot_map["total_shots"]
    assert abs(shot_map["sum_statsbomb_xg"] - 8.235283801) < 1e-6
    assert abs(shot_map["xg_per_shot"] - shot_map["sum_statsbomb_xg"] / shot_map["total_shots"]) < 1e-9

    assert sum(shot_map["shots_by_body_part"].values()) == shot_map["total_shots"]
    assert set(shot_map["shots_by_body_part"]) <= {"Right Foot", "Left Foot", "Head", "Other"}

    # Well-supported -- 44 real shots clears MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP.
    assert shot_map["shot_map_used_low_sample_flag"] is False

    # Every real shot's xG must be a genuine StatsBomb probability (0, 1],
    # and every shot's location must already be ADR-002-rescaled (this
    # feature's own real bug, found and fixed during its own validation --
    # raw StatsBomb location is in a 120x80 unit grid, not this project's
    # 100x68m pitch space every renderer draws against).
    for shot in shot_map["shots"]:
        assert 0.0 < shot["statsbomb_xg"] <= 1.0
        x, y = shot["location"]
        assert 0.0 <= x <= PITCH_LENGTH, f"shot x={x} outside the ADR-002 0-{PITCH_LENGTH} pitch space"
        assert 0.0 <= y <= PITCH_WIDTH, f"shot y={y} outside the ADR-002 0-{PITCH_WIDTH} pitch space"
        assert shot["body_part"] in {"Right Foot", "Left Foot", "Head", "Other"}
        assert shot["is_goal"] == (shot["outcome"] == "Goal")

    print(f"Messi shot map: {shot_map['total_shots']} shots, {shot_map['goals']} goals, "
          f"xg_per_shot={shot_map['xg_per_shot']:.3f}, by body part: {shot_map['shots_by_body_part']}")


def test_generate_player_shot_map_low_sample_real_data():
    """Real low-shot-count case (Step 4's own requirement): a player with
    exactly 1 real tagged shot must trip shot_map_used_low_sample_flag,
    the same honest transparency `generate_player_report`'s own
    heatmap_used_uniform_fallback already provides for general events,
    applied here to shots specifically.
    """
    shot_map = generate_player_shot_map(SAKAI_PLAYER_ID, SAKAI_MATCH_IDS)

    assert shot_map["total_shots"] == 1
    assert shot_map["total_shots"] < MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP
    assert shot_map["shot_map_used_low_sample_flag"] is True
    assert shot_map["goals"] == 0
    assert shot_map["xg_per_shot"] == shot_map["sum_statsbomb_xg"]  # trivially true at n=1, sanity-checked anyway


def test_generate_player_shot_map_zero_shots_no_crash():
    """A player with real cached events but zero real Shot events (a
    defender/goalkeeper case) must not crash, and must report an honest
    zero rather than a fabricated placeholder -- xg_per_shot specifically
    must be None (not 0.0, which would misleadingly claim a real
    zero-value average over zero real shots)."""
    shot_map = generate_player_shot_map(32602, [3869684])  # Kristijan Jakic -- zero tagged events at all, per test_reporting.py

    assert shot_map["total_shots"] == 0
    assert shot_map["goals"] == 0
    assert shot_map["sum_statsbomb_xg"] == 0.0
    assert shot_map["xg_per_shot"] is None
    assert shot_map["shots_by_body_part"] == {}
    assert shot_map["shots"] == []
    assert shot_map["shot_map_used_low_sample_flag"] is True


def test_generate_player_shot_map_missing_match_is_skipped_not_fabricated():
    """Same discipline generate_player_report already establishes: a
    match_id with no cached/fetchable events is skipped with a printed
    warning, never silently fabricated into the shot map."""
    shot_map = generate_player_shot_map(MESSI_PLAYER_ID, [999999999])
    assert shot_map["total_shots"] == 0
    assert shot_map["shots"] == []
    assert shot_map["matches_requested"] == 1


def test_render_shot_map_real_data(tmp_path):
    shot_map = generate_player_shot_map(MESSI_PLAYER_ID, MESSI_MATCH_IDS)
    output_path = tmp_path / "messi_shot_map.png"

    render_shot_map(shot_map, str(output_path))

    image = _assert_non_trivial_image(output_path)
    print(f"shot map render: {output_path}, shape={image.shape}, std={image.std():.2f}")


def test_render_shot_map_low_sample_real_data(tmp_path):
    """The LOW SAMPLE banner (same visual convention
    _draw_positional_distribution/_draw_heatmap already use) must render
    without crashing even for a real 1-shot player -- and the image must
    still be a real, non-trivial render (pitch outline + the single shot
    marker + the warning banner), not blank."""
    shot_map = generate_player_shot_map(SAKAI_PLAYER_ID, SAKAI_MATCH_IDS)
    output_path = tmp_path / "sakai_shot_map.png"

    render_shot_map(shot_map, str(output_path))

    _assert_non_trivial_image(output_path)


def test_render_shot_map_handles_zero_shots_no_crash(tmp_path):
    """Mirrors test_render_player_dashboard_handles_missing_position_locations's
    own discipline: a shot map with zero real shots must render (the "No
    shot data" fallback) rather than crash."""
    shot_map = generate_player_shot_map(32602, [3869684])
    output_path = tmp_path / "empty_shot_map.png"

    render_shot_map(shot_map, str(output_path))

    _assert_non_trivial_image(output_path)


# ============================================================================
# ADR-021 condition-2 compliance fix: generate_player_shot_map_aggregated /
# render_shot_map_aggregated -- the PUBLIC-deployment counterpart to the
# raw per-shot functions above. The core guarantee under test throughout
# this section: NO field on the aggregated dict can be traced back to one
# specific shot's exact location -- confirmed here by asserting the
# `"shots"` key is simply absent (not emptied, not present-but-redacted).
# ============================================================================


def test_generate_player_shot_map_aggregated_real_data_never_returns_raw_shots():
    """Messi, same 8 real matches as the raw-variant test above: the
    aggregated dict must carry the SAME real summary scalars (total_shots,
    goals, sum_statsbomb_xg, xg_per_shot, shots_by_body_part,
    shot_map_used_low_sample_flag -- ADR-021's own reasoning already
    treats these as condition-2-compliant, reused unchanged) while NEVER
    exposing a `"shots"` field at all."""
    aggregated = generate_player_shot_map_aggregated(MESSI_PLAYER_ID, MESSI_MATCH_IDS)
    raw = generate_player_shot_map(MESSI_PLAYER_ID, MESSI_MATCH_IDS)

    assert "shots" not in aggregated
    assert aggregated["total_shots"] == raw["total_shots"] == 44
    assert aggregated["goals"] == raw["goals"] == 9
    assert aggregated["sum_statsbomb_xg"] == raw["sum_statsbomb_xg"]
    assert aggregated["xg_per_shot"] == raw["xg_per_shot"]
    assert aggregated["shots_by_body_part"] == raw["shots_by_body_part"]
    assert aggregated["shot_map_used_low_sample_flag"] == raw["shot_map_used_low_sample_flag"] is False

    # Grid shape/convention: same GRID_COLS x GRID_ROWS habit_memory's own
    # positional heatmap uses.
    density = aggregated["shot_density_grid"]
    mean_xg = aggregated["mean_xg_grid"]
    assert len(density) == GRID_COLS and len(density[0]) == GRID_ROWS
    assert len(mean_xg) == GRID_COLS and len(mean_xg[0]) == GRID_ROWS

    # Density is a real probability distribution over cells -- sums to 1.0
    # across the whole grid, matching team_comparison.py's own
    # activity-grid density convention exactly.
    assert abs(sum(sum(col) for col in density) - 1.0) < 1e-9

    # Every populated cell's mean xG must be a genuine StatsBomb
    # probability in (0, 1]; every cell with shots in the raw variant must
    # correspond to a non-None mean_xg cell in the aggregated variant
    # (and vice versa) -- the two are reductions of the exact same shots.
    populated_cells = 0
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            if mean_xg[col][row] is not None:
                assert 0.0 < mean_xg[col][row] <= 1.0
                populated_cells += 1
                assert density[col][row] > 0.0
            else:
                assert density[col][row] == 0.0
    assert populated_cells > 0

    print(f"Messi aggregated shot map: {populated_cells} populated cells, density sums to 1.0")


def test_generate_player_shot_map_aggregated_low_sample_real_data():
    """Sakai (1 real shot): the low-sample flag must survive into the
    aggregated variant identically, and the single shot must land in
    exactly one grid cell."""
    aggregated = generate_player_shot_map_aggregated(SAKAI_PLAYER_ID, SAKAI_MATCH_IDS)

    assert "shots" not in aggregated
    assert aggregated["total_shots"] == 1
    assert aggregated["shot_map_used_low_sample_flag"] is True

    density = aggregated["shot_density_grid"]
    mean_xg = aggregated["mean_xg_grid"]
    populated_cells = sum(1 for col in density for v in col if v > 0.0)
    assert populated_cells == 1
    assert sum(sum(col) for col in density) == 1.0
    non_none_xg_cells = sum(1 for col in mean_xg for v in col if v is not None)
    assert non_none_xg_cells == 1


def test_generate_player_shot_map_aggregated_zero_shots_no_crash():
    """Zero real shots: an honest all-zero density grid and an all-None
    mean-xG grid -- never a fabricated placeholder, matching
    generate_player_shot_map's own xg_per_shot=None discipline for this
    same case."""
    aggregated = generate_player_shot_map_aggregated(32602, [3869684])

    assert "shots" not in aggregated
    assert aggregated["total_shots"] == 0
    assert aggregated["xg_per_shot"] is None
    density = aggregated["shot_density_grid"]
    mean_xg = aggregated["mean_xg_grid"]
    assert all(v == 0.0 for col in density for v in col)
    assert all(v is None for col in mean_xg for v in col)


def test_render_shot_map_aggregated_real_data_no_individual_marker(tmp_path):
    """Messi: renders without crashing, and the render function itself
    never touches a `"shots"` key (the dict passed in genuinely doesn't
    have one) -- the strongest available proxy, at the function-contract
    level, for "no individual shot location was ever plotted"."""
    aggregated = generate_player_shot_map_aggregated(MESSI_PLAYER_ID, MESSI_MATCH_IDS)
    assert "shots" not in aggregated  # re-confirmed here: this is exactly what gets rendered
    output_path = tmp_path / "messi_shot_map_aggregated.png"

    render_shot_map_aggregated(aggregated, str(output_path))

    image = _assert_non_trivial_image(output_path)
    print(f"aggregated shot map render: {output_path}, shape={image.shape}, std={image.std():.2f}")


def test_render_shot_map_aggregated_low_sample_real_data(tmp_path):
    aggregated = generate_player_shot_map_aggregated(SAKAI_PLAYER_ID, SAKAI_MATCH_IDS)
    output_path = tmp_path / "sakai_shot_map_aggregated.png"

    render_shot_map_aggregated(aggregated, str(output_path))

    _assert_non_trivial_image(output_path)


def test_render_shot_map_aggregated_handles_zero_shots_no_crash(tmp_path):
    aggregated = generate_player_shot_map_aggregated(32602, [3869684])
    output_path = tmp_path / "empty_shot_map_aggregated.png"

    render_shot_map_aggregated(aggregated, str(output_path))

    _assert_non_trivial_image(output_path)
