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
from production.src.reporting.player_report import (
    MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP,
    generate_player_shot_map,
)
from production.src.reporting.player_visualizer import render_shot_map

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
