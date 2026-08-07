"""Player Dashboard validation (additive new feature, on top of Milestone
40's Player Report / Milestone 42's Player Visualizer): the new
match-level functions in player_report.py (`generate_player_match_summary`,
`generate_player_match_touch_map[_aggregated]`,
`generate_player_match_timeline[_aggregated]`) and their visualizers.

Deliberately end-to-end against REAL, already-cached match data, same
discipline as test_shot_map.py -- reuses the SAME MESSI_PLAYER_ID and one
of MESSI_MATCH_IDS those files already use, so no new network fetch is
needed here.

Does NOT modify, re-test, or duplicate coverage of `generate_player_report`/
`generate_player_shot_map` themselves -- those are unchanged and already
covered by test_reporting.py/test_shot_map.py.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
from PIL import Image

from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH
from production.src.pipeline.habit_memory import GRID_COLS, GRID_ROWS
from production.src.reporting.player_report import (
    TIMELINE_BUCKET_MINUTES,
    generate_player_match_summary,
    generate_player_match_timeline,
    generate_player_match_timeline_aggregated,
    generate_player_match_touch_map,
    generate_player_match_touch_map_aggregated,
)
from production.src.reporting.player_visualizer import (
    render_player_match_timeline,
    render_player_match_timeline_aggregated,
    render_player_match_touch_map,
    render_player_match_touch_map_aggregated,
)

MESSI_PLAYER_ID = 5503
MESSI_MATCH_IDS = [3773386, 3857264, 3857289, 3857300, 3869151, 3869321, 3869519, 3869685]
MESSI_ARGENTINA_POLAND_MATCH_ID = 3857264  # real, verified: 272 tagged events, all with `location`


def _assert_non_trivial_image(path) -> np.ndarray:
    assert path.exists() and path.stat().st_size > 0
    image = np.array(Image.open(path).convert("RGB"))
    assert image.std() > 5.0, f"{path} looks blank/near-uniform (std={image.std():.2f})"
    return image


# --- Match summary (unconditionally aggregate, no ADR-021 gating) --------


def test_generate_player_match_summary_real_data():
    """Messi, 8 real matches: real per-match minutes/team/opponent/event
    counts -- not placeholders. VERIFIED directly against real cached
    events before trusting this test's own numbers (see this module's
    docstring)."""
    summary = generate_player_match_summary(MESSI_PLAYER_ID, MESSI_MATCH_IDS)

    assert summary["player_id"] == MESSI_PLAYER_ID
    assert summary["matches_requested"] == len(MESSI_MATCH_IDS)
    assert summary["matches_player_appeared_in"] == len(summary["matches"])
    assert summary["matches_player_appeared_in"] > 0

    match = next(m for m in summary["matches"] if m["match_id"] == MESSI_ARGENTINA_POLAND_MATCH_ID)
    assert match["team"] == "Argentina"
    assert match["opponent"] == "Poland"
    assert match["event_type_counts"]["Pass"] == 70
    assert match["event_type_counts"]["Shot"] == 7
    assert match["total_tagged_events"] == sum(match["event_type_counts"].values())
    assert match["minutes_played"] > 0

    # Matches sorted by match_id -- a stable, predictable table order.
    assert summary["matches"] == sorted(summary["matches"], key=lambda m: m["match_id"])


def test_generate_player_match_summary_missing_match_skipped_not_fabricated():
    summary = generate_player_match_summary(MESSI_PLAYER_ID, [999999999])
    assert summary["matches_requested"] == 1
    assert summary["matches_player_appeared_in"] == 0
    assert summary["matches"] == []


# --- Touch map (RAW, ADR-021 condition-2 gated) + aggregated counterpart --


def test_generate_player_match_touch_map_real_data():
    touch_map = generate_player_match_touch_map(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)

    assert touch_map["no_data"] is False
    assert touch_map["total_touches"] == 272
    assert len(touch_map["touches"]) == 272
    assert touch_map["touch_map_used_low_sample_flag"] is False

    for touch in touch_map["touches"]:
        x, y = touch["location"]
        assert 0.0 <= x <= PITCH_LENGTH, f"touch x={x} outside the ADR-002 0-{PITCH_LENGTH} pitch space"
        assert 0.0 <= y <= PITCH_WIDTH, f"touch y={y} outside the ADR-002 0-{PITCH_WIDTH} pitch space"
        assert touch["minute"] >= 0.0


def test_generate_player_match_touch_map_no_data_for_unfetchable_match():
    touch_map = generate_player_match_touch_map(MESSI_PLAYER_ID, 999999999)
    assert touch_map["no_data"] is True
    assert touch_map["touches"] == []


def test_generate_player_match_touch_map_aggregated_no_raw_leak_real_data():
    """The actual ADR-021 compliance guarantee: the aggregated variant
    must carry a grid-binned density summing to 1.0 (same GRID_COLS x
    GRID_ROWS convention as the shot map), and must NEVER include the raw
    `touches` list -- popped, local-only, before the return dict is built
    (same discipline `generate_player_shot_map_aggregated` already
    established)."""
    touch_map_agg = generate_player_match_touch_map_aggregated(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)

    assert "touches" not in touch_map_agg
    assert touch_map_agg["total_touches"] == 272
    assert len(touch_map_agg["touch_density_grid"]) == GRID_COLS
    assert len(touch_map_agg["touch_density_grid"][0]) == GRID_ROWS
    assert abs(sum(sum(col) for col in touch_map_agg["touch_density_grid"]) - 1.0) < 1e-9


def test_render_player_match_touch_map_real_data(tmp_path):
    touch_map = generate_player_match_touch_map(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)
    output_path = tmp_path / "messi_touch_map.png"
    render_player_match_touch_map(touch_map, str(output_path))
    image = _assert_non_trivial_image(output_path)
    print(f"touch map render: {output_path}, shape={image.shape}, std={image.std():.2f}")


def test_render_player_match_touch_map_aggregated_real_data(tmp_path):
    touch_map_agg = generate_player_match_touch_map_aggregated(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)
    output_path = tmp_path / "messi_touch_map_aggregated.png"
    render_player_match_touch_map_aggregated(touch_map_agg, str(output_path))
    _assert_non_trivial_image(output_path)


# --- Key-event timeline (RAW, ADR-021 condition-2 gated) + aggregated -----


def test_generate_player_match_timeline_real_data():
    timeline = generate_player_match_timeline(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)

    assert timeline["no_data"] is False
    assert timeline["total_events"] == 272
    assert len(timeline["timeline"]) == 272

    # Chronological order is (period, index) -- NOT strictly-increasing
    # `minute`. Real finding, verified directly against this match's own
    # cached events before writing this assertion: period 1's last Messi
    # event lands at minute 47:59 (injury time), and period 2's first
    # Messi event is recorded at minute 45:00 (the second-half clock
    # display restarts at 45 without regard for how far period 1's
    # injury time ran) -- a genuine backward jump in `minute` across the
    # half-time boundary that does NOT indicate an ordering bug. What
    # IS guaranteed is that within a single period, minute is
    # non-decreasing -- checked indirectly below (no `period` field is
    # exposed on a timeline entry by design, see player_report.py's
    # allowlist), by confirming the whole sequence has exactly ONE
    # backward jump (the real half-time boundary), not scattered ones
    # (which would indicate a genuine ordering bug).
    backward_jumps = sum(
        1 for i in range(1, len(timeline["timeline"]))
        if timeline["timeline"][i]["minute"] < timeline["timeline"][i - 1]["minute"]
    )
    assert backward_jumps == 1, (
        f"expected exactly one backward minute jump (the real half-time boundary), got {backward_jumps}"
    )

    shot_entries = [e for e in timeline["timeline"] if e["event_type"] == "Shot"]
    assert len(shot_entries) == 7
    for shot_entry in shot_entries:
        assert "statsbomb_xg" in shot_entry["detail"]
        # The compliance-critical guard: a real Shot event's own sub-dict
        # carries a `freeze_frame` list of ~15 OTHER real, individually-
        # located players -- VERIFIED present in the real underlying event
        # (see player_report.py's own _TIMELINE_DETAIL_KEYS comment). This
        # allowlisted `detail` dict must NEVER include it.
        assert "freeze_frame" not in shot_entry["detail"]
        assert "location" not in shot_entry["detail"]
        assert "end_location" not in shot_entry["detail"]


def test_generate_player_match_timeline_no_data_for_unfetchable_match():
    timeline = generate_player_match_timeline(MESSI_PLAYER_ID, 999999999)
    assert timeline["no_data"] is True
    assert timeline["timeline"] == []


def test_generate_player_match_timeline_aggregated_no_raw_leak_real_data():
    """The actual ADR-021 compliance guarantee: the aggregated variant
    must carry only event-TYPE counts per coarse time bucket, and must
    NEVER include the raw, individually-enumerated `timeline` -- no exact
    minute, no per-event outcome/body-part detail, anywhere."""
    timeline_agg = generate_player_match_timeline_aggregated(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)

    assert "timeline" not in timeline_agg
    assert timeline_agg["bucket_size_minutes"] == TIMELINE_BUCKET_MINUTES
    assert timeline_agg["total_events"] == 272

    total_bucketed = sum(sum(b["event_type_counts"].values()) for b in timeline_agg["buckets"])
    assert total_bucketed == 272  # every real event lands in exactly one bucket, none dropped

    for bucket in timeline_agg["buckets"]:
        assert bucket["bucket_end_minute"] - bucket["bucket_start_minute"] == TIMELINE_BUCKET_MINUTES
        # No individual-event field anywhere in a bucket.
        assert "minute" not in bucket
        assert "detail" not in bucket


def test_render_player_match_timeline_real_data(tmp_path):
    timeline = generate_player_match_timeline(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)
    output_path = tmp_path / "messi_timeline.png"
    render_player_match_timeline(timeline, str(output_path))
    _assert_non_trivial_image(output_path)


def test_render_player_match_timeline_aggregated_real_data(tmp_path):
    timeline_agg = generate_player_match_timeline_aggregated(MESSI_PLAYER_ID, MESSI_ARGENTINA_POLAND_MATCH_ID)
    output_path = tmp_path / "messi_timeline_aggregated.png"
    render_player_match_timeline_aggregated(timeline_agg, str(output_path))
    _assert_non_trivial_image(output_path)
