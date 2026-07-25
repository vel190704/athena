"""Milestone 26 validation: ByteTrack player tracking + apparent pixel
velocity.

Velocities are apparent pixel motion, not calibrated player velocity --
camera motion is not yet compensated for. See `tracker.py`'s module
docstring for the full methodological note; this is restated in the test
output itself (Step 2.7) so it can never be read as "tracking is done."

SKIPPED (not failed) if `data/raw/test_match.mp4` doesn't exist. Per the
sourcing note for this milestone: prefer SoccerNet's own licensed clips
(once Milestone 25's NDA process completes) over YouTube-scraped footage,
which sits in uncertain copyright/ToS territory and should never become
this project's standard test-data source. If a YouTube clip is used as a
stopgap for this milestone only, it should be treated as strictly private
local test material -- never referenced in a committed artifact, and this
test file makes no assumption about how `test_match.mp4` was obtained,
only that it may or may not be present locally.
"""

from pathlib import Path

import pytest

from production.src.cv.tracker import run_tracking

TEST_MATCH_VIDEO_PATH = Path("data/raw/test_match.mp4")


def test_bytetrack_player_tracking_on_real_footage():
    if not TEST_MATCH_VIDEO_PATH.exists():
        pytest.skip(
            f"No local test video found at {TEST_MATCH_VIDEO_PATH}. Prefer a real SoccerNet "
            "clip once Milestone 25's NDA access is available; a private local broadcast clip "
            "is an acceptable stopgap for this milestone only, but is not provided in this "
            "environment. Place a clip at this path to run this test for real."
        )

    output = run_tracking(str(TEST_MATCH_VIDEO_PATH))

    assert len(output) > 0, "run_tracking returned no frames"

    # Step 2.4: at least 2 unique track_ids must persist across MULTIPLE
    # frames -- proving identity persistence, not independent per-frame
    # detection with no continuity between frames.
    frame_counts_by_track_id: dict[int, int] = {}
    for frame in output:
        for track in frame["tracks"]:
            frame_counts_by_track_id[track["track_id"]] = (
                frame_counts_by_track_id.get(track["track_id"], 0) + 1
            )
    persistent_track_ids = [tid for tid, count in frame_counts_by_track_id.items() if count > 1]
    assert len(persistent_track_ids) >= 2, (
        f"expected at least 2 track_ids persisting across multiple frames, found "
        f"{len(persistent_track_ids)}: {frame_counts_by_track_id}"
    )

    # Step 2.5: at least one persistent track must show REAL displacement
    # (still pixel-space) -- proving velocity is actually being measured,
    # not just defaulted to zero everywhere.
    nonzero_velocity_found = any(
        track["vel_pixels_per_sec"] != [0.0, 0.0]
        for frame in output
        for track in frame["tracks"]
        if track["track_id"] in persistent_track_ids
    )
    assert nonzero_velocity_found, (
        "no persistent track showed nonzero vel_pixels_per_sec -- displacement does not appear "
        "to be measured"
    )

    total_tracks = sum(len(frame["tracks"]) for frame in output)
    likely_id_switch_count = sum(
        1 for frame in output for track in frame["tracks"] if track["likely_id_switch"]
    )
    print(f"\n=== ByteTrack summary ({len(output)} frames, {total_tracks} total track "
          "observations) ===")
    print(f"Unique track_ids seen: {len(frame_counts_by_track_id)}")
    print(f"Track_ids persisting across >1 frame: {len(persistent_track_ids)}")
    # A nonzero count here is a real diagnostic about tracker reliability
    # on THIS footage (players crossing paths, brief occlusion causing
    # ByteTrack to reassign an ID) -- not something to hide, and not
    # evidence this milestone failed.
    print(f"likely_id_switch flagged: {likely_id_switch_count} / {total_tracks} track "
          f"observations ({likely_id_switch_count / total_tracks:.1%} if nonzero)")

    print("\nFirst 5 frames:")
    for frame in output[:5]:
        print(f"  frame_num={frame['frame_num']}:")
        for track in frame["tracks"]:
            print(
                f"    track_id={track['track_id']} pos={track['pos']} "
                f"vel_pixels_per_sec={track['vel_pixels_per_sec']} "
                f"likely_id_switch={track['likely_id_switch']}"
            )

    print(
        "\nReminder: velocities are apparent pixel motion, not calibrated player velocity -- "
        "camera motion is not yet compensated for."
    )
