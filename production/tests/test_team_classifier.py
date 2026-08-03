"""Milestone 28 validation: jersey-color extraction (circular hue
averaging) and masking-aware team clustering.

HONEST SCOPE NOTE (read before treating this as proof of real-world
readiness): the synthetic swatches below are SOLID, FULLY-SATURATED
colors. They validate the CLUSTERING MECHANICS ONLY -- real jersey
patterns (stripes, sponsor logos, printed numbers), lighting/shadow
variation across a pitch, and motion blur are NOT represented here and
remain a real, untested risk for actual broadcast footage. This test
proves the statistics are implemented correctly, not that the pipeline is
ready for real jerseys.
"""

import cv2
import numpy as np
import pytest
from sklearn.cluster import KMeans

from production.src.cv.team_classifier import (
    _build_cluster_features,
    classify_teams,
    extract_jersey_color,
)

BOX_WIDTH = 40
BOX_HEIGHT = 100
BOX_SPACING = 60

BLUE_BGR = (255, 0, 0)
RED_BGR = (0, 0, 255)
YELLOW_BGR = (0, 255, 255)
BLACK_BGR = (0, 0, 0)


def _make_roster_image_and_bboxes():
    """5 blue, 5 red, 1 yellow (GK), 1 black (referee) -- solid,
    fully-saturated rectangles laid out side by side."""
    colors_by_track_id = {}
    bboxes_by_track_id = {}

    track_id = 1
    x = 10
    for _ in range(5):
        colors_by_track_id[track_id] = BLUE_BGR
        bboxes_by_track_id[track_id] = [x, 10, BOX_WIDTH, BOX_HEIGHT]
        track_id += 1
        x += BOX_SPACING
    for _ in range(5):
        colors_by_track_id[track_id] = RED_BGR
        bboxes_by_track_id[track_id] = [x, 10, BOX_WIDTH, BOX_HEIGHT]
        track_id += 1
        x += BOX_SPACING
    colors_by_track_id[track_id] = YELLOW_BGR  # goalkeeper
    bboxes_by_track_id[track_id] = [x, 10, BOX_WIDTH, BOX_HEIGHT]
    goalkeeper_track_id = track_id
    track_id += 1
    x += BOX_SPACING
    colors_by_track_id[track_id] = BLACK_BGR  # referee
    bboxes_by_track_id[track_id] = [x, 10, BOX_WIDTH, BOX_HEIGHT]
    referee_track_id = track_id
    x += BOX_SPACING

    canvas = np.zeros((BOX_HEIGHT + 20, x, 3), dtype=np.uint8)
    for tid, bbox in bboxes_by_track_id.items():
        bx, by, bw, bh = bbox
        canvas[by : by + bh, bx : bx + bw] = colors_by_track_id[tid]

    return canvas, bboxes_by_track_id, goalkeeper_track_id, referee_track_id


def test_extract_jersey_color_on_solid_swatches_matches_known_hsv():
    """Sanity check: solid, fully-saturated BGR swatches must extract to
    their known OpenCV HSV hue (blue=120, red=0, yellow=30)."""
    canvas, bboxes, gk_id, ref_id = _make_roster_image_and_bboxes()

    blue_color = extract_jersey_color(canvas, bboxes[1])
    red_color = extract_jersey_color(canvas, bboxes[6])
    yellow_color = extract_jersey_color(canvas, bboxes[gk_id])

    assert blue_color[0] == pytest.approx(120.0, abs=1.0)
    assert red_color[0] == pytest.approx(0.0, abs=1.0)
    assert yellow_color[0] == pytest.approx(30.0, abs=1.0)


def test_team_classification_separates_blue_and_red_isolates_outliers():
    """5 blue + 5 red must cluster into exactly two team groups; the
    yellow GK and black referee must be flagged 'outlier', not forced into
    either team."""
    canvas, bboxes, gk_id, ref_id = _make_roster_image_and_bboxes()

    players_data = [
        {"track_id": tid, "color": extract_jersey_color(canvas, bbox)}
        for tid, bbox in bboxes.items()
    ]
    roles = classify_teams(players_data, random_state=42)

    blue_track_ids = list(range(1, 6))
    red_track_ids = list(range(6, 11))

    blue_roles = {roles[tid] for tid in blue_track_ids}
    red_roles = {roles[tid] for tid in red_track_ids}

    assert len(blue_roles) == 1, f"blue players split across roles: {blue_roles}"
    assert len(red_roles) == 1, f"red players split across roles: {red_roles}"
    assert blue_roles != red_roles, "blue and red were assigned to the SAME team"
    assert blue_roles.pop() in {"team_A", "team_B"}
    assert red_roles.pop() in {"team_A", "team_B"}

    assert roles[gk_id] == "outlier", f"goalkeeper (yellow) should be 'outlier', got {roles[gk_id]}"
    assert roles[ref_id] == "outlier", f"referee (black) should be 'outlier', got {roles[ref_id]}"

    print("\n=== Team classification summary ===")
    for tid, bbox in bboxes.items():
        color = extract_jersey_color(canvas, bbox)
        print(f"  track_id={tid:2d} HSV=({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f}) "
              f"-> role={roles[tid]}")


def test_iterative_refit_meaningfully_changes_centroids_vs_naive_fit():
    """Proves the masking-avoidance refit is NOT a no-op: replicates
    classify_teams's own naive-fit and candidate-removal-then-refit steps
    (same feature construction, same random_state) and asserts the
    REFINED centroids differ measurably from the NAIVE single-pass
    centroids -- i.e. the outliers really were pulling the naive fit
    before they were removed and the centroids were recomputed.
    """
    canvas, bboxes, gk_id, ref_id = _make_roster_image_and_bboxes()
    players_data = [
        {"track_id": tid, "color": extract_jersey_color(canvas, bbox)}
        for tid, bbox in bboxes.items()
    ]
    features = _build_cluster_features(players_data)

    naive_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features)
    naive_centroids = naive_kmeans.cluster_centers_
    naive_labels = naive_kmeans.labels_
    naive_distances = np.linalg.norm(features - naive_centroids[naive_labels], axis=1)

    num_candidates = max(1, round(0.2 * len(players_data)))
    candidate_indices = set(np.argsort(naive_distances)[-num_candidates:].tolist())
    candidate_track_ids = [players_data[i]["track_id"] for i in candidate_indices]

    inlier_mask = np.array([i not in candidate_indices for i in range(len(players_data))])
    refined_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(features[inlier_mask])
    refined_centroids = refined_kmeans.cluster_centers_

    # Match naive/refined centroids up by nearest pairing (cluster index
    # labels aren't guaranteed to correspond 0<->0, 1<->1 between the two
    # independent fits) before comparing.
    pairing_distances = np.linalg.norm(
        naive_centroids[:, None, :] - refined_centroids[None, :, :], axis=2
    )
    row_ind = [0, 1] if pairing_distances[0, 0] + pairing_distances[1, 1] <= pairing_distances[0, 1] + pairing_distances[1, 0] else [1, 0]
    total_centroid_shift = sum(
        np.linalg.norm(naive_centroids[i] - refined_centroids[j])
        for i, j in enumerate(row_ind)
    )

    print("\n=== Naive vs. refined centroid comparison ===")
    print(f"Candidate outliers removed before refit: track_ids={candidate_track_ids}")
    print(f"Naive centroids:\n{naive_centroids}")
    print(f"Refined centroids:\n{refined_centroids}")
    print(f"Total centroid shift (L2, matched pairs): {total_centroid_shift:.4f}")

    assert gk_id in candidate_track_ids or ref_id in candidate_track_ids, (
        "expected the naive fit's largest-distance candidates to include at least one of the "
        "known outliers (goalkeeper/referee)"
    )
    assert total_centroid_shift > 0.05, (
        f"refined centroids barely differ from naive centroids (shift={total_centroid_shift:.4f}) "
        "-- the iterative refit does not appear to be doing anything"
    )


def test_circular_hue_averaging_within_a_single_crop():
    """THE critical circular-hue check: a SINGLE crop that is half hue~2
    pixels and half hue~177 pixels -- both clearly 'red' -- must average
    to a hue near the 0/179 wraparound (red), NOT near hue 90 (green),
    which is what a naive LINEAR mean of the raw hue values would
    incorrectly produce. This is the only kind of test that can actually
    exercise the wraparound bug: a single solid-color crop cannot, since
    any averaging method trivially reproduces a uniform crop's one hue.
    """
    half_width, height = 20, 40
    hsv_crop = np.zeros((height, half_width * 2, 3), dtype=np.uint8)
    hsv_crop[:, :half_width] = [2, 200, 200]
    hsv_crop[:, half_width:] = [177, 200, 200]
    bgr_crop = cv2.cvtColor(hsv_crop, cv2.COLOR_HSV2BGR)

    # extract_jersey_color samples the middle 40%/upper 60% of the given
    # bbox -- use the full crop as the bbox so that window still straddles
    # the left/right hue=2 / hue=177 split at the crop's horizontal center.
    circular_result = extract_jersey_color(bgr_crop, [0, 0, half_width * 2, height])
    circular_hue = circular_result[0]

    # Explicit regression comparison: what would a NAIVE linear mean of
    # the same raw hue values have produced? Computed independently here
    # (not by calling any function from the codebase) purely to
    # demonstrate the divergence -- no naive implementation is kept in
    # production code.
    full_hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    naive_linear_mean_hue = float(full_hsv[:, :, 0].astype(np.float64).mean())

    print("\n=== Circular hue averaging: wraparound crop (half hue=2, half hue=177) ===")
    print(f"Circular mean hue (extract_jersey_color): {circular_hue:.2f}")
    print(f"Naive linear mean hue (independent computation): {naive_linear_mean_hue:.2f}")

    # Circular mean must land near the wraparound point (red): either
    # close to 0 or close to 179.
    assert circular_hue < 10.0 or circular_hue > 170.0, (
        f"circular mean hue={circular_hue:.2f} is not near the 0/179 wraparound (red) as expected"
    )
    # And must NOT be anywhere near green (hue ~90), which is what the bug
    # would produce.
    assert abs(circular_hue - 90.0) > 30.0

    # The naive linear mean, computed independently, DOES land near green
    # -- this is the regression proof that the fix matters, not just that
    # some averaging happens.
    assert abs(naive_linear_mean_hue - 89.5) < 2.0, (
        f"expected the naive linear mean to be near 89.5 (demonstrating the bug), got "
        f"{naive_linear_mean_hue:.2f}"
    )


def test_circular_hue_wraparound_colors_are_close_in_clustering_feature_space():
    """Clustering-feature-level check (deliberately isolated from the full
    classify_teams pipeline -- see note below): two SEPARATE players with
    solid hue~2 and hue~177 crops (both clearly 'reddish' to a human) must
    be CLOSE together in the `_build_cluster_features` representation,
    proving the (cos, sin) encoding avoids treating wraparound-adjacent
    hues as maximally distant. A naive representation using the raw hue
    SCALAR directly (2 vs 177) would compute a distance of 175 -- nearly
    the maximum possible on a 0-179 scale -- despite these being nearly
    the same color; this test asserts the ACTUAL feature distance is tiny
    by comparison.

    Deliberately does NOT route this through the full `classify_teams`
    pipeline: mixing these two near-duplicate-of-red points into a roster
    of PERFECTLY noiseless (zero natural variance) synthetic swatches hits
    the documented threshold-degeneracy limitation in `classify_teams`'s
    docstring (the "2x median inlier distance" threshold collapses toward
    0 when the true inlier clusters have zero spread, making it
    hypersensitive to any nonzero deviation -- including a legitimate,
    barely-different red). That is a real, separately-documented property
    of testing against zero-variance data, not something this test is
    designed to probe; this test isolates the one thing Step 3.2 actually
    asks for -- that the circular feature representation correctly places
    wraparound-adjacent hues close together -- without that unrelated
    interaction.
    """
    red_features = _build_cluster_features([{"track_id": 0, "color": [0.0, 255.0, 255.0]}])[0]
    low_hue_features = _build_cluster_features([{"track_id": 0, "color": [2.0, 255.0, 255.0]}])[0]
    high_hue_features = _build_cluster_features([{"track_id": 0, "color": [177.0, 255.0, 255.0]}])[0]

    circular_distance_low = np.linalg.norm(red_features - low_hue_features)
    circular_distance_high = np.linalg.norm(red_features - high_hue_features)
    naive_raw_hue_distance = abs(2.0 - 177.0)

    print("\n=== Circular feature-space distance (hue~2 / hue~177 vs. pure red hue=0) ===")
    print(f"Circular feature distance, hue=2 vs hue=0:   {circular_distance_low:.4f}")
    print(f"Circular feature distance, hue=177 vs hue=0: {circular_distance_high:.4f}")
    print(f"Naive raw-hue-scalar distance, hue=2 vs hue=177: {naive_raw_hue_distance:.1f}")

    assert circular_distance_low < 0.15
    assert circular_distance_high < 0.15
    assert naive_raw_hue_distance > 170.0  # the naive artifact this feature representation avoids
