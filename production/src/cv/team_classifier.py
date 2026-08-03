"""Milestone 28 (Module 4): jersey-color extraction and unsupervised team
classification.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25 -- nothing in `production/src/models`, `production/src/
pipeline`, `production/src/spatial`, `production/src/physics`, or
`production/src/serving` is imported by or imports from this module.

Bounding boxes here use the same top-left-origin `[x, y, w, h]` pixel
convention as `detector.py`/`tracker.py`. Images are assumed BGR (OpenCV's
default `cv2.imread`/`cv2.VideoCapture` frame convention).
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Crop fraction of the bbox used for color sampling: middle 40% of width
# (avoids arm/background bleed at the box edges), upper 60% of height
# (captures the torso/jersey area, avoiding shorts and grass bleed near
# the bottom of the box).
CROP_WIDTH_FRACTION = 0.4
CROP_HEIGHT_FRACTION = 0.6

# OpenCV's 8-bit HSV hue channel is 0-179, representing the full 0-360
# degree hue circle at half resolution (so true_degrees = h * 2).
HUE_TO_RADIANS = 2.0 * np.pi / 180.0


def extract_jersey_color(image: np.ndarray, bbox: list[float]) -> list[float]:
    """Extracts a single dominant `[H, S, V]` color estimate for the jersey
    within `bbox` (`[x, y, w, h]`, top-left origin, matching
    `detector.py`/`tracker.py`'s convention) in `image` (BGR).

    Crops the CENTER of the box (middle 40% width, upper 60% height) before
    sampling, to avoid grass/background bleed at the box edges and shorts
    bleed near the bottom -- see `CROP_WIDTH_FRACTION`/`CROP_HEIGHT_FRACTION`.

    CIRCULAR HUE AVERAGING (the critical detail this function exists to get
    right): Hue is a CIRCULAR quantity -- OpenCV's 0-179 range wraps around,
    with 0 and 179 both representing red. A naive arithmetic mean of hue
    values silently produces nonsensical results near this wraparound (e.g.
    averaging hues of 2 and 177 -- both clearly red -- gives ~90, which is
    GREEN). This function instead converts each pixel's hue to a unit
    vector (`cos(2h*pi/180), sin(2h*pi/180)`), averages the VECTORS, and
    converts the average vector's angle back to a hue via `atan2`. This is
    the standard circular-mean construction and is the only correct way to
    average an angular quantity.

    Saturation and Value are NOT circular (they're bounded linear scales,
    0-255), so a plain arithmetic mean is correct for them -- do not
    "fix" those to use the circular construction too.

    Returns `[H_circular_mean, S_mean, V_mean]`.

    LIMITATION (a single dominant-hue estimate per crop, not a mode/
    histogram): patterned kits (stripes, sponsor logos, printed numbers)
    will pull this single circular-mean estimate away from the kit's
    actual primary color, since every pixel in the crop -- logo pixels
    included -- contributes equally to the average. Using the mode of a
    coarse hue histogram instead of a single circular mean is a reasonable
    future refinement if patterned kits prove problematic in practice;
    this milestone validates the simpler circular-mean approach only, on
    solid-color synthetic swatches (see `test_team_classifier.py`'s
    explicit caveat about this).
    """
    x, y, w, h = bbox
    image_h, image_w = image.shape[:2]

    crop_x1 = round(x + (1.0 - CROP_WIDTH_FRACTION) / 2.0 * w)
    crop_x2 = round(x + (1.0 + CROP_WIDTH_FRACTION) / 2.0 * w)
    crop_y1 = round(y)
    crop_y2 = round(y + CROP_HEIGHT_FRACTION * h)

    crop_x1 = max(0, min(crop_x1, image_w))
    crop_x2 = max(0, min(crop_x2, image_w))
    crop_y1 = max(0, min(crop_y1, image_h))
    crop_y2 = max(0, min(crop_y2, image_h))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        raise ValueError(
            f"bbox {bbox} produced an empty crop after clipping to the image bounds "
            f"({image_w}x{image_h}) -- cannot extract a color."
        )

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hues = crop_hsv[:, :, 0].astype(np.float64).flatten()
    saturations = crop_hsv[:, :, 1].astype(np.float64).flatten()
    values = crop_hsv[:, :, 2].astype(np.float64).flatten()

    angles = hues * HUE_TO_RADIANS
    mean_x = np.cos(angles).mean()
    mean_y = np.sin(angles).mean()
    mean_angle = np.arctan2(mean_y, mean_x)  # (-pi, pi]

    h_circular = (mean_angle / HUE_TO_RADIANS) % 180.0  # wraps negative angles back into [0, 180)

    return [float(h_circular), float(saturations.mean()), float(values.mean())]


def _hue_to_unit_vector(hue: float) -> tuple[float, float]:
    angle = hue * HUE_TO_RADIANS
    return np.cos(angle), np.sin(angle)


def _build_cluster_features(players_data: list[dict]) -> np.ndarray:
    """Converts each player's `[H, S, V]` color into a clustering feature
    vector `[cos(H), sin(H), S/255, V/255]`.

    Using raw hue directly (a single scalar 0-179) would create a spurious
    Euclidean-distance artifact at the wraparound: hue 2 and hue 177 are
    nearly the SAME color but would appear maximally far apart (175 units)
    to a naive distance metric on the raw scalar. Representing hue as its
    two circular components fixes this -- their Euclidean distance in
    (cos, sin) space correctly reflects true hue closeness regardless of
    where on the circle the colors fall. S and V are normalized to
    [0, 1] (dividing by 255) to keep them roughly comparable in scale to
    the cos/sin components (each in [-1, 1]), rather than dominating the
    distance purely because of their larger raw numeric range.
    """
    features = []
    for player in players_data:
        h, s, v = player["color"]
        cos_h, sin_h = _hue_to_unit_vector(h)
        features.append([cos_h, sin_h, s / 255.0, v / 255.0])
    return np.array(features, dtype=np.float64)


def classify_teams(players_data: list[dict], random_state: int = 42) -> dict[int, str]:
    """Separates players into two teams via unsupervised clustering on
    jersey color, isolating goalkeepers/referees (whose kit color usually
    differs sharply from both outfield teams) as `"outlier"` rather than
    forcing them into one of the two team clusters.

    `players_data`: `[{"track_id": int, "color": [H, S, V]}, ...]` (e.g.
    from `extract_jersey_color`).

    MASKING-AWARE OUTLIER DETECTION (the critical detail this function
    exists to get right): fitting a single `KMeans(n_clusters=2)` on ALL
    points -- outliers included -- and then flagging distance-based
    outliers from THAT SAME fit is a well-known statistical failure mode
    called "masking": outliers can pull the centroids toward themselves,
    and any spread statistic (e.g. a std-dev) computed over a contaminated
    set is itself inflated by the very points it's meant to catch,
    systematically under-flagging them. This function instead uses an
    ITERATIVE REFIT:
      1. Fit an initial KMeans(n_clusters=2) on ALL points.
      2. Compute each point's distance to its assigned centroid.
      3. Tentatively flag the top `outlier_candidate_fraction` of points by
         distance as CANDIDATE outliers.
      4. REMOVE the candidates and REFIT KMeans(n_clusters=2) on only the
         remaining points -- these REFINED centroids are no longer pulled
         by the outliers.
      5. Re-evaluate EVERY original point's distance to the nearest
         REFINED centroid. The threshold is `2x the MEDIAN distance`
         among the points used to refit (a robust statistic, much less
         outlier-sensitive than a std-dev computed over a contaminated
         set) -- any point exceeding it is flagged `"outlier"`.
      6. Every other point is assigned to `"team_A"`/`"team_B"` by nearest
         refined centroid.

    KNOWN LIMITATION of the `2x median inlier distance` threshold: it is
    only meaningful when inliers have genuine natural color variance. On
    PERFECTLY noiseless synthetic swatches (every "red" player literally
    identical to every other), the median inlier distance collapses toward
    0, making the threshold hypersensitive to even tiny, legitimate
    deviations (e.g. a slightly different-but-still-clearly-red hue can
    get flagged as an outlier alongside genuine outliers). Real jersey
    crops have actual lighting/shadow/compression variance, which keeps
    this threshold non-degenerate in practice; this is a property of
    testing against zero-variance synthetic data, not a flaw in the
    thresholding approach itself, but worth knowing before reading too
    much into a synthetic test that mixes near-duplicate colors with
    perfectly-duplicate ones.

    Returns `{track_id: "team_A" | "team_B" | "outlier"}`.
    """
    track_ids = [p["track_id"] for p in players_data]
    features = _build_cluster_features(players_data)
    n_points = len(players_data)

    if n_points < 2:
        raise ValueError("classify_teams requires at least 2 players to cluster")

    # Step 1: naive single-pass fit on ALL points (kept around so callers/
    # tests can confirm the refit step actually changes anything).
    naive_kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    naive_labels = naive_kmeans.fit_predict(features)
    naive_centroids = naive_kmeans.cluster_centers_

    # Step 2: distance of each point to ITS OWN assigned naive centroid.
    naive_distances = np.linalg.norm(features - naive_centroids[naive_labels], axis=1)

    # Step 3: tentatively flag the largest-distance points as candidates
    # for removal before refitting -- a heuristic fraction, not the final
    # outlier decision (that happens in step 5, against ALL points).
    outlier_candidate_fraction = 0.2
    num_candidates = max(1, round(outlier_candidate_fraction * n_points))
    candidate_outlier_indices = set(
        np.argsort(naive_distances)[-num_candidates:].tolist()
    )

    inlier_mask = np.array([i not in candidate_outlier_indices for i in range(n_points)])
    inlier_features = features[inlier_mask]

    # Step 4: REFIT on the candidate-outlier-free subset.
    refined_kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    refined_kmeans.fit(inlier_features)
    refined_centroids = refined_kmeans.cluster_centers_

    # Step 5: re-evaluate EVERY original point against the REFINED
    # centroids -- distance to whichever refined centroid is nearest.
    distances_to_refined = np.linalg.norm(
        features[:, None, :] - refined_centroids[None, :, :], axis=2
    )  # [n_points, 2]
    nearest_refined_cluster = np.argmin(distances_to_refined, axis=1)
    nearest_refined_distance = np.min(distances_to_refined, axis=1)

    # Threshold from the INLIER set's distances to their own nearest
    # refined centroid -- a robust median, not a std-dev over a
    # contaminated set.
    inlier_nearest_distances = nearest_refined_distance[inlier_mask]
    median_inlier_distance = float(np.median(inlier_nearest_distances))
    outlier_threshold = 2.0 * median_inlier_distance

    roles: dict[int, str] = {}
    for i, track_id in enumerate(track_ids):
        if nearest_refined_distance[i] > outlier_threshold:
            roles[track_id] = "outlier"
        else:
            cluster_index = nearest_refined_cluster[i]
            roles[track_id] = "team_A" if cluster_index == 0 else "team_B"

    return roles
