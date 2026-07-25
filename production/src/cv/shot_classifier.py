"""Milestone 31 (Module 4): shot/camera-cut classification.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25 -- nothing in `production/src/models`, `production/src/
pipeline`, `production/src/spatial`, `production/src/physics`, or
`production/src/serving` is imported by or imports from this module.

PURPOSE: a lightweight, 2-feature heuristic gate deciding whether a frame
looks like a usable "tactical wide view" (grass-dominated, structurally
edge-rich -- pitch lines, players) before it's worth running the rest of
the CV pipeline (detection, tracking, the Milestone 30 adapter) on it at
all. This is a CHEAP PRE-FILTER, not a scene classifier -- it measures
exactly two signals (green-pixel ratio, Canny edge density) and nothing
about semantic content.

TWO REAL LIMITATIONS THIS HEURISTIC CANNOT SOLVE -- read before assuming a
passing frame is trustworthy:

1. **Replay vs. live.** This heuristic CANNOT distinguish a REPLAY of a
   genuine tactical wide shot from a LIVE tactical wide shot -- the two are
   visually near-identical (same grass, same lines, same player density) by
   the EXACT features this classifier measures. A slow-motion replay of a
   goal looks, to green-ratio and edge-density, like the live moment it
   replays. Distinguishing them requires a different signal entirely (an
   on-screen replay graphic, slow-motion frame-rate detection, or a
   scoreboard/timestamp discontinuity) -- explicitly out of scope here, and
   a candidate for a future ADR if it becomes necessary (e.g. once
   possession-chain timing starts double-counting replayed action as if it
   were new live play).

2. **Zoom changes within an otherwise-valid view.** This heuristic CANNOT
   detect a camera ZOOM change within what still looks like a tactical
   view. A frame can pass this filter (grass-dominated, structured edges)
   while the camera has zoomed in/out from the angle the CURRENT homography
   (`calibration.py`, Milestone 27) was calibrated against -- meaning a
   PASSING frame is NOT a guarantee that the existing calibration is still
   valid. This connects directly to Milestone 27's own documented scope
   note that it "does not yet handle continuous re-calibration across
   panning/zooming feeds": that gap and this classifier's blind spot are
   the SAME underlying problem (no continuous camera-motion awareness), not
   two unrelated limitations that happen to sit in different modules.

ASYMMETRIC COST PRINCIPLE (a design goal for future threshold tuning, not
yet acted on since no real-footage tuning has happened -- see below): a
FALSE NEGATIVE here (skipping a genuinely valid frame) costs exactly one
lost data point. A FALSE POSITIVE (processing a close-up/invalid frame as
if it were a valid tactical view) feeds FABRICATED player positions
directly into `BiomechanicalPitchControl` and DeepHit -- a materially more
expensive mistake, since it corrupts a training/inference sample rather
than merely omitting one. Future threshold tuning on real footage should
deliberately bias toward skipping when uncertain, not toward maximizing
raw classification accuracy symmetrically.

`green_ratio_threshold=0.25` and `min_edge_density=0.05` are UNVALIDATED
GUESSES pending real broadcast footage (SoccerNet NDA access still pending
per Milestone 25) -- consistent with how every other hand-tuned constant in
this project (Kalman Q/R, pitch-control radii, outlier-detection
thresholds) has been flagged as provisional until validated against real
data, not presented as tuned.
"""

import cv2
import numpy as np

# Green hue band in OpenCV's 0-179 H scale (roughly 70-170 degrees on the
# standard 0-360 hue circle) -- grass under broadcast lighting typically
# falls within this band. S/V thresholds are on the NORMALIZED [0, 1]
# scale (raw OpenCV S/V, which are 0-255, divided by 255) -- excludes
# desaturated/very dark pixels (shadow, out-of-focus dark regions) that
# happen to fall in the green hue band without visually reading as grass.
GREEN_HUE_MIN = 35
GREEN_HUE_MAX = 85
GREEN_SATURATION_MIN = 0.2
GREEN_VALUE_MIN = 0.2


def compute_shot_features(image: np.ndarray, canny_low: int = 50, canny_high: int = 150) -> tuple[float, float]:
    """Computes the two raw signals `is_tactical_view` thresholds:
    `(green_ratio, edge_density)`, each a fraction in `[0, 1]`.

    `canny_low`/`canny_high` are exposed as explicit named parameters
    (Canny's hysteresis thresholds), not hidden inside a hardcoded call --
    consistent with this project's practice of never hiding hand-tuned
    constants (Kalman Q/R, pitch-control radii, outlier thresholds, etc.).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    value = hsv[:, :, 2].astype(np.float64) / 255.0

    green_mask = (
        (hue >= GREEN_HUE_MIN)
        & (hue <= GREEN_HUE_MAX)
        & (saturation > GREEN_SATURATION_MIN)
        & (value > GREEN_VALUE_MIN)
    )
    green_ratio = float(green_mask.mean())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_density = float((edges > 0).mean())

    return green_ratio, edge_density


def is_tactical_view(
    image: np.ndarray,
    green_ratio_threshold: float = 0.25,
    min_edge_density: float = 0.05,
    canny_low: int = 50,
    canny_high: int = 150,
) -> bool:
    """`True` if `image` looks like a usable tactical wide view: BOTH
    `green_ratio > green_ratio_threshold` AND
    `edge_density > min_edge_density`. See this module's docstring for the
    two real limitations this 2-feature heuristic cannot solve (replay-vs-
    live; zoom changes within an otherwise-valid view) and the asymmetric
    cost principle that should guide future threshold tuning on real
    footage.
    """
    green_ratio, edge_density = compute_shot_features(image, canny_low=canny_low, canny_high=canny_high)
    return green_ratio > green_ratio_threshold and edge_density > min_edge_density
