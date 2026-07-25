"""Milestone 27 (Module 4): pitch calibration / homography.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25 -- nothing in `production/src/models`, `production/src/
pipeline`, `production/src/spatial`, `production/src/physics`, or
`production/src/serving` is imported by or imports from this module.

SCOPE: this module computes the PROJECTION MATH given KNOWN pixel<->meter
correspondences (`cv2.findHomography` + `cv2.perspectiveTransform`). It
does NOT detect pitch lines/keypoints automatically in a broadcast frame --
automatic keypoint detection (e.g. via SoccerNet's own calibration model)
is deferred to a future phase. It also does NOT handle continuous
re-calibration across a panning/zooming broadcast feed -- this milestone
validates the homography math for a SINGLE FIXED camera view/homography
only. A panning/zooming feed would need a fresh homography (or continuous
re-estimation) per shot/cut, which this module does not attempt.

Coordinate convention: destination ("meter") points are in this project's
EXISTING 100 x 68 pitch space (ADR-002) -- the same space
`feature_extractor.py`/`control.py` already operate in. This module
produces coordinates in that space; it does not redefine it, and nothing
here writes into or imports from those existing modules.
"""

import cv2
import numpy as np


def compute_homography(src_pixels, dst_meters, method: int = 0) -> np.ndarray:
    """Computes the 3x3 homography mapping `src_pixels` (broadcast pixel
    coordinates) onto `dst_meters` (this project's 100x68m pitch space),
    via `cv2.findHomography`.

    `method` defaults to `0` (plain least-squares / DLT), appropriate for
    a small number of EXACT, manually-provided correspondences -- this
    milestone's ground-truth pitch landmarks (pitch corners, spot
    markings), known precisely, with no detection noise. Once a future
    phase (31+) introduces AUTOMATICALLY-DETECTED keypoints -- inherently
    noisier, with the possibility of outlier mismatches -- `method=cv2.RANSAC`
    should be used instead, so that a small number of bad correspondences
    don't corrupt the whole homography estimate the way they would under
    plain least-squares.

    `src_pixels`/`dst_meters`: array-like of `(u, v)` / `(x, y)` pairs, at
    least 4 non-collinear points (the classic ill-conditioning trap here is
    calibrating from near-collinear points, e.g. all along one touchline or
    the center circle only -- prefer well-spread landmarks like the pitch
    corners).

    Returns the 3x3 homography matrix (`None` if `cv2.findHomography`
    could not find a solution -- callers should check for this rather than
    assume success).
    """
    src = np.asarray(src_pixels, dtype=np.float32)
    dst = np.asarray(dst_meters, dtype=np.float32)
    matrix, _inlier_mask = cv2.findHomography(src, dst, method=method)
    return matrix


def transform_points(matrix: np.ndarray, pixel_points) -> np.ndarray:
    """Applies a homography (as returned by `compute_homography`) to a
    plain list/array of `[u, v]` pixel points, returning an `(N, 2)` numpy
    array of transformed meter-space points.

    `cv2.perspectiveTransform` requires points shaped as an `(N, 1, 2)`
    float32 array -- a common OpenCV shape gotcha (passing a plain `(N, 2)`
    array raises or silently misbehaves depending on the cv2 build). This
    function handles that reshape internally so callers can pass a plain
    `[[u, v], ...]` list or `(N, 2)` array without needing to know about it.
    """
    points = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, matrix)
    return transformed.reshape(-1, 2)
