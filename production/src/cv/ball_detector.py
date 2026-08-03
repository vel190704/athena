"""Milestone 29 (Module 4): ball detection -- pretrained YOLO plus a
shape-aware fallback heuristic.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25 -- nothing in `production/src/models`, `production/src/
pipeline`, `production/src/spatial`, `production/src/physics`, or
`production/src/serving` is imported by or imports from this module.

RELIABILITY NOTE (read before treating this module's output as trustworthy
without a downstream sanity check): ball detection is, honestly, the LEAST
reliable component of this CV pipeline. The ball is small, moves fast,
is frequently occluded by players, and generic COCO "sports ball"
detections are prone to false-positive on other round objects in a
broadcast frame (a player's head at distance, a referee's card, a
ball-shaped sponsor logo). The shape-aware fallback below is a best-effort
backstop, not a guarantee -- and its own core assumption ("the ball hasn't
moved far since the last known position") is weakest during exactly the
fast through-balls, crosses, and shots that matter most for tactical
prediction. Treat both paths as a signal to be sanity-checked (e.g.
against plausible ball speed, per Milestone 26's `LIKELY_ID_SWITCH_PIXEL_
VELOCITY_CEILING` precedent), not as ground truth.

Bounding boxes/positions here use the same top-left-origin `[x, y, w, h]`
pixel convention as `detector.py`/`tracker.py`. Positions are returned in
PIXEL space (`ball_pos_pixels`), never calibrated meters -- consistent
with Milestone 26's `vel_pixels_per_sec` unit-honesty convention.
Calibration (`calibration.py`, Milestone 27) is a separate downstream step
this module does not perform.
"""

import math

import cv2
import numpy as np

from production.src.cv.detector import _load_model

# COCO class 32 is "sports ball" in every standard Ultralytics
# COCO-pretrained checkpoint.
COCO_SPORTS_BALL_CLASS_ID = 32

# Intentionally LOWER than the person detector's 0.5 (Milestone 25). The
# ball is small and easy for the detector to miss; this is a deliberate
# tradeoff accepting more false positives in exchange for not missing real
# detections, not an oversight.
DEFAULT_BALL_CONFIDENCE_THRESHOLD = 0.3

# REAL-FOOTAGE FINDING, not a hypothetical: on an actual 1284x728 broadcast
# tactical-wide-shot clip, the ball's true pixel footprint is only ~5x5px
# (a player at median bbox height ~33px implies a ~4px ball, by real-world
# player-height/ball-diameter ratio). Ultralytics' `model.predict()`
# defaults `imgsz` to 640, which downscales a frame this size just enough
# to erase that object entirely -- direct A/B testing on this clip found
# ZERO "sports ball" class candidates at ANY confidence down to 0.01 at the
# 640 default, across every sampled frame, versus a real, moving,
# plausible-confidence (~0.3-0.7) detection recovered at imgsz=1920 on the
# majority of sampled frames. Like every other hand-tuned constant in this
# project, 1920 is evidence-based on exactly ONE real clip, not a
# generally-validated default -- a fixed pixel value also does not scale
# correctly to a different native video resolution (upscaling an
# already-larger frame to 1920 would shrink it instead); a longer-term fix
# should size this relative to the source video's own resolution, not a
# hardcoded constant.
DEFAULT_BALL_DETECTION_IMGSZ = 1920


def detect_ball(
    image,
    confidence_threshold: float = DEFAULT_BALL_CONFIDENCE_THRESHOLD,
    model_checkpoint: str = "yolov8m.pt",
    imgsz: int = DEFAULT_BALL_DETECTION_IMGSZ,
) -> dict | None:
    """Runs a pretrained (COCO-trained) YOLO detector on `image` and
    returns the highest-confidence "sports ball" (COCO class 32)
    detection, or `None` if nothing scored above `confidence_threshold`.

    `image` may be a file path or an already-loaded BGR numpy array
    (ultralytics' `predict` accepts either).

    If multiple candidate balls are detected, the HIGHEST-CONFIDENCE one
    is returned -- but on real broadcast footage, generic COCO "sports
    ball" detections are prone to false-positive on other round objects
    (a player's head at distance, a referee's card, a ball-shaped sponsor
    logo). This is an EXPECTED, not hypothetical, limitation -- see this
    module's docstring.

    Returns `{"ball_pos_pixels": [x, y], "bbox": [x, y, w, h],
    "confidence": float}` (top-left-origin pixel position/box, matching
    `detector.py`/`tracker.py`'s convention), or `None`.

    `imgsz` defaults to `DEFAULT_BALL_DETECTION_IMGSZ` (see that constant's
    comment) rather than Ultralytics' own built-in 640 -- the ball is small
    enough, relative to a typical broadcast wide shot, that the default
    resize can erase it before the model ever sees it.
    """
    model = _load_model(model_checkpoint)
    results = model.predict(source=image, conf=confidence_threshold, imgsz=imgsz, verbose=False)

    best_detection = None
    for result in results:
        boxes = result.boxes
        for box_xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            if int(cls_id.item()) != COCO_SPORTS_BALL_CLASS_ID:
                continue
            confidence = conf.item()
            if best_detection is not None and confidence <= best_detection["confidence"]:
                continue

            x1, y1, x2, y2 = (v.item() for v in box_xyxy)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            best_detection = {
                "ball_pos_pixels": [x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0],
                "bbox": bbox,
                "confidence": confidence,
            }

    return best_detection


def detect_ball_fallback(
    image: np.ndarray,
    prev_ball_pos: list[float],
    search_radius_px: float = 100.0,
    min_ball_area_px: float = 20.0,
    max_ball_area_px: float = 2000.0,
    circularity_threshold: float = 0.7,
    brightness_threshold: int = 180,
) -> dict | None:
    """Shape-aware fallback for when `detect_ball` finds nothing: searches
    a `search_radius_px` window around `prev_ball_pos` for a small, round,
    bright blob.

    Deliberately NOT "biggest bright blob in the search window" -- real
    broadcast frames routinely contain OTHER, LARGER bright objects near
    the ball (pitch line markings, advertising boards, goalkeeper gloves,
    sock tape) that a pure brightness/size heuristic would pick in error.
    Every candidate contour found in the thresholded search window is
    filtered on BOTH:
      - `circularity` = `4*pi*area / perimeter^2` (~1.0 for a true circle,
        much lower for elongated/irregular shapes like a line marking or
        an advertising board edge) -- must exceed `circularity_threshold`.
      - `area` (in pixels) -- must fall within
        `[min_ball_area_px, max_ball_area_px]`, an explicit, tunable range
        (a ball's apparent pixel size depends heavily on camera distance/
        zoom, so this range should be tuned per-source, not trusted as a
        universal constant).
    Among candidates passing BOTH filters, the one CLOSEST to
    `prev_ball_pos` is returned -- not necessarily the largest or most
    circular -- since proximity to the last known position is this
    fallback's actual detection signal.

    CORE ASSUMPTION, STATED EXPLICITLY: this fallback assumes the ball
    hasn't moved far since `prev_ball_pos`. That assumption is WEAKEST
    exactly during fast through-balls, crosses, and shots -- the
    highest-value moments for tactical prediction. A fallback miss during
    exactly these moments should be expected, not treated as a rare edge
    case.

    Returns `{"ball_pos_pixels": [x, y], "bbox": [x, y, w, h],
    "circularity": float}` in ORIGINAL image coordinates, or `None` if no
    candidate passed both filters.
    """
    image_h, image_w = image.shape[:2]
    prev_x, prev_y = prev_ball_pos

    crop_x1 = max(0, round(prev_x - search_radius_px))
    crop_y1 = max(0, round(prev_y - search_radius_px))
    crop_x2 = min(image_w, round(prev_x + search_radius_px))
    crop_y2 = min(image_h, round(prev_y + search_radius_px))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _thresh_value, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_distance = math.inf
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_ball_area_px or area > max_ball_area_px:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter**2)
        if circularity < circularity_threshold:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        # Convert back to ORIGINAL image coordinates (contour coords are
        # relative to the cropped search window).
        center_x = crop_x1 + x + w / 2.0
        center_y = crop_y1 + y + h / 2.0

        distance = math.hypot(center_x - prev_x, center_y - prev_y)
        if distance < best_distance:
            best_distance = distance
            best_candidate = {
                "ball_pos_pixels": [center_x, center_y],
                "bbox": [crop_x1 + x, crop_y1 + y, w, h],
                "circularity": circularity,
            }

    return best_candidate
