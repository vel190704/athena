"""Milestone 29 validation: ball detection (YOLO + shape-aware fallback).

BALL DETECTION RELIABILITY WARNING (printed again at the end of this file's
tests, per Step 2.4 -- see `ball_detector.py`'s module docstring for the
full version): ball detection reliability is expected to be LOW on real
broadcast footage due to occlusion, motion blur, and false-positive-prone
round-object confusion (heads, cards, logos). This module provides a
best-effort signal and a shape-filtered fallback, not a guarantee -- and
the fallback's implicit "small motion" assumption is weakest during
exactly the fast passes/shots most valuable to detect.
"""

from pathlib import Path

import cv2
import numpy as np

from production.src.cv.ball_detector import detect_ball, detect_ball_fallback

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Real photograph (CC BY-SA 4.0, Wikimedia Commons -- see fixtures/ATTRIBUTION.md).
# A cv2.circle()-drawn flat disc lacks the lighting/texture/shading signature a
# photo-trained COCO detector actually learned to recognize; testing the YOLO
# path against a synthetic circle would risk a false negative that reflects
# the test image's unrealism, not a bug in detect_ball. This is a real photo
# specifically so the YOLO test is meaningful.
REAL_BALL_PHOTO_PATH = FIXTURES_DIR / "soccer_ball_kick.jpg"

GREEN_BACKGROUND_BGR = (34, 139, 34)
WHITE_BGR = (255, 255, 255)


def test_detect_ball_yolo_on_real_photo():
    """THE YOLO-path test: must run against a REAL photograph (see module
    docstring for why a synthetic circle would not be a meaningful test
    here).
    """
    assert REAL_BALL_PHOTO_PATH.exists(), f"missing test fixture: {REAL_BALL_PHOTO_PATH}"

    result = detect_ball(str(REAL_BALL_PHOTO_PATH))

    assert result is not None, "expected a sports-ball detection on a real photo of a ball being kicked"
    assert "ball_pos_pixels" in result
    assert "bbox" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    print(f"\nYOLO ball detection on real photo {REAL_BALL_PHOTO_PATH.name}:")
    print(f"  ball_pos_pixels={result['ball_pos_pixels']}, bbox={result['bbox']}, "
          f"confidence={result['confidence']:.4f}")


def _make_fallback_test_image():
    """Green background, a small white CIRCLE (the ball) and a larger,
    non-circular white RECTANGLE (a distractor simulating a pitch line
    marking or advertising board) -- both within the fallback's search
    radius of `circle_center` below.
    """
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    canvas[:, :] = GREEN_BACKGROUND_BGR

    circle_center = (150, 150)
    circle_radius = 10
    cv2.circle(canvas, circle_center, circle_radius, WHITE_BGR, thickness=-1)

    # Rectangle: area (50*20=1000px) is LARGER than the circle's
    # (pi*10^2=~314px) but its circularity (4*pi*area/perimeter^2 =~0.64)
    # is well below the 0.7 threshold -- this is what actually
    # distinguishes it from the ball, not its size.
    rectangle_top_left = (190, 140)
    rectangle_bottom_right = (240, 160)
    cv2.rectangle(canvas, rectangle_top_left, rectangle_bottom_right, WHITE_BGR, thickness=-1)

    return canvas, circle_center


def test_detect_ball_fallback_rejects_larger_noncircular_distractor():
    """THE key robustness proof for this milestone: the fallback must
    return the small CIRCLE's centroid, not the larger RECTANGLE, proving
    the circularity/size filter actually discriminates by shape -- a
    single-object test could not prove this (it could pass even with a
    naive "biggest bright blob" implementation).
    """
    image, circle_center = _make_fallback_test_image()

    result = detect_ball_fallback(image, prev_ball_pos=list(circle_center), search_radius_px=100.0)

    assert result is not None, "expected the fallback to find the circle"
    print(f"\nFallback result: {result}")

    recovered_x, recovered_y = result["ball_pos_pixels"]
    assert abs(recovered_x - circle_center[0]) < 3.0, (
        f"recovered position {result['ball_pos_pixels']} is not close to the circle's true "
        f"center {circle_center} -- the fallback may have picked the rectangle instead"
    )
    assert abs(recovered_y - circle_center[1]) < 3.0

    # Explicit check that the returned box is the SMALL circle, not the
    # larger rectangle (rectangle area is 1000px, circle's bounding box
    # area is ~400px for a radius-10 circle -- comfortably distinct).
    bbox_area = result["bbox"][2] * result["bbox"][3]
    assert bbox_area < 700, f"returned bbox area {bbox_area} looks like the rectangle, not the circle"

    assert result["circularity"] > 0.7


def test_detect_ball_fallback_returns_none_when_only_distractor_in_range():
    """Sanity check on the filter itself, isolated: with ONLY the
    rectangle (no circle) in the image, the fallback must return None
    rather than falling back to the rectangle as "the best available"
    match.
    """
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    canvas[:, :] = GREEN_BACKGROUND_BGR
    cv2.rectangle(canvas, (190, 140), (240, 160), WHITE_BGR, thickness=-1)

    result = detect_ball_fallback(canvas, prev_ball_pos=[150.0, 150.0], search_radius_px=100.0)
    assert result is None


def test_no_ball_anywhere_both_paths_return_none_cleanly():
    """Neither a circle nor a rectangle -- plain background. Both
    `detect_ball` (YOLO) and `detect_ball_fallback` must return `None`
    without crashing, confirming graceful degradation when the ball
    simply isn't visible (occlusion, off-screen) -- a common real
    scenario, not an edge case to special-case away.
    """
    blank_image = np.zeros((300, 400, 3), dtype=np.uint8)
    blank_image[:, :] = GREEN_BACKGROUND_BGR

    yolo_result = detect_ball(blank_image)
    fallback_result = detect_ball_fallback(blank_image, prev_ball_pos=[150.0, 150.0], search_radius_px=100.0)

    assert yolo_result is None
    assert fallback_result is None

    print(
        "\nBall detection reliability is expected to be low on real broadcast footage due to "
        "occlusion, motion blur, and false-positive-prone round-object confusion (heads, cards, "
        "logos). This module provides a best-effort signal and a shape-filtered fallback, not a "
        "guarantee -- and the fallback's implicit 'small motion' assumption is weakest during "
        "exactly the fast passes/shots most valuable to detect."
    )
