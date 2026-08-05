"""Milestone 25 validation: SoccerNet acquisition + baseline YOLO
player-detection accuracy.

Three layers, deliberately kept separate:
  1. Pure geometry/matching unit tests (`metrics.py`) -- run unconditionally,
     no external data or model download required.
  2. A password-handling check for `acquisition.download_sample_dataset` --
     runs unconditionally; confirms the "no password" path behaves exactly
     as designed (Step 2.1) rather than crashing or silently no-op-ing.
  3. The real SoccerNet-data end-to-end detection/metrics test -- SKIPPED,
     not failed, if `data/raw/soccernet/tracking/train` isn't populated.
     SoccerNet's tracking dataset requires a real, individually-issued NDA
     password (see `acquisition.py`'s module docstring) that this test
     environment does not have; this is an expected, documented gap, not a
     bug in this milestone's implementation.
"""


import logging

import pytest

from production.src.cv.acquisition import (
    SOCCERNET_LOCAL_DIR,
    TRACKING_TASK,
    download_sample_dataset,
    load_sample_frames_and_labels,
)
from production.src.cv.detector import run_baseline_detection
from production.src.cv.metrics import (
    compute_iou,
    compute_precision_recall_f1,
    match_detections_to_ground_truth,
)

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
NUM_SAMPLE_FRAMES = 10


# ============================================================================
# Layer 1: pure geometry/matching unit tests -- no external data needed.
# ============================================================================

def test_compute_iou_known_cases():
    identical = [10.0, 10.0, 20.0, 20.0]
    assert compute_iou(identical, identical) == pytest.approx(1.0)

    non_overlapping = [100.0, 100.0, 20.0, 20.0]
    assert compute_iou(identical, non_overlapping) == 0.0

    # box_a = [0,0,10,10] (area 100), box_b = [5,5,10,10] (area 100),
    # intersection = [5,5,5,5] (area 25), union = 100+100-25 = 175
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [5.0, 5.0, 10.0, 10.0]
    assert compute_iou(box_a, box_b) == pytest.approx(25.0 / 175.0)


def test_match_detections_deduplicates_greedily():
    """Two predictions both overlap the SAME single ground-truth box, with
    different IoUs. A naive "count every pair independently" approach
    would double-count this as 2 true positives; the greedy one-to-one
    matcher must assign only the higher-IoU prediction, leaving the other
    as an unmatched (false-positive) prediction.
    """
    ground_truths = [{"bbox": [0.0, 0.0, 10.0, 10.0]}]
    predictions = [
        {"bbox": [0.0, 0.0, 10.0, 10.0]},  # IoU = 1.0 with the GT
        {"bbox": [1.0, 1.0, 10.0, 10.0]},  # partial overlap, lower IoU
    ]

    matches, unmatched_preds, unmatched_gts = match_detections_to_ground_truth(
        predictions, ground_truths, iou_threshold=IOU_THRESHOLD
    )

    assert len(matches) == 1
    matched_pred_index, matched_gt_index, iou = matches[0]
    assert matched_pred_index == 0  # the perfect-IoU prediction wins the match
    assert matched_gt_index == 0
    assert iou == pytest.approx(1.0)
    assert unmatched_preds == [1]  # the second prediction is an unmatched FP
    assert unmatched_gts == []


def test_compute_precision_recall_f1_known_scenario():
    """2 frames, hand-computed expected P/R/F1:
      frame 0: 2 GT boxes, 2 predictions, both match -> tp=2, fp=0, fn=0
      frame 1: 1 GT box, 2 predictions (1 matches, 1 is a spurious FP,
               e.g. a crowd detection) -> tp=1, fp=1, fn=0
    Aggregate: tp=3, fp=1, fn=0 -> precision=3/4=0.75, recall=3/3=1.0,
    f1 = 2*0.75*1.0/(0.75+1.0) = 0.857142...
    """
    ground_truths_by_frame = {
        0: [{"bbox": [0.0, 0.0, 10.0, 10.0]}, {"bbox": [50.0, 50.0, 10.0, 10.0]}],
        1: [{"bbox": [0.0, 0.0, 10.0, 10.0]}],
    }
    predictions_by_frame = {
        0: [{"bbox": [0.0, 0.0, 10.0, 10.0]}, {"bbox": [50.0, 50.0, 10.0, 10.0]}],
        1: [{"bbox": [0.0, 0.0, 10.0, 10.0]}, {"bbox": [200.0, 200.0, 10.0, 10.0]}],
    }

    result = compute_precision_recall_f1(
        predictions_by_frame, ground_truths_by_frame, iou_threshold=IOU_THRESHOLD
    )

    assert result["total_tp"] == 3
    assert result["total_fp"] == 1
    assert result["total_fn"] == 0
    assert result["precision"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(2 * 0.75 * 1.0 / (0.75 + 1.0))


# ============================================================================
# Layer 2: password-handling check (Step 5.1) -- runs unconditionally.
# ============================================================================

def test_download_sample_dataset_without_password_logs_instructions_and_returns_none(
    caplog, monkeypatch
):
    """Engineering-hygiene pass: acquisition.py converted from print() to
    logging (module-level logger.getLogger(__name__), no basicConfig at
    import time -- same pattern as train.py/alert_store.py). This
    instructional message now goes through logger.warning(), not stdout
    -- observed via caplog, not capsys."""
    monkeypatch.delenv("SOCCERNET_PASSWORD", raising=False)

    with caplog.at_level(logging.WARNING, logger="production.src.cv.acquisition"):
        result = download_sample_dataset(num_games=1, password=None)

    assert result is None
    assert "NDA" in caplog.text or "password" in caplog.text.lower()
    assert "soccer-net.org" in caplog.text


# ============================================================================
# Layer 3: real-data end-to-end test -- SKIPPED without cached SoccerNet data.
# ============================================================================

def _soccernet_train_dir_available():
    train_dir = SOCCERNET_LOCAL_DIR / TRACKING_TASK / "train"
    if not train_dir.exists():
        return None
    sequence_dirs = [p for p in train_dir.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists()]
    return train_dir if sequence_dirs else None


def _select_diverse_frame_ids(frame_paths: dict, num_frames: int) -> list[int]:
    """Spreads `num_frames` frame IDs as evenly as possible across the full
    available range in `frame_paths` -- NOT the first `num_frames`
    consecutive frames, which would be near-duplicate broadcast images
    (same camera angle, same instant) and give a falsely stable, non-
    representative metric.
    """
    sorted_ids = sorted(frame_paths.keys())
    if len(sorted_ids) <= num_frames:
        return sorted_ids
    step = len(sorted_ids) / num_frames
    return [sorted_ids[int(i * step)] for i in range(num_frames)]


def test_soccernet_baseline_detection_accuracy():
    """Step 4: real SoccerNet broadcast footage, real YOLO detections, real
    IoU-matched P/R/F1. SKIPPED (not failed) if no cached data is present --
    see this file's module docstring and `acquisition.py` for why.
    """
    train_dir = _soccernet_train_dir_available()
    if train_dir is None:
        pytest.skip(
            f"No cached SoccerNet tracking data found at {SOCCERNET_LOCAL_DIR / TRACKING_TASK / 'train'}. "
            "SoccerNet's tracking dataset requires a real, NDA-issued download password -- run "
            "production.src.cv.acquisition.download_sample_dataset(password=...) with a real "
            "password first (see that function's printed instructions for how to obtain one)."
        )

    sequence_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    print(f"\n[test_cv_detector] {len(sequence_dirs)} sequence(s) available under {train_dir}")

    # Spread the 10 sample frames across AS MANY of the available sequences
    # as possible (real clip diversity), falling back to spreading across
    # time within a single sequence if only one is available -- explicitly
    # noted as a diversity limitation in that case, per Step 4.2.
    frames_needed = NUM_SAMPLE_FRAMES
    sequences_to_use = sequence_dirs if len(sequence_dirs) <= frames_needed else sequence_dirs[:frames_needed]
    frames_per_sequence = max(1, frames_needed // len(sequences_to_use))

    if len(sequence_dirs) == 1:
        print(
            "[test_cv_detector] Only 1 sequence available -- diversity is limited to spacing "
            "frames across TIME within this single clip, not across different clips."
        )

    selected_frames = []  # list of (sequence_dir_name, frame_id, image_path, gt_boxes)
    for sequence_dir in sequences_to_use:
        data = load_sample_frames_and_labels(sequence_dir.name, extract_dir=train_dir)
        diverse_ids = _select_diverse_frame_ids(data["frame_paths"], frames_per_sequence)
        for frame_id in diverse_ids:
            selected_frames.append(
                (sequence_dir.name, frame_id, data["frame_paths"][frame_id], data["labels"].get(frame_id, []))
            )
        if len(selected_frames) >= frames_needed:
            break
    selected_frames = selected_frames[:frames_needed]

    assert len(selected_frames) > 0, "no frames could be selected from the available SoccerNet data"

    predictions_by_frame = {}
    ground_truths_by_frame = {}
    print(f"\n=== Per-frame detection ({len(selected_frames)} frames, conf>={CONFIDENCE_THRESHOLD}) ===")
    for i, (seq_name, frame_id, image_path, gt_boxes) in enumerate(selected_frames):
        predictions = run_baseline_detection(image_path, confidence_threshold=CONFIDENCE_THRESHOLD)
        predictions_by_frame[i] = predictions
        ground_truths_by_frame[i] = gt_boxes
        print(f"  frame {i}: sequence={seq_name} frame_id={frame_id} "
              f"predictions={len(predictions)} ground_truth_boxes={len(gt_boxes)}")

    result = compute_precision_recall_f1(
        predictions_by_frame, ground_truths_by_frame, iou_threshold=IOU_THRESHOLD
    )

    print(f"\n=== Aggregate metrics (single operating point: conf={CONFIDENCE_THRESHOLD}, "
          f"IoU>={IOU_THRESHOLD} -- NOT full mAP, see metrics.py docstring) ===")
    print(f"  TP={result['total_tp']} FP={result['total_fp']} FN={result['total_fn']}")
    print(f"  Precision: {result['precision']:.4f} (reported, NOT gated -- see rationale below)")
    print(f"  Recall:    {result['recall']:.4f} (hard-gated: must exceed 0.5)")
    print(f"  F1:        {result['f1']:.4f} (reported, NOT gated)")

    # Rough, explicitly-labeled off-pitch/crowd spot-check: since pitch
    # calibration doesn't exist yet (Phase 27), this is a HEURISTIC based on
    # vertical position only, not a real determination -- flagged as such.
    print("\n=== Rough false-positive spot-check (heuristic, NOT real pitch calibration) ===")
    likely_crowd_count = 0
    likely_onpitch_count = 0
    for frame_index, frame_result in result["per_frame"].items():
        predictions = predictions_by_frame[frame_index]
        for pred_index in frame_result["unmatched_pred_indices"]:
            fp_box = predictions[pred_index]["bbox"]
            fp_y_center = fp_box[1] + fp_box[3] / 2.0
            # Broadcast wide shots typically place the crowd/stands in the
            # upper portion of the frame and the pitch lower/center -- a
            # rough heuristic only, explicitly not a substitute for real
            # calibration (Phase 27).
            if fp_y_center < 300:
                likely_crowd_count += 1
            else:
                likely_onpitch_count += 1
    total_fp = result["total_fp"]
    print(f"  {likely_crowd_count}/{total_fp} false positives in the upper-frame region "
          "(heuristically consistent with crowd/stands)")
    print(f"  {likely_onpitch_count}/{total_fp} false positives lower in the frame "
          "(likely on-pitch misclassifications, staff, or ball boys)")

    assert result["recall"] > 0.5, (
        f"Recall={result['recall']:.4f} is unexpectedly low -- a person-detector should find most "
        "real players/officials even amid crowd noise; this would indicate a genuine detection "
        "problem, unlike low precision (expected and diagnostic, not gated)."
    )
