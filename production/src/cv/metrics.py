"""Milestone 25: IoU-based detection matching and P/R/F1 computation.

Kept separate from `detector.py` (runs the model) and `acquisition.py`
(fetches data) specifically so this geometry/matching logic can be
unit-tested with synthetic boxes, independent of a real YOLO checkpoint or
real (NDA-gated) SoccerNet data being available.
"""


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU between two `[x, y, w, h]` boxes (top-left origin, pixel units,
    matching both the ground-truth format from `acquisition.py` and
    `detector.run_baseline_detection`'s output format)."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_detections_to_ground_truth(
    predictions: list[dict], ground_truths: list[dict], iou_threshold: float = 0.5
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy, ONE-TO-ONE (COCO-style) IoU matching between `predictions`
    and `ground_truths` (each a list of `{"bbox": [x, y, w, h], ...}`
    dicts).

    Deliberately NOT "count every prediction/GT pair with IoU >= threshold
    independently" -- that would double-count true positives whenever a
    box overlaps multiple candidates on either side (common in a crowded
    broadcast frame), silently inflating precision/recall. Instead: every
    (prediction, GT) pair with IoU >= `iou_threshold` is a CANDIDATE match;
    candidates are sorted by IoU descending and greedily assigned, with
    each prediction and each GT box used in AT MOST ONE final match.

    Returns `(matches, unmatched_pred_indices, unmatched_gt_indices)`,
    where `matches` is a list of `(pred_index, gt_index, iou)` triples.
    """
    candidates = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truths):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pi, gi in candidates:
        if pi in matched_pred_indices or gi in matched_gt_indices:
            continue
        matched_pred_indices.add(pi)
        matched_gt_indices.add(gi)
        matches.append((pi, gi, iou))

    unmatched_pred_indices = [i for i in range(len(predictions)) if i not in matched_pred_indices]
    unmatched_gt_indices = [i for i in range(len(ground_truths)) if i not in matched_gt_indices]
    return matches, unmatched_pred_indices, unmatched_gt_indices


def compute_precision_recall_f1(
    predictions_by_frame: dict, ground_truths_by_frame: dict, iou_threshold: float = 0.5
) -> dict:
    """Aggregate (micro-averaged) P/R/F1 across all frames in
    `predictions_by_frame` (`frame_id -> [prediction dicts]`) against
    `ground_truths_by_frame` (`frame_id -> [ground-truth dicts]`), via
    `match_detections_to_ground_truth` per frame. TP/FP/FN are SUMMED
    across all frames before dividing -- a micro-average across the pooled
    set of all boxes, not an average of per-frame ratios (which would
    weight a frame with few boxes as heavily as one with many).

    SINGLE FIXED operating point (`iou_threshold` for matching,
    `confidence_threshold` baked into whichever predictions were passed
    in). This is NOT full mAP (a precision-recall curve integrated across
    multiple confidence thresholds) -- mAP is explicitly out of scope for
    this milestone. Do not read `f1` here as a substitute for it.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_frame = {}

    for frame_id, predictions in predictions_by_frame.items():
        ground_truths = ground_truths_by_frame.get(frame_id, [])
        matches, unmatched_preds, unmatched_gts = match_detections_to_ground_truth(
            predictions, ground_truths, iou_threshold=iou_threshold
        )
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_frame[frame_id] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "unmatched_pred_indices": unmatched_preds,
            "unmatched_gt_indices": unmatched_gts,
        }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_frame": per_frame,
    }
