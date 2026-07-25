"""Milestone 25 (Module 4): baseline person detection on broadcast frames.

Standalone from every existing ML/physics/API module -- see
`acquisition.py`'s module docstring for the isolation statement that
applies equally here.
"""

from pathlib import Path

from ultralytics import YOLO

# COCO class 0 is "person" in every standard Ultralytics COCO-pretrained
# checkpoint (the class-index-to-name mapping is fixed by the COCO dataset
# convention these checkpoints were trained on, not something this project
# defines).
COCO_PERSON_CLASS_ID = 0

_model_cache: dict[str, YOLO] = {}


def _load_model(model_checkpoint: str) -> YOLO:
    """Cached by checkpoint name so repeated calls (e.g. once per frame in
    the validation test) don't reload the network weights every time."""
    if model_checkpoint not in _model_cache:
        _model_cache[model_checkpoint] = YOLO(model_checkpoint)
    return _model_cache[model_checkpoint]


def run_baseline_detection(
    image_path: str | Path,
    confidence_threshold: float = 0.5,
    model_checkpoint: str = "yolov8m.pt",
) -> list[dict]:
    """Runs a pretrained (COCO-trained) YOLO detector on one image and
    returns predicted human bounding boxes.

    `model_checkpoint` is a named, pinned default -- not an implicit
    "whatever ultralytics defaults to" choice -- so any P/R/F1 numbers
    computed from this function's output are reproducible against a known,
    citable checkpoint, the same discipline this project applies to citing
    exact MLflow run_ids for anything logged or compared against later.

    SCOPE NOTE (read before interpreting results): this filters to COCO
    class 0 ("person") only. That single class is matched against ALL
    human ground-truth categories (player, goalkeeper, referee) for this
    baseline's purposes, since role/team distinction is out of scope until
    Phase 28. It will ALSO inevitably detect spectators, staff, ball boys,
    and anyone else visible in a broadcast wide shot, since there is no
    pitch-region restriction available yet -- that requires Phase 27's
    calibration work. Precision being crushed by these off-pitch/crowd
    detections is an EXPECTED, INFORMATIVE limitation of running detection
    before calibration exists, not a bug this function should try to work
    around.

    Returns a list of `{"bbox": [x, y, w, h], "confidence": float}` dicts,
    in the same top-left-origin `[x, y, w, h]` pixel format as the
    ground-truth boxes from `acquisition.load_sample_frames_and_labels`
    (YOLO's native output is `[x1, y1, x2, y2]`; this function converts).
    """
    model = _load_model(model_checkpoint)
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for box_xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            if int(cls_id.item()) != COCO_PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = (v.item() for v in box_xyxy)
            detections.append(
                {"bbox": [x1, y1, x2 - x1, y2 - y1], "confidence": conf.item()}
            )

    return detections
