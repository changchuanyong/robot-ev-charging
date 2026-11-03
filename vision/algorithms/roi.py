from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO


TARGET_CLASS = 0
CONF_THRES = 0.25
MIN_PADDING = 20
PAD_RATIO = 0.08


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def compute_padding(x1: int, y1: int, x2: int, y2: int) -> int:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    return max(MIN_PADDING, int(max(bw, bh) * PAD_RATIO))


def extract_best_roi(
    frame: np.ndarray,
    model: YOLO,
    target_class: int,
    conf_thres: float,
) -> Tuple[
    Optional[np.ndarray],                 # roi
    Optional[Tuple[int, int, int, int]], # det_bbox
    Optional[Tuple[int, int, int, int]], # crop_bbox
    Optional[float],                     # score
    Optional[int],                       # pad
]:
    """
    取单类别任务中置信度最高的检测框，并裁出 ROI。

    返回:
        roi        : 裁剪后的 ROI
        det_bbox   : 原始检测框 (x1, y1, x2, y2)
        crop_bbox  : 扩展后的裁切框 (x1, y1, x2, y2)
        best_score : 置信度
        pad        : 本次外扩像素
    """
    h, w = frame.shape[:2]

    try:
        results = model(frame, conf=conf_thres, verbose=False)
    except Exception:
        return None, None, None, None, None

    if not results:
        return None, None, None, None, None

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return None, None, None, None, None

    best_box = None
    best_score = -1.0

    for box in boxes:
        cls_id = int(box.cls.item())
        score = float(box.conf.item())

        if cls_id != target_class:
            continue

        if score > best_score:
            best_score = score
            best_box = box

    if best_box is None:
        return None, None, None, None, None

    det_x1, det_y1, det_x2, det_y2 = map(int, best_box.xyxy[0].tolist())
    det_bbox = (det_x1, det_y1, det_x2, det_y2)

    pad = compute_padding(det_x1, det_y1, det_x2, det_y2)

    crop_x1 = clamp(det_x1 - pad, 0, w - 1)
    crop_y1 = clamp(det_y1 - pad, 0, h - 1)
    crop_x2 = clamp(det_x2 + pad, 0, w - 1)
    crop_y2 = clamp(det_y2 + pad, 0, h - 1)

    crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, None, None, None, None

    roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    return roi, det_bbox, crop_bbox, best_score, pad
