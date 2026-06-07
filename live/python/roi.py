from __future__ import annotations

import os
import time
import json
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ======================
# 你只改这里
# ======================
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "runs" / "detect_retrain" / "weights" / "best.pt"
IMAGE_PATH = ROOT_DIR / "dataset" / "live" / "latest.jpg"

OUT_DIR = ROOT_DIR / "dataset" / "live"
ROI_LATEST_PATH = OUT_DIR / "latest_roi.jpg"
ROI_META_PATH = OUT_DIR / "latest_roi_meta.json"
VIS_LATEST_PATH = OUT_DIR / "latest_vis.jpg"

TARGET_CLASS = 0         # 你的数据集里 port -> 0
CONF_THRES = 0.25
MIN_PADDING = 20         # 最小外扩像素
PAD_RATIO = 0.08         # 按检测框尺寸自适应外扩
SLEEP_SHORT = 0.03
SHOW_WINDOW = (
    os.environ.get("VISION_SHOW_ROI_WINDOW")
    if os.environ.get("VISION_SHOW_ROI_WINDOW") is not None
    else os.environ.get("VISION_SHOW_BASE_WINDOWS", "1")
) == "1"
KEY_WINDOWS_ONLY = os.environ.get("VISION_KEY_WINDOWS", "0") == "1"

SAVE_RECENT = False      # 这里只做单独 ROI，默认关掉缓存
KEEP_RECENT = 100
RECENT_DIR = OUT_DIR / "recent_roi"

ENABLE_CROP_SMOOTHING = True
CROP_SMOOTH_IOU_MIN = 0.45
CROP_SMOOTH_ALPHA = 0.65
CROP_MAX_CENTER_SHIFT_RATIO = 0.35
CROP_MAX_SIZE_CHANGE_RATIO = 0.45
# ======================


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def atomic_imwrite(save_path: Path, image) -> bool:
    """原子写入，避免读到半张图。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_name(save_path.stem + "_tmp" + save_path.suffix)

    ok = cv2.imwrite(str(tmp_path), image)
    if not ok:
        return False

    try:
        if save_path.exists():
            save_path.unlink()
        tmp_path.replace(save_path)
        return True
    except OSError:
        return False


def atomic_write_json(save_path: Path, data: dict) -> bool:
    """原子写 JSON，避免读到半截文件。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_name(save_path.stem + "_tmp" + save_path.suffix)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if save_path.exists():
            save_path.unlink()
        tmp_path.replace(save_path)
        return True
    except OSError:
        return False


def compute_padding(x1: int, y1: int, x2: int, y2: int) -> int:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    return max(MIN_PADDING, int(max(bw, bh) * PAD_RATIO))


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_center_size(bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return x1 + w / 2.0, y1 + h / 2.0, w, h


def is_crop_jump(
    prev_bbox: tuple[int, int, int, int],
    curr_bbox: tuple[int, int, int, int],
) -> bool:
    prev_cx, prev_cy, prev_w, prev_h = bbox_center_size(prev_bbox)
    curr_cx, curr_cy, curr_w, curr_h = bbox_center_size(curr_bbox)

    ref_dim = max(prev_w, prev_h, 1.0)
    center_shift = np.hypot(curr_cx - prev_cx, curr_cy - prev_cy) / ref_dim
    size_change = max(
        abs(curr_w - prev_w) / prev_w,
        abs(curr_h - prev_h) / prev_h,
    )
    return (
        center_shift > CROP_MAX_CENTER_SHIFT_RATIO
        or size_change > CROP_MAX_SIZE_CHANGE_RATIO
    )


def smooth_crop_bbox(
    prev_bbox: tuple[int, int, int, int] | None,
    curr_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], str, float]:
    if not ENABLE_CROP_SMOOTHING or prev_bbox is None:
        return curr_bbox, "raw", 0.0

    iou = bbox_iou(prev_bbox, curr_bbox)
    if iou < CROP_SMOOTH_IOU_MIN or is_crop_jump(prev_bbox, curr_bbox):
        return curr_bbox, "reset", iou

    h, w = image_shape[:2]
    values = []
    for prev_v, curr_v in zip(prev_bbox, curr_bbox):
        value = CROP_SMOOTH_ALPHA * curr_v + (1.0 - CROP_SMOOTH_ALPHA) * prev_v
        values.append(int(round(value)))

    x1 = clamp(values[0], 0, w - 1)
    y1 = clamp(values[1], 0, h - 1)
    x2 = clamp(values[2], x1 + 1, w)
    y2 = clamp(values[3], y1 + 1, h)
    return (x1, y1, x2, y2), "smoothed", iou


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


def save_recent_roi(roi, recent_files: deque):
    """只保留最近 KEEP_RECENT 张 ROI。"""
    RECENT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() * 1000) % 1000)
    save_path = RECENT_DIR / f"roi_{timestamp}_{millis:03d}.jpg"

    ok = cv2.imwrite(str(save_path), roi)
    if not ok:
        return

    recent_files.append(save_path)

    while len(recent_files) > KEEP_RECENT:
        old_file = recent_files.popleft()
        try:
            if old_file.exists():
                old_file.unlink()
        except OSError:
            pass


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")
    print(f"Watching image   : {IMAGE_PATH}")
    print(f"Latest ROI       : {ROI_LATEST_PATH}")
    print(f"Latest ROI Meta  : {ROI_META_PATH}")
    print(f"Latest VIS       : {VIS_LATEST_PATH}")
    print("Press ESC to quit.")

    last_mtime = 0
    last_valid_roi = None
    last_crop_bbox = None
    recent_files = deque(sorted(RECENT_DIR.glob("*.jpg"))) if RECENT_DIR.exists() else deque()

    while len(recent_files) > KEEP_RECENT:
        old_file = recent_files.popleft()
        try:
            if old_file.exists():
                old_file.unlink()
        except OSError:
            pass

    while True:
        if not os.path.exists(IMAGE_PATH):
            time.sleep(0.05)
            continue

        try:
            file_size = os.path.getsize(IMAGE_PATH)
            mtime = os.path.getmtime(IMAGE_PATH)
        except (OSError, PermissionError):
            time.sleep(SLEEP_SHORT)
            continue

        if file_size <= 0:
            time.sleep(SLEEP_SHORT)
            continue

        if mtime == last_mtime:
            time.sleep(SLEEP_SHORT)
            continue

        try:
            frame = cv2.imread(str(IMAGE_PATH))
        except (cv2.error, OSError, PermissionError):
            time.sleep(SLEEP_SHORT)
            continue

        if frame is None or frame.size == 0:
            time.sleep(SLEEP_SHORT)
            continue

        last_mtime = mtime
        h, w = frame.shape[:2]

        roi, det_bbox, crop_bbox, score, pad = extract_best_roi(
            frame=frame,
            model=model,
            target_class=TARGET_CLASS,
            conf_thres=CONF_THRES,
        )

        vis = frame.copy()

        if roi is not None and det_bbox is not None and crop_bbox is not None and score is not None and pad is not None:
            det_x1, det_y1, det_x2, det_y2 = det_bbox
            raw_crop_bbox = crop_bbox
            crop_bbox, crop_stabilization, crop_iou = smooth_crop_bbox(
                prev_bbox=last_crop_bbox,
                curr_bbox=raw_crop_bbox,
                image_shape=frame.shape,
            )
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
            roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            last_crop_bbox = crop_bbox

            # 画原始检测框（绿色）
            cv2.rectangle(vis, (det_x1, det_y1), (det_x2, det_y2), (0, 255, 0), 2)

            # 画实际裁切框（黄色）
            cv2.rectangle(vis, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 2)

            cv2.putText(
                vis,
                f"port {score:.3f}",
                (det_x1, max(det_y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                vis,
                f"pad={pad} {crop_stabilization}",
                (crop_x1, min(crop_y2 + 25, h - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # 保存最新 ROI
            atomic_imwrite(ROI_LATEST_PATH, roi)
            last_valid_roi = roi.copy()

            # 保存坐标 JSON
            meta = {
                "source_image": str(IMAGE_PATH),
                "roi_image": str(ROI_LATEST_PATH),
                "image_width": int(w),
                "image_height": int(h),
                "target_class": int(TARGET_CLASS),
                "confidence_threshold": float(CONF_THRES),
                "score": float(score),
                "padding": int(pad),
                "crop_stabilization": crop_stabilization,
                "crop_iou_with_previous": float(crop_iou),

                # 原始检测框
                "det_bbox_xyxy": {
                    "x1": int(det_x1),
                    "y1": int(det_y1),
                    "x2": int(det_x2),
                    "y2": int(det_y2)
                },
                "raw_crop_bbox_xyxy": {
                    "x1": int(raw_crop_bbox[0]),
                    "y1": int(raw_crop_bbox[1]),
                    "x2": int(raw_crop_bbox[2]),
                    "y2": int(raw_crop_bbox[3])
                },

                # 实际裁切框（这个最重要）
                "crop_bbox_xyxy": {
                    "x1": int(crop_x1),
                    "y1": int(crop_y1),
                    "x2": int(crop_x2),
                    "y2": int(crop_y2)
                },

                # ROI 尺寸
                "roi_width": int(crop_x2 - crop_x1),
                "roi_height": int(crop_y2 - crop_y1),

                # 方便后续映射 ROI 内坐标 -> 原图坐标
                "roi_origin_in_full_image": {
                    "x": int(crop_x1),
                    "y": int(crop_y1)
                },

                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            atomic_write_json(ROI_META_PATH, meta)

            if SAVE_RECENT:
                save_recent_roi(roi, recent_files)

            status_text = "ROI updated"
            status_color = (0, 255, 0)

            print("===== ROI Updated =====")
            print(f"score     : {score:.4f}")
            print(f"det_bbox  : {det_bbox}")
            print(f"crop_bbox : {crop_bbox} ({crop_stabilization}, prev_iou={crop_iou:.3f})")
            print(f"roi_size  : {crop_x2 - crop_x1} x {crop_y2 - crop_y1}")
        else:
            status_text = "No target - keep last ROI"
            status_color = (0, 0, 255)

        cv2.putText(
            vis,
            status_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
        )

        cv2.putText(
            vis,
            f"Conf: {CONF_THRES:.2f}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        atomic_imwrite(VIS_LATEST_PATH, vis)

        if SHOW_WINDOW:
            cv2.imshow("YOLO ROI Watch", vis)

            if (not KEY_WINDOWS_ONLY) and last_valid_roi is not None:
                cv2.imshow("Latest ROI", last_valid_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        else:
            time.sleep(SLEEP_SHORT)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
