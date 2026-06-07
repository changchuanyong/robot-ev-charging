from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT_DIR / "dataset" / "live"

ROI_META_PATH = LIVE_DIR / "latest_roi_meta.json"
ROI_PATH = LIVE_DIR / "latest_roi_enhanced.jpg"
EDGES_PATH = LIVE_DIR / "latest_edges.jpg"
CENTERS_PATH = LIVE_DIR / "latest_fitted_centers.json"
POSE_PATH = LIVE_DIR / "latest_pose.json"
QUALITY_PATH = LIVE_DIR / "latest_pipeline_quality.json"

EXPECTED_LAYOUT_NAMES = {
    "main_left",
    "main_right",
    "top_left",
    "top_mid",
    "top_right",
    "center",
    "bottom_left",
    "bottom_mid",
    "bottom_right",
}

MIN_EXPECTED_CANDIDATES = 7
GOOD_REPROJ_ERROR_PX = 8.0
WARN_REPROJ_ERROR_PX = 15.0
GOOD_EDGE_DENSITY_RANGE = (0.012, 0.120)
WARN_EDGE_DENSITY_RANGE = (0.006, 0.180)
WARN_LAYOUT_RATIO = 0.90
FAIL_LAYOUT_RATIO = 1.25
WARN_FIT_SHIFT_PX = 3.0
FAIL_FIT_SHIFT_PX = 8.0


def read_json(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, data: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if path.exists():
            path.unlink()
        tmp.replace(path)
        return True
    except OSError:
        return False


def status_level(status: str) -> int:
    return {"ok": 0, "warn": 1, "fail": 2}.get(status, 2)


def worst_status(*statuses: str) -> str:
    return max(statuses, key=status_level) if statuses else "fail"


def check_edge_density(edges: np.ndarray | None) -> dict[str, Any]:
    if edges is None or edges.size == 0:
        return {"status": "fail", "reason": "edges_missing", "edge_density": None}

    if edges.ndim == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    density = float(np.count_nonzero(edges > 0) / edges.size)
    good_lo, good_hi = GOOD_EDGE_DENSITY_RANGE
    warn_lo, warn_hi = WARN_EDGE_DENSITY_RANGE

    if good_lo <= density <= good_hi:
        status = "ok"
        reason = "edge_density_good"
    elif warn_lo <= density <= warn_hi:
        status = "warn"
        reason = "edge_density_borderline"
    else:
        status = "fail"
        reason = "edge_density_out_of_range"

    return {
        "status": status,
        "reason": reason,
        "edge_density": round(density, 6),
        "good_range": list(GOOD_EDGE_DENSITY_RANGE),
        "warn_range": list(WARN_EDGE_DENSITY_RANGE),
    }


def check_candidates(centers: list[dict[str, Any]]) -> dict[str, Any]:
    fitted = [c for c in centers if c.get("fitted_center") is not None]
    layout_names = {str(c.get("layout_name", "")) for c in centers}
    missing = sorted(EXPECTED_LAYOUT_NAMES - layout_names)

    anchors = [c for c in centers if c.get("is_main_anchor")]
    fit_shifts = []
    layout_ratios = []

    for item in fitted:
        raw = item.get("raw_center")
        fit = item.get("fitted_center")
        if isinstance(raw, list) and isinstance(fit, list) and len(raw) == 2 and len(fit) == 2:
            fit_shifts.append(float(np.hypot(float(raw[0]) - float(fit[0]), float(raw[1]) - float(fit[1]))))

        accept = float(item.get("layout_accept_dist", 0.0) or 0.0)
        dist = float(item.get("layout_dist", 0.0) or 0.0)
        if accept > 0:
            layout_ratios.append(dist / accept)

    max_fit_shift = max(fit_shifts) if fit_shifts else None
    max_layout_ratio = max(layout_ratios) if layout_ratios else None

    status = "ok"
    reasons = []

    if len(fitted) < MIN_EXPECTED_CANDIDATES or len(anchors) < 2:
        status = "fail"
        reasons.append("too_few_fitted_candidates_or_anchors")
    elif missing:
        status = "warn"
        reasons.append("layout_names_missing")

    if max_layout_ratio is not None:
        if max_layout_ratio > FAIL_LAYOUT_RATIO:
            status = worst_status(status, "fail")
            reasons.append("layout_ratio_fail")
        elif max_layout_ratio > WARN_LAYOUT_RATIO:
            status = worst_status(status, "warn")
            reasons.append("layout_ratio_warn")

    if max_fit_shift is not None:
        if max_fit_shift > FAIL_FIT_SHIFT_PX:
            status = worst_status(status, "fail")
            reasons.append("fit_shift_fail")
        elif max_fit_shift > WARN_FIT_SHIFT_PX:
            status = worst_status(status, "warn")
            reasons.append("fit_shift_warn")

    return {
        "status": status,
        "reasons": reasons or ["candidate_quality_good"],
        "candidate_count": len(centers),
        "fitted_count": len(fitted),
        "anchor_count": len(anchors),
        "missing_layout_names": missing,
        "max_layout_ratio": round(max_layout_ratio, 4) if max_layout_ratio is not None else None,
        "median_fit_shift_px": round(median(fit_shifts), 4) if fit_shifts else None,
        "max_fit_shift_px": round(max_fit_shift, 4) if max_fit_shift is not None else None,
    }


def check_pose(pose: dict[str, Any]) -> dict[str, Any]:
    if not pose:
        return {"status": "fail", "reason": "pose_missing"}

    reproj = pose.get("reproj_error_px", pose.get("selection_error_px"))
    inlier_count = int(pose.get("inlier_count", 0) or 0)
    placeholder_intrinsics = bool(pose.get("camera_intrinsics_placeholder", False))
    all_in_front = bool(pose.get("all_in_front", False))

    status = "ok"
    reasons = []

    if reproj is None:
        status = "fail"
        reasons.append("reproj_error_missing")
    else:
        reproj = float(reproj)
        if reproj > WARN_REPROJ_ERROR_PX:
            status = "fail"
            reasons.append("reproj_error_high")
        elif reproj > GOOD_REPROJ_ERROR_PX:
            status = "warn"
            reasons.append("reproj_error_warn")

    if inlier_count < MIN_EXPECTED_CANDIDATES:
        status = worst_status(status, "fail")
        reasons.append("too_few_pnp_inliers")

    if not all_in_front:
        status = worst_status(status, "fail")
        reasons.append("points_not_all_in_front")

    if placeholder_intrinsics:
        status = worst_status(status, "warn")
        reasons.append("camera_intrinsics_placeholder")

    return {
        "status": status,
        "reasons": reasons or ["pose_quality_good"],
        "method": pose.get("method"),
        "reproj_error_px": round(float(reproj), 4) if reproj is not None else None,
        "inlier_count": inlier_count,
        "all_in_front": all_in_front,
        "camera_intrinsics_placeholder": placeholder_intrinsics,
    }


def check_roi(meta: dict[str, Any], roi: np.ndarray | None) -> dict[str, Any]:
    score = meta.get("score")
    roi_w = int(meta.get("roi_width", 0) or 0)
    roi_h = int(meta.get("roi_height", 0) or 0)
    image_w = int(meta.get("image_width", 0) or 0)
    image_h = int(meta.get("image_height", 0) or 0)

    status = "ok"
    reasons = []

    if roi is None or roi.size == 0:
        return {"status": "fail", "reasons": ["roi_missing"]}

    if score is None or float(score) < 0.50:
        status = "warn"
        reasons.append("det_score_low")

    if roi_w <= 0 or roi_h <= 0:
        status = worst_status(status, "fail")
        reasons.append("roi_size_invalid")
    elif image_w > 0 and image_h > 0:
        area_ratio = (roi_w * roi_h) / float(image_w * image_h)
        if area_ratio < 0.05 or area_ratio > 0.65:
            status = worst_status(status, "warn")
            reasons.append("roi_area_ratio_unusual")
    else:
        area_ratio = None

    return {
        "status": status,
        "reasons": reasons or ["roi_quality_good"],
        "det_score": round(float(score), 4) if score is not None else None,
        "roi_width": roi_w,
        "roi_height": roi_h,
        "roi_area_ratio": round(area_ratio, 6) if "area_ratio" in locals() and area_ratio is not None else None,
        "crop_stabilization": meta.get("crop_stabilization"),
        "crop_iou_with_previous": meta.get("crop_iou_with_previous"),
    }


def main() -> int:
    meta = read_json(ROI_META_PATH, {})
    centers = read_json(CENTERS_PATH, [])
    pose = read_json(POSE_PATH, {})
    roi = cv2.imread(str(ROI_PATH))
    edges = cv2.imread(str(EDGES_PATH), cv2.IMREAD_GRAYSCALE)

    roi_check = check_roi(meta, roi)
    edge_check = check_edge_density(edges)
    candidate_check = check_candidates(centers if isinstance(centers, list) else [])
    pose_check = check_pose(pose if isinstance(pose, dict) else {})

    overall = worst_status(
        roi_check["status"],
        edge_check["status"],
        candidate_check["status"],
        pose_check["status"],
    )

    report = {
        "overall_status": overall,
        "roi": roi_check,
        "edges": edge_check,
        "candidates": candidate_check,
        "pose": pose_check,
    }

    ok = write_json(QUALITY_PATH, report)
    if not ok:
        print(f"[FAIL] could not write quality report: {QUALITY_PATH}")
        return 1

    print("===== Pipeline Quality =====")
    print(f"overall      : {overall}")
    print(f"roi          : {roi_check['status']} {roi_check['reasons']}")
    print(f"edges        : {edge_check['status']} density={edge_check.get('edge_density')}")
    print(
        "candidates   : "
        f"{candidate_check['status']} count={candidate_check['candidate_count']} "
        f"max_layout_ratio={candidate_check['max_layout_ratio']} "
        f"max_fit_shift={candidate_check['max_fit_shift_px']}"
    )
    print(
        "pose         : "
        f"{pose_check['status']} reproj={pose_check.get('reproj_error_px')} "
        f"inliers={pose_check.get('inlier_count')}"
    )
    print(f"report       : {QUALITY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
