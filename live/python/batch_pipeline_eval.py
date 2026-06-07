from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
PY_DIR = Path(__file__).resolve().parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import Canny  # noqa: E402
import enhance  # noqa: E402
import latest_candidate_contours as contours_step  # noqa: E402
import layout_prior  # noqa: E402
import pnp2  # noqa: E402
import roi as roi_step  # noqa: E402
from center_fit import export_fitted_centers, fit_ellipse_on_kept_candidates  # noqa: E402


DEFAULT_MODEL_PATH = ROOT_DIR / "runs" / "detect_retrain" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "batch_eval"
DEFAULT_IMAGE_DIRS = [
    ROOT_DIR / "yolo_port" / "images" / "val",
    ROOT_DIR / "yolo_port" / "images" / "train",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_LAYOUTS = {
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
SMALL_HOLE_LAYOUTS = {
    "top_mid",
    "center",
    "bottom_left",
    "bottom_right",
}

EDGE_TOO_SPARSE = 0.006
EDGE_TOO_NOISY = 0.180
PNP_WARN_REPROJ_PX = 12.0
PNP_FAIL_REPROJ_PX = 20.0
PNP_LAYOUT_MAX_HYPOTHESES = 8


def list_images(image_dirs: list[Path], max_images: int) -> list[Path]:
    images: list[Path] = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
                if len(images) >= max_images:
                    return images
    return images


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if path.exists():
        path.unlink()
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "status",
        "failure_categories",
        "roi_score",
        "roi_area_ratio",
        "edge_density",
        "candidate_count",
        "missing_layouts",
        "layout_variant",
        "layout_variant_count",
        "pnp_reproj_error_px",
        "pnp_inlier_count",
        "pnp_method",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def layout_name_to_model_name(layout_name: str) -> str:
    return pnp2.LAYOUT_TO_MODEL_LABEL.get(layout_name, layout_name)


def build_image_points_from_centers(
    centers: list[dict[str, Any]],
    offset_x: float,
    offset_y: float,
) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for item in centers:
        layout_name = str(item.get("layout_name", ""))
        if layout_name == "":
            continue

        center = item.get("fitted_center") or item.get("raw_center")
        if not isinstance(center, list) or len(center) < 2:
            continue

        model_name = layout_name_to_model_name(layout_name)
        points[model_name] = (
            float(center[0]) + float(offset_x),
            float(center[1]) + float(offset_y),
        )
    return points


def reset_candidate_layout(candidate):
    candidate.layout_name = ""
    candidate.layout_group = ""
    candidate.layout_u = 0.0
    candidate.layout_v = 0.0
    candidate.layout_dist = 999.0
    candidate.layout_score = 0.0
    candidate.layout_accept_dist = 999.0
    candidate.is_main_anchor = False
    candidate.is_top_backup = False
    candidate.fitted_center = None
    candidate.fit_method = "none"
    candidate.ellipse_axes = None
    candidate.ellipse_angle = None


def clone_candidates(candidates):
    clones = [dataclasses.replace(c) for c in candidates]
    for clone in clones:
        reset_candidate_layout(clone)
    return clones


def candidate_index(candidates, target) -> int | None:
    for idx, candidate in enumerate(candidates):
        if candidate is target:
            return idx
    return None


def build_layout_variant_from_assignment(
    raw_candidates,
    left_anchor,
    right_anchor,
    assignment,
    model,
    variant_name: str,
):
    clones = clone_candidates(raw_candidates)
    left_idx = candidate_index(raw_candidates, left_anchor)
    right_idx = candidate_index(raw_candidates, right_anchor)
    if left_idx is None or right_idx is None:
        return None

    left_clone = clones[left_idx]
    right_clone = clones[right_idx]
    left_clone.is_main_anchor = True
    right_clone.is_main_anchor = True
    left_clone.layout_name = "main_left"
    right_clone.layout_name = "main_right"
    left_clone.layout_group = "anchor"
    right_clone.layout_group = "anchor"
    left_clone.layout_u, left_clone.layout_v = -0.5, 0.0
    right_clone.layout_u, right_clone.layout_v = 0.5, 0.0
    left_clone.layout_dist = 0.0
    right_clone.layout_dist = 0.0
    left_clone.layout_score = left_clone.score
    right_clone.layout_score = right_clone.score
    left_clone.layout_accept_dist = 0.0
    right_clone.layout_accept_dist = 0.0

    kept = [left_clone, right_clone]
    used_indices = {left_idx, right_idx}

    for zone_name in [z[0] for z in layout_prior.STANDARD_LAYOUT_ZONES]:
        if zone_name not in assignment:
            continue
        cand, u, v, dist, accept_dist, zone_group, score = assignment[zone_name]
        idx = candidate_index(raw_candidates, cand)
        if idx is None or idx in used_indices:
            continue
        clone = clones[idx]
        layout_prior.assign_candidate_to_zone(clone, zone_name, zone_group, u, v, dist, accept_dist)
        clone.layout_score = score
        kept.append(clone)
        used_indices.add(idx)

    def sort_key(c):
        if c.is_main_anchor:
            return (-10.0, c.center[0])
        return (c.layout_v, c.layout_u)

    kept.sort(key=sort_key)
    return {
        "name": variant_name,
        "kept": kept,
        "model": model,
    }


def generate_layout_variants(raw_candidates, roi_shape):
    variants = []
    pair_hypotheses = layout_prior.find_main_big_hole_pair_hypotheses(raw_candidates, roi_shape)
    if not pair_hypotheses:
        return variants

    best_pair_score = pair_hypotheses[0][0]
    for rank, (pair_score, i, j) in enumerate(pair_hypotheses[:PNP_LAYOUT_MAX_HYPOTHESES], start=1):
        if pair_score < best_pair_score * layout_prior.MAIN_PAIR_RELATIVE_SCORE_MIN:
            continue

        c1 = raw_candidates[i]
        c2 = raw_candidates[j]
        if c1.center[0] <= c2.center[0]:
            left_anchor, right_anchor = c1, c2
        else:
            left_anchor, right_anchor = c2, c1

        model = layout_prior.build_layout_model(left_anchor, right_anchor)
        if model is None:
            continue

        global_assignment, _global_score, _global_coverage = layout_prior.build_global_layout_assignment(
            raw_candidates,
            left_anchor,
            right_anchor,
            model,
        )
        global_variant = build_layout_variant_from_assignment(
            raw_candidates,
            left_anchor,
            right_anchor,
            global_assignment,
            model,
            f"global_pair_{rank}",
        )
        if global_variant is not None:
            variants.append(global_variant)

        nearest_assignment, _nearest_score, _nearest_coverage = layout_prior.build_nearest_layout_assignment(
            raw_candidates,
            left_anchor,
            right_anchor,
            model,
        )
        nearest_variant = build_layout_variant_from_assignment(
            raw_candidates,
            left_anchor,
            right_anchor,
            nearest_assignment,
            model,
            f"nearest_pair_{rank}",
        )
        if nearest_variant is not None:
            variants.append(nearest_variant)

    return variants


def score_pose_variant(
    variant,
    crop_x1,
    crop_y1,
    object_points,
    camera_matrix,
    dist_coeffs,
    verbose: bool,
):
    if verbose:
        fitted_candidates = fit_ellipse_on_kept_candidates(variant["kept"])
    else:
        with open(Path("nul"), "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                fitted_candidates = fit_ellipse_on_kept_candidates(variant["kept"])

    centers = export_fitted_centers(fitted_candidates)
    layout_names = {str(item.get("layout_name", "")) for item in centers}
    missing_layouts = sorted(EXPECTED_LAYOUTS - layout_names)
    image_points = build_image_points_from_centers(centers, offset_x=crop_x1, offset_y=crop_y1)

    pose = None
    pnp_error = None
    try:
        pose = pnp2.solve_target_pose(
            image_points_dict=image_points,
            object_points_dict=object_points,
            K=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
    except Exception as exc:
        pnp_error = str(exc)

    reproj = float(pose["reproj_error"]) if pose is not None else 9999.0
    inliers = int(pose["inlier_count"]) if pose is not None else 0
    count = len(centers)
    missing_count = len(missing_layouts)

    selection_score = (
        80.0 * count
        - 120.0 * missing_count
        + 12.0 * inliers
        - 8.0 * reproj
    )
    if pose is None:
        selection_score -= 500.0

    return {
        "variant_name": variant["name"],
        "score": selection_score,
        "candidates": fitted_candidates,
        "centers": centers,
        "layout_names": layout_names,
        "missing_layouts": missing_layouts,
        "pose": pose,
        "pnp_error": pnp_error,
        "candidate_count": count,
        "reproj": reproj,
        "inliers": inliers,
    }


def classify_result(result: dict[str, Any]) -> list[str]:
    categories: list[str] = []

    if not result.get("roi_ok", False):
        categories.append("roi_missing")
        return categories

    roi_score = result.get("roi_score")
    roi_area_ratio = result.get("roi_area_ratio")
    if roi_score is not None and roi_score < 0.50:
        categories.append("roi_low_score")
    if roi_area_ratio is not None and (roi_area_ratio < 0.05 or roi_area_ratio > 0.65):
        categories.append("roi_suspicious_size")

    edge_density = result.get("edge_density")
    if edge_density is None:
        categories.append("canny_missing")
    elif edge_density < EDGE_TOO_SPARSE:
        categories.append("canny_too_sparse")
    elif edge_density > EDGE_TOO_NOISY:
        categories.append("canny_too_noisy")

    missing_layouts = set(result.get("missing_layouts", []))
    candidate_count = int(result.get("candidate_count", 0) or 0)
    if candidate_count < 9 or missing_layouts:
        categories.append("layout_incomplete")

    if missing_layouts & SMALL_HOLE_LAYOUTS:
        categories.append("small_holes_missing")

    if {"main_left", "main_right"} & missing_layouts:
        categories.append("big_hole_or_outer_ring_unstable")

    if result.get("pnp_ok") is False:
        categories.append("pnp_failed")
    else:
        reproj = result.get("pnp_reproj_error_px")
        if reproj is not None and reproj > PNP_FAIL_REPROJ_PX:
            categories.append("pnp_point_matching_unstable")
        elif reproj is not None and reproj > PNP_WARN_REPROJ_PX:
            categories.append("pnp_reproj_warn")

    return categories


def evaluate_one_image(
    image_path: Path,
    model: YOLO,
    object_points: dict[str, tuple[float, float, float]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    verbose: bool,
) -> dict[str, Any]:
    frame = cv2.imread(str(image_path))
    if frame is None or frame.size == 0:
        return {
            "image": str(image_path),
            "status": "fail",
            "roi_ok": False,
            "failure_categories": "image_read_failed",
        }

    result: dict[str, Any] = {
        "image": str(image_path.relative_to(ROOT_DIR)) if image_path.is_relative_to(ROOT_DIR) else str(image_path),
        "status": "ok",
        "roi_ok": False,
        "pnp_ok": False,
    }

    roi_img, det_bbox, crop_bbox, score, _pad = roi_step.extract_best_roi(
        frame=frame,
        model=model,
        target_class=roi_step.TARGET_CLASS,
        conf_thres=roi_step.CONF_THRES,
    )

    if roi_img is None or det_bbox is None or crop_bbox is None or score is None:
        result["status"] = "fail"
        result["failure_categories"] = "roi_missing"
        return result

    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    roi_area_ratio = ((crop_x2 - crop_x1) * (crop_y2 - crop_y1)) / float(frame.shape[0] * frame.shape[1])
    result.update(
        {
            "roi_ok": True,
            "roi_score": round(float(score), 6),
            "roi_area_ratio": round(float(roi_area_ratio), 6),
            "roi_bbox_xyxy": [int(v) for v in crop_bbox],
        }
    )

    enhanced_results = enhance.preprocess_roi(roi_img)
    enhanced_gray = enhanced_results["final"]
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    _gray, _enhanced, _smooth, _grad_mag, edges, low, high, morph_thresh = Canny.adaptive_canny_pipeline(enhanced_bgr)
    edge_density = float(np.count_nonzero(edges > 0) / edges.size)
    result.update(
        {
            "edge_density": round(edge_density, 6),
            "canny_low": int(low),
            "canny_high": int(high),
            "morph_thresh": int(morph_thresh),
        }
    )

    old_reject_log = contours_step.PRINT_REJECT_LOG
    contours_step.PRINT_REJECT_LOG = verbose
    try:
        if verbose:
            raw_candidates, _all_contours, _layout_model = contours_step.filter_candidate_contours(
                edges,
                enhanced_bgr,
                apply_layout=False,
            )
        else:
            with open(Path("nul"), "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull):
                    raw_candidates, _all_contours, _layout_model = contours_step.filter_candidate_contours(
                        edges,
                        enhanced_bgr,
                        apply_layout=False,
                    )
    finally:
        contours_step.PRINT_REJECT_LOG = old_reject_log

    variants = generate_layout_variants(raw_candidates, enhanced_bgr.shape)
    scored_variants = [
        score_pose_variant(
            variant,
            crop_x1=crop_x1,
            crop_y1=crop_y1,
            object_points=object_points,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            verbose=verbose,
        )
        for variant in variants
    ]
    scored_variants.sort(key=lambda item: item["score"], reverse=True)

    if scored_variants:
        best_variant = scored_variants[0]
        fitted_candidates = best_variant["candidates"]
        centers = best_variant["centers"]
        result["layout_variant"] = best_variant["variant_name"]
        result["layout_variant_count"] = len(scored_variants)
    else:
        fitted_candidates = []
        centers = []
        result["layout_variant"] = ""
        result["layout_variant_count"] = 0

    layout_names = {str(item.get("layout_name", "")) for item in centers}
    missing_layouts = sorted(EXPECTED_LAYOUTS - layout_names)
    result.update(
        {
            "candidate_count": len(centers),
            "layout_names": sorted(layout_names),
            "missing_layouts": missing_layouts,
        }
    )

    if scored_variants and scored_variants[0]["pose"] is not None:
        pose = scored_variants[0]["pose"]
        result.update(
            {
                "pnp_ok": True,
                "pnp_method": pose["method"],
                "pnp_reproj_error_px": round(float(pose["reproj_error"]), 6),
                "pnp_selection_error_px": round(float(pose["selection_error"]), 6),
                "pnp_inlier_count": int(pose["inlier_count"]),
                "pnp_outlier_names": pose["outlier_names"],
                "pnp_per_point_errors_px": {
                    name: round(float(err), 6)
                    for name, err in pose["per_point_errors_px"].items()
                },
            }
        )
    else:
        result["pnp_error"] = scored_variants[0]["pnp_error"] if scored_variants else "no_layout_variant"

    categories = classify_result(result)
    result["failure_categories"] = ";".join(categories)
    if categories:
        result["status"] = "warn" if result.get("pnp_ok") and "pnp_failed" not in categories else "fail"

    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    for row in rows:
        for category in str(row.get("failure_categories", "")).split(";"):
            if not category:
                continue
            category_counts[category] = category_counts.get(category, 0) + 1

    reproj_values = [
        float(row["pnp_reproj_error_px"])
        for row in rows
        if row.get("pnp_reproj_error_px") not in (None, "")
    ]

    return {
        "total": len(rows),
        "ok": sum(1 for row in rows if row.get("status") == "ok"),
        "warn": sum(1 for row in rows if row.get("status") == "warn"),
        "fail": sum(1 for row in rows if row.get("status") == "fail"),
        "category_counts": dict(sorted(category_counts.items())),
        "mean_reproj_error_px": round(float(np.mean(reproj_values)), 6) if reproj_values else None,
        "median_reproj_error_px": round(float(np.median(reproj_values)), 6) if reproj_values else None,
        "max_reproj_error_px": round(float(np.max(reproj_values)), 6) if reproj_values else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate the ROI -> contour -> PnP pipeline.")
    parser.add_argument(
        "--image-dir",
        action="append",
        type=Path,
        help="Image directory to evaluate. Can be passed multiple times. Defaults to yolo_port images.",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true", help="Print contour/layout debug logs for every image.")
    parser.add_argument("--strict-exit", action="store_true", help="Return non-zero when any image fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dirs = args.image_dir if args.image_dir else DEFAULT_IMAGE_DIRS
    images = list_images(image_dirs, max_images=args.max_images)
    if not images:
        print("No images found for batch evaluation.")
        return 1

    if not args.model.exists():
        print(f"YOLO model not found: {args.model}")
        return 1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    object_points = pnp2.load_object_points_from_csv(pnp2.MODEL_CSV_PATH)
    camera_matrix, dist_coeffs, camera_is_placeholder = pnp2.load_camera_intrinsics(pnp2.CAMERA_CONFIG_PATH)
    if camera_is_placeholder:
        print("[WARN] camera_intrinsics.json is still approximate/placeholder.")

    rows: list[dict[str, Any]] = []
    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx:03d}/{len(images):03d}] {image_path}")
        row = evaluate_one_image(
            image_path=image_path,
            model=model,
            object_points=object_points,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            verbose=args.verbose,
        )
        rows.append(row)
        print(
            "    "
            f"status={row.get('status')} "
            f"candidates={row.get('candidate_count', 0)} "
            f"missing={','.join(row.get('missing_layouts', []))} "
            f"reproj={row.get('pnp_reproj_error_px', '')} "
            f"categories={row.get('failure_categories', '')}"
        )

    summary = summarize(rows)
    write_csv(run_dir / "summary.csv", rows)
    atomic_write_json(run_dir / "details.json", {"summary": summary, "rows": rows})
    atomic_write_json(args.output_dir / "latest_details.json", {"summary": summary, "rows": rows})

    failed_rows = [row for row in rows if row.get("status") != "ok"]
    write_csv(run_dir / "failures.csv", failed_rows)

    print("\n===== Batch Evaluation Summary =====")
    print(f"total : {summary['total']}")
    print(f"ok    : {summary['ok']}")
    print(f"warn  : {summary['warn']}")
    print(f"fail  : {summary['fail']}")
    print(f"mean reproj px   : {summary['mean_reproj_error_px']}")
    print(f"median reproj px : {summary['median_reproj_error_px']}")
    print("failure categories:")
    for name, count in summary["category_counts"].items():
        print(f"  {name}: {count}")
    print(f"\nSaved: {run_dir}")
    if args.strict_exit and summary["fail"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
