from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vision.model_config import current_model_path

import enhance  # noqa: E402
import latest_candidate_contours as contours_step  # noqa: E402
import pnp2  # noqa: E402
from pose_correspondence import select_correspondences  # noqa: E402
import roi as roi_step  # noqa: E402
from observed_layout import DEFAULT_PRIOR_PATH, load_prior  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "batch_eval"
DEFAULT_IMAGE_DIRS = [
    ROOT_DIR / "dataset" / "vision" / "images" / "historical",
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
EDGE_TOO_NOISY = 0.350
PNP_WARN_REPROJ_PX = 16.0
PNP_FAIL_REPROJ_PX = 20.0


def list_images(image_dirs: list[Path], max_images: int) -> list[Path]:
    images: list[Path] = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
                if len(images) >= max_images:
                    return images
    return images


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists():
        path.unlink()
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "status",
        "legacy_status",
        "split",
        "pose_quality_ok",
        "pose_rejection_reasons",
        "failure_categories",
        "roi_score",
        "roi_area_ratio",
        "edge_density",
        "candidate_count",
        "missing_layouts",
        "layout_variant",
        "layout_variant_count",
        "pnp_reproj_error_px",
        "pnp_reproj_rms_px",
        "pnp_max_error_px",
        "pnp_inlier_count",
        "pnp_method",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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
    visualization_dir: Path | None = None,
    observed_prior: dict | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    # OpenCV's Windows filename API cannot read this workspace's Chinese path.
    try:
        image_bytes = image_path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR) if image_bytes else None
    except (OSError, cv2.error):
        frame = None
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
        "split": image_path.parent.name if image_path.parent.name in {"train", "val", "test"} else "unknown",
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
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

    _gray, _enhanced, _smooth, _grad_mag, edges, low, high, morph_thresh = enhance.adaptive_canny_pipeline(enhanced_bgr)
    edge_density = float(np.count_nonzero(edges > 0) / edges.size)
    result.update(
        {
            "edge_density": round(edge_density, 6),
            "canny_low": int(low),
            "canny_high": int(high),
            "morph_thresh": int(morph_thresh),
        }
    )

    contour_diagnostics: dict = {}
    old_reject_log = contours_step.PRINT_REJECT_LOG
    contours_step.PRINT_REJECT_LOG = verbose
    try:
        if verbose:
            raw_candidates, _all_contours, _layout_model = contours_step.filter_candidate_contours(
                edges,
                enhanced_bgr,
                apply_layout=False,
                diagnostics=contour_diagnostics,
                gray_source=roi_img,
            )
        else:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull):
                    raw_candidates, _all_contours, _layout_model = contours_step.filter_candidate_contours(
                        edges,
                        enhanced_bgr,
                        apply_layout=False,
                        diagnostics=contour_diagnostics,
                        gray_source=roi_img,
                    )
    finally:
        contours_step.PRINT_REJECT_LOG = old_reject_log

    selection = select_correspondences(raw_candidates, enhanced_bgr.shape, object_points,
                                       camera_matrix, dist_coeffs, (crop_x1, crop_y1),
                                       observed_prior=observed_prior)
    scored_variants = [selection["best"]] if selection["best"] is not None else []
    result["contour_diagnostics"] = contour_diagnostics
    result["correspondence_diagnostics"] = selection["diagnostics"]
    result["pose_hypotheses"] = selection["hypotheses"]
    result["raw_candidate_count"] = len(raw_candidates)
    result["raw_candidate_centers_full"] = [[float(c.center[0]+crop_x1), float(c.center[1]+crop_y1)]
                                             for c in raw_candidates]

    if scored_variants:
        best_variant = scored_variants[0]
        fitted_candidates = best_variant["candidates"]
        centers = best_variant["centers"]
        result["association_points_full"] = {name: list(point) for name, point in
                                             best_variant["image_points"].items()}
        result["layout_variant"] = best_variant["variant_name"]
        result["layout_variant_count"] = len(selection["hypotheses"])
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
                "pnp_reproj_rms_px": round(float(pose["reproj_rms_error"]), 6),
                "pnp_max_error_px": round(float(pose["max_reproj_error"]), 6),
                "pnp_fit_count": int(pose["fit_count"]),
                "pnp_all_in_front": bool(pose["all_in_front"]),
                "pnp_inlier_ratio": float(pose["inlier_ratio"]),
                "pnp_selection_error_px": round(float(pose["selection_error"]), 6),
                "pnp_inlier_count": int(pose["inlier_count"]),
                "pnp_outlier_names": pose["outlier_names"],
                "pnp_per_point_errors_px": {
                    name: round(float(err), 6)
                    for name, err in pose["per_point_errors_px"].items()
                },
            }
        )
        result["image_points_full"] = {name: list(point) for name, point in
                                        scored_variants[0]["image_points"].items()}
        projected, _ = cv2.projectPoints(pose["object_points"], pose["rvec"], pose["tvec"],
                                         camera_matrix, dist_coeffs)
        result["projected_points_full"] = dict(zip(pose["matched_names"], projected.reshape(-1, 2).tolist()))
    else:
        result["pnp_error"] = scored_variants[0]["pnp_error"] if scored_variants else "no_layout_variant"

    categories = classify_result(result)
    result["failure_categories"] = ";".join(categories)
    result["legacy_status"] = ("warn" if result.get("pnp_ok") else "fail") if categories else "ok"
    reasons = pose_rejection_reasons(result)
    result["pose_quality_ok"] = not reasons
    result["pose_rejection_reasons"] = reasons
    result["status"] = "fail" if reasons else result["legacy_status"]
    result["processing_ms"] = round((time.perf_counter() - started) * 1000, 3)
    if visualization_dir is not None:
        save_diagnostic_image(frame, result, visualization_dir / (image_path.stem + ".jpg"))

    return result


def pose_rejection_reasons(row: dict) -> list[str]:
    """Offline geometric gate only; camera calibration is a separate control requirement."""
    if not row.get("pnp_ok"):
        return ["pnp_failed"]
    reasons = []
    if not row.get("pnp_all_in_front", False):
        reasons.append("invalid_depth")
    if not np.isfinite(float(row.get("pnp_reproj_error_px", float("inf")))):
        reasons.append("nonfinite_residual")
    elif float(row["pnp_reproj_error_px"]) > PNP_FAIL_REPROJ_PX:
        reasons.append("all_point_mean_exceeds_20px")
    if int(row.get("pnp_inlier_count", 0)) < 4:
        reasons.append("fewer_than_four_10px_inliers")
    if row.get("correspondence_diagnostics", {}).get("ambiguous", False):
        reasons.append("ambiguous_correspondences")
    if row.get("correspondence_diagnostics", {}).get("selection_quality_ok") is False:
        reasons.append("correspondence_quality_rejected")
    return reasons


def save_diagnostic_image(frame: np.ndarray, row: dict, path: Path) -> None:
    canvas = frame.copy()
    for point in row.get("raw_candidate_centers_full", []):
        cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3, (150, 150, 150), 1)
    for name, point in row.get("image_points_full", {}).items():
        center = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, center, 5, (0, 220, 0), 2)
        cv2.putText(canvas, name, (center[0]+6, center[1]-6), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 1)
        projected = row.get("projected_points_full", {}).get(name)
        if projected is not None:
            end = tuple(np.rint(projected).astype(int))
            cv2.line(canvas, center, end, (0, 0, 255), 2)
            cv2.drawMarker(canvas, end, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
    label = f"{row['status']} | points={row.get('candidate_count', 0)} | mean={row.get('pnp_reproj_error_px')} px"
    cv2.putText(canvas, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, canvas)
    if not ok:
        raise OSError(f"Cannot save diagnostic image: {path}")
    path.write_bytes(encoded.tobytes())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    for row in rows:
        for reason in row.get("pose_rejection_reasons", []):
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
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
        "pose_rejection_counts": dict(sorted(rejection_counts.items())),
        "mean_reproj_error_px": round(float(np.mean(reproj_values)), 6) if reproj_values else None,
        "median_reproj_error_px": round(float(np.median(reproj_values)), 6) if reproj_values else None,
        "max_reproj_error_px": round(float(np.max(reproj_values)), 6) if reproj_values else None,
        "solver_success_count": sum(bool(row.get("pnp_ok")) for row in rows),
        "pose_quality_ok_count": sum(bool(row.get("pose_quality_ok")) for row in rows),
        "full_nine_point_count": sum(row.get("candidate_count") == 9 for row in rows),
        "mean_matched_point_count": round(float(np.mean([row.get("candidate_count", 0) for row in rows])), 6) if rows else None,
        "legacy_status_counts": {status: sum(row.get("legacy_status", row.get("status")) == status for row in rows)
                                 for status in ("ok", "warn", "fail")},
        "residual_population": "all returned poses, including rejected poses; all matched points",
        "median_processing_ms": round(float(np.median([r["processing_ms"] for r in rows if "processing_ms" in r])), 3)
        if any("processing_ms" in r for r in rows) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate the ROI -> contour -> PnP pipeline.")
    parser.add_argument(
        "--image-dir",
        action="append",
        type=Path,
        help="Image directory to evaluate recursively. Defaults to dataset/vision/images/historical for the fixed historical intrinsics.",
    )
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--model", type=Path, help="Explicit experiment weights; otherwise use config/vision_model.json")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--observed-layout-prior", type=Path, default=DEFAULT_PRIOR_PATH,
                        help="Learned 2D correspondence prior; physical PnP geometry stays in the CSV")
    layout.add_argument("--legacy-layout", action="store_true", help="Use the previous CAD-layout matcher for comparison")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true", help="Print contour/layout debug logs for every image.")
    parser.add_argument("--strict-exit", action="store_true", help="Return non-zero when any image fails.")
    parser.add_argument("--save-visualizations", action="store_true", help="Save observed/projected point overlays.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.legacy_layout:
        args.observed_layout_prior = None
    image_dirs = args.image_dir if args.image_dir else DEFAULT_IMAGE_DIRS
    images = list_images(image_dirs, max_images=args.max_images)
    if not images:
        print("No images found for batch evaluation.")
        return 1

    args.model = current_model_path(verify=True) if args.model is None else args.model.resolve()
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
    observed_prior = None
    if args.observed_layout_prior is not None:
        observed_prior = load_prior(args.observed_layout_prior)
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
            visualization_dir=run_dir / "visualizations" if args.save_visualizations else None,
            observed_prior=observed_prior,
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
    provenance = {
        "weights": str(args.model.resolve()),
        "weights_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "observed_layout_prior": str(args.observed_layout_prior.resolve()) if args.observed_layout_prior else None,
        "observed_layout_prior_sha256": hashlib.sha256(args.observed_layout_prior.read_bytes()).hexdigest()
        if args.observed_layout_prior else None,
        "camera_config_sha256": hashlib.sha256(pnp2.CAMERA_CONFIG_PATH.read_bytes()).hexdigest(),
        "model_points_sha256": hashlib.sha256(pnp2.MODEL_CSV_PATH.read_bytes()).hexdigest(),
        "camera_is_placeholder": camera_is_placeholder,
        "independent_test_set": False,
        "evaluation_note": "Existing train/val development images; validation was used for model selection and inspected during development.",
        "thresholds_px": {"legacy_warn_mean": PNP_WARN_REPROJ_PX, "quality_fail_mean": PNP_FAIL_REPROJ_PX,
                           "inlier": pnp2.RANSAC_REPROJ_ERROR_PX},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), PY_DIR / "pnp2.py", PY_DIR / "layout_prior.py",
                           PY_DIR / "pose_correspondence.py", PY_DIR / "latest_candidate_contours.py",
                           PY_DIR / "center_fit.py", PY_DIR / "enhance.py"]},
    }
    if observed_prior is not None:
        provenance["source_sha256"]["observed_layout.py"] = hashlib.sha256((PY_DIR / "observed_layout.py").read_bytes()).hexdigest()
    report = {"summary": summary, "by_split": {split: summarize([r for r in rows if r.get("split") == split])
              for split in sorted({r.get("split", "unknown") for r in rows})}, "provenance": provenance, "rows": rows}
    write_csv(run_dir / "summary.csv", rows)
    atomic_write_json(run_dir / "details.json", report)
    atomic_write_json(args.output_dir / "latest_details.json", report)

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
