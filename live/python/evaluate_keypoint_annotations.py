"""Compare named observations and automatic points with a fixed PnP backend."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import cv2
import numpy as np

from keypoint_annotations import LABELS, normalized_image_key, sha256_file, validate_annotations, visible_points
import pnp2


ROOT_DIR = Path(__file__).resolve().parents[2]


def _repo_path(root: Path, value: str) -> Path:
    path = (root / normalized_image_key(value)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves repository: {value}")
    return path


def validate_inputs(annotations: dict, batch: dict, root: Path, batch_report_path: Path) -> dict:
    """Bind annotations to immutable image bytes, report, geometry and solver."""
    validate_annotations(annotations)
    provenance = annotations["provenance"]
    report_path = batch_report_path.resolve()
    if report_path != _repo_path(root, provenance["batch_report_path"]):
        raise ValueError("Batch report path differs from annotation provenance")
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != provenance["batch_report_sha256"]:
        raise ValueError("Batch report hash changed")
    if json.loads(report_bytes) != batch:
        raise ValueError("Loaded batch report differs from bound report bytes")
    batch_provenance = batch.get("provenance", {})
    for path_field, hash_field in (("camera_config_path", "camera_config_sha256"),
                                   ("model_points_path", "model_points_sha256")):
        actual_hash = sha256_file(_repo_path(root, provenance[path_field]))
        if actual_hash != provenance[hash_field] or actual_hash != batch_provenance.get(hash_field):
            raise ValueError(f"Input hash changed: {path_field}")
    source_hash = sha256_file(Path(pnp2.__file__))
    if (source_hash != provenance["pnp_source_sha256"]
            or source_hash != batch_provenance.get("source_sha256", {}).get("pnp2.py")):
        raise ValueError("PnP source hash changed")
    rows = {normalized_image_key(row["image"]): row for row in batch["rows"]}
    if len(rows) != len(batch["rows"]):
        raise ValueError("Batch report contains duplicate image identities")
    camera = json.loads(_repo_path(root, provenance["camera_config_path"]).read_text(encoding="utf-8"))
    camera_size = (camera.get("image_width"), camera.get("image_height"))
    for entry in annotations["images"]:
        key = normalized_image_key(entry["image"])
        if key not in rows:
            raise ValueError(f"Image missing from batch report: {key}")
        row = rows[key]
        image_path = _repo_path(root, key)
        if sha256_file(image_path) != entry["image_sha256"] or row.get("image_sha256") != entry["image_sha256"]:
            raise ValueError(f"Image hash changed: {key}")
        frame = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or (frame.shape[1], frame.shape[0]) != (entry["width"], entry["height"]):
            raise ValueError(f"Image dimensions changed: {key}")
        if camera_size != (entry["width"], entry["height"]):
            raise ValueError(f"Image size does not match fixed camera intrinsics: {key}")
        if entry["auto_points"] != row.get("image_points_full", {}):
            raise ValueError(f"Automatic points changed from report: {key}")
        if not row.get("pnp_ok", False) and entry["auto_points"]:
            raise ValueError(f"Unsolved image unexpectedly supplies selected automatic points: {key}")
        if entry.get("roi_bbox_xyxy") != row.get("roi_bbox_xyxy", []):
            raise ValueError(f"ROI changed from report: {key}")
        if entry["split"] != row.get("split", "unknown"):
            raise ValueError(f"Split changed from report: {key}")
    return rows


def residual_stats(values: list[float]) -> dict:
    return {"count": len(values), "mean_px": statistics.mean(values) if values else None,
            "median_px": statistics.median(values) if values else None,
            "max_px": max(values) if values else None}


def solve_points(points: dict, model: dict, matrix: np.ndarray, distortion: np.ndarray) -> dict:
    result = {"solved": False, "point_count": len(points), "labels": list(points)}
    if len(points) < 4:
        return {**result, "reason": "fewer_than_four_points"}
    try:
        cv2.setRNGSeed(0)
        pose = pnp2.solve_target_pose(points, model, matrix, distortion)
    except (ValueError, RuntimeError, cv2.error) as exc:
        return {**result, "reason": str(exc)}
    projected = cv2.projectPoints(pose["object_points"], pose["rvec"], pose["tvec"], matrix, distortion)[0]
    return {**result, "solved": True, "method": pose["method"],
            "mean_px": float(pose["reproj_error"]), "rms_px": float(pose["reproj_rms_error"]),
            "max_px": float(pose["max_reproj_error"]),
            "inlier_count": int(pose["inlier_count"]), "inlier_threshold_px": float(pose["inlier_threshold_px"]),
            "per_point_errors_px": pose["per_point_errors_px"],
            "projected_points": dict(zip(pose["matched_names"], projected.reshape(-1, 2).tolist())),
            "rvec": pose["rvec"].reshape(3).tolist(), "tvec_model_units": pose["tvec"].reshape(3).tolist()}


def evaluate_image(entry: dict, batch_row: dict, model: dict, matrix: np.ndarray, distortion: np.ndarray) -> dict:
    observed = visible_points(entry)
    automatic = entry["auto_points"]
    common = [name for name in LABELS if name in observed and name in automatic]
    distances = {name: float(np.linalg.norm(np.asarray(observed[name]) - automatic[name])) for name in common}
    return {"image": normalized_image_key(entry["image"]), "split": entry["split"],
            "review_status": entry["review_status"], "reviewer": entry["reviewer"],
            "visible_annotation_count": len(observed), "automatic_point_count": len(automatic),
            "visibility_counts": {state: sum(p["visibility"] == state for p in entry["points"].values())
                                  for state in ("visible", "uncertain", "occluded", "out_of_frame", "unlabeled")},
            "common_labels": common, "missing_automatic_labels": [name for name in observed if name not in automatic],
            "automatic_labels_without_visible_reference": [name for name in automatic if name not in observed],
            "center_deviation_px": distances,
            "original_pose_quality_ok": batch_row.get("pose_quality_ok"),
            "original_pose_rejection_reasons": batch_row.get("pose_rejection_reasons", []),
            "original_report_mean_px": batch_row.get("pnp_reproj_error_px"),
            "annotation_visible": solve_points(observed, model, matrix, distortion),
            "automatic_all": solve_points(automatic, model, matrix, distortion),
            "annotation_common": solve_points({name: observed[name] for name in common}, model, matrix, distortion),
            "automatic_common": solve_points({name: automatic[name] for name in common}, model, matrix, distortion)}


def summarize_group(rows: list[dict], *, reviewed: bool) -> dict:
    summary = {"image_count": len(rows), "human_reviewed": reviewed,
               "independent_test_set": False, "physical_pose_accuracy_verified": False,
               "annotation_visible_point_count": sum(row["visible_annotation_count"] for row in rows),
               "automatic_selected_point_count": sum(row["automatic_point_count"] for row in rows),
               "common_name_point_count": sum(len(row["common_labels"]) for row in rows),
               "missing_automatic_on_visible_count": sum(len(row["missing_automatic_labels"]) for row in rows),
               "center_deviation_px": residual_stats([v for row in rows for v in row["center_deviation_px"].values()]),
               "original_quality_pass_count": sum(row["original_pose_quality_ok"] is True for row in rows),
               "branches": {}}
    for branch in ("annotation_visible", "automatic_all", "annotation_common", "automatic_common"):
        solved = [row[branch] for row in rows if row[branch]["solved"]]
        summary["branches"][branch] = {
            "solved_images": len(solved), "failed_images": [row["image"] for row in rows if not row[branch]["solved"]],
            "image_mean_residual_px": residual_stats([pose["mean_px"] for pose in solved])}
    paired = [row for row in rows if row["annotation_common"]["solved"] and row["automatic_common"]["solved"]]
    summary["paired_same_images_and_point_names"] = {
        "image_count": len(paired), "images": [row["image"] for row in paired],
        "annotation_image_mean_residual_px": residual_stats([row["annotation_common"]["mean_px"] for row in paired]),
        "automatic_image_mean_residual_px": residual_stats([row["automatic_common"]["mean_px"] for row in paired]),
        "automatic_minus_annotation_mean_px": residual_stats(
            [row["automatic_common"]["mean_px"] - row["annotation_common"]["mean_px"] for row in paired])}
    return summary


def evaluate_annotations(annotations: dict, batch_rows: dict, model: dict, matrix: np.ndarray,
                         distortion: np.ndarray, *, include_drafts: bool = False) -> dict:
    """Run only after validate_inputs; drafts never enter the reviewed aggregate."""
    validate_annotations(annotations)
    if list(model) != list(LABELS):
        raise ValueError("Model label order differs from annotation schema")
    groups = {"human_reviewed": []}
    if include_drafts:
        groups["draft_diagnostic"] = []
    pending = []
    for entry in annotations["images"]:
        key = normalized_image_key(entry["image"])
        if entry["review_status"] != "reviewed":
            pending.append(key)
            if not include_drafts:
                continue
        group = "human_reviewed" if entry["review_status"] == "reviewed" else "draft_diagnostic"
        groups[group].append(evaluate_image(entry, batch_rows[key], model, matrix, distortion))
    return {"schema_version": 1, "independent_test_set": False,
            "physical_pose_accuracy_verified": False, "selected_images": len(annotations["images"]),
            "pending_human_review": pending, "drafts_included_for_diagnosis": include_drafts,
            "feature_definition": annotations["feature_definition"],
            "notes": ["Draft visual annotations are not human ground truth; their statistics remain separate.",
                      "All residuals include every observed point in that branch, including outliers.",
                      "Compare paired_same_images_and_point_names to avoid changing the evaluated point set.",
                      "Center deviation includes localization and identity errors; it is not a proven identity error rate.",
                      "Low reprojection residual does not establish correspondence correctness or physical pose accuracy."],
            "groups": {name: summarize_group(rows, reviewed=name == "human_reviewed") for name, rows in groups.items()},
            "rows": {name: rows for name, rows in groups.items()}}


def validate_output_path(output_path: Path, annotations_path: Path, batch_report_path: Path,
                         annotations: dict, root: Path) -> None:
    """A diagnostic report must never replace a bound input or its evaluator."""
    provenance = annotations["provenance"]
    protected = [annotations_path, batch_report_path, Path(pnp2.__file__), Path(__file__),
                 Path(__file__).with_name("keypoint_annotations.py"),
                 _repo_path(root, provenance["camera_config_path"]),
                 _repo_path(root, provenance["model_points_path"])]
    protected.extend(_repo_path(root, entry["image"]) for entry in annotations["images"])
    output = output_path.resolve()
    for source in protected:
        if output == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError(f"Output must not overwrite an input or source: {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    parser.add_argument("--include-drafts", action="store_true", help="Separate unreviewed diagnosis; never human truth")
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    batch = json.loads(args.batch_report.read_text(encoding="utf-8"))
    rows = validate_inputs(annotations, batch, args.root, args.batch_report)
    validate_output_path(args.output, args.annotations, args.batch_report, annotations, args.root)
    provenance = annotations["provenance"]
    model = pnp2.load_object_points_from_csv(_repo_path(args.root, provenance["model_points_path"]))
    matrix, distortion, placeholder = pnp2.load_camera_intrinsics(_repo_path(args.root, provenance["camera_config_path"]))
    result = evaluate_annotations(annotations, rows, model, matrix, distortion, include_drafts=args.include_drafts)
    result["provenance"] = {**provenance, "annotations_sha256": sha256_file(args.annotations),
                            "inputs_verified": True, "camera_is_placeholder": placeholder,
                            "evaluator_sha256": sha256_file(Path(__file__)),
                            "schema_validator_sha256": sha256_file(Path(__file__).with_name("keypoint_annotations.py"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected_images": result["selected_images"], "pending_human_review": len(result["pending_human_review"]),
                      "groups": result["groups"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
