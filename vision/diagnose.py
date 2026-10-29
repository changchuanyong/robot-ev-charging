"""Schema-2 RGB diagnostics with frozen points and explicit camera/geometry variants."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from .contracts import distort_normalized, intrinsics_signature, rectify_points, validate_camera
from .pipeline import PosePipeline, ROOT, select_detection

if str(ROOT / "live/python") not in sys.path:
    sys.path.insert(0, str(ROOT / "live/python"))
from keypoint_annotations import LABELS, sha256_file, validate_annotations, visible_points
from prepare_capture_review import build_capture_annotations
from prepare_keypoint_review import repo_path, require_same_manifest
import pnp2


def distance_summary(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {"count": int(values.size), "mean_px": float(np.mean(values)) if values.size else None,
            "rms_px": float(np.sqrt(np.mean(values**2))) if values.size else None,
            "median_px": float(np.median(values)) if values.size else None,
            "max_px": float(np.max(values)) if values.size else None}


def solve_raw_points(points: dict, model: dict, camera: dict) -> dict:
    """Solve rectified pixels; report every residual in the original color image."""
    result = {"solved": False, "point_count": len(points), "points_raw_px": points,
              "residual_frame": "raw_color_pixels", "vector_definition": "projected_minus_observed"}
    if len(points) < 4:
        return {**result, "reason": "fewer_than_four_points"}
    try:
        matrix, coefficients, distortion = validate_camera(camera, (camera["image_height"], camera["image_width"], 3))
        labels = [label for label in LABELS if label in points]
        raw = np.asarray([points[label] for label in labels], dtype=float)
        rectified = rectify_points(raw, matrix, coefficients, distortion)
        cv2.setRNGSeed(0)
        pose = pnp2.solve_target_pose(dict(zip(labels, rectified)), model, matrix, np.zeros(5))
        rotation = cv2.Rodrigues(pose["rvec"])[0]
        xyz = (rotation @ np.asarray([model[name] for name in labels]).T).T + np.asarray(pose["tvec"]).reshape(3)
        if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
            raise ValueError("invalid_camera_depth")
        projected = distort_normalized(xyz[:, :2] / xyz[:, 2, None], coefficients, distortion)
        projected = projected * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]
        vectors = projected - raw
        errors = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(errors).all():
            raise ValueError("nonfinite_residual")
        return {**result, "solved": True, "reason": "ok", "labels": labels, "method": pose["method"],
                **distance_summary(errors), "inlier_count": int(np.count_nonzero(errors <= pnp2.RANSAC_REPROJ_ERROR_PX)),
                "inlier_threshold_px": pnp2.RANSAC_REPROJ_ERROR_PX,
                "per_hole": {name: {"observed_px": raw[i].tolist(), "projected_px": projected[i].tolist(),
                                     "residual_vector_px": vectors[i].tolist(), "distance_px": float(errors[i])}
                             for i, name in enumerate(labels)},
                "rvec": np.asarray(pose["rvec"]).reshape(3).tolist(),
                "tvec_mm": np.asarray(pose["tvec"]).reshape(3).tolist()}
    except (KeyError, TypeError, ValueError, RuntimeError, cv2.error) as exc:
        return {**result, "reason": f"{type(exc).__name__}: {exc}"}


class DiagnosticFrontend(PosePipeline):
    """Existing extractor with failure evidence retained before PnP/gating."""
    def extract(self, image, camera, geometry):
        output = {"attempted": True, "candidate_points_raw_px": [], "named_points_raw_px": {},
                  "association_quality_ok": False, "ambiguous": False, "reason": "not_processed"}
        try:
            self._load()
            matrix, coefficients, distortion = validate_camera(camera, image.shape)
            detections = self._detector(image, conf=float(self.config["confidence"]), verbose=False)
            boxes = [{"xyxy": [float(x) for x in box.xyxy[0].tolist()], "confidence": float(box.conf.item())}
                     for detection in detections for box in detection.boxes
                     if int(box.cls.item()) == int(self.config["target_class"])]
            selected = select_detection(boxes, image.shape, self.config.get("target_window"))
            x1, y1, x2, y2 = selected["xyxy"]
            padding = max(20, int(max(x2 - x1, y2 - y1) * .08))
            x1, y1 = max(0, int(x1) - padding), max(0, int(y1) - padding)
            x2, y2 = min(image.shape[1], int(x2) + padding), min(image.shape[0], int(y2) + padding)
            output.update(roi_bbox_xyxy=[x1, y1, x2, y2], detection_score=selected["confidence"])
            if x2 <= x1 or y2 <= y1:
                raise ValueError("invalid_detection_bbox")
            crop = image[y1:y2, x1:x2].copy()
            enhanced = self._modules["enhance"].preprocess_roi(crop)["final"]
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            edges = self._modules["Canny"].adaptive_canny_pipeline(enhanced_bgr)[4]
            contours, diagnostics = self._modules["latest_candidate_contours"], {}
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                candidates, _, _ = contours.filter_candidate_contours(
                    contours.prepare_edges(edges), enhanced_bgr, apply_layout=False,
                    diagnostics=diagnostics, gray_source=crop)
            # Candidate recall is deliberately measured before identity selection and PnP.
            for candidate in candidates:
                center = next((value for value in (candidate.aperture_center, candidate.fitted_center, candidate.center)
                               if value is not None), None)
                if np.asarray(center).shape == (2,) and np.isfinite(center).all():
                    output["candidate_points_raw_px"].append((np.asarray(center) + [x1, y1]).tolist())
            output["contour_diagnostics"] = diagnostics
            selection = self._modules["pose_correspondence"].select_correspondences(
                candidates, enhanced_bgr.shape, geometry["points_mm"], matrix, np.zeros(5), (x1, y1),
                point_transform=lambda points: rectify_points(points, matrix, coefficients, distortion),
                max_mean_error_px=float(self.config["max_reprojection_rms_px"]),
                min_inlier_count=int(self.config["min_inliers"]),
                min_correspondence_count=max(6, int(self.config["min_inliers"])), observed_prior=self.observed_prior)
            details = selection["diagnostics"]
            output["correspondence_diagnostics"] = details
            output["ambiguous"] = bool(details.get("ambiguous", False))
            output["association_quality_ok"] = bool(details.get("association_quality_ok", details.get("selection_quality_ok")))
            if selection["best"] is None:
                raise ValueError("no_layout_variant")
            output["named_points_raw_px"] = {name: np.asarray(point).tolist()
                                              for name, point in selection["best"]["image_points"].items()}
            output["selection_quality_ok"] = bool(details.get("selection_quality_ok", False))
            output["reason"] = selection["best"].get("pnp_error") or "ok"
        except Exception as exc:
            output["reason"] = f"{type(exc).__name__}: {exc}"
        return output


def point_metrics(reference: dict, frontend: dict, tolerance: float) -> dict:
    """One-to-one candidate recall and named pixel agreement, including failed poses."""
    if not frontend.get("attempted"):
        return {"available": False, "reason": "frontend_not_run"}
    names = list(reference)
    candidates = np.asarray(frontend["candidate_points_raw_px"], dtype=float).reshape(-1, 2)
    truth = np.asarray(list(reference.values()), dtype=float).reshape(-1, 2)
    distances = np.linalg.norm(truth[:, None, :] - candidates[None, :, :], axis=2)
    ownership = {}

    def assign(row, seen):
        for col in np.argsort(distances[row]):
            if distances[row, col] > tolerance or int(col) in seen:
                continue
            seen.add(int(col))
            if int(col) not in ownership or assign(ownership[int(col)], seen):
                ownership[int(col)] = row
                return True
        return False

    for row in range(len(names)):
        assign(row, set())
    labels = {}
    automatic = frontend["named_points_raw_px"]
    for name, point in reference.items():
        if name not in automatic:
            labels[name] = {"status": "missing"}
            continue
        distances_named = {other: float(np.linalg.norm(np.asarray(automatic[name]) - xy)) for other, xy in reference.items()}
        nearby = [other for other, distance in distances_named.items() if distance <= tolerance]
        status = "correct" if nearby == [name] else "wrong_name" if len(nearby) == 1 else "ambiguous" if nearby else "unlocalized"
        labels[name] = {"status": status, "same_name_distance_px": distances_named[name], "nearby_reference_labels": nearby}
    counts = {status: sum(item["status"] == status for item in labels.values())
              for status in ("correct", "wrong_name", "ambiguous", "unlocalized", "missing")}
    return {"available": True, "tolerance_px": tolerance, "visible_reference_count": len(reference),
            "candidate_matched_count": len(ownership), "candidate_count": len(candidates),
            "candidate_recall": len(ownership) / len(reference) if reference else None,
            "named_counts": counts, "per_hole": labels,
            "named_accuracy_on_visible": counts["correct"] / len(reference) if reference else None,
            "automatic_labels_without_visible_reference": [name for name in automatic if name not in reference]}


def acceptance(pose, frontend, camera, geometry, config):
    reasons = []
    if not pose["solved"]:
        reasons.append("pnp_unsolved")
    else:
        for field, limit, reason in (("rms_px", config["max_reprojection_rms_px"], "rms_exceeded"),
                                     ("max_px", config["max_point_error_px"], "point_error_exceeded")):
            if pose[field] > limit:
                reasons.append(reason)
        if pose["inlier_count"] < config["min_inliers"]:
            reasons.append("insufficient_inliers")
    if not frontend.get("association_quality_ok") or frontend.get("ambiguous"):
        reasons.append("association_rejected")
    if not camera.get("validated") or not camera.get("version"):
        reasons.append("camera_not_independently_validated")
    if geometry.get("validated") is not True or geometry.get("status") != "measured_print":
        reasons.append("printed_geometry_not_validated")
    return {"accepted": not reasons, "reasons": reasons, "authorizes_robot_motion": False}


def validate_capture_annotations(path: Path, root: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_annotations(data)
    if data["schema_version"] != 2:
        raise ValueError("Use legacy evaluator for schema 1; new-camera diagnosis requires schema 2")
    manifest = repo_path(root, data["provenance"]["manifest_path"])
    expected = build_capture_annotations(manifest, root)
    require_same_manifest(data, expected)
    camera_path = repo_path(root, data["provenance"]["camera_config_path"])
    camera = json.loads(camera_path.read_text(encoding="utf-8-sig"))
    if camera.get("device_name", "").split()[-1] not in {"D415", "D435"}:
        raise ValueError("Diagnostic capture requires a D415 or D435 identity")
    return data, camera


def evaluate(sessions: list[tuple[dict, dict]], geometries: list[dict], config: dict,
             *, root: Path = ROOT, camera_overrides=(), tolerance_px: float, frontend=None) -> dict:
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("positive_frozen_annotation_tolerance_required")
    if not geometries or len({g["version"] for g in geometries}) != len(geometries):
        raise ValueError("unique_geometry_versions_required")
    rows, pending, seen, groups = [], [], set(), {}
    matched_override_ids = set()
    for annotations, sdk in sessions:
        variants = {"sdk": sdk}
        for camera in camera_overrides:
            if camera.get("serial") != sdk["serial"]:
                continue
            for key in ("device_name", "stream", "image_width", "image_height"):
                if camera.get(key) != sdk.get(key):
                    raise ValueError(f"camera_override_identity_mismatch:{key}")
            validate_camera(camera, (sdk["image_height"], sdk["image_width"], 3))
            version = camera.get("version")
            if not isinstance(version, str) or not version or version in variants:
                raise ValueError("unique_camera_override_version_required")
            if camera.get("intrinsics_sha256") != intrinsics_signature(camera):
                raise ValueError("camera_override_fingerprint_mismatch")
            variants[version] = camera
            matched_override_ids.add(id(camera))
        for entry in annotations["images"]:
            identity = entry["image_sha256"]
            if identity in seen:
                raise ValueError("duplicate_image_content_across_sessions")
            seen.add(identity)
            metadata = entry.get("capture_metadata", {})
            baseline_geometry = geometries[0]
            if baseline_geometry["status"] == "measured_print" and (
                    baseline_geometry["specimen_id"] != metadata.get("specimen_id") or
                    baseline_geometry["printing_batch_id"] != metadata.get("printing_batch_id")):
                raise ValueError("frontend_geometry_does_not_match_capture_specimen")
            for kind, value in (("session", entry["session_id"]), ("batch", entry["batch_id"]),
                                ("specimen", metadata.get("specimen_id")), ("printing_batch", metadata.get("printing_batch_id"))):
                if value:
                    if (kind, value) in groups and groups[kind, value] != entry["split"]:
                        raise ValueError(f"{kind}_crosses_dataset_splits:{value}")
                    groups[kind, value] = entry["split"]
            if entry["review_status"] != "reviewed":
                pending.append(entry["image"])
                continue
            human = visible_points(entry)
            image_path = repo_path(root, entry["image"])
            image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            extracted = frontend.extract(image, sdk, geometries[0]) if frontend else {
                "attempted": False, "candidate_points_raw_px": [], "named_points_raw_px": {}, "reason": "manual_only"}
            auto = extracted["named_points_raw_px"]
            common = [name for name in LABELS if name in human and name in auto]
            comparisons, skipped_geometry_variants = {}, []
            for camera_name, camera in variants.items():
                for geometry in geometries:
                    if geometry["status"] == "measured_print" and (
                            geometry["specimen_id"] != metadata.get("specimen_id") or
                            geometry["printing_batch_id"] != metadata.get("printing_batch_id")):
                        skipped_geometry_variants.append({"geometry_version": geometry["version"],
                                                          "camera_variant": camera_name, "reason": "different_measured_specimen"})
                        continue
                    model = geometry["points_mm"]
                    branches = {"human_all": solve_raw_points(human, model, camera),
                                "automatic_all": solve_raw_points(auto, model, camera),
                                "human_common": solve_raw_points({k: human[k] for k in common}, model, camera),
                                "automatic_common": solve_raw_points({k: auto[k] for k in common}, model, camera)}
                    key = f"{camera_name}/{geometry['version']}"
                    comparisons[key] = {"camera_variant": camera_name, "camera_sha256": intrinsics_signature(camera),
                                        "geometry_version": geometry["version"], "geometry_status": geometry["status"],
                                        "geometry_sha256": geometry.get("source_sha256"), "branches": branches,
                                        "final_gate": acceptance(branches["automatic_all"], extracted, camera, geometry, config)}
            rows.append({"image": entry["image"], "image_sha256": identity, "session_id": entry["session_id"],
                         "camera_serial": sdk["serial"], "camera_model": sdk["device_name"], "split": entry["split"],
                         "capture_metadata": metadata, "human_points_raw_px": human, "common_labels": common,
                         "frontend": extracted, "point_metrics": point_metrics(human, extracted, tolerance_px),
                         "comparisons": comparisons, "skipped_geometry_variants": skipped_geometry_variants})
    if len(matched_override_ids) != len(camera_overrides):
        raise ValueError("camera_override_has_no_matching_capture_session")
    summary = {}
    for key in sorted({key for row in rows for key in row["comparisons"]}):
        relevant = [row for row in rows if key in row["comparisons"]]
        summary[key] = {"image_count": len(relevant), "branches": {},
                        "automatic_pnp_solved_count": sum(row["comparisons"][key]["branches"]["automatic_all"]["solved"] for row in relevant),
                        "final_accepted_count": sum(row["comparisons"][key]["final_gate"]["accepted"] for row in relevant)}
        paired = [row for row in relevant if all(row["comparisons"][key]["branches"][name]["solved"]
                                                for name in ("human_common", "automatic_common"))]
        summary[key]["paired_human_automatic"] = {
            "image_count": len(paired), "images": [row["image"] for row in paired],
            "human_image_mean_px": distance_summary(row["comparisons"][key]["branches"]["human_common"]["mean_px"] for row in paired),
            "automatic_image_mean_px": distance_summary(row["comparisons"][key]["branches"]["automatic_common"]["mean_px"] for row in paired),
            "automatic_minus_human_mean_px": distance_summary(
                row["comparisons"][key]["branches"]["automatic_common"]["mean_px"] -
                row["comparisons"][key]["branches"]["human_common"]["mean_px"] for row in paired)}
        for branch in ("human_all", "automatic_all", "human_common", "automatic_common"):
            poses = [row["comparisons"][key]["branches"][branch] for row in relevant]
            solved = [pose for pose in poses if pose["solved"]]
            summary[key]["branches"][branch] = {
                "solved_count": len(solved), "failed_count": len(poses) - len(solved),
                "per_image_mean_distance_px": distance_summary(pose["mean_px"] for pose in solved),
                "pooled_per_hole_distance_px": distance_summary(hole["distance_px"] for pose in solved for hole in pose["per_hole"].values())}
    metrics = [row["point_metrics"] for row in rows if row["point_metrics"]["available"]]
    visible = sum(row["visible_reference_count"] for row in metrics)
    matched = sum(row["candidate_matched_count"] for row in metrics)
    baseline = f"sdk/{geometries[0]['version']}"
    paired_variants = {}
    for key in summary:
        if key == baseline:
            continue
        paired_variants[key] = {}
        for branch in ("human_all", "automatic_all", "human_common", "automatic_common"):
            pairs = [row for row in rows if baseline in row["comparisons"] and key in row["comparisons"]
                     and row["comparisons"][baseline]["branches"][branch]["solved"]
                     and row["comparisons"][key]["branches"][branch]["solved"]]
            paired_variants[key][branch] = {"image_count": len(pairs), "images": [row["image"] for row in pairs],
                "baseline_image_mean_px": distance_summary(row["comparisons"][baseline]["branches"][branch]["mean_px"] for row in pairs),
                "variant_image_mean_px": distance_summary(row["comparisons"][key]["branches"][branch]["mean_px"] for row in pairs)}
    strata = {}
    for field in ("distance", "tilt", "image_position", "camera_model", "camera_serial"):
        strata[field] = {}
        def stratum_value(row):
            value = (row[field] if field in ("camera_model", "camera_serial") else
                     row["capture_metadata"].get("conditions", {}).get(field, "unrecorded"))
            return str(value)
        for value in sorted({stratum_value(row) for row in rows}):
            subset = [row for row in rows if stratum_value(row) == value]
            strata[field][value] = {"image_count": len(subset), "comparisons": {}}
            for key in summary:
                for branch in ("human_all", "automatic_all"):
                    solved = [row["comparisons"][key]["branches"][branch] for row in subset
                              if key in row["comparisons"] and row["comparisons"][key]["branches"][branch]["solved"]]
                    strata[field][value]["comparisons"][f"{key}/{branch}"] = {
                        "solved_count": len(solved), "image_mean_distance_px": distance_summary(pose["mean_px"] for pose in solved)}
    synthetic_rows = sum(row["capture_metadata"].get("synthetic") is True for row in rows)
    return {"schema_version": 2, "kind": "camera_geometry_diagnostic_v1", "independent_test_set": False,
            "synthetic_data_present": bool(synthetic_rows),
            "physical_pose_accuracy_verified": False, "motion_authorized": False,
            "reviewed_image_count": len(rows), "pending_review_images": pending, "tolerance_px": tolerance_px,
            "frontend_points_frozen_using": {"camera": "session SDK intrinsics", "geometry_version": geometries[0]["version"]},
            "stages": {"frontend_evaluated_images": len(metrics), "visible_reference_count": visible,
                       "candidate_matched_count": matched, "candidate_recall": matched / visible if visible else None,
                       "named_counts": {status: sum(row["named_counts"][status] for row in metrics)
                                        for status in ("correct", "wrong_name", "ambiguous", "unlocalized", "missing")}},
            "summary": summary, "paired_variants_against_baseline": paired_variants, "strata": strata, "rows": rows,
            "notes": ["All PnP branches retain every supplied named point, including outliers.",
                      "Common branches use identical image and hole-name sets; camera/geometry variants freeze raw points.",
                      "Compare rows by capture_metadata.conditions (distance, tilt, image_position).",
                      "Correct means unique pixel agreement within the declared tolerance; it is not absolute pose truth.",
                      "These development diagnostics never establish independent test accuracy or authorize robot motion."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--geometry", type=Path, action="append", required=True)
    parser.add_argument("--camera-override", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, default=ROOT / "config/realsense_config.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tolerance-px", type=float, required=True, help="Freeze after repeat-annotation study; not a tuning parameter")
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New file; existing reports are preserved")
    args = parser.parse_args(argv)
    from .geometry_reference import load_geometry_reference
    sessions = [validate_capture_annotations(path, args.root) for path in args.annotations]
    geometries = [load_geometry_reference(path) for path in args.geometry]
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    cameras = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.camera_override]
    result = evaluate(sessions, geometries, config, root=args.root, camera_overrides=cameras,
                      tolerance_px=args.tolerance_px, frontend=None if args.manual_only else DiagnosticFrontend(config))
    result["provenance"] = {"annotations": [{"path": str(path), "sha256": sha256_file(path)} for path in args.annotations],
                            "geometry": [{"path": str(path), "sha256": sha256_file(path)} for path in args.geometry],
                            "camera_overrides": [{"path": str(path), "sha256": sha256_file(path)} for path in args.camera_override],
                            "config_sha256": sha256_file(args.config), "evaluator_sha256": sha256_file(Path(__file__)),
                            "pnp_sha256": sha256_file(Path(pnp2.__file__))}
    result["provenance"]["pipeline_config"] = config
    result["provenance"]["source_sha256"] = {
        name: sha256_file(ROOT / name) for name in (
            "vision/pipeline.py", "vision/contracts.py", "live/python/pose_correspondence.py",
            "live/python/observed_layout.py", "live/python/latest_candidate_contours.py",
            "live/python/center_fit.py", "live/python/enhance.py", "live/python/Canny.py")}
    result["provenance"]["frontend_assets"] = {}
    if not args.manual_only:
        for key in ("weights", "observed_layout_prior"):
            if config.get(key):
                asset = Path(config[key])
                if not asset.is_absolute():
                    asset = ROOT / asset
                result["provenance"]["frontend_assets"][key] = {
                    "path": str(asset), "sha256": sha256_file(asset) if asset.is_file() else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"report": str(args.output), "reviewed_images": result["reviewed_image_count"],
                      "pending_images": len(result["pending_review_images"]), "stages": result["stages"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
