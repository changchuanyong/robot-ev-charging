"""Fit/check RGB intrinsics using measured chessboards and disjoint held-out views."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .contracts import distort_normalized, intrinsics_signature, rectify_points, validate_camera


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def board_points(board):
    for key in ("columns", "rows"):
        if type(board.get(key)) is not int or board[key] < 3:
            raise ValueError("board_requires_at_least_3x3_inner_corners")
    size = board.get("measured_square_mm")
    if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0:
        raise ValueError("measured_positive_square_size_required")
    for key in ("board_id", "measured_by", "measured_at", "instrument"):
        if not isinstance(board.get(key), str) or not board[key].strip():
            raise ValueError(f"board_measurement_requires_{key}")
    points = np.zeros((board["columns"] * board["rows"], 3), np.float32)
    points[:, :2] = np.mgrid[0:board["columns"], 0:board["rows"]].T.reshape(-1, 2) * size
    return points


def evaluate_board_view(objects, pixels, camera):
    """Independent per-view pose is a nuisance parameter; no camera fitting here."""
    matrix, coefficients, model = validate_camera(camera, (camera["image_height"], camera["image_width"], 3))
    rectified = rectify_points(np.asarray(pixels, float).reshape(-1, 2), matrix, coefficients, model)
    ok, rvec, tvec = cv2.solvePnP(objects, rectified, matrix, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("board_pose_unsolved")
    xyz = (cv2.Rodrigues(rvec)[0] @ np.asarray(objects).T).T + tvec.reshape(3)
    if np.any(xyz[:, 2] <= 0):
        raise ValueError("board_behind_camera")
    projected = distort_normalized(xyz[:, :2] / xyz[:, 2, None], coefficients, model)
    projected = projected * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]
    residual = projected - np.asarray(pixels).reshape(-1, 2)
    distances = np.linalg.norm(residual, axis=1)
    return {"mean_px": float(np.mean(distances)), "rms_px": float(np.sqrt(np.mean(distances**2))),
            "max_px": float(np.max(distances)), "residual_vectors_px": residual.tolist(),
            "per_corner_distance_px": distances.tolist(), "rvec": rvec.reshape(3).tolist(),
            "tvec_mm": tvec.reshape(3).tolist()}


def calibrate_views(camera, board, fit_views, validation_views, version):
    """Views contain immutable image IDs/hashes and observed raw chessboard corners."""
    objects = board_points(board)
    if not isinstance(version, str) or not version.strip():
        raise ValueError("calibration_version_required")
    validate_camera(camera, (camera["image_height"], camera["image_width"], 3))
    if len(fit_views) < 6 or len(validation_views) < 3:
        raise ValueError("at_least_six_fit_and_three_heldout_views_required")
    hashes, names = set(), set()
    for view in [*fit_views, *validation_views]:
        if view["image_sha256"] in hashes or view["image"] in names:
            raise ValueError("fit_and_validation_images_must_be_unique_and_disjoint")
        hashes.add(view["image_sha256"])
        names.add(view["image"])
        pixels = np.asarray(view["pixels"], float)
        if pixels.shape != (len(objects), 2) or not np.isfinite(pixels).all():
            raise ValueError("invalid_chessboard_corners")
        if np.any(pixels < 0) or np.any(pixels >= [camera["image_width"], camera["image_height"]]):
            raise ValueError("chessboard_corner_outside_image")
    image_points = [np.asarray(view["pixels"], np.float32).reshape(-1, 1, 2) for view in fit_views]
    rms, matrix, coefficients, _, _ = cv2.calibrateCamera(
        [objects.copy() for _ in fit_views], image_points, (camera["image_width"], camera["image_height"]), None, None)
    if not np.isfinite(rms):
        raise ValueError("calibration_failed")
    fitted = copy.deepcopy(camera)
    fitted.update(camera_matrix=matrix.tolist(), dist_coeffs=coefficients.reshape(5).tolist(),
                  distortion_model="brown_conrady", version=version, validated=False,
                  source="measured_chessboard_fit", is_placeholder=False, is_approximate=False,
                  validation_status="pending_independent_review")
    fitted["source_sdk_intrinsics_sha256"] = intrinsics_signature(camera)
    fitted["intrinsics_sha256"] = intrinsics_signature(fitted)
    validate_camera(fitted, (camera["image_height"], camera["image_width"], 3))
    report_rows = []
    for split, views in (("fit", fit_views), ("heldout", validation_views)):
        for view in views:
            report_rows.append({"image": view["image"], "image_sha256": view["image_sha256"], "split": split,
                                "corners_raw_px": np.asarray(view["pixels"]).tolist(),
                                "sdk": evaluate_board_view(objects, view["pixels"], camera),
                                "fitted": evaluate_board_view(objects, view["pixels"], fitted)})
    summary = {}
    for split in ("fit", "heldout"):
        subset = [row for row in report_rows if row["split"] == split]
        summary[split] = {"view_count": len(subset)}
        for name in ("sdk", "fitted"):
            errors = np.array([error for row in subset for error in row[name]["per_corner_distance_px"]])
            summary[split][name] = {"pooled_corner_mean_px": float(np.mean(errors)),
                                   "pooled_corner_rms_px": float(np.sqrt(np.mean(errors**2))),
                                   "max_corner_distance_px": float(np.max(errors))}
    return fitted, {"schema_version": 1, "kind": "rgb_intrinsics_check_v1", "version": version,
                    "source_camera_sha256": intrinsics_signature(camera), "fitted_camera_sha256": intrinsics_signature(fitted),
                    "board": board, "fit_opencv_rms_px": float(rms), "summary": summary, "rows": report_rows,
                    "validated": False, "physical_pose_accuracy_verified": False,
                    "notes": ["Held-out images do not enter calibrateCamera; each view still needs its own estimated board pose.",
                              "Board pose shares the evaluated camera model and is not independent absolute pose truth.",
                              "Review held-out corner residuals, pose/edge coverage and board measurements before accepting this calibration.",
                              "Recommended collection budget is 30 fit/check plus 10 independent-pose validation images per camera."]}


def load_views(spec_path):
    """Every listed image must belong to the bound finalized capture session."""
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if spec.get("schema_version") != 1:
        raise ValueError("intrinsics_check_spec_requires_schema_1")
    board = spec["board"]
    board_points(board)
    if not isinstance(spec.get("capture_manifest"), str) or not spec["capture_manifest"].strip():
        raise ValueError("capture_manifest_path_required")
    for key in ("fit_images", "validation_images"):
        if not isinstance(spec.get(key), list) or any(not isinstance(name, str) or not name for name in spec[key]):
            raise ValueError(f"image_path_list_required:{key}")
    manifest_path = (spec_path.parent / spec["capture_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("kind") != "realsense_rgb_capture" or manifest.get("status") not in {"completed", "failed", "interrupted"}:
        raise ValueError("finalized_RGB_capture_session_required")
    session = manifest_path.parent
    camera_path = session / "camera.json"
    if manifest.get("camera_file") != "camera.json" or _sha(camera_path) != manifest["camera_sha256"]:
        raise ValueError("capture_camera_binding_changed")
    camera = json.loads(camera_path.read_text(encoding="utf-8-sig"))
    if camera.get("intrinsics_sha256") != intrinsics_signature(camera):
        raise ValueError("capture_intrinsics_fingerprint_changed")
    if camera.get("device_name", "").split()[-1] not in {"D415", "D435"}:
        raise ValueError("D415_or_D435_required")
    frames = {frame["image"]: frame for frame in manifest["frames"]}
    if len(frames) != len(manifest["frames"]):
        raise ValueError("duplicate_capture_frame_paths")
    groups, rejected = {}, []
    for group in ("fit_images", "validation_images"):
        groups[group] = []
        for name in spec[group]:
            image_path = (session / name).resolve()
            if not image_path.is_relative_to(session) or name not in frames:
                raise ValueError("image_not_bound_to_capture_session")
            digest = _sha(image_path)
            if digest != frames[name]["image_sha256"]:
                raise ValueError("capture_image_changed")
            image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("unreadable_calibration_image")
            validate_camera(camera, image.shape)
            ok, corners = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                (board["columns"], board["rows"]), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if not ok:
                rejected.append({"image": name, "group": group, "reason": "chessboard_not_found", "image_sha256": digest})
            else:
                groups[group].append({"image": name, "image_sha256": digest, "pixels": corners.reshape(-1, 2).tolist()})
    # Reject duplicate requests even when a duplicate image failed corner extraction.
    requested = [*spec["fit_images"], *spec["validation_images"]]
    if len(requested) != len(set(requested)):
        raise ValueError("fit_and_validation_images_must_be_disjoint")
    requested_hashes = [frames[name]["image_sha256"] for name in requested]
    if len(requested_hashes) != len(set(requested_hashes)):
        raise ValueError("duplicate_calibration_image_content")
    return camera, board, groups, rejected, {"spec_sha256": _sha(spec_path), "capture_manifest_sha256": _sha(manifest_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Board measurements, session manifest, fit/validation image lists")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; prior calibrations are preserved")
    args = parser.parse_args(argv)
    camera, board, groups, rejected, provenance = load_views(args.spec)
    fitted, report = calibrate_views(camera, board, groups["fit_images"], groups["validation_images"], args.version)
    report.update(rejected_views=rejected, provenance=provenance)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in (("camera.fitted.json", fitted), ("intrinsics_report.json", report)):
        with (args.output_dir / name).open("x", encoding="utf-8") as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": report["summary"], "validated": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
