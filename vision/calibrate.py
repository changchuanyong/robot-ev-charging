"""Solve/independently validate eye-in-hand calibration from paired static observations.

Input JSON uses metres, `source: measured|synthetic`, and `pairs` containing
id, split (train|validation), T_base_flange, camera_time, robot_time, and either
T_camera_board or image. Images additionally require camera and board objects.
The camera/robot times must already share a measured clock mapping. Output
does not invent the independent plug TCP or force-sensor extrinsics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import atomic_json, rectify_points, validate_camera


def transform(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("finite_4x4_transform_required")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("invalid_homogeneous_transform")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("rotation_must_be_SO3")
    return matrix


def rotation_angle(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))


def board_pose(image_path: Path, camera: dict, board: dict) -> np.ndarray:
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"calibration_image_unreadable: {image_path}")
    matrix, coefficients, model = validate_camera(camera, image.shape)
    columns, rows = int(board["columns"]), int(board["rows"])
    square = float(board["square_size_m"])
    if min(columns, rows) < 3 or not np.isfinite(square) or square <= 0:
        raise ValueError("invalid_measured_checkerboard_dimensions")
    found, corners = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (columns, rows))
    if not found:
        raise ValueError(f"checkerboard_not_found: {image_path}")
    points = np.zeros((columns * rows, 3), dtype=float)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square
    pixels = rectify_points(corners.reshape(-1, 2), matrix, coefficients, model)
    ok, rvec, tvec = cv2.solvePnP(points, pixels, matrix, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("checkerboard_PnP_failed")
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
    if np.any(((pose[:3, :3] @ points.T).T + pose[:3, 3])[:, 2] <= 0):
        raise ValueError("checkerboard_behind_camera")
    return transform(pose)


def prepare_pairs(data: dict, base_dir: Path = Path(".")) -> tuple[list, list]:
    import cv2
    if data.get("translation_unit") != "m":
        raise ValueError("calibration_input_must_use_metres")
    max_skew = float(data["max_pair_skew_s"])
    if not np.isfinite(max_skew) or max_skew <= 0:
        raise ValueError("positive_measured_pair_skew_bound_required")
    identifiers = set()
    poses = []
    training, validation = [], []
    for row in data["pairs"]:
        identifier = str(row["id"])
        if not identifier or identifier in identifiers:
            raise ValueError("duplicate_or_missing_pose_id")
        identifiers.add(identifier)
        if row["split"] not in {"train", "validation"}:
            raise ValueError("split_must_be_train_or_validation")
        camera_time, robot_time = float(row["camera_time"]), float(row["robot_time"])
        if not np.isfinite([camera_time, robot_time]).all() or abs(camera_time - robot_time) > max_skew:
            raise ValueError("unsynchronized_calibration_pair")
        flange = transform(row["T_base_flange"])
        if any(np.allclose(flange, other, atol=1e-7) for other in poses):
            raise ValueError("repeated_robot_pose_is_not_an_independent_sample")
        poses.append(flange)
        if "T_camera_board" in row:
            board = transform(row["T_camera_board"])
        else:
            board = board_pose(base_dir / row["image"], data["camera"], data["board"])
        pair = (identifier, flange, board)
        (training if row["split"] == "train" else validation).append(pair)
    if len(training) < 3 or len(validation) < 2:
        raise ValueError("at_least_three_train_and_two_independent_validation_poses_required")
    # Tsai needs relative rotations about more than one direction. Translation-
    # only or single-axis data cannot determine the full camera extrinsic.
    for pose_index in (1, 2):
        relative_vectors = []
        reference = training[0][pose_index][:3, :3]
        for pair in training[1:]:
            relative = reference.T @ pair[pose_index][:3, :3]
            relative_vectors.append(cv2.Rodrigues(relative)[0].reshape(3))
        singular = np.linalg.svd(np.asarray(relative_vectors), compute_uv=False)
        if len(singular) < 2 or singular[1] < np.deg2rad(2):
            raise ValueError("degenerate_rotation_coverage")
    return training, validation


def average_pose(poses: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4)
    u, _, vt = np.linalg.svd(sum(pose[:3, :3] for pose in poses))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    result[:3, :3] = u @ correction @ vt
    result[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
    return result


def residuals(pairs, camera: np.ndarray, board_reference: np.ndarray) -> list[dict]:
    reference_inverse = np.linalg.inv(board_reference)
    results = []
    for identifier, flange, board in pairs:
        error = reference_inverse @ flange @ camera @ board
        results.append({"id": identifier, "translation_m": float(np.linalg.norm(error[:3, 3])),
                        "rotation_rad": rotation_angle(error[:3, :3])})
    return results


def solve_calibration(data: dict, base_dir: Path = Path(".")) -> dict:
    import cv2
    training, validation = prepare_pairs(data, base_dir)
    rotation, translation = cv2.calibrateHandEye(
        [pair[1][:3, :3] for pair in training], [pair[1][:3, 3] for pair in training],
        [pair[2][:3, :3] for pair in training], [pair[2][:3, 3] for pair in training],
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    camera = np.eye(4)
    camera[:3, :3], camera[:3, 3] = rotation, translation.reshape(3)
    camera = transform(camera)
    reference = average_pose([flange @ camera @ board for _, flange, board in training])
    train_results = residuals(training, camera, reference)
    validation_results = residuals(validation, camera, reference)
    acceptance = data.get("acceptance")
    passed = False
    if acceptance is not None:
        translation_limit = float(acceptance["max_translation_m"])
        rotation_limit = float(acceptance["max_rotation_rad"])
        if not np.isfinite([translation_limit, rotation_limit]).all() or min(translation_limit, rotation_limit) <= 0:
            raise ValueError("positive_independent_validation_limits_required")
        passed = all(row["translation_m"] <= translation_limit and row["rotation_rad"] <= rotation_limit
                     for row in validation_results)
    source = data.get("source", "unspecified")
    source_hash = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    version = str(data.get("version") or "handeye-" + source_hash[:12])
    validated = bool(passed and source == "measured" and data.get("pairing_verified", False))
    return {
        "schema_version": 1, "mount": "eye_in_hand", "translation_unit": "m", "version": version,
        "validated": validated, "source": source, "method": "OpenCV_Tsai",
        "opencv_version": cv2.__version__, "input_sha256": source_hash,
        "camera_serial": data.get("camera_serial"),
        "T_flange_camera": camera.tolist(), "T_base_board_reference": reference.tolist(),
        "T_flange_tool": None, "T_tool_sensor": None,
        "tool_calibration_validated": False, "force_reference_validated": False,
        "train_pose_count": len(training), "validation_pose_count": len(validation),
        "train_residuals": train_results, "validation_residuals": validation_results,
        "acceptance": acceptance, "acceptance_passed": passed,
        "validation_scope": "Held-out fixed-board consistency; no external absolute-accuracy claim.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples.resolve() == args.output.resolve():
        parser.error("output must differ from the source observations")
    data = json.loads(args.samples.read_text(encoding="utf-8-sig"))
    result = solve_calibration(data, args.samples.resolve().parent)
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "validated": result["validated"],
                      "acceptance_passed": result["acceptance_passed"]}, ensure_ascii=False))
    return 0 if result["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
