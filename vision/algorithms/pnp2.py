from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np


# =========================================================
# 历史批量评估的模型与内参路径
# =========================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_CSV_PATH = ROOT_DIR / "config" / "charging_port_model.csv"
CAMERA_CONFIG_PATH = ROOT_DIR / "config" / "camera_intrinsics.json"
RANSAC_REPROJ_ERROR_PX = 10.0
RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS = 500
RANSAC_MIN_INLIER_RATIO = 0.70

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, bool]:
    data = load_json(path)

    if "camera_matrix" not in data:
        raise KeyError(f"camera intrinsics config missing camera_matrix: {path}")
    if "dist_coeffs" not in data:
        raise KeyError(f"camera intrinsics config missing dist_coeffs: {path}")

    K = np.array(data["camera_matrix"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {K.shape}")

    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if dist_coeffs.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            "dist_coeffs must contain 4, 5, 8, 12, or 14 values, "
            f"got {dist_coeffs.size}"
        )

    return K, dist_coeffs, bool(data.get("is_placeholder", False))


def normalize_header_map(fieldnames: List[str]) -> Dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames if name is not None}


def find_first_existing(header_map: Dict[str, str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in header_map:
            return header_map[c]
    return None


def load_object_points_from_csv(csv_path: Path) -> Dict[str, Tuple[float, float, float]]:
    """
    读取三维模型点。兼容：
    点名列：
        name / point_name / id / label

    坐标列：
        x / x_mm / model_x
        y / y_mm / model_y
        z / z_mm / model_z   (没有则默认 0)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"模型 csv 不存在: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "charging_port_model.csv 缺少表头。"
                "请至少包含 label,x_mm,y_mm,z_mm 或 name,x,y,z 这样的表头。"
            )

        header_map = normalize_header_map(reader.fieldnames)

        name_col = find_first_existing(header_map, ["name", "point_name", "id", "label"])
        x_col = find_first_existing(header_map, ["x", "x_mm", "model_x"])
        y_col = find_first_existing(header_map, ["y", "y_mm", "model_y"])
        z_col = find_first_existing(header_map, ["z", "z_mm", "model_z"])

        if name_col is None or x_col is None or y_col is None:
            raise ValueError(
                "charging_port_model.csv 表头不符合要求。\n"
                "当前支持：\n"
                "  点名列: name / point_name / id / label\n"
                "  X列   : x / x_mm / model_x\n"
                "  Y列   : y / y_mm / model_y\n"
                "  Z列   : z / z_mm / model_z（可选）"
            )

        object_points: Dict[str, Tuple[float, float, float]] = {}

        for row_idx, row in enumerate(reader, start=2):
            name = str(row[name_col]).strip()
            if name == "":
                continue

            try:
                x = float(row[x_col])
                y = float(row[y_col])
                z = float(row[z_col]) if (z_col is not None and str(row[z_col]).strip() != "") else 0.0
            except Exception as e:
                raise ValueError(f"第 {row_idx} 行模型点解析失败: {e}")

            object_points[name] = (x, y, z)

    if len(object_points) < 4:
        raise ValueError(f"模型点不足 4 个，当前只有 {len(object_points)} 个")

    return object_points


def build_correspondences(
    object_points_dict: Dict[str, Tuple[float, float, float]],
    image_points_dict: Dict[str, Tuple[float, float]]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names = [name for name in object_points_dict.keys() if name in image_points_dict]

    if len(names) < 4:
        raise ValueError(
            f"有效对应点不足 4 个。当前只匹配到 {len(names)} 个点: {names}\n"
            "请检查 charging_port_model.csv 里的 label 是否与 latest_fitted_centers.json 中的点名一致。"
        )

    object_points = np.array([object_points_dict[name] for name in names], dtype=np.float64)
    image_points = np.array([image_points_dict[name] for name in names], dtype=np.float64)

    return names, object_points, image_points


def is_planar_points(object_points: np.ndarray, atol: float = 1e-9) -> bool:
    if object_points.shape[0] < 4:
        return False
    z0 = object_points[0, 2]
    return np.allclose(object_points[:, 2], z0, atol=atol)


def compute_reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    proj = proj.reshape(-1, 2)
    return np.linalg.norm(proj - image_points, axis=1)


def build_per_point_errors(
    names: list[str],
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, float]:
    errors = compute_reprojection_errors(
        object_points, image_points, rvec, tvec, K, dist_coeffs
    )
    return {name: float(err) for name, err in zip(names, errors)}


def all_points_in_front(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray
) -> bool:
    R, _ = cv2.Rodrigues(rvec)
    cam_pts = (R @ object_points.T + tvec.reshape(3, 1)).T
    return bool(np.isfinite(cam_pts).all() and np.all(cam_pts[:, 2] > 0))


def validate_pnp_inputs(object_points, image_points, K, dist_coeffs):
    if object_points.shape != (len(image_points), 3) or image_points.shape != (len(object_points), 2):
        raise ValueError("PnP requires Nx3 object points and Nx2 image points")
    if K.shape != (3, 3) or dist_coeffs.size not in (4, 5, 8, 12, 14):
        raise ValueError("invalid PnP camera parameter shape")
    if not all(np.isfinite(value).all() for value in (object_points, image_points, K, dist_coeffs)):
        raise ValueError("nonfinite PnP input")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0., 0., 1.]):
        raise ValueError("invalid PnP camera matrix")
    for name, points in (("object", object_points), ("image", image_points)):
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError(f"degenerate {name} correspondences: duplicate points")
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[0] == 0 or singular[1] <= singular[0] * 1e-6:
            raise ValueError(f"degenerate {name} correspondences: collinear points")


def evaluate_pose_candidate(object_points, image_points, rvec, tvec, K, dist_coeffs,
                            source, idx=0, fit_count=None):
    """Every solver uses the same full-point residuals and measured inlier threshold."""
    rvec, tvec = np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64)
    if rvec.size != 3 or tvec.size != 3 or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    rvec, tvec = rvec.reshape(3, 1).copy(), tvec.reshape(3, 1).copy()
    try:
        if not all_points_in_front(object_points, rvec, tvec):
            return None
        errors = compute_reprojection_errors(object_points, image_points, rvec, tvec, K, dist_coeffs)
    except cv2.error:
        return None
    if not np.isfinite(errors).all():
        return None
    rms = float(np.sqrt(np.mean(errors**2)))
    if not np.isfinite(rms):
        return None
    inliers = np.flatnonzero(errors <= RANSAC_REPROJ_ERROR_PX)
    return {
        "idx": idx, "source": source, "rvec": rvec, "tvec": tvec,
        "reproj_error": float(np.mean(errors)), "reproj_rms_error": rms,
        "max_reproj_error": float(np.max(errors)), "selection_error": rms,
        "all_in_front": True, "inlier_indices": inliers.tolist(),
        "inlier_count": int(len(inliers)), "inlier_ratio": float(len(inliers) / len(errors)),
        "inlier_reproj_error": float(np.mean(errors[inliers])) if len(inliers) else None,
        "fit_count": len(object_points) if fit_count is None else int(fit_count),
        "errors": errors, "refined": False,
    }


def refine_pose_candidate(candidate, object_points, image_points, K, dist_coeffs, fit_indices=None):
    """Keep an initial pose if LM worsens any full-point mean/RMS/maximum residual."""
    if candidate is None:
        return None
    indices = np.arange(len(object_points)) if fit_indices is None else fit_indices
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[indices], image_points[indices], K, dist_coeffs,
            candidate["rvec"].copy(), candidate["tvec"].copy(),
        )
    except cv2.error:
        return candidate
    refined = evaluate_pose_candidate(object_points, image_points, rvec, tvec, K, dist_coeffs,
                                      candidate["source"], candidate["idx"], len(indices))
    metrics = ("reproj_error", "reproj_rms_error", "max_reproj_error")
    if refined is not None and all(refined[key] <= candidate[key] + 1e-9 for key in metrics):
        refined["refined"] = True
        return refined
    return candidate


def solve_pnp_ransac_candidate(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict | None:
    if len(object_points) < 6:
        return None

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K,
            dist_coeffs,
            iterationsCount=RANSAC_ITERATIONS,
            reprojectionError=RANSAC_REPROJ_ERROR_PX,
            confidence=RANSAC_CONFIDENCE,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < 4:
        return None

    inlier_idx = np.array(inliers, dtype=np.int32).reshape(-1)
    min_inliers = int(np.ceil(RANSAC_MIN_INLIER_RATIO * len(object_points)))
    if len(inlier_idx) < min_inliers:
        return None

    candidate = evaluate_pose_candidate(object_points, image_points, rvec, tvec, K, dist_coeffs,
                                        "RANSAC_ITERATIVE", fit_count=len(inlier_idx))
    candidate = refine_pose_candidate(candidate, object_points, image_points, K, dist_coeffs, inlier_idx)
    if candidate is None or candidate["inlier_count"] < min_inliers:
        return None
    return candidate


def solve_target_pose(
    image_points_dict: Dict[str, Tuple[float, float]],
    object_points_dict: Dict[str, Tuple[float, float, float]],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
):
    names, object_points, image_points = build_correspondences(
        object_points_dict=object_points_dict,
        image_points_dict=image_points_dict
    )

    K = np.asarray(K, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
    validate_pnp_inputs(object_points, image_points, K, dist_coeffs)

    planar = is_planar_points(object_points)
    candidates = []

    if planar:
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
            )
        except cv2.error:
            ok, rvecs, tvecs = False, [], []
        if ok and rvecs is not None and tvecs is not None:
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                candidate = evaluate_pose_candidate(object_points, image_points, rvec, tvec,
                                                    K, dist_coeffs, "IPPE", i)
                candidate = refine_pose_candidate(candidate, object_points, image_points, K, dist_coeffs)
                if candidate is not None:
                    candidates.append(candidate)
        # IPPE can return only NaNs for an exact symmetric planar projection.
        # Recover from the original full correspondence set, never a fabricated subset.
        if not candidates:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                ok = False
            if ok:
                candidate = evaluate_pose_candidate(object_points, image_points, rvec, tvec,
                                                    K, dist_coeffs, "ITERATIVE_PLANAR")
                candidate = refine_pose_candidate(candidate, object_points, image_points, K, dist_coeffs)
                if candidate is not None:
                    candidates.append(candidate)
    else:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )
        except cv2.error:
            ok = False
        if ok:
            candidate = evaluate_pose_candidate(object_points, image_points, rvec, tvec,
                                                K, dist_coeffs, "EPNP")
            candidate = refine_pose_candidate(candidate, object_points, image_points, K, dist_coeffs)
            if candidate is not None:
                candidates.append(candidate)

    ransac_candidate = solve_pnp_ransac_candidate(
        object_points=object_points,
        image_points=image_points,
        K=K,
        dist_coeffs=dist_coeffs,
    )
    if ransac_candidate is not None:
        candidates.append(ransac_candidate)

    if not candidates:
        raise RuntimeError("no valid finite positive-depth pose candidate")
    candidates.sort(
        key=lambda c: (
            c["selection_error"],
            c["reproj_error"],
            c["max_reproj_error"],
            -c["inlier_count"],
        )
    )
    best = candidates[0]
    inlier_indices = set(int(i) for i in best["inlier_indices"])
    inlier_names = [name for i, name in enumerate(names) if i in inlier_indices]
    outlier_names = [name for i, name in enumerate(names) if i not in inlier_indices]
    per_point_errors = {name: float(err) for name, err in zip(names, best["errors"])}

    return {
        "method": best["source"],
        "matched_names": names,
        "inlier_names": inlier_names,
        "outlier_names": outlier_names,
        "object_points": object_points,
        "image_points": image_points,
        "rvec": best["rvec"],
        "tvec": best["tvec"],
        "reproj_error": best["reproj_error"],
        "selection_error": best["selection_error"],
        "per_point_errors_px": per_point_errors,
        "all_in_front": best["all_in_front"],
        "candidate_count": len(candidates),
        "inlier_count": best["inlier_count"],
        "inlier_ratio": best["inlier_ratio"],
        "inlier_threshold_px": RANSAC_REPROJ_ERROR_PX,
        "inlier_reproj_error": best["inlier_reproj_error"],
        "reproj_rms_error": best["reproj_rms_error"],
        "max_reproj_error": best["max_reproj_error"],
        "fit_count": best["fit_count"],
        "refined": best["refined"],
    }
