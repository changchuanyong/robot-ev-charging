from __future__ import annotations

import os
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np


# =========================================================
# 路径配置
# =========================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
ROI_META_PATH = ROOT_DIR / "dataset" / "live" / "latest_roi_meta.json"
IMAGE_POINTS_JSON_PATH = ROOT_DIR / "dataset" / "live" / "latest_fitted_centers.json"
MODEL_CSV_PATH = ROOT_DIR / "config" / "charging_port_model.csv"
CAMERA_CONFIG_PATH = ROOT_DIR / "config" / "camera_intrinsics.json"
FULL_IMAGE_PATH = ROOT_DIR / "dataset" / "live" / "latest.jpg"

POSE_JSON_PATH = ROOT_DIR / "dataset" / "live" / "latest_pose.json"
POSE_VIS_PATH = ROOT_DIR / "dataset" / "live" / "latest_pose_vis.jpg"

SHOW_WINDOW = os.environ.get("VISION_PIPELINE_MODE", "0") != "1"
POINTS_ARE_IN_ROI = True
AXIS_LEN_MM = 20.0
RANSAC_REPROJ_ERROR_PX = 10.0
RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS = 500
WARN_REPROJ_ERROR_PX = 5.0
OUTLIER_SELECTION_PENALTY_PX = 1.5
RANSAC_MIN_INLIER_RATIO = 0.70

# =========================================================
# 你的 layout_name -> 模型点 label 映射
# 如果后面发现左右极性定义反了，只改这里
# =========================================================
LAYOUT_TO_MODEL_LABEL = {
    "top_left": "S_neg",
    "top_mid": "CC2",
    "top_right": "S_pos",
    "center": "CC1",
    "main_left": "DC_neg",
    "main_right": "DC_pos",
    "bottom_left": "A_neg",
    "bottom_mid": "PE",
    "bottom_right": "A_pos",
}


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


def atomic_write_json(path: Path, data: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + "_tmp" + path.suffix)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if path.exists():
            path.unlink()
        tmp_path.replace(path)
        return True
    except OSError:
        return False


def load_roi_offset(meta_path: Path) -> Tuple[float, float, dict]:
    meta = load_json(meta_path)

    if "roi_origin_in_full_image" in meta:
        ox = float(meta["roi_origin_in_full_image"]["x"])
        oy = float(meta["roi_origin_in_full_image"]["y"])
        return ox, oy, meta

    if "crop_bbox_xyxy" in meta:
        ox = float(meta["crop_bbox_xyxy"]["x1"])
        oy = float(meta["crop_bbox_xyxy"]["y1"])
        return ox, oy, meta

    raise KeyError("latest_roi_meta.json 中缺少 roi_origin_in_full_image 或 crop_bbox_xyxy")


def parse_image_points(json_path: Path) -> Dict[str, Tuple[float, float]]:
    """
    兼容多种格式，并特别支持你当前 latest_fitted_centers.json 的格式：

    当前文件格式：
    [
      {
        "layout_name": "top_left",
        "fitted_center": [406.408, 242.301]
      },
      ...
    ]

    解析后会自动映射成：
    {
      "S_neg": (406.408, 242.301),
      ...
    }
    """
    data = load_json(json_path)
    points: Dict[str, Tuple[float, float]] = {}

    # -------------------------------------------------
    # 格式 A：list[dict]
    # 每项里用 layout_name + fitted_center
    # -------------------------------------------------
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            raw_name = None
            if "layout_name" in item:
                raw_name = str(item["layout_name"])
            elif "name" in item:
                raw_name = str(item["name"])
            elif "id" in item:
                raw_name = str(item["id"])
            elif "label" in item:
                raw_name = str(item["label"])

            if raw_name is None:
                continue

            pt = None
            if "fitted_center" in item and isinstance(item["fitted_center"], (list, tuple)) and len(item["fitted_center"]) >= 2:
                pt = (float(item["fitted_center"][0]), float(item["fitted_center"][1]))
            elif "raw_center" in item and isinstance(item["raw_center"], (list, tuple)) and len(item["raw_center"]) >= 2:
                pt = (float(item["raw_center"][0]), float(item["raw_center"][1]))
            elif "x" in item and "y" in item:
                pt = (float(item["x"]), float(item["y"]))
            elif "u" in item and "v" in item:
                pt = (float(item["u"]), float(item["v"]))
            elif "cx" in item and "cy" in item:
                pt = (float(item["cx"]), float(item["cy"]))
            elif "center_x" in item and "center_y" in item:
                pt = (float(item["center_x"]), float(item["center_y"]))

            if pt is None:
                continue

            mapped_name = LAYOUT_TO_MODEL_LABEL.get(raw_name, raw_name)
            points[mapped_name] = pt

    # -------------------------------------------------
    # 格式 B：dict
    # -------------------------------------------------
    elif isinstance(data, dict):
        if "points" in data and isinstance(data["points"], dict):
            data = data["points"]

        for name, val in data.items():
            if isinstance(val, dict):
                if "x" in val and "y" in val:
                    points[str(name)] = (float(val["x"]), float(val["y"]))
                    continue
                if "u" in val and "v" in val:
                    points[str(name)] = (float(val["u"]), float(val["v"]))
                    continue
                if "cx" in val and "cy" in val:
                    points[str(name)] = (float(val["cx"]), float(val["cy"]))
                    continue
                if "center_x" in val and "center_y" in val:
                    points[str(name)] = (float(val["center_x"]), float(val["center_y"]))
                    continue
                if "center" in val and isinstance(val["center"], (list, tuple)) and len(val["center"]) >= 2:
                    points[str(name)] = (float(val["center"][0]), float(val["center"][1]))
                    continue
                if "fitted_center" in val and isinstance(val["fitted_center"], (list, tuple)) and len(val["fitted_center"]) >= 2:
                    points[str(name)] = (float(val["fitted_center"][0]), float(val["fitted_center"][1]))
                    continue

            if isinstance(val, (list, tuple)) and len(val) >= 2:
                points[str(name)] = (float(val[0]), float(val[1]))
                continue

    if len(points) == 0:
        raise ValueError(
            "无法解析 latest_fitted_centers.json。\n"
            "当前已支持：layout_name + fitted_center 这种列表格式。"
        )

    return points


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


def roi_points_to_full_points(
    points_in_roi: Dict[str, Tuple[float, float]],
    offset_x: float,
    offset_y: float
) -> Dict[str, Tuple[float, float]]:
    points_in_full = {}
    for name, (u, v) in points_in_roi.items():
        points_in_full[name] = (float(u) + offset_x, float(v) + offset_y)
    return points_in_full


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


def compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist_coeffs)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - image_points, axis=1)
    return float(np.mean(err))


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
    return bool(np.all(cam_pts[:, 2] > 0))


def solve_pnp_ransac_candidate(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict | None:
    if len(object_points) < 6:
        return None

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
    if not ok or inliers is None or len(inliers) < 4:
        return None

    inlier_idx = np.array(inliers, dtype=np.int32).reshape(-1)
    min_inliers = int(np.ceil(RANSAC_MIN_INLIER_RATIO * len(object_points)))
    if len(inlier_idx) < min_inliers:
        return None

    inlier_object = object_points[inlier_idx]
    inlier_image = image_points[inlier_idx]

    refined_ok, refined_rvec, refined_tvec = cv2.solvePnP(
        inlier_object,
        inlier_image,
        K,
        dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if refined_ok:
        rvec, tvec = refined_rvec, refined_tvec

    return {
        "idx": 0,
        "source": "RANSAC_ITERATIVE",
        "rvec": rvec,
        "tvec": tvec,
        "reproj_error": compute_reprojection_error(
            object_points, image_points, rvec, tvec, K, dist_coeffs
        ),
        "selection_error": (
            compute_reprojection_error(
                inlier_object, inlier_image, rvec, tvec, K, dist_coeffs
            )
            + OUTLIER_SELECTION_PENALTY_PX * (len(object_points) - len(inlier_idx))
        ),
        "all_in_front": all_points_in_front(object_points, rvec, tvec),
        "inlier_indices": inlier_idx.tolist(),
        "inlier_count": int(len(inlier_idx)),
    }


def solve_target_pose_legacy(
    image_points_dict: Dict[str, Tuple[float, float]],
    object_points_dict: Dict[str, Tuple[float, float, float]],
    K: np.ndarray,
    dist_coeffs: np.ndarray,
):
    names, object_points, image_points = build_correspondences(
        object_points_dict=object_points_dict,
        image_points_dict=image_points_dict
    )

    planar = is_planar_points(object_points)

    if planar:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )

        if not ok or rvecs is None or len(rvecs) == 0:
            raise RuntimeError("IPPE 求解失败")

        candidates = []
        for i in range(len(rvecs)):
            rvec = rvecs[i]
            tvec = tvecs[i]

            err = compute_reprojection_error(
                object_points, image_points, rvec, tvec, K, dist_coeffs
            )
            front = all_points_in_front(object_points, rvec, tvec)

            candidates.append({
                "idx": i,
                "rvec": rvec,
                "tvec": tvec,
                "reproj_error": err,
                "all_in_front": front,
            })

        candidates.sort(key=lambda c: (0 if c["all_in_front"] else 1, c["reproj_error"]))
        best = candidates[0]

        return {
            "method": "IPPE",
            "matched_names": names,
            "object_points": object_points,
            "image_points": image_points,
            "rvec": best["rvec"],
            "tvec": best["tvec"],
            "reproj_error": best["reproj_error"],
            "all_in_front": best["all_in_front"],
            "candidate_count": len(candidates),
        }

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not ok:
        raise RuntimeError("EPNP 求解失败")

    err = compute_reprojection_error(
        object_points, image_points, rvec, tvec, K, dist_coeffs
    )

    return {
        "method": "EPNP",
        "matched_names": names,
        "object_points": object_points,
        "image_points": image_points,
        "rvec": rvec,
        "tvec": tvec,
        "reproj_error": err,
        "all_in_front": all_points_in_front(object_points, rvec, tvec),
        "candidate_count": 1,
    }


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

    planar = is_planar_points(object_points)
    candidates = []

    if planar:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )

        if not ok or rvecs is None or len(rvecs) == 0:
            raise RuntimeError("IPPE solve failed")

        for i in range(len(rvecs)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            candidates.append({
                "idx": i,
                "source": "IPPE",
                "rvec": rvec,
                "tvec": tvec,
                "reproj_error": compute_reprojection_error(
                    object_points, image_points, rvec, tvec, K, dist_coeffs
                ),
                "selection_error": compute_reprojection_error(
                    object_points, image_points, rvec, tvec, K, dist_coeffs
                ),
                "all_in_front": all_points_in_front(object_points, rvec, tvec),
                "inlier_indices": list(range(len(names))),
                "inlier_count": len(names),
            })
    else:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok:
            raise RuntimeError("EPNP solve failed")

        candidates.append({
            "idx": 0,
            "source": "EPNP",
            "rvec": rvec,
            "tvec": tvec,
            "reproj_error": compute_reprojection_error(
                object_points, image_points, rvec, tvec, K, dist_coeffs
            ),
            "selection_error": compute_reprojection_error(
                object_points, image_points, rvec, tvec, K, dist_coeffs
            ),
            "all_in_front": all_points_in_front(object_points, rvec, tvec),
            "inlier_indices": list(range(len(names))),
            "inlier_count": len(names),
        })

    ransac_candidate = solve_pnp_ransac_candidate(
        object_points=object_points,
        image_points=image_points,
        K=K,
        dist_coeffs=dist_coeffs,
    )
    if ransac_candidate is not None:
        candidates.append(ransac_candidate)

    candidates.sort(
        key=lambda c: (
            0 if c["all_in_front"] else 1,
            c["selection_error"],
            -c["inlier_count"],
        )
    )
    best = candidates[0]
    inlier_indices = set(int(i) for i in best["inlier_indices"])
    inlier_names = [name for i, name in enumerate(names) if i in inlier_indices]
    outlier_names = [name for i, name in enumerate(names) if i not in inlier_indices]
    per_point_errors = build_per_point_errors(
        names, object_points, image_points, best["rvec"], best["tvec"], K, dist_coeffs
    )

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
    }


def draw_pose_result(
    full_image_path: Path,
    image_points_dict: Dict[str, Tuple[float, float]],
    matched_names: List[str],
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    save_path: Path,
    axis_len_mm: float = 20.0,
    show_window: bool = True,
):
    if not full_image_path.exists():
        print(f"[警告] 原图不存在，跳过可视化: {full_image_path}")
        return

    img = cv2.imread(str(full_image_path))
    if img is None or img.size == 0:
        print("[警告] 原图读取失败，跳过可视化")
        return

    for name in matched_names:
        u, v = image_points_dict[name]
        x = int(round(u))
        y = int(round(v))
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            img,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    axis_3d = np.array([
        [0.0, 0.0, 0.0],
        [axis_len_mm, 0.0, 0.0],
        [0.0, axis_len_mm, 0.0],
        [0.0, 0.0, -axis_len_mm],
    ], dtype=np.float64)

    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist_coeffs)
    axis_2d = axis_2d.reshape(-1, 2).astype(int)

    o = tuple(axis_2d[0])
    x_axis = tuple(axis_2d[1])
    y_axis = tuple(axis_2d[2])
    z_axis = tuple(axis_2d[3])

    cv2.line(img, o, x_axis, (0, 0, 255), 3)
    cv2.line(img, o, y_axis, (0, 255, 0), 3)
    cv2.line(img, o, z_axis, (255, 0, 0), 3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), img)
    if ok:
        print(f"位姿可视化已保存: {save_path}")

    if show_window:
        cv2.namedWindow("pose_vis", cv2.WINDOW_NORMAL)
        cv2.imshow("pose_vis", img)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    roi_offset_x, roi_offset_y, _ = load_roi_offset(ROI_META_PATH)
    image_points_raw = parse_image_points(IMAGE_POINTS_JSON_PATH)
    object_points_dict = load_object_points_from_csv(MODEL_CSV_PATH)
    camera_matrix, dist_coeffs, camera_is_placeholder = load_camera_intrinsics(CAMERA_CONFIG_PATH)

    print("===== Loaded Data =====")
    print(f"ROI offset: ({roi_offset_x:.3f}, {roi_offset_y:.3f})")
    print(f"Image points raw ({len(image_points_raw)}): {list(image_points_raw.keys())}")
    print(f"Model points ({len(object_points_dict)}): {list(object_points_dict.keys())}")
    print(f"Camera config: {CAMERA_CONFIG_PATH}")
    if camera_is_placeholder:
        print("[WARN] camera_intrinsics.json still uses placeholder calibration values.")

    if POINTS_ARE_IN_ROI:
        image_points_full = roi_points_to_full_points(
            image_points_raw,
            offset_x=roi_offset_x,
            offset_y=roi_offset_y
        )
    else:
        image_points_full = image_points_raw

    print("Image points full:")
    for k, v in image_points_full.items():
        print(f"  {k}: ({v[0]:.3f}, {v[1]:.3f})")

    result = solve_target_pose(
        image_points_dict=image_points_full,
        object_points_dict=object_points_dict,
        K=camera_matrix,
        dist_coeffs=dist_coeffs
    )

    rvec = result["rvec"]
    tvec = result["tvec"]
    R, _ = cv2.Rodrigues(rvec)

    print("===== Pose Solve Done =====")
    print(f"method           : {result['method']}")
    print(f"matched_names    : {result['matched_names']}")
    print(f"inlier_names     : {result['inlier_names']}")
    print(f"outlier_names    : {result['outlier_names']}")
    print(f"candidate_count  : {result['candidate_count']}")
    print(f"inlier_count     : {result['inlier_count']}")
    print(f"all_in_front     : {result['all_in_front']}")
    print(f"selection_error  : {result['selection_error']:.6f}")
    print(f"reproj_error(px) : {result['reproj_error']:.6f}")
    if result["reproj_error"] > WARN_REPROJ_ERROR_PX:
        print(f"[WARN] reprojection error is above {WARN_REPROJ_ERROR_PX:.1f}px.")
    print("per_point_errors(px):")
    for name, err in result["per_point_errors_px"].items():
        print(f"  {name}: {err:.3f}")
    print("rvec:")
    print(rvec.reshape(-1))
    print("tvec:")
    print(tvec.reshape(-1))
    print("R:")
    print(R)

    pose_json = {
        "method": result["method"],
        "matched_names": result["matched_names"],
        "inlier_names": result["inlier_names"],
        "outlier_names": result["outlier_names"],
        "inlier_count": int(result["inlier_count"]),
        "selection_error_px": float(result["selection_error"]),
        "reproj_error_px": float(result["reproj_error"]),
        "per_point_errors_px": {
            name: float(err)
            for name, err in result["per_point_errors_px"].items()
        },
        "all_in_front": bool(result["all_in_front"]),
        "roi_offset": {
            "x": float(roi_offset_x),
            "y": float(roi_offset_y)
        },
        "camera_config": str(CAMERA_CONFIG_PATH),
        "camera_intrinsics_placeholder": bool(camera_is_placeholder),
        "camera_matrix": [[float(v) for v in row] for row in camera_matrix.tolist()],
        "dist_coeffs": [float(x) for x in dist_coeffs.reshape(-1)],
        "rvec": [float(x) for x in rvec.reshape(-1)],
        "tvec": [float(x) for x in tvec.reshape(-1)],
        "R": [[float(v) for v in row] for row in R.tolist()],
        "image_points_full": {
            name: {
                "x": float(image_points_full[name][0]),
                "y": float(image_points_full[name][1]),
            }
            for name in result["matched_names"]
        },
        "object_points_mm": {
            name: {
                "x": float(object_points_dict[name][0]),
                "y": float(object_points_dict[name][1]),
                "z": float(object_points_dict[name][2]),
            }
            for name in result["matched_names"]
        }
    }

    ok = atomic_write_json(POSE_JSON_PATH, pose_json)
    if ok:
        print(f"位姿结果已保存: {POSE_JSON_PATH}")

    draw_pose_result(
        full_image_path=FULL_IMAGE_PATH,
        image_points_dict=image_points_full,
        matched_names=result["matched_names"],
        rvec=rvec,
        tvec=tvec,
        K=camera_matrix,
        dist_coeffs=dist_coeffs,
        save_path=POSE_VIS_PATH,
        axis_len_mm=AXIS_LEN_MM,
        show_window=SHOW_WINDOW,
    )


if __name__ == "__main__":
    main()
