"""Depth-only evidence for an RGB diagnostic; never a robot motion permission.

Points and their raw RGB pixels come from SDK projection and a depth-to-color
rigid transform. An accepted fit is only a candidate plane within the detected
box: a box cannot identify which printed step is the intended aperture plane.
"""
from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from .contracts import intrinsics_signature


# Reproducible diagnostic heuristics, not calibrated insertion/safety limits.
THRESHOLDS = {
    "roi_inset_fraction": .1, "hole_radius_fraction": .025, "hole_radius_min_px": 6.,
    "edge_absolute_m": .005, "edge_relative": .01, "min_surface_samples": 80,
    "max_fit_samples": 4000, "ransac_iterations": 80, "plane_tolerance_m": .003,
    "min_inlier_fraction": .85, "max_plane_rms_m": .002,
    "min_second_axis_std_m": .003, "min_rgb_span_fraction": .35,
}


def _positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _finite_json(value):
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    return value


def _interior_depth_mask(raw, scale):
    """Exclude zeros, depth jumps, and their immediate native-depth neighbors."""
    z = raw.astype(float) * scale
    valid = raw > 0
    safe = valid.copy()
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        tolerance = np.maximum(THRESHOLDS["edge_absolute_m"],
                               THRESHOLDS["edge_relative"] * np.minimum(z[left], z[right]))
        jump = (~valid[left] | ~valid[right] | (np.abs(z[left] - z[right]) > tolerance))
        safe[left] &= ~jump
        safe[right] &= ~jump
    return cv2.erode(safe.astype(np.uint8), np.ones((3, 3), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def _fit_plane(points, pixels, box):
    count = len(points)
    if count < THRESHOLDS["min_surface_samples"]:
        return {"solved": False, "reason": "insufficient_surface_samples"}
    if count > THRESHOLDS["max_fit_samples"]:
        indices = np.linspace(0, count - 1, THRESHOLDS["max_fit_samples"], dtype=int)
        points, pixels = points[indices], pixels[indices]
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    tolerance = THRESHOLDS["plane_tolerance_m"]
    for _ in range(THRESHOLDS["ransac_iterations"]):
        triple = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(triple[1] - triple[0], triple[2] - triple[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        residuals = np.abs((points - triple[0]) @ normal)
        inliers = residuals <= tolerance
        if inliers.sum() > best.sum():
            best = inliers
    fraction = float(best.mean())
    if fraction < THRESHOLDS["min_inlier_fraction"]:
        return {"solved": False, "reason": "mixed_or_nonplanar_surface", "inlier_fraction": fraction}
    centroid = points[best].mean(axis=0)
    _, singular, vectors = np.linalg.svd(points[best] - centroid, full_matrices=False)
    normal = vectors[-1]
    if normal[2] > 0:
        normal = -normal
    residuals = np.abs((points - centroid) @ normal)
    inliers = residuals <= tolerance
    fraction = float(inliers.mean())
    rms = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if np.any(inliers) else math.inf
    if fraction < THRESHOLDS["min_inlier_fraction"] or rms > THRESHOLDS["max_plane_rms_m"]:
        return {"solved": False, "reason": "noisy_or_mixed_surface", "inlier_fraction": fraction}
    spans = np.ptp(pixels[inliers], axis=0) / (np.asarray(box[2:]) - box[:2])
    if (singular[1] / math.sqrt(int(best.sum())) < THRESHOLDS["min_second_axis_std_m"]
            or np.any(spans < THRESHOLDS["min_rgb_span_fraction"])):
        return {"solved": False, "reason": "insufficient_plane_spatial_support"}
    return {
        "solved": True, "kind": "detected_roi_candidate_plane",
        "aperture_reference_verified": False, "normal_color_camera": normal.tolist(),
        "centroid_color_mm": (centroid * 1000).tolist(),
        "offset_mm": float(-normal @ centroid * 1000), "rms_mm": rms * 1000,
        "inlier_fraction": fraction, "sample_count": len(points),
        "inlier_count": int(inliers.sum()), "surface_samples_before_downsampling": count,
        "normal_convention": "color_optical_frame_z_nonpositive",
    }


def _compare_pose(plane, pose, geometry):
    if not plane["solved"] or not pose.get("solved"):
        return None
    try:
        model = np.asarray(list(geometry["points_mm"].values()), dtype=float)
        rotation_vector = np.asarray(pose["rvec"], dtype=float).reshape(3)
        translation = np.asarray(pose["tvec_mm"], dtype=float).reshape(3)
        if model.ndim != 2 or model.shape[1] != 3 or len(model) < 3:
            return None
        if not all(np.isfinite(value).all() for value in (model, rotation_vector, translation)):
            return None
        center = model.mean(axis=0)
        _, singular, vectors = np.linalg.svd(model - center, full_matrices=False)
        if singular[1] < 1e-6 or np.max(np.abs((model - center) @ vectors[-1])) > .001:
            return None  # A stepped model does not define one aperture plane.
        rotation = cv2.Rodrigues(rotation_vector)[0]
        normal = np.asarray(plane["normal_color_camera"])
        angle = math.degrees(math.acos(float(np.clip(abs(normal @ (rotation @ vectors[-1])), 0, 1))))
        distance = abs(float(normal @ (rotation @ center + translation) + plane["offset_mm"]))
        return {"normal_angle_deg": angle, "plane_distance_mm": distance,
                "reference_verified": False, "geometry_status": geometry.get("status"),
                "note": "candidate_ROI_surface_may_differ_from_modeled_aperture_plane"}
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, cv2.error):
        return None


def analyze_depth(packet, color_frame, color_camera, frontend, pose, geometry):
    """Return finite JSON evidence; rejected depth never changes the RGB pose.

    Scene validity uses native depth pixels. ROI coverage counts unique RGB
    pixels receiving actual projected samples; it deliberately does not fill
    gaps and is resolution dependent. It is not an aligned-depth validity rate.
    """
    packet = packet or {}
    calibration, frame = packet.get("calibration") or {}, packet.get("frame") or {}
    result = {
        "schema_version": 1, "kind": "depth_assisted_diagnostic", "method_version": "roi_plane_v1",
        "authorizes_robot_motion": False, "status": "invalid", "reasons": list(packet.get("reasons", [])),
        "frame": packet.get("frame"), "color_frame": dict(color_frame), "calibration": calibration,
        "raw_pixels_sha256": None, "pair_skew_ms": None, "scene": {}, "roi": None,
        "plane": {"solved": False, "reason": "invalid_depth"}, "pnp_consistency": None,
        "thresholds": dict(THRESHOLDS),
    }
    reasons = result["reasons"]
    raw = packet.get("raw_z16")
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint16 or raw.ndim != 2 or raw.size == 0:
        reasons.append("missing_or_invalid_raw_depth")
        return _finite_json(result)
    result["raw_pixels_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
    if packet.get("color_frame") is not None and packet["color_frame"] != color_frame:
        reasons.append("depth_color_frame_binding_mismatch")
    color_time, depth_time = color_frame.get("device_timestamp_ms"), frame.get("device_timestamp_ms")
    if not _positive(color_time):
        reasons.append("invalid_color_timestamp")
    if not _positive(depth_time):
        reasons.append("invalid_depth_timestamp")
    domains = {"hardware_clock", "system_time", "global_time"}
    same_domain = (color_frame.get("timestamp_domain") in domains
                   and frame.get("timestamp_domain") == color_frame.get("timestamp_domain"))
    if not same_domain:
        reasons.append("depth_timestamp_domain_mismatch")
    max_skew = calibration.get("max_pair_skew_ms")
    if not _positive(max_skew):
        reasons.append("invalid_depth_pair_skew_limit")
    elif same_domain and _positive(color_time) and _positive(depth_time):
        result["pair_skew_ms"] = abs(float(color_time - depth_time))
        if result["pair_skew_ms"] > max_skew:
            reasons.append("depth_pair_skew_exceeded")
    scale = calibration.get("depth_scale_m")
    profile = calibration.get("depth_profile", {})
    if not _positive(scale):
        reasons.append("invalid_depth_scale")
    if (profile.get("height"), profile.get("width"), profile.get("format")) != (*raw.shape, "z16"):
        reasons.append("depth_profile_mismatch")
    try:
        if calibration.get("color_intrinsics_sha256") != intrinsics_signature(color_camera):
            reasons.append("depth_color_intrinsics_mismatch")
    except (KeyError, TypeError, ValueError):
        reasons.append("invalid_color_intrinsics")
    count = int(np.count_nonzero(raw))
    result["scene"] = {"valid_fraction": count / raw.size, "valid_pixels": count,
                       "total_pixels": raw.size, "median_depth_mm":
                       float(np.median(raw[raw > 0]) * scale * 1000) if count and _positive(scale) else None,
                       "depth_frame": "depth_optical", "validity_definition": "nonzero_native_Z16_pixels"}
    if count == 0:
        reasons.append("no_valid_depth")
    if reasons:
        result["reasons"] = list(dict.fromkeys(reasons))
        return _finite_json(result)
    points, pixels = packet.get("points_color_m"), packet.get("color_pixels")
    if (not isinstance(points, np.ndarray) or points.shape != (*raw.shape, 3)
            or not isinstance(pixels, np.ndarray) or pixels.shape != (*raw.shape, 2)):
        reasons.append("missing_or_invalid_depth_projection")
        return _finite_json(result)
    width, height = color_camera["image_width"], color_camera["image_height"]
    valid = ((raw > 0) & np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
             & np.isfinite(pixels).all(axis=2) & (pixels[:, :, 0] >= 0) & (pixels[:, :, 0] < width)
             & (pixels[:, :, 1] >= 0) & (pixels[:, :, 1] < height))
    result["scene"]["projected_valid_samples"] = int(valid.sum())
    try:
        box = np.asarray(frontend.get("selected_bbox_xyxy", []), dtype=float)
    except (TypeError, ValueError):
        box = np.array([])
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        result.update(status="scene_only", reasons=["target_not_detected"],
                      plane={"solved": False, "reason": "no_detected_target_ROI"})
        return _finite_json(result)
    box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    inset = (box[2:] - box[:2]) * THRESHOLDS["roi_inset_fraction"]
    inner = np.r_[box[:2] + inset, box[2:] - inset]
    region = (valid & (pixels[:, :, 0] >= inner[0]) & (pixels[:, :, 0] < inner[2])
              & (pixels[:, :, 1] >= inner[1]) & (pixels[:, :, 1] < inner[3]))
    surface = region & _interior_depth_mask(raw, scale)
    radius = max(THRESHOLDS["hole_radius_min_px"], float(min(box[2:] - box[:2]))
                 * THRESHOLDS["hole_radius_fraction"])
    for value in frontend.get("named_points_raw_px", {}).values():
        hole = np.asarray(value, dtype=float)
        if hole.shape == (2,) and np.isfinite(hole).all():
            surface &= np.sum((pixels - hole) ** 2, axis=2) > radius ** 2
    projected = np.floor(pixels[region]).astype(np.int64)
    area = max(0, math.ceil(inner[2]) - math.ceil(inner[0])) * max(0, math.ceil(inner[3]) - math.ceil(inner[1]))
    coverage = len(np.unique(projected, axis=0)) / area if area else 0.
    result["roi"] = {
        "bbox_xyxy": box.tolist(), "inset_bbox_xyxy": inner.tolist(),
        "projected_valid_samples": int(region.sum()), "surface_samples": int(surface.sum()),
        "measurement_coverage_fraction": min(1., coverage), "hole_exclusion_radius_px": radius,
        "median_color_z_mm": float(np.median(points[region, 2]) * 1000) if np.any(region) else None,
        "coverage_definition": "unique_RGB_pixels_with_projected_measurements_divided_by_inset_ROI_pixels",
    }
    result["plane"] = _fit_plane(points[surface], pixels[surface], inner)
    if result["plane"]["solved"]:
        result["status"] = "candidate_plane"
        result["reasons"] = ["aperture_plane_identity_unverified"]
        result["pnp_consistency"] = _compare_pose(result["plane"], pose, geometry)
    else:
        result["status"] = "roi_insufficient" if int(surface.sum()) < THRESHOLDS["min_surface_samples"] else "plane_rejected"
        result["reasons"] = [result["plane"]["reason"]]
    return _finite_json(result)
