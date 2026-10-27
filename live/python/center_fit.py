from __future__ import annotations

import math

import cv2
import numpy as np

from candidate_types import CandidateContour


FIT_MIN_POINTS = 20
FIT_MIN_POINTS_HARD = 5
FIT_MAX_ASPECT_RATIO = 5.0
FIT_MIN_AXIS = 3.0
FIT_CENTER_IN_BBOX_MARGIN = 0.35
FIT_MAX_CENTER_SHIFT_RATIO = 0.22
FIT_MIN_ANGULAR_COVERAGE = 0.65
FIT_MAX_RESIDUAL_AXIS_RATIO = 0.05
FIT_RESIDUAL_PIXEL_FLOOR = 1.5


def ellipse_fit_quality(contour: np.ndarray, ellipse: tuple) -> tuple[float, float]:
    """Radial RMS in pixels and observed angular coverage of an ellipse boundary."""
    (cx, cy), (axis_x, axis_y), angle = ellipse
    theta = math.radians(angle)
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    local = (np.asarray(contour, dtype=float).reshape(-1, 2) - [cx, cy]) @ rotation
    normalized = local / [axis_x / 2.0, axis_y / 2.0]
    radius = np.linalg.norm(normalized, axis=1)
    if np.any(radius <= 1e-9) or not np.isfinite(radius).all():
        return float("inf"), 0.0
    radial_distance = np.linalg.norm(local * (1.0 - 1.0 / radius[:, None]), axis=1)
    phases = np.sort(np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2.0 * math.pi))
    gaps = np.diff(np.r_[phases, phases[0] + 2.0 * math.pi])
    return float(np.sqrt(np.mean(radial_distance ** 2))), float(1.0 - gaps.max() / (2.0 * math.pi))


def ellipse_center_valid(
    center: tuple[float, float],
    bbox: tuple[int, int, int, int],
    margin_ratio: float,
) -> bool:
    cx, cy = center
    x, y, w, h = bbox

    mx = w * margin_ratio
    my = h * margin_ratio

    return (x - mx <= cx <= x + w + mx) and (y - my <= cy <= y + h + my)


def fit_ellipse_on_kept_candidates(
    kept_candidates: list[CandidateContour],
) -> list[CandidateContour]:
    print("\n===== Stage 5: fitEllipse on kept candidates =====")

    fitted_results: list[CandidateContour] = []

    for cand in kept_candidates:
        cnt = cand.contour
        n_points = len(cnt)

        cand.fitted_center = cand.center
        cand.fit_method = "moments"
        cand.ellipse_axes = None
        cand.ellipse_angle = None
        cand.ellipse_area = None
        cand.fit_valid = False
        cand.fit_residual_px = None
        cand.fit_coverage = 0.0
        cand.fit_reject_reason = "insufficient_points"

        if n_points < FIT_MIN_POINTS_HARD:
            print(
                f"fit=fallback_points   "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"points={n_points}"
            )
            fitted_results.append(cand)
            continue

        if n_points < FIT_MIN_POINTS:
            print(
                f"fit=fallback_sparse   "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"points={n_points}"
            )
            fitted_results.append(cand)
            continue

        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            cand.fit_reject_reason = "fit_error"
            print(
                f"fit=fallback_error    "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})"
            )
            fitted_results.append(cand)
            continue

        (cx, cy), (ma, mi), angle = ellipse

        if not np.isfinite([cx, cy, ma, mi, angle]).all():
            cand.fit_reject_reason = "nonfinite_fit"
            fitted_results.append(cand)
            continue

        if ma < FIT_MIN_AXIS or mi < FIT_MIN_AXIS:
            cand.fit_reject_reason = "small_axis"
            print(
                f"fit=fallback_axis     "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"axes=({ma:.1f},{mi:.1f})"
            )
            fitted_results.append(cand)
            continue

        ellipse_aspect_ratio = max(ma, mi) / (min(ma, mi) + 1e-6)
        if ellipse_aspect_ratio > FIT_MAX_ASPECT_RATIO:
            cand.fit_reject_reason = "extreme_aspect"
            print(
                f"fit=fallback_aspect   "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"e_aspect={ellipse_aspect_ratio:.2f}"
            )
            fitted_results.append(cand)
            continue

        if not ellipse_center_valid((cx, cy), cand.bbox, FIT_CENTER_IN_BBOX_MARGIN):
            cand.fit_reject_reason = "center_outside_bbox"
            print(
                f"fit=fallback_bbox     "
                f"raw=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"fit=({cx:6.1f},{cy:6.1f})"
            )
            fitted_results.append(cand)
            continue

        center_shift = math.hypot(cx - cand.center[0], cy - cand.center[1])
        _, _, bbox_w, bbox_h = cand.bbox
        max_shift = FIT_MAX_CENTER_SHIFT_RATIO * max(bbox_w, bbox_h)
        if center_shift > max_shift:
            cand.fit_reject_reason = "center_shift"
            print(
                f"fit=fallback_shift    "
                f"raw=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"fit=({cx:6.1f},{cy:6.1f})  "
                f"shift={center_shift:5.1f}/{max_shift:4.1f}"
            )
            fitted_results.append(cand)
            continue

        residual, coverage = ellipse_fit_quality(cnt, ellipse)
        cand.fit_residual_px = residual if math.isfinite(residual) else None
        cand.fit_coverage = coverage
        if coverage < FIT_MIN_ANGULAR_COVERAGE:
            cand.fit_reject_reason = "partial_boundary"
            fitted_results.append(cand)
            continue
        if residual > max(FIT_RESIDUAL_PIXEL_FLOOR, FIT_MAX_RESIDUAL_AXIS_RATIO * min(ma, mi)):
            cand.fit_reject_reason = "boundary_residual"
            fitted_results.append(cand)
            continue

        cand.fitted_center = (float(cx), float(cy))
        cand.fit_method = "ellipse"
        cand.ellipse_axes = (float(ma), float(mi))
        cand.ellipse_angle = float(angle)
        cand.ellipse_area = float(math.pi * ma * mi / 4.0)
        cand.fit_valid = True
        cand.fit_reject_reason = ""

        print(
            f"fit=ellipse           "
            f"raw=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
            f"fit=({cx:6.1f},{cy:6.1f})  "
            f"axes=({ma:5.1f},{mi:5.1f})  "
            f"angle={angle:6.1f}"
        )

        fitted_results.append(cand)

    return fitted_results


def export_fitted_centers(fitted_candidates: list[CandidateContour]) -> list[dict]:
    out = []
    for i, c in enumerate(fitted_candidates):
        item = {
            "index": i,
            "raw_center": [round(c.center[0], 3), round(c.center[1], 3)],
            "fitted_center": None if c.fitted_center is None else [round(c.fitted_center[0], 3), round(c.fitted_center[1], 3)],
            "fit_method": c.fit_method,
            "fit_valid": c.fit_valid,
            "fit_residual_px": None if c.fit_residual_px is None else round(c.fit_residual_px, 6),
            "fit_coverage": round(c.fit_coverage, 6),
            "fit_reject_reason": c.fit_reject_reason,
            "contour_source": c.contour_source,
            "hole_group_id": c.hole_group_id,
            "aperture_diameter_px": c.aperture_diameter_px,
            "aperture_center": None if c.aperture_center is None else list(c.aperture_center),
            "bbox": [int(c.bbox[0]), int(c.bbox[1]), int(c.bbox[2]), int(c.bbox[3])],
            "area": round(c.area, 3),
            "ellipse_area": None if c.ellipse_area is None else round(c.ellipse_area, 3),
            "score": round(c.score, 6),
            "layout_name": c.layout_name,
            "layout_group": c.layout_group,
            "layout_dist": round(c.layout_dist, 6),
            "layout_accept_dist": round(c.layout_accept_dist, 6),
            "is_main_anchor": c.is_main_anchor,
            "is_top_backup": c.is_top_backup,
            "ellipse_axes": None if c.ellipse_axes is None else [round(c.ellipse_axes[0], 3), round(c.ellipse_axes[1], 3)],
            "ellipse_angle": None if c.ellipse_angle is None else round(c.ellipse_angle, 3),
        }
        out.append(item)
    return out
