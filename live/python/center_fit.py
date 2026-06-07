from __future__ import annotations

import math

import cv2

from candidate_types import CandidateContour


FIT_MIN_POINTS = 20
FIT_MIN_POINTS_HARD = 5
FIT_MAX_ASPECT_RATIO = 5.0
FIT_MIN_AXIS = 3.0
FIT_CENTER_IN_BBOX_MARGIN = 0.35
FIT_MAX_CENTER_SHIFT_RATIO = 0.22


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
            print(
                f"fit=fallback_error    "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})"
            )
            fitted_results.append(cand)
            continue

        (cx, cy), (ma, mi), angle = ellipse

        if ma < FIT_MIN_AXIS or mi < FIT_MIN_AXIS:
            print(
                f"fit=fallback_axis     "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"axes=({ma:.1f},{mi:.1f})"
            )
            fitted_results.append(cand)
            continue

        ellipse_aspect_ratio = max(ma, mi) / (min(ma, mi) + 1e-6)
        if ellipse_aspect_ratio > FIT_MAX_ASPECT_RATIO:
            print(
                f"fit=fallback_aspect   "
                f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"e_aspect={ellipse_aspect_ratio:.2f}"
            )
            fitted_results.append(cand)
            continue

        if not ellipse_center_valid((cx, cy), cand.bbox, FIT_CENTER_IN_BBOX_MARGIN):
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
            print(
                f"fit=fallback_shift    "
                f"raw=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                f"fit=({cx:6.1f},{cy:6.1f})  "
                f"shift={center_shift:5.1f}/{max_shift:4.1f}"
            )
            fitted_results.append(cand)
            continue

        cand.fitted_center = (float(cx), float(cy))
        cand.fit_method = "ellipse"
        cand.ellipse_axes = (float(ma), float(mi))
        cand.ellipse_angle = float(angle)

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
            "bbox": [int(c.bbox[0]), int(c.bbox[1]), int(c.bbox[2]), int(c.bbox[3])],
            "area": round(c.area, 3),
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
