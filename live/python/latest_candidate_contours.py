from __future__ import annotations

import math

import cv2
import numpy as np

from center_fit import CandidateContour, fit_ellipse_on_kept_candidates
from layout_prior import apply_standard_layout_prior


MIN_CONTOUR_AREA = 60
MAX_CONTOUR_AREA_RATIO = 0.22

MIN_CIRCULARITY = 0.15
MAX_ASPECT_RATIO = 4.00
MIN_FILL_RATIO = 0.05

BORDER_MARGIN_RATIO = 0.03
PRIOR_RX_RATIO = 0.40
PRIOR_RY_RATIO = 0.33
PRIOR_VALUE_MAX = 0.90

DUP_CENTER_DIST = 4
MAX_KEEP = 40
DARK_WINDOW_RATIO = 0.30
DARK_CONTRAST_OFFSET = 3.0


PRINT_REJECT_LOG = True


def prepare_edges(edges: np.ndarray) -> np.ndarray:
    if len(edges.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    return binary


def contour_center(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if abs(m["m00"]) > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    x, y, w, h = cv2.boundingRect(cnt)
    return x + w / 2.0, y + h / 2.0


def is_near_border(bbox: tuple[int, int, int, int], roi_shape: tuple[int, int], margin_ratio: float) -> bool:
    h_img, w_img = roi_shape[:2]
    x, y, w, h = bbox

    mx = int(w_img * margin_ratio)
    my = int(h_img * margin_ratio)

    return (x <= mx) or (y <= my) or (x + w >= w_img - mx) or (y + h >= h_img - my)


def inside_structure_prior(cx: float, cy: float, roi_shape: tuple[int, int]) -> tuple[bool, float]:
    h_img, w_img = roi_shape[:2]
    roi_cx = w_img / 2.0
    roi_cy = h_img / 2.0

    rx = PRIOR_RX_RATIO * w_img
    ry = PRIOR_RY_RATIO * h_img

    dx = cx - roi_cx
    dy = cy - roi_cy

    value = (dx * dx) / (rx * rx + 1e-6) + (dy * dy) / (ry * ry + 1e-6)
    return value <= 1.0, value


def area_preference(area_norm: float) -> float:
    if area_norm < 0.0015:
        return 0.05
    if area_norm < 0.004:
        return 0.50
    if area_norm < 0.035:
        return 1.00
    return 0.50


def compute_candidate_features(cnt: np.ndarray, roi_shape: tuple[int, int],
                               allow_open: bool = False) -> CandidateContour | None:
    area = float(cv2.contourArea(cnt))
    if area <= 1 and not allow_open:
        return None

    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter <= 1:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 1 or h <= 1:
        return None

    cx, cy = contour_center(cnt)

    circularity = 4.0 * math.pi * area / (perimeter * perimeter + 1e-6)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    fill_ratio = area / (w * h + 1e-6)

    inside, prior_value = inside_structure_prior(cx, cy, roi_shape)

    roi_h, roi_w = roi_shape[:2]
    roi_area = roi_h * roi_w
    area_norm = area / (roi_area + 1e-6)

    area_score = area_preference(area_norm)

    center_score = max(0.0, min(1.0, 1.0 - prior_value))
    circularity_score = max(0.0, min(1.0, circularity))
    aspect_score = max(0.0, min(1.0, 1.0 / aspect_ratio))
    fill_score = max(0.0, min(1.0, fill_ratio))

    score = (
        0.25 * circularity_score +
        0.10 * aspect_score +
        0.10 * fill_score +
        0.30 * center_score +
        0.25 * area_score
    )

    return CandidateContour(
        contour=cnt,
        center=(cx, cy),
        bbox=(x, y, w, h),
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        fill_ratio=fill_ratio,
        prior_value=prior_value,
        inside_prior=inside,
        score=score,
    )


def reject_log(reason: str, feat: CandidateContour) -> None:
    if not PRINT_REJECT_LOG:
        return

    print(
        f"reject={reason:>16s}  "
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"area={feat.area:7.1f}  "
        f"circ={feat.circularity:5.3f}  "
        f"aspect={feat.aspect_ratio:4.2f}  "
        f"fill={feat.fill_ratio:4.2f}  "
        f"prior={feat.prior_value:4.2f}  "
        f"score={feat.score:5.3f}"
    )


def keep_log(tag: str, feat: CandidateContour) -> None:
    print(
        f"{tag:<20s}"
        f"center=({feat.center[0]:6.1f},{feat.center[1]:6.1f})  "
        f"area={feat.area:7.1f}  "
        f"circ={feat.circularity:5.3f}  "
        f"aspect={feat.aspect_ratio:4.2f}  "
        f"fill={feat.fill_ratio:4.2f}  "
        f"prior={feat.prior_value:4.2f}  "
        f"score={feat.score:5.3f}"
    )


def ellipse_radius(points: np.ndarray, candidate: CandidateContour) -> np.ndarray:
    """Normalized radius in the measured ellipse, without changing image points."""
    angle = math.radians(candidate.ellipse_angle)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    local = (np.asarray(points) - candidate.fitted_center) @ rotation
    return np.linalg.norm(local / (np.asarray(candidate.ellipse_axes) / 2), axis=-1)


def same_hole_boundaries(first: CandidateContour, second: CandidateContour) -> bool:
    """Concentric or shifted nested rings belong to one measured aperture.

    Both centres must be supported by the inner ellipse. The area-ratio and
    containment checks prevent the connector's broad outer face swallowing its
    individual holes, and do not merge nearby non-nested apertures.
    """
    if not (first.fit_valid and second.fit_valid):
        return False
    inner, outer = sorted((first, second), key=lambda c: c.ellipse_area)
    if inner.ellipse_area / outer.ellipse_area < 0.12:
        return False
    if ellipse_radius(np.asarray(outer.fitted_center), inner) > 1.0:
        return False
    theta = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    boundary = np.column_stack([np.cos(theta), np.sin(theta)]) * (np.asarray(inner.ellipse_axes) / 2)
    angle = math.radians(inner.ellipse_angle)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    boundary = boundary @ rotation.T + inner.fitted_center
    return bool(np.mean(ellipse_radius(boundary, outer) <= 1.05) >= 0.95)


def dark_boundary_contrast(gray: np.ndarray, candidate: CandidateContour) -> float:
    """Require a dark interior with a brighter observed surrounding band."""
    cx, cy = candidate.fitted_center
    radius = 0.75 * max(candidate.ellipse_axes)
    x0, x1 = max(0, int(cx - radius)), min(gray.shape[1], int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(gray.shape[0], int(cy + radius) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    radii = ellipse_radius(np.stack([xx, yy], axis=-1), candidate)
    values = gray[y0:y1, x0:x1]
    inside, band = values[radii < 0.65], values[(radii > 1.1) & (radii < 1.4)]
    if min(inside.size, band.size) < 5:
        return float("-inf")
    return float(np.median(band)) - float(np.median(inside))


def filter_candidate_contours(
    edges: np.ndarray,
    roi_bgr: np.ndarray,
    apply_layout: bool = True,
    diagnostics: dict | None = None,
    gray_source: np.ndarray | None = None,
) -> tuple[list[CandidateContour], list[np.ndarray], LayoutModel | None]:
    h_img, w_img = roi_bgr.shape[:2]
    roi_area = h_img * w_img
    max_area = MAX_CONTOUR_AREA_RATIO * roi_area

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    sources = ["canny"] * len(contours)
    hierarchy = hierarchy.reshape(-1, 4) if hierarchy is not None else np.empty((0, 4), dtype=np.int32)
    gray = None
    if gray_source is not None:
        if gray_source.dtype != np.uint8 or gray_source.shape[:2] != (h_img, w_img):
            raise ValueError("gray_source_must_match_ROI_uint8")
        if gray_source.ndim == 3 and gray_source.shape[2] == 3:
            gray = cv2.cvtColor(gray_source, cv2.COLOR_BGR2GRAY)
        elif gray_source.ndim == 2:
            gray = gray_source
        else:
            raise ValueError("gray_source_requires_gray_or_BGR")
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        block = max(31, int(min(gray.shape) * DARK_WINDOW_RATIO)) | 1
        dark_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, block, DARK_CONTRAST_OFFSET)
        extra, extra_hierarchy = cv2.findContours(dark_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if extra:
            shifted = extra_hierarchy.reshape(-1, 4).copy()
            shifted[shifted >= 0] += len(contours)
            hierarchy = np.vstack([hierarchy, shifted])
            contours.extend(extra)
            sources.extend(["adaptive_dark"] * len(extra))
    if diagnostics is not None:
        diagnostics.update({"contour_count": len(contours), "rejection_counts": {},
                            "rejected_candidates": [],
                            "candidate_trace": {"coordinate_frame": "roi_pixels", "max_keep": MAX_KEEP,
                                                "outcomes": []},
                            "contour_sources": {source: sources.count(source) for source in set(sources)}})

    def snapshot(feat: CandidateContour) -> dict:
        def point(value):
            return None if value is None else [float(x) for x in value]

        center = next(value for value in (feat.aperture_center, feat.fitted_center, feat.center)
                      if value is not None)
        return {"contour_index": int(feat.contour_index), "contour_parent": int(feat.contour_parent),
                "source": feat.contour_source, "center": point(feat.center),
                "fitted_center": point(feat.fitted_center), "aperture_center": point(feat.aperture_center),
                "measurement_center": point(center), "bbox": [int(x) for x in feat.bbox],
                "fit_valid": bool(feat.fit_valid), "fit_reason": feat.fit_reject_reason,
                "fit_residual_px": feat.fit_residual_px, "ellipse_axes": point(feat.ellipse_axes),
                "ellipse_angle": feat.ellipse_angle, "ellipse_area": feat.ellipse_area,
                "aperture_diameter_px": feat.aperture_diameter_px, "area": float(feat.area),
                "score": float(feat.score), "hole_group_id": int(feat.hole_group_id)}

    def reject(reason: str, feat: CandidateContour, *, representative=None,
               aperture_transferred=False) -> None:
        reject_log(reason, feat)
        if diagnostics is not None:
            counts = diagnostics["rejection_counts"]
            counts[reason] = counts.get(reason, 0) + 1
            outcome = {**snapshot(feat), "stage": "filtered", "reason": reason}
            if representative is not None:
                outcome.update(stage="merged", representative_contour_index=int(representative.contour_index),
                               aperture_transferred=bool(aperture_transferred),
                               representative_aperture_center_after_merge=(
                                   None if representative.aperture_center is None
                                   else [float(x) for x in representative.aperture_center]),
                               representative_aperture_diameter_after_merge=representative.aperture_diameter_px)
            diagnostics["candidate_trace"]["outcomes"].append(outcome)
            # Keep meaningful rejected boundaries for overlays, without storing contours.
            if max(feat.area, feat.ellipse_area or 0.0) >= MIN_CONTOUR_AREA:
                diagnostics["rejected_candidates"].append({
                    "reason": reason, "center": list(feat.center), "bbox": list(feat.bbox),
                    "area": round(feat.area, 3), "ellipse_area": feat.ellipse_area,
                    "fit_reason": feat.fit_reject_reason,
                    "source": feat.contour_source,
                })

    raw_candidates: list[CandidateContour] = []

    print("===== Stage 1&2: geometry + coarse prior filtering =====")
    for contour_index, cnt in enumerate(contours):
        feat = compute_candidate_features(cnt, roi_bgr.shape, allow_open=not apply_layout)
        if feat is None:
            if diagnostics is not None:
                diagnostics["candidate_trace"]["outcomes"].append({
                    "contour_index": contour_index, "contour_parent": int(hierarchy[contour_index, 3]),
                    "source": sources[contour_index], "center": list(contour_center(cnt)),
                    "bbox": list(cv2.boundingRect(cnt)), "stage": "feature_rejected",
                    "reason": "degenerate_contour", "fit_valid": False, "fit_reason": "not_fitted",
                    "fitted_center": None, "aperture_center": None, "measurement_center": None,
                })
            continue
        feat.contour_index = contour_index
        feat.contour_parent = int(hierarchy[contour_index, 3])
        feat.contour_source = sources[contour_index]

        if not apply_layout or feat.contour_source == "adaptive_dark":
            # A gap in a Canny ring produces a contour that retraces the arc and
            # encloses almost no area. Validate the observed boundary before
            # using enclosed area or circularity as a proxy for hole geometry.
            fit_ellipse_on_kept_candidates([feat])
        if feat.fit_valid and not apply_layout:
            old_center_score = max(0.0, min(1.0, 1.0 - feat.prior_value))
            feat.inside_prior, feat.prior_value = inside_structure_prior(*feat.fitted_center, roi_bgr.shape)
            feat.score += 0.30 * (max(0.0, min(1.0, 1.0 - feat.prior_value)) - old_center_score)
        if feat.contour_source == "adaptive_dark":
            if not feat.fit_valid:
                reject("dark_invalid_fit", feat)
                continue
            if dark_boundary_contrast(gray, feat) < DARK_CONTRAST_OFFSET:
                reject("dark_polarity", feat)
                continue
            feat.aperture_diameter_px = math.sqrt(feat.ellipse_axes[0] * feat.ellipse_axes[1])
            feat.aperture_center = feat.fitted_center
        measured_area = feat.ellipse_area if feat.fit_valid and not apply_layout else feat.area
        if measured_area != feat.area:
            feat.score += 0.25 * (area_preference(measured_area / roi_area)
                                  - area_preference(feat.area / roi_area))

        if measured_area < MIN_CONTOUR_AREA:
            reject("area_small", feat)
            continue

        if measured_area > max_area:
            reject("area_large", feat)
            continue

        # Raw extraction keeps off-centre holes for geometric matching. Only an
        # actually clipped boundary is unusable before the connector pose is known.
        border_margin = BORDER_MARGIN_RATIO if apply_layout else 0.0
        if is_near_border(feat.bbox, roi_bgr.shape, border_margin):
            reject("near_border", feat)
            continue

        if feat.circularity < MIN_CIRCULARITY and not (feat.fit_valid and not apply_layout):
            reject("circularity", feat)
            continue

        if feat.aspect_ratio > MAX_ASPECT_RATIO:
            reject("aspect", feat)
            continue

        if feat.fill_ratio < MIN_FILL_RATIO and not (feat.fit_valid and not apply_layout):
            reject("fill_ratio", feat)
            continue

        if apply_layout and not feat.inside_prior:
            reject("outside_prior", feat)
            continue

        if apply_layout and feat.prior_value > PRIOR_VALUE_MAX:
            reject("prior_too_far", feat)
            continue

        raw_candidates.append(feat)
        keep_log("keep_raw", feat)

    if apply_layout:
        fit_ellipse_on_kept_candidates(raw_candidates)
    raw_candidates.sort(key=lambda c: (c.fit_valid, c.score,
                                      -(c.fit_residual_px if c.fit_residual_px is not None else float("inf"))),
                        reverse=True)
    if diagnostics is not None:
        diagnostics["raw_candidate_count"] = len(raw_candidates)
        diagnostics["valid_fit_count"] = sum(c.fit_valid for c in raw_candidates)
        diagnostics["fit_rejection_counts"] = {}
        for cand in raw_candidates:
            if not cand.fit_valid:
                counts = diagnostics["fit_rejection_counts"]
                counts[cand.fit_reject_reason] = counts.get(cand.fit_reject_reason, 0) + 1

    def duplicate(first: CandidateContour, second: CandidateContour) -> bool:
        c1 = first.fitted_center if first.fit_valid else first.center
        c2 = second.fitted_center if second.fit_valid else second.center
        distance = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        if distance < DUP_CENTER_DIST:
            return True
        return same_hole_boundaries(first, second)

    print("\n===== Stage 3: duplicate removal =====")
    kept: list[CandidateContour] = []
    for cand in raw_candidates:
        duplicated = False
        for old in kept:
            if duplicate(cand, old):
                aperture_transferred = False
                if cand.aperture_diameter_px is not None and (
                        old.aperture_diameter_px is None or cand.aperture_diameter_px < old.aperture_diameter_px):
                    old.aperture_diameter_px = cand.aperture_diameter_px
                    old.aperture_center = cand.aperture_center
                    aperture_transferred = True
                reject("duplicate", cand, representative=old, aperture_transferred=aperture_transferred)
                print(
                    f"reject=duplicate       "
                    f"center=({cand.center[0]:6.1f},{cand.center[1]:6.1f})  "
                    f"score={cand.score:5.3f}  "
                    f"close_to=({old.center[0]:6.1f},{old.center[1]:6.1f})"
                )
                duplicated = True
                break

        if not duplicated:
            cand.hole_group_id = len(kept)
            kept.append(cand)
            keep_log("keep_final_dup", cand)

    if diagnostics is not None:
        diagnostics["uncapped_candidate_count"] = len(kept)
        for rank, cand in enumerate(kept):
            diagnostics["candidate_trace"]["outcomes"].append({
                **snapshot(cand), "stage": "kept" if rank < MAX_KEEP else "capped",
                "reason": None if rank < MAX_KEEP else "max_keep", "rank_before_cap": rank + 1,
            })
    kept = kept[:MAX_KEEP]
    if diagnostics is not None:
        diagnostics["kept_candidate_count"] = len(kept)
        diagnostics["kept_sources"] = {source: sum(c.contour_source == source for c in kept)
                                       for source in set(sources)}
    if not apply_layout:
        return kept, contours, None

    kept_layout, layout_model = apply_standard_layout_prior(kept, roi_bgr.shape)
    return kept_layout, contours, layout_model
