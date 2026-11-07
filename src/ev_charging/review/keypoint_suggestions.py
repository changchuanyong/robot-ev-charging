"""Unconfirmed image-space hole suggestions; never synthesize missing centers."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ev_charging.review.keypoint_annotations import LABELS
from ev_charging.vision.algorithms.center_fit import fit_ellipse_on_kept_candidates
from ev_charging.vision.algorithms.observed_layout import _review_topology
from ev_charging.vision.model_config import resolve_model_config
from ev_charging.vision.pipeline import padded_crop, preprocess_candidates


def promote_dark_apertures(candidates, crop, contours):
    """Verify already fitted Canny centers against raw-image dark-hole evidence."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    promoted = []
    for item in candidates:
        if item.contour_source != "canny" or item.aperture_center is not None or not item.fit_valid:
            continue
        contrast = contours.dark_boundary_contrast(gray, item)
        if contrast >= contours.DARK_CONTRAST_OFFSET:
            item.aperture_center = item.fitted_center
            item.aperture_diameter_px = math.sqrt(np.prod(item.ellipse_axes))
            item.review_only_aperture = True
            promoted.append(
                {
                    "center": list(item.aperture_center),
                    "contrast": contrast,
                    "contour_index": item.contour_index,
                }
            )
    return promoted


def stable_dark_apertures(crop, contours):
    """Find measured dark boundaries that persist across several intensity levels."""
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    low, high = np.percentile(gray, [2, 40])
    levels = np.unique(np.linspace(low, high, 32).astype(np.uint8))
    groups = []
    for level in levels:
        _, mask = cv2.threshold(gray, int(level), 255, cv2.THRESH_BINARY_INV)
        boundaries, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for boundary in boundaries:
            area = cv2.contourArea(boundary)
            if not contours.MIN_CONTOUR_AREA <= area <= crop.shape[0] * crop.shape[1] * 0.22:
                continue
            if contours.is_near_border(cv2.boundingRect(boundary), crop.shape, 0.0):
                continue
            candidate = contours.compute_candidate_features(boundary, crop.shape)
            if candidate is None:
                continue
            fit_ellipse_on_kept_candidates([candidate], verbose=False)
            if not candidate.fit_valid:
                continue
            contrast = contours.dark_boundary_contrast(gray, candidate)
            if contrast < contours.DARK_CONTRAST_OFFSET:
                continue
            diameter = math.sqrt(np.prod(candidate.ellipse_axes))
            group = next(
                (
                    items
                    for items in groups
                    if np.linalg.norm(np.asarray(candidate.fitted_center) - items[0][1].fitted_center)
                    <= max(3.0, diameter * 0.08)
                    and 0.7 <= diameter / items[0][2] <= 1.4
                ),
                None,
            )
            item = (int(level), candidate, diameter, contrast)
            if group is None:
                groups.append([item])
            else:
                group.append(item)
    stable = []
    for group in groups:
        if len({item[0] for item in group}) < 3:
            continue
        median = np.median([item[1].fitted_center for item in group], axis=0)
        _, candidate, diameter, contrast = min(
            group, key=lambda item: np.linalg.norm(np.asarray(item[1].fitted_center) - median)
        )
        # Select an observed contour's center, never the projected or averaged center.
        candidate.aperture_center = candidate.fitted_center
        candidate.aperture_diameter_px = diameter
        candidate.contour_source = "threshold_dark"
        candidate.inside_prior, candidate.prior_value = contours.inside_structure_prior(
            *candidate.fitted_center, crop.shape
        )
        stable.append((len(group), contrast, candidate))
    return [
        item[2]
        for item in sorted(stable, key=lambda item: (item[0], item[1], item[2].score), reverse=True)[:20]
    ]


def complete_apertures(selection, candidates, offset):
    """Complete a supported review layout using unique measured inner apertures."""
    points = dict(selection["image_points"])
    diagnostics = selection["diagnostics"]
    if (
        not {"DC_neg", "DC_pos", "PE"} <= points.keys()
        or diagnostics.get("ambiguous")
        or not diagnostics.get("structural_support_ok")
    ):
        return points, []
    span = float(np.linalg.norm(np.asarray(points["DC_pos"]) - points["DC_neg"]))
    pe = next(
        (candidate for candidate in selection["candidates"] if candidate.layout_name == "bottom_mid"), None
    )
    if pe is None or not pe.aperture_diameter_px or span <= 0:
        return points, []
    added = []
    for label, predicted in selection.get("association_points", {}).items():
        if label in points or label in ("DC_neg", "DC_pos", "PE"):
            continue
        eligible = []
        for candidate in candidates:
            if (
                not candidate.fit_valid
                or candidate.aperture_center is None
                or candidate.review_only_aperture
                or not candidate.inside_prior
            ):
                continue
            if not 0.04 * span <= candidate.aperture_diameter_px <= pe.aperture_diameter_px / 1.2:
                continue
            xy = np.asarray(candidate.aperture_center) + offset
            if any(np.linalg.norm(xy - np.asarray(point)) < 8 for point in points.values()):
                continue
            distance = float(np.linalg.norm(xy - np.asarray(predicted)))
            if distance <= 0.08 * span:
                eligible.append((distance, candidate, xy))
        if not eligible:
            continue
        eligible.sort(key=lambda item: item[0])
        distance, candidate, xy = eligible[0]
        if any(np.linalg.norm(other - xy) > 4 for _, _, other in eligible[1:]):
            continue
        proposal = {**points, label: tuple(xy)}
        kept, _ = _review_topology(
            {LABELS.index(name): index for index, name in enumerate(proposal)},
            np.asarray(list(proposal.values())),
        )
        if LABELS.index(label) in kept:
            points[label] = tuple(xy)
            added.append(
                {
                    "label": label,
                    "xy": xy.tolist(),
                    "source": candidate.contour_source,
                    "distance_to_association_px": distance,
                    "diameter_px": candidate.aperture_diameter_px,
                }
            )
    return points, added


def valid_points(points, width, height):
    result = {}
    if isinstance(points, dict):
        for name, point in points.items():
            if (
                name in LABELS
                and isinstance(point, (list, tuple))
                and len(point) == 2
                and all(type(value) in (int, float) and np.isfinite(value) for value in point)
                and 0 <= point[0] < width
                and 0 <= point[1] < height
            ):
                result[name] = list(point)
    return result


class SuggestionEngine:
    def __init__(self, root, library):
        self.root, self.library = Path(root).resolve(), Path(library).resolve()
        self.detector = None
        self.detector_key = None
        self.modules = None

    def current_model(self):
        return resolve_model_config(
            {"model_registry": "config/vision_model.json"}, root=self.root, verify=True
        )

    def _load_detector(self, model):
        key = (model["weights"], model["weights_sha256"])
        if key != self.detector_key:
            from ev_charging.runtime import configure_caches

            configure_caches(self.root)
            from ultralytics import YOLO

            detector = YOLO(model["weights"])
            self.detector, self.detector_key = detector, key
        return self.detector

    def suggest(self, entry, image, *, model=None, diagnostics=False):
        model = model or self.current_model()
        provenance = {
            "source": "observed_2d_layout",
            "weights_sha256": model["weights_sha256"],
            "weights": model["weights"],
            "model_name": Path(model["weights"]).parent.parent.name + "/" + Path(model["weights"]).name,
        }
        if entry.get("status") == "excluded":
            return {**provenance, "points": {}, "reason": "该图已排除，未自动生成孔位；仍可手工标注。"}
        predictions = self._load_detector(model)(image, conf=0.25, verbose=False)
        boxes = [
            {"xyxy": item.xyxy[0].tolist(), "confidence": float(item.conf.item())}
            for prediction in predictions
            for item in prediction.boxes
            if int(item.cls.item()) == 0
        ]
        trace = {"detected_boxes": boxes}
        if len(boxes) != 1:
            return {
                **provenance,
                "points": {},
                "reason": "当前权重未检测到唯一充电口，请手工标注。",
                **({"diagnostics": trace} if diagnostics else {}),
            }
        box = boxes[0]["xyxy"]
        if self.modules is None:
            from ev_charging.vision.algorithms import latest_candidate_contours, observed_layout

            self.modules = [latest_candidate_contours, observed_layout]
            self.prior = self.modules[1].load_prior(self.root / "config/observed_layout_prior.json")
        height, width = image.shape[:2]
        crop, (x1, y1, x2, y2) = padded_crop(image, box)
        contours, layout = self.modules
        contour_trace = {} if diagnostics else None
        enhanced, candidates, _, _ = preprocess_candidates(crop, diagnostics=contour_trace)
        promoted = promote_dark_apertures(candidates, crop, contours)
        selected = layout.select_observed_correspondences(
            candidates, enhanced.shape, self.prior, (x1, y1), review_mode=True, prefitted=True
        )
        extras = []
        if (
            {"DC_neg", "DC_pos", "PE"} <= selected["image_points"].keys()
            and not selected["diagnostics"].get("ambiguous")
            and selected["diagnostics"].get("structural_support_ok")
        ):
            extras = stable_dark_apertures(crop, contours)
        completed, added = complete_apertures(selected, candidates + extras, np.asarray([x1, y1]))
        if diagnostics:
            by_index = {candidate.contour_index: candidate for candidate in candidates}
            for outcome in contour_trace["candidate_trace"]["outcomes"]:
                if outcome["stage"] == "kept":
                    candidate = by_index[outcome["contour_index"]]
                    outcome["aperture_center"] = (
                        list(candidate.aperture_center) if candidate.aperture_center is not None else None
                    )
                    outcome["aperture_diameter_px"] = candidate.aperture_diameter_px
            trace.update(
                roi_bbox_xyxy=[x1, y1, x2, y2],
                contours=contour_trace,
                promoted_apertures=promoted,
                layout=selected["diagnostics"],
                association_points={
                    name: np.asarray(point).tolist()
                    for name, point in selected.get("association_points", {}).items()
                },
                completed_apertures=added,
                stable_dark_candidate_count=len(extras),
            )
        points = {name: np.asarray(point).tolist() for name, point in completed.items()}
        return {
            **provenance,
            "points": valid_points(points, width, height),
            "quality_ok": selected["diagnostics"]["selection_quality_ok"],
            "reason": f"已找到 {len(points)} 个有图像证据的候选，请逐点复核。"
            if points
            else "主孔尺度、孔径比例或朝向证据不足，请手工标注。",
            **({"diagnostics": trace} if diagnostics else {}),
        }
