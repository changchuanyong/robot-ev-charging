"""Unconfirmed image-space hole suggestions; never synthesize missing centers."""
from __future__ import annotations

import contextlib
import importlib
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from live.python.keypoint_annotations import LABELS
from ..model_config import resolve_model_config
from .review_apertures import promote_dark_apertures


def valid_points(points, width, height):
    result = {}
    if isinstance(points, dict):
        for name, point in points.items():
            if (name in LABELS and isinstance(point, (list, tuple)) and len(point) == 2
                    and all(type(value) in (int, float) and np.isfinite(value) for value in point)
                    and 0 <= point[0] < width and 0 <= point[1] < height):
                result[name] = list(point)
    return result


class SuggestionEngine:
    def __init__(self, root, library):
        self.root, self.library = Path(root).resolve(), Path(library).resolve()
        self.detector = None
        self.detector_key = None
        self.modules = None

    def current_model(self):
        return resolve_model_config({"model_registry": "config/vision_model.json"}, root=self.root, verify=True)

    def _load_detector(self, model):
        key = (model["weights"], model["weights_sha256"])
        if key != self.detector_key:
            from train.python.runtime_env import configure_caches
            configure_caches(self.root)
            from ultralytics import YOLO
            detector = YOLO(model["weights"])
            self.detector, self.detector_key = detector, key
        return self.detector

    def suggest(self, entry, image, *, model=None, diagnostics=False):
        model = model or self.current_model()
        provenance = {"source": "observed_2d_layout", "weights_sha256": model["weights_sha256"],
                      "weights": model["weights"],
                      "model_name": Path(model["weights"]).parent.parent.name + "/" + Path(model["weights"]).name}
        if entry.get("status") == "excluded":
            return {**provenance, "points": {}, "reason": "该图已排除，未自动生成孔位；仍可手工标注。"}
        predictions = self._load_detector(model)(image, conf=.25, verbose=False)
        boxes = [{"xyxy": item.xyxy[0].tolist(), "confidence": float(item.conf.item())}
                 for prediction in predictions for item in prediction.boxes if int(item.cls.item()) == 0]
        trace = {"detected_boxes": boxes}
        if len(boxes) != 1:
            return {**provenance, "points": {}, "reason": "当前权重未检测到唯一充电口，请手工标注。",
                    **({"diagnostics": trace} if diagnostics else {})}
        box = boxes[0]["xyxy"]
        if self.modules is None:
            path = str(self.root / "live/python")
            if path not in sys.path:
                sys.path.insert(0, path)
            self.modules = [importlib.import_module(name) for name in
                            ("enhance", "latest_candidate_contours", "observed_layout")]
            self.prior = self.modules[2].load_prior(self.root / "config/observed_layout_prior.json")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        padding = max(20, int(max(x2-x1, y2-y1)*.08))
        x1, y1 = max(0, int(x1)-padding), max(0, int(y1)-padding)
        x2, y2 = min(width, int(x2)+padding), min(height, int(y2)+padding)
        crop = image[y1:y2, x1:x2].copy()
        enhance, contours, layout = self.modules
        enhanced = cv2.cvtColor(enhance.preprocess_roi(crop)["final"], cv2.COLOR_GRAY2BGR)
        edges = enhance.adaptive_canny_pipeline(enhanced)[4]
        contour_trace = {} if diagnostics else None
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            candidates, _, _ = contours.filter_candidate_contours(
                contours.prepare_edges(edges), enhanced, apply_layout=False, gray_source=crop,
                diagnostics=contour_trace)
            promoted = promote_dark_apertures(candidates, crop, contours)
            selected = layout.select_observed_correspondences(candidates, enhanced.shape, self.prior, (x1, y1), review_mode=True)
        if diagnostics:
            by_index = {candidate.contour_index: candidate for candidate in candidates}
            for outcome in contour_trace["candidate_trace"]["outcomes"]:
                if outcome["stage"] == "kept":
                    candidate = by_index[outcome["contour_index"]]
                    outcome["aperture_center"] = list(candidate.aperture_center) if candidate.aperture_center is not None else None
                    outcome["aperture_diameter_px"] = candidate.aperture_diameter_px
            trace.update(roi_bbox_xyxy=[x1, y1, x2, y2], contours=contour_trace,
                         promoted_apertures=promoted,
                         layout=selected["diagnostics"],
                         association_points={name: np.asarray(point).tolist()
                                             for name, point in selected.get("association_points", {}).items()})
        points = {name: np.asarray(point).tolist() for name, point in selected["image_points"].items()}
        return {**provenance, "points": valid_points(points, width, height),
                "quality_ok": selected["diagnostics"]["selection_quality_ok"],
                "reason": "主孔尺度、孔径比例或朝向证据不足，未输出不确定孔名；请手工补点。",
                **({"diagnostics": trace} if diagnostics else {})}
