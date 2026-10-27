"""One immutable frame through the existing vision algorithms, without latest-file races."""
from __future__ import annotations

import contextlib
import csv
import importlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from .contracts import FramePacket, distort_normalized, empty_observation, intrinsics_signature, rectify_points, validate_camera


ROOT = Path(__file__).resolve().parents[1]


def calibrated_visual_budget(spec: dict | None, camera: dict, rms_px: float,
                             depth_m: float) -> tuple[list | None, str | None]:
    """Retrieve an independently validated SI action budget within its calibration region.

    This is not a conversion of pixel residuals into position covariance. Missing
    or out-of-domain metadata leaves F-mode unavailable, while pose diagnostics
    can still be used independently.
    """
    if not isinstance(spec, dict) or spec.get("validated") is not True:
        return None, None
    try:
        values = np.asarray(spec["budget_m_rad"], dtype=float)
        depth_range = np.asarray(spec["depth_range_m"], dtype=float)
        max_rms = float(spec["max_reprojection_rms_px"])
        matches = (spec["camera_intrinsics_sha256"] == intrinsics_signature(camera)
                   and bool(spec["version"]) and bool(spec["source"]))
        if (values.shape != (6,) or not np.isfinite(values).all() or np.any(values <= 0)
                or depth_range.shape != (2,) or not np.isfinite(depth_range).all()
                or depth_range[0] <= 0 or depth_range[0] >= depth_range[1]
                or not np.isfinite(max_rms) or max_rms <= 0
                or rms_px > max_rms or not depth_range[0] <= depth_m <= depth_range[1]
                or not matches):
            return None, None
        return values.tolist(), str(spec["version"])
    except (KeyError, TypeError, ValueError):
        return None, None


def select_detection(boxes: list[dict], image_shape: tuple, target_window: list | None) -> dict:
    """A complete connector has one identity; confidence alone cannot select between two."""
    height, width = image_shape[:2]
    if target_window is not None:
        window = np.asarray(target_window, dtype=float)
        if (window.shape != (4,) or not np.isfinite(window).all() or np.any(window < 0)
                or np.any(window > 1) or window[0] >= window[2] or window[1] >= window[3]):
            raise ValueError("target_window_requires_normalized_xyxy")
        candidates = []
        for box in boxes:
            x1, y1, x2, y2 = box["xyxy"]
            cx, cy = (x1+x2)/(2*width), (y1+y2)/(2*height)
            if window[0] <= cx <= window[2] and window[1] <= cy <= window[3]:
                candidates.append(box)
    else:
        candidates = boxes
    if not candidates:
        raise ValueError("target_not_detected")
    if len(candidates) != 1:
        raise ValueError("ambiguous_complete_target_use_target_window")
    return candidates[0]


class PosePipeline:
    def __init__(self, config: dict, detector=None):
        self.config = config
        self.target_id = config["target_id"]
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("target_id_required")
        for key in ("max_frame_age_s", "max_reprojection_rms_px", "max_point_error_px"):
            if not np.isfinite(float(config[key])) or float(config[key]) <= 0:
                raise ValueError(f"positive_{key}_required")
        if not 0 < float(config["confidence"]) <= 1 or int(config["min_inliers"]) < 4:
            raise ValueError("invalid_detector_or_inlier_threshold")
        self.last_frame = None
        self._detector = detector
        self._modules = None

    def _load(self):
        if self._modules is not None:
            return
        # The old scripts use sibling imports. These functions do not execute
        # their main() blocks and do not read intermediate image files.
        script_path = str(ROOT / "live" / "python")
        if script_path not in sys.path:
            sys.path.insert(0, script_path)
        names = ("enhance", "Canny", "latest_candidate_contours", "center_fit", "pnp2", "pose_correspondence")
        modules = {name: importlib.import_module(name) for name in names}
        detector = self._detector
        if detector is None:
            from ultralytics import YOLO
            weights = Path(self.config["weights"])
            if not weights.is_absolute():
                weights = ROOT / weights
            if not weights.is_file():
                raise FileNotFoundError(f"YOLO weights missing: {weights}")
            detector = YOLO(str(weights))
        model_path = Path(self.config["model_points"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        with model_path.open(encoding="utf-8-sig", newline="") as stream:
            headers = csv.DictReader(stream).fieldnames or []
        if not {"x_mm", "y_mm", "z_mm"}.issubset(headers):
            raise ValueError("control_model_requires_explicit_mm_columns")
        self.model = modules["pnp2"].load_object_points_from_csv(model_path)
        if not np.isfinite(np.asarray(list(self.model.values()), dtype=float)).all():
            raise ValueError("nonfinite_model_points")
        self._modules, self._detector = modules, detector

    def _extract_points(self, image: np.ndarray, *, projection: tuple) -> tuple[dict, dict]:
        import cv2
        self._load()
        result = self._detector(image, conf=float(self.config["confidence"]), verbose=False)
        boxes = []
        if result:
            for box in result[0].boxes:
                if int(box.cls.item()) == int(self.config["target_class"]):
                    boxes.append({"xyxy": [float(x) for x in box.xyxy[0].tolist()],
                                  "confidence": float(box.conf.item())})
        selected = select_detection(boxes, image.shape, self.config.get("target_window"))
        x1, y1, x2, y2 = selected["xyxy"]
        padding = max(20, int(max(x2-x1, y2-y1) * 0.08))
        height, width = image.shape[:2]
        x1, y1 = max(0, int(x1)-padding), max(0, int(y1)-padding)
        x2, y2 = min(width, int(x2)+padding), min(height, int(y2)+padding)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid_detection_bbox")
        crop = image[y1:y2, x1:x2].copy()
        enhanced = self._modules["enhance"].preprocess_roi(crop)["final"]
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        edges = self._modules["Canny"].adaptive_canny_pipeline(enhanced_bgr)[4]
        contours = self._modules["latest_candidate_contours"]
        # Existing functions are verbose. Keep their diagnostics out of control
        # timings and never persist large unbounded buffers.
        diagnostics = {}
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            candidates, _, _ = contours.filter_candidate_contours(
                contours.prepare_edges(edges), enhanced_bgr, apply_layout=False,
                diagnostics=diagnostics, gray_source=crop)
        matrix, coefficients, distortion_model = projection
        selection = self._modules["pose_correspondence"].select_correspondences(
            candidates, enhanced_bgr.shape, self.model, matrix, np.zeros(5), (x1, y1),
            point_transform=lambda points: rectify_points(points, matrix, coefficients, distortion_model),
            max_mean_error_px=float(self.config["max_reprojection_rms_px"]),
            min_inlier_count=int(self.config["min_inliers"]),
            min_correspondence_count=max(6, int(self.config["min_inliers"])))
        if selection["best"] is None:
            raise ValueError("no_layout_variant")
        return selection["best"]["image_points"], {
            "detection_score": selected["confidence"], "crop_xyxy": [x1, y1, x2, y2],
            "contour_diagnostics": diagnostics, "correspondence_diagnostics": selection["diagnostics"],
        }

    def process(self, packet: FramePacket, now: float | None = None) -> dict:
        observation = empty_observation(packet, self.target_id, "not_processed")
        try:
            current = time.monotonic() if now is None else now
            if packet.frame_id == self.last_frame:
                raise ValueError("frame_already_processed")
            self.last_frame = packet.frame_id
            if packet.image.ndim != 3 or packet.image.shape[2] != 3 or packet.image.dtype != np.uint8:
                raise ValueError("BGR_uint8_frame_required")
            if packet.source != "offline":
                if not packet.time_sync_valid:
                    raise ValueError("camera_clock_not_synchronized")
                if not np.isfinite([packet.captured_at, packet.received_at, current]).all():
                    raise ValueError("invalid_frame_timestamp")
                if packet.captured_at > packet.received_at + 0.005 or current < packet.received_at:
                    raise ValueError("invalid_frame_time_order")
                if current - packet.captured_at > float(self.config["max_frame_age_s"]):
                    raise ValueError("frame_expired_before_processing")
            matrix, coefficients, model = validate_camera(packet.camera, packet.image.shape)
            raw_points, metadata = self._extract_points(packet.image, projection=(matrix, coefficients, model))
            labels = list(raw_points)
            if len(labels) < int(self.config["min_inliers"]):
                raise ValueError("insufficient_hole_correspondences")
            raw = np.asarray([raw_points[label] for label in labels])
            rectified = rectify_points(raw, matrix, coefficients, model)
            image_points = {label: tuple(point) for label, point in zip(labels, rectified)}
            pnp = self._modules["pnp2"]
            result = pnp.solve_target_pose(image_points, self.model, matrix, np.zeros(5))
            import cv2
            rotation = cv2.Rodrigues(result["rvec"])[0]
            translation_mm = np.asarray(result["tvec"]).reshape(3)
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation_mm / 1000.0
            ordered_labels = result["matched_names"]
            camera_points = (rotation @ np.asarray([self.model[x] for x in ordered_labels]).T).T + translation_mm
            if np.any(camera_points[:, 2] <= 0) or not np.isfinite(camera_points).all():
                raise ValueError("invalid_camera_depth")
            projected = distort_normalized(camera_points[:, :2]/camera_points[:, 2, None], coefficients, model)
            projected = projected * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]
            errors = np.linalg.norm(projected - np.asarray([raw_points[x] for x in ordered_labels]), axis=1)
            if not np.isfinite(projected).all() or not np.isfinite(errors).all():
                raise ValueError("nonfinite_raw_reprojection")
            calibrated = bool(packet.camera.get("validated") and packet.camera.get("version"))
            quality = {
                "reprojection_rms_px": float(np.sqrt(np.mean(errors**2))),
                "reprojection_mean_px": float(np.mean(errors)), "max_error_px": float(np.max(errors)),
                "inlier_count": int(np.count_nonzero(errors <= pnp.RANSAC_REPROJ_ERROR_PX)),
                "rectified_inlier_count": int(result["inlier_count"]),
                "inlier_threshold_px": pnp.RANSAC_REPROJ_ERROR_PX,
                "all_in_front": bool(result["all_in_front"]),
                "camera_calibrated": calibrated, "residual_frame": "raw_color_pixels",
                "method": result["method"], "per_point_error_px": dict(zip(ordered_labels, errors.tolist())),
                "intrinsic_source": packet.camera.get("source"),
                "intrinsic_calibration_version": packet.camera.get("version"),
            }
            budget, budget_version = calibrated_visual_budget(
                self.config.get("visual_budget"), packet.camera,
                quality["reprojection_rms_px"], transform[2, 3],
            )
            quality.update({"visual_budget": budget, "budget_version": budget_version,
                            "budget_source": self.config.get("visual_budget", {}).get("source")
                            if budget is not None else None})
            observation.update({"T_camera_target": transform.tolist(), "quality": quality,
                                "image_points_full": {k: list(v) for k, v in raw_points.items()},
                                "matched_labels": ordered_labels, "metadata": metadata})
            reasons = []
            if metadata.get("correspondence_diagnostics", {}).get("ambiguous", False):
                reasons.append("ambiguous_correspondences")
            if metadata.get("correspondence_diagnostics", {}).get("selection_quality_ok") is False:
                reasons.append("correspondence_quality_rejected")
            if not calibrated:
                reasons.append("intrinsics_not_independently_validated")
            if quality["inlier_count"] < int(self.config["min_inliers"]):
                reasons.append("insufficient_inliers")
            if quality["reprojection_rms_px"] > float(self.config["max_reprojection_rms_px"]):
                reasons.append("reprojection_rms_exceeded")
            if quality["max_error_px"] > float(self.config["max_point_error_px"]):
                reasons.append("point_error_exceeded")
            finished = time.monotonic() if now is None else now
            if packet.source != "offline" and finished - packet.captured_at > float(self.config["max_frame_age_s"]):
                reasons.append("frame_expired_during_processing")
            observation["valid"] = not reasons
            observation["reason"] = ";".join(reasons) if reasons else "ok"
        except Exception as exc:
            observation["reason"] = f"{type(exc).__name__}: {exc}"
            observation["valid"] = False
            observation["T_camera_target"] = None
        observation["processed_at"] = time.monotonic() if now is None else now
        return observation
