"""Save a completed diagnostic frame without mixing it with newer camera pixels."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .contracts import atomic_capture_json as atomic_json
from .capture_session import CaptureSession
from .contracts import intrinsics_signature, validate_camera
from .model_config import resolve_model_config


ROOT = Path(__file__).resolve().parents[1]


def _finite(value):
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _copy(value):
    return json.loads(json.dumps(_finite(value), ensure_ascii=False, allow_nan=False))


def _json_bytes(value):
    return (json.dumps(_finite(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _write_new(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _asset(path):
    if not path:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return {"path": str(path.resolve()), "sha256": _hash(path.read_bytes()) if path.is_file() else None}


class PreviewSession:
    def __init__(self, output_root, camera, batch_id, *, pipeline_config, geometry, metadata=None,
                 depth_calibration=None, split="unknown"):
        self._camera = _copy(camera)
        self._signature = intrinsics_signature(self._camera)
        self._geometry = _copy(geometry)
        self._depth_calibration = None
        if depth_calibration is not None:
            # Calibration cannot silently turn nonfinite values into null.
            self._depth_calibration = json.loads(json.dumps(depth_calibration, allow_nan=False))
            self._validate_depth_calibration()
        pipeline = _copy(resolve_model_config(pipeline_config))
        assets = {key: _asset(pipeline.get(key)) for key in
                  ("weights", "observed_layout_prior", "model_points", "geometry_reference")}
        code = {name: _asset(name) for name in
                ("vision/preview.py", "vision/preview_render.py", "vision/preview_session.py",
                 "vision/preview_capture.py", "vision/rgb_controls.py", "vision/collect.py",
                 "vision/diagnose.py", "vision/model_config.py", "vision/capture_session.py", "vision/contracts.py",
                 "live/python/pose_correspondence.py", "live/python/pnp2.py")}
        if self._depth_calibration is not None:
            code.update({name: _asset(name) for name in
                         ("vision/rgbd_capture.py", "vision/depth_diagnostic.py")})
        self._capture = CaptureSession(output_root, camera, batch_id, split,
                                       "Live diagnostic samples; automatic points are not human labels.",
                                       metadata=metadata)
        self.path = self._capture.path
        self._frames = {}
        self._pending_finish = None
        self._manifest = {
            "schema_version": 2 if self._depth_calibration is not None else 1,
            "kind": "realsense_preview_diagnostics",
            "source": "live_diagnostic", "authorizes_robot_motion": False,
            "capture_manifest": "manifest.json", "camera_file": "camera.json",
            "camera_sha256": _hash((self.path / "camera.json").read_bytes()),
            "intrinsics_sha256": self._signature, "status": "recording", "samples": [],
            "pipeline_config": pipeline, "pipeline_config_sha256": _hash(_json_bytes(pipeline)),
            "geometry": self._geometry, "assets": assets, "code": code,
        }
        try:
            (self.path / "diagnostics").mkdir()
            (self.path / "overlays").mkdir()
            if self._depth_calibration is not None:
                (self.path / "depth").mkdir()
                calibration_bytes = _json_bytes(self._depth_calibration)
                _write_new(self.path / "depth_camera.json", calibration_bytes)
                self._manifest.update({"depth_camera_file": "depth_camera.json",
                                       "depth_camera_sha256": _hash(calibration_bytes)})
            atomic_json(self.path / "preview_manifest.json", self._manifest)
        except BaseException as exc:
            self._mark_failed(exc)
            raise

    @property
    def saved_count(self):
        return len(self._manifest["samples"])

    def _mark_failed(self, exc):
        try:
            self.finish("failed", f"{type(exc).__name__}: {exc}")
        except Exception as failure:
            exc.add_note(f"Could not persist final failure status: {failure}")

    def _validate_depth_calibration(self):
        calibration = self._depth_calibration
        depth = calibration.get("depth_camera", {})
        profile = calibration.get("depth_profile", {})
        dimensions = [profile.get(key) for key in ("height", "width", "fps")]
        if (calibration.get("schema_version") != 1
                or calibration.get("color_intrinsics_sha256") != self._signature
                or any(type(value) is not int or value <= 0 for value in dimensions)
                or profile.get("format") != "z16"
                or depth.get("stream") != "depth"
                or depth.get("serial") != self._camera.get("serial")
                or depth.get("device_model") != self._camera.get("device_model")
                or (depth.get("image_height"), depth.get("image_width"), depth.get("fps")) != tuple(dimensions)):
            raise ValueError("invalid_depth_calibration_identity_or_profile")
        scale, skew = calibration.get("depth_scale_m"), calibration.get("max_pair_skew_ms")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in (scale, skew)):
            raise ValueError("positive_depth_scale_and_pair_tolerance_required")
        extrinsics = calibration.get("depth_to_color", {})
        rotation = np.asarray(extrinsics.get("rotation"), dtype=float)
        translation = np.asarray(extrinsics.get("translation_m"), dtype=float)
        if (rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
            raise ValueError("invalid_depth_to_color_extrinsics")
        signature = intrinsics_signature(depth)
        if depth.get("intrinsics_sha256", signature) != signature:
            raise ValueError("depth_intrinsics_fingerprint_mismatch")

    def _depth_record(self, diagnostic, sample):
        if self._depth_calibration is None:
            if sample is not None or diagnostic.get("depth") is not None:
                raise ValueError("depth_sample_requires_session_calibration")
            return None, None
        depth = diagnostic.get("depth")
        if not isinstance(sample, dict) or not isinstance(depth, dict):
            raise ValueError("depth_enabled_session_requires_sample_and_diagnostic")
        if (sample.get("calibration") != self._depth_calibration
                or depth.get("calibration") != self._depth_calibration):
            raise ValueError("depth_calibration_mismatch")
        color_frame = _copy(sample.get("color_frame"))
        depth_frame = _copy(sample.get("frame"))
        if color_frame != _copy(diagnostic.get("frame")):
            raise ValueError("depth_sample_color_frame_mismatch")
        if depth_frame != _copy(depth.get("frame")):
            raise ValueError("depth_sample_frame_mismatch")
        if not isinstance(depth.get("status"), str) or not isinstance(depth.get("reasons"), list):
            raise ValueError("depth_diagnostic_status_and_reasons_required")
        raw = sample.get("raw_z16")
        pixel_hash = None
        if raw is not None:
            profile = self._depth_calibration["depth_profile"]
            if (not isinstance(raw, np.ndarray) or raw.dtype != np.uint16
                    or raw.shape != (profile["height"], profile["width"])):
                raise ValueError("raw_uint16_depth_profile_required")
            if not isinstance(sample.get("frame"), dict):
                raise ValueError("raw_depth_requires_frame_metadata")
            pixel_hash = _hash(raw.tobytes())
        elif not depth["reasons"]:
            raise ValueError("missing_depth_requires_explicit_reason")
        if depth.get("raw_pixels_sha256") != pixel_hash:
            raise ValueError("diagnostic_depth_pixels_mismatch")
        return raw, {"available": raw is not None, "file": None, "file_sha256": None,
                     "raw_pixels_sha256": pixel_hash, "frame": depth_frame,
                     "color_frame": color_frame,
                     "depth_scale_m": self._depth_calibration["depth_scale_m"],
                     "calibration_file": self._manifest["depth_camera_file"],
                     "calibration_sha256": self._manifest["depth_camera_sha256"],
                     "status": depth["status"], "reasons": _copy(depth["reasons"])}

    def save(self, image, diagnostic, overlay, *, depth_sample=None):
        if self._manifest["status"] != "recording" or self._pending_finish is not None:
            raise RuntimeError("preview_session_already_finished")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError("raw_uint8_BGR_image_required")
        validate_camera(self._camera, image.shape)
        if diagnostic.get("image_pixels_sha256") != _hash(image.tobytes()):
            raise ValueError("diagnostic_image_pixels_mismatch")
        if (diagnostic.get("source") != "live_diagnostic"
                or diagnostic.get("authorizes_robot_motion") is not False):
            raise ValueError("diagnostic_source_must_not_authorize_motion")
        camera = diagnostic.get("camera", {})
        if intrinsics_signature(camera) != self._signature or any(
                camera.get(key) != self._camera.get(key) for key in
                ("device_name", "device_model", "fps", "pixel_format", "validated", "version")):
            raise ValueError("diagnostic_camera_mismatch")
        if camera.get("intrinsics_sha256", self._signature) != self._signature:
            raise ValueError("diagnostic_camera_fingerprint_mismatch")
        if any(diagnostic.get("geometry", {}).get(key) != self._geometry.get(key)
               for key in ("version", "status", "validated", "source_sha256")):
            raise ValueError("diagnostic_geometry_mismatch")
        if (not isinstance(overlay, np.ndarray) or overlay.dtype != np.uint8
                or overlay.ndim != 3 or overlay.shape[2] != 3 or min(overlay.shape[:2]) < 1):
            raise ValueError("overlay_uint8_BGR_required")
        frame = diagnostic.get("frame", {})
        frame = {key: frame[key] for key in
                 ("frame_number", "device_timestamp_ms", "timestamp_domain", "received_at_utc")}
        raw_depth, depth_record = self._depth_record(diagnostic, depth_sample)
        identity = {**frame, "image_pixels_sha256": diagnostic["image_pixels_sha256"]}
        if depth_record is not None:
            identity["depth"] = {key: depth_record[key] for key in
                                 ("raw_pixels_sha256", "frame", "color_frame", "calibration_sha256")}
        previous = self._frames.get(frame["frame_number"])
        if previous is not None:
            if identity != previous:
                raise ValueError("previous_frame_metadata_or_pixels_changed")
            return None
        try:
            record = self._capture.save(image, **frame)
            if record is None:
                return None
            stem = Path(record["image"]).stem
            if raw_depth is not None:
                encoded, pixels = cv2.imencode(".png", raw_depth)
                if not encoded:
                    raise OSError("raw_depth_PNG_encoding_failed")
                depth_bytes = pixels.tobytes()
                depth_record.update({"file": f"depth/{stem}.png", "file_sha256": _hash(depth_bytes)})
                _write_new(self.path / depth_record["file"], depth_bytes)
            overlay_file = f"overlays/{stem}.png"
            encoded, pixels = cv2.imencode(".png", overlay)
            if not encoded:
                raise OSError("overlay_PNG_encoding_failed")
            overlay_bytes = pixels.tobytes()
            _write_new(self.path / overlay_file, overlay_bytes)
            diagnostic_file = f"diagnostics/{stem}.json"
            payload = {**_copy(diagnostic), "schema_version": self._manifest["schema_version"],
                       "image": record["image"], "image_sha256": record["image_sha256"],
                       "camera_file": "camera.json", "camera_sha256": self._manifest["camera_sha256"],
                       "intrinsics_sha256": self._signature,
                       "overlay": overlay_file, "overlay_sha256": _hash(overlay_bytes)}
            if depth_record is not None:
                payload["depth_capture"] = depth_record
            diagnostic_bytes = _json_bytes(payload)
            _write_new(self.path / diagnostic_file, diagnostic_bytes)
            sample = {**record, "image_pixels_sha256": diagnostic["image_pixels_sha256"],
                      "diagnostic": diagnostic_file, "diagnostic_sha256": _hash(diagnostic_bytes),
                      "overlay": overlay_file, "overlay_sha256": _hash(overlay_bytes)}
            if depth_record is not None:
                sample["depth"] = depth_record
            updated = {**self._manifest, "samples": [*self._manifest["samples"], sample]}
            atomic_json(self.path / "preview_manifest.json", updated)
            self._manifest = updated
            self._frames[frame["frame_number"]] = identity
            return dict(sample)
        except BaseException as exc:
            self._mark_failed(exc)
            raise

    def finish(self, status="completed", error=None):
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("invalid_preview_final_status")
        if error is not None and not isinstance(error, str):
            raise ValueError("preview_error_must_be_text")
        if self._manifest["status"] != "recording":
            return
        if self._pending_finish is None:
            self._pending_finish = {**self._manifest, "status": status, "error": error,
                                    "finished_at": datetime.now(timezone.utc).isoformat()}
        # Retrying a partially written pair must keep the original finish outcome.
        updated = self._pending_finish
        self._capture.finish(updated["status"], updated["error"])
        atomic_json(self.path / "preview_manifest.json", updated)
        self._manifest = updated
        self._pending_finish = None
