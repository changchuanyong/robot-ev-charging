"""Lossless RGB capture sessions; importing this module never opens a device."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import cv2
import numpy as np

from .contracts import atomic_capture_json as atomic_json
from .contracts import intrinsics_signature, validate_camera
from .model_config import resolve_model_config


KIND = "realsense_rgb_capture"
ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_batch(root: Path, batch_id: str, split: str, metadata: dict) -> None:
    for path in root.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("kind") == KIND and manifest.get("batch_id") == batch_id
                and manifest.get("split") != split):
            raise ValueError("capture_batch_cannot_cross_dataset_splits")
        if manifest.get("kind") == KIND and manifest.get("split") != split:
            for key in ("specimen_id", "printing_batch_id"):
                if metadata.get(key) and manifest.get("metadata", {}).get(key) == metadata[key]:
                    raise ValueError(f"capture_{key}_cannot_cross_dataset_splits")


class CaptureSession:
    def __init__(self, output_root: Path, camera: dict, batch_id: str,
                 split: str = "train", notes: str = "", *, metadata: dict | None = None):
        if not isinstance(batch_id, str) or not batch_id.strip() or batch_id != batch_id.strip():
            raise ValueError("nonempty_batch_id_without_surrounding_spaces_required")
        if split not in {"train", "val", "test", "unknown"} or not isinstance(notes, str):
            raise ValueError("invalid_capture_split_or_notes")
        metadata = json.loads(json.dumps({} if metadata is None else metadata, allow_nan=False))
        if not isinstance(metadata, dict):
            raise ValueError("capture_metadata_must_be_object")
        for key in ("specimen_id", "printing_batch_id", "geometry_version"):
            if metadata.get(key) is not None and (not isinstance(metadata[key], str) or not metadata[key].strip()
                                                  or metadata[key] != metadata[key].strip()):
                raise ValueError(f"invalid_capture_metadata_{key}")
        if metadata.get("source_kind", "unknown") not in {"printed_connector", "real_connector", "unknown"}:
            raise ValueError("invalid_capture_source_kind")
        if not isinstance(metadata.get("conditions", {}), dict):
            raise ValueError("capture_conditions_must_be_object")
        # A JSON copy isolates the recorded stream calibration from caller mutations.
        self._camera = json.loads(json.dumps(camera, allow_nan=False))
        for key in ("serial", "device_name"):
            if not isinstance(self._camera.get(key), str) or not self._camera[key].strip():
                raise ValueError("camera_serial_and_device_name_required")
        if self._camera.get("stream") != "color":
            raise ValueError("RGB_requires_color_stream_intrinsics")
        if self._camera.get("validated") is not False:
            raise ValueError("factory_capture_intrinsics_must_not_claim_validation")
        dimensions = [self._camera.get(key) for key in ("image_height", "image_width")]
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("positive_integer_camera_dimensions_required")
        validate_camera(self._camera, tuple(dimensions) + (3,))
        signature = intrinsics_signature(self._camera)
        if self._camera.get("intrinsics_sha256", signature) != signature:
            raise ValueError("camera_stream_fingerprint_mismatch")
        self._camera["intrinsics_sha256"] = signature
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        _check_batch(root, batch_id, split, metadata)
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex
        self.path = root / session_id
        self.path.mkdir()  # Never reuse or clean up an existing session.
        try:
            (self.path / "images").mkdir()
            camera_path = self.path / "camera.json"
            atomic_json(camera_path, self._camera)
            self._camera_sha256 = hashlib.sha256(camera_path.read_bytes()).hexdigest()
            self._manifest = {
                "schema_version": 1, "kind": KIND, "session_id": session_id,
                "created_at": _utc_now(), "batch_id": batch_id, "split": split,
                "notes": notes, "status": "recording", "camera_file": "camera.json",
                "camera_sha256": self._camera_sha256, "frames": [],
                "metadata": metadata,
            }
            atomic_json(self.path / "manifest.json", self._manifest)
            # Recheck after publishing: concurrent creators cannot both keep conflicting splits.
            _check_batch(root, batch_id, split, metadata)
        except BaseException:
            for name in ("manifest.json", "camera.json"):
                (self.path / name).unlink(missing_ok=True)
            images = self.path / "images"
            if images.exists():
                images.rmdir()
            self.path.rmdir()
            raise
        self._frame_numbers: set[int] = set()
        self._image_hashes: set[str] = set()

    def save(self, image: np.ndarray, frame_number: int, device_timestamp_ms: float,
             timestamp_domain: str, received_at_utc: str) -> dict | None:
        if self._manifest["status"] != "recording":
            raise RuntimeError("capture_session_already_finished")
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError("full_resolution_uint8_BGR_image_required")
        validate_camera(self._camera, image.shape)
        if type(frame_number) is not int or frame_number < 0:
            raise ValueError("nonnegative_integer_device_frame_number_required")
        if (not isinstance(device_timestamp_ms, (int, float)) or isinstance(device_timestamp_ms, bool)
                or not math.isfinite(device_timestamp_ms) or device_timestamp_ms < 0):
            raise ValueError("finite_nonnegative_device_timestamp_required")
        if timestamp_domain not in {"hardware_clock", "system_time", "global_time"}:
            raise ValueError("unsupported_device_timestamp_domain")
        try:
            received = datetime.fromisoformat(received_at_utc)
        except (TypeError, ValueError) as exc:
            raise ValueError("UTC_receive_timestamp_required") from exc
        if received.utcoffset() is None or received.utcoffset().total_seconds() != 0:
            raise ValueError("UTC_receive_timestamp_required")
        if hashlib.sha256((self.path / "camera.json").read_bytes()).hexdigest() != self._camera_sha256:
            raise ValueError("session_camera_file_changed")
        if frame_number in self._frame_numbers:
            return None
        encoded_ok, encoded = cv2.imencode(".png", image)
        if not encoded_ok:
            raise OSError("PNG_encoding_failed")
        image_bytes = encoded.tobytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest in self._image_hashes:
            self._frame_numbers.add(frame_number)
            return None
        record = {
            "image": f"images/frame_{len(self._manifest['frames']) + 1:06d}.png",
            "image_sha256": digest, "width": image.shape[1], "height": image.shape[0],
            "frame_number": frame_number, "device_timestamp_ms": float(device_timestamp_ms),
            "timestamp_domain": timestamp_domain, "received_at_utc": received_at_utc,
        }
        path = self.path / record["image"]
        created = False
        updated = {**self._manifest, "frames": [*self._manifest["frames"], record]}
        try:
            with path.open("xb") as stream:
                created = True
                stream.write(image_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(self.path / "manifest.json", updated)
        except BaseException:
            if created:
                path.unlink(missing_ok=True)
            raise
        self._manifest = updated
        self._frame_numbers.add(frame_number)
        self._image_hashes.add(digest)
        return dict(record)

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("invalid_capture_final_status")
        if error is not None and not isinstance(error, str):
            raise ValueError("capture_error_must_be_text")
        if self._manifest["status"] != "recording":
            return
        updated = {**self._manifest, "status": status, "finished_at": _utc_now(), "error": error}
        atomic_json(self.path / "manifest.json", updated)
        self._manifest = updated


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
                ("vision/preview.py", "vision/preview_render.py", "vision/rgb_controls.py",
                 "vision/collect.py", "vision/launch.py",
                 "vision/diagnose.py", "vision/model_config.py", "vision/capture_session.py", "vision/contracts.py",
                 "live/python/pose_correspondence.py", "live/python/pnp2.py")}
        if self._depth_calibration is not None:
            code.update({name: _asset(name) for name in
                         ("vision/depth_diagnostic.py",)})
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
