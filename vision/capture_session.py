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

from .contracts import atomic_json, intrinsics_signature, validate_camera


KIND = "realsense_rgb_capture"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_batch(root: Path, batch_id: str, split: str) -> None:
    for path in root.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("kind") == KIND and manifest.get("batch_id") == batch_id
                and manifest.get("split") != split):
            raise ValueError("capture_batch_cannot_cross_dataset_splits")


class CaptureSession:
    def __init__(self, output_root: Path, camera: dict, batch_id: str,
                 split: str = "train", notes: str = ""):
        if not isinstance(batch_id, str) or not batch_id.strip() or batch_id != batch_id.strip():
            raise ValueError("nonempty_batch_id_without_surrounding_spaces_required")
        if split not in {"train", "val", "test", "unknown"} or not isinstance(notes, str):
            raise ValueError("invalid_capture_split_or_notes")
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
        _check_batch(root, batch_id, split)
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
            }
            atomic_json(self.path / "manifest.json", self._manifest)
            # Recheck after publishing: concurrent creators cannot both keep conflicting splits.
            _check_batch(root, batch_id, split)
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
