"""D415/D435 RGB acquisition with synchronized, calibration-bound pose input."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from ev_charging.capture.collect import CaptureColorCamera
from ev_charging.paths import PROJECT_ROOT
from ev_charging.vision.contracts import FrameClock, FramePacket, intrinsics_signature, validate_camera


def validated_camera_override(sdk_camera: dict, config: dict) -> dict:
    """Apply only an explicitly reviewed calibration bound to this exact RGB profile."""
    value = config.get("intrinsics_override_file")
    if value is None:
        return sdk_camera
    if not isinstance(value, str) or not value.strip():
        raise ValueError("intrinsics_override_file_requires_path")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    fitted = json.loads(path.read_text(encoding="utf-8-sig"))
    if fitted.get("validated") is not True or config.get("intrinsics_validated") is not True:
        raise ValueError("intrinsics_override_requires_explicit_validation")
    if not fitted.get("version") or fitted["version"] != config.get("calibration_version"):
        raise ValueError("intrinsics_override_version_mismatch")
    for key in ("serial", "device_name", "stream", "image_width", "image_height", "fps", "pixel_format"):
        if fitted.get(key) != sdk_camera.get(key) or fitted.get(key) is None:
            raise ValueError(f"intrinsics_override_profile_mismatch:{key}")
    if config.get("device_model") not in {"D415", "D435"} or (
        fitted["device_name"].split()[-1] != config["device_model"]
    ):
        raise ValueError("intrinsics_override_device_model_mismatch")
    validate_camera(fitted, (sdk_camera["image_height"], sdk_camera["image_width"], 3))
    signature = intrinsics_signature(fitted)
    if signature != fitted.get("intrinsics_sha256") or signature != config.get("intrinsics_sha256"):
        raise ValueError("intrinsics_override_fingerprint_mismatch")
    sdk_signature = intrinsics_signature(sdk_camera)
    if fitted.get("source_sdk_intrinsics_sha256") != sdk_signature:
        raise ValueError("intrinsics_override_source_sdk_changed")
    approval = fitted.get("validation_record", {})
    for key in ("reviewer", "reviewed_at", "report_path", "report_sha256"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            raise ValueError(f"intrinsics_override_validation_record_requires_{key}")
    try:
        date = datetime.fromisoformat(approval["reviewed_at"].replace("Z", "+00:00"))
        limit = float(approval["max_heldout_rms_px"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("intrinsics_override_invalid_validation_record") from exc
    if date.tzinfo is None or not np.isfinite(limit) or limit <= 0:
        raise ValueError("intrinsics_override_invalid_validation_record")
    report_path = (path.parent / approval["report_path"]).resolve()
    content = report_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != approval["report_sha256"]:
        raise ValueError("intrinsics_override_validation_report_changed")
    report = json.loads(content)
    heldout = report.get("summary", {}).get("heldout", {})
    residual = heldout.get("fitted", {}).get("pooled_corner_rms_px")
    if (
        report.get("kind") != "rgb_intrinsics_check_v1"
        or report.get("version") != fitted["version"]
        or report.get("source_camera_sha256") != sdk_signature
        or report.get("fitted_camera_sha256") != signature
        or type(heldout.get("view_count")) is not int
        or heldout["view_count"] < 3
        or isinstance(residual, bool)
        or not isinstance(residual, (int, float))
        or not np.isfinite(residual)
        or residual < 0
        or residual > limit
    ):
        raise ValueError("intrinsics_override_heldout_validation_failed")
    return fitted


class RealSenseCamera:
    def __init__(self, config: dict):
        self.capture = CaptureColorCamera({"warmup_frames": 0, "timeout_ms": 2000, **config})
        self.config = self.capture.config
        self.session = uuid.uuid4().hex
        self.clock = FrameClock(float(config["max_frame_age_s"]))

    def __enter__(self):
        import pyrealsense2 as rs

        try:
            self.capture.__enter__()
            device = self.capture.pipeline.get_active_profile().get_device()
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
            self.camera = dict(self.capture.camera)
            self.camera.update(
                validated=bool(self.config.get("intrinsics_validated", False)),
                version=self.config.get("calibration_version"),
            )
            if self.config.get("intrinsics_override_file") is not None:
                self.camera = validated_camera_override(self.camera, self.config)
            elif self.camera["validated"] and (
                not self.camera["version"]
                or self.config.get("intrinsics_sha256") != self.camera["intrinsics_sha256"]
            ):
                raise ValueError("validated_intrinsics_require_matching_version_and_stream_fingerprint")
            self.startup_deadline = time.monotonic() + float(self.config.get("clock_startup_timeout_s", 5.0))
        except BaseException:
            self.close()
            raise
        return self

    def read(self) -> FramePacket:
        if self.capture.pipeline is None:
            raise RuntimeError("camera_not_open")
        while True:
            image, metadata = self.capture.read()
            received_mono = time.monotonic()
            received_wall = time.time()
            number = metadata["frame_number"]
            timestamp = metadata["device_timestamp_ms"]
            domain = metadata["timestamp_domain"]
            # Global timestamp negotiation can take several initial frames.
            # The published observation remains invalid during this interval.
            if (
                domain == "hardware_clock"
                and received_mono < self.startup_deadline
                and self.clock.last_frame == -1
            ):
                continue
            captured = self.clock.map(number, timestamp, domain, received_mono, received_wall)
            break
        return FramePacket(
            image=image,
            frame_id=f"{self.session}:{number}",
            captured_at=captured,
            received_at=received_mono,
            source="realsense",
            time_sync_valid=True,
            camera=self.camera,
            camera_time_ms=timestamp,
            timestamp_domain=domain,
        )

    def close(self):
        self.capture.close()

    def __exit__(self, *_):
        self.close()
