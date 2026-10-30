"""D435 color acquisition. Hardware access happens only inside the context manager."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .contracts import FrameClock, FramePacket, intrinsics_signature, validate_camera
from .win_privilege import ensure_camera_privilege


def validated_camera_override(sdk_camera: dict, config: dict) -> dict:
    """Apply only an explicitly reviewed calibration bound to this exact RGB profile."""
    value = config.get("intrinsics_override_file")
    if value is None:
        return sdk_camera
    if not isinstance(value, str) or not value.strip():
        raise ValueError("intrinsics_override_file_requires_path")
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    fitted = json.loads(path.read_text(encoding="utf-8-sig"))
    if fitted.get("validated") is not True or config.get("intrinsics_validated") is not True:
        raise ValueError("intrinsics_override_requires_explicit_validation")
    if not fitted.get("version") or fitted["version"] != config.get("calibration_version"):
        raise ValueError("intrinsics_override_version_mismatch")
    for key in ("serial", "device_name", "stream", "image_width", "image_height", "fps", "pixel_format"):
        if fitted.get(key) != sdk_camera.get(key) or fitted.get(key) is None:
            raise ValueError(f"intrinsics_override_profile_mismatch:{key}")
    if fitted["device_name"].split()[-1] != "D435":
        raise ValueError("intrinsics_override_requires_D435")
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
    if (report.get("kind") != "rgb_intrinsics_check_v1" or report.get("version") != fitted["version"]
            or report.get("source_camera_sha256") != sdk_signature or report.get("fitted_camera_sha256") != signature
            or type(heldout.get("view_count")) is not int or heldout["view_count"] < 3
            or isinstance(residual, bool) or not isinstance(residual, (int, float))
            or not np.isfinite(residual) or residual < 0 or residual > limit):
        raise ValueError("intrinsics_override_heldout_validation_failed")
    return fitted


class RealSenseCamera:
    def __init__(self, config: dict):
        self.config = config
        self.pipeline = None
        self.session = uuid.uuid4().hex
        self.clock = FrameClock(float(config["max_frame_age_s"]))

    def __enter__(self):
        import pyrealsense2 as rs
        ensure_camera_privilege(sdk=rs)
        self.rs = rs
        serial = self.config.get("serial")
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError("configure_exact_D435_serial_before_acquisition")
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, int(self.config["width"]), int(self.config["height"]),
                             rs.format.bgr8, int(self.config["fps"]))
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(config)
            device = profile.get_device()
            actual_serial = device.get_info(rs.camera_info.serial_number)
            device_name = device.get_info(rs.camera_info.name)
            if actual_serial != serial:
                raise ValueError("camera_serial_mismatch")
            if device_name.split()[-1] != "D435":
                raise ValueError("configured_device_is_not_a_D435_series_camera")
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
            video = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = video.get_intrinsics()
            if (intrinsics.width, intrinsics.height, video.fps(), video.format()) != (
                    int(self.config["width"]), int(self.config["height"]), int(self.config["fps"]), rs.format.bgr8):
                raise ValueError("actual_D435_color_profile_differs_from_config")
            model = intrinsics.model.name
            self.camera = {
                "stream": "color", "serial": actual_serial,
                "device_name": device_name,
                "fps": video.fps(), "pixel_format": "bgr8",
                "image_width": intrinsics.width, "image_height": intrinsics.height,
                "camera_matrix": [[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]],
                "dist_coeffs": list(intrinsics.coeffs), "distortion_model": model,
                "is_placeholder": False, "is_approximate": False,
                "validated": bool(self.config.get("intrinsics_validated", False)),
                "version": self.config.get("calibration_version"),
                "source": "device_color_stream",
            }
            self.camera["intrinsics_sha256"] = intrinsics_signature(self.camera)
            validate_camera(self.camera, (intrinsics.height, intrinsics.width, 3))
            if self.config.get("intrinsics_override_file") is not None:
                self.camera = validated_camera_override(self.camera, self.config)
            elif self.camera["validated"] and (
                    not self.camera["version"]
                    or self.config.get("intrinsics_sha256") != self.camera["intrinsics_sha256"]):
                raise ValueError("validated_intrinsics_require_matching_version_and_stream_fingerprint")
            self.startup_deadline = time.monotonic() + float(self.config.get("clock_startup_timeout_s", 5.0))
        except Exception:
            self.close()
            raise
        return self

    def read(self) -> FramePacket:
        if self.pipeline is None:
            raise RuntimeError("camera_not_open")
        while True:
            frames = self.pipeline.wait_for_frames(int(self.config.get("timeout_ms", 2000)))
            received_mono = time.monotonic()
            received_wall = time.time()
            frame = frames.get_color_frame()
            if not frame:
                raise ValueError("missing_color_frame")
            number = frame.get_frame_number()
            timestamp = frame.get_timestamp()
            domain = frame.get_frame_timestamp_domain().name
            # Global timestamp negotiation can take several initial frames.
            # The published observation remains invalid during this interval.
            if domain == "hardware_clock" and received_mono < self.startup_deadline and self.clock.last_frame == -1:
                continue
            captured = self.clock.map(number, timestamp, domain, received_mono, received_wall)
            break
        return FramePacket(
            image=np.asanyarray(frame.get_data()).copy(),
            frame_id=f"{self.session}:{number}", captured_at=captured,
            received_at=received_mono, source="realsense", time_sync_valid=True,
            camera=self.camera, camera_time_ms=timestamp, timestamp_domain=domain,
        )

    def close(self):
        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                # start() may have failed before streaming began.
                pass

    def __exit__(self, *_):
        self.close()
