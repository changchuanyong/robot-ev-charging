"""D435 color acquisition. Hardware access happens only inside the context manager."""
from __future__ import annotations

import time
import uuid

import numpy as np

from .contracts import FrameClock, FramePacket, intrinsics_signature


class RealSenseCamera:
    def __init__(self, config: dict):
        self.config = config
        self.pipeline = None
        self.session = uuid.uuid4().hex
        self.clock = FrameClock(float(config["max_frame_age_s"]))

    def __enter__(self):
        import pyrealsense2 as rs
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
            if "D435" not in device_name:
                raise ValueError("configured_device_is_not_a_D435_series_camera")
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            model = intrinsics.model.name
            self.camera = {
                "stream": "color", "serial": actual_serial,
                "device_name": device_name,
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
            if self.camera["validated"] and (
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
