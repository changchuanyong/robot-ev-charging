"""Optional RGB + raw depth acquisition for diagnostics, without robot output."""
from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np

from .collect import CaptureColorCamera, list_devices, validate_config
from .contracts import intrinsics_signature, validate_camera
from .win_privilege import ensure_camera_privilege


def validate_depth_config(config):
    for key in ("width", "height", "fps"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"invalid_depth_{key}")
    skew = config.get("max_pair_skew_ms")
    if (isinstance(skew, bool) or not isinstance(skew, (int, float))
            or not math.isfinite(skew) or skew <= 0):
        raise ValueError("invalid_max_pair_skew_ms")


def _stream_camera(profile, stream, serial, name, model):
    intrinsics = profile.get_intrinsics()
    camera = {
        "stream": stream, "serial": serial, "device_name": name, "device_model": model,
        "image_width": intrinsics.width, "image_height": intrinsics.height,
        "fps": profile.fps(), "pixel_format": "bgr8" if stream == "color" else "z16",
        "camera_matrix": [[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]],
        "dist_coeffs": list(intrinsics.coeffs), "distortion_model": intrinsics.model.name,
        "source": f"device_{stream}_stream", "validated": False, "version": None,
        "is_placeholder": False, "is_approximate": False,
    }
    # Validate matrix and distortion without claiming depth is an RGB stream.
    validate_camera({**camera, "stream": "color"}, (intrinsics.height, intrinsics.width, 3))
    camera["intrinsics_sha256"] = intrinsics_signature(camera)
    return camera


def _extrinsics(extrinsics):
    # rs2_extrinsics.rotation is column-major; persisted matrices are row-major.
    rotation = np.asarray(extrinsics.rotation, dtype=float).reshape(3, 3, order="F")
    translation = np.asarray(extrinsics.translation, dtype=float)
    if (translation.shape != (3,) or not np.isfinite(translation).all()
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("invalid_depth_to_color_extrinsics")
    return {"rotation": rotation.tolist(), "translation_m": translation.tolist()}


def _frame_metadata(frame, received_at):
    return {"frame_number": frame.get_frame_number(),
            "device_timestamp_ms": frame.get_timestamp(),
            "timestamp_domain": frame.get_frame_timestamp_domain().name,
            "received_at_utc": received_at}


class CaptureRGBDCamera(CaptureColorCamera):
    """Preserve SDK Z16 and map SDK pointcloud vertices to the color optical frame.

    ``read_rgbd`` returns a copied RGB image, color frame metadata, and a depth
    packet. Unusable pairs keep raw depth for diagnosis but have no derived 3D
    points. The legacy ``read`` two-item contract remains available.
    """
    def __init__(self, config, depth_config):
        validate_config(config)
        validate_depth_config(depth_config)
        super().__init__(config)
        self.depth_config = dict(depth_config)
        self._previous = {}

    def __enter__(self):
        import pyrealsense2 as rs
        ensure_camera_privilege(sdk=rs)
        serial, model = self.config["serial"], self.config["device_model"]
        matches = [device for device in list_devices(rs) if device["serial"] == serial]
        if len(matches) != 1 or matches[0]["name"].split()[-1] != model:
            raise ValueError("RGBD_camera_model_or_serial_mismatch")
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, self.config["width"], self.config["height"],
                             rs.format.bgr8, self.config["fps"])
        config.enable_stream(rs.stream.depth, self.depth_config["width"], self.depth_config["height"],
                             rs.format.z16, self.depth_config["fps"])
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(config)
            device = profile.get_device()
            actual_serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            if actual_serial != serial or name.split()[-1] != model:
                raise ValueError("RGBD_started_camera_identity_mismatch")
            color = profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            for video, requested, pixel_format in (
                    (color, self.config, rs.format.bgr8), (depth, self.depth_config, rs.format.z16)):
                intrinsic = video.get_intrinsics()
                if (intrinsic.width, intrinsic.height, video.fps(), video.format()) != (
                        requested["width"], requested["height"], requested["fps"], pixel_format):
                    raise ValueError("RGBD_actual_stream_profile_mismatch")
            self.camera = _stream_camera(color, "color", serial, name, model)
            scale = float(device.first_depth_sensor().get_depth_scale())
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("invalid_depth_scale_m")
            self.depth_calibration = {
                "schema_version": 1, "source": "device_sdk", "validated": False,
                "depth_camera": _stream_camera(depth, "depth", serial, name, model),
                "color_intrinsics_sha256": self.camera["intrinsics_sha256"],
                "depth_scale_m": scale, "depth_to_color": _extrinsics(depth.get_extrinsics_to(color)),
                "depth_profile": {key: self.depth_config[key] for key in ("width", "height", "fps")},
                "max_pair_skew_ms": self.depth_config["max_pair_skew_ms"],
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "point_frame": "camera_color_optical", "point_unit": "m",
                "depth_pixels": "native_unaligned_unfiltered_z16",
            }
            self.depth_calibration["depth_profile"]["format"] = "z16"
            self._pointcloud = rs.pointcloud()
            self._previous = {}
            self._start_rgb_controls(device, rs)
            for _ in range(self.config["warmup_frames"]):
                self.read_rgbd()
        except BaseException:
            self.close()
            raise
        return self

    def _check_frame(self, stream, frame, reasons):
        timestamp = frame["device_timestamp_ms"]
        if not math.isfinite(timestamp) or timestamp <= 0:
            reasons.append(f"invalid_{stream}_timestamp")
            return
        if frame["timestamp_domain"] not in {"hardware_clock", "system_time", "global_time"}:
            reasons.append(f"invalid_{stream}_timestamp_domain")
            return
        previous = self._previous.get(stream)
        if previous is not None and (
                frame["frame_number"] <= previous["frame_number"]
                or (frame["timestamp_domain"] == previous["timestamp_domain"]
                    and timestamp <= previous["device_timestamp_ms"])):
            reasons.append(f"repeated_or_reordered_{stream}_frame")
            return
        if previous is not None and frame["timestamp_domain"] != previous["timestamp_domain"]:
            reasons.append(f"{stream}_timestamp_domain_changed")
        self._previous[stream] = frame

    def read_rgbd(self):
        if self.pipeline is None:
            raise RuntimeError("RGBD_camera_not_started")
        frames = self.pipeline.wait_for_frames(self.config["timeout_ms"])
        received_at = datetime.now(timezone.utc).isoformat()
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color:
            raise ValueError("RGBD_missing_color_frame")
        image = np.asanyarray(color.get_data()).copy()
        if image.dtype != np.uint8 or image.shape != (self.config["height"], self.config["width"], 3):
            raise ValueError("RGBD_invalid_color_pixels")
        frame = _frame_metadata(color, received_at)
        if self.rgb_controls is not None:
            frame["rgb_settings"] = self.rgb_controls.frame_settings(color)
        packet = {"raw_z16": None, "frame": None, "color_frame": dict(frame),
                  "calibration": self.depth_calibration,
                  "points_color_m": None, "color_pixels": None, "pair_skew_ms": None, "reasons": []}
        reasons = packet["reasons"]
        self._check_frame("color", frame, reasons)
        if not depth:
            reasons.append("missing_depth")
            return image, frame, packet
        depth_frame = _frame_metadata(depth, received_at)
        packet["frame"] = depth_frame
        self._check_frame("depth", depth_frame, reasons)
        raw = np.asanyarray(depth.get_data()).copy()
        shape = (self.depth_config["height"], self.depth_config["width"])
        if raw.dtype != np.uint16 or raw.shape != shape:
            reasons.append("invalid_depth_pixels")
            return image, frame, packet
        packet["raw_z16"] = raw
        if frame["timestamp_domain"] != depth_frame["timestamp_domain"]:
            reasons.append("depth_timestamp_domain_mismatch")
        else:
            skew = abs(frame["device_timestamp_ms"] - depth_frame["device_timestamp_ms"])
            if math.isfinite(skew):
                packet["pair_skew_ms"] = skew
                if skew > self.depth_config["max_pair_skew_ms"]:
                    reasons.append("depth_pair_skew_exceeded")
        if reasons:
            return image, frame, packet
        try:
            self._pointcloud.map_to(color)
            cloud = self._pointcloud.calculate(depth)
            vertices = np.asanyarray(cloud.get_vertices()).view(np.float32).reshape(*shape, 3)
            texture = np.asanyarray(cloud.get_texture_coordinates()).view(np.float32).reshape(*shape, 2)
        except (RuntimeError, ValueError) as exc:
            reasons.append("depth_projection_failed")
            packet["error"] = f"{type(exc).__name__}: {exc}"
            return image, frame, packet
        transform = self.depth_calibration["depth_to_color"]
        # SDK vertices already use metres. Do not multiply them by depth_scale_m.
        xyz = vertices @ np.asarray(transform["rotation"], dtype=np.float32).T
        xyz += np.asarray(transform["translation_m"], dtype=np.float32)
        uv = texture * np.array([self.config["width"], self.config["height"]], dtype=np.float32)
        invalid = (raw == 0) | ~np.isfinite(vertices).all(axis=2) | (vertices[:, :, 2] <= 0)
        xyz[invalid] = np.nan
        uv[invalid] = np.nan
        packet["points_color_m"], packet["color_pixels"] = xyz, uv
        return image, frame, packet

    def read(self):
        image, frame, _ = self.read_rgbd()
        return image, frame
