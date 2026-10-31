"""Control-facing observations. Times are host monotonic seconds; lengths are metres."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


def intrinsics_signature(camera: dict) -> str:
    fields = ("stream", "serial", "image_width", "image_height", "camera_matrix", "dist_coeffs", "distortion_model")
    encoded = json.dumps({key: camera[key] for key in fields}, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    """Replace the complete message, including invalid observations, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


_RETRY_DELAYS_S = (.025, .05, .1, .2, .4)
_RETRYABLE_WINERRORS = {5, 32, 33}


def atomic_capture_json(path: Path, data: dict) -> None:
    """Retry temporary Windows sharing conflicts for capture files, outside control loops."""
    for attempt in range(len(_RETRY_DELAYS_S) + 1):
        try:
            atomic_json(path, data)
            return
        except PermissionError as exc:
            if (getattr(exc, "winerror", None) not in _RETRYABLE_WINERRORS
                    or attempt == len(_RETRY_DELAYS_S)):
                raise
            time.sleep(_RETRY_DELAYS_S[attempt])


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    frame_id: str
    captured_at: float
    received_at: float
    source: str
    time_sync_valid: bool
    camera: dict
    camera_time_ms: float | None = None
    timestamp_domain: str = "offline"


def empty_observation(packet: FramePacket | None, target_id: str, reason: str) -> dict:
    return {
        "schema_version": 1, "valid": False, "reason": reason,
        "source": packet.source if packet else "unavailable",
        "frame_id": packet.frame_id if packet else None,
        "target_id": target_id,
        "captured_at": packet.captured_at if packet else None,
        "received_at": packet.received_at if packet else None,
        "processed_at": time.monotonic(), "clock": "host_monotonic",
        "time_sync_valid": bool(packet and packet.time_sync_valid),
        "camera_time_ms": packet.camera_time_ms if packet else None,
        "timestamp_domain": packet.timestamp_domain if packet else None,
        "camera_frame": "camera_color_optical", "translation_unit": "m",
        "calibration_version": packet.camera.get("version") if packet else None,
        "camera_serial": packet.camera.get("serial") if packet else None,
        "T_camera_target": None, "quality": {},
    }


def validate_camera(camera: dict, image_shape: tuple) -> tuple[np.ndarray, np.ndarray, str]:
    if camera.get("is_placeholder") or camera.get("is_approximate"):
        raise ValueError("placeholder_camera_intrinsics")
    if camera.get("stream", "color") != "color":
        raise ValueError("RGB_requires_color_stream_intrinsics")
    if (camera.get("image_height"), camera.get("image_width")) != tuple(image_shape[:2]):
        raise ValueError("camera_resolution_mismatch")
    matrix = np.asarray(camera.get("camera_matrix"), dtype=float)
    coefficients = np.asarray(camera.get("dist_coeffs"), dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(matrix[0, 1]) > 1e-12 or abs(matrix[1, 0]) > 1e-12):
        raise ValueError("invalid_camera_matrix")
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("five_finite_distortion_coefficients_required")
    model = camera.get("distortion_model", "brown_conrady")
    if model not in {"none", "brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"}:
        raise ValueError("unsupported_distortion_model: " + str(model))
    if model == "none" and np.any(coefficients != 0):
        raise ValueError("nonzero_coefficients_for_undistorted_stream")
    return matrix, coefficients, model


def distort_normalized(points: np.ndarray, coefficients: np.ndarray, model: str) -> np.ndarray:
    """RealSense forward projection convention, distinct from OpenCV for modified Brown.

    Formula reference: librealsense src/rs.cpp, rs2_project_point_to_pixel.
    Inverse Brown here follows that SDK's forward projection semantics.
    """
    points = np.asarray(points, dtype=float)
    if model == "none":
        return points.copy()
    x, y = points[:, 0], points[:, 1]
    k1, k2, p1, p2, k3 = coefficients
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    if model in {"modified_brown_conrady", "inverse_brown_conrady"}:
        a, b = x * radial, y * radial
    elif model == "brown_conrady":
        a, b = x, y
    else:
        raise ValueError("unsupported_distortion_model")
    return np.column_stack((x*radial + 2*p1*a*b + p2*(r2 + 2*a*a),
                            y*radial + 2*p2*a*b + p1*(r2 + 2*b*b)))


def rectify_points(points: np.ndarray, matrix: np.ndarray,
                   coefficients: np.ndarray, model: str) -> np.ndarray:
    """Numerically invert forward projection, retaining K and using zero distortion for PnP."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("invalid_image_points")
    focal = np.array([matrix[0, 0], matrix[1, 1]])
    center = matrix[:2, 2]
    observed = (points - center) / focal
    estimate = observed.copy()
    epsilon = 1e-6
    for _ in range(20):
        forward = distort_normalized(estimate, coefficients, model)
        residual = forward - observed
        if np.max(np.abs(residual), initial=0) < 1e-11:
            break
        jx = (distort_normalized(estimate + [epsilon, 0], coefficients, model) - forward) / epsilon
        jy = (distort_normalized(estimate + [0, epsilon], coefficients, model) - forward) / epsilon
        jacobian = np.stack([jx, jy], axis=-1)
        try:
            update = np.linalg.solve(jacobian, residual[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("distortion_inverse_singular") from exc
        estimate -= update
        if not np.isfinite(estimate).all():
            raise ValueError("distortion_inverse_nonfinite")
    if np.max(np.abs(distort_normalized(estimate, coefficients, model) - observed), initial=0) > 1e-8:
        raise ValueError("distortion_inverse_not_converged")
    return estimate * focal + center


class FrameClock:
    """Map SDK global/system epoch timestamps to this PC's monotonic clock.

    Hardware-clock timestamps without a measured mapping are rejected. Clock
    steps and buffered old frames are rejected rather than stamped as new.
    """
    def __init__(self, max_age_s: float, max_clock_step_s: float = 0.01):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s_must_be_positive")
        self.max_age_s = max_age_s
        self.max_clock_step_s = max_clock_step_s
        self.last_frame = -1
        self.last_camera_time = None
        self.wall_offset = None

    def map(self, frame_number: int, camera_ms: float, domain: str,
            received_mono: float, received_wall: float) -> float:
        if domain not in {"global_time", "system_time"}:
            raise ValueError("camera_clock_not_synchronized")
        if not np.isfinite([camera_ms, received_mono, received_wall]).all():
            raise ValueError("nonfinite_camera_timestamp")
        if frame_number <= self.last_frame or (self.last_camera_time is not None and camera_ms <= self.last_camera_time):
            raise ValueError("repeated_or_reordered_camera_frame")
        offset = received_wall - received_mono
        if self.wall_offset is not None and abs(offset - self.wall_offset) > self.max_clock_step_s:
            raise ValueError("host_wall_clock_changed_restart_required")
        age = received_wall - camera_ms / 1000.0
        if age < -0.005 or age > self.max_age_s:
            raise ValueError("camera_frame_outside_age_bound")
        self.last_frame, self.last_camera_time = frame_number, camera_ms
        self.wall_offset = offset
        return received_mono - max(0.0, age)
