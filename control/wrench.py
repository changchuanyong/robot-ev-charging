"""Normalize force data to the insertion task axes at the plug control point.

The current channel is transformed and bias-compensated but is not low-pass
filtered; independent hard protection must use it. The second channel is for
the historical learning window. Absolute data age belongs to the supervisor;
this processor rejects invalid samples and excessive update intervals.
"""
from __future__ import annotations

import numpy as np

from .geometry import invert, transform_wrench, validate_transform
from .types import RobotSample


def _vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return result.copy()


class WrenchProcessor:
    def __init__(self, config: dict):
        self.T_tool_sensor = validate_transform(config["T_tool_sensor"])
        self.version = config["version"]
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("wrench transform needs a version")
        if type(config["external_compensated"]) is not bool:
            raise ValueError("external_compensated must explicitly be true or false")
        self.external_compensated = config["external_compensated"]
        if config.get("bias_frozen") is not True:
            raise ValueError("force bias must be frozen before a contact attempt")
        # Bias is always defined at the sensor origin, in sensor axes, N/Nm.
        # reset() never changes this calibration; contact cannot zero real load.
        self.bias = _vector(config["bias"], 6, "sensor bias")
        self.filter_tau = float(config["filter_tau"])
        self.max_gap = float(config["max_gap"])
        if (not np.isfinite(self.filter_tau) or self.filter_tau < 0
                or not np.isfinite(self.max_gap) or self.max_gap <= 0):
            raise ValueError("filter_tau must be nonnegative and max_gap positive")
        self.gravity = None
        if not self.external_compensated:
            if "gravity" not in config:
                raise ValueError("raw force requires explicit mass, center of mass and base gravity")
            gravity = config["gravity"]
            mass = float(gravity["mass"])
            center = _vector(gravity["center_of_mass_sensor"], 3, "sensor-frame center of mass")
            acceleration = _vector(gravity["gravity_base"], 3, "base-frame gravity")
            if not np.isfinite(mass) or mass < 0 or np.linalg.norm(acceleration) <= 0:
                raise ValueError("gravity compensation needs nonnegative mass and nonzero gravity")
            self.gravity = mass, center, acceleration
        self.reset()

    def reset(self) -> None:
        """Reset only the historical filter, preserving the frozen force bias."""
        self._filtered = None

    def _sensor_wrench(self, sample: RobotSample, R_base_task: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base_tool = validate_transform(sample.pose)
        source = _vector(sample.wrench, 6, "wrench")
        metadata = sample.metadata
        if not isinstance(metadata, dict):
            raise ValueError("force metadata must be a dictionary")
        compensated = metadata.get("external_compensated")
        if type(compensated) is not bool or compensated != self.external_compensated:
            raise ValueError("force compensation metadata does not match the configured channel")
        declared_version = metadata.get("wrench_version", self.version)
        if declared_version != self.version:
            raise ValueError("wrench calibration version mismatch")
        channel = metadata.get("force_channel")
        if not self.external_compensated:
            if channel != "force_data" or sample.wrench_frame != "sensor":
                raise ValueError("raw gravity compensation requires force_data at the sensor origin")
        elif channel == "force_data":
            raise ValueError("raw force_data cannot be declared externally compensated")
        R_base_sensor = base_tool[:3, :3] @ self.T_tool_sensor[:3, :3]
        if sample.wrench_frame == "sensor":
            return source, R_base_sensor
        if metadata.get("moment_origin") != "tool":
            raise ValueError("non-sensor wrench must explicitly use the plug/tool moment origin")
        if sample.wrench_frame == "task" and metadata.get("simulation") is not True:
            raise ValueError("task-frame force shortcut is reserved for explicit simulation")
        if sample.wrench_frame == "tool":
            sensor_source = invert(self.T_tool_sensor)
        elif sample.wrench_frame in ("base", "task"):
            # These inputs have task/base-oriented axes but their moment
            # origin remains the tool point, not the base origin.
            sensor_source = invert(self.T_tool_sensor)
            R_base_source = np.eye(3) if sample.wrench_frame == "base" else R_base_task
            sensor_source[:3, :3] = R_base_sensor.T @ R_base_source
        else:
            raise ValueError(f"unsupported wrench frame: {sample.wrench_frame}")
        return transform_wrench(source, sensor_source), R_base_sensor

    def process(self, sample: RobotSample, R_base_task, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if not sample.valid or not np.isfinite(sample.timestamp):
            raise ValueError("invalid robot sample for wrench processing")
        if not np.isfinite(dt) or dt <= 0 or dt > self.max_gap:
            raise ValueError("force processing interval is invalid or exceeds max_gap")
        task_rotation = np.asarray(R_base_task, dtype=float)
        task_transform = np.eye(4)
        if task_rotation.shape != (3, 3):
            raise ValueError("R_base_task must be a proper 3x3 rotation")
        task_transform[:3, :3] = task_rotation
        validate_transform(task_transform)
        sensor_wrench, R_base_sensor = self._sensor_wrench(sample, task_rotation)
        if self.gravity is not None:
            mass, center, gravity_base = self.gravity
            gravity_force = mass * (R_base_sensor.T @ gravity_base)
            sensor_wrench = sensor_wrench - np.r_[gravity_force, np.cross(center, gravity_force)]
        sensor_wrench = sensor_wrench - self.bias
        tool_wrench = transform_wrench(sensor_wrench, self.T_tool_sensor)
        task_tool = np.eye(4)
        task_tool[:3, :3] = task_rotation.T @ np.asarray(sample.pose)[:3, :3]
        current = transform_wrench(tool_wrench, task_tool)
        if not np.all(np.isfinite(current)):
            raise ValueError("force normalization overflow")
        if self._filtered is None:
            filtered = current.copy()
        else:
            alpha = 1.0 if self.filter_tau == 0 else -np.expm1(-dt / self.filter_tau)
            filtered = self._filtered + alpha * (current - self._filtered)
        if not np.all(np.isfinite(filtered)):
            raise ValueError("force filter overflow")
        self._filtered = filtered
        return current.copy(), filtered.copy()
