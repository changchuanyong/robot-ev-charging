"""Rigid transforms and wrenches. Translation is metres; rotation is radians.

``T_target_source`` maps source coordinates to target coordinates. Pose errors
and offsets use a spatial (left-multiplied) rotation vector, not Euler addition.
"""
from __future__ import annotations

import numpy as np


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result.copy()


def _rotation(value) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0)):
        raise ValueError("rotation must be a proper orthonormal 3x3 matrix")
    return rotation


def validate_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0)):
        raise ValueError("transform must be a finite homogeneous 4x4 matrix")
    _rotation(transform[:3, :3])
    return transform.copy()


def skew(vector) -> np.ndarray:
    x, y, z = _vector(vector, 3, "vector")
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def invert(transform) -> np.ndarray:
    transform = validate_transform(transform)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def exp_so3(rotvec) -> np.ndarray:
    vector = _vector(rotvec, 3, "rotation vector")
    theta = np.linalg.norm(vector)
    cross = skew(vector)
    if theta < 1e-7:
        a = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        b = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta**2
    return np.eye(3) + a * cross + b * (cross @ cross)


def log_so3(rotation) -> np.ndarray:
    rotation = _rotation(rotation)
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    vee = np.array([rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1]])
    if theta < 1e-7:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        # The antisymmetric part vanishes at pi. The unit eigenvector with
        # eigenvalue one remains well-conditioned; its sign at exactly pi is
        # necessarily ambiguous, but either sign reconstructs the same R.
        _, vectors = np.linalg.eigh((rotation + rotation.T) / 2.0)
        axis = vectors[:, -1]
        if np.dot(axis, vee) < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def pose_from_xyz_rpy(pose6) -> np.ndarray:
    """Extrinsic XYZ angles: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    x, y, z, roll, pitch, yaw = _vector(pose6, 6, "xyz_rpy")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = [x, y, z]
    return result


def xyz_rpy_from_pose(transform) -> np.ndarray:
    transform = validate_transform(transform)
    rotation = transform[:3, :3]
    cp = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    pitch = np.arctan2(-rotation[2, 0], cp)
    if cp > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.r_[transform[:3, 3], roll, pitch, yaw]


def transform_wrench(wrench, T_target_source) -> np.ndarray:
    """Change both axes and moment origin of [Fx,Fy,Fz,Mx,My,Mz].

    The translation is the source origin expressed from the target origin.
    A rotation-only change is insufficient when the sensor and TCP differ.
    """
    wrench = _vector(wrench, 6, "wrench")
    transform = validate_transform(T_target_source)
    force = transform[:3, :3] @ wrench[:3]
    moment = transform[:3, :3] @ wrench[3:] + np.cross(transform[:3, 3], force)
    return np.r_[force, moment]


def pose_error(reference, actual) -> np.ndarray:
    """Reference minus actual, expressed in their shared parent frame."""
    reference, actual = validate_transform(reference), validate_transform(actual)
    return np.r_[reference[:3, 3] - actual[:3, 3],
                 log_so3(reference[:3, :3] @ actual[:3, :3].T)]


def offset_pose(transform, delta6, R_base_task) -> np.ndarray:
    """Apply a task-frame displacement to a base-frame tool reference."""
    result = validate_transform(transform)
    delta = _vector(delta6, 6, "offset")
    rotation = _rotation(R_base_task)
    result[:3, 3] += rotation @ delta[:3]
    result[:3, :3] = exp_so3(rotation @ delta[3:]) @ result[:3, :3]
    return result
