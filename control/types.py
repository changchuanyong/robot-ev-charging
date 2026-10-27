"""Boundary types. Lengths m, rotations rad, forces N, moments N m.

Runtime timestamps use the host monotonic clock. Recorded timestamps are replayed
on their original relative time line, never interpreted as fresh live feedback.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RobotSample:
    seq: int
    timestamp: float
    pose: np.ndarray
    wrench: np.ndarray
    joints: tuple = ()
    valid: bool = True
    reason: str = ""
    wrench_frame: str = "task"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "seq": self.seq, "timestamp": self.timestamp,
            "pose": np.asarray(self.pose).tolist(), "wrench": np.asarray(self.wrench).tolist(),
            "joints": list(self.joints), "valid": self.valid, "reason": self.reason,
            "wrench_frame": self.wrench_frame, "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            int(data["seq"]), float(data["timestamp"]), np.asarray(data["pose"], float),
            np.asarray(data["wrench"], float), tuple(data.get("joints", ())),
            bool(data.get("valid", False)), str(data.get("reason", "")),
            str(data.get("wrench_frame", "unknown")), dict(data.get("metadata", {})),
        )


@dataclass
class Decision:
    state: str
    command: np.ndarray | None = None
    stop: bool = False
    finished: bool = False
    success: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "state": self.state,
            "command": None if self.command is None else self.command.tolist(),
            "stop": self.stop, "finished": self.finished, "success": self.success,
            "reason": self.reason, "details": self.details,
        }
