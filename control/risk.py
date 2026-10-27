"""Thesis equations (3-16)--(3-18): signed margins of a bounded wrench domain.

This is an engineering margin, not a probability of jamming. Calibration
readiness is separate from mathematical validity, allowing explicit simulation
domains to be tested without representing them as measured hardware limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np


@dataclass(frozen=True)
class RiskResult:
    valid: bool
    values: np.ndarray
    outside: bool
    reason: str


class EngineeringRisk:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.version = str(config.get("version", ""))
        self.calibrated = config.get("calibrated", False) is True
        self.scale = np.asarray(config["scale"], dtype=float)
        self.A = np.asarray(config["A"], dtype=float)
        self.b = np.asarray(config["b"], dtype=float)
        self.reference = np.asarray(config["reference"], dtype=float)
        if not self.version:
            raise ValueError("engineering domain needs a version")
        if (self.scale.shape != (6,) or not np.all(np.isfinite(self.scale))
                or np.any(self.scale <= 0)):
            raise ValueError("risk scale must contain six positive finite values")
        if (self.A.ndim != 2 or self.A.shape[1] != 6 or self.A.shape[0] < 12
                or self.b.shape != (self.A.shape[0],)
                or not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.b))):
            raise ValueError("domain needs finite A[m,6] and b[m], including signed axis bounds")
        if self.reference.shape != (6,) or not np.all(np.isfinite(self.reference)):
            raise ValueError("domain reference must be a finite normalized six-vector")
        if np.any(np.linalg.norm(self.A, axis=1) <= 1e-12):
            raise ValueError("zero domain rows are not allowed")
        # Explicit positive/negative axis bounds guarantee a bounded domain
        # without an optional linear-program solver. Extra coupling rows are
        # allowed, but cannot replace the documented independent load limits.
        nonzero = np.abs(self.A) > 1e-12
        for axis in range(6):
            axis_rows = np.sum(nonzero, axis=1) == 1
            column = self.A[axis_rows, axis]
            if not (np.any(column > 1e-12) and np.any(column < -1e-12)):
                raise ValueError(f"missing explicit positive/negative bound for axis {axis}")
        slack = self.b - self.A @ self.reference
        if np.any(slack <= 1e-10):
            raise ValueError("domain reference must be strictly inside every constraint")
        self._reference_margins = self._margins(slack)

    def _margins(self, slack: np.ndarray) -> np.ndarray:
        margins = np.empty((2, 6))
        for side, sign in enumerate((-1.0, 1.0)):
            for axis in range(6):
                coefficient = sign * self.A[:, axis]
                active = coefficient > 1e-12
                margins[side, axis] = np.min(slack[active] / coefficient[active])
        return margins

    def evaluate(self, wrench) -> RiskResult:
        try:
            wrench = np.asarray(wrench, dtype=float)
        except (TypeError, ValueError):
            return RiskResult(False, np.ones(6), False, "invalid_wrench")
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            return RiskResult(False, np.ones(6), False, "invalid_wrench")
        normalized = wrench / self.scale
        slack = self.b - self.A @ normalized
        if not np.all(np.isfinite(slack)):
            return RiskResult(False, np.ones(6), False, "invalid_wrench")
        if np.any(slack < -1e-10):
            return RiskResult(False, np.ones(6), True, "outside_engineering_domain")
        margins = self._margins(np.maximum(slack, 0.0))
        values = np.max(np.clip(1.0 - margins / self._reference_margins, 0.0, 1.0), axis=0)
        return RiskResult(True, values, False, "ok")
