"""Shared finite-motion admission; F adds a four-axis visual correction domain.

Current measured risk admits a trial. It does not predict post-action loading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import exp_so3


ACTIONS = ("+x", "-x", "+y", "-y", "+Rx", "-Rx", "+Ry", "-Ry", "Hold")
CORRECTION_AXES = (0, 1, 3, 4)
_AXES = (0, 0, 1, 1, 3, 3, 4, 4)


def action_vector(name: str) -> np.ndarray:
    if name not in ACTIONS:
        raise ValueError(f"unknown corrective action: {name}")
    vector = np.zeros(6)
    index = ACTIONS.index(name)
    if index < 8:
        vector[_AXES[index]] = 1.0 if index % 2 == 0 else -1.0
    return vector


@dataclass(frozen=True)
class SynergyResult:
    allowed: tuple[str, ...]
    reasons: dict[str, str]
    valid: bool


class SynergyFilter:
    def __init__(self, config: dict, risk_model=None):
        if config.get("rules_version") != "finite-trial-v2" or "force_response" in config:
            raise ValueError("legacy synergy configuration requires explicit finite-trial-v2 migration")
        self.config = config
        self.risk_model = risk_model
        self.validated = config.get("validated", False) is True
        self.max_offset = self._six(config["max_offset"], "max_offset")
        self.action_step = self._six(config["action_step"], "action_step")
        if np.any(self.max_offset <= 0) or np.any(self.action_step < 0):
            raise ValueError("synergy max_offset must be positive and action_step nonnegative")
        if np.any(self.action_step[[0, 1, 3, 4]] <= 0):
            raise ValueError("all four active correction axes need a positive action step")
        self.pins = []
        for pin in config.get("pins", []):
            point = np.asarray(pin["point_m"], dtype=float)
            clearance, engagement = float(pin["clearance_m"]), float(pin["engagement_m"])
            if (point.shape != (3,) or not np.all(np.isfinite(point))
                    or not np.isfinite(clearance) or clearance <= 0
                    or not np.isfinite(engagement) or engagement < 0):
                raise ValueError("pin geometry needs finite point, positive clearance and nonnegative engagement")
            self.pins.append((point.copy(), clearance, engagement))
        if np.any(self.action_step[[2, 5]] != 0):
            raise ValueError("finite corrections require zero z/Rz steps")
        self.max_risk = float(config["admission_max_risk"])
        if not np.isfinite(self.max_risk) or not 0 < self.max_risk <= 1:
            raise ValueError("admission_max_risk must be in (0,1]")

    @staticmethod
    def _six(value, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite six-vector")
        return value.copy()

    @staticmethod
    def _unavailable(reason: str) -> SynergyResult:
        return SynergyResult(("Hold",), {action: reason for action in ACTIONS[:-1]}, False)

    def _geometry_valid(self, offset: np.ndarray) -> bool:
        rotation = exp_so3(offset[3:])
        for point, clearance, engagement in self.pins:
            # A finite straight pin's lateral displacement is affine along its
            # axis; the norm is convex, so its maximum occurs at an endpoint.
            for z in (0.0, engagement):
                endpoint = point + np.array([0.0, 0.0, z])
                shifted = offset[:3] + rotation @ endpoint - endpoint
                if np.linalg.norm(shifted[:2]) > clearance:
                    return False
        return True

    def endpoint_reason(self, offset, vision_budget=None):
        offset = self._six(offset, "offset")
        if np.any(np.abs(offset) > self.max_offset + 1e-12):
            return "travel_or_key_budget"
        if not self._geometry_valid(offset):
            return "pin_clearance_or_key_geometry"
        if vision_budget is not None:
            budget = np.asarray(vision_budget, float)
            if budget.shape != (4,) or not np.isfinite(budget).all() or np.any(budget <= 0):
                return "invalid_visual_budget"
            if np.any(np.abs(offset[list(CORRECTION_AXES)]) > budget + 1e-12):
                return "visual_correction_budget"
        return ""

    def allowed(self, pose_offset6, vision_budget4, risk6, wrench6,
                phase: str = "INSERT") -> SynergyResult:
        try:
            offset = self._six(pose_offset6, "pose_offset")
            wrench = self._six(wrench6, "wrench")
            reason = self.endpoint_reason(offset, vision_budget4)
        except (TypeError, ValueError):
            return self._unavailable("invalid_input")
        if phase != "INSERT":
            return self._unavailable("phase_disallows_correction")
        if not self.validated or not self.pins:
            return self._unavailable("geometry_not_validated")
        if reason:
            return self._unavailable("current_" + reason)
        if self.risk_model is not None:
            current = self.risk_model.evaluate(wrench)
            try:
                risk = self._six(risk6, "risk")
            except (TypeError, ValueError):
                return self._unavailable("invalid_risk")
            if (not current.valid or not self.risk_model.calibrated or
                    not np.allclose(current.values, risk, atol=1e-7, rtol=0)):
                return self._unavailable("risk_input_inconsistent")
            if np.max(risk) > self.max_risk:
                return SynergyResult(("Hold",), dict.fromkeys(ACTIONS[:-1], "current_risk_admission"), True)
        accepted, reasons = [], {}
        for action in ACTIONS[:-1]:
            proposed = offset + action_vector(action) * self.action_step
            reason = self.endpoint_reason(proposed, vision_budget4)
            if reason:
                reasons[action] = reason
            else:
                accepted.append(action)
        accepted.append("Hold")
        return SynergyResult(tuple(accepted), reasons, True)
