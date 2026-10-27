"""Conservative implementation of thesis section 4.5.3, A_valid.

The thesis specifies the meaning of g_v/g_g/g_f, not numeric calibrated rules.
Here g_v uses a calibrated SI correction envelope; g_g checks both endpoints of
each engaged pin; g_f uses measured per-action wrench responses. The last rule
is a local calibrated approximation, not a guarantee about future contact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import exp_so3


ACTIONS = ("+x", "-x", "+y", "-y", "+Rx", "-Rx", "+Ry", "-Ry", "Hold")
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
        response = config.get("force_response")
        self.response = None
        self.response_version = None
        self.response_risk_version = None
        self.response_step = None
        if response is not None:
            if not isinstance(response, dict):
                raise ValueError("force_response needs matrix, version, risk_version and calibration metadata")
            matrix = np.asarray(response["delta_wrench"], dtype=float)
            if matrix.shape != (6, 8) or not np.all(np.isfinite(matrix)):
                raise ValueError("force response delta_wrench must have shape [6,8]")
            if response.get("calibrated", False) is True:
                self.response = matrix.copy()
            self.response_version = response.get("version")
            self.response_risk_version = response.get("risk_version")
            self.response_step = self._six(response.get("action_step", []), "force response action_step")
        self.risk_tolerance = float(config.get("risk_tolerance", 1e-9))
        if not np.isfinite(self.risk_tolerance) or not 0 <= self.risk_tolerance <= 1e-3:
            raise ValueError("risk_tolerance must be a small numerical tolerance, in [0,0.001]")

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

    def allowed(self, pose_offset6, vision_budget6, risk6, wrench6,
                phase: str = "INSERT") -> SynergyResult:
        try:
            offset = self._six(pose_offset6, "pose_offset")
            budget = self._six(vision_budget6, "vision_budget")
            risk = self._six(risk6, "risk")
            wrench = self._six(wrench6, "wrench")
        except (TypeError, ValueError):
            return self._unavailable("invalid_input")
        if phase != "INSERT":
            return self._unavailable("phase_disallows_correction")
        if np.any(budget <= 0) or np.any(risk < 0) or np.any(risk > 1):
            return self._unavailable("invalid_budget_or_risk")
        if not self.validated or not self.pins:
            return self._unavailable("geometry_or_visual_envelope_not_validated")
        if (self.response is None or not self.response_version
                or self.risk_model is None or not self.risk_model.calibrated
                or self.response_risk_version != self.risk_model.version
                or self.response_step is None
                or not np.allclose(self.response_step, self.action_step, atol=1e-12, rtol=0)):
            return self._unavailable("force_response_not_validated_for_this_domain_and_step")
        current = self.risk_model.evaluate(wrench)
        if not current.valid or not np.allclose(current.values, risk, atol=1e-7, rtol=0):
            return self._unavailable("risk_input_inconsistent")
        envelope = np.minimum(budget, self.max_offset)
        if np.any(np.abs(offset) > envelope):
            return self._unavailable("current_pose_outside_correction_envelope")
        accepted, reasons = [], {}
        for index, action in enumerate(ACTIONS[:-1]):
            proposed = offset + action_vector(action) * self.action_step
            if np.any(np.abs(proposed) > envelope):
                reasons[action] = "visual_or_travel_budget"
            elif not self._geometry_valid(proposed):
                reasons[action] = "pin_clearance_or_key_geometry"
            else:
                predicted = self.risk_model.evaluate(wrench + self.response[:, index])
                if not predicted.valid:
                    reasons[action] = "predicted_wrench_outside_domain"
                elif np.any(predicted.values > current.values + self.risk_tolerance):
                    reasons[action] = "predicted_risk_increase"
                else:
                    accepted.append(action)
        accepted.append("Hold")
        return SynergyResult(tuple(accepted), reasons, True)
