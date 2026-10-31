"""Shared finite-motion admission; F adds a four-axis visual correction domain.

Current measured risk admits a trial. It does not predict post-action loading.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from copy import deepcopy

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


class FiniteTrial:
    def __init__(self, config):
        self.config = deepcopy(config)
        if config.get("version") != "finite-trial-v2":
            raise ValueError("finite trial version must equal finite-trial-v2")
        for key in ("baseline_s", "trial_timeout_s", "observation_s", "confirmation_s",
                    "min_advance_m", "load_drop", "max_wait_s"):
            value = config[key]
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"finite_trial.{key} must be positive")
        if type(config["max_trials"]) is not int or config["max_trials"] < 1:
            raise ValueError("finite_trial.max_trials must be a positive integer")
        for key in ("endpoint_tolerance", "cumulative_budget"):
            a = np.asarray(config[key], float)
            if a.shape != (6,) or not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f"finite_trial.{key} must be a positive six-vector")
        self.phase = "CANDIDATE"
        self.samples = deque()
        self.observations = []
        self.count = 0
        self.cumulative = np.zeros(6)
        self.increment = np.zeros(6)
        self.record = None
        self.last_result = None
        self.wait_started = None

    @property
    def active(self):
        return self.phase != "CANDIDATE"

    @property
    def freezes_advance(self):
        return self.phase in ("TRIAL", "OBSERVE")

    def add_baseline(self, now, wrench, load):
        if self.active:
            return
        if self.wait_started is None:
            self.wait_started = now
        self.samples.append((now, float(load)))
        while len(self.samples) > 1 and self.samples[1][0] <= now - self.config["baseline_s"]:
            self.samples.popleft()

    def ready(self, now):
        return (len(self.samples) >= 2 and
                now - self.samples[0][0] + 1e-12 >= self.config["baseline_s"])

    def start(self, action, now, start_pose, start_bias, step, rate, depth):
        if self.active:
            raise ValueError("cannot replace an active locked trial")
        if action == "Hold" or not self.ready(now):
            raise ValueError("trial requires a non-Hold action and complete baseline")
        delta = action_vector(action) * np.asarray(step, float)
        speeds = np.asarray(rate, float)
        active = np.flatnonzero(delta)
        if len(active) != 1 or speeds.shape != (6,) or speeds[active[0]] <= 0:
            raise ValueError("trial requires one finite correction axis with positive speed")
        if self.count >= self.config["max_trials"]:
            raise ValueError("finite_trial_count_exhausted")
        if np.any(self.cumulative + np.abs(delta) > self.config["cumulative_budget"]):
            raise ValueError("finite_trial_cumulative_budget")
        self.count += 1
        self.cumulative += np.abs(delta)
        self.delta, self.rate = delta.copy(), speeds.copy()
        self.start_pose = np.asarray(start_pose, float).copy()
        self.start_bias = np.asarray(start_bias, float).copy()
        self.increment[:] = 0
        self.observations = []
        self.phase = "TRIAL"
        self.record = {
            "version": "finite-trial-v2", "trial_id": str(self.count), "phase": self.phase,
            "action": action, "outcome": None, "started_at": now,
            "start_pose": self.start_pose.tolist(), "start_bias": self.start_bias.tolist(),
            "total_increment": delta.tolist(), "trial_deadline": now + self.config["trial_timeout_s"],
            "observation_deadline": None, "confirmation_deadline": None,
            "baseline_load": float(np.median([v for _, v in self.samples])),
            "baseline_started_at": self.samples[0][0], "baseline_ended_at": self.samples[-1][0],
            "observation_load": None, "trial_peak": [0.] * 6, "release_clue": False,
            "measured_advance_m": 0., "loads_allowed": True, "start_depth_m": depth,
        }

    def finish(self, outcome, now, reason=""):
        self.record.update(outcome=outcome, ended_at=now, reason=reason, phase=self.phase)
        self.last_result = deepcopy(self.record)
        self.phase = "CANDIDATE"
        self.wait_started = now
        self.samples.clear()
        return self.last_result

    def update(self, now, dt, actual_increment, depth, wrench, load, loads_allowed):
        """Return a stop reason. The command never exceeds the locked increment."""
        if not self.active:
            return ""
        r = self.record
        r["trial_peak"] = np.maximum(r["trial_peak"], np.abs(wrench)).tolist()
        r["loads_allowed"] = bool(r["loads_allowed"] and loads_allowed)
        if not loads_allowed:
            self.finish("invalid", now, "trial_load_admission_lost")
            return "trial_load_admission_lost"
        if self.phase == "TRIAL":
            if now > r["trial_deadline"] + 1e-12:
                self.finish("unconfirmed", now, "finite_trial_timeout")
                return "finite_trial_timeout"
            reached = (np.all(np.abs(self.increment - self.delta) <= 1e-12) and
                       np.all(np.abs(actual_increment - self.delta) <= self.config["endpoint_tolerance"]))
            if reached:
                self.phase = "OBSERVE"
                r["observation_started_at"] = now
                r["observation_deadline"] = now + self.config["observation_s"]
                self.observations.append(float(load))
            elif now >= r["trial_deadline"]:
                self.finish("unconfirmed", now, "finite_trial_timeout")
                return "finite_trial_timeout"
            else:
                remaining = self.delta - self.increment
                effective_dt = min(dt, max(0., r["trial_deadline"] - now))
                amount = np.minimum(np.abs(remaining), self.rate * effective_dt)
                self.increment += np.sign(remaining) * amount
        elif self.phase == "OBSERVE":
            if np.any(np.abs(actual_increment - self.delta) > self.config["endpoint_tolerance"]):
                self.finish("invalid", now, "trial_endpoint_lost")
                return "trial_endpoint_lost"
            if now <= r["observation_deadline"] + 1e-12:
                self.observations.append(float(load))
            if now + 1e-12 >= r["observation_deadline"]:
                r["observation_load"] = float(np.median(self.observations))
                r["release_clue"] = r["baseline_load"] - r["observation_load"] >= self.config["load_drop"]
                r["confirmation_start_depth_m"] = depth
                r["confirmation_deadline"] = now + self.config["confirmation_s"]
                self.phase = "CONFIRM"
        elif self.phase == "CONFIRM":
            r["measured_advance_m"] = float(depth - r["confirmation_start_depth_m"])
            if now > r["confirmation_deadline"] + 1e-12:
                self.finish("no_progress", now, "advance_confirmation_timeout")
            elif r["measured_advance_m"] >= self.config["min_advance_m"]:
                self.finish("effective", now)
            elif now >= r["confirmation_deadline"]:
                self.finish("no_progress", now, "advance_not_confirmed")
        if self.active:
            r["phase"] = self.phase
        return ""

    def snapshot(self):
        if self.active:
            result = deepcopy(self.record)
            result["commanded_increment"] = self.increment.tolist()
            return result
        return {"version": "finite-trial-v2", "phase": self.phase,
                "trials_completed": self.count, "cumulative_increment": self.cumulative.tolist()}
