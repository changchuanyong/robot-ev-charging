"""Feedback-driven insertion supervisor (thesis S0-S5 and SE).

There are no sockets in this module. Decisions are proposals; an execution
adapter must acknowledge commands/stops and supply independent actual feedback.
"""
from collections import deque
from copy import deepcopy

import numpy as np

from .admittance import Admittance, advance_scale
from .config import METHODS, require_config
from .finite_trial import FiniteTrial
from .geometry import invert, log_so3, offset_pose, pose_error, validate_transform
from .risk import EngineeringRisk
from .synergy import ACTIONS, CORRECTION_AXES, SynergyFilter, action_vector
from .types import Decision, RobotSample
from .wrench import WrenchProcessor


class InsertionController:
    def __init__(self, config, policy=None, *, probe_action=None):
        self.config = require_config(config)
        self.probe_action = probe_action
        if config.get("supervised_probe") is not None:
            from .probe import OperatorProbePolicy, prepare_probe
            if probe_action is None or prepare_probe(config, probe_action) != config:
                raise ValueError("supervised probe configuration requires explicit probe_action")
            if not isinstance(policy, OperatorProbePolicy) or policy.requested_action != probe_action:
                raise ValueError("probe policy must match the explicitly selected direction")
        elif probe_action is not None:
            raise ValueError("probe_action requires a separately prepared calibration configuration")
        self.flags = METHODS[self.config["method"]]
        self.policy = policy
        if self.flags["learning"] and (policy is None or not policy.model_ready):
            raise ValueError("selected method requires a ready, trained GRU checkpoint")
        self.admittance = Admittance(config["admittance"])
        self.risk_model = EngineeringRisk(config["risk"]) if self.flags["risk"] else None
        self.synergy = SynergyFilter(config["synergy"], self.risk_model) if self.flags["learning"] else None
        self.trial = FiniteTrial(config["finite_trial"]) if self.flags["learning"] else None
        self.wrench_processor = WrenchProcessor(config["wrench"])
        self.state = "VISION"
        self.reason = ""
        self.entered_at = self.started_at = self.last_at = None
        self.last_seq = None
        self.last_sample_time = None
        self.last_pose = self.last_command = None
        self.command_velocity = np.zeros(6)
        self.actual_velocity = np.zeros(6)
        self.R_base_task = np.eye(3)
        self.target = self.pre_pose = self.nominal = self.contact_pose = None
        self.bias = np.zeros(6)
        self.reference_pose = None
        self.vision_budget = None
        self.frozen_vision = None
        self.history = deque()
        self.progress = deque()
        self.contact_since = self.release_since = self.complete_since = self.pause_since = None
        self.handover_contact_since = None
        self.completion_depths = deque()
        self.stop_confirmed = False
        self.retreat_start = None
        self.retreat_candidate = None
        self.retreat_stationary_since = None
        self.current_wrench = np.zeros(6)
        self.last_retreat_load = None
        self.nominal_distance = 0.
        self.action = "Hold"
        self.last_action_time = None
        self.last_action_prediction = None
        self.events = []

    def _transition(self, state, now, reason=""):
        previous = self.state
        self.state, self.entered_at = state, now
        self.events.append({"at": now, "from": previous, "to": state, "reason": reason})

    def _fail(self, reason, now, **details):
        if self.trial and self.trial.active:
            self.trial.record["trial_peak"] = np.maximum(
                self.trial.record["trial_peak"], np.abs(self.current_wrench)).tolist()
            self.trial.finish("invalid", now, reason)
            details.update(finite_trial=self.trial.snapshot(), trial_result=self.trial.last_result)
        if self.state != "STOPPED":
            self._transition("STOPPED", now, reason)
        self.reason = reason
        self.action = "Hold"
        return Decision("STOPPED", stop=True, finished=True, reason=reason, details=details)

    def acknowledge_stop(self):
        """Call only after the driver confirms its explicit stop operation."""
        self.stop_confirmed = True

    def request_retreat(self, sample, now):
        """An explicit, conditional recovery after a verified stop.

        Data/overload/unknown-execution faults cannot be automatically retried.
        One bounded retreat ends the attempt; a new attempt needs fresh vision.
        """
        t = self.config["task"]
        timing, limits = self.config["timing"], self.config["limits"]
        if (self.state != "STOPPED" or not self.stop_confirmed or not t["retreat_enabled"]
                or self.reason not in {"progress_stall", "insert_travel_exhausted", "pause_timeout"}):
            return False
        try:
            if (not sample.valid or not np.isfinite(now) or not np.isfinite(sample.timestamp)
                    or isinstance(sample.seq, (bool, np.bool_)) or not isinstance(sample.seq, (int, np.integer))
                    or not -timing["future_tolerance"] <= now - sample.timestamp <= timing["max_sample_age"]
                    or self.last_seq is None or sample.seq <= self.last_seq
                    or sample.timestamp <= self.last_sample_time):
                raise ValueError("invalid recovery feedback")
            pose = validate_transform(sample.pose)
            if np.any(pose[:3, 3] < limits["workspace_min"]) or np.any(pose[:3, 3] > limits["workspace_max"]):
                raise ValueError("recovery pose outside workspace")
            # Use a separate processor so a recovery check cannot alter the
            # frozen bias or the contact-stage historical filter.
            current, _ = WrenchProcessor(self.config["wrench"]).process(sample, self.R_base_task, self.config["period"])
            if (np.any(np.abs(current) >= limits["wrench"])
                    or np.any(np.abs(current) > t["retreat_load_limit"])):
                raise ValueError("recovery load not permitted")
            target = pose[:3, 3] - self.R_base_task[:, 2] * t["retreat_distance"]
            if np.any(target < limits["workspace_min"]) or np.any(target > limits["workspace_max"]):
                raise ValueError("retreat target outside workspace")
        except (ValueError, TypeError, KeyError):
            self.retreat_candidate = None
            self.retreat_stationary_since = None
            return False
        previous = self.retreat_candidate
        candidate = {"seq": sample.seq, "timestamp": sample.timestamp, "at": now, "pose": pose.copy()}
        if previous is None:
            self.retreat_candidate = candidate
            self.retreat_stationary_since = now
            return False
        elapsed = sample.timestamp - previous["timestamp"]
        if (sample.seq <= previous["seq"] or elapsed <= 0
                or elapsed > timing["max_period"] or now <= previous["at"]
                or now - previous["at"] > timing["max_period"]):
            self.retreat_candidate = None
            self.retreat_stationary_since = None
            return False
        velocity = self._task_delta(pose, previous["pose"]) / elapsed
        self.retreat_candidate = candidate
        if np.any(np.abs(velocity) > t["retreat_stop_velocity"]):
            self.retreat_stationary_since = None
            return False
        if self.retreat_stationary_since is None:
            self.retreat_stationary_since = now
        if now - self.retreat_stationary_since < t["retreat_stationary_duration"]:
            return False
        self.retreat_start = pose.copy()
        self.last_command = self.retreat_start.copy()
        self.command_velocity[:] = 0
        self.current_wrench = current.copy()
        self.actual_velocity = velocity
        self.last_at, self.last_seq = now, sample.seq
        self.last_sample_time, self.last_pose = sample.timestamp, pose.copy()
        self.wrench_processor.reset()
        self.last_retreat_load = float(np.linalg.norm(current))
        self._transition("RETREAT", now, "explicit_conditional_retreat")
        self.stop_confirmed = False
        return True

    def _task_delta(self, reference, actual):
        delta = pose_error(reference, actual)
        return np.r_[self.R_base_task.T @ delta[:3], self.R_base_task.T @ delta[3:]]

    def _depth(self, pose):
        if self.contact_pose is None:
            return 0.
        return float(self.R_base_task[:, 2] @ (pose[:3, 3] - self.contact_pose[:3, 3]))

    def _limit(self, desired, dt, *, commit=True):
        limits = self.config["limits"]
        desired = validate_transform(desired)
        # Bounds are checked before interpolation, so an unreachable target is
        # never hidden behind a series of small apparently valid commands.
        if np.any(desired[:3, 3] < limits["workspace_min"]) or np.any(desired[:3, 3] > limits["workspace_max"]):
            raise ValueError("workspace_limit")
        delta = self._task_delta(desired, self.last_command)
        desired_velocity = delta / dt
        velocity = np.clip(desired_velocity, -np.asarray(limits["velocity"]), limits["velocity"])
        if self.trial and self.trial.freezes_advance:
            trial_rate = np.asarray(self.config["learning"]["bias_rate"])
            velocity = np.clip(velocity, -trial_rate, trial_rate)
        acceleration = np.asarray(limits["acceleration"])
        velocity = np.clip(velocity, self.command_velocity - acceleration * dt,
                           self.command_velocity + acceleration * dt)
        step = np.clip(velocity * dt, -np.asarray(limits["step"]), limits["step"])
        if self.trial and self.trial.freezes_advance:
            # A bounded trial never commands beyond its locked endpoint, even
            # if the previous velocity would overshoot while decelerating.
            step = np.where(step * delta >= 0,
                            np.sign(delta) * np.minimum(np.abs(step), np.abs(delta)), 0.)
        # Do not jump through a new target when decelerating. A reversal can
        # take several periods; it is recorded as a command limit.
        result = offset_pose(self.last_command, step, self.R_base_task)
        if np.any(result[:3, 3] < limits["workspace_min"]) or np.any(result[:3, 3] > limits["workspace_max"]):
            raise ValueError("workspace_limit")
        limited = not np.allclose(step, delta, rtol=0, atol=1e-10)
        if commit:
            self.command_velocity = step / dt
            self.last_command = result.copy()
        return result, limited

    def _accept_vision(self, observation, now):
        if not observation or observation.get("valid") is not True:
            return "vision_invalid"
        c = self.config
        try:
            if observation.get("schema_version") != 1 or observation.get("clock") != "host_monotonic":
                return "vision_schema_or_clock"
            timestamp = float(observation["captured_at"])
            received, processed = float(observation["received_at"]), float(observation["processed_at"])
            if (not np.isfinite([timestamp, received, processed]).all()
                    or not timestamp <= received <= processed <= now + c["timing"]["future_tolerance"]):
                return "vision_timestamp_order"
            if not np.isfinite(timestamp) or not -c["timing"]["future_tolerance"] <= now - timestamp <= c["timing"]["max_vision_age"]:
                return "vision_stale"
            if observation["target_id"] != c["target_id"]:
                return "target_identity_mismatch"
            if not observation.get("frame_id") or observation.get("translation_unit") != "m":
                return "vision_units_or_identity"
            if observation.get("camera_frame") != "camera_color_optical":
                return "vision_camera_frame"
            if observation.get("calibration_version") != c["calibration"]["expected_camera_version"]:
                return "vision_camera_calibration_version"
            if observation.get("camera_serial") != c["calibration"]["expected_camera_serial"]:
                return "vision_camera_serial"
            if not c["simulation_only"] and (observation.get("source") != "realsense" or observation.get("time_sync_valid") is not True):
                return "vision_source_or_clock_unverified"
            quality = observation["quality"]
            experiment = c.get("experiment", {})
            if experiment.get("formal") is True:
                expected = experiment.get("geometry_reference")
                actual = quality.get("geometry_reference")
                if (not isinstance(expected, dict) or not isinstance(actual, dict)
                        or actual.get("status") != "measured_print" or actual.get("validated") is not True
                        or not expected.get("version") or not expected.get("sha256")
                        or actual.get("version") != expected["version"]
                        or actual.get("source_sha256") != expected["sha256"]):
                    return "formal_print_geometry_mismatch"
            if quality.get("camera_calibrated") is not True or quality.get("all_in_front") is not True:
                return "vision_calibration_or_depth"
            metrics = np.array([quality["reprojection_rms_px"], quality["max_error_px"], quality["inlier_count"]], float)
            if not np.isfinite(metrics).all() or np.any(metrics < 0):
                return "vision_metrics_invalid"
            if metrics[0] > c["vision"]["max_rms_px"] or metrics[1] > c["vision"]["max_error_px"] or metrics[2] < c["vision"]["min_inliers"]:
                return "vision_quality_rejected"
            candidates = []
            for candidate in self.history:
                earliest = float(candidate.timestamp)
                metadata = candidate.metadata
                latest = float(metadata.get("received_at", earliest))
                if not c["simulation_only"]:
                    if (metadata.get("timestamp_source") != "host_receive_minus_validated_transport_bound"
                            or "received_at" not in metadata or "transport_latency_bound_s" not in metadata):
                        continue
                if "transport_latency_bound_s" in metadata:
                    bound = float(metadata["transport_latency_bound_s"])
                    if (not np.isfinite(bound) or bound < 0
                            or not np.isclose(latest - earliest, bound, atol=1e-9, rtol=1e-7)):
                        continue
                if (not np.isfinite([earliest, latest]).all() or latest < earliest
                        or latest > now + c["timing"]["future_tolerance"]):
                    continue
                candidates.append((max(abs(earliest - timestamp), abs(latest - timestamp)), candidate))
            if not candidates:
                return "vision_robot_time_interval_unverified"
            worst_skew, paired = min(candidates, key=lambda pair: pair[0])
            if worst_skew > c["timing"].get("max_pose_skew", c["period"]):
                return "vision_robot_pair_not_synchronized"
            cal = c["calibration"]
            if cal.get("insertion_axis_verified") is not True:
                return "insertion_axis_unverified"
            target_task = np.eye(4)
            target_task[:3, :3] = cal["R_target_task"]
            validate_transform(target_task)
            base_flange = validate_transform(paired.pose) @ invert(cal["T_flange_tool"])
            base_target = base_flange @ validate_transform(cal["T_flange_camera"]) @ validate_transform(observation["T_camera_target"])
            target = base_target @ validate_transform(cal["T_target_plug"])
            # PnP's model normal need not point into the connector. The
            # physically verified task axes are a separate calibrated rotation
            # from both the camera/model axes and the plug mating orientation.
            R_base_task = base_target[:3, :3] @ target_task[:3, :3]
            pre_pose = target.copy()
            pre_pose[:3, 3] -= R_base_task[:, 2] * c["task"]["pre_distance"]
            for p in (pre_pose[:3, 3], target[:3, 3]):
                if np.any(p < c["limits"]["workspace_min"]) or np.any(p > c["limits"]["workspace_max"]):
                    return "vision_target_outside_workspace"
            if self.flags["synergy"]:
                budget = np.asarray(quality.get("visual_budget"), float)
                if (budget.shape != (4,) or not np.isfinite(budget).all() or np.any(budget <= 0)
                        or quality.get("budget_version") != cal["expected_budget_version"]):
                    return "visual_correction_budget_unavailable"
                cal_budget = np.r_[cal["position_budget"], cal["rotation_budget"]]
                if cal_budget.shape != (6,) or not np.isfinite(cal_budget).all() or np.any(cal_budget <= 0):
                    return "visual_budget_calibration_invalid"
                vision_budget = np.minimum(budget, cal_budget[list(CORRECTION_AXES)])
            else:
                vision_budget = None
            # Rejected observations must not rotate the force-monitoring axes
            # or partially replace a previous target/reference.
            self.target, self.pre_pose = target, pre_pose
            self.R_base_task, self.vision_budget = R_base_task, vision_budget
            self.frozen_vision = deepcopy(observation)
        except (KeyError, ValueError, TypeError):
            return "vision_contract_invalid"
        return ""

    def step(self, sample: RobotSample, now: float, observation=None):
        if self.state == "COMPLETE":
            return Decision(self.state, stop=True, finished=True, success=True, reason="mechanical_insertion_complete")
        if self.state == "STOPPED":
            return Decision(self.state, stop=True, finished=True, reason=self.reason)
        timing, task, limits = self.config["timing"], self.config["task"], self.config["limits"]
        if not np.isfinite(now) or not np.isfinite(sample.timestamp):
            return self._fail("invalid_timestamp", now)
        if self.started_at is None:
            self.started_at = self.entered_at = now
        dt = self.config["period"] if self.last_at is None else now - self.last_at
        if dt <= 0 or dt > min(timing["max_period"], self.admittance.max_dt):
            return self._fail("control_period_invalid", now, dt=dt)
        if not sample.valid:
            return self._fail("feedback_invalid", now, source_reason=sample.reason)
        if isinstance(sample.seq, (bool, np.bool_)) or not isinstance(sample.seq, (int, np.integer)) or sample.seq < 0:
            return self._fail("invalid_feedback_sequence", now)
        if not -timing["future_tolerance"] <= now - sample.timestamp <= timing["max_sample_age"]:
            return self._fail("feedback_stale", now)
        if self.last_seq is not None and (sample.seq <= self.last_seq or sample.timestamp <= self.last_sample_time):
            return self._fail("feedback_repeated_or_reordered", now)
        try:
            pose = validate_transform(sample.pose)
            if sample.joints and not np.isfinite(np.asarray(sample.joints, float)).all():
                raise ValueError("nonfinite joints")
            if self.last_pose is not None:
                self.actual_velocity = self._task_delta(pose, self.last_pose) / (sample.timestamp - self.last_sample_time)
            if self.last_command is None:
                self.last_command = pose.copy()
            elif np.any(np.abs(self._task_delta(self.last_command, pose)) > limits["tracking_error"]):
                return self._fail("tracking_error", now)
            if np.any(pose[:3, 3] < limits["workspace_min"]) or np.any(pose[:3, 3] > limits["workspace_max"]):
                return self._fail("actual_workspace_limit", now)
            self.history.append(sample)
            while self.history and now - self.history[0].timestamp > timing["max_vision_age"] + timing["max_period"]:
                self.history.popleft()
            # Freeze the task axes before translating and monitoring the wrench.
            vision_reason = ""
            if self.state == "VISION":
                vision_reason = self._accept_vision(observation, now)
            current, filtered = self.wrench_processor.process(sample, self.R_base_task, dt)
        except (ValueError, TypeError, KeyError) as exc:
            return self._fail("feedback_contract_invalid", now, error=str(exc))
        self.last_at, self.last_seq = now, sample.seq
        self.last_sample_time, self.last_pose = sample.timestamp, pose.copy()
        self.current_wrench = current.copy()
        if self.trial and self.trial.active:
            self.trial.record["trial_peak"] = np.maximum(
                self.trial.record["trial_peak"], np.abs(current)).tolist()
        if np.any(np.abs(current) >= limits["wrench"]):
            return self._fail("hard_wrench_limit", now, wrench=current.tolist())
        if now - self.started_at > timing["timeout"]:
            return self._fail("task_timeout", now)
        details = {"dt": dt, "wrench": current.tolist(), "filtered_wrench": filtered.tolist(),
                   "depth": self._depth(pose), "action": "Hold", "risk": None,
                   "risk_valid": None, "limited": False, "advance_requested": False,
                   "advance_scale": 0., "bias": self.bias.tolist()}
        if self.state == "VISION":
            if vision_reason:
                details["vision_rejection"] = vision_reason
                return Decision(self.state, reason=vision_reason, details=details)
            self._transition("PREALIGN", now)
        desired = self.last_command.copy()
        load = np.array([abs(current[2]), np.linalg.norm(current[:2]), np.linalg.norm(current[3:5])])
        if self.state == "PREALIGN":
            desired = self.pre_pose
            error = self._task_delta(self.pre_pose, pose)
            if np.linalg.norm(error[:3]) <= task["pre_position_tolerance"] and np.linalg.norm(error[3:]) <= task["pre_angle_tolerance"]:
                self.nominal = pose.copy()
                self.reference_pose = pose.copy()
                self._transition("APPROACH", now)
        elif self.state == "APPROACH":
            if np.any(load >= task["contact_enter"]):
                if self.contact_since is None:
                    self.contact_since = now
                if now - self.contact_since >= task["contact_duration"]:
                    self.contact_pose = pose.copy()
                    self.nominal = pose.copy()
                    self.bias[:] = 0
                    self.nominal_distance = 0.
                    self.admittance.reset(velocity=self.actual_velocity)
                    if self.policy:
                        self.policy.reset()
                    self.last_action_time = self.last_action_prediction = None
                    self.handover_contact_since = now
                    self._transition("CONTACT", now)
                    desired = pose.copy()
            else:
                self.contact_since = None
            if self.state == "APPROACH":
                advance = np.zeros(6)
                advance[2] = task["approach_speed"] * dt
                desired = offset_pose(self.last_command, advance, self.R_base_task)
                travel = self._task_delta(pose, self.reference_pose)[2]
                if travel > task["max_approach_travel"]:
                    return self._fail("contact_not_found", now, **details)
        elif self.state == "CONTACT":
            desired = self.nominal.copy()
            if np.all(load < task["contact_exit"]):
                self.handover_contact_since = None
                if self.release_since is None:
                    self.release_since = now
                if now - self.release_since > task["release_duration"]:
                    return self._fail("contact_lost_during_handover", now, **details)
            else:
                self.release_since = None
                if self.handover_contact_since is None:
                    self.handover_contact_since = now
            if (self.handover_contact_since is not None
                    and now - self.handover_contact_since >= task["confirm_duration"]
                    and np.all(np.abs(self.actual_velocity) <= task["handover_velocity"])):
                # The measured stop, rather than a fixed delay alone, completes
                # the handover. Reference and internal displacement start here.
                self.nominal = pose.copy()
                self.admittance.reset()
                self._transition("INSERT", now)
        elif self.state == "INSERT":
            decision = self._insert(pose, current, filtered, now, dt, details)
            if isinstance(decision, Decision):
                return decision
            desired = decision
        elif self.state == "RETREAT":
            norm = float(np.linalg.norm(current))
            if np.any(np.abs(current) > task["retreat_load_limit"]) or norm > self.last_retreat_load + .02:
                return self._fail("retreat_load_increased", now, **details)
            self.last_retreat_load = norm
            distance = -self._task_delta(pose, self.retreat_start)[2]
            if distance >= task["retreat_distance"]:
                return self._fail("retreated_attempt_ended", now, **details)
            increment = np.zeros(6)
            increment[2] = -task["retreat_speed"] * dt
            desired = offset_pose(self.last_command, increment, self.R_base_task)
        try:
            command, limited = self._limit(desired, dt)
        except ValueError as exc:
            return self._fail(str(exc), now, **details)
        if self.state == "INSERT" and limited and not (self.trial and self.trial.active):
            # Do not keep integrating a hidden reference or learning offset
            # behind a clipped command. Keep the feasible command as nominal.
            self.nominal = command.copy()
            self.bias[:] = 0
            self.admittance.sync(np.zeros(6), np.zeros(6))
            details["anti_windup"] = "rebase_to_feasible_command"
        details.update(limited=limited, nominal=None if self.nominal is None else self.nominal.tolist(),
                       bias=self.bias.tolist(), admittance=self.admittance.displacement.tolist(),
                       admittance_velocity=self.admittance.velocity.tolist(), damping=self.admittance.damping.tolist())
        return Decision(self.state, command=command, details=details)

    def _insert(self, pose, current, filtered, now, dt, details):
        task, limits, scheduling = self.config["task"], self.config["limits"], self.config["scheduling"]
        risk = None
        if self.risk_model:
            result = self.risk_model.evaluate(current)
            details.update(risk=result.values.tolist(), risk_valid=result.valid, risk_reason=result.reason)
            if not result.valid:
                return self._fail("risk_domain_invalid", now, **details)
            risk = result.values
        force_scale, lateral_scale, tilt_scale, twist_scale = scheduling["load_scales"]
        chi = max(abs(current[2]) / force_scale, np.linalg.norm(current[:2]) / lateral_scale,
                  np.linalg.norm(current[3:5]) / tilt_scale, abs(current[5]) / twist_scale)
        rho = max(chi, risk[2] if risk is not None else 0.)
        # The independent load supervisor remains on even in fixed baseline A.
        scale = advance_scale(rho, scheduling["low"], scheduling["high"]) if self.flags["risk"] else float(chi < scheduling["high"])
        if self.trial and (self.trial.freezes_advance or
                           (self.probe_action is not None and not self.trial.active) or
                           (not self.trial.active and not self.trial.ready(now))):
            scale = 0.
        depth = self._depth(pose)
        attitude = np.linalg.norm(log_so3(self.target[:3, :3] @ pose[:3, :3].T))
        at_goal = depth >= task["depth"] - task["depth_tolerance"] and attitude <= task["attitude_tolerance"]
        if self.probe_action is not None and at_goal and not self.trial.active:
            return self._fail("probe_start_already_at_goal", now, **details)
        advance = np.zeros(6)
        advance[2] = task["insert_speed"] * scale * dt
        proposed_nominal = offset_pose(self.nominal, advance, self.R_base_task)
        correction = self.admittance.displacement.copy()
        if not at_goal and not (self.trial and self.trial.freezes_advance):
            correction = self.admittance.step(current, dt, risk if self.flags["risk"] else None)
        base_desired = self.last_command.copy() if at_goal else offset_pose(proposed_nominal, self.bias + correction, self.R_base_task)
        allowed, rejections = ACTIONS, {}
        probabilities = self.policy.update(filtered, now, self.config["target_id"]) if self.policy else None
        if self.policy:
            details.update(probabilities=None if probabilities is None else np.asarray(probabilities).tolist(),
                           decision_policy_source=getattr(self.policy, "source", "trained_gru"),
                           learning_reason=self.policy.last_reason,
                           inference_seconds=self.policy.last_inference_seconds)
        if self.trial:
            self.trial.add_baseline(now, filtered, chi)
            actual_offset = self._task_delta(pose, self.target)
            actual_offset[2] = 0.
            budget = self.vision_budget if self.flags["synergy"] else None
            base_offset = self._task_delta(self.last_command if self.trial.active else base_desired, self.target)
            base_offset[2] = 0.
            permit = self.synergy.allowed(base_offset, budget, risk, current)
            allowed, rejections = list(permit.allowed), dict(permit.reasons)
            details["synergy_valid"] = permit.valid
            if not permit.valid:
                details["action_rejections"] = rejections
                return self._fail("synergy_constraints_invalid", now, **details)
            actual_reason = self.synergy.endpoint_reason(actual_offset, budget)
            if actual_reason:
                return self._fail("actual_" + actual_reason, now, **details)
            loads_allowed = chi < scheduling["high"] and (risk is None or np.max(risk) <= self.synergy.max_risk)
            was_phase = self.trial.phase
            if self.trial.active:
                endpoint = offset_pose(self.trial.start_pose, self.trial.delta, self.R_base_task)
                endpoint_offset = self._task_delta(endpoint, self.target)
                endpoint_offset[2] = 0.
                reason = self.synergy.endpoint_reason(endpoint_offset, budget)
                if (np.any(np.abs(self._task_delta(endpoint, self.contact_pose)) > limits["total_offset"])
                        or np.any(endpoint[:3, 3] < limits["workspace_min"])
                        or np.any(endpoint[:3, 3] > limits["workspace_max"])):
                    reason = "travel_or_workspace_limit"
                if reason:
                    self.trial.finish("invalid", now, reason)
                    details["trial_result"] = self.trial.last_result
                    return self._fail("locked_endpoint_" + reason, now, **details)
                actual_increment = self._task_delta(pose, self.trial.start_pose)
                reason = self.trial.update(now, dt, actual_increment, depth, current, chi, loads_allowed)
                if reason:
                    details.update(finite_trial=self.trial.snapshot(), trial_result=self.trial.last_result)
                    return self._fail(reason, now, **details)
                self.action = self.trial.record["action"] if self.trial.active else "Hold"
                # Only independently observed endpoint arrival unlocks the
                # confirmation phase. Rebase once; never add the trial twice.
                if was_phase == "OBSERVE" and self.trial.phase == "CONFIRM":
                    self.nominal = self.last_command.copy()
                    self.bias[:] = 0.
                    self.admittance.sync(np.zeros(6), np.zeros(6))
                    correction[:] = 0.
                    proposed_nominal = offset_pose(self.nominal, advance, self.R_base_task)
                if not self.trial.active:
                    details["trial_result"] = self.trial.last_result
                    if self.probe_action is not None:
                        details.update(probe_completed=True, probe_action=self.probe_action,
                                       advance_requested=False, advance_scale=0.)
                        return self._fail("supervised_probe_completed", now, **details)
                    self.policy.reset()
                elif self.probe_action is not None and at_goal:
                    return self._fail("probe_reached_depth_boundary_before_confirmation", now, **details)
            else:
                if at_goal or not loads_allowed or not self.trial.ready(now):
                    allowed = ["Hold"]
                for action in tuple(allowed):
                    if action == "Hold":
                        continue
                    increment = action_vector(action) * self.synergy.action_step
                    endpoint = offset_pose(self.last_command, increment, self.R_base_task)
                    endpoint_offset = self._task_delta(endpoint, self.target)
                    endpoint_offset[2] = 0.
                    reason = self.synergy.endpoint_reason(endpoint_offset, budget)
                    if (np.any(np.abs(self._task_delta(endpoint, self.contact_pose)) > limits["total_offset"])
                            or np.any(endpoint[:3, 3] < limits["workspace_min"])
                            or np.any(endpoint[:3, 3] > limits["workspace_max"])):
                        reason = "locked_endpoint_travel_or_workspace"
                    proposed_bias = self.bias + increment
                    if np.any(np.abs(proposed_bias) > np.asarray(self.config["learning"]["bias_budget"]) + 1e-12):
                        reason = "accumulated_bias_budget"
                    if reason:
                        allowed.remove(action)
                        rejections[action] = reason
                self.action = self.policy.choose(probabilities, allowed, now)
                if self.action != "Hold":
                    try:
                        self.trial.start(self.action, now, self.last_command, self.bias,
                                         self.synergy.action_step, self.config["learning"]["bias_rate"], depth)
                        self.trial.record["endpoint_pose"] = offset_pose(
                            self.trial.start_pose, self.trial.delta, self.R_base_task).tolist()
                        self.trial.record.update(
                            selection_probabilities=np.asarray(probabilities).tolist(),
                            selection_confidence=float(probabilities[ACTIONS.index(self.action)]),
                            selection_prediction_at=self.policy.prediction_time,
                            decision_policy_source=getattr(self.policy, "source", "trained_gru"),
                            peak_source="compensated_unfiltered_task_wrench")
                    except ValueError as exc:
                        return self._fail(str(exc), now, **details)
                    self.nominal = self.last_command.copy()
                    self.bias[:] = 0.
                    self.admittance.sync(np.zeros(6), np.zeros(6))
                    correction[:] = 0.
                    self.trial.update(now, dt, np.zeros(6), depth, current, chi, loads_allowed)
                elif (self.trial.wait_started is not None and
                      now - self.trial.wait_started > self.config["finite_trial"]["max_wait_s"]):
                    return self._fail("finite_trial_candidate_timeout", now, **details)
            details.update(allowed_actions=list(allowed), action_rejections=rejections,
                           finite_trial=self.trial.snapshot(), rules_version=self.config["rules_version"],
                           learning_reason=self.policy.last_reason,
                           selection_confidence=(self.trial.record.get("selection_confidence")
                                                 if self.trial.active else None))
            if self.trial.freezes_advance:
                scale = 0.
                self.progress.clear()
                self.pause_since = None
                details.update(action=self.action, advance_scale=0., advance_requested=False, load_gate=chi)
                desired = offset_pose(self.trial.start_pose, self.trial.increment, self.R_base_task)
                if (np.any(np.abs(self._task_delta(pose, self.contact_pose)) > limits["total_offset"])
                        or np.any(np.abs(self._task_delta(desired, self.contact_pose)) > limits["total_offset"])):
                    return self._fail("finite_trial_total_travel_limit", now, **details)
                return desired
        else:
            self.action = "Hold"
        if self.policy:
            details.update(probabilities=None if probabilities is None else np.asarray(probabilities).tolist(),
                           learning_reason=self.policy.last_reason,
                           inference_seconds=self.policy.last_inference_seconds)
            if not self.trial.active and self.policy.hold_timed_out(now):
                return self._fail("learning_hold_timeout", now, **details)
        details.update(action=self.action, advance_scale=scale, advance_requested=scale > 0., load_gate=chi)
        if at_goal:
            # Reaching a depth once is not stable seating. Stop nominal advance
            # and require low actual velocity and a bounded measured depth range
            # for every sample of the complete_hold window.
            self.progress.clear()
            self.pause_since = None
            details.update(advance_requested=False, advance_scale=0., completion_hold=True)
            self.nominal = self.last_command.copy()
            # This feasible pose already contains the previous learned and
            # admittance corrections. A loss of seating must not add them a
            # second time when nominal advancement resumes.
            self.bias[:] = 0
            self.admittance.sync(np.zeros(6), np.zeros(6))
            self.last_action_time = self.last_action_prediction = None
            if np.any(np.abs(self.actual_velocity) > task["complete_velocity"]):
                self.complete_since = None
                self.completion_depths.clear()
            else:
                if self.complete_since is None:
                    self.complete_since = now
                self.completion_depths.append(depth)
                if max(self.completion_depths) - min(self.completion_depths) > task["complete_depth_range"]:
                    self.complete_since = now
                    self.completion_depths.clear()
                    self.completion_depths.append(depth)
            if self.complete_since is not None and now - self.complete_since >= task["complete_hold"]:
                self._transition("COMPLETE", now)
                return Decision(self.state, stop=True, finished=True, success=True,
                                reason="mechanical_insertion_complete", details=details)
            return self.last_command.copy()
        self.complete_since = None
        self.completion_depths.clear()
        if scale <= 0:
            if self.pause_since is None:
                self.pause_since = now
            if now - self.pause_since > task["max_pause"]:
                return self._fail("pause_timeout", now, **details)
            self.progress.clear()
        else:
            self.pause_since = None
            self.nominal_distance += task["insert_speed"] * scale * dt
            if self.nominal_distance > task["max_insert_travel"]:
                return self._fail("insert_travel_exhausted", now, **details)
        if scale > 0 and abs(current[2]) >= task["stall_load"]:
            self.progress.append((now, depth))
            while len(self.progress) > 1 and self.progress[1][0] <= now - task["stall_window"]:
                self.progress.popleft()
            if len(self.progress) > 1 and now - self.progress[0][0] >= task["stall_window"]:
                if depth - self.progress[0][1] < task["stall_min_progress"]:
                    return self._fail("progress_stall", now, **details)
        else:
            self.progress.clear()
        offset = self._task_delta(pose, self.contact_pose)
        if np.any(np.abs(offset) > limits["total_offset"]):
            return self._fail("actual_total_travel_limit", now, **details)
        self.nominal = proposed_nominal
        desired = offset_pose(self.nominal, self.bias + correction, self.R_base_task)
        if self.synergy:
            full_offset = self._task_delta(desired, self.target)
            full_offset[2] = 0.
            if self.synergy.endpoint_reason(full_offset, self.vision_budget if self.flags["synergy"] else None):
                return self._fail("full_command_synergy_constraint", now, **details)
        if np.any(np.abs(self._task_delta(desired, self.contact_pose)) > limits["total_offset"]):
            return self._fail("command_total_travel_limit", now, **details)
        return desired
