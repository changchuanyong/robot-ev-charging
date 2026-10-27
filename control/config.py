"""Explicit configurations: toy values are never accepted by the hardware runner."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


METHODS = {
    "A": {"learning": False, "risk": False, "synergy": False},
    "B": {"learning": False, "risk": True, "synergy": False},
    "C": {"learning": True, "risk": True, "synergy": False},
    "D": {"learning": True, "risk": False, "synergy": False},
    "E": {"learning": True, "risk": True, "synergy": False},
    "F": {"learning": True, "risk": True, "synergy": True},
}


def simulation_config():
    """A small synthetic contact fixture, NOT RM75 or connector acceptance limits."""
    eye = np.eye(4).tolist()
    step = [.00001, .00001, 0., .0002, .0002, 0.]
    return {
        "schema_version": 1, "version": "synthetic-fixture-v1", "simulation_only": True,
        "method": "B", "target_id": "demo-port", "period": .01,
        "timing": {"max_sample_age": .035, "max_vision_age": .25,
                   "max_period": .03, "future_tolerance": .002, "max_pose_skew": .012, "timeout": 18.},
        "calibration": {"version": "synthetic-v1", "validated": True,
                        "expected_camera_version": "synthetic-v1", "expected_budget_version": "synthetic-v1",
                        "expected_camera_serial": "SIMULATED",
                        "R_target_task": np.eye(3).tolist(), "insertion_axis_verified": True,
                        "T_flange_camera": eye, "T_flange_tool": eye,
                        "T_target_plug": eye, "position_budget": [.004, .004, .02],
                        "rotation_budget": [.08, .08, .03]},
        "vision": {"max_rms_px": 3., "max_error_px": 6., "min_inliers": 7},
        "wrench": {"version": "synthetic-v1", "T_tool_sensor": eye,
                   "external_compensated": True, "bias": [0.] * 6,
                   "bias_frozen": True, "filter_tau": .025, "max_gap": .04},
        "limits": {"wrench": [10., 10., 12., 1., 1., 1.],
                   "workspace_min": [-.03, -.03, -.03], "workspace_max": [.03, .03, .05],
                   "velocity": [.012, .012, .008, .08, .08, .08],
                   "acceleration": [.2, .2, .2, 1., 1., 1.],
                   "step": [.001, .001, .001, .005, .005, .005],
                   "tracking_error": [.005, .005, .005, .08, .08, .08],
                   "total_offset": [.005, .005, .02, .1, .1, .03]},
        "task": {"pre_distance": .003, "pre_position_tolerance": .0001,
                 "pre_angle_tolerance": .01, "approach_speed": .002,
                 "insert_speed": .003, "contact_enter": [.15, .4, .08],
                 "contact_exit": [.07, .2, .04], "contact_duration": .05,
                 "release_duration": .1, "confirm_duration": .08,
                 "depth": .006, "depth_tolerance": .00015,
                 "attitude_tolerance": .04, "complete_hold": .15,
                 "complete_velocity": [.0005, .0005, .0005, .01, .01, .01],
                 "complete_depth_range": .0002,
                 "handover_velocity": [.001, .001, .001, .01, .01, .01],
                 "max_approach_travel": .012, "max_insert_travel": .016,
                 "stall_window": 1.5, "stall_min_progress": .00012,
                 "stall_load": .2, "max_pause": 3., "max_attempts": 1,
                 "retreat_enabled": False, "retreat_distance": .003,
                 "retreat_stop_velocity": [.0001, .0001, .0001, .001, .001, .001],
                 "retreat_stationary_duration": .1,
                 "retreat_speed": .001, "retreat_load_limit": [1., 1., 2., .2, .2, .2]},
        "admittance": {"mass": [2., 2., 2., .1, .1, .1],
                       "stiffness": [80., 80., 800., 1., 1., 1.],
                       "damping_min": [12., 12., 30., .8, .8, .8],
                       "damping_max": [30., 30., 50., 2., 2., 2.],
                       "damping_rate": [30., 30., 30., 3., 3., 3.], "dt_margin": .75},
        "risk": {"version": "synthetic-box-v1", "calibrated": True,
                 "scale": [5., 5., 6., .5, .5, .5],
                 "A": np.vstack((np.eye(6), -np.eye(6))).tolist(),
                 "b": [1.] * 12, "reference": [0.] * 6},
        "scheduling": {"load_scales": [6., 5., .5, .5], "low": .35, "high": .85},
        "learning": {"window_size": 8, "sample_period": .01, "max_gap": .035,
                     "max_age": .05, "confidence": .8, "consistent_windows": 3,
                     "min_hold_s": .05, "max_hold_s": 3., "model_path": None,
                     "bias_rate": [.0002, .0002, 0., .004, .004, 0.],
                     "bias_budget": [.001, .001, 0., .02, .02, 0.]},
        "synergy": {"validated": True, "max_offset": [.004, .004, .02, .08, .08, .03],
                    "action_step": step,
                    "pins": [{"point_m": [x, y, 0.], "clearance_m": .006,
                              "engagement_m": .01} for x, y in [(-.017, 0.), (.017, 0.), (0., .021)]],
                    "force_response": {"calibrated": True, "version": "synthetic-zero-response",
                                       "risk_version": "synthetic-box-v1", "action_step": step,
                                       "delta_wrench": np.zeros((6, 8)).tolist()}},
        "hardware": None,
    }


def hardware_template():
    """All application-dependent physical numbers must be supplied explicitly."""
    c = simulation_config()
    c.update(version="UNCONFIGURED", simulation_only=False, target_id=None, method="A", period=None)
    for section in ("timing", "vision", "limits", "task", "admittance", "scheduling"):
        c[section] = {k: None for k in c[section]}
    c["task"].update(retreat_enabled=False, max_attempts=1)
    c["calibration"] = {k: None for k in c["calibration"]}
    c["calibration"].update(validated=False, version=None)
    c["calibration"]["insertion_axis_verified"] = False
    c["wrench"] = {k: None for k in c["wrench"]}
    c["wrench"].update(external_compensated=True, bias_frozen=True)
    c["risk"] = {"version": None, "calibrated": False, "scale": None,
                 "A": None, "b": None, "reference": None}
    c["synergy"] = {"validated": False, "max_offset": None, "action_step": None,
                    "pins": [], "force_response": None}
    c["learning"] = {k: None for k in c["learning"]}
    c["hardware"] = {"host": None, "port": 8080, "controller_version": None,
                     "controller_version_confirmed": False, "expected_sdk_version": "1.1.6",
                     "expected_software": {}, "expected_arm_model": "RM_75",
                     "expected_force_type": None, "tcp_name": None, "work_frame_name": None,
                     "udp": {"ip": None, "port": None, "cycle": None},
                     "follow": False, "trajectory_mode": 0, "command_period": None,
                     "max_sample_age": None, "command_gap_timeout": None,
                     "max_translation_step": None, "max_rotation_step": None,
                     "transport_latency_bound_s": None, "acceptance_record": None,
                     "stop_feedback_timeout": None,
                     "stop_tolerance": {"joint_speed_rad_s": None, "linear_speed_m_s": None,
                                        "angular_speed_rad_s": None, "hold_s": None}}
    return c


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    return config


def validate_config(config, *, hardware=False):
    """Return actionable errors without connecting any device."""
    errors = []
    if not isinstance(config, dict):
        return ["configuration must be a JSON object"]
    for section in ("timing", "vision", "task", "scheduling", "limits", "calibration",
                    "admittance", "wrench", "risk", "learning", "synergy"):
        if not isinstance(config.get(section), dict):
            errors.append(f"{section} must be a JSON object")
    if errors:
        return errors
    if config.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    method = config.get("method")
    if not isinstance(method, str) or method not in METHODS:
        errors.append("method must be A, B, C, D, E, or F")
        return errors
    flags = METHODS[method]
    if type(config.get("simulation_only")) is not bool:
        errors.append("simulation_only must be an explicit boolean")
    if hardware and config.get("simulation_only") is not False:
        errors.append("hardware refuses a simulation configuration")
    for key in ("version", "target_id"):
        if not config.get(key) or config.get(key) == "UNCONFIGURED":
            errors.append(f"{key} is unconfigured")

    def number(value, name, positive=True):
        try:
            if isinstance(value, bool) or not np.isfinite(value) or (positive and value <= 0):
                raise ValueError()
        except (TypeError, ValueError):
            errors.append(f"{name} must be a finite {'positive ' if positive else ''}number")

    number(config.get("period"), "period")
    for section in ("timing", "vision", "task", "scheduling"):
        template = simulation_config()[section]
        for name in template:
            value = config.get(section, {}).get(name)
            if name == "retreat_enabled":
                if not isinstance(value, bool):
                    errors.append("task.retreat_enabled must be boolean")
            elif isinstance(template[name], list):
                try:
                    a = np.asarray(value, float)
                    if a.shape != (len(template[name]),) or not np.isfinite(a).all() or (a <= 0).any():
                        raise ValueError()
                except (ValueError, TypeError):
                    errors.append(f"{section}.{name} must have {len(template[name])} positive entries")
            else:
                number(value, f"{section}.{name}")
    for name, expected in simulation_config()["limits"].items():
        try:
            a = np.asarray(config["limits"][name], float)
            if a.shape != (len(expected),) or not np.isfinite(a).all():
                raise ValueError()
            if not name.startswith("workspace") and (a <= 0).any():
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors.append(f"limits.{name} has invalid shape or values")
    try:
        if not np.all(np.asarray(config["limits"]["workspace_min"]) < config["limits"]["workspace_max"]):
            errors.append("workspace_min must be below workspace_max")
        if not np.all(np.asarray(config["task"]["contact_exit"]) < config["task"]["contact_enter"]):
            errors.append("contact_exit must be below contact_enter")
        if not 0 <= config["scheduling"]["low"] < config["scheduling"]["high"] <= 1:
            errors.append("scheduling thresholds must satisfy 0 <= low < high <= 1")
        if config["task"]["max_attempts"] != 1:
            errors.append("automatic retry is disabled; max_attempts must equal 1")
        if config["period"] > config["timing"]["max_period"]:
            errors.append("period exceeds timing.max_period")
    except (KeyError, TypeError, ValueError):
        pass  # Individual fields above already explain missing configuration.
    from .geometry import validate_transform
    for name in ("T_flange_camera", "T_flange_tool", "T_target_plug"):
        try:
            validate_transform(config["calibration"][name])
        except (KeyError, ValueError, TypeError):
            errors.append(f"calibration.{name} must be a measured rigid transform")
    if not config["calibration"].get("expected_camera_version"):
        errors.append("calibration.expected_camera_version must freeze the RGB calibration identity")
    try:
        rotation = np.asarray(config["calibration"]["R_target_task"], float)
        if rotation.shape != (3,3):
            raise ValueError("rotation shape")
        task_transform = np.eye(4)
        task_transform[:3,:3] = rotation
        validate_transform(task_transform)
    except (ValueError, KeyError, TypeError):
        errors.append("calibration.R_target_task must specify the task axes in target coordinates")
    try:
        from .wrench import WrenchProcessor
        WrenchProcessor(config["wrench"])
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"wrench: {exc}")
    try:
        from .admittance import Admittance
        ad = Admittance(config["admittance"])
        if config["timing"]["max_period"] > ad.max_dt:
            errors.append("timing.max_period exceeds the admittance discrete interval")
    except (KeyError, ValueError, TypeError) as exc:
        errors.append(f"admittance: {exc}")
    if flags["risk"] or flags["synergy"]:
        try:
            from .risk import EngineeringRisk
            risk = EngineeringRisk(config["risk"])
            if hardware and not risk.calibrated:
                errors.append("risk domain has not been calibrated")
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(f"risk: {exc}")
    if flags["learning"]:
        try:
            from .learning import ForcePolicy
            ForcePolicy({**config["learning"], "model_path": None})
            for name in ("bias_rate", "bias_budget"):
                a = np.asarray(config["learning"][name], float)
                if a.shape != (6,) or not np.isfinite(a).all() or (a < 0).any() or (a[[0,1,3,4]] <= 0).any() or (a[[2,5]] != 0).any():
                    raise ValueError(f"{name} needs positive x/y/Rx/Ry and zero z/Rz entries")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"learning: {exc}")
    if flags["synergy"]:
        try:
            from .synergy import SynergyFilter
            SynergyFilter(config["synergy"])
            for name in ("position_budget", "rotation_budget"):
                a = np.asarray(config["calibration"][name], float)
                if a.shape != (3,) or not np.isfinite(a).all() or (a <= 0).any():
                    raise ValueError(f"{name} must contain three calibrated positive SI limits")
            if not config["calibration"].get("expected_budget_version"):
                raise ValueError("expected_budget_version required for F")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"synergy: {exc}")
    if hardware:
        cal = config.get("calibration", {})
        if cal.get("validated") is not True or not cal.get("version"):
            errors.append("independent hand-eye and tool calibration acceptance required")
        if not cal.get("expected_camera_serial") or cal.get("expected_camera_serial") == "SIMULATED":
            errors.append("calibration.expected_camera_serial must identify the calibrated D435")
        if cal.get("insertion_axis_verified") is not True:
            errors.append("physical insertion axis and keyed orientation have not been verified")
        h = config.get("hardware") or {}
        if not isinstance(h, dict):
            errors.append("hardware must be a JSON object")
            return errors
        for name in ("host", "controller_version", "tcp_name", "work_frame_name",
                     "udp", "command_period", "expected_sdk_version", "expected_force_type",
                     "acceptance_record", "transport_latency_bound_s", "stop_feedback_timeout", "stop_tolerance"):
            if h.get(name) is None or h.get(name) == "":
                errors.append(f"hardware.{name} is unconfigured")
        if flags["learning"] and not config.get("learning", {}).get("model_path"):
            errors.append("learning method requires a trained checkpoint")
        if flags["synergy"] and config.get("synergy", {}).get("validated") is not True:
            errors.append("F requires independently validated geometry and action response")
        if not h.get("expected_software"):
            errors.append("hardware.expected_software must freeze the verified firmware identity")
        number(h.get("stop_feedback_timeout"), "hardware.stop_feedback_timeout")
        stop_tolerance = h.get("stop_tolerance")
        if not isinstance(stop_tolerance, dict):
            errors.append("hardware.stop_tolerance must define measured stationarity thresholds")
        else:
            for name in ("joint_speed_rad_s", "linear_speed_m_s", "angular_speed_rad_s", "hold_s"):
                number(stop_tolerance.get(name), f"hardware.stop_tolerance.{name}")
        try:
            if h["command_period"] != config["period"]:
                errors.append("hardware.command_period must equal period")
        except KeyError:
            pass
    return errors


def require_config(config, *, hardware=False):
    errors = validate_config(config, hardware=hardware)
    if errors:
        raise ValueError("Configuration is not ready:\n- " + "\n- ".join(errors))
    return deepcopy(config)
