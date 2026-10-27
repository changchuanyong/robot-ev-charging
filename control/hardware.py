"""RealMan API2 adapter. Import/construction never connects to a device.

SDK contract checked against Robotic_Arm 1.1.6 and the vendor API2 documentation:
https://develop.realman-robotics.com/robot/apipython/classes/udpConfig/
https://develop.realman-robotics.com/robot/apipython/classes/movePlan/
https://develop.realman-robotics.com/robot/apipython/classes/motionControl/
"""
from __future__ import annotations

import copy
import importlib
from importlib.metadata import version
import ipaddress
import threading
import time

import numpy as np

from .geometry import invert, log_so3, pose_from_xyz_rpy, validate_transform, xyz_rpy_from_pose
from .types import RobotSample


def _checked(result, operation):
    if isinstance(result, tuple) and len(result) == 2:
        status, data = result
        if status != 0:
            raise RuntimeError(f"{operation} failed: {status}")
        return data
    if result != 0:
        raise RuntimeError(f"{operation} failed: {result}")
    return None


def _pose_from_packet(pose):
    if isinstance(pose, dict):
        position, euler = pose["position"], pose["euler"]
        pose = [position[k] for k in ("x", "y", "z")] + [euler[k] for k in ("rx", "ry", "rz")]
    return pose_from_xyz_rpy(pose)


class RealManRobot:
    """One explicitly configured RM75. stop() latches until a new connection.

    UDP reports receiver times, not camera-compatible sensor capture timestamps.
    The configured, experimentally verified transport bound is subtracted from
    reception time, so downstream age gates use a conservative sample age.
    This software watchdog cannot replace a controller/hardware emergency stop.
    """
    def __init__(self, config, *, sdk=None, clock=time.monotonic):
        self.config = copy.deepcopy(config)
        self._sdk = sdk
        self._clock = clock
        self._robot = None
        self._callback_handle = None
        self._packet = None
        self._packet_time = None
        self._seq = 0
        self._data_lock = threading.Lock()
        self._api_lock = threading.RLock()
        self._watchdog_exit = threading.Event()
        self._watchdog = None
        self._old_udp = None
        self._udp_changed = False
        self._active = False
        self._stopped = False
        self._last_command_time = None
        self._last_command = None
        self._connected = False
        self.fault = ""
        self.stop_result = None
        self.stop_command_ack = False
        self.stop_feedback_confirmed = False
        self._stop_ack_time = None
        self.identity = {}
        self.T_base_work = np.eye(4)

    def _validate_config(self):
        c = self.config
        required = ("host", "port", "controller_version", "expected_sdk_version", "expected_software",
                    "expected_arm_model", "expected_force_type", "tcp_name", "work_frame_name",
                    "T_flange_tool", "udp", "command_period", "follow", "trajectory_mode",
                    "max_sample_age", "command_gap_timeout", "max_translation_step", "max_rotation_step",
                    "transport_latency_bound_s")
        if any(c.get(key) is None for key in required):
            raise ValueError("hardware config incomplete: " + ", ".join(k for k in required if c.get(k) is None))
        ipaddress.IPv4Address(c["host"])
        ipaddress.IPv4Address(c["udp"]["ip"])
        if c["controller_version"] not in (3, 4) or c["trajectory_mode"] != 0:
            raise ValueError("adapter requires explicitly identified gen3/gen4 controller and trajectory_mode=0")
        if c["expected_arm_model"] != "RM_75" or c["expected_force_type"] not in ("6F", "6FB", "6FB-V"):
            raise ValueError("expected RM_75 with a six-axis force sensor")
        if not c["expected_sdk_version"] or not isinstance(c["expected_software"], dict) or not c["expected_software"]:
            raise ValueError("freeze the measured SDK and firmware identity before motion")
        if not c["tcp_name"] or not c["work_frame_name"]:
            raise ValueError("explicit TCP and work-frame names required")
        validate_transform(c["T_flange_tool"])
        if type(c["follow"]) is not bool or type(c["udp"].get("cycle")) is not int or c["udp"]["cycle"] < 1:
            raise ValueError("follow must be bool; UDP cycle must be a positive count of 5ms")
        for port in (c["port"], c["udp"]["port"]):
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError("invalid TCP/UDP port")
        for key in ("command_period", "max_sample_age", "command_gap_timeout", "max_translation_step",
                    "max_rotation_step", "transport_latency_bound_s"):
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if c["max_sample_age"] <= c["transport_latency_bound_s"]:
            raise ValueError("sample-age limit must exceed the measured transport bound")
        if c["follow"] and (c["command_period"] > 0.010 or c["command_gap_timeout"] > 0.010):
            raise ValueError("high-follow requires a measured command period/watchdog limit <=10ms")
        if c["command_gap_timeout"] < c["command_period"]:
            raise ValueError("watchdog period shorter than command period")

    def _udp_object(self, values):
        custom = self._sdk.rm_udp_custom_config_t()
        for key, value in values.get("custom_config", {}).items():
            if not hasattr(custom, key):
                raise ValueError(f"unknown SDK UDP custom field {key}")
            setattr(custom, key, value)
        return self._sdk.rm_realtime_push_config_t(
            values["cycle"], values["enable"], values["port"], values["force_coordinate"], values["ip"], custom)

    def connect(self):
        self._validate_config()
        if self._connected:
            raise RuntimeError("robot already connected")
        self._packet = None
        self._packet_time = None
        self._last_command = None
        self._last_command_time = None
        self._seq = 0
        self._old_udp = None
        self._udp_changed = False
        self.fault = ""
        if self._sdk is None:
            installed_version = version("Robotic_Arm")
            if installed_version != self.config["expected_sdk_version"]:
                raise RuntimeError(f"SDK version mismatch: {installed_version}")
            self._sdk = importlib.import_module("Robotic_Arm.rm_robot_interface")
        self._robot = self._sdk.RoboticArm(self._sdk.rm_thread_mode_e.RM_TRIPLE_MODE_E)
        try:
            handle = self._robot.rm_create_robot_arm(self.config["host"], self.config["port"])
            if handle.id <= 0:
                raise RuntimeError(f"RM75 connection failed: handle={handle.id}")
            self._connected = True
            info = _checked(self._robot.rm_get_robot_info(), "robot identity")
            software = _checked(self._robot.rm_get_arm_software_info(), "robot software")
            if info.get("arm_dof") != 7 or info.get("arm_model") != self.config["expected_arm_model"] or info.get("force_type") != self.config["expected_force_type"]:
                raise RuntimeError(f"unexpected robot identity: {info}")
            generation = info.get("robot_controller_version")
            if generation is None:
                if self.config.get("controller_version_confirmed") is not True:
                    raise RuntimeError("SDK did not identify controller generation; explicit verified configuration required")
            elif generation != self.config["controller_version"]:
                raise RuntimeError("controller generation mismatch")
            for key, expected in self.config["expected_software"].items():
                if key not in software or software[key] != expected:
                    raise RuntimeError(f"firmware mismatch: {key}")
            tool = _checked(self._robot.rm_get_current_tool_frame(), "current tool")
            work = _checked(self._robot.rm_get_current_work_frame(), "current work frame")
            if tool.get("name") != self.config["tcp_name"] or work.get("name") != self.config["work_frame_name"]:
                raise RuntimeError("active tool/work frame does not match configuration")
            if not np.allclose(_pose_from_packet(tool["pose"]), self.config["T_flange_tool"], atol=2e-6, rtol=0):
                raise RuntimeError("configured tool installation differs from controller TCP")
            self.T_base_work = _pose_from_packet(work["pose"])
            self.identity = {"robot": info, "software": software, "tool": tool, "work": work,
                             "sdk_version": self.config["expected_sdk_version"]}
            self._old_udp = _checked(self._robot.rm_get_realtime_push(), "read previous UDP config")
            # Verify that it can be reconstructed BEFORE changing device configuration.
            self._udp_object(self._old_udp)
            self._callback_handle = self._sdk.rm_realtime_arm_state_callback_ptr(self._on_packet)
            self._robot.rm_realtime_arm_state_call_back(self._callback_handle)
            udp = self.config["udp"]
            new_udp = {"cycle": udp["cycle"], "enable": True, "port": udp["port"],
                       "force_coordinate": 0, "ip": udp["ip"], "custom_config": {"joint_speed": 1}}
            self._udp_changed = True
            _checked(self._robot.rm_set_realtime_push(self._udp_object(new_udp)), "configure UDP feedback")
            actual_udp = _checked(self._robot.rm_get_realtime_push(), "verify UDP configuration")
            if any(actual_udp.get(k) != new_udp[k] for k in ("cycle", "enable", "port", "force_coordinate", "ip")):
                raise RuntimeError("UDP feedback configuration was not applied")
            if actual_udp.get("custom_config", {}).get("joint_speed") != 1:
                raise RuntimeError("joint-speed feedback must be enabled for stop verification")
            self._stopped = False
            self._watchdog_exit.clear()
            self._watchdog = threading.Thread(target=self._watch, name="rm75-watchdog", daemon=True)
            self._watchdog.start()
        except BaseException:
            self.close()
            raise

    def _on_packet(self, data):
        received = self._clock()
        try:
            packet = copy.deepcopy(data if isinstance(data, dict) else data.to_dict())
            if packet.get("arm_ip") != self.config["host"]:
                return
        except Exception as exc:
            packet = {"callback_error": str(exc)}
        with self._data_lock:
            self._seq += 1
            self._packet, self._packet_time = packet, received

    def read(self):
        if not self._connected:
            raise RuntimeError("robot is not connected")
        with self._data_lock:
            packet, received, seq = copy.deepcopy(self._packet), self._packet_time, self._seq
        if packet is None:
            raise RuntimeError("no RM75 UDP feedback received")
        timestamp = received - self.config["transport_latency_bound_s"]
        metadata = {"received_at": received, "timestamp_source": "host_receive_minus_validated_transport_bound",
                    "transport_latency_bound_s": self.config["transport_latency_bound_s"],
                    "external_compensated": True, "force_channel": "udp_external",
                    "wrench_version": self.config.get("wrench_version", "unconfigured"),
                    "identity": self.identity}
        try:
            if self.fault:
                raise ValueError(self.fault)
            if packet.get("errCode") != 0:
                raise ValueError("invalid UDP packet")
            if int(packet["err"]["err_len"]) != 0:
                raise ValueError(f"robot error: {packet['err']}")
            joint = packet["joint_status"]
            if len(joint["joint_en_flag"]) != 7 or not all(joint["joint_en_flag"]) or any(joint["joint_err_code"]):
                raise ValueError("joint disabled or joint error")
            joints = np.asarray(joint["joint_position"], float)
            force = packet["force_sensor"]
            wrench = np.asarray(force["zero_force"], float)
            raw = np.asarray(force["force"], float)
            if force["coordinate"] != 0:
                raise ValueError("force feedback is not in the sensor frame")
            if joints.shape != (7,) or wrench.shape != (6,) or raw.shape != (6,) or not np.isfinite(np.r_[joints, wrench, raw]).all():
                raise ValueError("invalid force or joint dimensions/values")
            if not 0 <= self._clock() - timestamp <= self.config["max_sample_age"]:
                raise ValueError("stale UDP feedback")
            pose = self.T_base_work @ _pose_from_packet(packet["waypoint"])
            metadata["raw_wrench"] = raw.tolist()
            if "joint_speed" in joint:
                speeds = np.asarray(joint["joint_speed"], dtype=float)
                if speeds.shape != (7,) or not np.isfinite(speeds).all():
                    raise ValueError("invalid joint speed feedback")
                metadata["joint_speeds_rad_s"] = np.deg2rad(speeds).tolist()
            metadata["arm_current_status"] = packet.get("arm_current_status")
            return RobotSample(seq, timestamp, pose, wrench, tuple(np.deg2rad(joints)),
                               wrench_frame="sensor", metadata=metadata)
        except (ValueError, KeyError, TypeError) as exc:
            return RobotSample(seq, timestamp, np.eye(4), np.full(6, np.nan), valid=False,
                               reason=str(exc), wrench_frame="sensor", metadata=metadata)

    def command(self, T_base_tool, dt):
        T = validate_transform(T_base_tool)
        with self._api_lock:
            if not self._connected or self._stopped or self.fault:
                raise RuntimeError("robot command stream is not available")
            now = self._clock()
            if not np.isfinite(dt) or not 0 < dt <= self.config["command_gap_timeout"]:
                self.stop()
                raise RuntimeError("command time step exceeded its accepted limit")
            if self._last_command_time is not None and now - self._last_command_time > self.config["command_gap_timeout"]:
                self.stop()
                raise RuntimeError("command stream missed its deadline")
            sample = self.read()
            if not sample.valid:
                self.stop()
                raise RuntimeError(f"cannot command with invalid feedback: {sample.reason}")
            previous = self._last_command if self._last_command is not None else sample.pose
            dp = np.linalg.norm(T[:3, 3] - previous[:3, 3])
            dr = np.linalg.norm(log_so3(T[:3, :3] @ previous[:3, :3].T))
            if dp > self.config["max_translation_step"] or dr > self.config["max_rotation_step"]:
                self.stop()
                raise RuntimeError("command exceeded accepted Cartesian step")
            pose = xyz_rpy_from_pose(invert(self.T_base_work) @ T).tolist()
            self._active = True  # A failed send may still have reached the controller.
            self.stop_command_ack = False
            self.stop_feedback_confirmed = False
            self._last_command_time = now
            try:
                _checked(self._robot.rm_movep_canfd(pose, self.config["follow"], trajectory_mode=0, radio=0), "pose pass-through")
            except BaseException:
                self.stop()
                raise
            self._last_command = T.copy()

    def _watch(self):
        while not self._watchdog_exit.wait(min(0.005, self.config["command_period"])):
            if self._active:
                now = self._clock()
                with self._data_lock:
                    received = self._packet_time
                stale = received is None or now - received + self.config["transport_latency_bound_s"] > self.config["max_sample_age"]
                missed = self._last_command_time is None or now - self._last_command_time > self.config["command_gap_timeout"]
                if stale or missed:
                    self.fault = "feedback_watchdog" if stale else "command_watchdog"
                    try:
                        self.stop()
                    except Exception as exc:
                        self.fault += f"; stop failed: {exc}"

    def stop(self):
        with self._api_lock:
            self._stopped = True
            if self._connected:
                self.stop_result = self._robot.rm_set_arm_stop()
                self.stop_command_ack = self.stop_result == 0
                self._stop_ack_time = self._clock() if self.stop_command_ack else None
                _checked(self.stop_result, "controlled stop")
            self._active = False

    def wait_stopped(self, timeout, tol):
        """Confirm measured stationarity, independently of the stop API return.

        tol requires joint_speed_rad_s, linear_speed_m_s, angular_speed_rad_s
        and hold_s, all from the accepted platform configuration. It never rearms
        the latched stream. Missing/stale/invalid feedback returns False.
        """
        keys = ("joint_speed_rad_s", "linear_speed_m_s", "angular_speed_rad_s", "hold_s")
        if (not np.isfinite(timeout) or timeout <= 0 or any(k not in tol for k in keys)
                or any(not np.isfinite(tol[k]) or tol[k] <= 0 for k in keys)):
            raise ValueError("explicit positive stop verification timeout and tolerances required")
        self.stop_feedback_confirmed = False
        if not self._stopped or not self.stop_command_ack:
            return False
        deadline, previous, stationary_since = time.monotonic() + timeout, None, None
        while time.monotonic() < deadline:
            try:
                sample = self.read()
            except RuntimeError:
                return False
            if not sample.valid:
                return False
            if sample.timestamp < self._stop_ack_time:
                self._watchdog_exit.wait(min(0.005, max(0, deadline - time.monotonic())))
                continue
            if previous is not None and sample.seq == previous.seq:
                self._watchdog_exit.wait(min(0.005, max(0, deadline - time.monotonic())))
                continue
            speeds = np.asarray(sample.metadata.get("joint_speeds_rad_s", []), float)
            if speeds.shape != (7,) or np.max(np.abs(speeds)) > tol["joint_speed_rad_s"]:
                stationary_since = None
            elif previous is not None:
                dt = sample.timestamp - previous.timestamp
                if dt <= 0:
                    return False
                linear = np.linalg.norm(sample.pose[:3, 3] - previous.pose[:3, 3]) / dt
                angular = np.linalg.norm(log_so3(sample.pose[:3, :3] @ previous.pose[:3, :3].T)) / dt
                if linear <= tol["linear_speed_m_s"] and angular <= tol["angular_speed_rad_s"]:
                    if stationary_since is None:
                        stationary_since = previous.timestamp
                    if sample.timestamp - stationary_since >= tol["hold_s"]:
                        self.stop_feedback_confirmed = True
                        return True
                else:
                    stationary_since = None
            previous = sample
            self._watchdog_exit.wait(min(0.005, max(0, deadline - time.monotonic())))
        return False

    def close(self):
        self._watchdog_exit.set()
        if self._watchdog and self._watchdog is not threading.current_thread():
            self._watchdog.join(timeout=1.0)
        errors = []
        with self._api_lock:
            if self._connected:
                if self._active:
                    try:
                        self.stop()
                    except Exception as exc:
                        errors.append(str(exc))
                if self._udp_changed and self._old_udp is not None:
                    try:
                        _checked(self._robot.rm_set_realtime_push(self._udp_object(self._old_udp)), "restore UDP configuration")
                    except Exception as exc:
                        errors.append(str(exc))
                try:
                    _checked(self._robot.rm_delete_robot_arm(), "delete robot handle")
                except Exception as exc:
                    errors.append(str(exc))
            self._connected = False
            self._active = False
            self._stopped = True
            self._robot = None
        if errors:
            self.fault = "; ".join(errors)
            raise RuntimeError(self.fault)
