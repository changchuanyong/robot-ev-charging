import copy
import importlib.util
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from control.hardware import RealManRobot


class FakeSDK:
    rm_thread_mode_e = SimpleNamespace(RM_TRIPLE_MODE_E=2)

    def __init__(self):
        self.created = 0
        self.robot = FakeRobot()

    def RoboticArm(self, mode):
        self.created += 1
        return self.robot

    @staticmethod
    def rm_udp_custom_config_t():
        return SimpleNamespace(joint_speed=0)

    @staticmethod
    def rm_realtime_push_config_t(cycle, enable, port, force_coordinate, ip, custom):
        return {"cycle": cycle, "enable": enable, "port": port, "force_coordinate": force_coordinate,
                "ip": ip, "custom_config": vars(custom)}

    @staticmethod
    def rm_realtime_arm_state_callback_ptr(callback):
        return callback


class FakeRobot:
    def __init__(self):
        self.calls = []
        self.info = {"arm_model": "RM_75", "arm_dof": 7, "force_type": "6FB", "robot_controller_version": 3}
        self.udp = {"cycle": 20, "enable": False, "port": 8089, "force_coordinate": 2,
                    "ip": "192.0.2.5", "custom_config": {"joint_speed": 0}}
        self.original_udp = copy.deepcopy(self.udp)
        self.send_result = 0
        self.stop_event = threading.Event()

    def rm_create_robot_arm(self, host, port):
        self.calls.append("connect")
        return SimpleNamespace(id=1)

    def rm_get_robot_info(self):
        return 0, self.info

    def rm_get_arm_software_info(self):
        return 0, {"product_version": "test-firmware"}

    def rm_get_current_tool_frame(self):
        return 0, {"name": "plug", "pose": [0, 0, 0.1, 0, 0, 0]}

    def rm_get_current_work_frame(self):
        return 0, {"name": "base", "pose": [0, 0, 0, 0, 0, 0]}

    def rm_get_realtime_push(self):
        return 0, copy.deepcopy(self.udp)

    def rm_set_realtime_push(self, config):
        self.udp = copy.deepcopy(config)
        self.calls.append("udp")
        return 0

    def rm_realtime_arm_state_call_back(self, callback):
        self.callback = callback

    def rm_movep_canfd(self, pose, follow, trajectory_mode, radio):
        self.calls.append(("send", pose))
        return self.send_result

    def rm_set_arm_stop(self):
        self.calls.append("stop")
        self.stop_event.set()
        return 0

    def rm_delete_robot_arm(self):
        self.calls.append("delete")
        return 0


def config():
    tool = np.eye(4)
    tool[2, 3] = 0.1
    return {"host": "192.0.2.1", "port": 8080, "controller_version": 3,
            "expected_sdk_version": "1.1.6", "expected_software": {"product_version": "test-firmware"},
            "expected_arm_model": "RM_75", "expected_force_type": "6FB", "tcp_name": "plug",
            "work_frame_name": "base", "T_flange_tool": tool.tolist(),
            "udp": {"ip": "192.0.2.2", "port": 8089, "cycle": 1}, "command_period": 0.01,
            "follow": False, "trajectory_mode": 0, "max_sample_age": 0.1,
            "command_gap_timeout": 0.02, "max_translation_step": 0.001, "max_rotation_step": 0.01,
            "transport_latency_bound_s": 0.005}


def packet():
    return {"arm_ip": "192.0.2.1", "errCode": 0, "err": {"err_len": 0, "err": []},
            "waypoint": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}, "euler": {"rx": 0, "ry": 0, "rz": 0}},
            "force_sensor": {"coordinate": 0, "force": [11, 12, 13, 1, 2, 3], "zero_force": [1, 2, 3, 0.1, 0.2, 0.3]},
            "joint_status": {"joint_position": [90] * 7, "joint_speed": [0] * 7,
                             "joint_en_flag": [True] * 7, "joint_err_code": [0] * 7}}


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.now = 1.0
        self.sdk = FakeSDK()
        self.adapter = RealManRobot(config(), sdk=self.sdk, clock=lambda: self.now)

    def tearDown(self):
        self.adapter.close()

    def test_construct_is_inert_and_connect_does_not_move(self):
        self.assertEqual(self.sdk.created, 0)
        self.adapter.connect()
        self.assertFalse(any(isinstance(c, tuple) and c[0] == "send" for c in self.sdk.robot.calls))
        self.assertNotIn("stop", self.sdk.robot.calls)
        self.adapter.close()
        self.assertEqual(self.sdk.robot.udp, self.sdk.robot.original_udp)

    def test_reject_identity_and_restore_resources(self):
        self.sdk.robot.info["force_type"] = "B"
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.adapter.connect()
        self.assertIn("delete", self.sdk.robot.calls)
        self.assertNotIn("udp", self.sdk.robot.calls)

    def test_sdk_units_and_sensor_reference_are_preserved(self):
        self.adapter.connect()
        self.sdk.robot.callback(packet())
        sample = self.adapter.read()
        self.assertTrue(sample.valid)
        np.testing.assert_allclose(sample.pose[:3, 3], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(sample.wrench, [1, 2, 3, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(sample.joints, [np.pi / 2] * 7)
        self.assertEqual(sample.wrench_frame, "sensor")
        self.assertTrue(sample.metadata["external_compensated"])
        self.assertEqual(self.adapter.read().seq, sample.seq)
        self.assertAlmostEqual(sample.timestamp, 0.995)

    def test_invalid_packet_and_age_prevent_commands(self):
        self.adapter.connect()
        self.sdk.robot.callback(packet())
        self.now = 2.0
        self.assertFalse(self.adapter.read().valid)
        with self.assertRaisesRegex(RuntimeError, "invalid feedback"):
            self.adapter.command(np.eye(4), 0.01)
        self.assertIn("stop", self.sdk.robot.calls)

    def test_send_failure_stops_and_latches(self):
        self.adapter.connect()
        self.sdk.robot.callback(packet())
        self.sdk.robot.send_result = -1
        with self.assertRaisesRegex(RuntimeError, "pass-through"):
            self.adapter.command(self.adapter.read().pose, 0.01)
        self.assertIn("stop", self.sdk.robot.calls)
        with self.assertRaisesRegex(RuntimeError, "not available"):
            self.adapter.command(self.adapter.read().pose, 0.01)

    def test_cartesian_step_and_frame_validation(self):
        self.adapter.connect()
        self.sdk.robot.callback(packet())
        target = self.adapter.read().pose
        target[0, 3] += 0.1
        with self.assertRaisesRegex(RuntimeError, "Cartesian step"):
            self.adapter.command(target, 0.01)
        self.assertIn("stop", self.sdk.robot.calls)

    def test_force_coordinate_and_joint_errors_fail_closed(self):
        self.adapter.connect()
        bad = packet()
        bad["force_sensor"]["coordinate"] = 2
        self.sdk.robot.callback(bad)
        self.assertFalse(self.adapter.read().valid)
        bad = packet()
        bad["joint_status"]["joint_en_flag"][2] = False
        self.sdk.robot.callback(bad)
        self.assertFalse(self.adapter.read().valid)

    def test_stop_ack_is_distinct_from_measured_stop(self):
        self.adapter.connect()
        self.sdk.robot.callback(packet())
        self.adapter.stop()
        self.assertTrue(self.adapter.stop_command_ack)
        self.assertFalse(self.adapter.stop_feedback_confirmed)
        tolerances = {"joint_speed_rad_s": 0.01, "linear_speed_m_s": 0.001,
                      "angular_speed_rad_s": 0.01, "hold_s": 0.02}
        self.assertFalse(self.adapter.wait_stopped(0.015, tolerances))
        def producer():
            for _ in range(6):
                time.sleep(0.01)
                self.now += 0.01
                self.sdk.robot.callback(packet())
        thread = threading.Thread(target=producer)
        thread.start()
        try:
            self.assertTrue(self.adapter.wait_stopped(0.2, tolerances))
            self.assertTrue(self.adapter.stop_feedback_confirmed)
        finally:
            thread.join()

    def test_independent_watchdog_stops_a_stalled_sender(self):
        self.adapter.connect()
        self.sdk.robot.callback(packet())
        self.adapter.command(self.adapter.read().pose, 0.01)
        self.now += 0.03
        self.assertTrue(self.sdk.robot.stop_event.wait(0.2))
        self.assertIn("watchdog", self.adapter.fault)

    @unittest.skipUnless(importlib.util.find_spec("Robotic_Arm"), "optional RM SDK is unavailable")
    def test_actual_sdk_structure_without_connecting(self):
        from Robotic_Arm.rm_robot_interface import rm_realtime_arm_joint_state_t

        self.adapter.connect()  # Injected FakeSDK only; the real class is a data structure.
        sample = rm_realtime_arm_joint_state_t()
        sample.arm_ip = b"192.0.2.1"
        sample.waypoint.position.x = 0.1
        sample.waypoint.position.y = 0.2
        sample.waypoint.position.z = 0.3
        for i in range(7):
            sample.joint_status.joint_en_flag[i] = True
        sample.force_sensor.zero_force[0] = 2.5
        self.sdk.robot.callback(sample)
        result = self.adapter.read()
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.wrench[0], 2.5)


if __name__ == "__main__":
    unittest.main()
