"""Synthetic PosePipeline -> controller contract checks; no camera or robot access."""
from copy import deepcopy
import importlib.util
import json
import unittest

import numpy as np

from control.config import simulation_config
from control.geometry import exp_so3, invert
from control.state_machine import InsertionController
from control.types import RobotSample
from vision.contracts import FramePacket
from vision.pipeline import PosePipeline, ROOT


class ReadyTestPolicy:
    """Only marks the test's visual F gate ready; never used for inference."""
    model_ready = True


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for synthetic PnP integration")
class VisionControllerIntegrationTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.config = simulation_config()
        self.config["method"] = "A"
        self.config["target_id"] = "integration-port"
        self.config["limits"]["workspace_min"] = [-2., -2., -2.]
        self.config["limits"]["workspace_max"] = [2., 2., 2.]
        self.config["calibration"].update(
            version="integration-handeye-v7", expected_camera_version="integration-intrinsics-v1",
            expected_camera_serial="integration-test-camera",
        )
        tool = np.eye(4)
        tool[:3, :3], tool[:3, 3] = exp_so3([0.1, -0.05, 0.02]), [0.01, -0.02, 0.07]
        camera_mount = np.eye(4)
        camera_mount[:3, :3], camera_mount[:3, 3] = exp_so3([-0.05, 0.1, 0.03]), [0.03, 0.01, 0.02]
        self.config["calibration"]["T_flange_tool"] = tool.tolist()
        self.config["calibration"]["T_flange_camera"] = camera_mount.tolist()
        self.camera = {
            "stream": "color", "serial": "integration-test-camera",
            "image_width": 1280, "image_height": 720,
            "camera_matrix": [[800., 0., 640.], [0., 805., 360.], [0., 0., 1.]],
            "dist_coeffs": [0.01, -0.005, 0.001, -0.002, 0.001],
            "distortion_model": "brown_conrady", "validated": True,
            "is_placeholder": False, "is_approximate": False,
            "version": "integration-intrinsics-v1", "source": "synthetic_test",
        }
        vision_config = json.loads((ROOT / "config/realsense_config.json").read_text())
        vision_config["target_id"] = self.config["target_id"]
        self.pipeline = PosePipeline(vision_config, detector=object())
        self.pipeline._load()
        labels = list(self.pipeline.model)
        model = np.asarray([self.pipeline.model[label] for label in labels])
        self.T_camera_target = np.eye(4)
        self.T_camera_target[:3, :3] = exp_so3([0.12, -0.08, 0.03])
        self.T_camera_target[:3, 3] = [0.01, -0.005, 0.5]
        points, _ = cv2.projectPoints(model, np.array([0.12, -0.08, 0.03]),
                                     np.array([10., -5., 500.]), np.asarray(self.camera["camera_matrix"]),
                                     np.asarray(self.camera["dist_coeffs"]))
        self.points = dict(zip(labels, points.reshape(-1, 2)))
        # Synthetic feature correspondences still pass through real PnP,
        # distortion inversion, raw reprojection, SI conversion and schema.
        self.pipeline._extract_points = lambda image, **kwargs: (self.points, {"test": "integration"})
        self.frame_number = 0

    def observation(self, captured=100., received=100.005, processed=100.01, source="offline"):
        self.frame_number += 1
        packet = FramePacket(np.zeros((720, 1280, 3), dtype=np.uint8),
                             f"integration:{self.frame_number}", captured, received, source,
                             source == "realsense", self.camera)
        result = self.pipeline.process(packet, now=processed)
        self.assertTrue(result["valid"], result["reason"])
        return result

    def sample(self, seq=1, timestamp=100., pose=None, **metadata):
        return RobotSample(seq, timestamp, np.eye(4) if pose is None else pose.copy(), np.zeros(6),
                           wrench_frame="sensor", metadata={"simulation": True,
                           "external_compensated": True, "force_channel": "synthetic_external",
                           "wrench_version": self.config["wrench"]["version"], **metadata})

    def controller(self, hardware_gate=False, method="A"):
        config = deepcopy(self.config)
        config["simulation_only"] = not hardware_gate
        config["method"] = method
        return InsertionController(config, ReadyTestPolicy() if method == "F" else None)

    def assert_visual_rejection(self, controller, sample, observation, now, reason_part=None):
        decision = controller.step(sample, now, observation)
        self.assertEqual(decision.state, "VISION", decision.to_dict())
        self.assertIsNone(decision.command)
        if reason_part:
            self.assertIn(reason_part, decision.reason)
        return decision

    def test_actual_schema_accepted_only_in_simulation_for_offline_source(self):
        observation = self.observation()
        controller = self.controller()
        decision = controller.step(self.sample(), 100.01, observation)
        self.assertEqual(decision.state, "PREALIGN", decision.to_dict())
        self.assertIsNotNone(decision.command)
        self.assertEqual(controller.frozen_vision["frame_id"], observation["frame_id"])
        # No socket/driver is constructed. Exercise only the hardware source gate.
        self.assert_visual_rejection(self.controller(hardware_gate=True), self.sample(), observation,
                                     100.01, "source")

    def test_tool_pose_is_converted_to_flange_before_handeye_chain(self):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = exp_so3([0.02, 0.04, -0.08]), [0.1, 0.2, 0.3]
        observation = self.observation()
        controller = self.controller()
        decision = controller.step(self.sample(pose=pose), 100.01, observation)
        self.assertEqual(decision.state, "PREALIGN")
        expected = (pose @ invert(self.config["calibration"]["T_flange_tool"])
                    @ np.asarray(self.config["calibration"]["T_flange_camera"])
                    @ self.T_camera_target)
        np.testing.assert_allclose(controller.target, expected, atol=1e-6)
        self.assertNotEqual(self.config["calibration"]["version"], observation["calibration_version"])

    def test_pair_uses_capture_time_robot_pose_instead_of_current_pose(self):
        controller = self.controller()
        earlier = np.eye(4)
        earlier[:3, 3] = [0.01, 0., 0.]
        waiting = controller.step(self.sample(1, 100., earlier), 100., None)
        self.assertEqual(waiting.state, "VISION")
        current = earlier.copy()
        current[0, 3] += 0.001
        observation = self.observation(captured=100., received=100.004, processed=100.01)
        decision = controller.step(self.sample(2, 100.01, current), 100.01, observation)
        self.assertEqual(decision.state, "PREALIGN", decision.to_dict())
        expected = (earlier @ invert(self.config["calibration"]["T_flange_tool"])
                    @ np.asarray(self.config["calibration"]["T_flange_camera"])
                    @ self.T_camera_target)
        np.testing.assert_allclose(controller.target, expected, atol=1e-6)

    def test_transport_time_interval_must_fit_entire_pairing_tolerance(self):
        self.config["timing"]["max_sample_age"] = 0.08
        observation = self.observation(captured=99.98, received=100.01, processed=100.02, source="realsense")
        sample = self.sample(timestamp=99.98, received_at=100.02, transport_latency_bound_s=0.04,
                             timestamp_source="host_receive_minus_validated_transport_bound")
        self.assert_visual_rejection(self.controller(hardware_gate=True), sample, observation, 100.02,
                                     "synchronized")

    def test_wrong_schema_clock_serial_and_time_order_are_rejected(self):
        observation = self.observation(source="realsense")
        changes = [{"schema_version": 2}, {"clock": "wall_clock"},
                   {"camera_serial": "another_camera"}, {"processed_at": float("nan")},
                   {"received_at": 100.02}, {"processed_at": 99.99},
                   {"time_sync_valid": False}, {"calibration_version": "different-intrinsics"}]
        for change in changes:
            with self.subTest(change=change):
                changed = deepcopy(observation)
                changed.update(change)
                self.assert_visual_rejection(self.controller(hardware_gate=True), self.sample(), changed, 100.01)

    def test_failed_pipeline_observation_never_freezes_previous_valid_pose(self):
        self.observation()
        self.points = {}
        packet = FramePacket(np.zeros((720, 1280, 3), dtype=np.uint8), "integration:failed", 100., 100.005,
                             "offline", False, self.camera)
        invalid = self.pipeline.process(packet, now=100.01)
        self.assertFalse(invalid["valid"])
        self.assertIsNone(invalid["T_camera_target"])
        controller = self.controller()
        self.assert_visual_rejection(controller, self.sample(), invalid, 100.01, "invalid")
        self.assertIsNone(controller.frozen_vision)

    def test_rejected_vision_cannot_change_task_frame_or_partial_target(self):
        observation = self.observation()
        out_of_workspace = deepcopy(observation)
        out_of_workspace["T_camera_target"][0][3] = 20.
        for controller, candidate in [(self.controller(), out_of_workspace),
                                      (self.controller(method="F"), observation)]:
            self.assert_visual_rejection(controller, self.sample(), candidate, 100.01)
            self.assertIsNone(controller.target)
            self.assertIsNone(controller.pre_pose)
            self.assertIsNone(controller.frozen_vision)
            np.testing.assert_array_equal(controller.R_base_task, np.eye(3))

    def test_target_identity_and_pose_are_frozen_after_preapproach_acceptance(self):
        controller = self.controller()
        observation = self.observation()
        first = controller.step(self.sample(), 100.01, observation)
        frozen = deepcopy(controller.frozen_vision)
        target = controller.target.copy()
        replacement = deepcopy(observation)
        replacement["target_id"] = "other-port"
        replacement["T_camera_target"][0][3] += 1.
        replacement["valid"] = False
        followup = controller.step(self.sample(2, 100.02, first.command), 100.02, replacement)
        self.assertEqual(followup.state, "PREALIGN")
        np.testing.assert_array_equal(controller.target, target)
        self.assertEqual(controller.frozen_vision, frozen)


if __name__ == "__main__":
    unittest.main()
