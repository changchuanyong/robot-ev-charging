from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from vision.contracts import (FrameClock, FramePacket, atomic_json, distort_normalized,
                              empty_observation, intrinsics_signature, rectify_points, validate_camera)
from vision.pipeline import PosePipeline, ROOT, calibrated_visual_budget, select_detection


HAS_CV2 = importlib.util.find_spec("cv2") is not None


def camera_config():
    return {
        "stream": "color", "serial": "test_camera", "image_width": 1280, "image_height": 720,
        "camera_matrix": [[800., 0., 640.], [0., 805., 360.], [0., 0., 1.]],
        "dist_coeffs": [0.01, -0.005, 0.001, -0.002, 0.001],
        "distortion_model": "brown_conrady", "validated": True,
        "is_placeholder": False, "is_approximate": False, "version": "test-only-v1",
        "source": "synthetic_test",
    }


class ObservationContractTests(unittest.TestCase):
    def test_placeholder_and_wrong_resolution_are_rejected(self):
        camera = camera_config()
        camera["is_placeholder"] = True
        with self.assertRaisesRegex(ValueError, "placeholder"):
            validate_camera(camera, (720, 1280, 3))
        camera["is_placeholder"] = False
        with self.assertRaisesRegex(ValueError, "resolution"):
            validate_camera(camera, (1080, 1920, 3))
        camera["stream"] = "depth"
        with self.assertRaisesRegex(ValueError, "color_stream"):
            validate_camera(camera, (720, 1280, 3))

    def test_distortion_roundtrip_for_all_supported_models(self):
        camera = camera_config()
        matrix = np.asarray(camera["camera_matrix"])
        coefficients = np.asarray(camera["dist_coeffs"])
        ideal = np.array([[-0.4, -0.3], [0, 0], [0.5, 0.35]])
        for model in ("brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"):
            raw = distort_normalized(ideal, coefficients, model) * [800, 805] + [640, 360]
            recovered = rectify_points(raw, matrix, coefficients, model)
            np.testing.assert_allclose(recovered, ideal * [800, 805] + [640, 360], atol=1e-7)

    def test_sdk_projection_matches_distortion_convention(self):
        if importlib.util.find_spec("pyrealsense2") is None:
            self.skipTest("pyrealsense2 not installed")
        import pyrealsense2 as rs
        camera = camera_config()
        points = np.array([[-0.35, 0.15], [0.25, -0.2], [0, 0]])
        for model in ("brown_conrady", "modified_brown_conrady", "inverse_brown_conrady"):
            intrinsics = rs.intrinsics()
            intrinsics.width, intrinsics.height = 1280, 720
            intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy = 800, 805, 640, 360
            intrinsics.model = getattr(rs.distortion, model)
            intrinsics.coeffs = camera["dist_coeffs"]
            expected = np.array([rs.rs2_project_point_to_pixel(intrinsics, [*p, 1.0]) for p in points])
            actual = distort_normalized(points, np.asarray(camera["dist_coeffs"]), model) * [800, 805] + [640, 360]
            np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_invalid_observation_replaces_old_pose_atomically(self):
        temporary_root = ROOT / "artifacts" / "test-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = temporary_root / ("vision-" + uuid.uuid4().hex)
        temporary.mkdir()
        output = temporary / "pose.json"
        try:
            atomic_json(output, {"valid": True, "T_camera_target": np.eye(4).tolist()})
            atomic_json(output, empty_observation(None, "port_1", "disconnected"))
            saved = json.loads(output.read_text())
            self.assertFalse(saved["valid"])
            self.assertIsNone(saved["T_camera_target"])
            self.assertEqual(saved["reason"], "disconnected")
            self.assertEqual(len(list(output.parent.iterdir())), 1)
        finally:
            if output.exists():
                output.unlink()
            temporary.rmdir()

    def test_clock_rejects_hardware_epoch_duplicates_and_stale_frames(self):
        clock = FrameClock(0.2)
        self.assertAlmostEqual(clock.map(1, 999980, "global_time", 100, 1000), 99.98)
        with self.assertRaisesRegex(ValueError, "repeated"):
            clock.map(1, 999980, "global_time", 100.02, 1000.02)
        with self.assertRaisesRegex(ValueError, "synchronized"):
            clock.map(2, 1000020, "hardware_clock", 100.04, 1000.04)
        with self.assertRaisesRegex(ValueError, "age_bound"):
            clock.map(2, 999990, "global_time", 100.5, 1000.5)
        with self.assertRaisesRegex(ValueError, "clock_changed"):
            clock.map(2, 1000020, "global_time", 100.04, 1002.04)

    def test_target_identity_does_not_jump_to_higher_confidence_connector(self):
        boxes = [{"xyxy": [100, 100, 200, 200], "confidence": 0.7},
                 {"xyxy": [800, 100, 900, 200], "confidence": 0.99}]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            select_detection(boxes, (720, 1280), None)
        self.assertEqual(select_detection(boxes, (720, 1280), [0, 0, 0.4, 1]), boxes[0])

    def test_visual_budget_requires_independent_calibration_and_domain_match(self):
        camera = camera_config()
        spec = {"validated": True, "budget_m_rad": [0.001]*2+[0.01]*2,
                "axes": ["x", "y", "Rx", "Ry"], "rules_version": "finite-trial-v2",
                "depth_range_m": [0.2, 0.8], "max_reprojection_rms_px": 2,
                "camera_intrinsics_sha256": intrinsics_signature(camera),
                "version": "budget-test", "source": "held-out-pose-validation"}
        self.assertEqual(calibrated_visual_budget(spec, camera, 1, 0.4), (spec["budget_m_rad"], "budget-test"))
        self.assertEqual(calibrated_visual_budget(None, camera, 1, 0.4), (None, None))
        self.assertEqual(calibrated_visual_budget(spec, camera, 3, 0.4), (None, None))
        self.assertEqual(calibrated_visual_budget(spec, camera, 1, 1.4), (None, None))
        camera["serial"] = "other_camera"
        self.assertEqual(calibrated_visual_budget(spec, camera, 1, 0.4), (None, None))


@unittest.skipUnless(HAS_CV2, "OpenCV needed for pose/calibration tests")
class PosePipelineTests(unittest.TestCase):
    def setUp(self):
        import cv2
        config = json.loads((ROOT / "config/realsense_config.json").read_text())
        self.pipeline = PosePipeline(config, detector=object())
        self.pipeline._load()
        self.camera = camera_config()
        self.rotation = np.array([0.15, -0.12, 0.05])
        self.translation = np.array([10., -5., 500.])
        names = list(self.pipeline.model)
        points = np.asarray([self.pipeline.model[name] for name in names])
        raw, _ = cv2.projectPoints(points, self.rotation, self.translation,
                                   np.asarray(self.camera["camera_matrix"]), np.asarray(self.camera["dist_coeffs"]))
        self.points = dict(zip(names, raw.reshape(-1, 2)))
        self.pipeline._extract_points = lambda image, **kwargs: (self.points, {"test": True})

    def packet(self, identifier="test:1", camera=None):
        return FramePacket(np.zeros((720, 1280, 3), dtype=np.uint8), identifier,
                           100.0, 100.01, "realsense", True, camera or self.camera)

    def test_same_frame_pose_is_in_metres_and_low_residual(self):
        result = self.pipeline.process(self.packet(), now=100.02)
        self.assertTrue(result["valid"], result["reason"])
        np.testing.assert_allclose(np.asarray(result["T_camera_target"])[:3, 3], self.translation/1000, atol=1e-7)
        self.assertLess(result["quality"]["reprojection_rms_px"], 1e-4)
        self.assertEqual(result["frame_id"], "test:1")
        self.assertEqual(result["captured_at"], 100.0)
        self.assertIsNone(result["quality"]["visual_budget"])

    def test_failed_next_frame_never_reuses_successful_pose(self):
        result = self.pipeline.process(self.packet(), now=100.02)
        self.assertTrue(result["valid"])
        self.points = {}
        invalid = self.pipeline.process(self.packet("test:2"), now=100.03)
        self.assertFalse(invalid["valid"])
        self.assertIsNone(invalid["T_camera_target"])
        self.assertEqual(invalid["frame_id"], "test:2")

    def test_nonfinite_raw_projection_is_rejected(self):
        self.assertTrue(self.pipeline.process(self.packet(), now=100.02)["valid"])
        for identifier, value in (("test:nan", np.nan), ("test:inf", np.inf)):
            with self.subTest(value=value):
                with patch("vision.pipeline.distort_normalized",
                           side_effect=lambda points, *args: np.full_like(points, value)):
                    result = self.pipeline.process(self.packet(identifier), now=100.02)
                self.assertFalse(result["valid"])
                self.assertIn("nonfinite_raw_reprojection", result["reason"])
                self.assertIsNone(result["T_camera_target"])
                json.dumps(result, allow_nan=False)

    def test_inlier_gate_uses_raw_pixel_residuals(self):
        def shifted_projection(points, coefficients, model):
            projected = distort_normalized(points, coefficients, model)
            projected[:3, 0] += 12.0 / self.camera["camera_matrix"][0][0]
            return projected

        with patch("vision.pipeline.distort_normalized", side_effect=shifted_projection):
            result = self.pipeline.process(self.packet(), now=100.02)
        quality = result["quality"]
        self.assertEqual(quality["rectified_inlier_count"], 9)
        self.assertEqual(quality["inlier_count"], 6)
        self.assertEqual(quality["inlier_threshold_px"], 10.0)
        self.assertEqual(quality["residual_frame"], "raw_color_pixels")
        self.assertLess(quality["reprojection_rms_px"], self.pipeline.config["max_reprojection_rms_px"])
        self.assertLess(quality["max_error_px"], self.pipeline.config["max_point_error_px"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "insufficient_inliers")

    def test_duplicate_expired_and_unvalidated_inputs_are_not_control_valid(self):
        self.pipeline.process(self.packet(), now=100.02)
        duplicate = self.pipeline.process(self.packet(), now=100.03)
        self.assertIn("already_processed", duplicate["reason"])
        expired = self.pipeline.process(self.packet("test:2"), now=101)
        self.assertIn("expired", expired["reason"])
        camera = dict(self.camera, validated=False)
        unvalidated = self.pipeline.process(self.packet("test:3", camera), now=100.02)
        self.assertFalse(unvalidated["valid"])
        self.assertIn("not_independently_validated", unvalidated["reason"])


@unittest.skipUnless(HAS_CV2, "OpenCV needed for hand-eye calibration")
class HandEyeCalibrationTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.camera = np.eye(4)
        self.camera[:3, :3] = cv2.Rodrigues(np.array([0.12, -0.2, 0.08]))[0]
        self.camera[:3, 3] = [0.03, 0.02, 0.06]
        board = np.eye(4)
        board[:3, 3] = [0.4, 0.05, 0.7]
        rng = np.random.default_rng(7)
        pairs = []
        for i in range(14):
            flange = np.eye(4)
            flange[:3, :3] = cv2.Rodrigues(rng.uniform(-0.5, 0.5, 3))[0]
            flange[:3, 3] = rng.uniform(-0.15, 0.15, 3)
            camera_board = np.linalg.inv(self.camera) @ np.linalg.inv(flange) @ board
            pairs.append({"id": str(i), "split": "train" if i < 10 else "validation",
                          "T_base_flange": flange.tolist(), "T_camera_board": camera_board.tolist(),
                          "camera_time": float(i), "robot_time": float(i)})
        self.data = {"translation_unit": "m", "source": "synthetic", "pairing_verified": True,
                     "max_pair_skew_s": 0.01, "pairs": pairs,
                     "acceptance": {"max_translation_m": 1e-5, "max_rotation_rad": 1e-5}}

    def test_synthetic_recovery_and_holdout_validation(self):
        from vision.calibrate import solve_calibration
        result = solve_calibration(self.data)
        np.testing.assert_allclose(result["T_flange_camera"], self.camera, atol=1e-6)
        self.assertTrue(result["acceptance_passed"])
        self.assertFalse(result["validated"], "Synthetic tests are never hardware calibration")
        self.assertIsNone(result["T_flange_tool"])

    def test_holdout_corruption_prevents_acceptance(self):
        from vision.calibrate import solve_calibration
        self.data["pairs"][-1]["T_camera_board"][0][3] += 0.03
        result = solve_calibration(self.data)
        self.assertFalse(result["acceptance_passed"])

    def test_overlap_unsynchronized_pairs_and_degenerate_motion_are_rejected(self):
        from vision.calibrate import solve_calibration
        import copy
        data = copy.deepcopy(self.data)
        data["pairs"][-1]["T_base_flange"] = data["pairs"][0]["T_base_flange"]
        with self.assertRaisesRegex(ValueError, "independent"):
            solve_calibration(data)
        data = copy.deepcopy(self.data)
        data["pairs"][0]["robot_time"] += 1
        with self.assertRaisesRegex(ValueError, "unsynchronized"):
            solve_calibration(data)
        data = copy.deepcopy(self.data)
        for row in data["pairs"]:
            pose = np.asarray(row["T_base_flange"])
            pose[:3, :3] = np.eye(3)
            row["T_base_flange"] = pose.tolist()
        with self.assertRaisesRegex(ValueError, "degenerate"):
            solve_calibration(data)


if __name__ == "__main__":
    unittest.main()
