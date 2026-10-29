"""Pairing, native depth validity and conservative detected-ROI plane evidence."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

from vision.contracts import intrinsics_signature
from vision.depth_diagnostic import analyze_depth
from vision.preview_session import PreviewSession


class DepthDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.camera = {
            "stream": "color", "serial": "test-camera", "image_width": 160, "image_height": 160,
            "camera_matrix": [[200., 0, 80.], [0, 200., 80.], [0, 0, 1]],
            "dist_coeffs": [0.] * 5, "distortion_model": "none",
        }
        self.frame = {"frame_number": 12, "device_timestamp_ms": 1000.,
                      "timestamp_domain": "hardware_clock", "received_at_utc": "2026-10-29T12:00:00+00:00"}
        yy, xx = np.mgrid[:80, :80]
        self.pixels = np.stack([xx * 2., yy * 2.], axis=2)
        self.points = np.concatenate([(self.pixels - 80) / 200 * .5,
                                      np.full((80, 80, 1), .5)], axis=2)
        self.packet = {
            "raw_z16": np.full((80, 80), 500, dtype=np.uint16),
            "frame": {**self.frame, "frame_number": 9, "device_timestamp_ms": 999.},
            "color_frame": dict(self.frame), "points_color_m": self.points,
            "color_pixels": self.pixels, "reasons": [],
            "calibration": {
                "schema_version": 1, "depth_scale_m": .001,
                "color_intrinsics_sha256": intrinsics_signature(self.camera),
                "max_pair_skew_ms": 20., "depth_profile": {"width": 80, "height": 80, "fps": 15, "format": "z16"},
                "depth_to_color": {"rotation": np.eye(3).tolist(), "translation_m": [0., 0., 0.]},
            },
        }
        self.frontend = {"selected_bbox_xyxy": [10., 10., 150., 150.], "named_points_raw_px": {}}
        self.pose = {"solved": True, "rvec": [0., 0., 0.], "tvec_mm": [0., 0., 500.]}
        self.geometry = {"status": "nominal_reference", "points_mm": {
            "a": [-10., -10., 0.], "b": [10., -10., 0.], "c": [10., 10., 0.], "d": [-10., 10., 0.]}}

    def analyze(self, **overrides):
        result = analyze_depth(overrides.get("packet", self.packet), self.frame, self.camera,
                               overrides.get("frontend", self.frontend), self.pose, self.geometry)
        json.dumps(result, allow_nan=False)
        return result

    def test_candidate_plane_has_color_units_normal_and_unverified_reference(self):
        result = self.analyze()
        self.assertEqual(result["status"], "candidate_plane")
        self.assertFalse(result["authorizes_robot_motion"])
        plane = result["plane"]
        np.testing.assert_allclose(plane["normal_color_camera"], [0, 0, -1], atol=1e-9)
        self.assertAlmostEqual(plane["centroid_color_mm"][2], 500.)
        self.assertAlmostEqual(plane["offset_mm"], 500.)
        self.assertLess(plane["rms_mm"], 1e-6)
        self.assertFalse(plane["aperture_reference_verified"])
        self.assertAlmostEqual(result["pnp_consistency"]["plane_distance_mm"], 0.)
        self.assertFalse(result["pnp_consistency"]["reference_verified"])
        self.assertAlmostEqual(result["pair_skew_ms"], 1.)
        self.assertEqual(result["scene"]["valid_fraction"], 1.)
        self.assertGreater(result["roi"]["measurement_coverage_fraction"], .2)
        self.assertLess(result["roi"]["measurement_coverage_fraction"], .3)

    def test_raw_depth_camera_z_is_not_substituted_for_color_z(self):
        self.packet["points_color_m"] = self.points + [0, 0, .1]
        self.packet["calibration"]["depth_to_color"]["translation_m"] = [0, 0, .1]
        result = self.analyze()
        self.assertEqual(result["scene"]["median_depth_mm"], 500.)
        self.assertAlmostEqual(result["roi"]["median_color_z_mm"], 600.)
        self.assertAlmostEqual(result["plane"]["centroid_color_mm"][2], 600.)
        self.assertAlmostEqual(result["pnp_consistency"]["plane_distance_mm"], 100.)

    def test_no_detection_never_labels_scene_plane_as_connector(self):
        for value in (None, [], "invalid-box"):
            result = self.analyze(frontend={"selected_bbox_xyxy": value, "reason": "target_not_detected"})
            self.assertEqual(result["status"], "scene_only")
            self.assertEqual(result["scene"]["valid_fraction"], 1.)
            self.assertIsNone(result["roi"])
            self.assertFalse(result["plane"]["solved"])

    def test_missing_and_zero_depth_retain_explicit_failure(self):
        for packet in ({}, {**self.packet, "raw_z16": None},
                       {**self.packet, "raw_z16": np.zeros((80, 80), dtype=np.uint16)}):
            with self.subTest(packet_keys=list(packet)):
                result = self.analyze(packet=packet)
                self.assertEqual(result["status"], "invalid")
                self.assertTrue(result["reasons"])
                self.assertFalse(result["plane"]["solved"])
        result = self.analyze(packet={**self.packet, "raw_z16": np.zeros((80, 80), dtype=np.uint16)})
        self.assertEqual(result["scene"]["valid_fraction"], 0.)
        self.assertIsNone(result["scene"]["median_depth_mm"])

    def test_missing_frame_analysis_can_be_saved_with_rgb(self):
        self.camera.update(device_name="RealSense D415", device_model="D415", fps=15,
                           pixel_format="bgr8", validated=False, version=None)
        depth_camera = {**self.camera, "stream": "depth", "image_width": 80, "image_height": 80,
                        "pixel_format": "z16"}
        self.packet["calibration"]["depth_camera"] = depth_camera
        packet = {**self.packet, "raw_z16": None, "frame": None,
                  "points_color_m": None, "color_pixels": None, "reasons": ["missing_depth"]}
        depth = self.analyze(packet=packet)
        self.assertIsNone(depth["frame"])
        self.assertIsNone(depth["raw_pixels_sha256"])
        self.assertEqual(depth["status"], "invalid")
        artifact_root = Path(__file__).resolve().parents[1] / "artifacts/test-tmp"
        root = artifact_root / ("missing-depth-analysis-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        try:
            session = PreviewSession(root, self.camera, "missing-depth-analysis", pipeline_config={},
                                     geometry=self.geometry, depth_calibration=packet["calibration"])
            image = np.zeros((160, 160, 3), dtype=np.uint8)
            diagnostic = {"source": "live_diagnostic", "authorizes_robot_motion": False,
                          "frame": self.frame, "camera": self.camera, "geometry": self.geometry,
                          "image_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "depth": depth}
            saved = session.save(image, diagnostic, image.copy(), depth_sample=packet)
            self.assertIsNone(saved["depth"]["frame"])
            self.assertFalse(saved["depth"]["available"])
            self.assertIsNone(saved["depth"]["file"])
            self.assertTrue((session.path / saved["image"]).is_file())
            session.finish()
        finally:
            if not root.resolve().is_relative_to(artifact_root.resolve()):
                raise ValueError("test_directory_outside_workspace_artifacts")
            shutil.rmtree(root)

    def test_time_domain_skew_and_invalid_timestamp_are_rejected(self):
        for changed, reason in (({"timestamp_domain": "global_time"}, "depth_timestamp_domain_mismatch"),
                                ({"device_timestamp_ms": 900.}, "depth_pair_skew_exceeded"),
                                ({"device_timestamp_ms": 0.}, "invalid_depth_timestamp"),
                                ({"device_timestamp_ms": float("nan")}, "invalid_depth_timestamp")):
            packet = {**self.packet, "frame": {**self.packet["frame"], **changed}}
            result = self.analyze(packet=packet)
            self.assertEqual(result["status"], "invalid")
            self.assertIn(reason, result["reasons"])
            self.assertEqual(result["raw_pixels_sha256"], hashlib.sha256(packet["raw_z16"].tobytes()).hexdigest())
            if reason == "depth_timestamp_domain_mismatch":
                self.assertIsNone(result["pair_skew_ms"])

    def test_stale_acquisition_and_wrong_color_binding_are_rejected(self):
        stale = {**self.packet, "reasons": ["repeated_or_reordered_depth_frame"],
                 "points_color_m": None, "color_pixels": None}
        result = self.analyze(packet=stale)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("repeated_or_reordered_depth_frame", result["reasons"])
        wrong = {**self.packet, "color_frame": {**self.frame, "frame_number": 13}}
        self.assertIn("depth_color_frame_binding_mismatch", self.analyze(packet=wrong)["reasons"])

    def test_mismatched_profile_scale_and_intrinsics_are_rejected(self):
        for changed, reason in (({"depth_scale_m": 0.}, "invalid_depth_scale"),
                                ({"depth_profile": {"width": 81, "height": 80, "format": "z16"}}, "depth_profile_mismatch"),
                                ({"color_intrinsics_sha256": "other-camera"}, "depth_color_intrinsics_mismatch")):
            packet = {**self.packet, "calibration": {**self.packet["calibration"], **changed}}
            self.assertIn(reason, self.analyze(packet=packet)["reasons"])

    def test_equal_step_planes_are_rejected(self):
        raw = self.packet["raw_z16"].copy()
        raw[:, 40:] = 550
        points = self.points.copy()
        points[:, 40:] *= 1.1
        result = self.analyze(packet={**self.packet, "raw_z16": raw, "points_color_m": points})
        self.assertEqual(result["status"], "plane_rejected")
        self.assertFalse(result["plane"]["solved"])
        self.assertIn("mixed_or_nonplanar_surface", result["reasons"])

    def test_hole_neighborhoods_and_depth_edges_are_excluded(self):
        baseline = self.analyze()
        frontend = {**self.frontend, "named_points_raw_px": {"hole": [80., 80.]}}
        with_hole = self.analyze(frontend=frontend)
        self.assertLess(with_hole["roi"]["surface_samples"], baseline["roi"]["surface_samples"])
        self.assertEqual(with_hole["roi"]["projected_valid_samples"], baseline["roi"]["projected_valid_samples"])
        raw = self.packet["raw_z16"].copy()
        raw[35:45, 35:45] = 0
        with_gap = self.analyze(packet={**self.packet, "raw_z16": raw})
        self.assertLess(with_gap["roi"]["surface_samples"], baseline["roi"]["surface_samples"] - 100)
        self.assertAlmostEqual(with_gap["scene"]["valid_fraction"], 1 - 100 / 6400)

    def test_tilted_plane_and_nonplanar_model(self):
        points = self.points.copy()
        points[:, :, 2] += .2 * points[:, :, 0]
        raw = np.rint(points[:, :, 2] * 1000).astype(np.uint16)
        result = self.analyze(packet={**self.packet, "raw_z16": raw, "points_color_m": points})
        expected = np.array([.2, 0, -1]) / np.linalg.norm([.2, 0, -1])
        np.testing.assert_allclose(result["plane"]["normal_color_camera"], expected, atol=1e-8)
        self.assertAlmostEqual(result["pnp_consistency"]["normal_angle_deg"], 11.309932474, places=6)
        self.geometry["points_mm"]["a"][2] = 5.
        self.assertIsNone(self.analyze()["pnp_consistency"])

    def test_nonfinite_projected_points_do_not_create_plane_or_modify_input(self):
        packet = copy.deepcopy(self.packet)
        packet["points_color_m"][:] = np.nan
        raw_before = packet["raw_z16"].copy()
        result = self.analyze(packet=packet)
        self.assertFalse(result["plane"]["solved"])
        self.assertEqual(result["roi"]["surface_samples"], 0)
        np.testing.assert_array_equal(packet["raw_z16"], raw_before)


if __name__ == "__main__":
    unittest.main()
