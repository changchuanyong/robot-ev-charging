"""Synthetic camera/geometry tests; these do not establish physical accuracy."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import cv2
import numpy as np

from vision.capture_session import CaptureSession
from vision.contracts import distort_normalized, intrinsics_signature
from vision.diagnose import (ROOT, DiagnosticFrontend, evaluate, point_metrics,
                             solve_raw_points, validate_capture_annotations)
from vision.geometry_reference import load_geometry_reference
from vision.intrinsics import board_points, calibrate_views, evaluate_board_view
from prepare_capture_review import build_capture_annotations


def camera_config():
    camera = {"serial": "synthetic-d435", "device_name": "Intel RealSense D435", "stream": "color",
              "image_width": 1280, "image_height": 720, "camera_matrix": [[900., 0., 640.], [0., 910., 360.], [0., 0., 1.]],
              "dist_coeffs": [.15, -.04, .002, .003, .01], "distortion_model": "modified_brown_conrady",
              "validated": False, "version": None, "source": "device_color_stream",
              "is_placeholder": False, "is_approximate": False}
    camera["intrinsics_sha256"] = intrinsics_signature(camera)
    return camera


def project(objects, camera, rvec=(.3, -.2, .05), tvec=(20., 10., 350.)):
    xyz = (cv2.Rodrigues(np.array(rvec))[0] @ np.asarray(objects).T).T + np.array(tvec)
    norm = distort_normalized(xyz[:, :2] / xyz[:, 2, None], np.array(camera["dist_coeffs"]), camera["distortion_model"])
    matrix = np.array(camera["camera_matrix"])
    return norm * [matrix[0, 0], matrix[1, 1]] + matrix[:2, 2]


class CameraDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.geometry = load_geometry_reference(ROOT / "config/geometry_nominal.json")
        self.camera = camera_config()
        self.points = dict(zip(self.geometry["points_mm"], project(list(self.geometry["points_mm"].values()), self.camera).tolist()))
        self.config = json.loads((ROOT / "config/realsense_config.json").read_text())

    def test_realsense_raw_pixel_residual_vectors_and_statistical_definitions(self):
        pose = solve_raw_points(self.points, self.geometry["points_mm"], self.camera)
        self.assertTrue(pose["solved"], pose)
        self.assertLess(pose["max_px"], .001)
        shifted = copy.deepcopy(self.points)
        shifted["CC1"][0] += 3
        pose = solve_raw_points(shifted, self.geometry["points_mm"], self.camera)
        values = []
        for label, hole in pose["per_hole"].items():
            delta = np.asarray(hole["projected_px"]) - shifted[label]
            np.testing.assert_allclose(delta, hole["residual_vector_px"])
            values.append(float(np.linalg.norm(delta)))
        self.assertAlmostEqual(pose["rms_px"], np.sqrt(np.mean(np.array(values)**2)))
        self.assertAlmostEqual(pose["mean_px"], np.mean(values))
        self.assertAlmostEqual(pose["max_px"], max(values))

    def test_failed_pose_retains_every_input_point(self):
        points = dict(list(self.points.items())[:3])
        pose = solve_raw_points(points, self.geometry["points_mm"], self.camera)
        self.assertFalse(pose["solved"])
        self.assertEqual(pose["points_raw_px"], points)

    def test_candidate_recall_is_one_to_one_and_identity_is_separate(self):
        truth = {"S_neg": [10., 10.], "CC2": [12., 10.], "S_pos": [25., 10.]}
        frontend = {"attempted": True, "candidate_points_raw_px": [[11., 10.], [25., 10.]],
                    "named_points_raw_px": {"S_neg": [25., 10.], "S_pos": [25., 10.]}}
        result = point_metrics(truth, frontend, 1.5)
        self.assertEqual(result["candidate_matched_count"], 2)
        self.assertEqual(result["named_counts"]["wrong_name"], 1)
        self.assertEqual(result["named_counts"]["missing"], 1)
        self.assertEqual(result["named_counts"]["correct"], 1)

    def test_frontend_exception_after_candidates_does_not_erase_candidates(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        pipeline = DiagnosticFrontend(self.config, detector=lambda *_args, **_kw: [SimpleNamespace(boxes=[
            SimpleNamespace(cls=np.array(0), conf=np.array(.9), xyxy=np.array([[100., 100., 400., 400.]]))])])
        pipeline._load()
        candidate = SimpleNamespace(aperture_center=(20., 25.), fitted_center=None, center=(21., 26.))
        contour_module = MagicMock()
        contour_module.filter_candidate_contours.return_value = [candidate], None, None
        pipeline._modules["latest_candidate_contours"] = contour_module
        with patch.object(pipeline._modules["pose_correspondence"], "select_correspondences", side_effect=ValueError("bad_correspondence")):
            result = pipeline.extract(np.zeros((720, 1280, 3), np.uint8), self.camera, self.geometry)
        self.assertEqual(result["candidate_points_raw_px"], [[96., 101.]])
        self.assertIn("bad_correspondence", result["reason"])
        self.assertEqual(result["named_points_raw_px"], {})


class BoundCaptureDiagnosticTests(CameraDiagnosticTests):
    def setUp(self):
        super().setUp()
        self.root = ROOT / "artifacts/test-tmp" / ("new-diagnostic-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        session = CaptureSession(self.root / "captures", self.camera, "session-batch", metadata={
            "specimen_id": "printed-01", "printing_batch_id": "print-01", "source_kind": "printed_connector",
            "geometry_version": self.geometry["version"], "conditions": {"tilt": "oblique", "distance": "near", "image_position": "edge"}})
        session.save(np.zeros((720, 1280, 3), np.uint8), 1, 10., "hardware_clock", "2026-10-29T01:00:00Z")
        session.finish()
        self.annotations = build_capture_annotations(session.path / "manifest.json", self.root)
        entry = self.annotations["images"][0]
        entry.update(review_status="reviewed", reviewer="synthetic_test", reviewed_at="2026-10-29T01:00:01Z")
        for name, point in self.points.items():
            entry["points"][name].update(xy=point, visibility="visible", source="human")
        self.path = self.root / "reviewed.json"
        self.path.write_text(json.dumps(self.annotations), encoding="utf-8")

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise RuntimeError("invalid_test_cleanup_target")
        shutil.rmtree(self.root)

    def test_bound_schema2_and_manual_variants_have_paired_pixel_sets(self):
        annotations, camera = validate_capture_annotations(self.path, self.root)
        override = copy.deepcopy(camera)
        override.update(version="checked-v1", validated=True)
        override["camera_matrix"][0][0] *= 1.03
        override["intrinsics_sha256"] = intrinsics_signature(override)
        alternate = copy.deepcopy(self.geometry)
        alternate["version"] = "alternate-geometry"
        alternate["points_mm"]["CC1"][2] = 5.
        result = evaluate([(annotations, camera)], [self.geometry, alternate], self.config, root=self.root,
                          camera_overrides=[override], tolerance_px=2.)
        self.assertEqual(len(result["rows"][0]["comparisons"]), 4)
        self.assertEqual(result["stages"]["frontend_evaluated_images"], 0)
        self.assertFalse(result["physical_pose_accuracy_verified"])
        self.assertEqual(result["strata"]["tilt"]["oblique"]["image_count"], 1)
        for comparison in result["rows"][0]["comparisons"].values():
            self.assertEqual(comparison["branches"]["human_all"]["points_raw_px"], self.points)
            self.assertFalse(comparison["final_gate"]["accepted"])
        for variants in result["paired_variants_against_baseline"].values():
            self.assertEqual(variants["human_all"]["image_count"], 1)

    def test_manifest_binding_prevents_specimen_or_camera_edits_in_annotations(self):
        tampered = copy.deepcopy(self.annotations)
        tampered["images"][0]["capture_metadata"]["specimen_id"] = "wrong"
        self.path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_capture_annotations(self.path, self.root)

    def test_override_cannot_change_camera_identity_or_image_resolution(self):
        for key, value in (("image_width", 640), ("device_name", "Intel RealSense D415"), ("serial", "wrong")):
            override = copy.deepcopy(self.camera)
            override.update(version="checked", validated=True)
            override[key] = value
            override["intrinsics_sha256"] = intrinsics_signature(override)
            with self.subTest(key=key), self.assertRaises(ValueError):
                evaluate([(self.annotations, self.camera)], [self.geometry], self.config,
                         root=self.root, camera_overrides=[override], tolerance_px=2.)

    def test_duplicate_images_across_sessions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate_image"):
            evaluate([(self.annotations, self.camera)] * 2, [self.geometry], self.config,
                     root=self.root, tolerance_px=2.)

    def test_capture_specimen_and_print_batch_cannot_cross_splits(self):
        for metadata in ({"specimen_id": "printed-01"}, {"printing_batch_id": "print-01"}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "cross_dataset_splits"):
                CaptureSession(self.root / "captures", self.camera, "different-session", "test", metadata=metadata)


class IntrinsicCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.camera = camera_config()
        self.camera.update(distortion_model="brown_conrady", dist_coeffs=[.05, -.02, .001, .001, 0.])
        self.camera["intrinsics_sha256"] = intrinsics_signature(self.camera)
        self.board = {"columns": 8, "rows": 5, "measured_square_mm": 12., "board_id": "synthetic",
                      "measured_by": "fixture", "measured_at": "2026-10-29", "instrument": "synthetic"}
        self.objects = board_points(self.board)
        self.views = []
        rng = np.random.default_rng(7)
        for i in range(20):
            pixels = project(self.objects, self.camera, (.05 + i*.02, -.25 + (i % 5)*.12, .03*(i % 4)),
                             (-75. + (i % 5)*30., -40. + (i % 3)*15., 280. + i*8.))
            pixels += rng.normal(0, .03, pixels.shape)
            self.views.append({"image": f"image{i}.png", "image_sha256": f"{i:064x}", "pixels": pixels})

    def test_fit_uses_only_training_views_and_reports_heldout_raw_residuals(self):
        sdk = copy.deepcopy(self.camera)
        sdk["camera_matrix"][0][0] *= 1.1
        sdk["intrinsics_sha256"] = intrinsics_signature(sdk)
        fitted, report = calibrate_views(sdk, self.board, self.views[:12], self.views[12:], "fit-v1")
        self.assertFalse(fitted["validated"])
        self.assertEqual(report["summary"]["heldout"]["view_count"], 8)
        self.assertLess(report["summary"]["heldout"]["fitted"]["pooled_corner_rms_px"], .2)
        self.assertLess(report["summary"]["heldout"]["fitted"]["pooled_corner_rms_px"],
                        report["summary"]["heldout"]["sdk"]["pooled_corner_rms_px"])
        changed = copy.deepcopy(self.views[12:])
        for view in changed:
            view["pixels"][:, 0] += np.linspace(0, 2., len(self.objects))
        fitted_again, _ = calibrate_views(sdk, self.board, self.views[:12], changed, "fit-v1")
        self.assertEqual(fitted["camera_matrix"], fitted_again["camera_matrix"])
        self.assertEqual(fitted["dist_coeffs"], fitted_again["dist_coeffs"])

    def test_fit_validation_overlap_and_missing_measurements_fail(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            calibrate_views(self.camera, self.board, self.views[:12], self.views[11:], "v")
        bad = dict(self.board, measured_square_mm=None)
        with self.assertRaisesRegex(ValueError, "measured_positive"):
            calibrate_views(self.camera, bad, self.views[:12], self.views[12:], "v")

    def test_board_validation_uses_realsense_modified_brown_projection(self):
        camera = camera_config()
        pixels = project(self.objects, camera)
        result = evaluate_board_view(self.objects, pixels, camera)
        self.assertLess(result["max_px"], .001)


if __name__ == "__main__":
    unittest.main()
