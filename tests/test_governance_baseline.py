"""Frozen images, control decisions and capture integrity before governance."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ev_charging.capture import intake
from ev_charging.capture.capture_session import CaptureSession
from ev_charging.control.simulate import run_simulation
from ev_charging.paths import PROJECT_ROOT
from ev_charging.training.train_port import CoverageAudit
from ev_charging.vision.algorithms.roi import extract_best_roi
from ev_charging.vision.geometry_reference import load_geometry_reference
from ev_charging.vision.locate.diagnose import solve_raw_points
from tests.governance_helpers import FixedDetector, assert_json_close, baseline, camera, vision_result


class FrozenVisionTests(unittest.TestCase):
    def test_fixed_image_outputs(self):
        fixture = baseline()
        for case in fixture["images"]:
            with self.subTest(image=case["image"]):
                self.assertEqual(
                    hashlib.sha256((PROJECT_ROOT / case["image"]).read_bytes()).hexdigest(), case["sha256"]
                )
                assert_json_close(self, vision_result(case, fixture["config"]), case["expected"])

    def test_legacy_roi_bounds_and_error_classification(self):
        image = np.zeros((40, 50, 3), np.uint8)
        model = FixedDetector([{"xyxy": [0, 0, 50, 40], "confidence": 0.9}])
        roi, detection, crop, score, padding = extract_best_roi(image, model, 0, 0.25)
        self.assertEqual((detection, crop, score, padding), ((0, 0, 50, 40), (0, 0, 49, 39), 0.9, 20))
        self.assertEqual(roi.shape, (39, 49, 3))
        self.assertEqual(extract_best_roi(image, FixedDetector([]), 0, 0.25), (None,) * 5)

    def test_projected_nine_hole_pose_and_pixel_convention(self):
        model = load_geometry_reference(PROJECT_ROOT / "config/geometry_nominal.json")["points_mm"]
        intrinsics = camera(1920, 1080)
        labels = list(model)
        pixels, _ = cv2.projectPoints(
            np.asarray([model[name] for name in labels], float),
            np.array([0.1, -0.05, 0.04]),
            np.array([0.0, 0.0, 450.0]),
            np.asarray(intrinsics["camera_matrix"], float),
            np.zeros(5),
        )
        points = dict(zip(labels, pixels.reshape(-1, 2).tolist()))
        pose = solve_raw_points(points, model, intrinsics)
        self.assertTrue(pose["solved"], pose)
        self.assertEqual(pose["residual_frame"], "raw_color_pixels")
        self.assertEqual(pose["inlier_count"], 9)
        np.testing.assert_allclose(pose["tvec_mm"], [0.0, 0.0, 450.0], atol=1e-5, rtol=0)
        self.assertLess(pose["rms_px"], 1e-5)


class FrozenControlTests(unittest.TestCase):
    def test_all_software_scenarios(self):
        for expected in baseline()["control"]:
            with self.subTest(scenario=expected["scenario"]):
                result, _ = run_simulation(scenario=expected["scenario"])
                assert_json_close(self, result, expected)


class CaptureIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.library = self.root / "library"
        self.session = CaptureSession(self.root / "captures", camera(), "test-batch")
        self.saved = self.session.save(
            np.zeros((24, 32, 3), np.uint8), 1, 1.0, "hardware_clock", "2026-11-04T00:00:00+00:00"
        )
        # Isolate the split registry; never write the real project's configuration.
        for name, value in (
            ("check_capture_split", None),
            ("ensure_capture_split", None),
            ("read_split", {"groups": {}}),
        ):
            mocked = patch("ev_charging.datasets.vision.dataset_split." + name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_registration_is_idempotent_and_recoverable(self):
        first = intake.retry_session(self.session.path, self.library)
        self.assertEqual((first["imported"], first["pending_sync"]), (1, 0))
        record = self.library / "records" / f"{self.saved['image_sha256']}.json"
        original = record.read_bytes()
        self.assertEqual(intake.retry_session(self.session.path, self.library), first)
        self.assertEqual(record.read_bytes(), original)
        record.unlink()
        self.assertEqual(intake.retry_session(self.session.path, self.library)["imported"], 1)

    def test_corrupt_source_remains_pending(self):
        (self.session.path / self.saved["image"]).write_bytes(b"corrupt")
        result = intake.retry_session(self.session.path, self.library, attempts=1)
        self.assertEqual((result["imported"], result["pending_sync"]), (0, 1))
        self.assertIn("capture_file_hash_mismatch", result["errors"][0])


class TrainingCoverageTests(unittest.TestCase):
    def test_missing_and_unplanned_batches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = CoverageAudit(directory, {"entries": [{"split": "train", "image": "a.png"}]})
            with self.assertRaisesRegex(ValueError, "completed_batch_without_image_provenance"):
                audit.complete_batch(SimpleNamespace())
            with self.assertRaisesRegex(ValueError, "batch_contains_unplanned_images"):
                audit.complete_batch(
                    SimpleNamespace(_coverage_batch_paths=[str(Path(directory) / "other.png")])
                )


if __name__ == "__main__":
    unittest.main()
