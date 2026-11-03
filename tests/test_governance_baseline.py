"""Frozen images, control decisions and capture integrity before governance."""

import hashlib
import json
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
    def test_prefitted_candidates_preserve_measurements_without_refitting(self):
        from copy import deepcopy

        from ev_charging.vision.algorithms.center_fit import fit_ellipse_on_kept_candidates
        from ev_charging.vision.pipeline import padded_crop, preprocess_candidates

        case = baseline()["images"][-1]
        image = cv2.imdecode(
            np.frombuffer((PROJECT_ROOT / case["image"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        crop, _ = padded_crop(image, case["boxes"][0]["xyxy"])
        _, candidates, _, _ = preprocess_candidates(crop)
        repeated = fit_ellipse_on_kept_candidates(deepcopy(candidates), verbose=False)
        with patch.object(cv2, "fitEllipse", wraps=cv2.fitEllipse) as fit:
            reused = fit_ellipse_on_kept_candidates(deepcopy(candidates), verbose=False, reuse=True)
        fit.assert_not_called()
        self.assertGreater(len(reused), 0)
        for left, right in zip(repeated, reused):
            self.assertEqual(left.fit_valid, right.fit_valid)
            self.assertEqual(left.fit_reject_reason, right.fit_reject_reason)
            np.testing.assert_allclose(left.fitted_center, right.fitted_center, rtol=0, atol=1e-10)

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

    def test_completed_retry_does_not_rewrite_state(self):
        intake.retry_session(self.session.path, self.library)
        with patch.object(intake, "atomic_capture_json", wraps=intake.atomic_capture_json) as write:
            result = intake.retry_session(self.session.path, self.library)
        self.assertEqual(result["imported"], 1)
        write.assert_not_called()

    def test_display_cache_detects_changes_and_forces_final_refresh(self):
        intake.retry_session(self.session.path, self.library)
        worker = intake.CaptureIntake(self.session, self.library)
        self.addCleanup(worker.close)
        worker._pool.submit(lambda: None).result()
        worker._refresh(attempts=1)
        record = self.library / "records" / f"{self.saved['image_sha256']}.json"
        with patch.object(intake, "_json", wraps=intake._json) as read:
            worker._refresh(attempts=1)
        self.assertNotIn(record, [Path(call.args[0]) for call in read.call_args_list])
        entry = json.loads(record.read_bytes())
        entry["status"] = "excluded"
        record.write_text(json.dumps(entry), encoding="utf-8")
        worker._refresh(attempts=1)
        self.assertEqual(worker.poll()["pending_annotation"], 0)
        with patch.object(intake, "_json", wraps=intake._json) as read:
            worker._refresh(attempts=1, force=True)
        self.assertIn(record, [Path(call.args[0]) for call in read.call_args_list])

    def test_daily_sync_excludes_legacy_sources_but_explicit_import_preserves_them(self):
        from ev_charging.datasets.vision.prepare_port_library import sync_library

        original = self.root / "yolo_port/images/train/old.png"
        original.parent.mkdir(parents=True)
        original.write_bytes((self.session.path / self.saved["image"]).read_bytes())
        self.assertEqual(sync_library(self.root, self.library, include_history=False)["unique_images"], 0)
        self.assertEqual(sync_library(self.root, self.library, include_history=True)["unique_images"], 1)
        self.assertTrue(original.is_file())


class TrainingCoverageTests(unittest.TestCase):
    def test_training_preparation_requests_daily_sync(self):
        from ev_charging.training.train_port import prepare_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "initial.pt"
            weights.write_bytes(b"fixture")
            with patch("ev_charging.datasets.vision.prepare_port_library.sync_library") as sync:
                with patch(
                    "ev_charging.training.train_port.validate_library", side_effect=ValueError("fixture stop")
                ):
                    with self.assertRaisesRegex(ValueError, "fixture stop"):
                        prepare_training(
                            root=root,
                            weights=weights,
                            expected_weights_sha256=hashlib.sha256(b"fixture").hexdigest(),
                        )
            self.assertFalse(sync.call_args.kwargs["include_history"])

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
