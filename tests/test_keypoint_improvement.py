"""Measured aperture completion and camera-bound offline pose diagnostics."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ev_charging.paths import PROJECT_ROOT
from ev_charging.review.keypoint_suggestions import (
    SuggestionEngine,
    complete_apertures,
    stable_dark_apertures,
)
from ev_charging.vision.algorithms import latest_candidate_contours as contours
from ev_charging.vision.evaluation.evaluate_library_keypoints import pose_diagnostic
from ev_charging.vision.geometry_reference import load_geometry_reference
from tests.governance_helpers import FixedDetector, baseline, camera


class ApertureCompletionTests(unittest.TestCase):
    def selection(self):
        points = {"DC_neg": (40.0, 50.0), "DC_pos": (80.0, 50.0), "PE": (60.0, 75.0), "CC2": (60.0, 25.0)}
        return {
            "image_points": points,
            "association_points": {**points, "S_neg": (43.0, 25.0)},
            "diagnostics": {"ambiguous": False, "structural_support_ok": True},
            "candidates": [SimpleNamespace(layout_name="bottom_mid", aperture_diameter_px=10.0)],
        }

    def candidate(self, xy=(44.0, 25.0), *, review_only=False):
        return SimpleNamespace(
            fit_valid=True,
            aperture_center=xy,
            aperture_diameter_px=5.0,
            review_only_aperture=review_only,
            inside_prior=True,
            contour_source="adaptive_dark",
        )

    def test_no_points_are_generated_without_measured_boundaries(self):
        selection = self.selection()
        points, added = complete_apertures(selection, [], np.zeros(2))
        self.assertEqual(points, selection["image_points"])
        self.assertEqual(added, [])
        points, added = complete_apertures(selection, [self.candidate(review_only=True)], np.zeros(2))
        self.assertEqual(points, selection["image_points"])
        self.assertEqual(added, [])

    def test_unique_measured_center_is_used_without_moving_existing_points(self):
        selection = self.selection()
        points, added = complete_apertures(selection, [self.candidate()], np.zeros(2))
        self.assertEqual(points["S_neg"], (44.0, 25.0))
        self.assertNotEqual(points["S_neg"], selection["association_points"]["S_neg"])
        for name, xy in selection["image_points"].items():
            self.assertEqual(points[name], xy)
        self.assertEqual([item["label"] for item in added], ["S_neg"])

    def test_ambiguous_or_unsupported_layout_is_not_completed(self):
        for key, value in (("ambiguous", True), ("structural_support_ok", False)):
            selection = self.selection()
            selection["diagnostics"][key] = value
            points, added = complete_apertures(selection, [self.candidate()], np.zeros(2))
            self.assertEqual(points, selection["image_points"])
            self.assertEqual(added, [])

    def test_stable_dark_region_uses_observed_ellipse_and_rejects_blank_image(self):
        image = np.full((128, 128, 3), 180, np.uint8)
        self.assertEqual(stable_dark_apertures(image, contours), [])
        cv2.ellipse(image, (64, 64), (18, 12), 25, 0, 360, (25, 25, 25), -1)
        candidates = stable_dark_apertures(image, contours)
        self.assertTrue(candidates)
        self.assertLess(
            min(np.linalg.norm(np.asarray(c.aperture_center) - [64, 64]) for c in candidates), 1.0
        )
        self.assertTrue(all(c.aperture_center == c.fitted_center for c in candidates))

    def test_fixed_photo_recovers_three_missing_labels(self):
        fixture = baseline()
        case = next(row for row in fixture["images"] if Path(row["image"]).name.startswith("0311084e449d"))
        image = cv2.imdecode(
            np.frombuffer((PROJECT_ROOT / case["image"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        engine = SuggestionEngine(PROJECT_ROOT, PROJECT_ROOT / "dataset/vision")
        with patch.object(engine, "_load_detector", return_value=FixedDetector(case["boxes"])):
            prediction = engine.suggest({"status": "ready"}, image, model=fixture["config"], diagnostics=True)
        self.assertEqual(len(prediction["points"]), 9)
        self.assertEqual(
            {row["label"] for row in prediction["diagnostics"]["completed_apertures"]},
            {"S_neg", "A_neg", "A_pos"},
        )
        # Human-edited reference coordinates frozen before this algorithm change.
        references = {"S_neg": [1304.44, 296.66], "A_neg": [1267.06, 429.32], "A_pos": [1363.94, 441.8]}
        for label, xy in references.items():
            self.assertLessEqual(np.linalg.norm(np.asarray(prediction["points"][label]) - xy), 5.0)


class PoseDiagnosticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.library = Path(directory.name)
        self.geometry = load_geometry_reference(PROJECT_ROOT / "config/geometry_cad_socket_openings.json")
        self.camera = camera(1920, 1080)
        points = self.geometry["points_mm"]
        labels = list(points)
        pixels, _ = cv2.projectPoints(
            np.asarray(list(points.values()), float),
            np.array([0.1, -0.05, 0.04]),
            np.array([0.0, 0.0, 450.0]),
            np.asarray(self.camera["camera_matrix"], float),
            np.zeros(5),
        )
        self.points = dict(zip(labels, pixels.reshape(-1, 2).tolist()))
        self.reference = {
            "width": 1920,
            "height": 1080,
            "points": {
                name: {"xy": xy, "visibility": "visible", "source": "human"}
                for name, xy in self.points.items()
            },
        }
        content = json.dumps(self.camera).encode()
        (self.library / "camera.json").write_bytes(content)
        self.record = {
            "camera_calibrations": [
                {"status": "verified", "file": "camera.json", "sha256": hashlib.sha256(content).hexdigest()}
            ]
        }

    def test_pose_requires_bound_camera_and_never_authorizes_motion(self):
        result = pose_diagnostic(
            self.reference, {"points": self.points}, self.record, self.library, [self.geometry]
        )
        self.assertEqual(result["status"], "evaluated")
        self.assertFalse(result["authorizes_robot_motion"])
        self.assertFalse(result["camera"]["independently_validated"])
        pose = result["poses"][self.geometry["version"]]["automatic"]
        self.assertTrue(pose["solved"])
        self.assertEqual(pose["point_count"], 9)
        self.assertLess(pose["rms_px"], 1e-5)

    def test_missing_or_changed_camera_is_not_replaced_with_a_default(self):
        missing = pose_diagnostic(self.reference, {"points": self.points}, {}, self.library, [self.geometry])
        self.assertEqual(missing["status"], "camera_missing_or_ambiguous")
        self.assertEqual(missing["poses"], {})
        (self.library / "camera.json").write_bytes(b"changed")
        changed = pose_diagnostic(
            self.reference, {"points": self.points}, self.record, self.library, [self.geometry]
        )
        self.assertEqual(changed["status"], "invalid_camera")
        self.assertEqual(changed["reason"], "camera_hash_mismatch")

    def test_cad_reference_keeps_xy_and_is_not_print_acceptance(self):
        nominal = load_geometry_reference(PROJECT_ROOT / "config/geometry_nominal.json")
        self.assertEqual(self.geometry["status"], "cad_extracted")
        self.assertFalse(self.geometry["validated"])
        for label, point in self.geometry["points_mm"].items():
            self.assertEqual(point[:2], nominal["points_mm"][label][:2])
            self.assertEqual(point[2], 0 if label in {"DC_neg", "DC_pos", "PE", "CC2"} else -10)
        rotation = np.asarray(self.geometry["cad_to_target"])[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_camera_identity_must_match_its_recorded_source(self):
        self.record["sources"] = [
            {
                "camera_serial": "ANOTHER-CAMERA",
                "camera_sha256": self.record["camera_calibrations"][0]["sha256"],
            }
        ]
        result = pose_diagnostic(
            self.reference, {"points": self.points}, self.record, self.library, [self.geometry]
        )
        self.assertEqual(result["status"], "invalid_camera")
        self.assertEqual(result["reason"], "camera_serial_mismatch")


if __name__ == "__main__":
    unittest.main()
