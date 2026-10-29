"""Annotation provenance and fair PnP comparisons, without devices or YOLO weights."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live" / "python"))
import pnp2
import evaluate_keypoint_annotations as evaluator
from evaluate_keypoint_annotations import evaluate_annotations, validate_inputs, validate_output_path
from keypoint_annotations import LABELS, sha256_file, validate_annotations, visible_points


class KeypointAnnotationTests(unittest.TestCase):
    def setUp(self):
        temporary_parent = ROOT / "artifacts" / "test-tmp"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        self.root = temporary_parent / ("keypoint-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.remove_fixture)
        self.model = pnp2.load_object_points_from_csv(ROOT / "config/charging_port_model.csv")
        self.matrix = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
        self.distortion = np.zeros(5)
        projection = cv2.projectPoints(np.array(list(self.model.values())), np.array([.25, -.3, .1]),
                                      np.array([0., 0., 400.]), self.matrix, self.distortion)[0].reshape(-1, 2)
        self.coordinates = dict(zip(LABELS, projection.tolist()))
        (self.root / "images").mkdir()
        (self.root / "images/a.png").write_bytes(cv2.imencode(".png", np.zeros((1080, 1920, 3), np.uint8))[1].tobytes())
        (self.root / "camera.json").write_text(json.dumps({"camera_matrix": self.matrix.tolist(),
            "dist_coeffs": self.distortion.tolist(), "image_width": 1920, "image_height": 1080,
            "is_placeholder": True}), encoding="utf-8")
        (self.root / "model.csv").write_bytes((ROOT / "config/charging_port_model.csv").read_bytes())
        self.batch_row = {"image": "images/a.png", "image_sha256": sha256_file(self.root / "images/a.png"),
                          "split": "train", "image_points_full": copy.deepcopy(self.coordinates),
                          "pnp_ok": True, "pose_quality_ok": True, "pose_rejection_reasons": [],
                          "roi_bbox_xyxy": [800, 400, 1100, 700], "pnp_reproj_error_px": 0.}
        self.batch = {"provenance": {"camera_config_sha256": sha256_file(self.root / "camera.json"),
                                    "model_points_sha256": sha256_file(self.root / "model.csv"),
                                    "source_sha256": {"pnp2.py": sha256_file(Path(pnp2.__file__))}},
                      "rows": [self.batch_row]}
        self.report_path = self.root / "batch.json"
        self.report_path.write_text(json.dumps(self.batch), encoding="utf-8")
        self.entry = {"image": "images/a.png", "image_sha256": self.batch_row["image_sha256"],
                      "width": 1920, "height": 1080, "split": "train", "selection_reasons": ["synthetic"],
                      "review_status": "draft", "reviewer": "", "reviewed_at": None, "notes": "synthetic fixture",
                      "roi_bbox_xyxy": self.batch_row["roi_bbox_xyxy"], "auto_points": copy.deepcopy(self.coordinates),
                      "points": {name: {"xy": xy, "visibility": "visible", "source": "ai_visual", "note": ""}
                                 for name, xy in self.coordinates.items()}}
        self.annotations = {"schema_version": 1, "purpose": "development_diagnostic", "independent_test_set": False,
                            "labels": list(LABELS), "feature_definition": "synthetic projected centers",
                            "provenance": {"batch_report_path": "batch.json", "batch_report_sha256": sha256_file(self.report_path),
                                           "camera_config_path": "camera.json", "model_points_path": "model.csv",
                                           "camera_config_sha256": self.batch["provenance"]["camera_config_sha256"],
                                           "model_points_sha256": self.batch["provenance"]["model_points_sha256"],
                                           "pnp_source_sha256": sha256_file(Path(pnp2.__file__))}, "images": [self.entry]}

    def remove_fixture(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts" / "test-tmp").resolve()):
            raise ValueError("Test cleanup path escaped its fixture directory")
        shutil.rmtree(self.root)

    def evaluate(self, include_drafts=False):
        return evaluate_annotations(self.annotations, {"images/a.png": self.batch_row}, self.model,
                                    self.matrix, self.distortion, include_drafts=include_drafts)

    def mark_reviewed(self):
        self.entry.update(review_status="reviewed", reviewer="test reviewer", reviewed_at="2026-10-27T12:00:00+08:00")

    def test_drafts_are_excluded_by_default_without_running_solver(self):
        with patch("evaluate_keypoint_annotations.solve_points") as solve:
            report = self.evaluate()
        solve.assert_not_called()
        self.assertEqual(report["pending_human_review"], ["images/a.png"])
        self.assertEqual(report["groups"]["human_reviewed"]["image_count"], 0)
        self.assertNotIn("draft_diagnostic", report["groups"])

    def test_explicit_draft_diagnosis_never_enters_reviewed_aggregate(self):
        report = self.evaluate(include_drafts=True)
        self.assertEqual(report["groups"]["human_reviewed"]["image_count"], 0)
        group = report["groups"]["draft_diagnostic"]
        self.assertEqual(group["image_count"], 1)
        self.assertFalse(group["human_reviewed"])
        self.assertFalse(group["physical_pose_accuracy_verified"])

    def test_reviewed_requires_a_person_complete_statuses_and_timestamp(self):
        self.entry["review_status"] = "reviewed"
        with self.assertRaisesRegex(ValueError, "requires reviewer"):
            validate_annotations(self.annotations)
        self.mark_reviewed()
        self.entry["points"]["PE"] = {"xy": None, "visibility": "unlabeled", "source": "unlabeled", "note": ""}
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            validate_annotations(self.annotations)
        self.entry["points"]["PE"].update(visibility="uncertain", source="human")
        validate_annotations(self.annotations)
        self.entry["reviewed_at"] = "2026-10-27T12:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            validate_annotations(self.annotations)

    def test_occluded_uncertain_and_out_of_frame_never_supply_pnp_points(self):
        for name, status in zip(LABELS, ("occluded", "uncertain", "out_of_frame")):
            self.entry["points"][name].update(xy=None, visibility=status)
        validate_annotations(self.annotations)
        self.assertEqual(len(visible_points(self.entry)), 6)
        report = self.evaluate(include_drafts=True)
        row = report["rows"]["draft_diagnostic"][0]
        self.assertEqual(row["annotation_visible"]["point_count"], 6)
        self.assertEqual(row["automatic_all"]["point_count"], 9)
        self.assertEqual(row["automatic_common"]["point_count"], 6)
        self.entry["points"]["S_neg"]["xy"] = [10., 20.]
        with self.assertRaisesRegex(ValueError, "Nonvisible"):
            validate_annotations(self.annotations)

    def test_invalid_coordinates_and_duplicate_content_are_rejected(self):
        for bad in ([float("nan"), 1.], [1., float("inf")], [1920., 1.], [-1., 1.], [True, 1.]):
            with self.subTest(xy=bad):
                modified = copy.deepcopy(self.annotations)
                modified["images"][0]["points"]["PE"]["xy"] = bad
                with self.assertRaises(ValueError):
                    validate_annotations(modified)
        duplicate = copy.deepcopy(self.entry)
        duplicate["image"] = "images/b.png"
        self.annotations["images"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            validate_annotations(self.annotations)

    def test_paths_cannot_escape_repository(self):
        self.entry["image"] = "../a.png"
        with self.assertRaisesRegex(ValueError, "repository-relative"):
            validate_annotations(self.annotations)

    def test_valid_inputs_are_bound_to_report_and_original_pixels(self):
        rows = validate_inputs(self.annotations, self.batch, self.root, self.report_path)
        self.assertEqual(list(rows), ["images/a.png"])
        self.entry["auto_points"]["PE"][0] += 1.
        with self.assertRaisesRegex(ValueError, "Automatic points changed"):
            validate_inputs(self.annotations, self.batch, self.root, self.report_path)

    def test_changed_image_bytes_are_rejected(self):
        with (self.root / "images/a.png").open("ab") as file:
            file.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Image hash changed"):
            validate_inputs(self.annotations, self.batch, self.root, self.report_path)

    def test_changed_report_camera_model_and_solver_hashes_are_rejected(self):
        for field in ("batch_report_sha256", "camera_config_sha256", "model_points_sha256", "pnp_source_sha256"):
            with self.subTest(field=field):
                modified = copy.deepcopy(self.annotations)
                modified["provenance"][field] = "0" * 64
                with self.assertRaisesRegex(ValueError, "hash changed"):
                    validate_inputs(modified, self.batch, self.root, self.report_path)

    def test_wrong_image_dimensions_are_rejected(self):
        self.entry["width"] = 1921
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            validate_inputs(self.annotations, self.batch, self.root, self.report_path)

    def test_exact_synthetic_points_recover_equal_common_set_residuals(self):
        self.mark_reviewed()
        report = self.evaluate()
        group = report["groups"]["human_reviewed"]
        self.assertEqual(group["common_name_point_count"], 9)
        self.assertEqual(group["center_deviation_px"]["max_px"], 0.)
        paired = group["paired_same_images_and_point_names"]
        self.assertEqual(paired["image_count"], 1)
        self.assertLess(paired["annotation_image_mean_residual_px"]["max_px"], 1e-6)
        self.assertAlmostEqual(paired["automatic_minus_annotation_mean_px"]["mean_px"], 0.)

    def test_missing_automatic_points_use_identical_names_in_paired_comparison(self):
        del self.entry["auto_points"]["S_neg"]
        self.entry["auto_points"]["A_pos"][0] += 40.
        report = self.evaluate(include_drafts=True)
        row = report["rows"]["draft_diagnostic"][0]
        self.assertEqual(row["missing_automatic_labels"], ["S_neg"])
        self.assertEqual(row["annotation_common"]["labels"], row["automatic_common"]["labels"])
        self.assertEqual(row["annotation_visible"]["point_count"], 9)
        self.assertEqual(row["annotation_common"]["point_count"], 8)
        self.assertAlmostEqual(row["center_deviation_px"]["A_pos"], 40.)
        self.assertGreater(row["automatic_common"]["mean_px"], row["annotation_common"]["mean_px"] + 1.)

    def test_no_automatic_pose_keeps_failure_in_denominator(self):
        self.entry["auto_points"] = {}
        report = self.evaluate(include_drafts=True)
        group = report["groups"]["draft_diagnostic"]
        self.assertEqual(group["image_count"], 1)
        self.assertEqual(group["branches"]["automatic_all"]["failed_images"], ["images/a.png"])
        self.assertEqual(group["branches"]["annotation_visible"]["solved_images"], 1)
        self.assertEqual(group["paired_same_images_and_point_names"]["image_count"], 0)
        self.assertIsNone(group["branches"]["automatic_all"]["image_mean_residual_px"]["mean_px"])

    def test_all_decided_point_statuses_require_a_source(self):
        for status in ("visible", "uncertain", "occluded", "out_of_frame"):
            with self.subTest(status=status):
                modified = copy.deepcopy(self.annotations)
                point = modified["images"][0]["points"]["PE"]
                point.update(visibility=status, source="unlabeled", xy=point["xy"] if status == "visible" else None)
                with self.assertRaisesRegex(ValueError, "needs an observation source"):
                    validate_annotations(modified)

    def test_output_cannot_replace_bound_inputs_or_source_files(self):
        annotations_path = self.root / "annotations.json"
        annotations_path.write_text(json.dumps(self.annotations), encoding="utf-8")
        protected = [annotations_path, self.report_path, self.root / "camera.json", self.root / "model.csv",
                     self.root / "images/a.png", Path(pnp2.__file__), Path(evaluator.__file__),
                     Path(evaluator.__file__).with_name("keypoint_annotations.py")]
        before = {path: path.read_bytes() for path in protected}
        for output in protected:
            with self.subTest(output=output):
                with patch.object(sys, "argv", ["evaluate_keypoint_annotations.py", "--annotations", str(annotations_path),
                        "--batch-report", str(self.report_path), "--root", str(self.root), "--output", str(output)]):
                    with patch.object(evaluator, "evaluate_annotations") as evaluate, patch.object(
                            Path, "write_text", side_effect=AssertionError("Output protection reached a write")):
                        with self.assertRaisesRegex(ValueError, "Output must not overwrite"):
                            evaluator.main()
                        evaluate.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in protected})
        validate_output_path(self.root / "diagnostic.json", annotations_path, self.report_path, self.annotations, self.root)


if __name__ == "__main__":
    unittest.main()
