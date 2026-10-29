"""Reviewed-label metrics must retain missing points and explicit visibility conflicts."""
import copy
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live/python"))
from evaluate_labeled_regression import compare_labeled, markdown_report, validate_comparison_inputs
from keypoint_annotations import LABELS, sha256_file
import pnp2


class LabeledRegressionTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("labeled-regression-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        (self.root / "images").mkdir()
        image = self.root / "images/a.png"
        image.write_bytes(cv2.imencode(".png", np.zeros((240, 320, 3), np.uint8))[1].tobytes())
        camera = self.root / "camera.json"
        camera.write_text(json.dumps({"image_width": 320, "image_height": 240}), encoding="utf-8")
        model = self.root / "model.csv"
        model.write_bytes((ROOT / "config/charging_port_model.csv").read_bytes())
        coordinates = {name: [20. + i * 25, 50. + (i % 3) * 25] for i, name in enumerate(LABELS)}
        self.row = {"image": "images/a.png", "image_sha256": sha256_file(image), "split": "train",
                    "image_points_full": copy.deepcopy(coordinates), "pnp_ok": True, "pose_quality_ok": True,
                    "roi_bbox_xyxy": [0, 0, 320, 240], "pose_rejection_reasons": []}
        self.before = {"provenance": {"camera_config_sha256": sha256_file(camera),
                                      "model_points_sha256": sha256_file(model), "weights_sha256": "old-weights",
                                      "source_sha256": {"pnp2.py": sha256_file(Path(pnp2.__file__)), "pose_correspondence.py": "old-source"}},
                       "rows": [self.row]}
        self.before_path = self.root / "before.json"
        self.before_path.write_text(json.dumps(self.before), encoding="utf-8")
        self.entry = {"image": "images/a.png", "image_sha256": self.row["image_sha256"],
                      "width": 320, "height": 240, "split": "train", "selection_reasons": ["synthetic"],
                      "review_status": "reviewed", "reviewer": "fixture", "reviewed_at": "2026-10-27T12:00:00+08:00",
                      "notes": "synthetic", "roi_bbox_xyxy": self.row["roi_bbox_xyxy"],
                      "auto_points": copy.deepcopy(coordinates),
                      "points": {name: {"xy": xy, "visibility": "visible", "source": "human", "note": ""}
                                 for name, xy in coordinates.items()}}
        self.annotations = {"schema_version": 1, "purpose": "development_diagnostic", "independent_test_set": False,
                            "labels": list(LABELS), "feature_definition": "synthetic named image points",
                            "provenance": {"batch_report_path": "before.json", "batch_report_sha256": sha256_file(self.before_path),
                                           "camera_config_path": "camera.json", "camera_config_sha256": sha256_file(camera),
                                           "model_points_path": "model.csv", "model_points_sha256": sha256_file(model),
                                           "pnp_source_sha256": sha256_file(Path(pnp2.__file__))}, "images": [self.entry]}
        self.after = copy.deepcopy(self.before)
        self.after_path = self.root / "after.json"
        self.write_after()

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("Test cleanup outside intended directory")
        shutil.rmtree(self.root)

    def write_after(self):
        self.after_path.write_text(json.dumps(self.after), encoding="utf-8")

    def compare(self):
        return compare_labeled(self.annotations, {"images/a.png": self.row},
                               {row["image"]: row for row in self.after["rows"]})

    def validate(self):
        self.write_after()
        return validate_comparison_inputs(self.annotations, self.before, self.after,
                                          self.root, self.before_path, self.after_path)

    def test_fewer_outputs_cannot_hide_missing_reference_points(self):
        self.after["rows"][0].update(image_points_full={}, pnp_ok=False, pose_quality_ok=False)
        report = self.compare()
        group = report["groups"]["all"]
        self.assertEqual(group["before"]["correct_visible_points"], 9)
        self.assertEqual(group["after"]["visible_reference_count"], 9)
        self.assertEqual(group["after"]["correct_visible_points"], 0)
        self.assertEqual(group["after"]["missing_visible_points"], 9)
        self.assertEqual(group["after"]["fully_supported_gate_pass_count"], 0)
        self.assertIsNone(group["after"]["center_error_on_returned_visible_points"]["mean_px"])
        self.assertEqual(group["paired_same_visible_labels"]["after"]["point_count"], 0)

    def test_occluded_output_is_separate_and_a_gate_conflict(self):
        self.entry["points"]["A_pos"].update(xy=None, visibility="occluded")
        self.after["rows"][0]["image_points_full"]["S_neg"][0] += 15
        group = self.compare()["groups"]["all"]["after"]
        self.assertEqual(group["visible_reference_count"], 8)
        self.assertEqual(group["correct_visible_points"], 7)
        self.assertEqual(group["wrong_visible_points"], 1)
        self.assertEqual(group["unsupported_nonvisible_outputs"], 1)
        self.assertEqual(group["center_error_on_returned_visible_points"]["point_count"], 8)
        self.assertEqual(group["labeled_failure_gate_pass_images"], ["images/a.png"])

    def test_partial_clean_gate_pass_is_not_complete_coverage(self):
        del self.after["rows"][0]["image_points_full"]["S_neg"]
        group = self.compare()["groups"]["all"]["after"]
        self.assertEqual(group["labeled_failure_gate_pass_count"], 0)
        self.assertEqual(group["gate_pass_with_missing_visible_count"], 1)
        self.assertEqual(group["fully_supported_gate_pass_count"], 0)
        self.assertEqual(group["correct_visible_points"], 8)

    def test_swapped_labels_are_wrong_even_when_both_land_on_holes(self):
        output = self.after["rows"][0]["image_points_full"]
        output["DC_neg"], output["DC_pos"] = output["DC_pos"], output["DC_neg"]
        self.assertEqual(self.compare()["groups"]["all"]["after"]["wrong_visible_points"], 2)

    def test_fixed_tolerance_includes_ten_pixels_and_keeps_paired_labels(self):
        self.after["rows"][0]["image_points_full"]["S_neg"][0] += 10
        del self.after["rows"][0]["image_points_full"]["S_pos"]
        group = self.compare()["groups"]["all"]
        self.assertEqual(group["after"]["correct_visible_points"], 8)
        self.assertEqual(group["after"]["wrong_visible_points"], 0)
        self.assertEqual(group["paired_same_visible_labels"]["before"]["point_count"], 8)
        self.assertEqual(group["paired_same_visible_labels"]["after"]["point_count"], 8)

    def test_unreviewed_images_are_pending_and_not_scored(self):
        self.entry.update(review_status="draft", reviewer="", reviewed_at=None)
        report = self.compare()
        self.assertEqual(report["pending_human_review"], ["images/a.png"])
        self.assertEqual(report["groups"]["all"]["reviewed_images"], 0)
        self.assertEqual(report["groups"]["all"]["after"]["visible_reference_count"], 0)

    def test_new_weights_and_frontend_source_are_allowed(self):
        self.after["provenance"]["weights_sha256"] = "new-weights"
        self.after["provenance"]["source_sha256"]["pose_correspondence.py"] = "new-source"
        old, new = self.validate()
        self.assertEqual(old["images/a.png"]["image_sha256"], new["images/a.png"]["image_sha256"])

    def test_after_hash_missing_image_and_split_changes_are_rejected(self):
        original = copy.deepcopy(self.after)
        for change, error in (({"image_sha256": "0" * 64}, "image hash"),
                              ({"image": "images/b.png"}, "missing from after"),
                              ({"split": "val"}, "split changed")):
            with self.subTest(change=change):
                self.after = copy.deepcopy(original)
                self.after["rows"][0].update(change)
                with self.assertRaisesRegex(ValueError, error):
                    self.validate()

    def test_fixed_camera_model_and_solver_cannot_change(self):
        original = copy.deepcopy(self.after)
        for field in ("camera_config_sha256", "model_points_sha256", "pnp2.py"):
            self.after = copy.deepcopy(original)
            if field == "pnp2.py":
                self.after["provenance"]["source_sha256"][field] = "0" * 64
            else:
                self.after["provenance"][field] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash changed"):
                self.validate()

    def test_bad_output_pixels_and_duplicate_rows_are_rejected(self):
        self.after["rows"][0]["image_points_full"]["S_neg"] = [float("nan"), 1.]
        with self.assertRaises(ValueError):
            self.validate()
        self.after = copy.deepcopy(self.before)
        self.after["rows"].append(copy.deepcopy(self.row))
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            self.validate()

    def test_markdown_contains_all_denominators_and_development_boundary(self):
        text = markdown_report(self.compare())
        self.assertIn("正确点", text)
        self.assertIn("漏点", text)
        self.assertIn("不是独立测试集", text)
        self.assertIn("| all | 1 | 9 |", text)


if __name__ == "__main__":
    unittest.main()
