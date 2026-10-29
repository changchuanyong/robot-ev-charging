"""Reviewed hole centers catch training boxes that crop or miss the target."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train" / "python"))
from check_reviewed_detection_labels import check_annotations, detection_bbox


class ReviewedDetectionLabelTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts" / "test-tmp" / ("reviewed-boxes-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.image = self.root / "yolo_port/images/train/example.png"
        self.image.parent.mkdir(parents=True)
        self.image.write_bytes(cv2.imencode(".png", np.zeros((150, 200, 3), np.uint8))[1].tobytes())
        self.label = self.root / "yolo_port/labels/train/example.txt"
        self.label.parent.mkdir(parents=True)
        self.label.write_text("0 0.5 0.5 0.8 0.8\n", encoding="utf-8")
        names = ["S_neg", "CC2", "S_pos", "CC1", "DC_neg", "DC_pos", "A_neg", "PE", "A_pos"]
        digest = "0" * 64
        self.annotations = {"schema_version": 1, "purpose": "development_diagnostic", "independent_test_set": False,
            "labels": names, "feature_definition": "synthetic visible centers", "provenance": {
                "batch_report_path": "batch.json", "batch_report_sha256": digest,
                "camera_config_path": "camera.json", "camera_config_sha256": digest,
                "model_points_path": "model.csv", "model_points_sha256": digest, "pnp_source_sha256": digest},
            "images": [{"image": "yolo_port/images/train/example.png",
                "image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(), "width": 200, "height": 150,
                "split": "train", "selection_reasons": ["synthetic"], "review_status": "reviewed",
                "reviewer": "test", "reviewed_at": "2026-10-27T12:00:00+08:00", "notes": "",
                "roi_bbox_xyxy": [], "auto_points": {},
                "points": {name: {"xy": [50 + 10 * i, 60], "visibility": "visible", "source": "human", "note": ""}
                           for i, name in enumerate(names)}}]}

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("Test fixture outside expected directory")
        shutil.rmtree(self.root)

    def test_valid_box_contains_every_reviewed_visible_point(self):
        report = check_annotations(self.annotations, self.root)
        self.assertEqual(report["reviewed_images"], 1)
        self.assertEqual(report["failed_images"], 0)
        self.assertEqual(report["outside_point_count"], 0)
        np.testing.assert_allclose(report["rows"][0]["bbox_xyxy"], [20., 15., 180., 135.])

    def test_shifted_box_reports_exact_excluded_holes(self):
        self.label.write_text("0 0.25 0.5 0.4 0.8\n", encoding="utf-8")
        report = check_annotations(self.annotations, self.root)
        self.assertEqual(report["failed_images"], 1)
        self.assertEqual(set(report["rows"][0]["outside_points"]), {"DC_pos", "A_neg", "PE", "A_pos"})

    def test_occluded_points_and_drafts_are_not_treated_as_observations(self):
        entry = self.annotations["images"][0]
        entry["points"]["A_pos"].update(xy=None, visibility="occluded")
        report = check_annotations(self.annotations, self.root)
        self.assertEqual(report["rows"][0]["visible_point_count"], 8)
        entry.update(review_status="draft", reviewer="", reviewed_at=None)
        report = check_annotations(self.annotations, self.root)
        self.assertEqual(report["reviewed_images"], 0)
        self.assertEqual(len(report["skipped_draft_images"]), 1)

    def test_changed_image_and_missing_label_fail(self):
        changed = copy.deepcopy(self.annotations)
        changed["images"][0]["image_sha256"] = "f" * 64
        report = check_annotations(changed, self.root)
        self.assertEqual(report["failed_images"], 1)
        self.assertIn("content differs", report["rows"][0]["errors"][0])
        self.label.unlink()
        self.assertEqual(check_annotations(self.annotations, self.root)["failed_images"], 1)

    def test_malformed_or_nonfinite_labels_fail(self):
        for content in ("0 nan 0.5 0.8 0.8", "0 0.5 0.5 -0.1 0.8", "0 0.5 0.5 0.8", "1 0.5 0.5 0.8 0.8"):
            with self.subTest(content=content):
                self.label.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    detection_bbox(self.label, 200, 150)


if __name__ == "__main__":
    unittest.main()
