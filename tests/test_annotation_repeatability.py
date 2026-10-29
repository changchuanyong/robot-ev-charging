import copy
import unittest

from vision.annotation_repeatability import compare_passes
from vision.geometry_reference import LABELS


def annotations():
    return {
        "schema_version": 2, "purpose": "development_diagnostic", "independent_test_set": False,
        "labels": list(LABELS), "feature_definition": "synthetic aperture centers",
        "provenance": {"kind": "capture_session", "manifest_path": "capture/manifest.json",
                       "camera_config_path": "capture/camera.json", "manifest_sha256": "a"*64,
                       "camera_config_sha256": "b"*64, "session_id": "s1", "batch_id": "b1",
                       "camera_serial": "synthetic"},
        "images": [{"image": "capture/image.png", "image_sha256": "c"*64, "width": 640, "height": 480,
                    "split": "train", "session_id": "s1", "batch_id": "b1", "camera_serial": "synthetic",
                    "selection_reasons": ["fixture"], "notes": "", "review_status": "reviewed",
                    "reviewer": "tester", "reviewed_at": "2026-10-29T12:00:00+08:00", "roi_bbox_xyxy": [],
                    "auto_points": {}, "points": {name: {"xy": [100+i*10, 200], "visibility": "visible",
                                                         "source": "human", "note": ""}
                                                  for i, name in enumerate(LABELS)}}],
    }


class AnnotationRepeatabilityTests(unittest.TestCase):
    def test_known_pixel_difference_does_not_freeze_tolerance(self):
        first = annotations()
        second = copy.deepcopy(first)
        for point in second["images"][0]["points"].values():
            point["xy"][0] += 3
            point["xy"][1] += 4
        report = compare_passes(first, second, {"independent_blind_passes": True, "record": "protocol-1"})
        self.assertEqual(report["point_count"], 9)
        self.assertEqual(report["p95_distance_px"], 5)
        self.assertFalse(report["tolerance_frozen"])
        self.assertFalse(report["count_budget_met"])

    def test_visibility_disagreement_is_retained(self):
        first = annotations()
        second = copy.deepcopy(first)
        second["images"][0]["points"]["PE"].update(xy=None, visibility="occluded")
        report = compare_passes(first, second, {"independent_blind_passes": True, "record": "protocol-1"})
        self.assertEqual(report["point_count"], 8)
        self.assertEqual(report["visibility_disagreements"][0]["label"], "PE")

    def test_different_images_or_unrecorded_blind_protocol_rejected(self):
        first = annotations()
        second = copy.deepcopy(first)
        with self.assertRaisesRegex(ValueError, "blind_annotation_protocol"):
            compare_passes(first, second, {})
        second["images"][0]["image_sha256"] = "d"*64
        with self.assertRaisesRegex(ValueError, "identical_image"):
            compare_passes(first, second, {"independent_blind_passes": True, "record": "protocol-1"})


if __name__ == "__main__":
    unittest.main()
