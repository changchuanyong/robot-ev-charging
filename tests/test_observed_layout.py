"""Supervised association must use held-out-independent templates and real apertures."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live/python"))
from candidate_types import CandidateContour
from keypoint_annotations import LABELS
from observed_layout import fit_observed_layout, select_observed_correspondences, validate_prior


NORMALIZED = {
    "S_neg": [-.35, -.43], "CC2": [0., -.60], "S_pos": [.35, -.43], "CC1": [0., -.12],
    "DC_neg": [-.5, 0.], "DC_pos": [.5, 0.], "A_neg": [-.44, .64], "PE": [0., .60], "A_pos": [.44, .64],
}


def annotations():
    rows = []
    for index, split in enumerate(["train", "train", "train", "val"]):
        points = {label: {"xy": (np.asarray(NORMALIZED[label]) * 240 + [400, 350]).tolist(),
                          "visibility": "visible", "source": "human", "note": ""} for label in LABELS}
        rows.append({"image": f"images/{split}/{index}.jpg", "image_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                     "width": 800, "height": 700, "split": split, "selection_reasons": [],
                     "review_status": "reviewed", "reviewer": "test", "reviewed_at": "2026-10-27T00:00:00Z",
                     "notes": "", "roi_bbox_xyxy": [], "auto_points": {}, "points": points})
    return {"schema_version": 1, "purpose": "development_diagnostic", "independent_test_set": False,
            "labels": list(LABELS), "feature_definition": "inner aperture", "images": rows,
            "provenance": {"batch_report_path": "batch.json", "batch_report_sha256": "a" * 64,
                           "camera_config_path": "camera.json", "camera_config_sha256": "b" * 64,
                           "model_points_path": "model.csv", "model_points_sha256": "c" * 64,
                           "pnp_source_sha256": "d" * 64}}


def candidates(angle=.0, missing=(), without_aperture=()):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result, truth = [], {}
    for index, label in enumerate(LABELS):
        if label in missing:
            continue
        center = rotation @ (np.asarray(NORMALIZED[label]) * 240) + [400, 350]
        radius = 36 if label.startswith("DC_") else 20 if label == "PE" else 8
        phases = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        contour = (center + radius * np.column_stack([np.cos(phases), np.sin(phases)])).astype(np.float32).reshape(-1, 1, 2)
        candidate = CandidateContour(contour, tuple(center), cv2.boundingRect(contour), float(cv2.contourArea(contour)),
                                     float(cv2.arcLength(contour, True)), .95, 1., .8, .2, True, .9)
        candidate.hole_group_id = index
        if label not in without_aperture:
            candidate.aperture_center = tuple(center)
            candidate.aperture_diameter_px = radius * 2
            candidate.contour_source = "adaptive_dark"
        result.append(candidate)
        truth[label] = center
    return result, truth


class ObservedLayoutTests(unittest.TestCase):
    def setUp(self):
        self.annotations = annotations()
        self.prior = fit_observed_layout(self.annotations)

    def test_fit_never_uses_validation_coordinates(self):
        altered = copy.deepcopy(self.annotations)
        for point in altered["images"][-1]["points"].values():
            point["xy"] = [10., 20.]
        self.assertEqual(self.prior, fit_observed_layout(altered))
        self.assertEqual(len(self.prior["training_images"]), 3)
        self.assertTrue(all("train" in row["image"] for row in self.prior["training_images"]))

    def test_fit_excludes_unreviewed_and_occluded_observations(self):
        altered = copy.deepcopy(self.annotations)
        altered["images"][0].update(review_status="draft", reviewer="", reviewed_at=None)
        with self.assertRaisesRegex(ValueError, "three reviewed train"):
            fit_observed_layout(altered)
        altered = copy.deepcopy(self.annotations)
        point = altered["images"][0]["points"]["CC1"]
        point.update(xy=None, visibility="occluded")
        prior = fit_observed_layout(altered)
        self.assertNotIn("CC1", prior["templates"][0]["points"])

    def test_rotation_and_full_image_offset_preserve_hole_identity(self):
        for angle in (0., .7, 2.7):
            with self.subTest(angle=angle):
                raw, truth = candidates(angle)
                selection = select_observed_correspondences(raw, (700, 800), self.prior, (100, 50))
                self.assertTrue(selection["diagnostics"]["selection_quality_ok"])
                self.assertEqual(set(selection["image_points"]), set(LABELS))
                for label in LABELS:
                    np.testing.assert_allclose(selection["image_points"][label], truth[label] + [100, 50], atol=.01)

    def test_large_roi_does_not_remove_observed_dc_pair(self):
        raw, _ = candidates()
        small = select_observed_correspondences(raw, (700, 800), self.prior)
        padded = select_observed_correspondences(raw, (3000, 4000), self.prior)
        self.assertEqual(small["image_points"], padded["image_points"])
        self.assertTrue(padded["diagnostics"]["selection_quality_ok"])

    def test_missing_hole_is_never_filled_with_template_prediction(self):
        raw, _ = candidates(missing=("A_neg",))
        selection = select_observed_correspondences(raw, (700, 800), self.prior)
        self.assertNotIn("A_neg", selection["image_points"])
        self.assertEqual(len(selection["image_points"]), 8)

    def test_outer_ring_support_is_not_emitted_as_aperture_measurement(self):
        raw, _ = candidates(without_aperture=("CC1",))
        selection = select_observed_correspondences(raw, (700, 800), self.prior)
        self.assertNotIn("CC1", selection["image_points"])
        self.assertIn("CC1", selection["diagnostics"]["excluded_without_aperture"])

    def test_no_cc2_aperture_cannot_certify_direction(self):
        raw, _ = candidates(without_aperture=("CC2",))
        selection = select_observed_correspondences(raw, (700, 800), self.prior)
        self.assertFalse(selection["diagnostics"]["selection_quality_ok"])
        self.assertNotIn("CC2", selection["image_points"])

    def test_ignored_outer_ring_alternatives_do_not_make_apertures_ambiguous(self):
        raw, _ = candidates(without_aperture=("A_neg",))
        ring = raw[6]
        alternate = copy.deepcopy(ring)
        alternate.hole_group_id = 20
        for candidate, shift in ((ring, -10.), (alternate, 10.)):
            candidate.contour[:, 0, 0] += shift
            candidate.center = (candidate.center[0] + shift, candidate.center[1])
            candidate.bbox = cv2.boundingRect(candidate.contour)
        raw.append(alternate)
        prior = copy.deepcopy(self.prior)
        for template, shift in zip(prior["templates"], (-10., 10., -10.)):
            template["points"]["A_neg"][0] += shift / 240
        selection = select_observed_correspondences(raw, (700, 800), prior)
        self.assertEqual(set(selection["image_points"]), set(LABELS) - {"A_neg"})
        self.assertFalse(selection["diagnostics"]["ambiguous"])
        self.assertTrue(selection["diagnostics"]["selection_quality_ok"])

    def test_nonfinite_template_is_rejected(self):
        bad = copy.deepcopy(self.prior)
        bad["templates"][0]["points"]["CC2"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid normalized"):
            validate_prior(bad)

    def test_different_aperture_assignments_still_require_clear_margin(self):
        raw, _ = candidates()
        alternate = copy.deepcopy(raw[6])
        alternate.hole_group_id = 20
        for candidate, shift in ((raw[6], -10.), (alternate, 10.)):
            candidate.contour[:, 0, 0] += shift
            candidate.center = (candidate.center[0] + shift, candidate.center[1])
            candidate.aperture_center = candidate.center
            candidate.bbox = cv2.boundingRect(candidate.contour)
        raw.append(alternate)
        prior = copy.deepcopy(self.prior)
        for template, shift in zip(prior["templates"], (-10., 10., -10.)):
            template["points"]["A_neg"][0] += shift / 240
        selection = select_observed_correspondences(raw, (700, 800), prior)
        self.assertTrue(selection["diagnostics"]["ambiguous"])
        self.assertFalse(selection["diagnostics"]["selection_quality_ok"])


if __name__ == "__main__":
    unittest.main()
