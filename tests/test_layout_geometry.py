import copy
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live" / "python"))

from candidate_types import CandidateContour
import layout_prior
import pnp2
from pose_correspondence import _minimum_assignment, select_correspondences


class LayoutGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = pnp2.load_object_points_from_csv(ROOT / "config" / "charging_port_model.csv")
        # A 350 px crop encloses the projected ~70 mm connector face plus margin.
        cls.K = np.array([[950., 0, 175], [0, 950., 175], [0, 0, 1.]])
        cls.roi_shape = (350, 350)

    def observations(self, roll=0, tilt=0, missing=()):
        # Model y points up: front view is a pi rotation around camera x.
        rz = cv2.Rodrigues(np.array([0., 0., roll]))[0]
        rx = cv2.Rodrigues(np.array([np.pi + tilt, 0., 0.]))[0]
        rvec = cv2.Rodrigues(rz @ rx)[0]
        labels = list(self.model)
        image = cv2.projectPoints(np.asarray(list(self.model.values())), rvec,
                                  np.array([0., 0., 220.]), self.K, np.zeros(5))[0].reshape(-1, 2)
        candidates, truth = [], {}
        for label, point in zip(labels, image):
            if label in missing:
                continue
            radius = 17 if label.startswith("DC_") else (13 if label == "PE" else 6)
            theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
            contour = (point + radius * np.column_stack([np.cos(theta), np.sin(theta)])).astype(np.float32).reshape(-1, 1, 2)
            area = float(cv2.contourArea(contour))
            prior = ((point[0] - 175) / 140) ** 2 + ((point[1] - 175) / 115.5) ** 2
            candidates.append(CandidateContour(contour, tuple(point), cv2.boundingRect(contour),
                                                area, float(cv2.arcLength(contour, True)),
                                                .95, 1., .8, float(prior), bool(prior <= 1), .85))
            # Synthetic circles are the aperture boundary, not tube silhouettes.
            candidates[-1].contour_source = "adaptive_dark"
            candidates[-1].aperture_diameter_px = 2 * radius
            candidates[-1].aperture_center = tuple(point)
            truth[label] = point
        return candidates, truth

    def test_geometry_comes_from_csv(self):
        normalized = layout_prior.normalized_model_points(self.model)
        np.testing.assert_allclose(normalized["top_left"], [-12 / 34, -22.2 / 34])
        np.testing.assert_allclose(normalized["center"], [0, -.3])
        np.testing.assert_allclose(normalized["bottom_right"], [14.25 / 34, 19.8 / 34])

    def test_exact_assignment_beats_greedy(self):
        assigned = _minimum_assignment(np.array([[1., 2., 100.], [1.1, 100., 100.]]))
        self.assertEqual(list(assigned), [1, 0])

    def test_rotated_and_tilted_correspondences(self):
        for roll, tilt in ((.7, 0), (1.7, .5), (-.8, -.4), (np.pi, .25)):
            with self.subTest(roll=roll, tilt=tilt):
                candidates, truth = self.observations(roll, tilt)
                result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
                best = result["best"]
                self.assertIsNotNone(best)
                self.assertEqual(best["candidate_count"], 9)
                for label, expected in truth.items():
                    np.testing.assert_allclose(best["image_points"][label], expected, atol=.01)
                self.assertLess(best["reproj"], .01)
                self.assertTrue(best["orientation_evidence"]["ok"])

    def test_missing_holes_are_not_fabricated(self):
        candidates, truth = self.observations(.6, .35, missing=("CC2", "A_neg"))
        best = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))["best"]
        self.assertEqual(set(best["image_points"]), set(truth))
        self.assertEqual(best["candidate_count"], 7)
        self.assertEqual(len(best["all_point_errors_px"]), 7)

    def test_raw_coordinates_survive_rectification(self):
        candidates, truth = self.observations(.7)
        offset = np.array([10., 20.])
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5),
                                        offset_xy=offset, point_transform=lambda points: points - offset)
        for label, expected in truth.items():
            np.testing.assert_allclose(result["best"]["image_points"][label], expected + offset, atol=.01)
        self.assertLess(result["best"]["reproj"], .01)

    def test_small_distant_blobs_cannot_be_dc_anchors(self):
        candidates, _ = self.observations()
        small = [c for c in candidates if c.area < 200]
        left, right = copy.deepcopy(small[:2])
        left.center, right.center = (10., 10.), (340., 340.)
        self.assertIsNone(layout_prior.score_main_big_hole_pair(left, right, self.roi_shape))

    def test_background_clutter_does_not_replace_large_dc_holes(self):
        candidates, truth = self.observations(.7, .4)
        original = copy.deepcopy(candidates[0])
        for target in ((20., 25.), (330., 25.), (20., 325.), (330., 325.), (175., 15.)):
            blob = copy.deepcopy(original)
            shift = np.asarray(target) - np.asarray(blob.center)
            blob.contour += shift.astype(np.float32)
            blob.center = target
            blob.aperture_center = target
            blob.bbox = cv2.boundingRect(blob.contour)
            blob.score = .99
            blob.inside_prior = False
            candidates.append(blob)
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertTrue(result["diagnostics"]["selection_quality_ok"])
        for label, expected in truth.items():
            np.testing.assert_allclose(result["best"]["image_points"][label], expected, atol=.01)

    def test_low_residual_five_point_subset_is_diagnostic_only(self):
        candidates, _ = self.observations(.7, .3, missing=("S_neg", "CC2", "S_pos", "A_neg"))
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertIsNotNone(result["best"])
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])
        self.assertFalse(result["best"]["selection_quality_ok"])

    def test_low_residual_background_centers_are_diagnostic_only(self):
        candidates, _ = self.observations(.7, .3)
        for candidate in candidates:
            candidate.inside_prior = False
            candidate.prior_value = 1.4
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertIsNotNone(result["best"])
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])
        self.assertFalse(result["diagnostics"]["structural_support_ok"])

    def test_signal_size_pair_cannot_substitute_missing_dc_holes(self):
        candidates, _ = self.observations(.7, .3, missing=("DC_neg", "DC_pos"))
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertIsNone(result["best"])
        self.assertEqual(result["diagnostics"]["anchor_pair_count"], 0)

    def test_two_small_background_clusters_cannot_replace_target(self):
        original, _ = self.observations()
        clusters = []
        for origin in (np.array([55., 55.]), np.array([285., 55.])):
            for candidate in copy.deepcopy(original):
                candidate.contour = ((candidate.contour - [175., 175.]) * .30 + origin).astype(np.float32)
                candidate.center = tuple((np.asarray(candidate.center) - 175) * .30 + origin)
                candidate.aperture_center = candidate.center
                candidate.aperture_diameter_px *= .30
                candidate.bbox = cv2.boundingRect(candidate.contour)
                candidate.area = float(cv2.contourArea(candidate.contour))
                candidate.inside_prior = False
                clusters.append(candidate)
        result = select_correspondences(clusters, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertIsNone(result["best"])
        self.assertEqual(result["diagnostics"]["anchor_pair_count"], 0)

    def test_true_dc_with_peripheral_signal_pattern_is_rejected(self):
        # Exact projective geometry can still describe background beyond the
        # connector region. DC support alone must not certify those signal holes.
        candidates, _ = self.observations()
        camera = self.K.copy()
        camera[:2, 2] -= 75
        for candidate in candidates:
            candidate.contour -= 75
            candidate.center = tuple(np.asarray(candidate.center) - 75)
            candidate.aperture_center = candidate.center
            candidate.bbox = cv2.boundingRect(candidate.contour)
            x, y = candidate.center
            candidate.prior_value = ((x - 100) / 80) ** 2 + ((y - 100) / 66) ** 2
            candidate.inside_prior = candidate.prior_value <= 1
        result = select_correspondences(candidates, (200, 200), self.model, camera, np.zeros(5))
        self.assertIsNotNone(result["best"])
        self.assertLess(result["best"]["reproj"], .01)
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])
        self.assertFalse(result["diagnostics"]["structural_support_ok"])

    def test_equal_pe_signal_sizes_do_not_certify_orientation(self):
        candidates, _ = self.observations(np.pi, .25)
        pe = candidates[list(self.model).index("PE")]
        pe.contour = (np.asarray(pe.center) + (pe.contour - pe.center) * (6 / 13)).astype(np.float32)
        pe.area = float(cv2.contourArea(pe.contour))
        pe.bbox = cv2.boundingRect(pe.contour)
        pe.aperture_diameter_px = 12
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertIsNotNone(result["best"])
        self.assertFalse(result["best"]["orientation_evidence"]["ok"])
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])

    def test_mixed_unverified_ring_boundaries_do_not_certify_orientation(self):
        candidates, _ = self.observations(np.pi, .25)
        cc2 = candidates[list(self.model).index("CC2")]
        cc2.contour_source = "canny"
        cc2.aperture_center = None
        cc2.aperture_diameter_px = None
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertIsNotNone(result["best"])
        self.assertFalse(result["best"]["orientation_evidence"]["ok"])
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])

    def test_nested_edges_with_one_physical_id_cannot_get_two_labels(self):
        candidates, _ = self.observations(np.pi, .25)
        pe = candidates[list(self.model).index("PE")]
        pe.hole_group_id = 7
        edge = copy.deepcopy(pe)
        shift = np.array([0., 14.])
        edge.contour += shift.astype(np.float32)
        edge.center = tuple(np.asarray(edge.center) + shift)
        edge.bbox = cv2.boundingRect(edge.contour)
        edge.score -= .1
        candidates.append(edge)
        result = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))
        self.assertEqual(result["diagnostics"]["valid_center_count"], 9)
        groups = [getattr(candidate, "hole_group_id", -1) for candidate in result["best"]["candidates"]]
        self.assertEqual(groups.count(7), 1)

    def test_aperture_measurement_preserves_original_contour_center_audit(self):
        candidates, truth = self.observations(.7, .3)
        for candidate in candidates:
            candidate.contour += np.array([2., 3.], dtype=np.float32)
            candidate.center = tuple(np.asarray(candidate.center) + [2., 3.])
            candidate.bbox = cv2.boundingRect(candidate.contour)
            candidate.contour_source = "canny"
        best = select_correspondences(candidates, self.roi_shape, self.model, self.K, np.zeros(5))["best"]
        for label, expected in truth.items():
            np.testing.assert_allclose(best["image_points"][label], expected, atol=.01)
        for center in best["centers"]:
            self.assertEqual(center["measurement_source"], "adaptive_dark")
            np.testing.assert_allclose(np.subtract(center["fitted_center"], center["measurement_center"]),
                                       [2., 3.], atol=.01)


if __name__ == "__main__":
    unittest.main()
