"""Boundary-quality regressions using synthetic observed contours only."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live" / "python"))
import center_fit
import latest_candidate_contours as contours


class ContourQualityTests(unittest.TestCase):
    def candidate(self, points):
        candidate = contours.compute_candidate_features(np.asarray(points, dtype=np.int32).reshape(-1, 1, 2), (400, 400))
        self.assertIsNotNone(candidate)
        return candidate

    def fit(self, candidate):
        with contextlib.redirect_stdout(io.StringIO()):
            center_fit.fit_ellipse_on_kept_candidates([candidate])
        return candidate

    def test_rotated_complete_boundary_has_observed_reliable_center(self):
        points = cv2.ellipse2Poly((200, 180), (55, 24), 47, 0, 360, 3)
        candidate = self.fit(self.candidate(points))
        self.assertTrue(candidate.fit_valid, candidate.fit_reject_reason)
        np.testing.assert_allclose(candidate.fitted_center, [200, 180], atol=0.3)
        self.assertGreater(candidate.fit_coverage, 0.95)
        self.assertLess(candidate.fit_residual_px, 1.0)

    def test_partial_arc_is_not_mistaken_for_observed_hole_center(self):
        points = cv2.ellipse2Poly((200, 180), (55, 24), 25, 0, 170, 3)
        candidate = self.fit(self.candidate(points))
        self.assertFalse(candidate.fit_valid)
        self.assertNotEqual(candidate.fit_reject_reason, "")

    def test_sparse_fallback_is_explicitly_unreliable(self):
        candidate = self.fit(self.candidate([[100, 100], [120, 100], [120, 120], [100, 120]]))
        self.assertFalse(candidate.fit_valid)
        self.assertEqual(candidate.fit_method, "moments")
        exported = center_fit.export_fitted_centers([candidate])[0]
        self.assertFalse(exported["fit_valid"])
        self.assertEqual(exported["fit_reject_reason"], "insufficient_points")

    def test_non_elliptic_boundary_is_not_promoted_by_fit_success(self):
        image = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(image, (150, 150), (250, 250), 255, -1)
        boundaries, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidate = self.fit(self.candidate(boundaries[0]))
        self.assertFalse(candidate.fit_valid)
        self.assertEqual(candidate.fit_reject_reason, "boundary_residual")

    def test_nonfinite_fit_is_not_accepted(self):
        points = cv2.ellipse2Poly((200, 180), (55, 24), 47, 0, 360, 3)
        with patch.object(cv2, "fitEllipse", return_value=((np.nan, 180), (48, 110), 47)):
            candidate = self.fit(self.candidate(points))
        self.assertFalse(candidate.fit_valid)
        self.assertEqual(candidate.fit_reject_reason, "nonfinite_fit")

    def extract(self, edges, gray_source=None):
        diagnostics = {}
        with contextlib.redirect_stdout(io.StringIO()):
            kept, _, _ = contours.filter_candidate_contours(
                edges, np.zeros((*edges.shape, 3), dtype=np.uint8), apply_layout=False,
                diagnostics=diagnostics,
                gray_source=gray_source,
            )
        return kept, diagnostics

    def test_offcenter_complete_boundary_survives_raw_extraction(self):
        edges = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(edges, (60, 75), (22, 14), 30, 0, 360, 255, 2)
        kept, diagnostics = self.extract(edges)
        valid = [candidate for candidate in kept if candidate.fit_valid]
        self.assertEqual(len(valid), 1)
        self.assertGreater(valid[0].prior_value, 1)
        self.assertGreater(diagnostics["rejection_counts"].get("duplicate", 0), 0)

    def test_nested_rings_merge_but_separate_holes_remain(self):
        edges = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(edges, (160, 200), 40, 255, 2)
        cv2.circle(edges, (165, 200), 30, 255, 2)
        cv2.circle(edges, (260, 200), 26, 255, 2)
        kept, _ = self.extract(edges)
        self.assertEqual(sum(candidate.fit_valid for candidate in kept), 2)

    def test_clipped_boundary_still_rejected_with_reason(self):
        edges = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(edges, (5, 200), 30, 255, 2)
        kept, diagnostics = self.extract(edges)
        self.assertEqual(kept, [])
        self.assertGreater(diagnostics["rejection_counts"].get("near_border", 0), 0)

    def test_open_canny_ring_is_recovered_without_fabricating_a_boundary(self):
        edges = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(edges, (200, 200), (58, 32), 35, 15, 345, 255, 1)
        kept, _ = self.extract(edges)
        valid = [candidate for candidate in kept if candidate.fit_valid]
        self.assertEqual(len(valid), 1)
        candidate = valid[0]
        self.assertLess(candidate.area, contours.MIN_CONTOUR_AREA)
        self.assertGreater(candidate.ellipse_area, 5000)
        self.assertGreater(candidate.fit_coverage, 0.85)
        np.testing.assert_allclose(candidate.fitted_center, [200, 200], atol=1.0)
        np.testing.assert_allclose(candidate.ellipse_area, np.pi * 58 * 32, rtol=0.03)
        _, expected_prior = contours.inside_structure_prior(*candidate.fitted_center, edges.shape)
        self.assertAlmostEqual(candidate.prior_value, expected_prior)

    def test_short_open_arc_and_nonelliptic_stroke_are_not_recovered(self):
        edges = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(edges, (120, 160), (45, 25), 35, 10, 155, 255, 1)
        cv2.polylines(edges, [np.array([[220, 100], [300, 100], [300, 200], [240, 200]])],
                      False, 255, 1)
        kept, _ = self.extract(edges)
        self.assertFalse(any(candidate.fit_valid for candidate in kept))

    def test_gray_source_recovers_observed_dark_hole_without_canny_edges(self):
        gray = np.tile(np.linspace(140, 220, 400, dtype=np.uint8), (400, 1))
        cv2.ellipse(gray, (200, 190), (42, 23), 35, 0, 360, 45, -1)
        gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
        kept, diagnostics = self.extract(np.zeros_like(gray), gray)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].contour_source, "adaptive_dark")
        self.assertTrue(kept[0].fit_valid)
        np.testing.assert_allclose(kept[0].fitted_center, [200, 190], atol=1.0)
        self.assertEqual(diagnostics["kept_sources"]["adaptive_dark"], 1)
        self.assertIsNotNone(kept[0].aperture_diameter_px)

    def test_shifted_nested_boundaries_are_one_hole(self):
        edges = np.zeros((500, 500), dtype=np.uint8)
        cv2.ellipse(edges, (200, 220), (70, 50), 0, 0, 360, 255, 2)
        cv2.ellipse(edges, (225, 220), (35, 25), 0, 0, 360, 255, 2)
        cv2.circle(edges, (350, 220), 30, 255, 2)
        kept, _ = self.extract(edges)
        self.assertEqual(len(kept), 2)
        self.assertEqual(len({candidate.hole_group_id for candidate in kept}), 2)

    def test_outer_face_and_nearby_non_nested_holes_do_not_merge(self):
        outer = self.fit(self.candidate(cv2.ellipse2Poly((200, 200), (110, 90), 0, 0, 360, 2)))
        central = self.fit(self.candidate(cv2.ellipse2Poly((200, 200), (12, 10), 0, 0, 360, 2)))
        left = self.fit(self.candidate(cv2.ellipse2Poly((170, 200), (30, 25), 0, 0, 360, 2)))
        right = self.fit(self.candidate(cv2.ellipse2Poly((215, 200), (30, 25), 0, 0, 360, 2)))
        self.assertFalse(contours.same_hole_boundaries(outer, central))
        self.assertFalse(contours.same_hole_boundaries(left, right))

    def test_canny_and_gray_source_do_not_duplicate_physical_holes(self):
        gray = np.full((400, 400), 180, dtype=np.uint8)
        edges = np.zeros_like(gray)
        cv2.ellipse(gray, (200, 200), (40, 25), 30, 0, 360, 40, -1)
        cv2.ellipse(edges, (200, 200), (40, 25), 30, 0, 360, 255, 2)
        kept, diagnostics = self.extract(edges, gray)
        self.assertEqual(len(kept), 1)
        self.assertGreater(diagnostics["contour_sources"]["adaptive_dark"], 0)
        self.assertGreater(diagnostics["rejection_counts"]["duplicate"], 0)
        self.assertIsNotNone(kept[0].aperture_diameter_px)
        np.testing.assert_allclose(kept[0].aperture_center, [200, 200], atol=1.0)

    def test_uniform_gray_source_does_not_invent_holes(self):
        gray = np.full((400, 400), 100, dtype=np.uint8)
        kept, _ = self.extract(np.zeros_like(gray), gray)
        self.assertEqual(kept, [])


if __name__ == "__main__":
    unittest.main()
