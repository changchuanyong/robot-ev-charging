"""PnP quality regressions with synthetic correspondences; no devices or weights."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "live" / "python"))
import pnp2


class PnpQualityTests(unittest.TestCase):
    def setUp(self):
        self.model = pnp2.load_object_points_from_csv(ROOT / "config/charging_port_model.csv")
        self.names = list(self.model)
        self.K = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
        self.dist = np.zeros(5)
        self.rvec = np.array([0.25, -0.3, 0.1])
        self.tvec = np.array([0., 0., 400.])
        self.objects = np.asarray([self.model[name] for name in self.names])
        self.points = cv2.projectPoints(self.objects, self.rvec, self.tvec, self.K, self.dist)[0].reshape(-1, 2)

    def solve(self, points=None, count=None, model=None, matrix=None, dist=None):
        count = len(self.names) if count is None else count
        points = self.points if points is None else points
        return pnp2.solve_target_pose(
            dict(zip(self.names[:count], points[:count])), self.model if model is None else model,
            self.K if matrix is None else matrix, self.dist if dist is None else dist,
        )

    def test_exact_pose_keeps_all_points_and_recovers_pose(self):
        result = self.solve()
        np.testing.assert_allclose(result["tvec"].reshape(3), self.tvec, atol=1e-6)
        self.assertEqual(result["fit_count"], 9)
        self.assertEqual(result["inlier_count"], 9)
        self.assertEqual(set(result["per_point_errors_px"]), set(self.names))
        self.assertLess(result["reproj_rms_error"], 1e-6)
        self.assertTrue(result["all_in_front"])

    def test_noisy_five_point_ippe_is_refined_without_worsening_full_metrics(self):
        points = self.points[:5] + np.random.default_rng(8).normal(0, 3, (5, 2))
        _, rotations, translations, _ = cv2.solvePnPGeneric(
            self.objects[:5], points, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE,
        )
        original = [pnp2.compute_reprojection_errors(self.objects[:5], points, r, t, self.K, self.dist)
                    for r, t in zip(rotations, translations)]
        result = self.solve(points, count=5)
        self.assertTrue(result["refined"])
        self.assertLess(result["reproj_rms_error"], min(np.sqrt(np.mean(e**2)) for e in original) - 0.01)
        self.assertLessEqual(result["reproj_error"], min(np.mean(e) for e in original))
        self.assertLessEqual(result["max_reproj_error"], min(np.max(e) for e in original))
        self.assertEqual(result["fit_count"], 5)

    def test_bad_correspondence_is_not_reported_as_an_inlier(self):
        for count, offset in ((5, [40., 0.]), (9, [60., 40.])):
            with self.subTest(count=count):
                points = self.points[:count].copy()
                points[-1] += offset
                result = self.solve(points, count=count)
                errors = result["per_point_errors_px"]
                names = self.names[:count]
                expected = [name for name in names if errors[name] <= 10.0]
                self.assertLess(len(expected), len(names))
                self.assertEqual(result["inlier_count"], len(expected))
                self.assertEqual(result["inlier_names"], expected)
                self.assertEqual(result["outlier_names"], [name for name in names if name not in expected])
                self.assertEqual(set(errors), set(names), "Rejected points remain in reported residuals")
                self.assertAlmostEqual(result["reproj_error"], np.mean(list(errors.values())))
                self.assertEqual(result["inlier_threshold_px"], 10.0)

    def test_subset_fit_cannot_win_with_a_worse_full_point_rms(self):
        points = self.points.copy()
        points[-1] += [60., 40.]
        _, rotations, translations, _ = cv2.solvePnPGeneric(
            self.objects, points, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE,
        )
        initial_rms = [np.sqrt(np.mean(pnp2.compute_reprojection_errors(
            self.objects, points, r, t, self.K, self.dist)**2)) for r, t in zip(rotations, translations)]
        result = self.solve(points)
        final_rms = np.sqrt(np.mean(np.asarray(list(result["per_point_errors_px"].values()))**2))
        self.assertLessEqual(final_rms, min(initial_rms) + 1e-8)
        self.assertAlmostEqual(result["selection_error"], final_rms)

    def test_nonfinite_correspondences_and_camera_are_rejected_before_solving(self):
        bad_points = self.points.copy()
        bad_points[0, 0] = np.nan
        bad_model = dict(self.model)
        bad_model[self.names[0]] = (np.inf, 0., 0.)
        bad_matrix = self.K.copy()
        bad_matrix[0, 0] = np.nan
        bad_dist = self.dist.copy()
        bad_dist[0] = np.inf
        for kwargs in ({"points": bad_points}, {"model": bad_model},
                       {"matrix": bad_matrix}, {"dist": bad_dist}):
            with self.subTest(kwargs=kwargs), patch.object(cv2, "solvePnPGeneric") as solver:
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.solve(**kwargs)
                solver.assert_not_called()

    def test_collinear_and_duplicate_correspondences_are_rejected(self):
        collinear = np.column_stack((np.arange(9), np.arange(9)))
        duplicate = self.points.copy()
        duplicate[1] = duplicate[0]
        collinear_model = {name: (float(i), 0., 0.) for i, name in enumerate(self.names)}
        for kwargs in ({"points": collinear}, {"points": duplicate}, {"model": collinear_model}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "degenerate"):
                self.solve(**kwargs)

    def test_invalid_depth_and_nonfinite_solver_results_are_rejected(self):
        for rotation, translation in ((self.rvec, -self.tvec), (np.full(3, np.nan), self.tvec),
                                      (self.rvec, np.full(3, np.inf))):
            with self.subTest(rotation=rotation, translation=translation):
                with patch.object(cv2, "solvePnPGeneric", return_value=(1, [rotation], [translation], None)):
                    with patch.object(cv2, "solvePnPRefineLM") as refine, patch.object(
                            cv2, "solvePnP", return_value=(False, None, None)):
                        with self.assertRaisesRegex(RuntimeError, "valid.*pose"):
                            self.solve(count=5)
                        refine.assert_not_called()

    def test_nonfinite_ippe_uses_full_point_planar_fallback(self):
        with patch.object(cv2, "solvePnPGeneric", return_value=(1, [np.full(3, np.nan)], [self.tvec], None)):
            result = self.solve(count=5)
        self.assertEqual(result["method"], "ITERATIVE_PLANAR")
        self.assertEqual(result["fit_count"], 5)
        self.assertLess(result["reproj_rms_error"], 1e-6)

    def test_bad_refinement_preserves_initial_pose(self):
        bad_results = [(self.rvec, -self.tvec), (np.full(3, np.nan), self.tvec),
                       (self.rvec, self.tvec + [30., 20., 0.])]
        for bad in bad_results:
            with self.subTest(refined=bad):
                with patch.object(cv2, "solvePnPGeneric", return_value=(1, [self.rvec], [self.tvec], None)):
                    with patch.object(cv2, "solvePnPRefineLM", return_value=bad):
                        result = self.solve(count=5)
                np.testing.assert_array_equal(result["rvec"].reshape(3), self.rvec)
                np.testing.assert_array_equal(result["tvec"].reshape(3), self.tvec)
                self.assertFalse(result["refined"])

    def test_refinement_exception_does_not_discard_valid_initial_pose(self):
        with patch.object(cv2, "solvePnPRefineLM", side_effect=cv2.error("refinement failed")):
            result = self.solve(count=5)
        self.assertFalse(result["refined"])
        self.assertLess(result["reproj_error"], 1e-6)

    def test_lower_rms_refinement_is_rejected_if_maximum_error_grows(self):
        points = self.points.copy()
        points[-1] += [60., 40.]
        _, rotations, translations, _ = cv2.solvePnPGeneric(
            self.objects, points, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE,
        )
        rotation, translation = rotations[1], translations[1]
        initial = pnp2.compute_reprojection_errors(self.objects, points, rotation, translation, self.K, self.dist)
        rr, tt = cv2.solvePnPRefineLM(self.objects, points, self.K, self.dist, rotation.copy(), translation.copy())
        refined = pnp2.compute_reprojection_errors(self.objects, points, rr, tt, self.K, self.dist)
        self.assertLess(np.mean(refined**2), np.mean(initial**2))
        self.assertGreater(np.max(refined), np.max(initial))
        with patch.object(cv2, "solvePnPGeneric", return_value=(1, [rotation], [translation], None)):
            with patch.object(pnp2, "solve_pnp_ransac_candidate", return_value=None):
                result = self.solve(points)
        self.assertFalse(result["refined"])
        self.assertAlmostEqual(result["max_reproj_error"], float(np.max(initial)))

    def test_nonplanar_model_still_recovers_pose(self):
        objects = self.objects.copy()
        objects[:, 2] = [0., 1., 4., 2., -2., 3., 1., 5., -3.]
        model = dict(zip(self.names, objects))
        points = cv2.projectPoints(objects, self.rvec, self.tvec, self.K, self.dist)[0].reshape(-1, 2)
        result = self.solve(points, model=model)
        np.testing.assert_allclose(result["tvec"].reshape(3), self.tvec, atol=1e-5)
        self.assertLess(result["reproj_rms_error"], 1e-5)
        self.assertEqual(result["inlier_count"], len(self.names))

    def test_one_invalid_ippe_branch_does_not_hide_valid_branch(self):
        with patch.object(cv2, "solvePnPGeneric", return_value=(2, [np.full(3, np.nan), self.rvec],
                                                               [self.tvec, self.tvec], None)):
            result = self.solve(count=5)
        self.assertTrue(np.isfinite(result["rvec"]).all())
        self.assertLess(result["reproj_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
