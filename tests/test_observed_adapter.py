"""Observed identities survive the PnP adapter; no camera or detector required."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live" / "python"))
from candidate_types import CandidateContour
from layout_prior import LAYOUT_TO_MODEL_LABEL
import observed_layout
import pose_correspondence


class ObservedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.offset = (321, 123)
        self.prior = {"test_prior": True}
        self.K = np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]])
        self.dist = np.zeros(5)
        self.points, self.candidates = {}, []
        for index, (layout, label) in enumerate(LAYOUT_TO_MODEL_LABEL.items()):
            local = (30. + 35. * (index % 3), 25. + 40. * (index // 3))
            point = tuple(np.add(local, self.offset))
            self.points[label] = point
            self.candidates.append(CandidateContour(
                contour=np.zeros((20, 1, 2), dtype=np.int32), center=local,
                bbox=(int(local[0]) - 10, int(local[1]) - 10, 20, 20),
                area=300., perimeter=60., circularity=.9, aspect_ratio=1.,
                fill_ratio=.8, prior_value=.5, inside_prior=True, score=.9,
                layout_name=layout, fitted_center=(local[0] + 2., local[1] - 1.),
                fit_method="ellipse", fit_valid=True, fit_coverage=.99,
                ellipse_axes=(20., 18.), ellipse_angle=0., ellipse_area=280.,
                aperture_center=local, aperture_diameter_px=19.,
                contour_source="adaptive_dark"))
        self.objects = {label: (float(index % 3), float(index // 3), 0.)
                        for index, label in enumerate(self.points)}
        self.selection = {
            "image_points": copy.deepcopy(self.points), "candidates": self.candidates,
            "diagnostics": {"selection_quality_ok": True, "ambiguous": False,
                            "hypothesis_count": 7, "score": 8.,
                            "structural_support_ok": True,
                            "orientation_evidence": {"ok": True}},
        }
        self.pose = {"all_in_front": True, "inlier_count": 9,
                     "matched_names": list(self.points),
                     "per_point_errors_px": {name: 1. for name in self.points},
                     "image_points": np.asarray(list(self.points.values()))}

    def call(self, *, selection=None, pose=None, solve_error=None, transform=None,
             min_correspondence_count=6):
        selection = self.selection if selection is None else selection
        pose = self.pose if pose is None else pose
        with patch.object(observed_layout, "select_observed_correspondences",
                          return_value=selection) as associate, \
                patch.object(pose_correspondence.pnp2, "solve_target_pose",
                             return_value=pose, side_effect=solve_error) as solve:
            result = pose_correspondence.select_correspondences(
                self.candidates, (180, 200), self.objects, self.K, self.dist,
                offset_xy=self.offset, point_transform=transform,
                min_correspondence_count=min_correspondence_count,
                observed_prior=self.prior)
        associate.assert_called_once_with(self.candidates, (180, 200), self.prior, self.offset)
        return result, solve

    def test_rectification_only_changes_pnp_input_not_observations(self):
        seen = []

        def rectify(points):
            seen.append(points.copy())
            return points * [1.1, .9] + [8., -3.]

        result, solve = self.call(transform=rectify)
        solve.assert_called_once()
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], list(self.points.values()))
        transformed, objects, matrix, distortion = solve.call_args.args
        self.assertEqual(list(transformed), list(self.points))
        for label, raw in self.points.items():
            np.testing.assert_allclose(transformed[label], np.array(raw) * [1.1, .9] + [8., -3.])
        self.assertIs(objects, self.objects)
        self.assertIs(matrix, self.K)
        self.assertIs(distortion, self.dist)
        self.assertEqual(result["best"]["image_points"], self.points)
        self.assertEqual(result["hypotheses"][0]["image_points"], self.points)
        for center, candidate in zip(result["best"]["centers"], self.candidates):
            self.assertEqual(center["measurement_center"], list(candidate.aperture_center))
        self.assertTrue(result["best"]["selection_quality_ok"])

    def test_unrectified_pnp_keeps_full_image_pixels_without_double_offset(self):
        result, solve = self.call()
        self.assertEqual(solve.call_args.args[0], self.points)
        self.assertEqual(result["best"]["image_points"], self.points)

    def test_pnp_order_and_fitted_positions_do_not_relabel_observations(self):
        pose = copy.deepcopy(self.pose)
        pose["matched_names"].reverse()
        pose["image_points"] = np.full((len(self.points), 2), -999.)
        pose["per_point_errors_px"] = dict(reversed(list(pose["per_point_errors_px"].items())))
        result, solve = self.call(pose=pose)
        solve.assert_called_once()
        best = result["best"]
        self.assertEqual(best["image_points"], self.points)
        self.assertEqual(list(best["image_points"]), list(self.points))
        self.assertEqual([item["layout_name"] for item in best["centers"]],
                         list(LAYOUT_TO_MODEL_LABEL))

    def test_low_pnp_error_does_not_override_failed_or_ambiguous_association(self):
        for accepted, ambiguous in ((False, False), (True, True), (False, True)):
            with self.subTest(accepted=accepted, ambiguous=ambiguous):
                selection = copy.deepcopy(self.selection)
                selection["diagnostics"].update(selection_quality_ok=accepted, ambiguous=ambiguous)
                result, _ = self.call(selection=selection)
                self.assertTrue(result["best"]["geometric_quality_ok"])
                self.assertFalse(result["best"]["selection_quality_ok"])
                self.assertFalse(result["diagnostics"]["selection_quality_ok"])
                self.assertEqual(result["diagnostics"]["association_quality_ok"], accepted)
                self.assertEqual(result["best"]["image_points"], self.points)

    def test_pnp_rejection_does_not_change_association_diagnostics(self):
        original = copy.deepcopy(self.selection["diagnostics"])
        pose = copy.deepcopy(self.pose)
        pose["per_point_errors_px"] = {name: 21. for name in self.points}
        result, solve = self.call(pose=pose)
        solve.assert_called_once()
        self.assertEqual(self.selection["diagnostics"], original)
        self.assertTrue(result["diagnostics"]["association_quality_ok"])
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])
        self.assertEqual(result["best"]["image_points"], self.points)

    def test_quality_requires_enough_named_observations_and_inliers(self):
        selection = copy.deepcopy(self.selection)
        selection["image_points"] = dict(list(self.points.items())[:5])
        selection["candidates"] = self.candidates[:5]
        pose = copy.deepcopy(self.pose)
        pose["matched_names"] = list(selection["image_points"])
        pose["per_point_errors_px"] = {name: 1. for name in selection["image_points"]}
        pose["inlier_count"] = 5
        result, _ = self.call(selection=selection, pose=pose)
        self.assertIsNotNone(result["best"]["pose"])
        self.assertFalse(result["best"]["selection_quality_ok"])
        pose = copy.deepcopy(self.pose)
        pose["inlier_count"] = 3
        result, _ = self.call(pose=pose)
        self.assertFalse(result["best"]["selection_quality_ok"])

    def test_negative_depth_or_nonfinite_residual_cannot_pass(self):
        for field, value in (("all_in_front", False),
                             ("per_point_errors_px", {**self.pose["per_point_errors_px"], "CC2": float("nan")}),
                             ("per_point_errors_px", {**self.pose["per_point_errors_px"], "CC2": float("inf")})):
            with self.subTest(field=field, value=value):
                pose = copy.deepcopy(self.pose)
                pose[field] = value
                result, _ = self.call(pose=pose)
                self.assertIsNone(result["best"]["pose"])
                self.assertFalse(result["best"]["selection_quality_ok"])
                self.assertIsNotNone(result["best"]["pnp_error"])
                self.assertEqual(result["best"]["image_points"], self.points)

    def test_solver_errors_retain_observations_but_reject_pose(self):
        for error in (ValueError("invalid input"), RuntimeError("no pose"), cv2.error("cv failure")):
            with self.subTest(error=type(error).__name__):
                result, _ = self.call(solve_error=error)
                self.assertIsNone(result["best"]["pose"])
                self.assertFalse(result["best"]["selection_quality_ok"])
                self.assertEqual(result["best"]["pnp_error"], str(error))
                self.assertEqual(result["best"]["image_points"], self.points)
                self.assertIsNone(result["hypotheses"][0]["reproj"])

    def test_invalid_rectification_is_rejected_before_solver(self):
        for bad in (np.zeros((9, 3)), np.zeros((8, 2)), np.full((9, 2), np.nan)):
            with self.subTest(shape=bad.shape):
                result, solve = self.call(transform=lambda _points: bad)
                solve.assert_not_called()
                self.assertFalse(result["best"]["selection_quality_ok"])
                self.assertEqual(result["best"]["pnp_error"], "invalid_point_transform")
                self.assertEqual(result["best"]["image_points"], self.points)

    def test_missing_model_labels_cannot_hide_unscored_observations(self):
        # PnP's normal intersection behavior can solve a smaller model. The
        # adapter must not count the omitted observed labels as pose support.
        del self.objects["CC1"]
        pose = copy.deepcopy(self.pose)
        pose["matched_names"].remove("CC1")
        del pose["per_point_errors_px"]["CC1"]
        pose["inlier_count"] = 8
        result, solve = self.call(pose=pose)
        solve.assert_not_called()
        self.assertFalse(result["best"]["selection_quality_ok"])
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])
        self.assertEqual(result["best"]["image_points"], self.points)

    def test_incomplete_solver_labels_or_residuals_cannot_pass(self):
        for field in ("matched_names", "per_point_errors_px"):
            with self.subTest(field=field):
                pose = copy.deepcopy(self.pose)
                if field == "matched_names":
                    pose[field].remove("CC1")
                else:
                    del pose[field]["CC1"]
                result, solve = self.call(pose=pose)
                solve.assert_called_once()
                self.assertIsNone(result["best"]["pose"])
                self.assertFalse(result["best"]["selection_quality_ok"])
                self.assertEqual(result["best"]["image_points"], self.points)

    def test_empty_selection_never_calls_solver(self):
        selection = copy.deepcopy(self.selection)
        selection["image_points"] = {}
        selection["candidates"] = []
        result, solve = self.call(selection=selection)
        solve.assert_not_called()
        self.assertIsNone(result["best"])
        self.assertEqual(result["hypotheses"], [])
        self.assertFalse(result["diagnostics"]["selection_quality_ok"])


if __name__ == "__main__":
    unittest.main()
