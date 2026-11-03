import contextlib
import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from vision.evaluation.evaluate_library_keypoints import ROOT, compare_reports, point_diagnostic, topology_warnings
from vision.review.keypoint_suggestions import promote_dark_apertures

from vision.algorithms import latest_candidate_contours as contours
from vision.algorithms.center_fit import fit_ellipse_on_kept_candidates
from vision.algorithms import observed_layout


class ApertureEvidenceTests(unittest.TestCase):
    def candidate(self):
        boundary = cv2.ellipse2Poly((60, 60), (22, 16), 20, 0, 360, 3).reshape(-1, 1, 2)
        item = contours.compute_candidate_features(boundary, (120, 120, 3), allow_open=True)
        with contextlib.redirect_stdout(io.StringIO()):
            fit_ellipse_on_kept_candidates([item])
        self.assertTrue(item.fit_valid)
        return item

    def test_fitted_dark_inner_boundary_is_recovered_without_moving_center(self):
        item = self.candidate()
        image = np.full((120, 120, 3), 100, np.uint8)
        cv2.ellipse(image, (60, 60), (22, 16), 20, 0, 360, (20, 20, 20), -1)
        original = item.fitted_center
        recovered = promote_dark_apertures([item], image, contours)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(item.aperture_center, original)
        self.assertGreater(item.aperture_diameter_px, 0)
        self.assertTrue(item.review_only_aperture)

    def test_bright_outer_wall_and_flat_region_are_not_apertures(self):
        for ring in (False, True):
            with self.subTest(ring=ring):
                item = self.candidate()
                image = np.full((120, 120, 3), 20, np.uint8)
                if ring:
                    cv2.ellipse(image, (60, 60), (22, 16), 20, 0, 360, (100, 100, 100), -1)
                    cv2.ellipse(image, (60, 60), (8, 6), 20, 0, 360, (20, 20, 20), -1)
                self.assertEqual(promote_dark_apertures([item], image, contours), [])
                self.assertIsNone(item.aperture_center)

    def test_invalid_fit_and_existing_aperture_are_not_overwritten(self):
        invalid, existing = self.candidate(), self.candidate()
        invalid.fit_valid = False
        existing.aperture_center = (59., 61.)
        existing.aperture_diameter_px = 15.
        image = np.zeros((120, 120, 3), np.uint8)
        self.assertEqual(promote_dark_apertures([invalid, existing], image, contours), [])
        self.assertIsNone(invalid.aperture_center)
        self.assertEqual(existing.aperture_center, (59., 61.))


class DiagnosticReportTests(unittest.TestCase):
    def test_missing_detection_and_retained_but_unmatched_are_distinguished(self):
        self.assertEqual(point_diagnostic([10, 10], None, {"detected_boxes": []}, 5)["stage"], "detector_missing")
        trace = {"detected_boxes": [{}], "roi_bbox_xyxy": [0, 0, 100, 100], "contours": {
            "candidate_trace": {"outcomes": [{"stage": "kept", "measurement_center": [11, 10],
                                               "aperture_center": None}]}}}
        self.assertEqual(point_diagnostic([10, 10], None, trace, 5)["stage"], "kept_without_aperture")

    def test_topology_warning_does_not_relabel_annotations(self):
        points = {"DC_neg": [20, 50], "DC_pos": [80, 50], "PE": [50, 80], "S_neg": [25, 75], "CC2": [50, 20]}
        row = {"points": {key: {"xy": xy, "visibility": "visible"} for key, xy in points.items()}}
        original = copy.deepcopy(row)
        self.assertEqual(topology_warnings(row), ["S_neg"])
        self.assertEqual(row, original)

    def test_comparison_rejects_different_reference_or_weights(self):
        report = {"annotation_sha256": "a", "model": {"weights_sha256": "w"},
                  "summary": {"diagnostic_tolerance_px": 5}, "rows": []}
        self.assertEqual(compare_reports(report, report)["lost_within_tolerance"], [])
        changed = copy.deepcopy(report)
        changed["model"]["weights_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "same_annotations"):
            compare_reports(report, changed)


class ReviewAnchorTests(unittest.TestCase):
    def pair(self, points, diameters, shape=(600, 600, 3)):
        candidates = [SimpleNamespace(aperture_diameter_px=d, score=1.) for d in diameters]
        return observed_layout._anchor_pairs(candidates, np.asarray(points, dtype=float),
                                             np.asarray(diameters, dtype=float), review_shape=shape)

    def test_small_fragment_pair_cannot_define_connector_axes(self):
        self.assertEqual(self.pair([[280, 300], [320, 310]], [12, 15]), [])

    def test_main_hole_and_pe_cannot_be_used_as_two_dc_holes(self):
        self.assertEqual(self.pair([[200, 300], [310, 410]], [60, 33]), [])

    def test_full_size_similar_main_holes_remain_eligible(self):
        self.assertEqual(len(self.pair([[200, 300], [400, 300]], [60, 65])), 1)

    def test_review_pair_constraints_scale_with_image(self):
        self.assertEqual(len(self.pair([[100, 150], [200, 150]], [30, 32.5], (300, 300, 3))), 1)

    def layout_fixture(self):
        positions = dict(zip(observed_layout.LABELS, [(-.35,-.45),(0,-.6),(.35,-.45),(0,-.2),
                                                     (-.5,0),(.5,0),(-.4,.6),(0,.6),(.4,.6)]))
        prior = {'schema_version':1,'kind':'observed_2d_layout','fit_split':'train',
                 'labels':list(observed_layout.LABELS),'fallback_centers':positions,
                 'templates':[{'points':copy.deepcopy(positions)} for _ in range(3)]}
        candidates=[]
        for name, xy in positions.items():
            point=np.asarray(xy)*200+[300,300]
            diameter=60 if name.startswith('DC') else 30 if name=='PE' else 12
            candidates.append(SimpleNamespace(aperture_center=point,fitted_center=point,
                                               aperture_diameter_px=diameter,ellipse_axes=(diameter,diameter),
                                               score=1.,inside_prior=True,review_only_aperture=False))
        return prior,candidates

    def test_rotating_connector_does_not_swap_hole_names(self):
        prior,candidates=self.layout_fixture()
        for candidate in candidates:
            candidate.aperture_center=600-candidate.aperture_center
            candidate.fitted_center=candidate.aperture_center
        with patch.object(observed_layout,'_prepare',return_value=candidates):
            result=observed_layout.select_observed_correspondences(candidates,(600,600,3),prior,review_mode=True)
        self.assertEqual(set(result['image_points']),set(observed_layout.LABELS))
        for name,candidate in zip(observed_layout.LABELS,candidates):
            np.testing.assert_allclose(result['image_points'][name],candidate.aperture_center)

    def test_signal_hole_cannot_be_assigned_to_center_hole(self):
        _,candidates=self.layout_fixture()
        points=np.asarray([candidate.aperture_center for candidate in candidates])
        points[0]=points[3]+[2,0]
        kept,rejected=observed_layout._review_topology(dict(enumerate(range(9))),points)
        self.assertNotIn(0,kept)
        self.assertIn(0,rejected)
        self.assertTrue({1,3,4,5,7}<=kept.keys())

    def test_laterally_offset_center_cannot_be_used_as_upper_signal_row(self):
        _,candidates=self.layout_fixture()
        points=np.asarray([candidate.aperture_center for candidate in candidates])
        points[0]=[265,250]
        kept,_=observed_layout._review_topology(dict(enumerate(range(9))),points)
        self.assertNotIn(0,kept)

    def test_perspective_compressed_signal_side_is_preserved(self):
        _,candidates=self.layout_fixture()
        points=np.asarray([candidate.aperture_center for candidate in candidates])
        points[2]=[323,215]
        kept,_=observed_layout._review_topology(dict(enumerate(range(9))),points)
        self.assertIn(2,kept)

    def test_ambiguous_small_hole_is_withheld_but_shared_anchors_remain(self):
        prior,candidates=self.layout_fixture()
        candidates[1].aperture_center=np.array([290.,180.])
        candidates[1].fitted_center=candidates[1].aperture_center
        alternative=copy.deepcopy(candidates[1])
        alternative.aperture_center=np.array([310.,180.])
        alternative.fitted_center=alternative.aperture_center
        candidates.append(alternative)
        prior['templates'][0]['points']['CC2']=(-.05,-.6)
        prior['templates'][1]['points']['CC2']=(.05,-.6)
        with patch.object(observed_layout,'_prepare',return_value=candidates):
            result=observed_layout.select_observed_correspondences(candidates,(600,600,3),prior,review_mode=True)
        self.assertNotIn('CC2',result['image_points'])
        self.assertTrue({'DC_neg','DC_pos','PE'}<=result['image_points'].keys())
        self.assertIn('CC2',result['diagnostics']['review_disputed_labels'])


if __name__ == "__main__":
    unittest.main()
