"""Measured dark apertures are a fallback, never an extension of a primary layout."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ev_charging.paths import PROJECT_ROOT
from ev_charging.review import keypoint_suggestions as suggestions
from ev_charging.vision.algorithms import observed_layout
from tests.governance_helpers import FixedDetector, baseline


def selection(points):
    return {
        "image_points": dict(points),
        "association_points": dict(points),
        "diagnostics": {
            "ambiguous": False,
            "structural_support_ok": True,
            "selection_quality_ok": bool(points),
            "anchor_pair_count": len(points),
        },
        "candidates": [],
    }


class DarkApertureFallbackTests(unittest.TestCase):
    def setUp(self):
        self.engine = suggestions.SuggestionEngine(".", ".")
        self.engine.modules = [None, observed_layout]
        self.engine.prior = {"schema_version": 1}
        self.engine._load_detector = lambda model: FixedDetector(
            [{"xyxy": [50, 50, 150, 150], "confidence": 0.9}]
        )
        self.image = np.zeros((200, 200, 3), np.uint8)
        self.model = {"weights": "weights.pt", "weights_sha256": "0" * 64}
        self.candidate = SimpleNamespace(contour_index=0)
        self.low_candidate = SimpleNamespace(contour_index=7)
        self.exposure_candidate = SimpleNamespace(contour_index=9)
        self.extra = SimpleNamespace(contour_index=1, aperture_center=(80.0, 80.0))
        self.low_candidates = [self.low_candidate]
        self.exposure_candidates = [self.exposure_candidate]

    def fake_preprocess(self, crop, diagnostics=None, profile="standard", **kwargs):
        if diagnostics is not None:
            diagnostics["candidate_trace"] = {"outcomes": []}
        if profile == "low_contrast":
            candidates = self.low_candidates
        elif profile == "exposure":
            candidates = self.exposure_candidates
        else:
            candidates = [self.candidate]
        return np.zeros((120, 120, 3), np.uint8), candidates, None, (0, 0, 0)

    def run_suggest(self, select_mock, extras):
        with (
            patch.object(suggestions, "preprocess_candidates", side_effect=self.fake_preprocess) as pre,
            patch.object(suggestions, "promote_dark_apertures", return_value=[]),
            patch.object(suggestions, "stable_dark_apertures", return_value=extras) as stable,
            patch.object(observed_layout, "select_observed_correspondences", select_mock),
        ):
            prediction = self.engine.suggest(
                {"status": "ready"}, self.image, model=self.model, diagnostics=True
            )
        return prediction, pre, stable

    def test_primary_layout_is_never_extended_by_measured_dark_boundaries(self):
        primary = selection({"DC_neg": (60.0, 60.0), "DC_pos": (100.0, 60.0)})
        select_mock = MagicMock(return_value=primary)
        prediction, pre, stable = self.run_suggest(select_mock, [self.extra])
        self.assertEqual(select_mock.call_count, 1)
        stable.assert_not_called()
        self.assertIsNone(prediction["diagnostics"]["fallback_source"])
        self.assertFalse(prediction["diagnostics"]["fallback_used"])
        self.assertIsNone(prediction["diagnostics"]["primary_layout"])
        self.assertEqual(
            prediction["points"], {name: list(xy) for name, xy in primary["image_points"].items()}
        )

    def test_empty_primary_retries_once_with_measured_dark_boundaries(self):
        retried = selection({"DC_neg": (60.0, 60.0), "PE": (80.0, 80.0)})
        select_mock = MagicMock(side_effect=[selection({}), retried])
        prediction, pre, stable = self.run_suggest(select_mock, [self.extra])
        self.assertEqual(select_mock.call_count, 2)
        stable.assert_called_once()
        self.assertEqual(select_mock.call_args_list[1].args[0], [self.candidate, self.extra])
        self.assertEqual(prediction["diagnostics"]["fallback_source"], "dark_aperture")
        self.assertEqual(
            prediction["points"], {name: list(xy) for name, xy in retried["image_points"].items()}
        )
        # Completion belongs to the primary evidence chain and is skipped here.
        self.assertEqual(prediction["diagnostics"]["completed_apertures"], [])
        self.assertEqual(pre.call_count, 1)

    def test_low_contrast_profile_is_tried_before_the_exposure_profile(self):
        low = selection({"DC_neg": (60.0, 60.0), "PE": (80.0, 80.0)})
        select_mock = MagicMock(side_effect=[selection({}), selection({}), low])
        prediction, pre, _ = self.run_suggest(select_mock, [self.extra])
        self.assertEqual(select_mock.call_count, 3)
        self.assertEqual(prediction["diagnostics"]["fallback_source"], "low_contrast")
        self.assertEqual(select_mock.call_args_list[2].args[0], [self.low_candidate, self.extra])
        self.assertIsNone(pre.call_args_list[0].kwargs.get("profile"))
        self.assertEqual(pre.call_args_list[1].kwargs.get("profile"), "low_contrast")
        self.assertEqual(
            [call.kwargs.get("profile") for call in pre.call_args_list],
            [None, "low_contrast"],
        )
        self.assertEqual(prediction["points"], {name: list(xy) for name, xy in low["image_points"].items()})

    def test_exposure_profile_runs_before_the_scale_free_gate(self):
        exposure = selection({"DC_neg": (60.0, 60.0), "PE": (80.0, 80.0)})
        select_mock = MagicMock(side_effect=[selection({}), selection({}), selection({}), exposure])
        prediction, pre, _ = self.run_suggest(select_mock, [self.extra])
        self.assertEqual(select_mock.call_count, 4)
        self.assertEqual(prediction["diagnostics"]["fallback_source"], "exposure")
        self.assertEqual(select_mock.call_args_list[3].args[0], [self.exposure_candidate, self.extra])
        self.assertEqual(
            [call.kwargs.get("profile") for call in pre.call_args_list],
            [None, "low_contrast", "exposure"],
        )
        self.assertEqual(prediction["diagnostics"]["completed_apertures"], [])

    def test_scale_free_anchor_gate_is_the_final_attempt(self):
        free = selection({"DC_neg": (60.0, 60.0), "PE": (80.0, 80.0)})
        select_mock = MagicMock(
            side_effect=[selection({}), selection({}), selection({}), selection({}), free]
        )
        prediction, _, _ = self.run_suggest(select_mock, [self.extra])
        self.assertEqual(select_mock.call_count, 5)
        self.assertEqual(prediction["diagnostics"]["fallback_source"], "scale_free_anchors")
        last = select_mock.call_args_list[4]
        self.assertEqual(last.args[0], [self.candidate, self.extra])
        self.assertTrue(last.kwargs.get("scale_free_anchors"))

    def test_no_candidate_source_leaves_the_result_empty(self):
        self.low_candidates = []
        self.exposure_candidates = []
        select_mock = MagicMock(return_value=selection({}))
        prediction, pre, stable = self.run_suggest(select_mock, [])
        self.assertEqual(select_mock.call_count, 2)
        stable.assert_called_once()
        self.assertEqual(pre.call_count, 3)
        self.assertIsNone(prediction["diagnostics"]["fallback_source"])
        self.assertEqual(prediction["points"], {})
        self.assertIn("证据不足", prediction["reason"])

    def test_retry_without_an_association_keeps_the_empty_result(self):
        select_mock = MagicMock(
            side_effect=[selection({}), selection({}), selection({}), selection({}), selection({})]
        )
        prediction, _, _ = self.run_suggest(select_mock, [self.extra])
        self.assertEqual(select_mock.call_count, 5)
        self.assertIsNone(prediction["diagnostics"]["fallback_source"])
        self.assertEqual(prediction["points"], {})

    def test_working_fixed_photo_keeps_its_output_without_the_fallback(self):
        fixture = baseline()
        case = next(
            row for row in fixture["images"] if row["image"].rsplit("/", 1)[-1].startswith("0311084e449d")
        )
        image = cv2.imdecode(
            np.frombuffer((PROJECT_ROOT / case["image"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR
        )
        engine = suggestions.SuggestionEngine(PROJECT_ROOT, PROJECT_ROOT / "dataset/vision")
        with patch.object(engine, "_load_detector", return_value=FixedDetector(case["boxes"])):
            prediction = engine.suggest({"status": "ready"}, image, model=fixture["config"], diagnostics=True)
        self.assertEqual(len(prediction["points"]), 9)
        self.assertIsNone(prediction["diagnostics"]["fallback_source"])


class AnchorGateTests(unittest.TestCase):
    def pairs(self, diameters, points, *, scale_free=False):
        candidates = [SimpleNamespace(aperture_diameter_px=float(value), score=0.6) for value in diameters]
        return observed_layout._anchor_pairs(
            candidates,
            np.asarray(points, float),
            np.asarray(diameters, float),
            review_shape=(700, 700),
            scale_free=scale_free,
        )

    def test_extent_gate_rejects_a_pair_the_scale_free_gate_accepts(self):
        points = [[100.0, 100.0], [170.0, 100.0]]
        diameters = [20.0, 21.0]
        self.assertEqual(self.pairs(diameters, points), [])
        self.assertEqual(len(self.pairs(diameters, points, scale_free=True)), 1)

    def test_scale_free_gate_requires_a_plausible_span(self):
        points = [[100.0, 100.0], [250.0, 100.0]]
        self.assertEqual(self.pairs([20.0, 21.0], points, scale_free=True), [])

    def test_scale_free_gate_requires_comparable_anchor_sizes(self):
        points = [[100.0, 100.0], [200.0, 100.0]]
        self.assertEqual(self.pairs([20.0, 33.0], points, scale_free=True), [])


if __name__ == "__main__":
    unittest.main()
