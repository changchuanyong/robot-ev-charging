"""Exposure calibration scoring and configuration contract."""

import json
import unittest

import cv2
import numpy as np

from ev_charging.capture import exposure_calibration as calibration
from ev_charging.capture.collect import validate_rgb_options
from ev_charging.paths import PROJECT_ROOT


class ScoreFrameTests(unittest.TestCase):
    def test_dark_disc_scores_above_a_flat_field(self):
        image = np.full((200, 200), 200, np.uint8)
        cv2.circle(image, (100, 100), 25, 40, -1)
        flat = np.full((200, 200), 120, np.uint8)
        disc, uniform = calibration.score_frame(image), calibration.score_frame(flat)
        self.assertGreater(disc["aperture_response"], uniform["aperture_response"])
        self.assertGreater(disc["score"], uniform["score"])
        self.assertEqual(uniform["score"], 0.0)

    def test_clipping_cancels_the_score(self):
        dark = calibration.score_frame(np.zeros((200, 200), np.uint8))
        self.assertEqual(dark["clip_low"], 1.0)
        self.assertEqual(dark["score"], 0.0)
        bright = calibration.score_frame(np.full((200, 200), 255, np.uint8))
        self.assertEqual(bright["clip_high"], 1.0)
        self.assertEqual(bright["score"], 0.0)

    def test_roi_box_is_validated_and_applied(self):
        image = np.zeros((100, 100), np.uint8)
        self.assertEqual(calibration.region_of_interest(image, (0, 0, 60, 60)).shape, (60, 60))
        with self.assertRaises(ValueError):
            calibration.score_frame(image, (10, 10, 20, 20))

    def test_box_parser_rejects_wrong_arity(self):
        self.assertEqual(calibration.parse_box("1,2,3,4"), (1, 2, 3, 4))
        with self.assertRaises(ValueError):
            calibration.parse_box("1,2,3")


class SettingTests(unittest.TestCase):
    def test_candidates_stay_inside_the_reported_range(self):
        exposure = {"min": 1.0, "max": 1000.0, "step": 1.0, "default": 100.0}
        values = calibration.exposure_candidates(exposure, count=5)
        self.assertTrue(values)
        self.assertTrue(all(exposure["min"] <= value <= exposure["max"] for value in values))
        self.assertEqual(values, sorted(set(values)))
        gain = {"min": 0.0, "max": 248.0, "step": 1.0, "default": 0.0}
        self.assertTrue(all(0.0 <= value <= 248.0 for value in calibration.gain_candidates(gain)))

    def test_degenerate_range_returns_a_single_value(self):
        limits = {"min": 120.0, "max": 120.0, "step": 1.0, "default": 120.0}
        self.assertEqual(calibration.exposure_candidates(limits), [120.0])

    def test_recommend_prefers_the_higher_score_then_less_clipping(self):
        rows = [
            {
                "status": "captured",
                "metrics": {"score": 1.0, "clip_low": 0.0},
                "applied": {"exposure": 100.0, "gain": 0.0},
            },
            {
                "status": "captured",
                "metrics": {"score": 1.0, "clip_low": 0.05},
                "applied": {"exposure": 200.0, "gain": 0.0},
            },
            {"status": "failed", "error": "unsupported"},
        ]
        best = calibration.recommend(rows)
        self.assertEqual(best["applied"]["exposure"], 100.0)
        self.assertEqual(
            calibration.rgb_options(best),
            {"enable_auto_exposure": 0, "exposure": 100.0, "gain": 0.0},
        )

    def test_failed_settings_are_not_recommended(self):
        self.assertIsNone(calibration.recommend([{"status": "failed"}]))
        self.assertIsNone(calibration.recommend([]))


class ConfigContractTests(unittest.TestCase):
    def test_collect_profile_exposes_a_valid_rgb_options_block(self):
        config = json.loads((PROJECT_ROOT / "config/d415.json").read_text(encoding="utf-8-sig"))
        options = config["profiles"]["collect"]["rgb_options"]
        validate_rgb_options(options)
        self.assertEqual(options["enable_auto_exposure"], 1)

    def test_calibration_output_can_be_pasted_back_into_the_profile(self):
        options = {"enable_auto_exposure": 0, "exposure": 150.0, "gain": 16.0}
        validate_rgb_options(options)
        payload = json.loads(json.dumps(options))
        self.assertEqual(set(payload), {"enable_auto_exposure", "exposure", "gain"})


if __name__ == "__main__":
    unittest.main()
