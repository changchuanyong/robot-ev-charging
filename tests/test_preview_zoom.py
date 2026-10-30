"""Display changes retain raw images and same-frame geometry."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from PIL import ImageDraw

from vision.depth_render import colorize_depth, render_depth
from vision.preview_render import PANEL_WIDTH, PROJECTED, render_preview


class PreviewZoomTests(unittest.TestCase):
    def test_zoom_enlarges_unique_target_without_mutating_raw_image_or_points(self):
        source = np.zeros((160, 200, 3), np.uint8)
        source[40:80, 60:100] = (15, 90, 170)
        original = source.copy()
        diagnostic = {"frontend": {"detected_boxes": [{"xyxy": [60, 40, 100, 80]}]},
                      "pose": {"solved": True, "per_hole": {
                          "PE": {"observed_px": [75, 60], "projected_px": [80, 60]}}}}
        before = deepcopy(diagnostic)
        result = render_preview(source, diagnostic, zoom=True)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(diagnostic, before)
        self.assertEqual(result.shape, (864, 200 + PANEL_WIDTH, 3))
        # The square crop is centered in the original 200 x 160 image area.
        np.testing.assert_array_equal(result[70, 5], [29, 25, 22])
        np.testing.assert_array_equal(result[30, 50], [15, 90, 170])
        projected = np.all(result[60:105, 85:120] == PROJECTED, axis=2)
        self.assertGreater(np.count_nonzero(projected), 10)

    def test_zoom_waits_for_unique_valid_target(self):
        source = np.full((80, 100, 3), (20, 80, 140), np.uint8)
        for boxes in ([], [{"xyxy": [10, 10, 30, 30]}, {"xyxy": [50, 10, 70, 30]}],
                      [{"xyxy": [float("nan"), 0, 10, 10]}], [{"xyxy": [150, 0, 200, 20]}]):
            diagnostic = {"frontend": {"detected_boxes": boxes}}
            with self.subTest(boxes=boxes):
                unzoomed = render_preview(source, diagnostic)
                zoomed = render_preview(source, diagnostic, zoom=True)
                np.testing.assert_array_equal(unzoomed[:80, :100], zoomed[:80, :100])

    def test_runtime_is_distinct_from_configured_fps_and_keeps_full_hd_pixels(self):
        source = np.empty((1080, 1920, 3), np.uint8)
        source[:] = (37, 111, 215)
        diagnostic = {"camera": {"fps": 30}, "frame": {"rgb_settings": {"sensor_options": {
            "values": {"enable_auto_exposure": 1, "exposure": 156, "gain": 64}}}}}
        texts = []
        with patch.object(ImageDraw.ImageDraw, "text", side_effect=lambda xy, value, **kw: texts.append(str(value))):
            result = render_preview(source, diagnostic, runtime={
                "capture_fps": 27.5, "analysis_fps": 11.2, "render_fps": 10.4, "frame_age_ms": 123})
        self.assertEqual(result.shape, (1080, 1920 + PANEL_WIDTH, 3))
        np.testing.assert_array_equal(result[:, :1920], source)
        self.assertTrue(any("30 fps" in value for value in texts))
        self.assertTrue(any("27.5" in value and "11.2" in value and "10.4" in value for value in texts))
        self.assertTrue(any("156" in value and "64" in value for value in texts))

    def test_multiline_sdk_errors_do_not_crash_or_change_diagnostic(self):
        source = np.zeros((80, 100, 3), np.uint8)
        diagnostic = {"assessment": {"reasons": ["OpenCV error\n" + "x" * 10000 + "\r\nsecond line"]},
                      "depth": {"reasons": ["SDK failure\nmissing frame"], "status": "invalid"}}
        before = deepcopy(diagnostic)
        render_preview(source, diagnostic, message="first\nsecond")
        render_depth(None, diagnostic, message="first\r\nsecond")
        self.assertEqual(diagnostic, before)

    def test_depth_text_rendering_preserves_native_colorized_pixels(self):
        raw = np.full((480, 640), 650, np.uint16)
        raw[20:40, 20:40] = 0
        packet = {"raw_z16": raw, "calibration": {"depth_scale_m": .001}}
        result = render_depth(packet, {"depth": {"status": "scene_only"}})
        np.testing.assert_array_equal(result[72:552, 16:656], colorize_depth(raw, .001))
        self.assertTrue(np.all(raw[20:40, 20:40] == 0))

    def test_display_width_scales_only_image_and_overlays_not_panel(self):
        source = np.zeros((1080, 1920, 3), np.uint8)
        diagnostic = {"frontend": {"detected_boxes": [{"xyxy": [600, 300, 1200, 900]}]},
                      "camera": {"image_width": 1920, "image_height": 1080}}
        before = deepcopy(diagnostic)
        result = render_preview(source, diagnostic, display_width=1280, runtime={})
        self.assertEqual(result.shape, (864, 1280 + PANEL_WIDTH, 3))
        # The native target top-left (600,300) maps to display (400,200).
        self.assertGreater(int(result[200, 400].max()), 0)
        self.assertEqual(int(result[300, 600].max()), 0)
        self.assertFalse(np.any(source))
        self.assertEqual(diagnostic, before)
        unscaled = render_preview(source, diagnostic, runtime={})
        np.testing.assert_array_equal(result[:, 1280:], unscaled[:864, 1920:])

    def test_zoom_is_applied_before_display_resize(self):
        source = np.zeros((160, 200, 3), np.uint8)
        source[40:80, 60:100] = (15, 90, 170)
        diagnostic = {"frontend": {"detected_boxes": [{"xyxy": [60, 40, 100, 80]}]}}
        result = render_preview(source, diagnostic, zoom=True, display_width=100)
        self.assertEqual(result.shape, (864, 100 + PANEL_WIDTH, 3))
        np.testing.assert_array_equal(result[15, 25], [15, 90, 170])
        np.testing.assert_array_equal(result[35, 2], [29, 25, 22])
        np.testing.assert_array_equal(source[40, 60], [15, 90, 170])

    def test_invalid_display_width_is_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "display_width"):
                render_preview(np.zeros((2, 2, 3), np.uint8), {}, display_width=value)


if __name__ == "__main__":
    unittest.main()
