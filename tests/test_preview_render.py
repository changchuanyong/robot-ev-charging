"""The preview is a same-frame overlay, never the source image or a stale pose."""
from copy import deepcopy
import unittest

import numpy as np

from vision.preview_render import PANEL_WIDTH, PROJECTED, render_preview


def diagnostic():
    return {
        "frontend": {"detected_boxes": [{"xyxy": [10, 10, 100, 80], "confidence": .9}],
                     "roi_bbox_xyxy": [5, 5, 110, 90], "candidate_points_raw_px": [[40, 40]],
                     "named_points_raw_px": {"PE": [40, 40]}, "reason": "ok"},
        "pose": {"solved": True, "mean_px": 20., "rms_px": 20., "max_px": 20.,
                 "tvec_mm": [0., 0., 300.], "per_hole": {
                     "PE": {"observed_px": [40, 40], "projected_px": [60, 40], "distance_px": 20.}}},
        "assessment": {"status": "warning", "reasons": ["point_error_exceeded"]},
        "camera": {"serial": "test-camera", "image_width": 120, "image_height": 100, "fps": 15,
                   "validated": False}, "geometry": {"version": "nominal", "validated": False},
        "frame": {"frame_number": 17}, "processing_ms": 80.,
    }


class PreviewRenderTests(unittest.TestCase):
    def test_overlays_preserve_source_pixels_and_input_diagnostic(self):
        source = np.full((100, 120, 3), 35, dtype=np.uint8)
        before, result = source.copy(), diagnostic()
        expected = deepcopy(result)
        rendered = render_preview(source, result, saved_count=2)
        np.testing.assert_array_equal(source, before)
        self.assertEqual(result, expected)
        self.assertEqual(rendered.shape, (720, 120 + PANEL_WIDTH, 3))
        self.assertFalse(np.shares_memory(source, rendered))
        np.testing.assert_array_equal(rendered[40, 60], PROJECTED)
        self.assertFalse(np.array_equal(rendered[:100, :120], source))

    def test_failure_does_not_show_old_projection_or_residual_values(self):
        source = np.full((100, 120, 3), 35, dtype=np.uint8)
        render_preview(source, diagnostic())
        failed = diagnostic()
        failed["frontend"] = {"named_points_raw_px": {}, "reason": "target_not_detected"}
        failed["pose"]["solved"] = False
        failed["assessment"] = {"status": "no_pose", "reasons": ["target_not_detected"]}
        stale = render_preview(source, failed)
        failed["pose"] = {"solved": False}
        clean = render_preview(source, failed)
        np.testing.assert_array_equal(stale, clean)
        np.testing.assert_array_equal(stale[:100, :120], source)

    def test_outside_and_invalid_coordinates_are_clipped_or_ignored(self):
        source = np.zeros((100, 120, 3), dtype=np.uint8)
        result = diagnostic()
        result["frontend"]["detected_boxes"] += [
            {"xyxy": [float("nan"), 0, 12, 20]}, {"xyxy": [-1e100, -1e100, 1e100, 1e100]},
            {"xyxy": [10, 10, 9, 9]}]
        result["frontend"]["candidate_points_raw_px"] += [[float("inf"), 0], None, [1, 2, 3], [-30, -20]]
        result["frontend"]["named_points_raw_px"]["CC1"] = [1e200, -1e200]
        result["pose"]["per_hole"]["PE"]["projected_px"] = [-1e200, 40]
        rendered = render_preview(source, result)
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertTrue(np.any(rendered[40, :35] != 0))
        np.testing.assert_array_equal(source, np.zeros_like(source))

    def test_empty_failed_frame_and_long_messages_remain_renderable(self):
        source = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = {"pose": {"solved": False}, "assessment": {"reasons": ["x" * 1000]}}
        rendered = render_preview(source, result, paused=True, message="保存 " * 200)
        self.assertEqual(rendered.shape, (720, 1280 + PANEL_WIDTH, 3))
        np.testing.assert_array_equal(rendered[:, :1280], source)


if __name__ == "__main__":
    unittest.main()
