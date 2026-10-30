"""Do not repaint unchanged panels or mistake configured fps for actual events."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vision import preview


class NeverFinishedExecutor:
    def __init__(self, **kwargs):
        pass

    def submit(self, function):
        return Future()

    def shutdown(self, **kwargs):
        pass


class PreviewPerformanceTests(unittest.TestCase):
    def test_waiting_on_inference_reuses_panels_but_polls_keyboard(self):
        image = np.zeros((6, 8, 3), np.uint8)
        camera = SimpleNamespace(camera={}, read_rgbd=lambda: (image, {"frame_number": 1}, {}))
        with (patch.object(preview, "ThreadPoolExecutor", NeverFinishedExecutor),
              patch.object(preview.time, "monotonic", return_value=1.),
              patch.object(preview, "render_preview", return_value=image) as color,
              patch.object(preview, "render_depth", return_value=image) as depth,
              patch.object(preview.cv2, "namedWindow"), patch.object(preview.cv2, "resizeWindow"),
              patch.object(preview.cv2, "imshow"), patch.object(preview.cv2, "destroyWindow"),
              patch.object(preview.cv2, "getWindowProperty", return_value=1),
              patch.object(preview.cv2, "waitKeyEx", side_effect=[-1, -1, -1, 27]) as keys):
            result = preview.run_preview(camera, None, {}, {}, MagicMock(saved_count=0))
        self.assertEqual(color.call_count, 1)
        self.assertEqual(depth.call_count, 1)
        self.assertEqual(keys.call_count, 4)
        self.assertEqual(result["analysis_mean_fps"], 0.)

    def test_fps_is_derived_from_completed_events(self):
        rate = preview.FrameRate()
        with patch.object(preview.time, "monotonic", side_effect=[10., 10.1, 10.2]):
            for _ in range(3):
                rate.tick()
        with patch.object(preview.time, "monotonic", return_value=10.2):
            self.assertAlmostEqual(rate.fps(), 10.)
        self.assertAlmostEqual(rate.mean_fps(), 10.)
        with patch.object(preview.time, "monotonic", return_value=13.):
            self.assertEqual(rate.fps(), 0.)

    def test_automatic_override_clears_old_manual_values_without_changing_source(self):
        import json
        config = json.loads((preview.ROOT / "config/d415_preview_usb3.json").read_text())
        config["rgb_options"] = {"exposure": 500, "gain": 90, "enable_auto_exposure": 0}
        updated = preview.configure_capture(config, quality="hd", auto_exposure=True)
        self.assertEqual((updated["width"], updated["height"], updated["fps"]), (1280, 720, 30))
        self.assertEqual(updated["rgb_options"], {"enable_auto_exposure": 1})
        self.assertEqual(config["rgb_options"]["exposure"], 500)

    def test_manual_override_disables_auto_and_profile_uses_actual_requested_dimensions(self):
        import json
        config = json.loads((preview.ROOT / "config/data_capture_d415.json").read_text())
        updated = preview.configure_capture(config, quality="full-hd", exposure=200, gain=80)
        self.assertEqual((updated["width"], updated["height"], updated["fps"]), (1920, 1080, 30))
        self.assertEqual(updated["rgb_options"], {"enable_auto_exposure": 0, "exposure": 200, "gain": 80})


if __name__ == "__main__":
    unittest.main()
