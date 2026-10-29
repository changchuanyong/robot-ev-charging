"""The displayed/saved RGB and depth must be the same immutable analysis input."""
from concurrent.futures import Future
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vision import preview
from vision.depth_render import colorize_depth, render_depth


class ImmediateExecutor:
    def __init__(self, **kwargs):
        pass

    def submit(self, function):
        future = Future()
        try:
            future.set_result(function())
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


class PreviewDepthTests(unittest.TestCase):
    def run_loop(self, keys, *, repeat_color_at=(), **kwargs):
        number = 0
        pixels = np.zeros((6, 8, 3), np.uint8)
        raw = np.zeros((3, 4), np.uint16)

        def read():
            nonlocal number
            number += 1
            frame_number = number - 1 if number in repeat_color_at else number
            pixels[:] = frame_number
            raw[:] = 500 + frame_number
            frame = {"frame_number": frame_number}
            # A deliberately reused buffer exercises snapshot ownership.
            reasons = ["repeated_or_reordered_color_frame"] if number in repeat_color_at else []
            return pixels, frame, {"raw_z16": raw, "color_frame": frame,
                                   "frame": {"frame_number": number + 100}, "reasons": reasons}

        camera = SimpleNamespace(camera={}, read_rgbd=read)
        session = MagicMock(saved_count=0)

        def process(frontend, image, camera, frame, geometry, config, *, depth_sample):
            self.assertEqual(int(depth_sample["raw_z16"][0, 0]), 500 + frame["frame_number"])
            return {"frame": frame, "depth": {"status": "scene_only"},
                    "assessment": {"status": "no_pose"}}

        with (patch.object(preview, "ThreadPoolExecutor", ImmediateExecutor),
              patch.object(preview, "process_frame", side_effect=process) as work,
              patch.object(preview, "render_preview", side_effect=lambda image, *a, **k: image.copy()),
              patch.object(preview, "render_depth", return_value=np.zeros((6, 8, 3), np.uint8)),
              patch.object(preview.cv2, "namedWindow"), patch.object(preview.cv2, "resizeWindow"),
              patch.object(preview.cv2, "imshow"), patch.object(preview.cv2, "destroyWindow") as close,
              patch.object(preview.cv2, "waitKeyEx", side_effect=keys),
              patch.object(preview.cv2, "getWindowProperty", return_value=1)):
            result = preview.run_preview(camera, None, {}, {}, session, **kwargs)
        self.assertEqual(close.call_count, 2)
        return session, result, work.call_count

    def test_depth_save_binds_completed_display_instead_of_latest_frame(self):
        session, result, _ = self.run_loop([-1, 0x740000, 27])
        image, diagnostic, _ = session.save.call_args.args
        packet = session.save.call_args.kwargs["depth_sample"]
        self.assertTrue(np.all(image == 1))
        self.assertTrue(np.all(packet["raw_z16"] == 501))
        self.assertEqual(packet["color_frame"], diagnostic["frame"])
        self.assertEqual(packet["frame"]["frame_number"], 101)
        self.assertEqual(result["last_depth_status"], "scene_only")

    def test_pause_freezes_depth_as_rgb_frames_continue(self):
        session, _, count = self.run_loop([-1, 0x750000, 0x740000, 0x750000, 27])
        self.assertEqual(count, 2)
        packet = session.save.call_args.kwargs["depth_sample"]
        self.assertTrue(np.all(packet["raw_z16"] == 501))

    def test_depth_result_can_be_saved_and_bounded_run_exits(self):
        session, result, _ = self.run_loop([-1, -1], frames=1, save_first=True)
        session.save.assert_called_once()
        self.assertEqual(result["processed_frames"], 1)

    def test_repeated_color_is_not_reanalyzed_or_rebound_to_new_depth(self):
        session, _, count = self.run_loop([-1, 0x740000, -1, 27], repeat_color_at=(2,))
        self.assertEqual(count, 2)
        self.assertEqual(session.save.call_args.kwargs["depth_sample"]["frame"]["frame_number"], 101)

    def test_colorization_preserves_z16_and_invalid_pixels_are_black(self):
        raw = np.array([[0, 400, 1500, 60000]], dtype=np.uint16)
        original = raw.copy()
        shown = colorize_depth(raw, .001)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(shown[0, 0], [0, 0, 0])
        self.assertFalse(np.all(shown[0, 1] == 0))
        np.testing.assert_array_equal(shown[0, 2], shown[0, 3])

    def test_invalid_depth_does_not_display_leftover_plane(self):
        raw = np.full((480, 640), 500, np.uint16)
        packet = {"raw_z16": raw, "calibration": {"depth_scale_m": .001}}
        before = hashlib.sha256(raw.tobytes()).hexdigest()
        diagnostic = {"depth": {"status": "invalid", "reasons": ["depth_pair_skew_exceeded"],
                                "plane": {"solved": True, "normal_color_camera": [0, 0, -1], "rms_mm": .5},
                                "pnp_consistency": {"plane_distance_mm": 3}}}
        stale = render_depth(packet, diagnostic)
        diagnostic["depth"]["plane"] = {"solved": False}
        diagnostic["depth"]["pnp_consistency"] = None
        np.testing.assert_array_equal(stale, render_depth(packet, diagnostic))
        self.assertEqual(before, hashlib.sha256(raw.tobytes()).hexdigest())
        self.assertEqual(stale.shape, (720, 1140, 3))
        self.assertEqual(render_depth(None, {}).shape, stale.shape)


if __name__ == "__main__":
    unittest.main()
