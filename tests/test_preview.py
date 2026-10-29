"""Preview frame association, pause/save handling, and SDK-only diagnostics."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from vision import preview
from vision.geometry_reference import load_geometry_reference


ROOT = Path(__file__).resolve().parents[1]


class ImmediateExecutor:
    def __init__(self, **kwargs):
        pass

    def submit(self, function):
        future = Future()
        try:
            future.set_result(function())
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/realsense_config.json").read_text())
        self.camera = json.loads((ROOT / "config/d415_camera_intrinsics.json").read_text())
        self.geometry = load_geometry_reference(ROOT / "config/geometry_nominal.json")
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.frame = {"frame_number": 7, "device_timestamp_ms": 500., "timestamp_domain": "hardware_clock",
                      "received_at_utc": "2026-10-29T09:00:00+00:00"}

    def test_sdk_intrinsics_can_estimate_but_never_authorize_robot(self):
        points = self.geometry["points_mm"]
        projected, _ = cv2.projectPoints(np.asarray(list(points.values()), dtype=float),
                                        np.array([.1, .05, 0.]), np.array([0., 0., 400.]),
                                        np.asarray(self.camera["camera_matrix"]), np.zeros(5))
        frontend = MagicMock()
        frontend.extract.return_value = {
            "named_points_raw_px": dict(zip(points, projected.reshape(-1, 2).tolist())),
            "association_quality_ok": True, "selection_quality_ok": True, "ambiguous": False, "reason": "ok"}
        result = preview.process_frame(frontend, self.image, self.camera, self.frame, self.geometry, self.config)
        self.assertTrue(result["pose"]["solved"])
        self.assertLess(result["pose"]["rms_px"], .01)
        self.assertEqual(result["assessment"]["status"], "estimated")
        self.assertFalse(result["camera"]["validated"])
        self.assertFalse(result["geometry"]["validated"])
        self.assertFalse(result["authorizes_robot_motion"])
        self.assertEqual(result["image_pixels_sha256"], hashlib.sha256(self.image.tobytes()).hexdigest())
        frontend.extract.return_value = {"named_points_raw_px": {}, "reason": "ValueError: target_not_detected"}
        missing = preview.process_frame(frontend, self.image, self.camera, self.frame, self.geometry, self.config)
        self.assertEqual(missing["assessment"]["status"], "no_pose")
        self.assertIn("target_not_detected", missing["assessment"]["reasons"][0])
        self.assertNotIn("per_hole", missing["pose"])

    def test_ambiguous_or_poor_pose_remains_warning(self):
        frontend = {"reason": "ok", "association_quality_ok": True, "ambiguous": True}
        pose = {"solved": True, "inlier_count": 3, "rms_px": 20., "max_px": 50.}
        result = preview.assess_preview(frontend, pose, self.config)
        self.assertEqual(result["status"], "warning")
        self.assertIn("ambiguous_correspondences", result["reasons"])
        self.assertIn("insufficient_inliers", result["reasons"])
        self.assertIn("rms_exceeded", result["reasons"])
        self.assertIn("point_error_exceeded", result["reasons"])

    def run_loop(self, keys, **kwargs):
        reads = iter(range(1, 20))
        camera = SimpleNamespace(camera=self.camera)

        def read():
            number = next(reads)
            return np.full((6, 8, 3), number, dtype=np.uint8), {**self.frame, "frame_number": number}

        camera.read = read
        session = MagicMock(saved_count=0)

        def process(frontend, image, camera, frame, geometry, config):
            return {"frame": frame, "assessment": {"status": "no_pose", "reasons": ["target_not_detected"]}}

        def render(image, diagnostic, **kwargs):
            overlay = image.copy()
            overlay[0, 0] = 255
            return overlay

        with (patch.object(preview, "ThreadPoolExecutor", ImmediateExecutor),
              patch.object(preview, "process_frame", side_effect=process) as work,
              patch.object(preview, "render_preview", side_effect=render),
              patch.object(preview.cv2, "namedWindow"), patch.object(preview.cv2, "resizeWindow"),
              patch.object(preview.cv2, "imshow"), patch.object(preview.cv2, "destroyWindow") as close,
              patch.object(preview.cv2, "waitKeyEx", side_effect=keys),
              patch.object(preview.cv2, "getWindowProperty", return_value=1)):
            result = preview.run_preview(camera, None, self.geometry, self.config, session, **kwargs)
        close.assert_called_once()
        return result, session, work.call_count

    def test_save_binds_displayed_result_not_newer_camera_frame(self):
        _, session, _ = self.run_loop([-1, ord("s"), ord("q")])
        image, diagnostic, overlay = session.save.call_args.args
        self.assertTrue(np.all(image == 1))
        self.assertEqual(diagnostic["frame"]["frame_number"], 1)
        self.assertTrue(np.all(overlay[0, 0] == 255))
        self.assertTrue(np.all(image[0, 0] == 1))

    def test_pause_keeps_same_result_while_camera_continues_draining(self):
        _, session, analyses = self.run_loop([-1, ord("p"), ord("s"), ord("p"), ord("q")])
        image, diagnostic, _ = session.save.call_args.args
        self.assertTrue(np.all(image == 1))
        self.assertEqual(diagnostic["frame"]["frame_number"], 1)
        self.assertEqual(analyses, 2)

    def test_function_keys_keep_extended_codes_for_chinese_input_mode(self):
        _, session, analyses = self.run_loop([-1, 0x750000, 0x740000, 0x750000, 27])
        self.assertEqual(session.save.call_args.args[1]["frame"]["frame_number"], 1)
        self.assertEqual(analyses, 2)

    def test_bounded_run_saves_first_failure_and_exits(self):
        result, session, analyses = self.run_loop([-1, -1], frames=1, save_first=True)
        self.assertEqual(result["processed_frames"], 1)
        self.assertEqual(analyses, 1)
        session.save.assert_called_once()

    def test_save_during_initial_processing_does_not_save_unanalyzed_frame(self):
        _, session, _ = self.run_loop([ord("s"), ord("q")])
        session.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
