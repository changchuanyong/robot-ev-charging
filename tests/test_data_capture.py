"""Offline D415 acquisition tests; the SDK and display are replaced, not images."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vision import collect
from vision.contracts import intrinsics_signature
from vision.realsense import RealSenseCamera


def config():
    result = json.loads((collect.ROOT / "config/data_capture_d415.json").read_text())
    result.update(serial="d415-exact-serial", width=8, height=6, warmup_frames=0)
    return result


def sdk_fixture(name="Intel RealSense D415"):
    sdk = MagicMock()
    sdk.camera_info = SimpleNamespace(name="name", serial_number="serial_number")
    sdk.stream = SimpleNamespace(color="color")
    sdk.format = SimpleNamespace(bgr8="bgr8")
    device = MagicMock()
    device.get_info.side_effect = lambda field: {"name": name, "serial_number": "d415-exact-serial"}[field]
    sdk.context.return_value.query_devices.return_value = [device]
    profile = sdk.pipeline.return_value.start.return_value
    profile.get_device.return_value = device
    video = profile.get_stream.return_value.as_video_stream_profile.return_value
    video.get_intrinsics.return_value = SimpleNamespace(
        width=8, height=6, fx=100., fy=101., ppx=4., ppy=3.,
        coeffs=[.1, 0., 0., 0., 0.], model=SimpleNamespace(name="brown_conrady"))
    video.fps.return_value = 30
    video.format.return_value = "bgr8"
    frame = sdk.pipeline.return_value.wait_for_frames.return_value.get_color_frame.return_value
    frame.get_data.return_value = np.full((6, 8, 3), 40, dtype=np.uint8)
    frame.get_frame_number.return_value = 125
    frame.get_timestamp.return_value = 4166.7
    frame.get_frame_timestamp_domain.return_value = SimpleNamespace(name="hardware_clock")
    return sdk, profile, video, frame


class CaptureCameraTests(unittest.TestCase):
    def test_d435_acquisition_requires_explicit_model_and_exact_serial(self):
        sdk, *_ = sdk_fixture("Intel RealSense D435")
        settings = config()
        settings["device_model"] = "D435"
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with collect.CaptureColorCamera(settings) as camera:
                self.assertEqual(camera.camera["device_model"], "D435")
                self.assertEqual(camera.camera["serial"], settings["serial"])
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_only_requested_color_stream_and_actual_intrinsics_are_used(self):
        sdk, _, _, frame = sdk_fixture()
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with collect.D415ColorCamera(config()) as camera:
                sdk.config.return_value.enable_device.assert_called_once_with("d415-exact-serial")
                sdk.config.return_value.enable_stream.assert_called_once_with("color", 8, 6, "bgr8", 30)
                self.assertFalse(camera.camera["validated"])
                self.assertIsNone(camera.camera["version"])
                self.assertEqual(camera.camera["intrinsics_sha256"], intrinsics_signature(camera.camera))
                self.assertEqual(camera.camera["camera_matrix"][0][0], 100.)
                image, metadata = camera.read()
                frame.get_data.return_value[:] = 99
                self.assertTrue(np.all(image == 40))
                self.assertEqual(metadata["timestamp_domain"], "hardware_clock")
                self.assertEqual(metadata["frame_number"], 125)
                self.assertTrue(metadata["received_at_utc"].endswith("+00:00"))
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_wrong_model_or_serial_never_starts_capture(self):
        for model, serial in (("Intel RealSense D435", "d415-exact-serial"),
                              ("Intel RealSense D415", "different-serial")):
            sdk, *_ = sdk_fixture(model)
            settings = config()
            settings["serial"] = serial
            with self.subTest(model=model, serial=serial), patch.dict("sys.modules", {"pyrealsense2": sdk}):
                with self.assertRaises(ValueError):
                    with collect.D415ColorCamera(settings):
                        self.fail("Wrong camera was accepted")
                sdk.pipeline.assert_not_called()

    def test_mismatched_actual_profile_stops_camera(self):
        for change in ("resolution", "fps", "format", "serial"):
            sdk, profile, video, _ = sdk_fixture()
            if change == "resolution":
                video.get_intrinsics.return_value.width = 9
            elif change == "fps":
                video.fps.return_value = 15
            elif change == "format":
                video.format.return_value = "rgb8"
            else:
                other = MagicMock()
                other.get_info.side_effect = lambda field: {"name": "Intel RealSense D415", "serial_number": "wrong"}[field]
                profile.get_device.return_value = other
            with self.subTest(change=change), patch.dict("sys.modules", {"pyrealsense2": sdk}):
                with self.assertRaises(ValueError):
                    with collect.D415ColorCamera(config()):
                        pass
                sdk.pipeline.return_value.stop.assert_called_once()

    def test_warmup_failure_or_interrupt_closes_started_pipeline(self):
        for error in (RuntimeError("timeout"), KeyboardInterrupt()):
            sdk, *_ = sdk_fixture()
            sdk.pipeline.return_value.wait_for_frames.side_effect = error
            settings = config()
            settings["warmup_frames"] = 1
            with self.subTest(error=type(error)), patch.dict("sys.modules", {"pyrealsense2": sdk}):
                with self.assertRaises(type(error)):
                    with collect.D415ColorCamera(settings):
                        pass
                sdk.pipeline.return_value.stop.assert_called_once()

    def test_d435_controller_still_rejects_d415(self):
        sdk, *_ = sdk_fixture()
        settings = {**config(), "max_frame_age_s": .5}
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with self.assertRaisesRegex(ValueError, "not_a_D435"):
                with RealSenseCamera(settings):
                    pass
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_missing_serial_is_rejected_before_sdk_import_or_access(self):
        settings = config()
        settings["serial"] = None
        sdk, *_ = sdk_fixture()
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaises(ValueError):
            collect.D415ColorCamera(settings)
        sdk.context.assert_not_called()


class CaptureLoopTests(unittest.TestCase):
    def test_interval_saves_without_keypress_and_stops_at_image_limit(self):
        camera, session = MagicMock(), MagicMock()
        camera.read.return_value = np.zeros((60, 80, 3), dtype=np.uint8), {"frame_number": 5}
        session.save.return_value = {"image": "images/frame_000001.png"}
        with ExitStack() as stack:
            for name in ("namedWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch.object(collect.cv2, name))
            stack.enter_context(patch.object(collect.cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch.object(collect.cv2, "waitKey", return_value=-1))
            stack.enter_context(patch.object(collect.time, "monotonic", side_effect=[0., 3.]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(collect.collect(camera, session, interval=2., max_images=1), 1)
        session.save.assert_called_once()

    def test_save_uses_full_original_image_not_preview_overlay(self):
        image = np.full((60, 80, 3), 100, dtype=np.uint8)
        camera, session = MagicMock(), MagicMock()
        camera.read.return_value = image, {"frame_number": 1}
        session.save.return_value = {"image": "images/frame_000001.png"}
        with ExitStack() as stack:
            for name in ("namedWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch.object(collect.cv2, name))
            stack.enter_context(patch.object(collect.cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch.object(collect.cv2, "waitKey", return_value=32))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(collect.collect(camera, session, max_images=1), 1)
        saved = session.save.call_args.args[0]
        self.assertIs(saved, image)
        self.assertTrue(np.all(saved == 100))

    def test_list_devices_does_not_start_stream_or_create_session(self):
        sdk, *_ = sdk_fixture()
        with (patch.dict("sys.modules", {"pyrealsense2": sdk}),
              patch("vision.capture_session.CaptureSession") as session,
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(collect.main(["--list-devices"]), 0)
        self.assertEqual(json.loads(output.getvalue())[0]["serial"], "d415-exact-serial")
        sdk.pipeline.assert_not_called()
        session.assert_not_called()

    def test_interruption_and_save_error_finalize_session_and_release_camera(self):
        for error, status, code in ((KeyboardInterrupt(), "interrupted", 130),
                                    (OSError("disk full"), "failed", 2)):
            sdk, *_ = sdk_fixture()
            with (self.subTest(status=status), patch.dict("sys.modules", {"pyrealsense2": sdk}),
                  patch.object(collect, "collect", side_effect=error),
                  patch("vision.capture_session.CaptureSession") as factory,
                  patch.object(collect.json, "loads", return_value=config()),
                  redirect_stdout(io.StringIO())):
                self.assertEqual(collect.main(["--serial", "d415-exact-serial", "--batch", "batch01"]), code)
                self.assertEqual(factory.return_value.finish.call_args.args[0], status)
                sdk.pipeline.return_value.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
