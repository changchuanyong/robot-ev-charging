"""RGBD identity, mapping, and pair rejection tests without camera access."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vision.rgbd_capture import CaptureRGBDCamera


def config():
    return {"device_model": "D415", "serial": "exact-serial", "width": 8, "height": 6,
            "fps": 15, "timeout_ms": 5000, "warmup_frames": 0}


def depth_config():
    return {"width": 4, "height": 3, "fps": 15, "max_pair_skew_ms": 40.}


def sdk_fixture():
    sdk = MagicMock()
    sdk.camera_info = SimpleNamespace(name="name", serial_number="serial", firmware_version="firmware",
                                      usb_type_descriptor="usb")
    sdk.stream = SimpleNamespace(color="color", depth="depth")
    sdk.format = SimpleNamespace(bgr8="bgr8", z16="z16")
    device = MagicMock()
    device.get_info.side_effect = {"name": "RealSense D415", "serial": "exact-serial",
                                   "firmware": "test", "usb": "2.1"}.__getitem__
    device.first_depth_sensor.return_value.get_depth_scale.return_value = .001
    sdk.context.return_value.query_devices.return_value = [device]
    profile = sdk.pipeline.return_value.start.return_value
    profile.get_device.return_value = device
    videos = {}
    for stream, width, height, fmt in (("color", 8, 6, "bgr8"), ("depth", 4, 3, "z16")):
        video = MagicMock()
        video.as_video_stream_profile.return_value = video
        video.get_intrinsics.return_value = SimpleNamespace(
            width=width, height=height, fx=100., fy=101., ppx=2., ppy=1.,
            coeffs=[0.] * 5, model=SimpleNamespace(name="brown_conrady"))
        video.fps.return_value, video.format.return_value = 15, fmt
        videos[stream] = video
    profile.get_stream.side_effect = videos.__getitem__
    # A 90-degree Z rotation, expressed in the SDK's COLUMN-MAJOR layout.
    videos["depth"].get_extrinsics_to.return_value = SimpleNamespace(
        rotation=[0., 1., 0., -1., 0., 0., 0., 0., 1.], translation=[.01, .02, .03])
    frames = sdk.pipeline.return_value.wait_for_frames.return_value
    color, depth = frames.get_color_frame.return_value, frames.get_depth_frame.return_value
    color.get_data.return_value = np.full((6, 8, 3), 40, np.uint8)
    depth.get_data.return_value = np.full((3, 4), 1000, np.uint16)
    for frame, number, timestamp in ((color, 20, 1000.), (depth, 12, 1001.)):
        frame.get_frame_number.return_value, frame.get_timestamp.return_value = number, timestamp
        frame.get_frame_timestamp_domain.return_value = SimpleNamespace(name="hardware_clock")
    points = sdk.pointcloud.return_value.calculate.return_value
    points.get_vertices.return_value = np.tile(np.array([.1, .2, 1.], np.float32), (12, 1))
    points.get_texture_coordinates.return_value = np.tile(np.array([.25, .5], np.float32), (12, 1))
    return sdk, videos, color, depth


class RGBDCaptureTests(unittest.TestCase):
    def test_exact_dual_streams_and_independent_metadata(self):
        sdk, videos, color, depth = sdk_fixture()
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureRGBDCamera(config(), depth_config()) as camera:
                calls = sdk.config.return_value.enable_stream.call_args_list
                self.assertEqual([call.args for call in calls],
                                 [("color", 8, 6, "bgr8", 15), ("depth", 4, 3, "z16", 15)])
                image, frame, packet = camera.read_rgbd()
                self.assertEqual(frame["frame_number"], 20)
                self.assertEqual(packet["frame"]["frame_number"], 12)
                self.assertEqual(packet["color_frame"], frame)
                self.assertIsNot(packet["color_frame"], frame)
                self.assertEqual(packet["reasons"], [])
                self.assertEqual(packet["pair_skew_ms"], 1.)
                self.assertEqual(camera.depth_calibration["color_intrinsics_sha256"],
                                 camera.camera["intrinsics_sha256"])
                self.assertEqual(camera.depth_calibration["depth_camera"]["stream"], "depth")
                self.assertFalse(camera.depth_calibration["validated"])
                self.assertEqual(camera.depth_calibration["depth_scale_m"], .001)
                color.get_data.return_value[:] = 99
                depth.get_data.return_value[:] = 0
                self.assertTrue(np.all(image == 40))
                self.assertTrue(np.all(packet["raw_z16"] == 1000))
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_sdk_metres_rotation_and_texture_mapping(self):
        sdk, _, _, depth = sdk_fixture()
        depth.get_data.return_value[0, 0] = 0
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureRGBDCamera(config(), depth_config()) as camera:
                _, _, packet = camera.read_rgbd()
        np.testing.assert_allclose(packet["points_color_m"][1, 1], [-.19, .12, 1.03], atol=1e-6)
        np.testing.assert_allclose(packet["color_pixels"][1, 1], [2., 3.])
        self.assertTrue(np.isnan(packet["points_color_m"][0, 0]).all())
        self.assertTrue(np.isnan(packet["color_pixels"][0, 0]).all())

    def test_bad_pairs_keep_raw_but_never_create_points(self):
        cases = (("skew", "depth_pair_skew_exceeded"),
                 ("domain", "depth_timestamp_domain_mismatch"),
                 ("zero", "invalid_depth_timestamp"),
                 ("nan", "invalid_depth_timestamp"),
                 ("unknown_domain", "invalid_depth_timestamp_domain"))
        for case, reason in cases:
            sdk, _, _, depth = sdk_fixture()
            if case == "skew":
                depth.get_timestamp.return_value = 1100.
            elif case == "zero":
                depth.get_timestamp.return_value = 0.
            elif case == "nan":
                depth.get_timestamp.return_value = float("nan")
            else:
                depth.get_frame_timestamp_domain.return_value.name = "global_time" if case == "domain" else "other"
            with self.subTest(case=case), patch.dict("sys.modules", {"pyrealsense2": sdk}):
                with CaptureRGBDCamera(config(), depth_config()) as camera:
                    image, _, packet = camera.read_rgbd()
                self.assertEqual(image.shape, (6, 8, 3))
                self.assertEqual(packet["raw_z16"].dtype, np.uint16)
                self.assertIn(reason, packet["reasons"])
                self.assertIsNone(packet["points_color_m"])
                sdk.pointcloud.return_value.calculate.assert_not_called()

    def test_missing_depth_preserves_rgb(self):
        sdk, _, _, _ = sdk_fixture()
        sdk.pipeline.return_value.wait_for_frames.return_value.get_depth_frame.return_value = None
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureRGBDCamera(config(), depth_config()) as camera:
                image, _, packet = camera.read_rgbd()
        self.assertEqual(image.shape, (6, 8, 3))
        self.assertEqual(packet["reasons"], ["missing_depth"])
        self.assertIsNone(packet["raw_z16"])
        self.assertIsNone(packet["frame"])

    def test_reused_or_reordered_frames_are_rejected_until_newer(self):
        sdk, _, color, depth = sdk_fixture()
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureRGBDCamera(config(), depth_config()) as camera:
                camera.read_rgbd()
                _, _, duplicate = camera.read_rgbd()
                self.assertIn("repeated_or_reordered_depth_frame", duplicate["reasons"])
                self.assertIn("repeated_or_reordered_color_frame", duplicate["reasons"])
                color.get_frame_number.return_value, depth.get_frame_number.return_value = 19, 11
                _, _, old = camera.read_rgbd()
                self.assertIsNone(old["points_color_m"])
                color.get_frame_number.return_value, depth.get_frame_number.return_value = 21, 13
                color.get_timestamp.return_value, depth.get_timestamp.return_value = 1067., 1068.
                _, _, new = camera.read_rgbd()
                self.assertEqual(new["reasons"], [])

    def test_frame_number_alone_cannot_override_stale_timestamp(self):
        sdk, _, color, depth = sdk_fixture()
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureRGBDCamera(config(), depth_config()) as camera:
                camera.read_rgbd()
                color.get_frame_number.return_value, depth.get_frame_number.return_value = 21, 13
                _, _, packet = camera.read_rgbd()
        self.assertIn("repeated_or_reordered_depth_frame", packet["reasons"])
        self.assertIsNone(packet["points_color_m"])

    def test_projection_failure_does_not_lose_raw_rgb_or_depth(self):
        sdk, *_ = sdk_fixture()
        sdk.pointcloud.return_value.calculate.side_effect = RuntimeError("unavailable")
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureRGBDCamera(config(), depth_config()) as camera:
                image, _, packet = camera.read_rgbd()
        self.assertEqual(image.shape, (6, 8, 3))
        self.assertIn("depth_projection_failed", packet["reasons"])
        self.assertEqual(packet["raw_z16"].dtype, np.uint16)

    def test_mismatched_actual_profiles_never_silently_downgrade(self):
        for stream in ("color", "depth"):
            sdk, videos, *_ = sdk_fixture()
            videos[stream].fps.return_value = 6
            with self.subTest(stream=stream), patch.dict("sys.modules", {"pyrealsense2": sdk}):
                with self.assertRaisesRegex(ValueError, "profile_mismatch"):
                    with CaptureRGBDCamera(config(), depth_config()):
                        pass
                sdk.pipeline.return_value.stop.assert_called_once()

    def test_wrong_identity_never_starts(self):
        sdk, *_ = sdk_fixture()
        settings = {**config(), "serial": "another-serial"}
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with self.assertRaisesRegex(ValueError, "model_or_serial"):
                with CaptureRGBDCamera(settings, depth_config()):
                    pass
        sdk.pipeline.assert_not_called()

    def test_invalid_depth_config_fails_without_camera_access(self):
        for key, value in (("width", True), ("height", 0), ("fps", 1.5),
                           ("max_pair_skew_ms", float("nan")), ("max_pair_skew_ms", 0.)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                CaptureRGBDCamera(config(), {**depth_config(), key: value})

    def test_startup_error_or_interrupt_stops_pipeline(self):
        for error in (RuntimeError("unsupported combination"), KeyboardInterrupt()):
            sdk, *_ = sdk_fixture()
            sdk.pipeline.return_value.start.side_effect = error
            with self.subTest(error=type(error)), patch.dict("sys.modules", {"pyrealsense2": sdk}):
                with self.assertRaises(type(error)):
                    with CaptureRGBDCamera(config(), depth_config()):
                        pass
            sdk.pipeline.return_value.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
