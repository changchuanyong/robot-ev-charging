"""Pure SDK doubles and scoped training compatibility checks; no device access."""

import sys
import tempfile
import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

from ev_charging.capture.collect import CaptureColorCamera, _stream_camera
from ev_charging.hardware.realsense import RealSenseCamera
from ev_charging.vision.contracts import intrinsics_signature


class FakeProfile:
    def __init__(self, width=32, format="bgr8"):
        self.intrinsics = SimpleNamespace(
            width=width,
            height=24,
            fx=1000.0,
            fy=1000.0,
            ppx=16.0,
            ppy=12.0,
            coeffs=[0.0] * 5,
            model=SimpleNamespace(name="none"),
        )
        self.pixel_format = format

    def get_intrinsics(self):
        return self.intrinsics

    def fps(self):
        return 30

    def format(self):
        return self.pixel_format

    def as_video_stream_profile(self):
        return self


def fake_sdk(*, width=32, model="D435"):
    video = FakeProfile(width)
    device = SimpleNamespace(
        get_info=lambda field: "TEST" if field == "serial" else f"Intel RealSense {model}",
        query_sensors=lambda: [],
    )
    profile = SimpleNamespace(get_device=lambda: device, get_stream=lambda stream: video)
    pipeline = SimpleNamespace(start=lambda config: profile, stopped=False)
    pipeline.stop = lambda: setattr(pipeline, "stopped", True)
    config = SimpleNamespace(enable_device=lambda serial: None, enable_stream=lambda *args: None)
    sdk = SimpleNamespace(
        config=lambda: config,
        pipeline=lambda: pipeline,
        context=lambda: SimpleNamespace(query_devices=lambda: [device]),
        camera_info=SimpleNamespace(serial_number="serial", name="name"),
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(bgr8="bgr8"),
    )
    return sdk, pipeline


class CameraContractTests(unittest.TestCase):
    def test_color_and_depth_metadata_have_original_fields(self):
        for stream in ("color", "depth"):
            metadata = _stream_camera(
                FakeProfile(format="bgr8" if stream == "color" else "z16"),
                stream,
                "TEST",
                "Intel RealSense D435",
                "D435",
            )
            self.assertEqual(metadata["camera_matrix"], [[1000.0, 0, 16.0], [0, 1000.0, 12.0], [0, 0, 1]])
            self.assertEqual(metadata["pixel_format"], "bgr8" if stream == "color" else "z16")
            self.assertEqual(metadata["stream"], stream)
            self.assertEqual(metadata["device_model"], "D435")
            self.assertFalse(metadata["validated"])
            self.assertEqual(metadata["intrinsics_sha256"], intrinsics_signature(metadata))

    def config(self):
        return {
            "serial": "TEST",
            "device_model": "D435",
            "width": 32,
            "height": 24,
            "fps": 30,
            "warmup_frames": 0,
            "max_frame_age_s": 0.5,
            "timeout_ms": 1000,
        }

    def test_acquisition_modes_preserve_validation_and_cleanup(self):
        for cls in (CaptureColorCamera, RealSenseCamera):
            sdk, pipeline = fake_sdk()
            with self.subTest(camera=cls.__name__), patch.dict(sys.modules, {"pyrealsense2": sdk}):
                with cls(self.config()) as source:
                    self.assertEqual(source.camera["serial"], "TEST")
                    self.assertFalse(source.camera["validated"])
                    self.assertEqual("device_model" in source.camera, cls is CaptureColorCamera)
                self.assertTrue(pipeline.stopped)

    def test_profile_mismatch_releases_started_stream(self):
        for cls in (CaptureColorCamera, RealSenseCamera):
            sdk, pipeline = fake_sdk(width=64)
            with self.subTest(camera=cls.__name__), patch.dict(sys.modules, {"pyrealsense2": sdk}):
                with self.assertRaises(ValueError):
                    with cls(self.config()):
                        self.fail("mismatched stream was accepted")
                self.assertTrue(pipeline.stopped)

    def test_control_camera_does_not_accept_unverified_calibration(self):
        sdk, pipeline = fake_sdk()
        config = {**self.config(), "intrinsics_validated": True, "calibration_version": "claimed"}
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            with self.assertRaisesRegex(ValueError, "validated_intrinsics_require_matching_version"):
                with RealSenseCamera(config):
                    self.fail("unbound calibration was accepted")
        self.assertTrue(pipeline.stopped)


class TrainingScopeTests(unittest.TestCase):
    def setUp(self):
        from ev_charging import runtime

        runtime.configure_caches()
        self.runtime = runtime
        self.modules = [import_module(name) for name in ("ultralytics.data.base", "ultralytics.data.dataset")]
        self.original = [
            (module, module.ThreadPool) for module in self.modules if hasattr(module, "ThreadPool")
        ]

    def assert_restored(self):
        for module, original in self.original:
            self.assertIs(module.ThreadPool, original)

    def test_serial_scope_restores_after_interrupt(self):
        with patch.object(self.runtime, "_named_pipes_available", return_value=False):
            with self.assertRaises(KeyboardInterrupt):
                with self.runtime.label_scan_scope() as serial:
                    self.assertTrue(serial)
                    for module, _ in self.original:
                        self.assertIs(module.ThreadPool, self.runtime._SerialPool)
                    raise KeyboardInterrupt
        self.assert_restored()

    def test_nested_scopes_preserve_outer_patch(self):
        with patch.object(self.runtime, "_named_pipes_available", return_value=False):
            with self.runtime.label_scan_scope():
                with self.runtime.label_scan_scope():
                    pass
                for module, _ in self.original:
                    self.assertIs(module.ThreadPool, self.runtime._SerialPool)
        self.assert_restored()

    def test_normal_scope_does_not_change_pools(self):
        with patch.object(self.runtime, "_named_pipes_available", return_value=True):
            with self.runtime.label_scan_scope() as serial:
                self.assertFalse(serial)
                self.assert_restored()

    def test_training_failure_before_model_load_restores_pools(self):
        from ev_charging.training.train_port import train_all

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.runtime, "_named_pipes_available", return_value=False):
                with patch(
                    "ev_charging.training.train_port.validate_device",
                    side_effect=RuntimeError("fixture stop"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "fixture stop"):
                        train_all(
                            directory,
                            {"entries": []},
                            {"epochs": 1, "imgsz": 32, "batch": 1, "device": "cpu"},
                        )
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
