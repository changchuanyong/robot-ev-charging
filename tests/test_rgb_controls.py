"""RGB option range, per-frame provenance, and restoration without hardware."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from vision.collect import CaptureColorCamera
from vision.rgb_controls import OPTION_NAMES, RGBControls, validate_rgb_options
from vision.rgbd_capture import CaptureRGBDCamera
from tests.test_data_capture import config as color_config, sdk_fixture as color_sdk
from tests.test_rgbd_capture import config as rgbd_config, depth_config, sdk_fixture as rgbd_sdk


class Sensor:
    def __init__(self):
        self.values = {"enable_auto_exposure": 1., "exposure": 100., "gain": 64.,
                       "enable_auto_white_balance": 1., "auto_exposure_priority": 1.}
        self.ranges = {name: SimpleNamespace(min=0., max=1., step=1., default=1.) for name in OPTION_NAMES}
        self.ranges["exposure"] = SimpleNamespace(min=10., max=1000., step=10., default=100.)
        self.ranges["gain"] = SimpleNamespace(min=0., max=128., step=1., default=64.)
        self.read_only = set()
        self.writes = []
        self.reads = 0
        self.fail_on = None
        self.ignore_writes = False

    def supports(self, option):
        return option in self.values

    def get_option_range(self, option):
        return self.ranges[option]

    def is_option_read_only(self, option):
        return option in self.read_only

    def get_option(self, option):
        self.reads += 1
        return self.values[option]

    def set_option(self, option, value):
        self.writes.append((option, value))
        if self.fail_on == option:
            raise RuntimeError("USB option write failed")
        if option in ("exposure", "gain") and self.values["enable_auto_exposure"]:
            raise RuntimeError("manual value while AE is enabled")
        if not self.ignore_writes:
            self.values[option] = value


def add_controls(sdk=None, sensor=None):
    sensor = sensor or Sensor()
    sdk = sdk or MagicMock()
    sdk.option = SimpleNamespace(**{name: name for name in OPTION_NAMES})
    fields = ("actual_exposure", "gain_level", "auto_exposure", "white_balance")
    sdk.frame_metadata_value = SimpleNamespace(**{name: name for name in fields})
    device = sdk.pipeline.return_value.start.return_value.get_device.return_value
    device.first_color_sensor.return_value = sensor
    return sdk, sensor, device


class RGBControlTests(unittest.TestCase):
    def setUp(self):
        platform = patch("vision.rgb_controls.sys.platform", "linux")
        platform.start()
        self.addCleanup(platform.stop)
        sleeping = patch("vision.rgb_controls.time.sleep")
        sleeping.start()
        self.addCleanup(sleeping.stop)

    def test_invalid_configuration_is_rejected_without_hardware(self):
        for options in (None, [], {"depth_exposure": 4}, {"exposure": True},
                        {"gain": float("nan")}, {"enable_auto_exposure": 2},
                        {"enable_auto_exposure": 1, "exposure": 200}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_rgb_options(options)
        validate_rgb_options({"enable_auto_exposure": True})

    def test_all_ranges_and_support_are_checked_before_any_write(self):
        for bad in ({"gain": 129}, {"exposure": 115}, {"exposure": 1}, {"enable_auto_white_balance": 0}):
            sdk, sensor, device = add_controls()
            if "enable_auto_white_balance" in bad:
                del sensor.values["enable_auto_white_balance"]
            controls = RGBControls(device, sdk)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                controls.apply({"auto_exposure_priority": 0, **bad})
            self.assertEqual(sensor.writes, [])
        sdk, sensor, device = add_controls()
        sensor.read_only.add("gain")
        with self.assertRaisesRegex(ValueError, "read_only"):
            RGBControls(device, sdk).apply({"gain": 65})
        self.assertEqual(sensor.writes, [])

    def test_manual_adjustments_disable_auto_and_use_sensor_steps(self):
        sdk, sensor, device = add_controls()
        controls = RGBControls(device, sdk)
        result = controls.adjust_exposure(1)
        self.assertEqual(sensor.writes[0], ("enable_auto_exposure", 0.))
        self.assertEqual(result["values"]["exposure"], 130.)
        controls.adjust_gain(1)
        self.assertEqual(sensor.values["gain"], 70.)
        controls.toggle_auto()
        self.assertEqual(sensor.values["enable_auto_exposure"], 1.)
        controls.restore()
        self.assertEqual(sensor.values, Sensor().values)

    def test_exposure_and_gain_clamp_to_supported_bounds(self):
        sdk, sensor, device = add_controls()
        controls = RGBControls(device, sdk)
        controls.apply({"exposure": 1000, "gain": 0})
        self.assertEqual(controls.adjust_exposure(1)["values"]["exposure"], 1000)
        self.assertEqual(controls.adjust_gain(-1)["values"]["gain"], 0)
        controls.restore()

    def test_restore_original_manual_values_after_auto_changed_them(self):
        sdk, sensor, device = add_controls()
        sensor.values["enable_auto_exposure"] = 0.
        original = deepcopy(sensor.values)
        controls = RGBControls(device, sdk)
        controls.apply({"enable_auto_exposure": 1, "auto_exposure_priority": 0})
        sensor.values.update(exposure=500., gain=100.)
        controls.restore()
        self.assertEqual(sensor.values, original)
        count = len(sensor.writes)
        controls.restore()
        self.assertEqual(len(sensor.writes), count)

    def test_ignored_write_is_not_claimed_as_applied(self):
        sdk, sensor, device = add_controls()
        controls = RGBControls(device, sdk)
        sensor.ignore_writes = True
        with self.assertRaisesRegex(RuntimeError, "readback_mismatch"):
            controls.apply({"auto_exposure_priority": 0})

    def test_unchanged_startup_options_do_not_write_or_reset_exposure_on_restore(self):
        sdk, sensor, device = add_controls()
        controls = RGBControls(device, sdk)
        controls.apply({"enable_auto_exposure": 1, "auto_exposure_priority": 1})
        controls.restore()
        self.assertEqual(sensor.writes, [])

    def test_windows_exposure_quantization_records_actual_and_shortcut_crosses_step(self):
        sdk, sensor, device = add_controls()
        sensor.values["exposure"] = 156.
        sensor.ranges["exposure"] = SimpleNamespace(min=1., max=10000., step=1., default=156.)
        with patch("vision.rgb_controls.sys.platform", "win32"):
            controls = RGBControls(device, sdk)
        original_set = sensor.set_option

        def quantized_set(option, value):
            original_set(option, controls._expected_readback(option, value))

        sensor.set_option = quantized_set
        result = controls.apply({"exposure": 195})
        self.assertEqual(result["values"]["exposure"], 156.)
        self.assertEqual(result["last_changes"]["exposure"],
                         {"requested": 195., "readback": 156., "status": "quantized"})
        self.assertEqual(controls.adjust_exposure(1)["values"]["exposure"], 312.)
        self.assertEqual(controls.adjust_exposure(-1)["values"]["exposure"], 156.)
        controls.restore()
        self.assertEqual(sensor.values["enable_auto_exposure"], 1.)

    def test_delayed_readback_is_checked_again_before_claiming_failure(self):
        sdk, sensor, device = add_controls()
        controls = RGBControls(device, sdk)
        original_get, original_set = sensor.get_option, sensor.set_option
        stale = []

        def delayed_set(option, value):
            if option == "auto_exposure_priority":
                stale.extend([1., 1.])
            original_set(option, value)

        sensor.set_option = delayed_set
        sensor.get_option = lambda option: stale.pop(0) if option == "auto_exposure_priority" and stale else original_get(option)
        self.assertEqual(controls.apply({"auto_exposure_priority": 0})["values"]["auto_exposure_priority"], 0.)

    def test_disabling_auto_preserves_exposure_when_driver_resets_it(self):
        sdk, sensor, device = add_controls()
        controls = RGBControls(device, sdk)
        original_set = sensor.set_option

        def resetting_set(option, value):
            original_set(option, value)
            if option == "enable_auto_exposure" and value == 0:
                sensor.values["exposure"] = 1000.

        sensor.set_option = resetting_set
        self.assertEqual(controls.toggle_auto()["values"]["exposure"], 100.)
        controls.adjust_gain(1)
        self.assertEqual(sensor.values["exposure"], 100.)
        controls.restore()
        self.assertEqual(sensor.values, Sensor().values)

    def test_frame_metadata_is_separate_from_cached_sensor_options(self):
        sdk, sensor, device = add_controls()
        controls = RGBControls(device, sdk)
        frame = MagicMock()
        actual = {"actual_exposure": 9000, "gain_level": 88, "auto_exposure": 1}
        frame.supports_frame_metadata.side_effect = lambda field: field in actual
        frame.get_frame_metadata.side_effect = actual.__getitem__
        reads = sensor.reads
        record = controls.frame_settings(frame)
        self.assertEqual(record["sensor_options"]["values"]["exposure"], 100)
        self.assertEqual(record["frame_metadata"]["values"]["actual_exposure"], 9000)
        self.assertEqual(sensor.reads, reads)
        record["sensor_options"]["values"]["gain"] = -1
        self.assertEqual(controls.frame_settings(frame)["sensor_options"]["values"]["gain"], 64)
        frame.get_frame_metadata.side_effect = RuntimeError("metadata unavailable")
        self.assertIn("actual_exposure", controls.frame_settings(frame)["frame_metadata"]["errors"])


class RGBCameraIntegrationTests(unittest.TestCase):
    def test_rgb_frame_contains_settings_and_exit_restores(self):
        sdk, _, _, frame = color_sdk()
        sdk, sensor, _ = add_controls(sdk)
        frame.supports_frame_metadata.return_value = False
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureColorCamera({**color_config(), "rgb_options": {"exposure": 200}}) as camera:
                _, metadata = camera.read()
                self.assertEqual(metadata["rgb_settings"]["sensor_options"]["values"]["exposure"], 200)
                self.assertEqual(metadata["rgb_settings"]["frame_metadata"]["values"], {})
            self.assertEqual(sensor.values, Sensor().values)
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_rgbd_binding_includes_same_settings_and_keeps_calibration(self):
        sdk, _, frame, _ = rgbd_sdk()
        sdk, sensor, _ = add_controls(sdk)
        frame.supports_frame_metadata.return_value = False
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with CaptureRGBDCamera({**rgbd_config(), "rgb_options": {"gain": 80}}, depth_config()) as camera:
                _, metadata, packet = camera.read_rgbd()
                self.assertEqual(packet["color_frame"], metadata)
                self.assertEqual(metadata["rgb_settings"]["sensor_options"]["values"]["gain"], 80)
                self.assertEqual(packet["reasons"], [])
            self.assertEqual(sensor.values, Sensor().values)
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_warmup_failure_restores_before_stopping(self):
        sdk, _, _, frame = color_sdk()
        sdk, sensor, _ = add_controls(sdk)
        sdk.pipeline.return_value.wait_for_frames.side_effect = RuntimeError("frame timeout")
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaisesRegex(RuntimeError, "frame timeout"):
            with CaptureColorCamera({**color_config(), "warmup_frames": 1,
                                     "rgb_options": {"exposure": 200}}):
                self.fail("Expected startup failure")
        self.assertEqual(sensor.values, Sensor().values)
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_startup_write_failure_restores_partial_changes(self):
        sdk, _, _, _ = color_sdk()
        sdk, sensor, _ = add_controls(sdk)
        sensor.fail_on = "gain"
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaisesRegex(RuntimeError, "USB option write failed"):
            with CaptureColorCamera({**color_config(), "rgb_options": {"gain": 80}}):
                self.fail("Expected SDK error")
        self.assertEqual(sensor.values["enable_auto_exposure"], 1.)
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_transient_write_failure_restores_original_state(self):
        sdk, _, _, _ = color_sdk()
        sdk, sensor, _ = add_controls(sdk)
        original_set = sensor.set_option

        def fail_once(option, value):
            if option == "gain":
                sensor.set_option = original_set
                raise RuntimeError("transient USB failure")
            original_set(option, value)

        sensor.set_option = fail_once
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaisesRegex(RuntimeError, "transient USB"):
            with CaptureColorCamera({**color_config(), "rgb_options": {"gain": 80}}):
                self.fail("Expected SDK error")
        self.assertEqual(sensor.values, Sensor().values)
        sdk.pipeline.return_value.stop.assert_called_once()

    def test_restore_failure_is_reported_but_pipeline_still_stops(self):
        sdk, _, _, _ = color_sdk()
        sdk, sensor, _ = add_controls(sdk)
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaisesRegex(RuntimeError, "restore_failed"):
            with CaptureColorCamera({**color_config(), "rgb_options": {"exposure": 200}}):
                sensor.fail_on = "exposure"
        self.assertEqual(sensor.values["enable_auto_exposure"], 1.)
        sdk.pipeline.return_value.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
