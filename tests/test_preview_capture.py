"""Acquisition stays bounded and preserves RGB/depth identity during slow analysis."""
from queue import Queue
from types import SimpleNamespace
import unittest

import numpy as np

from vision.preview_capture import PreviewCapture


class Source:
    camera = {}
    config = {"timeout_ms": 100}
    rgb_controls = None

    def __init__(self):
        self.samples = Queue()

    def read(self):
        sample = self.samples.get(timeout=.3)
        if isinstance(sample, Exception):
            raise sample
        return sample

    read_rgbd = read


def sample(number):
    return (np.full((2, 3, 3), number, np.uint8),
            {"frame_number": number, "device_timestamp_ms": number * 33.},
            {"raw_z16": np.full((2, 3), 500 + number, np.uint16)})


class PreviewCaptureTests(unittest.TestCase):
    def test_latest_complete_pair_replaces_backlog_without_mutating_held_pair(self):
        source = Source()
        with PreviewCapture(source) as camera:
            source.samples.put(sample(1))
            first = camera.read_rgbd()
            source.samples.put(sample(2))
            source.samples.put(sample(3))
            with camera._condition:
                self.assertTrue(camera._condition.wait_for(lambda: camera._count == 3, timeout=1))
            latest = camera.read_rgbd()
            self.assertEqual(latest[1]["frame_number"], 3)
            self.assertTrue(np.all(latest[0] == 3))
            self.assertTrue(np.all(latest[2]["raw_z16"] == 503))
            self.assertTrue(np.all(first[0] == 1))
            self.assertTrue(np.all(first[2]["raw_z16"] == 501))
            self.assertEqual(camera.capture_stats()["captured_frames"], 3)
        self.assertFalse(camera._thread.is_alive())

    def test_reader_failure_reaches_consumer_and_thread_stops(self):
        source = Source()
        source.samples.put(RuntimeError("device disconnected"))
        with PreviewCapture(source) as camera:
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                camera.read_rgbd()
        self.assertFalse(camera._thread.is_alive())

    def test_repeated_frames_are_not_counted_as_new_capture_fps(self):
        source = Source()
        for _ in range(3):
            source.samples.put(sample(1))
        with PreviewCapture(source) as camera:
            with camera._condition:
                self.assertTrue(camera._condition.wait_for(lambda: camera._version == 3, timeout=1))
            self.assertEqual(camera.capture_stats()["captured_frames"], 1)
            self.assertEqual(camera.capture_stats()["capture_fps"], 0.)

    def test_rgb_only_wrapper_does_not_advertise_depth(self):
        source = Source()
        source.read_rgbd = None
        # RGB source exposes only read, not a callable read_rgbd.
        camera = PreviewCapture(source)
        self.assertFalse(callable(camera.read_rgbd))

    def test_sensor_commands_are_bounded_and_applied_on_reader(self):
        source = Source()
        applied = []
        source.rgb_controls = SimpleNamespace(toggle_auto=lambda: (applied.append(True)
                                            or {"values": {"enable_auto_exposure": 0}}))
        camera = PreviewCapture(source)
        for _ in range(8):
            camera.change_rgb("toggle_auto")
        self.assertIn("已满", camera.change_rgb("toggle_auto"))
        self.assertEqual(applied, [])
        camera._apply_commands()
        self.assertEqual(len(applied), 8)
        self.assertIn("自动曝光 0", camera.control_message)


if __name__ == "__main__":
    unittest.main()
