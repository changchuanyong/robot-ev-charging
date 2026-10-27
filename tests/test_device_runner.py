"""Runner semantics using a fake driver only; no device calls are made."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from control.run import run_device
from control.types import Decision, RobotSample


@contextmanager
def work_directory():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "test_device_runner"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != root.resolve():
            raise RuntimeError("test cleanup target changed")
        shutil.rmtree(directory)


class FakeDevice:
    def __init__(self, *, stopped=True, stop_error=False, invalid=False):
        self.identity = {"fake": True}
        self.calls = []
        self.seq = 0
        self.stopped = stopped
        self.stop_error = stop_error
        self.invalid = invalid
        self.stop_command_ack = False

    def connect(self):
        self.calls.append("connect")

    def read(self):
        self.seq += 1
        return RobotSample(self.seq, time.monotonic() - 0.001, np.eye(4),
                           np.full(6, np.nan) if self.invalid else np.zeros(6),
                           valid=not self.invalid, reason="injected_sensor_failure")

    def stop(self):
        self.calls.append("stop")
        if self.stop_error:
            raise RuntimeError("injected stop failure")
        self.stop_command_ack = True

    def wait_stopped(self, timeout, tol):
        self.calls.append("confirm")
        return self.stopped

    def close(self):
        self.calls.append("close")


class RunnerTests(unittest.TestCase):
    def run_fake(self, device, path, *, collect=False):
        controller = SimpleNamespace(state="INSERT", events=[],
            step=lambda *args: Decision("COMPLETE", stop=True, finished=True, success=True, reason="mechanical_fit"),
            acknowledge_stop=lambda: controller.events.append("feedback_stop_confirmed"))
        config = {"period": 0.01, "timing": {"max_sample_age": 1.0, "max_period": 0.1},
                  "hardware": {"stop_feedback_timeout": 0.1,
                               "stop_tolerance": {"joint_speed_rad_s": 0.01, "linear_speed_m_s": 0.001,
                                                  "angular_speed_rad_s": 0.01, "hold_s": 0.02}}}
        with (patch("control.run.preflight", return_value=([], None)),
              patch("control.run.hardware_config", return_value={}),
              patch("control.run.InsertionController", return_value=controller),
              patch("control.hardware.RealManRobot", return_value=device)):
            run_device(config, path.parent / "no-observation.json", path, collect_only=collect, duration=0.01)

    def test_success_requires_feedback_stop(self):
        with work_directory() as directory:
            device = FakeDevice()
            path = directory / "result.jsonl"
            self.run_fake(device, path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            result = next(r for r in rows if r["kind"] == "result")
            self.assertTrue(result["success"])
            self.assertTrue(result["stop_command_ack"])
            self.assertTrue(result["stop_feedback_confirmed"])
            self.assertEqual(device.calls[-1], "close")

    def test_unconfirmed_stop_is_not_success(self):
        with work_directory() as directory:
            device = FakeDevice(stopped=False)
            path = directory / "result.jsonl"
            with self.assertRaisesRegex(RuntimeError, "measured stop"):
                self.run_fake(device, path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            result = next(r for r in rows if r["kind"] == "result")
            self.assertFalse(result["success"])
            self.assertTrue(result["stop_command_ack"])
            self.assertFalse(result["stop_feedback_confirmed"])
            self.assertEqual(device.calls[-1], "close")

    def test_stop_failure_still_closes_and_logs(self):
        with work_directory() as directory:
            device = FakeDevice(stop_error=True)
            path = directory / "result.jsonl"
            with self.assertRaisesRegex(RuntimeError, "injected stop"):
                self.run_fake(device, path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["kind"] == "stop_error" for row in rows))
            self.assertEqual(device.calls[-1], "close")

    def test_read_only_collection_does_not_send_stop_on_failure(self):
        with work_directory() as directory:
            device = FakeDevice(invalid=True)
            path = directory / "result.jsonl"
            with self.assertRaisesRegex(RuntimeError, "invalid collection"):
                self.run_fake(device, path, collect=True)
            self.assertNotIn("stop", device.calls)
            self.assertEqual(device.calls[-1], "close")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            sample = next(r["sample"] for r in rows if r["kind"] == "sample")
            self.assertFalse(sample["valid"])
            self.assertEqual(sample["wrench"], [None] * 6)


if __name__ == "__main__":
    unittest.main()
