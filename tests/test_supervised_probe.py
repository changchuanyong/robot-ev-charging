"""Calibration bootstrap tests use synthetic parameters and fake devices only."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from control.config import hardware_template, simulation_config
from control.evaluate import evaluate
from control.probe import OperatorProbePolicy, prepare_probe
from control.replay import replay
from control.run import preflight, run_device
from control.simulate import SimulatedRobot
from control.state_machine import InsertionController
from tests.test_device_runner import work_directory
from tests.test_hardware import config as fake_hardware_config


def probe_config():
    config = simulation_config()
    config["method"] = "C"
    config["learning"].update(consistent_windows=1, min_hold_s=0.)
    config["experiment"] = {"session_id": "synthetic-session", "specimen_id": "synthetic-print",
                            "printing_batch_id": "synthetic-batch"}
    config["hardware"] = {**fake_hardware_config(), "acceptance_record": "synthetic-test-only",
                          "stop_feedback_timeout": .1,
                          "stop_tolerance": {"joint_speed_rad_s": .01, "linear_speed_m_s": .001,
                                             "angular_speed_rad_s": .01, "hold_s": .02}}
    return config


class FakeClock:
    now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, dt):
        self.now += dt


class FakeProbeDevice:
    def __init__(self, config, clock, stopped=True):
        self.sim = SimulatedRobot(config)
        self.clock = clock
        self.identity = {"fake": True}
        self.calls = []
        self.stop_command_ack = False
        self.stopped = stopped

    def connect(self):
        self.calls.append("connect")

    def read(self):
        self.sample = self.sim.read(self.clock.now)
        return self.sample

    def command(self, target, dt):
        self.calls.append("command")
        self.sim.command(target, dt)

    def stop(self):
        self.calls.append("stop")
        self.stop_command_ack = True
        self.sim.stop()

    def wait_stopped(self, *_):
        self.calls.append("confirm")
        return self.stopped

    def close(self):
        self.calls.append("close")


class ProbeTests(unittest.TestCase):
    def test_explicit_preflight_only_skips_checkpoint_not_physical_acceptance(self):
        config = probe_config()
        config["simulation_only"] = False
        config["calibration"]["expected_camera_serial"] = "fake-calibrated-camera"
        errors, _ = preflight(config)
        self.assertIn("learning method requires a trained checkpoint", errors)
        errors, policy = preflight(config, probe_action="+x")
        self.assertEqual(errors, [])
        self.assertIsInstance(policy, OperatorProbePolicy)
        errors, _ = preflight(prepare_probe(config, "+x"))
        self.assertTrue(any("explicit" in e for e in errors))
        config["calibration"]["validated"] = False
        errors, _ = preflight(config, probe_action="+x")
        self.assertTrue(any("calibration acceptance" in e for e in errors))
        config = hardware_template()
        config["method"] = "C"
        config["experiment"] = probe_config()["experiment"]
        self.assertTrue(preflight(config, probe_action="+x")[0])

    def test_probe_marker_cannot_enter_ordinary_controller_or_formal_task(self):
        config = prepare_probe(probe_config(), "+x")
        policy = OperatorProbePolicy(config["learning"], "+x")
        with self.assertRaisesRegex(ValueError, "explicit"):
            InsertionController(config, policy)
        config["experiment"]["formal"] = True
        self.assertTrue(preflight(config, probe_action="+x")[0])
        with self.assertRaisesRegex(ValueError, "non-Hold"):
            prepare_probe(probe_config(), "Hold")

    def test_probe_has_bounded_pretrial_history_and_no_model_claim(self):
        config = probe_config()
        config["learning"]["model_path"] = "unused/checkpoint.pt"
        prepared = prepare_probe(config, "+x")
        self.assertIsNone(prepared["learning"]["model_path"])
        self.assertFalse(prepared["supervised_probe"]["model_used"])
        policy = OperatorProbePolicy(prepared["learning"], "+x")
        for i in range(8):
            self.assertIsNone(policy.update(np.zeros(6), i * .01, "probe"))
        self.assertEqual(policy.update(np.zeros(6), .08, "probe").tolist(), [1., 0, 0, 0, 0, 0, 0, 0, 0])

    def run_fake(self, path, stopped=True):
        config = probe_config()
        clock = FakeClock()
        device = FakeProbeDevice(config, clock, stopped)
        def fake_preflight(c, *, probe_action=None):
            return [], OperatorProbePolicy(c["learning"], probe_action)
        with (patch("control.run.preflight", side_effect=fake_preflight),
              patch("control.hardware.RealManRobot", return_value=device),
              patch("control.run.read_observation", side_effect=lambda *a, **k: device.sim.observation(device.sample)),
              patch("control.run.time.monotonic", side_effect=clock.monotonic),
              patch("control.run.time.sleep", side_effect=clock.sleep)):
            run_device(config, "unused.json", path, probe_action="+x")
        return device

    def test_fake_device_stops_after_one_trial_and_excludes_task_statistics(self):
        with work_directory() as directory:
            path = directory / "probe.jsonl"
            device = self.run_fake(path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            manifest = rows[0]
            self.assertEqual(manifest["source"], "hardware")
            self.assertEqual(manifest["method_identity"], "PROBE")
            self.assertFalse(manifest["model_used"])
            result = rows[-1]
            self.assertEqual(result["reason"], "supervised_probe_completed")
            self.assertTrue(result["probe_completed"])
            self.assertFalse(result["success"])
            self.assertTrue(result["stop_feedback_confirmed"])
            cycles = [r for r in rows if r["kind"] == "cycle"]
            trial_results = [r for r in cycles if r["decision"]["details"].get("trial_result")]
            self.assertEqual(len(trial_results), 1)
            self.assertEqual(trial_results[0]["decision"]["details"]["trial_result"]["trial_id"], "1")
            self.assertIsNone(trial_results[0]["decision"]["command"])
            candidates = [r for r in cycles if r["decision"]["state"] == "INSERT" and
                          r["decision"]["details"].get("finite_trial", {}).get("phase") == "CANDIDATE"]
            self.assertTrue(candidates)
            self.assertTrue(all(not r["decision"]["details"]["advance_requested"] for r in candidates))
            self.assertEqual(device.calls[-3:], ["stop", "confirm", "close"])
            self.assertTrue(replay(path)["reproduced"])
            with self.assertRaisesRegex(ValueError, "calibration-only"):
                evaluate([path])

    def test_probe_requires_actual_stop_confirmation(self):
        with work_directory() as directory:
            path = directory / "unconfirmed.jsonl"
            with self.assertRaisesRegex(RuntimeError, "measured stop"):
                self.run_fake(path, stopped=False)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            results = [r for r in rows if r["kind"] == "result"]
            self.assertTrue(all(r.get("success") is False for r in results))
            self.assertTrue(all(r.get("stop_feedback_confirmed") is False for r in results))
            trials = [r["decision"]["details"]["trial_result"] for r in rows if
                      r.get("kind") == "cycle" and r["decision"]["details"].get("trial_result")]
            self.assertEqual(trials[-1]["outcome"], "invalid")
            self.assertEqual(trials[-1]["pre_stop_outcome"], "effective")


if __name__ == "__main__":
    unittest.main()
