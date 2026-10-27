import json
from pathlib import Path
import unittest
import uuid

import numpy as np

from control.config import hardware_template, simulation_config, validate_config
from control.evaluate import evaluate, wilson
from control.logging import TaskLog
from control.replay import replay
from control.run import preflight, read_observation
from control.simulate import run_simulation


class RuntimeToolsTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / "artifacts" / ("runtime-test-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)

    def tearDown(self):
        for p in self.directory.iterdir():
            p.unlink()
        self.directory.rmdir()

    def test_empty_hardware_template_refused_without_sdk_or_connection(self):
        errors, policy = preflight(hardware_template())
        self.assertTrue(errors)
        self.assertIsNone(policy)

    def test_simulation_never_accepted_as_hardware(self):
        self.assertIn("hardware refuses a simulation configuration", validate_config(simulation_config(), hardware=True))

    def test_malformed_configuration_returns_errors(self):
        for bad in (None, [], {"task": None}):
            self.assertTrue(validate_config(bad))
        c = simulation_config()
        c["method"] = []
        self.assertTrue(validate_config(c))

    def test_invalid_channel_calibration_rejected_early(self):
        c = simulation_config()
        c["wrench"]["T_tool_sensor"] = None
        self.assertTrue(any("wrench" in e for e in validate_config(c)))

    def test_recording_replay_and_statistics(self):
        path = self.directory / "nominal.jsonl"
        result, _ = run_simulation(output=path)
        self.assertTrue(result["success"])
        report = replay(path)
        self.assertTrue(report["reproduced"], report)
        self.assertTrue(report["source_code_matches"])
        stats = evaluate([path])
        self.assertEqual(stats["groups"][0]["tasks"], 1)
        self.assertEqual(stats["groups"][0]["source"], "simulation")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate([path, path])

    def test_no_overwriting_previous_task(self):
        path = self.directory / "task.jsonl"
        with TaskLog(path, simulation_config(), "simulation"):
            pass
        with self.assertRaises(FileExistsError):
            TaskLog(path, simulation_config(), "simulation")

    def test_nan_failure_record_is_valid_json(self):
        path = self.directory / "fault.jsonl"
        with TaskLog(path, simulation_config(), "simulation") as log:
            log.write({"kind": "sample", "valid": False, "wrench": np.array([np.nan, np.inf])})
        last = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertFalse(last["valid"])
        self.assertEqual(last["wrench"], [None, None])

    def test_previous_run_observation_is_not_live(self):
        path = self.directory / "old_vision.json"
        path.write_text(json.dumps({"valid": True, "captured_at": 5.}), encoding="utf-8")
        result = read_observation(path, published_after_ns=path.stat().st_mtime_ns + 1)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "observation_predates_this_run")
        result = read_observation(path, captured_after=10.)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "capture_predates_this_run")

    def test_wilson_does_not_claim_certainty_for_one_success(self):
        low, high = wilson(1, 1)
        self.assertGreater(low, 0)
        self.assertLess(low, .3)
        self.assertAlmostEqual(high, 1.)
        self.assertIsNone(wilson(0, 0))


if __name__ == "__main__":
    unittest.main()
