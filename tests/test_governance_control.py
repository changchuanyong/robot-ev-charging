"""Exercise controller preparation and stop confirmation with software-only doubles."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ev_charging.control import config as configuration
from ev_charging.control import run
from ev_charging.control.types import Decision, RobotSample


class ControlBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def execute(self, *, confirmed=True, stale=False):
        robots, controllers = [], []

        class Robot:
            def __init__(self, config):
                self.identity = {"fixture": True}
                self.stop_command_ack = False
                self.closed = False
                self.commands = 0
                robots.append(self)

            def _validate_config(self):
                pass

            def connect(self):
                pass

            def read(self):
                return RobotSample(1, 9.0 if stale else 10.0, np.eye(4), np.zeros(6))

            def stop(self):
                self.stop_command_ack = True

            def wait_stopped(self, timeout, tolerance):
                return confirmed

            def command(self, *args):
                self.commands += 1
                raise AssertionError("fixture must never issue a motion command")

            def close(self):
                self.closed = True

        class Controller:
            def __init__(self, config, policy=None, *, probe_action=None):
                self.state = "VISION"
                self.events = []
                self.confirmed = False
                controllers.append(self)

            def step(self, sample, now, observation):
                return Decision("COMPLETE", stop=True, finished=True, success=True, reason="fixture_complete")

            def acknowledge_stop(self):
                self.confirmed = True

        config = configuration.simulation_config()
        config["hardware"] = {"stop_feedback_timeout": 0.1, "stop_tolerance": {}}
        path = self.root / "task.jsonl"
        error = None
        with patch("ev_charging.hardware.realman.RealManRobot", Robot):
            with patch.object(run, "InsertionController", Controller):
                with patch.object(run, "validate_config", return_value=[]):
                    with patch.object(run.time, "monotonic", return_value=10.0):
                        try:
                            run.run_device(config, self.root / "missing_pose.json", path)
                        except RuntimeError as exc:
                            error = exc
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return robots, controllers, records, error

    def test_confirmed_stop_preserves_success(self):
        robots, controllers, records, error = self.execute()
        self.assertIsNone(error)
        self.assertEqual(len(controllers), 1)
        self.assertTrue(robots[-1].closed)
        self.assertTrue(controllers[-1].confirmed)
        result = [row for row in records if row["kind"] == "result"][-1]
        self.assertTrue(result["success"])
        self.assertTrue(result["stop_feedback_confirmed"])
        self.assertEqual(sum(robot.commands for robot in robots), 0)

    def test_unconfirmed_stop_cannot_report_success(self):
        robots, controllers, records, error = self.execute(confirmed=False)
        self.assertIn("measured stop was not confirmed", str(error))
        self.assertTrue(robots[-1].closed)
        self.assertFalse(controllers[-1].confirmed)
        result = [row for row in records if row["kind"] == "result"][-1]
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "stop_feedback_unconfirmed")

    def test_stale_feedback_stops_before_sending(self):
        robots, _, records, error = self.execute(stale=True)
        self.assertIn("feedback became stale", str(error))
        self.assertTrue(robots[-1].stop_command_ack)
        self.assertTrue(robots[-1].closed)
        self.assertEqual(sum(robot.commands for robot in robots), 0)
        self.assertFalse([row for row in records if row["kind"] == "result"][-1]["success"])

    def test_public_preflight_rejects_unsafe_config_without_constructing_driver(self):
        with patch(
            "ev_charging.hardware.realman.RealManRobot", side_effect=AssertionError("driver constructed")
        ):
            errors, policy = run.preflight(configuration.simulation_config())
            self.assertIn("hardware refuses a simulation configuration", errors)
            self.assertIsNone(policy)
            self.assertEqual(run.preflight(None), (["configuration must be a JSON object"], None))

    def test_configuration_errors_remain_exact_and_defaults_are_built_once(self):
        cases = json.loads((Path(__file__).parent / "fixtures/control_validation.json").read_text())
        for case in cases:
            config = configuration.simulation_config()
            if case["field"]:
                target = config
                *parents, name = case["field"].split(".")
                for parent in parents:
                    target = target[parent]
                target[name] = case["value"]
            self.assertEqual(configuration.validate_config(config, hardware=case["hardware"]), case["errors"])
        config = configuration.simulation_config()
        with patch.object(
            configuration, "simulation_config", wraps=configuration.simulation_config
        ) as defaults:
            self.assertEqual(configuration.validate_config(config), [])
        defaults.assert_called_once()


if __name__ == "__main__":
    unittest.main()
