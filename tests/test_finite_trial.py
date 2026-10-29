"""Finite-trial feedback contracts use synthetic values, never robot limits."""
import unittest
import numpy as np

from control.config import simulation_config, validate_config
from control.finite_trial import FiniteTrial
from control.simulate import SimulatedRobot, run_simulation
from control.state_machine import InsertionController
from tests.test_state_machine import ControlledPolicy, drive_to, cycle


class FiniteTrialTests(unittest.TestCase):
    def fixture(self):
        config = simulation_config()
        trial = FiniteTrial(config["finite_trial"])
        for t, load in zip((0., .01, .02, .03, .04, .05), (1., 1., 10., 1., 1., 1.)):
            trial.add_baseline(t, np.zeros(6), load)
        trial.start("+x", .05, np.eye(4), np.zeros(6), config["synergy"]["action_step"],
                    config["learning"]["bias_rate"], 0.)
        return trial

    def reach(self, trial):
        for now in (.05, .061, .08, .099, .11):
            trial.update(now, .019, trial.increment.copy(), 0., np.zeros(6), .5, True)
        if trial.phase == "TRIAL":
            trial.update(.12, .01, trial.delta.copy(), 0., np.zeros(6), .5, True)
        self.assertEqual(trial.phase, "OBSERVE")

    def test_jitter_never_exceeds_step_or_resets_deadline(self):
        trial = self.fixture()
        deadline = trial.record["trial_deadline"]
        for now, dt in ((.05, .01), (.062, .012), (.089, .027), (.11, .021)):
            trial.update(now, dt, np.zeros(6), 0., np.zeros(6), 1., True)
            self.assertLessEqual(trial.increment[0], trial.delta[0])
            self.assertEqual(trial.record["trial_deadline"], deadline)
        self.assertEqual(trial.phase, "TRIAL")  # command alone is insufficient
        self.assertEqual(trial.record["baseline_load"], 1.)
        with self.assertRaisesRegex(ValueError, "active"):
            trial.start("-x", .12, np.eye(4), np.zeros(6), np.ones(6), np.ones(6), 0.)

    def test_late_endpoint_cannot_win_over_fixed_deadline(self):
        trial = self.fixture()
        trial.increment = trial.delta.copy()
        reason = trial.update(.36, .01, trial.delta, 0., np.zeros(6), .5, True)
        self.assertEqual(reason, "finite_trial_timeout")
        self.assertEqual(trial.last_result["outcome"], "unconfirmed")

    def test_observation_freezes_until_fixed_window_and_needs_actual_advance(self):
        trial = self.fixture()
        self.reach(trial)
        deadline = trial.record["observation_deadline"]
        trial.update(deadline - .001, .01, trial.delta, 0., np.zeros(6), .4, True)
        self.assertTrue(trial.freezes_advance)
        trial.update(deadline, .001, trial.delta, 0., np.zeros(6), .4, True)
        self.assertEqual(trial.phase, "CONFIRM")
        self.assertTrue(trial.record["release_clue"])
        self.assertIsNone(trial.last_result)
        trial.update(deadline + .05, .01, trial.delta, .00011, np.zeros(6), .4, True)
        self.assertEqual(trial.last_result["outcome"], "effective")

    def test_load_drop_without_advance_is_not_effective(self):
        trial = self.fixture()
        self.reach(trial)
        end = trial.record["observation_deadline"]
        trial.update(end, .01, trial.delta, 0., np.zeros(6), .1, True)
        trial.update(trial.record["confirmation_deadline"], .01, trial.delta, 0., np.zeros(6), .1, True)
        self.assertEqual(trial.last_result["outcome"], "no_progress")

    def test_late_advance_cannot_be_effective(self):
        trial = self.fixture()
        self.reach(trial)
        trial.update(trial.record["observation_deadline"], .01, trial.delta, 0., np.zeros(6), .5, True)
        trial.update(trial.record["confirmation_deadline"] + .001, .01, trial.delta,
                     .001, np.zeros(6), .5, True)
        self.assertEqual(trial.last_result["outcome"], "no_progress")

    def test_observation_endpoint_drift_and_load_violation_stop(self):
        trial = self.fixture()
        self.reach(trial)
        reason = trial.update(.13, .01, np.ones(6), 0., np.zeros(6), .5, True)
        self.assertEqual(reason, "trial_endpoint_lost")
        trial = self.fixture()
        reason = trial.update(.06, .01, np.zeros(6), 0., np.ones(6), 1., False)
        self.assertEqual(reason, "trial_load_admission_lost")
        self.assertFalse(trial.last_result["loads_allowed"])

    def test_legacy_config_rejected(self):
        config = simulation_config()
        config["schema_version"] = 1
        self.assertTrue(any("legacy" in e for e in validate_config(config)))

    def test_formal_vision_requires_frozen_measured_geometry_identity(self):
        for source, expected_state in ((None, "VISION"),
                ({"version": "print1", "source_sha256": "a" * 64, "status": "measured_print", "validated": True}, "PREALIGN"),
                ({"version": "print1", "source_sha256": "b" * 64, "status": "measured_print", "validated": True}, "VISION"),
                ({"version": "print1", "source_sha256": "a" * 64, "status": "nominal_reference", "validated": True}, "VISION")):
            config = simulation_config()
            config["experiment"] = {"formal": True, "geometry_reference": {"version": "print1", "sha256": "a" * 64}}
            robot = SimulatedRobot(config)
            controller = InsertionController(config)
            sample = robot.read(0.)
            obs = robot.observation(sample)
            obs["quality"]["geometry_reference"] = source
            obs["T_camera_target"][0][3] = .001
            decision = controller.step(sample, 0., obs)
            self.assertEqual(decision.state, expected_state)
            if expected_state == "VISION":
                self.assertEqual(decision.reason, "formal_print_geometry_mismatch")


class FiniteControllerTests(unittest.TestCase):
    def fixture(self, method="C", scenario="nominal"):
        config = simulation_config()
        config["method"] = method
        config["learning"].update(confidence=.8, consistent_windows=1, min_hold_s=0.)
        policy = ControlledPolicy(config["learning"], [1, 0, 0, 0, 0, 0, 0, 0, 0])
        return (*drive_to(config=config, policy=policy, scenario=scenario), config)

    def test_trial_and_observation_freeze_axial_command_and_deadline(self):
        controller, robot, now, config = self.fixture()
        z, deadline = None, None
        phases = set()
        for k in range(70):
            decision = cycle(controller, robot, now + k * .01)
            self.assertFalse(decision.stop, decision.reason)
            snap = decision.details.get("finite_trial", {})
            phase = snap.get("phase")
            phases.add(phase)
            if phase in ("TRIAL", "OBSERVE"):
                if z is None:
                    z = snap["start_pose"][2][3]
                    deadline = snap["trial_deadline"]
                self.assertAlmostEqual(decision.command[2, 3], z)
                self.assertFalse(decision.details["advance_requested"])
                self.assertEqual(snap["trial_deadline"], deadline)
            if controller.trial.last_result:
                self.assertEqual(controller.trial.last_result["outcome"], "effective")
                break
        self.assertTrue({"TRIAL", "OBSERVE", "CONFIRM"}.issubset(phases))

    def test_candidate_checks_locked_endpoint_before_first_command(self):
        controller, robot, now, config = self.fixture("F")
        robot.pose[0, 3] = .000012
        controller.last_command = robot.pose.copy()
        # A pending admittance/nominal reference is different from the actual
        # locked trial origin. Admission must not use the more generous one.
        controller.nominal[0, 3] = 0.
        controller.vision_budget[0] = .000015
        controller.trial.add_baseline(now - .06, np.zeros(6), .1)
        controller.trial.add_baseline(now - .01, np.zeros(6), .1)
        decision = cycle(controller, robot, now, [0, 0, -.25, 0, 0, 0])
        self.assertNotIn("+x", decision.details["allowed_actions"])
        self.assertEqual(decision.details["action_rejections"]["+x"], "visual_correction_budget")
        self.assertFalse(controller.trial.active)

    def test_trial_command_respects_rate_even_when_global_velocity_is_larger(self):
        controller, robot, now, config = self.fixture()
        previous = robot.pose.copy()
        for k in range(25):
            decision = cycle(controller, robot, now + k * .01)
            if decision.details.get("finite_trial", {}).get("phase") == "TRIAL":
                self.assertLessEqual(abs(decision.command[0, 3] - previous[0, 3]),
                                     config["learning"]["bias_rate"][0] * .01 + 1e-12)
            previous = robot.pose.copy()

    def test_failed_actual_trial_stops_without_reverse(self):
        controller, robot, now, _ = self.fixture(scenario="trial_timeout")
        for k in range(70):
            decision = cycle(controller, robot, now + k * .01)
            if decision.stop:
                break
        self.assertEqual(decision.reason, "finite_trial_timeout")
        self.assertEqual(decision.details["trial_result"]["outcome"], "unconfirmed")
        self.assertIsNone(decision.command)

    def test_hard_fault_retains_invalid_trial_result(self):
        controller, robot, now, _ = self.fixture()
        for k in range(12):
            decision = cycle(controller, robot, now + k * .01)
            if controller.trial.active:
                break
        decision = cycle(controller, robot, now + (k + 1) * .01, [20., 0, 0, 0, 0, 0])
        self.assertEqual(decision.reason, "hard_wrench_limit")
        self.assertEqual(decision.details["trial_result"]["outcome"], "invalid")
        self.assertEqual(decision.details["trial_result"]["trial_peak"][0], 20.)

    def test_trial_peak_retains_unfiltered_impulse_and_selection_audit(self):
        controller, robot, now, _ = self.fixture()
        for k in range(12):
            decision = cycle(controller, robot, now + k * .01)
            if controller.trial.active:
                break
        self.assertEqual(decision.details["probabilities"], [1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(decision.details["selection_confidence"], 1.)
        self.assertEqual(decision.details["finite_trial"]["selection_prediction_at"], now + k * .01)
        decision = cycle(controller, robot, now + (k + 1) * .01, [1., 0, -.25, 0, 0, 0])
        self.assertFalse(decision.stop, decision.reason)
        self.assertLess(decision.details["filtered_wrench"][0], 1.)
        self.assertEqual(decision.details["finite_trial"]["trial_peak"][0], 1.)
        self.assertEqual(decision.details["finite_trial"]["peak_source"], "compensated_unfiltered_task_wrench")

    def test_c_and_e_share_control_rules(self):
        records = []
        for method in ("C", "E"):
            controller, robot, now, _ = self.fixture(method)
            records.append([cycle(controller, robot, now + k * .01).to_dict() for k in range(30)])
        self.assertEqual(records[0], records[1])

    def test_jitter_and_lag_are_executable_software_fault_fixtures(self):
        for scenario in ("period_jitter", "feedback_lag"):
            result, decisions = run_simulation(scenario=scenario)
            self.assertTrue(result["stop_confirmed"])
            self.assertTrue(decisions[-1].finished)
            if scenario == "period_jitter":
                self.assertGreater(len({round(d.details["dt"], 6) for d in decisions if "dt" in d.details}), 1)

    def test_observation_and_confirmation_fault_scenarios_reach_expected_branches(self):
        for scenario, expected in (("observation_drift", "invalid"), ("confirmation_blocked", "no_progress")):
            config = simulation_config()
            config["method"] = "C"
            config["finite_trial"]["max_trials"] = 2
            config["learning"].update(confidence=.8, consistent_windows=1, min_hold_s=0.)
            policy = ControlledPolicy(config["learning"], [1, 0, 0, 0, 0, 0, 0, 0, 0])
            result, decisions = run_simulation(config, scenario, policy=policy)
            self.assertFalse(result["success"])
            outcomes = [d.details["trial_result"]["outcome"] for d in decisions if "trial_result" in d.details]
            self.assertIn(expected, outcomes)


if __name__ == "__main__":
    unittest.main()
