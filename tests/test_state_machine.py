"""Feedback and failure-injection tests; these do not command any hardware."""
import unittest

import numpy as np

from control.config import simulation_config
from control.geometry import exp_so3
from control.learning import ForcePolicy
from control.simulate import SimulatedRobot, run_simulation
from control.state_machine import InsertionController


class ControlledPolicy(ForcePolicy):
    """Deterministic classification fixture; production acceptance logic stays real."""
    def __init__(self, config, probabilities=None):
        super().__init__(config)
        self.model_ready = True
        self.fixture_probabilities = probabilities

    def update(self, wrench, timestamp, context_id):
        if self.fixture_probabilities is None:
            self.prediction_time = None
            self.last_probabilities = None
            return None
        self.prediction_time = timestamp
        self.last_probabilities = np.asarray(self.fixture_probabilities, float)
        return self.last_probabilities.copy()


def drive_to(wanted="INSERT", config=None, policy=None, scenario="nominal"):
    config = config or simulation_config()
    controller = InsertionController(config, policy)
    robot = SimulatedRobot(config, scenario)
    for k in range(3000):
        now = k * config["period"]
        sample = robot.read(now)
        observation = robot.observation(sample) if controller.state == "VISION" else None
        decision = controller.step(sample, now, observation)
        if decision.stop:
            robot.stop()
            controller.acknowledge_stop()
        elif decision.command is not None:
            robot.command(decision.command, config["period"])
        if controller.state == wanted:
            return controller, robot, now + config["period"]
        if decision.finished:
            raise AssertionError(f"fixture stopped before {wanted}: {decision.reason}")
    raise AssertionError(f"fixture never entered {wanted}")


def cycle(controller, robot, now, wrench=None, follow=True):
    sample = robot.read(now)
    if wrench is not None:
        sample.wrench = np.asarray(wrench, float)
    decision = controller.step(sample, now)
    if follow and decision.command is not None:
        robot.command(decision.command, controller.config["period"])
    return decision


class StateMachineTests(unittest.TestCase):
    def test_completion_hold_does_not_double_existing_correction_on_resume(self):
        controller, robot, now = drive_to()
        task = controller.config["task"]
        robot.pose[0,3] = .0005
        robot.pose[2,3] = controller.contact_pose[2,3] + task["depth"]
        controller.last_command = robot.pose.copy()
        controller.last_pose = robot.pose.copy()
        controller.nominal = robot.pose.copy()
        controller.bias[0] = .0003
        controller.admittance.displacement[0] = .0002
        first = cycle(controller, robot, now, [0,0,-.25,0,0,0])
        self.assertTrue(first.details["completion_hold"])
        np.testing.assert_allclose(controller.bias, 0)
        np.testing.assert_allclose(controller.admittance.displacement, 0)
        robot.pose[2,3] = controller.contact_pose[2,3] + task["depth"] - task["depth_tolerance"] - .00001
        resumed = cycle(controller, robot, now + .01, [0,0,-.25,0,0,0])
        self.assertFalse(resumed.finished)
        self.assertAlmostEqual(resumed.command[0,3], .0005)

    def test_nominal_completion_has_feedback_hold_and_explicit_stop(self):
        result, decisions = run_simulation()
        self.assertTrue(result["success"])
        self.assertTrue(result["stop_confirmed"])
        self.assertEqual(result["states"], ["PREALIGN", "APPROACH", "CONTACT", "INSERT", "COMPLETE"])
        self.assertTrue(decisions[-1].stop)
        hold = [d for d in decisions if d.details.get("completion_hold")]
        self.assertGreater(len(hold), 1)
        self.assertTrue(all(not d.details["advance_requested"] for d in hold))

    def test_blocked_actual_feedback_cannot_be_replaced_by_commanded_depth(self):
        result, decisions = run_simulation(scenario="blocked")
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "progress_stall")
        self.assertTrue(result["stop_confirmed"])
        self.assertLess(decisions[-1].details["depth"], simulation_config()["task"]["depth"])

    def test_command_only_does_not_complete_prealignment(self):
        config = simulation_config()
        controller, robot = InsertionController(config), SimulatedRobot(config)
        sample = robot.read(0.)
        observation = robot.observation(sample)
        observation["T_camera_target"][0][3] += .002
        decision = controller.step(sample, 0., observation)
        self.assertEqual(decision.state, "PREALIGN")
        # The fixture deliberately ignores every command: time alone cannot
        # turn the requested prealignment into measured arrival.
        for k in range(1, 80):
            decision = controller.step(robot.read(k * .01), k * .01)
            self.assertFalse(decision.success)
            self.assertEqual(decision.state, "PREALIGN")

    def test_approach_uses_the_verified_task_axis_not_the_pnp_normal(self):
        config = simulation_config()
        target_task = exp_so3([0, np.pi / 2, 0])
        config["calibration"]["R_target_task"] = target_task.tolist()
        controller, robot = InsertionController(config), SimulatedRobot(config)
        robot.pose[:3, 3] = [-config["task"]["pre_distance"], 0, 0]
        sample = robot.read(0.)
        sample.wrench[:] = 0
        first = controller.step(sample, 0., robot.observation(sample))
        self.assertEqual(first.state, "APPROACH")
        np.testing.assert_allclose(controller.R_base_task, target_task, atol=1e-12)
        robot.command(first.command, config["period"])
        previous = robot.pose.copy()
        second = cycle(controller, robot, .01, [0] * 6)
        displacement = second.command[:3, 3] - previous[:3, 3]
        self.assertGreater(displacement[0], 0)
        np.testing.assert_allclose(displacement[1:], [0, 0], atol=1e-12)

    def test_stale_repeated_and_reordered_feedback_stops(self):
        for case in ("stale", "duplicate_seq", "duplicate_time", "future", "nan_seq"):
            with self.subTest(case=case):
                config = simulation_config()
                controller, robot = InsertionController(config), SimulatedRobot(config)
                first = robot.read(0.)
                controller.step(first, 0., robot.observation(first))
                sample = robot.read(.01)
                if case == "stale":
                    sample.timestamp = -.1
                elif case == "duplicate_seq":
                    sample.seq = first.seq
                elif case == "duplicate_time":
                    sample.timestamp = first.timestamp
                elif case == "future":
                    sample.timestamp = .1
                else:
                    sample.seq = np.nan
                result = controller.step(sample, .01)
                self.assertTrue(result.stop)
                self.assertFalse(result.success)
                self.assertIsNone(result.command)

    def test_nonfinite_pose_load_and_joints_stop_before_command(self):
        for field in ("pose", "wrench", "joints"):
            with self.subTest(field=field):
                config = simulation_config()
                controller, robot = InsertionController(config), SimulatedRobot(config)
                sample = robot.read(0.)
                observation = robot.observation(sample)
                if field == "pose":
                    sample.pose[0, 3] = np.nan
                elif field == "wrench":
                    sample.wrench[0] = np.inf
                else:
                    sample.joints = tuple([np.nan] * 7)
                result = controller.step(sample, 0., observation)
                self.assertTrue(result.stop)
                self.assertIsNone(result.command)

    def test_hard_limit_and_invalid_risk_are_distinct_stops(self):
        for load, reason in ((-12., "hard_wrench_limit"), (-7., "risk_domain_invalid")):
            with self.subTest(reason=reason):
                controller, robot, now = drive_to()
                decision = cycle(controller, robot, now, [0, 0, load, 0, 0, 0])
                self.assertEqual(decision.reason, reason)
                self.assertTrue(decision.stop)
                self.assertIsNone(decision.command)

    def test_risk_pause_is_not_a_stall_event(self):
        config = simulation_config()
        config["task"].update(stall_window=.08, max_pause=.5)
        controller, robot, now = drive_to(config=config)
        for k in range(80):
            decision = cycle(controller, robot, now + k * .01, [0, 0, -5.4, 0, 0, 0])
            self.assertNotEqual(decision.reason, "progress_stall")
            self.assertEqual(decision.details.get("advance_scale"), 0.)
            if decision.finished:
                break
        self.assertEqual(decision.reason, "pause_timeout")

    def test_target_identity_and_placeholder_intrinsics_are_rejected(self):
        for field in ("target_id", "camera_calibrated", "calibration_version"):
            with self.subTest(field=field):
                config = simulation_config()
                controller, robot = InsertionController(config), SimulatedRobot(config)
                sample = robot.read(0.)
                observation = robot.observation(sample)
                if field == "camera_calibrated":
                    observation["quality"][field] = False
                else:
                    observation[field] = "wrong-identity"
                result = controller.step(sample, 0., observation)
                self.assertEqual(result.state, "VISION")
                self.assertIsNone(result.command)
                self.assertIsNone(controller.target)

    def test_contact_handover_requires_contact_to_remain_present(self):
        controller, robot, now = drive_to("CONTACT")
        for k in range(15):
            decision = cycle(controller, robot, now + k * .01, [0] * 6)
            self.assertNotEqual(decision.state, "INSERT")
            if decision.finished:
                break
        self.assertEqual(decision.reason, "contact_lost_during_handover")

    def test_contact_confirmation_restarts_after_a_brief_release(self):
        controller, robot, now = drive_to("CONTACT")
        for k in range(4):
            self.assertEqual(cycle(controller, robot, now + k * .01, [0] * 6).state, "CONTACT")
        for k in range(4, 10):
            self.assertEqual(cycle(controller, robot, now + k * .01, [0, 0, -.3, 0, 0, 0]).state, "CONTACT")

    def test_continuing_actual_motion_does_not_count_as_stable_completion(self):
        config = simulation_config()
        config["limits"]["tracking_error"][2] = .02
        controller, robot, now = drive_to(config=config)
        origin = controller.contact_pose[2, 3] + config["task"]["depth"]
        for k in range(25):
            robot.pose[2, 3] = origin + k * .00001
            decision = cycle(controller, robot, now + k * .01, follow=False)
            self.assertFalse(decision.success)
            self.assertEqual(decision.state, "INSERT")

    def test_depth_drift_restarts_completion_hold_even_below_velocity_limit(self):
        config = simulation_config()
        config["limits"]["tracking_error"][2] = .02
        config["task"]["complete_velocity"][2] = .02
        config["task"]["complete_depth_range"] = .00005
        controller, robot, now = drive_to(config=config)
        origin = controller.contact_pose[2, 3] + config["task"]["depth"]
        for k in range(30):
            robot.pose[2, 3] = origin + k * .00001
            decision = cycle(controller, robot, now + k * .01, follow=False)
            self.assertFalse(decision.success)

    def test_learning_hold_timeout_stops_nominal_advance(self):
        config = simulation_config()
        config["method"] = "C"
        config["learning"]["max_hold_s"] = .05
        policy = ControlledPolicy(config["learning"])
        controller, robot, now = drive_to(config=config, policy=policy)
        for k in range(12):
            decision = cycle(controller, robot, now + k * .01)
            if decision.finished:
                break
        self.assertEqual(decision.reason, "learning_hold_timeout")
        self.assertTrue(decision.stop)
        self.assertIsNone(decision.command)

    def test_limited_commands_do_not_accumulate_hidden_admittance(self):
        config = simulation_config()
        config["limits"]["velocity"][0] = .00001
        controller, robot, now = drive_to(config=config)
        clipped = 0
        for k in range(100):
            decision = cycle(controller, robot, now + k * .01, [1, 0, -.25, 0, 0, 0])
            self.assertFalse(decision.finished)
            if decision.details["limited"]:
                clipped += 1
                np.testing.assert_allclose(controller.admittance.displacement, 0.)
                np.testing.assert_allclose(controller.admittance.velocity, 0.)
        self.assertGreater(clipped, 90)
        previous = robot.pose.copy()
        decision = cycle(controller, robot, now + 1., [0, 0, -.25, 0, 0, 0])
        self.assertLessEqual(abs(decision.command[0, 3] - previous[0, 3]), .00001 * .01 + 1e-12)

    def make_f_controller(self, mutator=None):
        config = simulation_config()
        config["method"] = "F"
        config["learning"].update(confidence=.3, consistent_windows=1, min_hold_s=0.)
        if mutator:
            mutator(config)
        probabilities = [.6, .35, 0, 0, 0, 0, 0, 0, .05]
        policy = ControlledPolicy(config["learning"], probabilities)
        return (*drive_to(config=config, policy=policy), config)

    def test_f_masks_highest_probability_and_uses_full_calibrated_quantum(self):
        def reject_positive(config):
            config["synergy"]["force_response"]["delta_wrench"][2][0] = -.1
        controller, robot, now, config = self.make_f_controller(reject_positive)
        decision = cycle(controller, robot, now, [0, 0, -.25, 0, 0, 0])
        self.assertEqual(decision.details["action"], "-x")
        self.assertNotIn("+x", decision.details["allowed_actions"])
        self.assertAlmostEqual(controller.bias[0], -config["synergy"]["action_step"][0])
        second = cycle(controller, robot, now + .01, [0, 0, -.25, 0, 0, 0])
        self.assertEqual(second.details["action"], "Hold")
        self.assertIn("action_quantum_rate_limit", second.details["action_rejections"].values())

    def test_f_cannot_send_a_truncated_action_quantum(self):
        def restrictive(config):
            config["limits"]["velocity"][0] = .0001
        controller, robot, now, _ = self.make_f_controller(restrictive)
        decision = cycle(controller, robot, now, [0, 0, -.25, 0, 0, 0])
        self.assertEqual(decision.details["action"], "Hold")
        self.assertEqual(decision.details["action_rejections"]["+x"], "action_quantum_would_be_truncated")
        self.assertAlmostEqual(controller.bias[0], 0.)
        self.assertAlmostEqual(decision.command[0, 3], 0.)

    def test_f_checks_admittance_in_the_total_command_budget(self):
        controller, robot, now, _ = self.make_f_controller()
        controller.vision_budget[0] = .000001
        decision = cycle(controller, robot, now, [1, 0, -.25, 0, 0, 0])
        self.assertTrue(decision.stop)
        self.assertEqual(decision.reason, "synergy_constraints_invalid")
        self.assertIsNone(decision.command)

    def test_retreat_requires_stop_acknowledgement_and_fresh_low_load(self):
        config = simulation_config()
        config["task"]["retreat_enabled"] = True
        controller, robot, now = drive_to("STOPPED", config=config, scenario="blocked")
        controller.stop_confirmed = False
        self.assertFalse(controller.request_retreat(robot.read(now), now))
        controller.acknowledge_stop()
        sample = robot.read(now + .01)
        sample.wrench[2] = -30.
        self.assertFalse(controller.request_retreat(sample, now + .01))
        sample = robot.read(now + .02)
        sample.timestamp = np.nan
        self.assertFalse(controller.request_retreat(sample, now + .02))
        self.assertEqual(controller.state, "STOPPED")

    def test_retreat_needs_a_continuously_stationary_feedback_window(self):
        config = simulation_config()
        config["task"]["retreat_enabled"] = True
        controller, robot, now = drive_to("STOPPED", config=config, scenario="blocked")
        sample = robot.read(now)
        self.assertFalse(controller.request_retreat(sample, now))
        self.assertFalse(controller.request_retreat(sample, now + .01))  # duplicate cannot confirm
        accepted = False
        for k in range(2, 17):
            accepted = controller.request_retreat(robot.read(now + k * .01), now + k * .01)
            if accepted:
                break
        self.assertTrue(accepted)
        self.assertEqual(controller.state, "RETREAT")
        self.assertGreaterEqual(k * .01, config["task"]["retreat_stationary_duration"])

    def test_retreat_rejects_unknown_fault_and_motion(self):
        config = simulation_config()
        config["task"]["retreat_enabled"] = True
        controller, robot, now = drive_to("STOPPED", config=config, scenario="blocked")
        controller.reason = "hard_wrench_limit"
        self.assertFalse(controller.request_retreat(robot.read(now), now))
        controller.reason = "progress_stall"
        for k in range(15):
            robot.pose[0, 3] += .00001
            self.assertFalse(controller.request_retreat(robot.read(now + k * .01), now + k * .01))
        self.assertEqual(controller.state, "STOPPED")


if __name__ == "__main__":
    unittest.main()
