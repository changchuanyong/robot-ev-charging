"""Analytic tests use synthetic units/limits, never hardware acceptance values."""
import unittest

import numpy as np

from control.admittance import Admittance, advance_scale
from control.geometry import (
    exp_so3, invert, log_so3, offset_pose, pose_error, pose_from_xyz_rpy,
    transform_wrench, validate_transform, xyz_rpy_from_pose,
)
from control.risk import EngineeringRisk
from control.synergy import ACTIONS, SynergyFilter, action_vector


def box_domain(coupled=False):
    matrix = np.vstack([np.eye(6), -np.eye(6)])
    bound = np.ones(12)
    if coupled:
        row = np.array([1, 0, 0, 1, 0, 0])
        matrix = np.vstack([matrix, row, -row])
        bound = np.r_[bound, 1.2, 1.2]
    return dict(version="synthetic-unit-test-domain", scale=np.ones(6),
                A=matrix, b=bound, reference=np.zeros(6), calibrated=True)


def admittance_config():
    return dict(mass=[1] * 6, stiffness=[25] * 6, damping_min=[5] * 6,
                damping_max=[10] * 6, damping_rate=[100] * 6, dt_margin=0.8)


def synergy_config():
    step = [0.0001, 0.0001, 0, 0.02, 0.02, 0]
    return dict(
        rules_version="finite-trial-v2", admission_max_risk=.9,
        validated=True, max_offset=[0.1] * 6, action_step=step,
        pins=[dict(point_m=[0, 0, 0], clearance_m=0.1, engagement_m=0.05)],
    )


class GeometryTests(unittest.TestCase):
    def test_transform_inverse_and_rpy_roundtrip(self):
        for angles in ([0.3, -0.6, 1.2], [0.1, np.pi / 2, 0.8],
                       [-0.1, -np.pi / 2, 0.8]):
            pose = pose_from_xyz_rpy([0.1, -0.2, 0.3, *angles])
            np.testing.assert_allclose(invert(pose) @ pose, np.eye(4), atol=1e-12)
            np.testing.assert_allclose(pose_from_xyz_rpy(xyz_rpy_from_pose(pose)), pose, atol=1e-8)

    def test_so3_at_zero_small_angle_and_pi(self):
        axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14)
        for angle in (0.0, 1e-10, 0.2, np.pi - 1e-7, np.pi):
            rotation = exp_so3(angle * axis)
            np.testing.assert_allclose(exp_so3(log_so3(rotation)), rotation, atol=3e-8)

    def test_transform_rejects_reflection_and_nonhomogeneous_input(self):
        bad = np.eye(4)
        bad[0, 0] = -1
        with self.assertRaises(ValueError):
            validate_transform(bad)
        bad = np.eye(4)
        bad[3, 0] = 0.01
        with self.assertRaises(ValueError):
            invert(bad)

    def test_wrench_translation_must_change_moment(self):
        transform = np.eye(4)
        transform[:3, 3] = [0.1, 0, 0]
        original = [0, 10, 0, 0, 0, 0]
        shifted = transform_wrench(original, transform)
        np.testing.assert_allclose(shifted, [0, 10, 0, 0, 0, 1])
        np.testing.assert_allclose(transform_wrench(shifted, invert(transform)), original)

    def test_wrench_rotation_and_translation(self):
        transform = pose_from_xyz_rpy([0.1, 0, 0, 0, 0, np.pi / 2])
        shifted = transform_wrench([10, 0, 0, 2, 0, 0], transform)
        np.testing.assert_allclose(shifted, [0, 10, 0, 0, 2, 1], atol=1e-12)

    def test_task_offsets_are_rotated_into_base(self):
        base_tool = pose_from_xyz_rpy([0.3, 0.2, 0.1, 0, 0, 0.5])
        base_task = exp_so3([0, 0, np.pi / 2])
        delta = [0.01, 0, 0, 0.02, 0, 0]
        moved = offset_pose(base_tool, delta, base_task)
        np.testing.assert_allclose(pose_error(moved, base_tool), [0, 0.01, 0, 0, 0.02, 0], atol=1e-12)


class RiskTests(unittest.TestCase):
    def test_independent_box_reduces_to_absolute_normalized_load(self):
        config = box_domain()
        config["scale"] = [2, 3, 4, 0.1, 0.2, 0.3]
        domain = EngineeringRisk(config)
        normalized = np.array([-0.3, 0.2, -0.9, 0, 0.5, 1.0])
        result = domain.evaluate(normalized * domain.scale)
        self.assertTrue(result.valid)
        self.assertFalse(result.outside)
        np.testing.assert_allclose(result.values, np.abs(normalized))

    def test_coupling_reduces_two_directional_margins(self):
        result = EngineeringRisk(box_domain(coupled=True)).evaluate([0.5, 0, 0, 0.4, 0, 0])
        np.testing.assert_allclose(result.values, [0.7, 0, 0, 0.7, 0, 0])

    def test_asymmetric_reference_uses_each_signed_reference_margin(self):
        config = box_domain()
        config["reference"] = [0.2, 0, 0, 0, 0, 0]
        domain = EngineeringRisk(config)
        np.testing.assert_allclose(domain.evaluate(config["reference"]).values, np.zeros(6))
        self.assertAlmostEqual(domain.evaluate([0.6, 0, 0, 0, 0, 0]).values[0], 0.5)
        self.assertAlmostEqual(domain.evaluate([-0.4, 0, 0, 0, 0, 0]).values[0], 0.5)

    def test_domain_and_measurement_failure_are_explicit(self):
        domain = EngineeringRisk(box_domain())
        outside = domain.evaluate([1.01, 0, 0, 0, 0, 0])
        self.assertFalse(outside.valid)
        self.assertTrue(outside.outside)
        np.testing.assert_array_equal(outside.values, np.ones(6))
        invalid = domain.evaluate([np.nan] * 6)
        self.assertFalse(invalid.valid)
        self.assertFalse(invalid.outside)
        self.assertEqual(invalid.reason, "invalid_wrench")
        for mutation in ("scale", "reference", "bounds"):
            config = box_domain()
            if mutation == "scale":
                config["scale"][0] = 0
            elif mutation == "reference":
                config["reference"][0] = 1
            else:
                config["A"][6, 0] = 1
            with self.assertRaises(ValueError):
                EngineeringRisk(config)


class AdmittanceTests(unittest.TestCase):
    def test_constant_force_converges_to_stiffness_equilibrium(self):
        model = Admittance(admittance_config())
        for _ in range(4000):
            model.step([2, 0, 0, 0, 0, 0], 0.005)
        np.testing.assert_allclose(model.displacement, [2 / 25, 0, 0, 0, 0, 0], atol=1e-8)
        np.testing.assert_allclose(model.velocity, np.zeros(6), atol=1e-8)

    def test_unforced_model_dissipates_and_recovers(self):
        model = Admittance(admittance_config())
        model.sync([0.01] * 6, [0] * 6)
        energy = lambda: float(np.sum(model.mass * model.velocity**2 + model.stiffness * model.displacement**2) / 2)
        old = energy()
        for _ in range(1000):
            model.step(np.zeros(6), 0.005, risk=[0.7] * 6)
            self.assertLessEqual(energy(), old + 1e-12)
            old = energy()
        self.assertLess(old, 1e-10)

    def test_risk_decreases_damping_with_rate_limit(self):
        model = Admittance(admittance_config())
        model.step(np.zeros(6), 0.005, risk=np.ones(6))
        np.testing.assert_allclose(model.damping, [9.5] * 6)
        for _ in range(20):
            model.step(np.zeros(6), 0.005, risk=np.ones(6))
        np.testing.assert_allclose(model.damping, [5] * 6)

    def test_invalid_interval_does_not_advance_state(self):
        model = Admittance(admittance_config())
        self.assertAlmostEqual(model.max_dt, 0.128)
        for dt in (0, -0.01, np.nan, model.max_dt * 1.01):
            with self.assertRaises(ValueError):
                model.step(np.ones(6), dt)
        np.testing.assert_array_equal(model.displacement, np.zeros(6))
        with self.assertRaises(ValueError):
            model.step(np.ones(6), 0.005, risk=[1.01] * 6)

    def test_synchronization_removes_hidden_accumulation(self):
        model = Admittance(admittance_config())
        for _ in range(100):
            model.step([20, 0, 0, 0, 0, 0], 0.005)
        self.assertGreater(model.displacement[0], 0.1)
        model.sync([0.001, 0, 0, 0, 0, 0], np.zeros(6))
        result = model.step(np.zeros(6), 0.005)
        self.assertLess(result[0], 0.001)
        self.assertGreater(result[0], 0)

    def test_progress_gate_has_smooth_endpoints(self):
        self.assertEqual(advance_scale(0.1, 0.2, 0.8), 1)
        self.assertAlmostEqual(advance_scale(0.5, 0.2, 0.8), 0.5)
        self.assertEqual(advance_scale(0.9, 0.2, 0.8), 0)
        with self.assertRaises(ValueError):
            advance_scale(np.nan, 0.2, 0.8)


class SynergyTests(unittest.TestCase):
    def setUp(self):
        self.domain = EngineeringRisk(box_domain())
        self.offset = np.zeros(6)
        self.budget = np.ones(4) * 0.1
        self.wrench = np.zeros(6)

    def evaluate(self, model, **kwargs):
        return model.allowed(kwargs.get("offset", self.offset), kwargs.get("budget", self.budget),
                             kwargs.get("risk", self.domain.evaluate(self.wrench).values),
                             kwargs.get("wrench", self.wrench))

    def test_actions_have_no_axial_or_key_rotation_command(self):
        for action in ACTIONS:
            vector = action_vector(action)
            self.assertEqual(vector[2], 0)
            self.assertEqual(vector[5], 0)
        np.testing.assert_array_equal(action_vector("-Rx"), [0, 0, 0, -1, 0, 0])

    def test_unvalidated_geometry_returns_hold_and_invalid(self):
        config = synergy_config()
        config["validated"] = False
        result = self.evaluate(SynergyFilter(config, self.domain))
        self.assertFalse(result.valid)
        self.assertEqual(result.allowed, ("Hold",))

    def test_geometry_checks_pin_tip_beyond_feature_plane(self):
        config = synergy_config()
        config["pins"][0]["clearance_m"] = 0.0005
        result = self.evaluate(SynergyFilter(config, self.domain))
        self.assertTrue(result.valid)
        self.assertIn("+x", result.allowed)
        self.assertNotIn("+Rx", result.allowed)
        self.assertNotIn("-Ry", result.allowed)
        self.assertEqual(result.reasons["+Rx"], "pin_clearance_or_key_geometry")

    def test_vision_envelope_removes_direction_that_exceeds_budget(self):
        budget = np.array([0.001, 0.1, 0.1, 0.1])
        result = self.evaluate(SynergyFilter(synergy_config(), self.domain),
                               offset=[0.001, 0, 0, 0, 0, 0], budget=budget)
        self.assertNotIn("+x", result.allowed)
        self.assertIn("-x", result.allowed)

    def test_current_risk_admits_without_predicting_directional_load(self):
        config = synergy_config()
        wrench = [0.5, 0, 0, 0, 0, 0]
        result = self.evaluate(SynergyFilter(config, self.domain), wrench=wrench,
                               risk=self.domain.evaluate(wrench).values)
        self.assertIn("+x", result.allowed)
        self.assertIn("-x", result.allowed)
        high = np.array([.95, 0, 0, 0, 0, 0])
        result = self.evaluate(SynergyFilter(config, self.domain), wrench=high,
                               risk=self.domain.evaluate(high).values)
        self.assertEqual(result.allowed, ("Hold",))
        self.assertEqual(result.reasons["+x"], "current_risk_admission")

    def test_legacy_response_and_nonfinite_input_fail_closed(self):
        config = synergy_config()
        config["force_response"] = {}
        with self.assertRaisesRegex(ValueError, "legacy"):
            SynergyFilter(config, self.domain)
        model = SynergyFilter(synergy_config(), self.domain)
        self.assertFalse(self.evaluate(model, budget=[np.nan] * 6).valid)
        self.assertFalse(self.evaluate(model, risk=[0.2] * 6).valid)


if __name__ == "__main__":
    unittest.main()
