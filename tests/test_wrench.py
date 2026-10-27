"""Synthetic force-processing tests with explicit compensation semantics."""
import unittest

import numpy as np

from control.geometry import exp_so3, pose_from_xyz_rpy
from control.types import RobotSample
from control.wrench import WrenchProcessor


def processor_config():
    return dict(T_tool_sensor=np.eye(4).tolist(), external_compensated=True,
                bias=[0] * 6, filter_tau=0.1, max_gap=0.1,
                version="synthetic-wrench-v1", bias_frozen=True)


def sensor_sample(wrench, pose=None, compensated=True):
    return RobotSample(
        seq=1, timestamp=10.0, pose=np.eye(4) if pose is None else pose,
        wrench=np.asarray(wrench, dtype=float), wrench_frame="sensor",
        metadata=dict(external_compensated=compensated,
                      force_channel="zero_force_data" if compensated else "force_data"),
    )


class WrenchTests(unittest.TestCase):
    def test_sensor_offset_changes_tool_moment(self):
        config = processor_config()
        config["T_tool_sensor"][0][3] = 0.1
        current, filtered = WrenchProcessor(config).process(sensor_sample([0, 10, 0, 0, 0, 0]), np.eye(3), 0.01)
        np.testing.assert_allclose(current, [0, 10, 0, 0, 0, 1])
        np.testing.assert_allclose(filtered, current)

    def test_sensor_tool_and_task_orientations_are_distinct(self):
        config = processor_config()
        config["T_tool_sensor"] = pose_from_xyz_rpy([0, 0, 0, 0, 0, np.pi / 2]).tolist()
        base_tool = pose_from_xyz_rpy([0.3, 0.2, 0.1, 0, 0, np.pi / 2])
        base_task = exp_so3([0, 0, np.pi / 2])
        current, _ = WrenchProcessor(config).process(sensor_sample([1, 0, 0, 0, 0, 0], base_tool), base_task, 0.01)
        np.testing.assert_allclose(current, [0, 1, 0, 0, 0, 0], atol=1e-12)

    def test_raw_gravity_and_frozen_bias_are_removed_in_sensor_axes(self):
        config = processor_config()
        config["external_compensated"] = False
        config["bias"] = [1, 2, 3, 0.1, 0.2, 0.3]
        config["gravity"] = dict(mass=2, center_of_mass_sensor=[0.1, 0, 0], gravity_base=[0, 0, -9.81])
        # Tool/sensor rotate 90 degrees about Y: gravity becomes +sensor X.
        base_tool = pose_from_xyz_rpy([0, 0, 0, 0, np.pi / 2, 0])
        gravity_force = base_tool[:3, :3].T @ np.array([0, 0, -19.62])
        gravity_wrench = np.r_[gravity_force, np.cross([0.1, 0, 0], gravity_force)]
        external = np.array([0, 3, 0, 0, 0, 0])
        sample = sensor_sample(gravity_wrench + config["bias"] + external, base_tool, compensated=False)
        current, _ = WrenchProcessor(config).process(sample, np.eye(3), 0.01)
        np.testing.assert_allclose(current, external, atol=1e-12)

    def test_gravity_moment_is_removed_for_offset_center_of_mass(self):
        config = processor_config()
        config["external_compensated"] = False
        config["gravity"] = dict(mass=2, center_of_mass_sensor=[0.1, 0, 0], gravity_base=[0, 0, -10])
        current, _ = WrenchProcessor(config).process(sensor_sample([0, 0, -20, 0, 2, 0], compensated=False), np.eye(3), 0.01)
        np.testing.assert_allclose(current, np.zeros(6), atol=1e-12)

    def test_compensated_channel_never_subtracts_gravity_twice(self):
        config = processor_config()
        config["gravity"] = dict(mass=100, center_of_mass_sensor=[1, 2, 3], gravity_base=[0, 0, -10])
        current, _ = WrenchProcessor(config).process(sensor_sample([1, 2, 3, 4, 5, 6]), np.eye(3), 0.01)
        np.testing.assert_allclose(current, [1, 2, 3, 4, 5, 6])

    def test_compensation_channel_mismatch_is_rejected(self):
        processor = WrenchProcessor(processor_config())
        with self.assertRaises(ValueError):
            processor.process(sensor_sample([0] * 6, compensated=False), np.eye(3), 0.01)
        sample = sensor_sample([0] * 6)
        sample.metadata["force_channel"] = "force_data"
        with self.assertRaises(ValueError):
            processor.process(sample, np.eye(3), 0.01)
        sample = sensor_sample([0] * 6)
        sample.metadata.pop("external_compensated")
        with self.assertRaises(ValueError):
            processor.process(sample, np.eye(3), 0.01)

    def test_raw_channel_requires_gravity_parameters_and_sensor_origin(self):
        config = processor_config()
        config["external_compensated"] = False
        with self.assertRaises(ValueError):
            WrenchProcessor(config)
        config["gravity"] = dict(mass=0, center_of_mass_sensor=[0, 0, 0], gravity_base=[0, 0, -10])
        processor = WrenchProcessor(config)
        sample = sensor_sample([0] * 6, compensated=False)
        sample.wrench_frame = "base"
        with self.assertRaises(ValueError):
            processor.process(sample, np.eye(3), 0.01)

    def test_filter_uses_real_dt_and_current_channel_stays_unfiltered(self):
        processor = WrenchProcessor(processor_config())
        processor.process(sensor_sample([0] * 6), np.eye(3), 0.01)
        current, filtered = processor.process(sensor_sample([10, 0, 0, 0, 0, 0]), np.eye(3), 0.02)
        self.assertEqual(current[0], 10)
        self.assertAlmostEqual(filtered[0], 10 * (1 - np.exp(-0.2)))
        current[:] = 50
        filtered[:] = 50
        _, second = processor.process(sensor_sample([10, 0, 0, 0, 0, 0]), np.eye(3), 0.03)
        self.assertAlmostEqual(second[0], 10 * (1 - np.exp(-0.5)))

    def test_reset_preserves_bias(self):
        config = processor_config()
        config["bias"] = [1] * 6
        processor = WrenchProcessor(config)
        processor.process(sensor_sample([5] * 6), np.eye(3), 0.01)
        processor.reset()
        current, filtered = processor.process(sensor_sample([2] * 6), np.eye(3), 0.01)
        np.testing.assert_allclose(current, [1] * 6)
        np.testing.assert_allclose(filtered, current)

    def test_bad_intervals_nonfinite_loads_and_version_are_rejected(self):
        processor = WrenchProcessor(processor_config())
        for dt in (0, -0.01, np.nan, np.inf, 0.101):
            with self.assertRaises(ValueError):
                processor.process(sensor_sample([0] * 6), np.eye(3), dt)
        with self.assertRaises(ValueError):
            processor.process(sensor_sample([np.nan] * 6), np.eye(3), 0.01)
        sample = sensor_sample([0] * 6)
        sample.metadata["wrench_version"] = "other"
        with self.assertRaises(ValueError):
            processor.process(sample, np.eye(3), 0.01)
        sample.valid = False
        with self.assertRaises(ValueError):
            processor.process(sample, np.eye(3), 0.01)

    def test_task_shortcut_requires_simulation_and_tool_moment_origin(self):
        processor = WrenchProcessor(processor_config())
        sample = sensor_sample([0, 1, 0, 0, 0, 0])
        sample.wrench_frame = "task"
        sample.metadata["moment_origin"] = "tool"
        with self.assertRaises(ValueError):
            processor.process(sample, np.eye(3), 0.01)
        sample.metadata["simulation"] = True
        current, _ = processor.process(sample, np.eye(3), 0.01)
        np.testing.assert_allclose(current, sample.wrench)

    def test_base_expression_at_tool_origin_does_not_add_moment_twice(self):
        config = processor_config()
        config["T_tool_sensor"][0][3] = 0.1
        sample = sensor_sample([0, 10, 0, 0, 0, 1])
        sample.wrench_frame = "base"
        sample.metadata["moment_origin"] = "tool"
        current, _ = WrenchProcessor(config).process(sample, np.eye(3), 0.01)
        np.testing.assert_allclose(current, sample.wrench)
        sample.metadata.pop("moment_origin")
        with self.assertRaises(ValueError):
            WrenchProcessor(config).process(sample, np.eye(3), 0.01)


if __name__ == "__main__":
    unittest.main()
