"""Check installed entrypoints, resources, device isolation and shared runtimes."""

import importlib
import os
import pkgutil
import subprocess
import sys
import tempfile
import unittest
from importlib import metadata, resources
from unittest.mock import patch

import ev_charging
from ev_charging.paths import PACKAGE_ROOT, PROJECT_ROOT, find_project_root
from ev_charging.runtime import configure_caches


class PackageTests(unittest.TestCase):
    def test_all_modules_import_without_device_access(self):
        configure_caches()
        from ev_charging.hardware.realman import RealManRobot
        from ev_charging.vision import launch

        with (
            patch.object(RealManRobot, "__init__", side_effect=AssertionError("robot constructed")),
            patch.object(
                launch, "ensure_camera_privilege", side_effect=AssertionError("camera privilege requested")
            ),
        ):
            modules = list(pkgutil.walk_packages(ev_charging.__path__, "ev_charging."))
            self.assertGreater(len(modules), 60)
            for module in modules:
                with self.subTest(module=module.name):
                    importlib.import_module(module.name)

    def test_review_templates_are_packaged(self):
        package = resources.files("ev_charging.review")
        for name in ("keypoint_review.html", "port_box_review.html"):
            with self.subTest(template=name):
                self.assertIn("<html", package.joinpath(name).read_text(encoding="utf-8").lower())
        from ev_charging.review.review_keypoints import TEMPLATE as keypoints
        from ev_charging.review.review_library import TEMPLATE as boxes

        self.assertEqual(keypoints.parent, PACKAGE_ROOT / "review")
        self.assertEqual(boxes.parent, PACKAGE_ROOT / "review")

    def test_import_and_defaults_outside_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    "from ev_charging.paths import PROJECT_ROOT; "
                    "from ev_charging.vision.model_config import current_model_path; "
                    "from ev_charging.datasets.vision.training_library import DEFAULT_LIBRARY; "
                    "assert current_model_path(verify=True).is_file(); "
                    "assert DEFAULT_LIBRARY == PROJECT_ROOT / 'dataset/vision'",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_data_root_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"EV_CHARGING_ROOT": directory}):
                with self.assertRaises(RuntimeError):
                    find_project_root()
            with patch.dict(os.environ, {"EV_CHARGING_ROOT": str(PROJECT_ROOT)}):
                self.assertEqual(find_project_root(), PROJECT_ROOT)

    def test_registered_commands_show_help(self):
        entries = metadata.distribution("robot-ev-charging").entry_points
        commands = [entry for entry in entries if entry.group == "console_scripts"]
        self.assertEqual(len(commands), 5)
        for entry in commands:
            with self.subTest(command=entry.name):
                module = entry.value.split(":")[0]
                result = subprocess.run(
                    [sys.executable, "-B", "-m", module, "--help"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_mujoco_and_torch_share_numpy(self):
        import mujoco
        import numpy as np
        import torch

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        for _ in range(10):
            mujoco.mj_step(model, data)
        self.assertGreater(data.time, 0)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(
            torch.equal(torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)), torch.tensor([1.0, 2.0]))
        )


if __name__ == "__main__":
    unittest.main()
