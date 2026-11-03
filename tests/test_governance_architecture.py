"""Guard concrete ownership boundaries and retained public imports."""

import ast
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ev_charging.paths import PACKAGE_ROOT


class ArchitectureTests(unittest.TestCase):
    def test_help_can_import_with_explicit_root_without_data(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-X", "utf8", "-m", "ev_charging.control.simulate", "--help"],
                cwd=directory,
                env={**os.environ, "EV_CHARGING_ROOT": directory},
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_internal_import_graph_is_acyclic(self):
        modules = {
            "ev_charging."
            + p.relative_to(PACKAGE_ROOT).with_suffix("").as_posix().replace("/", "."): ast.parse(
                p.read_text(encoding="utf-8")
            )
            for p in PACKAGE_ROOT.rglob("*.py")
        }
        graph = {name: set() for name in modules}
        for name, tree in modules.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        target = node.module + "." + alias.name
                        target = target if target in modules else node.module
                        if target in modules:
                            graph[name].add(target)
        done = set()

        def visit(name, path):
            self.assertNotIn(name, path, " -> ".join([*path, name]))
            if name not in done:
                for target in graph[name]:
                    visit(target, [*path, name])
                done.add(name)

        for name in graph:
            visit(name, [])

    def test_removed_reverse_imports_do_not_return(self):
        forbidden = {
            "vision/algorithms/observed_layout.py": "ev_charging.vision.algorithms.pose_correspondence",
            "capture/intake.py": "ev_charging.datasets.vision.prepare_port_library",
            "control/run.py": "ev_charging.control.simulate",
            "control/experiments.py": "ev_charging.control.simulate",
        }
        for file, target in forbidden.items():
            with self.subTest(file=file):
                tree = ast.parse((PACKAGE_ROOT / file).read_text(encoding="utf-8"))
                self.assertNotIn(
                    target, [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
                )

    def test_existing_public_imports_are_same_implementations(self):
        from ev_charging.control import learning, logging, simulate
        from ev_charging.datasets.vision import prepare_port_library
        from ev_charging.review import review_library

        self.assertIs(simulate.make_policy, learning.make_policy)
        self.assertIs(simulate.load_log, logging.load_log)
        self.assertIs(prepare_port_library.write_review, review_library.write_review)

    def test_assignment_matches_exhaustive_small_problem(self):
        from ev_charging.vision.algorithms.layout_prior import _minimum_assignment

        cost = np.array([[2.0, 1.0, 8.0, 4.0], [1.0, 7.0, 2.0, 5.0], [3.0, 2.0, 9.0, 0.5]])
        chosen = _minimum_assignment(cost)
        expected = min(
            sum(cost[row, column] for row, column in enumerate(p))
            for p in itertools.permutations(range(cost.shape[1]), cost.shape[0])
        )
        self.assertEqual(len(set(chosen)), cost.shape[0])
        self.assertEqual(sum(cost[row, column] for row, column in enumerate(chosen)), expected)


if __name__ == "__main__":
    unittest.main()
