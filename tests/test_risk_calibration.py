"""Synthetic domain compilation tests; the numerical bounds are not hardware limits."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
import uuid

import numpy as np

from control.build_risk_domain import CHANNELS, UNITS, build_domain, main, validate_records
from control.risk import EngineeringRisk


class RiskDomainBuilderTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / "artifacts" / ("risk-builder-test-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        evidence = self.directory / "evidence.txt"
        evidence.write_text("Synthetic analytic fixture only. Not a hardware acceptance record.", encoding="utf-8")
        self.spec = {
            "schema_version": 1, "version": "synthetic-domain-v1", "coordinate_version": "test-axis-v1",
            "specimen_version": "test-specimen-v1", "tool_version": "test-tool-v1",
            "wrench_calibration_version": "test-force-v1", "frame": "task", "moment_origin": "plug",
            "units": UNITS, "reference_wrench": [0.] * 6, "reference_source": "fixture",
            "sources": [{"id": "fixture", "version": "test-only-v1", "path": "evidence.txt"}],
            "limits": {name: {"lower": -2., "upper": 3., "lower_source": "fixture", "upper_source": "fixture"}
                       for name in CHANNELS},
        }

    def tearDown(self):
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()

    def domain(self):
        return build_domain(self.spec, self.directory)

    def log(self, vectors, kind="cycle", source="hardware", identity_change=None, raw=False):
        identity = {key: self.spec[key] for key in ("coordinate_version", "specimen_version", "tool_version",
                                                   "wrench_calibration_version", "frame", "moment_origin", "units")}
        identity.update(split="validation", domain_version=self.spec["version"])
        identity.update(identity_change or {})
        task_id = "test-task-" + uuid.uuid4().hex
        rows = [{"kind": "manifest", "task_id": task_id, "source": source, "risk_validation": identity}]
        for vector in vectors:
            sample = {"valid": True, "wrench": vector, "wrench_frame": "sensor" if raw else "task",
                      "metadata": {"moment_origin": "tool"}}
            row = {"kind": kind, "task_id": task_id, "sample": sample}
            if kind == "cycle":
                row["decision"] = {"state": "INSERT", "details": {"wrench": vector}}
            rows.append(row)
        path = self.directory / (uuid.uuid4().hex + ".jsonl")
        path.write_text("\n".join(json.dumps(row, allow_nan=False) for row in rows), encoding="utf-8")
        return path

    def test_axis_limits_are_compiled_without_losing_asymmetry_or_sources(self):
        domain = self.domain()
        self.assertFalse(domain["calibrated"])
        self.assertEqual(domain["scale"], [3.] * 6)
        np.testing.assert_allclose(domain["b"], [1.] * 6 + [2/3] * 6)
        risk = EngineeringRisk(domain)
        self.assertTrue(risk.evaluate([2.99, 0, 0, 0, 0, 0]).valid)
        self.assertTrue(risk.evaluate([-1.99, 0, 0, 0, 0, 0]).valid)
        self.assertTrue(risk.evaluate([-2.01, 0, 0, 0, 0, 0]).outside)
        self.assertEqual(len(domain["provenance"]["constraint_sources"]), 12)
        expected_hash = hashlib.sha256((self.directory / "evidence.txt").read_bytes()).hexdigest()
        self.assertEqual(domain["provenance"]["sources"][0]["sha256"], expected_hash)

    def test_coupled_bounds_and_physical_reference_normalization(self):
        self.spec["scale"] = [2.] * 6
        self.spec["reference_wrench"] = [0.2, 0, 0, 0, 0, 0]
        self.spec["coupled"] = [{"a": [1, 0, 0, 1, 0, 0], "b": 0.5, "space": "normalized",
                                 "source": "fixture", "version": "synthetic-coupling-v1"}]
        domain = self.domain()
        self.assertEqual(domain["reference"][0], 0.1)
        self.assertEqual(len(domain["b"]), 13)
        self.assertTrue(EngineeringRisk(domain).evaluate([0.8, 0, 0, 0.8, 0, 0]).outside)

    def test_invalid_units_reference_evidence_and_unsigned_bounds_are_rejected(self):
        cases = []
        bad = deepcopy(self.spec); bad["units"][0] = "kN"; cases.append(bad)
        bad = deepcopy(self.spec); bad["reference_wrench"][0] = 3.; cases.append(bad)
        bad = deepcopy(self.spec); bad["sources"][0]["sha256"] = "0"*64; cases.append(bad)
        bad = deepcopy(self.spec); bad["limits"]["Fx"]["lower"] = 0; cases.append(bad)
        bad = deepcopy(self.spec); bad["limits"]["Fx"]["upper_source"] = "missing"; cases.append(bad)
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                build_domain(spec, self.directory)

    def test_independent_logs_report_outside_fraction_without_fitting(self):
        domain = self.domain()
        initial_bounds = deepcopy(domain["b"])
        path = self.log([[0]*6, [4, 0, 0, 0, 0, 0], [-2.5, 0, 0, 0, 0, 0]])
        result = validate_records(domain, [path])
        report = result["logs"][0]
        self.assertEqual(report["evaluable_samples"], 3)
        self.assertEqual(report["outside_samples"], 2)
        self.assertAlmostEqual(report["outside_fraction"], 2/3)
        self.assertEqual(domain["b"], initial_bounds)
        self.assertFalse(domain["calibrated"])
        self.assertFalse(result["boundaries_refitted"])

    def test_task_samples_supported_but_raw_sensor_samples_refused(self):
        domain = self.domain()
        valid = self.log([[0]*6], kind="sample", source="simulation")
        self.assertEqual(validate_records(domain, [valid])["logs"][0]["source"], "simulation")
        raw = self.log([[0]*6], kind="sample", raw=True)
        with self.assertRaisesRegex(ValueError, "raw data"):
            validate_records(domain, [raw])

    def test_validation_identity_split_and_duplicate_logs_are_rejected(self):
        for change in ({"split": "train"}, {"specimen_version": "another-sample"},
                       {"coordinate_version": "another-axis"}, {"domain_version": "another-domain"}):
            path = self.log([[0]*6], identity_change=change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_records(self.domain(), [path])
        path = self.log([[0]*6])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records(self.domain(), [path, path])

    def test_calibrated_flag_in_input_cannot_auto_approve_hardware(self):
        self.spec["calibrated"] = True
        domain = self.domain()
        self.assertFalse(domain["calibrated"])
        self.assertEqual(domain["acceptance"]["status"], "manual_review_required")

    def test_boundary_evidence_cannot_double_as_independent_validation(self):
        path = self.log([[0]*6])
        self.spec["sources"][0]["path"] = path.name
        with self.assertRaisesRegex(ValueError, "cannot reuse"):
            validate_records(self.domain(), [path])

    def test_cli_keeps_existing_output_and_always_emits_uncalibrated_domain(self):
        spec_path, output = self.directory / "spec.json", self.directory / "domain.json"
        spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        self.assertEqual(main(["--spec", str(spec_path), "--output", str(output)]), 0)
        first = output.read_bytes()
        self.assertFalse(json.loads(first)["calibrated"])
        with self.assertRaises(SystemExit):
            main(["--spec", str(spec_path), "--output", str(output)])
        self.assertEqual(output.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
