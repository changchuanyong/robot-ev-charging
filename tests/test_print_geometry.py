import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vision.geometry_reference import load_geometry_reference, section_circles, read_binary_stl
from control.validate_admittance import exact_step, validate_responses
from vision.pipeline import PosePipeline


ROOT = Path(__file__).resolve().parents[1]


class PrintGeometryTests(unittest.TestCase):
    def test_pipeline_uses_explicit_geometry_instead_of_legacy_csv(self):
        pipeline = PosePipeline({"target_id": "fixture", "max_frame_age_s": 1.,
                                 "max_reprojection_rms_px": 3., "max_point_error_px": 5.,
                                 "confidence": .5, "min_inliers": 6,
                                 "geometry_reference": "config/geometry_nominal.json",
                                 "model_points": "nonexistent-legacy-model.csv"}, detector=object())
        pipeline._load()
        self.assertEqual(pipeline.model["DC_neg"], [-17, -1.2, 0])
        self.assertEqual(pipeline.geometry_source["version"], "legacy-nominal-xy-planar-v1")
        self.assertFalse(pipeline.geometry_source["validated"])

    def test_nominal_explicitly_unvalidated_and_template_rejected(self):
        data = load_geometry_reference(ROOT / "config/geometry_nominal.json")
        self.assertFalse(data["validated"])
        self.assertEqual(data["points_mm"]["DC_neg"], [-17, -1.2, 0])
        with self.assertRaises(ValueError):
            load_geometry_reference(ROOT / "config/geometry_reference.template.json")

    def test_measurement_provenance_and_nonplanar_sections_required(self):
        data = load_geometry_reference(ROOT / "config/geometry_nominal.json")
        data.update(status="measured_print", validated=True, specimen_id="s1", printing_batch_id="p1")
        data["points_mm"]["CC1"][2] = 1.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"geometry.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measurement_record"):
                load_geometry_reference(path)
            data["measurement"] = dict(operator="tester", measured_at="2026-10-29T12:00:00+08:00",
                                       instrument="test fixture", evidence_path="measurements.json", evidence_sha256="a"*64)
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_geometry_reference(path)["points_mm"]["CC1"][2], 1.5)
            for mutate in (lambda d: d["points_mm"]["CC1"].__setitem__(2, float("nan")),
                           lambda d: d["sections"].__setitem__("CC1", None),
                           lambda d: d.__setitem__("units", "m")):
                invalid = copy.deepcopy(data)
                mutate(invalid)
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_geometry_reference(path)

    def test_mesh_section_recovers_offset_circle_and_rejects_boundary(self):
        angles = np.linspace(0, 2*np.pi, 97)
        ring = np.c_[3+2*np.cos(angles), 4+2*np.sin(angles), np.zeros(len(angles))]
        triangles = []
        for a, b in zip(ring[:-1], ring[1:]):
            triangles.extend([[a, b, b+[0, 0, 5]], [a, b+[0, 0, 5], a+[0, 0, 5]]])
        triangles = np.asarray(triangles)
        result = section_circles(triangles, 2, 2.5)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]["center_cad"], [3, 4, 2.5], atol=1e-7)
        self.assertAlmostEqual(result[0]["diameter"], 4, delta=.003)
        with self.assertRaisesRegex(ValueError, "hits_vertices"):
            section_circles(triangles, 2, 0.)

    def test_truncated_stl_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"invalid.stl"
            path.write_bytes(b"short")
            with self.assertRaises(ValueError):
                read_binary_stl(path)


class AnalyticAdmittanceTests(unittest.TestCase):
    def test_all_damping_regimes_start_at_zero_reach_equilibrium(self):
        for damping in (2., 10., 20.):
            result = exact_step([0., 100.], 1., damping, 25., 2.)
            np.testing.assert_allclose(result, [0., .08], atol=1e-10)

    def test_discrete_response_converges_under_dt_refinement(self):
        report = validate_responses()
        self.assertTrue(report["passes_dt_refinement"])
        self.assertEqual(len(report["cases"]), 9)


if __name__ == "__main__":
    unittest.main()
