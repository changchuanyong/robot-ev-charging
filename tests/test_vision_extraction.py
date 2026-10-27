"""Real pixel-to-correspondence integration with only the detector replaced."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision.contracts import rectify_points
from vision.pipeline import PosePipeline, ROOT


class VisionExtractionTests(unittest.TestCase):
    def setUp(self):
        config = json.loads((ROOT / "config/realsense_config.json").read_text())
        box = SimpleNamespace(cls=np.array(0), conf=np.array(0.99),
                              xyxy=np.array([[600., 110., 1000., 550.]]))
        detector = lambda image, **kwargs: [SimpleNamespace(boxes=[box])]
        self.pipeline = PosePipeline(config, detector=detector)
        self.pipeline._load()
        self.matrix = np.array([[1000., 0., 640.], [0., 1000., 360.], [0., 0., 1.]])
        self.rotation = np.array([np.pi, 0., 0.])
        self.translation = np.array([24., -4.5, 150.])

    def fixture(self, coefficients):
        names = list(self.pipeline.model)
        objects = np.asarray([self.pipeline.model[name] for name in names])
        pixels = cv2.projectPoints(objects, self.rotation, self.translation,
                                   self.matrix, coefficients)[0].reshape(-1, 2)
        image = np.full((720, 1280, 3), 160, dtype=np.uint8)
        for name, point in zip(names, pixels):
            radius = 34 if name.startswith("DC") else 22 if name == "PE" else 13
            cv2.circle(image, tuple(np.rint(point).astype(int)), radius,
                       (15, 15, 15), thickness=-1, lineType=cv2.LINE_AA)
        return image, dict(zip(names, pixels))

    def assert_measured_centers(self, measured, expected):
        self.assertEqual(set(measured), set(expected))
        for name in expected:
            with self.subTest(hole=name):
                # Rasterization, enhancement and the even-sized morphology kernel
                # shift contour centers slightly; incorrect labels are tens of pixels away.
                self.assertLess(np.linalg.norm(np.asarray(measured[name]) - expected[name]), 3.5)

    def test_actual_extraction_recovers_nine_holes_in_full_image_coordinates(self):
        coefficients = np.zeros(5)
        image, expected = self.fixture(coefficients)
        measured, metadata = self.pipeline._extract_points(
            image, projection=(self.matrix, coefficients, "brown_conrady"))
        self.assert_measured_centers(measured, expected)
        x1, y1, x2, y2 = metadata["crop_xyxy"]
        self.assertGreater(x1, 0)
        self.assertGreater(y1, 0)
        self.assertTrue(all(x1 < point[0] < x2 and y1 < point[1] < y2 for point in measured.values()))
        self.assertAlmostEqual(metadata["detection_score"], 0.99)
        self.assertGreaterEqual(metadata["contour_diagnostics"]["kept_candidate_count"], 9)
        diagnostics = metadata["correspondence_diagnostics"]
        self.assertGreater(diagnostics["evaluated_hypothesis_count"], 0)
        self.assertTrue(diagnostics["selection_quality_ok"])
        self.assertFalse(diagnostics["ambiguous"])

    def test_distortion_transform_keeps_returned_points_in_raw_pixels(self):
        coefficients = np.array([0.3, 0., 0., 0., 0.])
        image, expected_raw = self.fixture(coefficients)
        _, expected_ideal = self.fixture(np.zeros(5))
        with patch("vision.pipeline.rectify_points", wraps=rectify_points) as transform:
            measured, metadata = self.pipeline._extract_points(
                image, projection=(self.matrix, coefficients, "brown_conrady"))
        self.assertGreater(transform.call_count, 0)
        for call in transform.call_args_list:
            np.testing.assert_array_equal(call.args[1], self.matrix)
            np.testing.assert_array_equal(call.args[2], coefficients)
            self.assertEqual(call.args[3], "brown_conrady")
        self.assert_measured_centers(measured, expected_raw)
        # A rectified or ROI-local result would fail these raw-coordinate checks.
        self.assertGreater(np.linalg.norm(expected_raw["DC_pos"] - expected_ideal["DC_pos"]), 6.0)
        self.assertGreater(np.linalg.norm(np.asarray(measured["DC_pos"]) - expected_ideal["DC_pos"]), 4.0)
        self.assertGreater(metadata["correspondence_diagnostics"]["evaluated_hypothesis_count"], 0)


if __name__ == "__main__":
    unittest.main()
