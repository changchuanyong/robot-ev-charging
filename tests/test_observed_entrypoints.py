"""One reviewed development photo through the live and file-based entry points."""
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from vision.pipeline import PosePipeline, ROOT

sys.path.insert(0, str(ROOT / "live" / "python"))
import Canny
import batch_pipeline_eval
import enhance
import latest_candidate_contours as contours
import observed_layout
import pipeline_quality
import pnp2
from center_fit import export_fitted_centers


class ObservedEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imdecode(np.frombuffer(
            (ROOT / "yolo_port/images/val/val_port_0001.jpg").read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        cls.prior = observed_layout.load_prior(observed_layout.DEFAULT_PRIOR_PATH)
        cls.matrix = np.array([[1053.62, 0., 950.39], [0., 1047.51, 527.34], [0., 0., 1.]])
        # Human-reviewed approximate centers; this is a regression fixture,
        # not independent test accuracy or physical calibration evidence.
        cls.expected = {"S_neg": [918, 417], "CC2": [981, 392], "S_pos": [1022, 423],
                        "CC1": [977, 465], "DC_neg": [911, 475], "DC_pos": [1067, 485],
                        "PE": [1004, 561]}

    def test_live_and_file_entrypoints_agree_on_measured_holes(self):
        config = json.loads((ROOT / "config/realsense_config.json").read_text())
        self.assertEqual(config["observed_layout_prior"], "config/observed_layout_prior.json")
        box = SimpleNamespace(cls=np.array(0), conf=np.array(.99),
                              xyxy=np.array([[758., 298., 1228., 713.]]))
        pipeline = PosePipeline(config, detector=lambda *args, **kwargs: [SimpleNamespace(boxes=[box])])
        measured, metadata = pipeline._extract_points(
            self.image, projection=(self.matrix, np.zeros(5), "brown_conrady"))
        self.assertEqual(set(measured), set(self.expected))
        for name, expected in self.expected.items():
            self.assertLess(np.linalg.norm(np.asarray(measured[name]) - expected), 3)
        self.assertNotIn("A_neg", measured)  # Manually marked occluded.
        x1, y1, x2, y2 = metadata["crop_xyxy"]
        raw = self.image[y1:y2, x1:x2]
        enhanced = cv2.cvtColor(enhance.preprocess_roi(raw)["final"], cv2.COLOR_GRAY2BGR)
        edges = Canny.adaptive_canny_pipeline(enhanced)[4]
        with contextlib.redirect_stdout(io.StringIO()):
            selected, _, diagnostics = contours.extract_observed_candidates(edges, enhanced, raw, self.prior)
        serialized = export_fitted_centers(selected)
        with patch.object(pnp2, "load_json", return_value=serialized):
            parsed = pnp2.roi_points_to_full_points(pnp2.parse_image_points(None), x1, y1)
        self.assertEqual(set(parsed), set(measured))
        for name in parsed:
            np.testing.assert_allclose(parsed[name], measured[name], atol=.001)
        self.assertEqual(diagnostics["ambiguous"], metadata["correspondence_diagnostics"]["ambiguous"])

        # Exercise the default batch path, including Unicode paths on Windows.
        with patch.object(Path, "mkdir"), patch.object(Path, "write_bytes") as save:
            output = ROOT / "artifacts" / "复核"
            row = batch_pipeline_eval.evaluate_one_image(
                ROOT / "yolo_port/images/val/val_port_0001.jpg", pipeline._detector,
                pipeline.model, self.matrix, np.zeros(5), False, output, self.prior)
            self.assertTrue(save.call_args.args[0].startswith(b"\xff\xd8"))
        self.assertEqual(set(row["image_points_full"]), set(measured))
        for name in measured:
            np.testing.assert_allclose(row["image_points_full"][name], measured[name])

    def test_file_entrypoint_requires_matching_raw_roi(self):
        enhanced = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "raw_roi_missing_or_shape_mismatch"):
            contours.extract_observed_candidates(enhanced[:, :, 0], enhanced, None, self.prior)

    def test_file_main_reads_encoded_images_and_exports_aperture_centers(self):
        raw = self.image[261:750, 721:1265]
        enhanced = cv2.cvtColor(enhance.preprocess_roi(raw)["final"], cv2.COLOR_GRAY2BGR)
        edges = Canny.adaptive_canny_pipeline(enhanced)[4]
        inputs = {}
        for name, pixels in (("RAW_ROI_IMAGE_PATH", raw), ("ROI_IMAGE_PATH", enhanced),
                             ("EDGES_IMAGE_PATH", edges)):
            path = MagicMock(spec=Path)
            path.exists.return_value = True
            path.read_bytes.return_value = cv2.imencode(".png", pixels)[1].tobytes()
            inputs[name] = path
        with (patch.multiple(contours, **inputs, SHOW_WINDOW=False),
              patch.object(contours.cv2, "imread", side_effect=AssertionError("Use Unicode-safe file reads")),
              patch.object(contours, "atomic_imwrite"),
              patch.object(contours, "atomic_write_json") as save,
              contextlib.redirect_stdout(io.StringIO())):
            contours.main()
        centers = save.call_args.args[1]
        with patch.object(pnp2, "load_json", return_value=centers):
            points = pnp2.roi_points_to_full_points(pnp2.parse_image_points(None), 721, 261)
        self.assertEqual(set(points), set(self.expected))
        for name in points:
            self.assertLess(np.linalg.norm(np.asarray(points[name]) - self.expected[name]), 3)
        self.assertTrue(all(item["association_method"] == "observed_2d_layout" for item in centers))

    def test_association_failure_is_not_downgraded_to_missing_hole_warning(self):
        names = list(pipeline_quality.EXPECTED_LAYOUT_NAMES)[:7]
        centers = [{"layout_name": name, "fitted_center": [10, 10], "raw_center": [10, 10],
                    "is_main_anchor": index < 2, "association_quality_ok": False}
                   for index, name in enumerate(names)]
        result = pipeline_quality.check_candidates(centers)
        self.assertEqual(result["status"], "fail")
        self.assertIn("observed_association_rejected", result["reasons"])


if __name__ == "__main__":
    unittest.main()
