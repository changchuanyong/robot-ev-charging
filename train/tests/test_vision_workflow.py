"""Offline workflows spanning the capture, algorithm, library and review packages."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from vision.capture.capture_session import PreviewSession
from vision.capture.collect import load_capture_config, validate_rgb_options
from vision.capture.preview import render_depth, render_preview
from vision.library.prepare_port_library import write_review
from vision.library.training_library import register_image
from vision.pipeline import PosePipeline, ROOT
from vision.review.prepare_keypoint_review import render_review
from vision.review.review_keypoints import build_annotations


class VisionWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = np.full((48, 64, 3), 100, dtype=np.uint8)

    def test_preview_saves_original_pixels_and_resolvable_code_fingerprints(self):
        camera = {"stream": "color", "serial": "fixture", "device_name": "Fixture D415",
                  "device_model": "D415", "image_width": 64, "image_height": 48,
                  "fps": 30, "pixel_format": "bgr8", "validated": False, "version": None,
                  "camera_matrix": [[60., 0., 32.], [0., 60., 24.], [0., 0., 1.]],
                  "dist_coeffs": [0.] * 5, "distortion_model": "none"}
        geometry = {"version": "fixture", "status": "nominal", "validated": False,
                    "source_sha256": None}
        frame = {"frame_number": 1, "device_timestamp_ms": 1.,
                 "timestamp_domain": "hardware_clock", "received_at_utc": "2026-11-04T00:00:00+00:00"}
        diagnostic = {"image_pixels_sha256": hashlib.sha256(self.image.tobytes()).hexdigest(),
                      "source": "live_diagnostic", "authorizes_robot_motion": False,
                      "camera": camera, "geometry": geometry, "frame": frame}
        original = self.image.copy()
        overlay = render_preview(self.image, diagnostic)
        session = PreviewSession(self.root / "captures", camera, "fixture",
                                 pipeline_config={}, geometry=geometry)
        self.addCleanup(session.finish)
        saved = session.save(self.image, diagnostic, overlay)
        self.assertIsNone(session.save(self.image, diagnostic, overlay))
        session.finish()
        np.testing.assert_array_equal(self.image, original)
        np.testing.assert_array_equal(cv2.imread(str(session.path / saved["image"])), original)
        manifest = json.loads((session.path / "preview_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len(manifest["samples"]), 1)
        for asset in manifest["code"].values():
            self.assertEqual(asset["sha256"], hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest())

    def test_depth_display_preserves_raw_measurements(self):
        raw = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
        original = raw.copy()
        panel = render_depth({"raw_z16": raw, "calibration": {"depth_scale_m": .001}}, {})
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(panel.shape, (720, 1140, 3))
        self.assertEqual(panel.dtype, np.uint8)

    def test_offline_box_and_keypoint_review_use_packaged_templates(self):
        image = self.root / "source.png"
        self.assertTrue(cv2.imwrite(str(image), self.image))
        library = self.root / "dataset/vision"
        register_image(image, library_root=library, source={"import_kind": "fixture"})
        output = self.root / "boxes.html"
        report = write_review(library, output, weights=None)
        self.assertEqual(report["images"], 1)
        self.assertNotIn("/*__REVIEW_DATA__*/[]", output.read_text(encoding="utf-8"))
        page = render_review(build_annotations(library, root=self.root), self.root)
        self.assertNotIn("__KEYPOINT_PAYLOAD__", page)
        self.assertIn("data:image/png;base64,", page)

    def test_pipeline_loads_algorithm_assets_without_loading_a_detector(self):
        _, options = load_capture_config(ROOT / "config/d415.json", "preview")
        detector = object()
        pipeline = PosePipeline(options["vision"], detector=detector)
        pipeline._load()
        self.assertIs(pipeline._detector, detector)
        self.assertEqual(len(pipeline.model), 9)
        self.assertTrue(pipeline.observed_prior["templates"])
        validate_rgb_options({"enable_auto_exposure": 0, "exposure": 100})
        with self.assertRaisesRegex(ValueError, "conflicts_with_auto_exposure"):
            validate_rgb_options({"enable_auto_exposure": 1, "exposure": 100})


if __name__ == "__main__":
    unittest.main()
