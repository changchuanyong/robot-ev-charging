from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

from vision.contracts import atomic_json, intrinsics_signature
from vision.preview_session import PreviewSession


ROOT = Path(__file__).resolve().parents[1]


class PreviewSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("preview-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.overlay = np.full((20, 30, 3), 180, np.uint8)
        self.camera = {
            "serial": "preview-test", "device_name": "RealSense D415", "device_model": "D415",
            "stream": "color", "image_width": 8, "image_height": 6, "fps": 15, "pixel_format": "bgr8",
            "camera_matrix": [[8., 0., 4.], [0., 8., 3.], [0., 0., 1.]],
            "dist_coeffs": [0.] * 5, "distortion_model": "none", "validated": False,
            "version": None, "source": "device_color_stream", "is_placeholder": False, "is_approximate": False,
        }
        self.camera["intrinsics_sha256"] = intrinsics_signature(self.camera)
        self.geometry = {"version": "nominal-test", "status": "nominal_reference",
                         "validated": False, "source_sha256": "a" * 64}

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_directory_outside_workspace_artifacts")
        shutil.rmtree(self.root)

    def session(self):
        return PreviewSession(self.root, self.camera, "preview-test", pipeline_config={"confidence": .25},
                              geometry=self.geometry)

    def diagnostic(self):
        return {"source": "live_diagnostic", "authorizes_robot_motion": False,
                "image_pixels_sha256": hashlib.sha256(self.image.tobytes()).hexdigest(),
                "camera": copy.deepcopy(self.camera), "geometry": dict(self.geometry),
                "frame": {"frame_number": 4, "device_timestamp_ms": 267.,
                          "timestamp_domain": "hardware_clock", "received_at_utc": "2026-10-29T08:00:00+00:00"},
                "frontend": {"reason": "ValueError: target_not_detected", "named_points_raw_px": {}},
                "pose": {"solved": False, "reason": "fewer_than_four_points"},
                "assessment": {"status": "no_pose", "reasons": ["target_not_detected"]},
                "processing_ms": 24.5}

    def read(self, session, name):
        return json.loads((session.path / name).read_text(encoding="utf-8"))

    def test_failed_detection_saves_original_overlay_and_bound_evidence(self):
        session = self.session()
        diagnostic = self.diagnostic()
        record = session.save(self.image, diagnostic, self.overlay)
        raw = cv2.imdecode(np.frombuffer((session.path / record["image"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        overlay = cv2.imdecode(np.frombuffer((session.path / record["overlay"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(raw, self.image)
        np.testing.assert_array_equal(overlay, self.overlay)
        saved = self.read(session, record["diagnostic"])
        self.assertEqual(saved["frame"], diagnostic["frame"])
        self.assertEqual(saved["image_sha256"], hashlib.sha256((session.path / record["image"]).read_bytes()).hexdigest())
        self.assertEqual(saved["intrinsics_sha256"], self.camera["intrinsics_sha256"])
        self.assertEqual(saved["overlay_sha256"], hashlib.sha256((session.path / record["overlay"]).read_bytes()).hexdigest())
        self.assertFalse(saved["pose"]["solved"])
        self.assertFalse(saved["authorizes_robot_motion"])
        self.assertEqual(self.read(session, "manifest.json")["split"], "unknown")
        self.assertEqual(self.read(session, "preview_manifest.json")["samples"], [record])
        self.assertEqual(session.saved_count, 1)
        session.finish()
        self.assertEqual(self.read(session, "manifest.json")["status"], "completed")
        self.assertEqual(self.read(session, "preview_manifest.json")["status"], "completed")

    def test_wrong_pixels_camera_geometry_or_source_rejected_before_saving(self):
        session = self.session()
        for change in ("pixels", "camera", "fps", "geometry", "source", "motion"):
            diagnostic = self.diagnostic()
            if change == "pixels":
                diagnostic["image_pixels_sha256"] = "b" * 64
            elif change == "camera":
                diagnostic["camera"]["serial"] = "another"
            elif change == "fps":
                diagnostic["camera"]["fps"] = 30
            elif change == "geometry":
                diagnostic["geometry"]["version"] = "different"
            elif change == "source":
                diagnostic["source"] = "hardware"
            else:
                diagnostic["authorizes_robot_motion"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                session.save(self.image, diagnostic, self.overlay)
        self.assertEqual(self.read(session, "manifest.json")["frames"], [])
        self.assertEqual(session.saved_count, 0)

    def test_same_frame_skipped_but_changed_frame_metadata_rejected(self):
        session = self.session()
        diagnostic = self.diagnostic()
        session.save(self.image, diagnostic, self.overlay)
        self.assertIsNone(session.save(self.image, diagnostic, self.overlay))
        diagnostic["frame"]["device_timestamp_ms"] += 1
        with self.assertRaisesRegex(ValueError, "previous_frame_metadata"):
            session.save(self.image, diagnostic, self.overlay)
        self.assertEqual(session.saved_count, 1)

    def test_sidecar_write_failure_keeps_raw_image_and_no_complete_index(self):
        session = self.session()
        with patch("vision.preview_session._write_new", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                session.save(self.image, self.diagnostic(), self.overlay)
        capture = self.read(session, "manifest.json")
        preview = self.read(session, "preview_manifest.json")
        self.assertEqual(capture["status"], "failed")
        self.assertEqual(len(capture["frames"]), 1)
        self.assertTrue((session.path / capture["frames"][0]["image"]).is_file())
        self.assertEqual(preview["status"], "failed")
        self.assertEqual(preview["samples"], [])
        self.assertEqual(session.saved_count, 0)
        session.finish()
        self.assertEqual(self.read(session, "preview_manifest.json")["status"], "failed")

    def test_index_failure_does_not_advertise_a_complete_sample(self):
        session = self.session()

        def fail_sample_index(path, data):
            if path.name == "preview_manifest.json" and data["samples"]:
                raise OSError("index unavailable")
            return atomic_json(path, data)

        with patch("vision.preview_session.atomic_json", side_effect=fail_sample_index):
            with self.assertRaisesRegex(OSError, "index unavailable"):
                session.save(self.image, self.diagnostic(), self.overlay)
        self.assertEqual(self.read(session, "preview_manifest.json")["samples"], [])
        self.assertEqual(self.read(session, "preview_manifest.json")["status"], "failed")
        self.assertEqual(session.saved_count, 0)
        self.assertEqual(len(self.read(session, "manifest.json")["frames"]), 1)

    def test_failed_diagnostics_serialize_nonfinite_values_as_null(self):
        session = self.session()
        diagnostic = self.diagnostic()
        diagnostic["pose"]["invalid_error"] = np.float64(np.nan)
        record = session.save(self.image, diagnostic, self.overlay)
        self.assertIsNone(self.read(session, record["diagnostic"])["pose"]["invalid_error"])

    def test_finish_write_failure_can_retry_without_losing_saved_sample(self):
        for target in ("vision.capture_session.atomic_json", "vision.preview_session.atomic_json"):
            with self.subTest(target=target):
                session = self.session()
                sample = session.save(self.image, self.diagnostic(), self.overlay)
                before = {name: (session.path / sample[name]).read_bytes()
                          for name in ("image", "overlay", "diagnostic")}
                with patch(target, side_effect=OSError("temporary write failure")):
                    with self.assertRaisesRegex(OSError, "temporary write failure"):
                        session.finish()
                self.assertEqual(session._manifest["status"], "recording")
                self.assertEqual(session.saved_count, 1)
                self.assertEqual(self.read(session, "preview_manifest.json")["status"], "recording")
                with self.assertRaisesRegex(RuntimeError, "already_finished"):
                    session.save(self.image, self.diagnostic(), self.overlay)
                session.finish()
                self.assertEqual(self.read(session, "manifest.json")["status"], "completed")
                preview = self.read(session, "preview_manifest.json")
                self.assertEqual(preview["status"], "completed")
                self.assertEqual(preview["samples"], [sample])
                self.assertEqual(session._manifest["status"], "completed")
                for name, contents in before.items():
                    self.assertEqual((session.path / sample[name]).read_bytes(), contents)

    def test_finish_retry_cannot_change_outcome_after_first_manifest_written(self):
        session = self.session()
        session.save(self.image, self.diagnostic(), self.overlay)
        with patch("vision.preview_session.atomic_json", side_effect=OSError("index unavailable")):
            with self.assertRaises(OSError):
                session.finish("interrupted", "operator interruption")
        session.finish("failed", "retry after write failure")
        for name in ("manifest.json", "preview_manifest.json"):
            record = self.read(session, name)
            self.assertEqual(record["status"], "interrupted")
            self.assertEqual(record["error"], "operator interruption")


if __name__ == "__main__":
    unittest.main()
