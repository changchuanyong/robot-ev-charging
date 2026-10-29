"""Raw RGBD evidence stays bound to its displayed diagnostic and calibration."""
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

from vision.contracts import intrinsics_signature
from vision.depth_diagnostic import analyze_depth
from vision.preview_session import PreviewSession, _write_new


ROOT = Path(__file__).resolve().parents[1]


class PreviewDepthSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("preview-depth-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.overlay = np.full((20, 30, 3), 180, np.uint8)
        self.camera = {
            "serial": "depth-test", "device_name": "RealSense D415", "device_model": "D415",
            "stream": "color", "image_width": 8, "image_height": 6, "fps": 15, "pixel_format": "bgr8",
            "camera_matrix": [[8., 0., 4.], [0., 8., 3.], [0., 0., 1.]],
            "dist_coeffs": [0.] * 5, "distortion_model": "none", "validated": False,
            "version": None, "source": "device_color_stream", "is_placeholder": False, "is_approximate": False,
        }
        self.camera["intrinsics_sha256"] = intrinsics_signature(self.camera)
        depth_camera = {**self.camera, "stream": "depth", "pixel_format": "z16",
                        "source": "device_depth_stream", "image_width": 4, "image_height": 3}
        depth_camera["intrinsics_sha256"] = intrinsics_signature(depth_camera)
        self.calibration = {
            "schema_version": 1, "depth_camera": depth_camera,
            "color_intrinsics_sha256": self.camera["intrinsics_sha256"], "depth_scale_m": .001,
            "depth_to_color": {"rotation": np.eye(3).tolist(), "translation_m": [.015, 0., 0.]},
            "depth_profile": {"width": 4, "height": 3, "fps": 15, "format": "z16"},
            "max_pair_skew_ms": 40.,
        }
        self.geometry = {"version": "nominal-test", "status": "nominal_reference",
                         "validated": False, "source_sha256": "a" * 64}
        self.frame = {"frame_number": 4, "device_timestamp_ms": 267.,
                      "timestamp_domain": "hardware_clock", "received_at_utc": "2026-10-29T08:00:00+00:00"}

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_directory_outside_workspace_artifacts")
        shutil.rmtree(self.root)

    def session(self):
        return PreviewSession(self.root, self.camera, "depth-test", pipeline_config={},
                              geometry=self.geometry, depth_calibration=self.calibration)

    def evidence(self, raw=None, *, missing=False):
        if raw is None and not missing:
            raw = np.array([0, 1, 255, 256, 511, 512, 1000, 5000, 65535, 400, 800, 1200],
                           dtype=np.uint16).reshape(3, 4)
        depth_frame = None if missing else {**self.frame, "frame_number": 2, "device_timestamp_ms": 266.}
        packet = {"raw_z16": raw, "frame": depth_frame, "color_frame": dict(self.frame),
                  "calibration": copy.deepcopy(self.calibration), "points_color_m": None,
                  "color_pixels": None, "reasons": ["depth_missing"] if missing else []}
        diagnostic = {"source": "live_diagnostic", "authorizes_robot_motion": False,
                      "image_pixels_sha256": hashlib.sha256(self.image.tobytes()).hexdigest(),
                      "camera": copy.deepcopy(self.camera), "geometry": dict(self.geometry),
                      "frame": dict(self.frame), "frontend": {}, "pose": {"solved": False},
                      "assessment": {"status": "no_pose", "reasons": ["target_not_detected"]},
                      "depth": {"status": "unavailable" if missing else "no_target",
                                "reasons": ["depth_missing"] if missing else ["target_not_detected"],
                                "frame": copy.deepcopy(depth_frame),
                                "calibration": copy.deepcopy(self.calibration),
                                "raw_pixels_sha256": None if raw is None else hashlib.sha256(raw.tobytes()).hexdigest()}}
        return packet, diagnostic

    def read(self, session, name):
        return json.loads((session.path / name).read_text(encoding="utf-8"))

    def test_16bit_roundtrip_metadata_and_hashes_preserve_measurements(self):
        session = self.session()
        packet, diagnostic = self.evidence()
        sample = session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
        depth = sample["depth"]
        png = (session.path / depth["file"]).read_bytes()
        decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.dtype, np.uint16)
        np.testing.assert_array_equal(decoded, packet["raw_z16"])
        self.assertEqual(depth["file_sha256"], hashlib.sha256(png).hexdigest())
        self.assertEqual(depth["raw_pixels_sha256"], hashlib.sha256(decoded.tobytes()).hexdigest())
        self.assertEqual(depth["frame"], packet["frame"])
        self.assertEqual(depth["color_frame"], diagnostic["frame"])
        self.assertEqual(depth["depth_scale_m"], .001)
        self.assertEqual(self.read(session, depth["calibration_file"]), self.calibration)
        self.assertEqual(depth["calibration_sha256"], hashlib.sha256(
            (session.path / depth["calibration_file"]).read_bytes()).hexdigest())
        saved = self.read(session, sample["diagnostic"])
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(saved["depth_capture"], depth)
        self.assertEqual(self.read(session, "preview_manifest.json")["schema_version"], 2)
        self.assertEqual(self.read(session, "manifest.json")["schema_version"], 1)
        self.assertNotIn("points_color_m", saved["depth"])
        session.finish()
        self.assertEqual(self.read(session, "preview_manifest.json")["status"], "completed")

    def test_missing_depth_saves_rgb_without_claiming_a_depth_file(self):
        session = self.session()
        packet, diagnostic = self.evidence(missing=True)
        sample = session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
        self.assertFalse(sample["depth"]["available"])
        self.assertIsNone(sample["depth"]["file"])
        self.assertIsNone(sample["depth"]["raw_pixels_sha256"])
        self.assertEqual(sample["depth"]["reasons"], ["depth_missing"])
        self.assertEqual(list((session.path / "depth").iterdir()), [])
        self.assertTrue((session.path / sample["image"]).is_file())

    def test_invalid_zero_depth_and_bad_pair_are_saved_as_failed_measurements(self):
        session = self.session()
        packet, diagnostic = self.evidence(np.zeros((3, 4), dtype=np.uint16))
        packet["frame"]["device_timestamp_ms"] = 10.
        diagnostic["depth"].update({"status": "unavailable", "reasons": ["pair_skew", "no_valid_depth"],
                                    "frame": copy.deepcopy(packet["frame"])})
        sample = session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
        self.assertTrue(sample["depth"]["available"])
        self.assertEqual(sample["depth"]["status"], "unavailable")
        self.assertEqual(sample["depth"]["reasons"], ["pair_skew", "no_valid_depth"])

    def test_analyzer_nonfinite_depth_timestamp_saves_rejected_raw_evidence(self):
        for timestamp in (float("nan"), float("inf")):
            with self.subTest(timestamp=timestamp):
                session = self.session()
                packet, diagnostic = self.evidence()
                packet["frame"]["device_timestamp_ms"] = timestamp
                diagnostic["depth"] = analyze_depth(packet, self.frame, self.camera,
                                                     diagnostic["frontend"], diagnostic["pose"], self.geometry)
                self.assertEqual(diagnostic["depth"]["status"], "invalid")
                self.assertIn("invalid_depth_timestamp", diagnostic["depth"]["reasons"])
                sample = session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
                self.assertIsNone(sample["depth"]["frame"]["device_timestamp_ms"])
                self.assertTrue(sample["depth"]["available"])
                self.assertTrue((session.path / sample["depth"]["file"]).is_file())
                self.assertIn("invalid_depth_timestamp", sample["depth"]["reasons"])
                self.assertIsNone(session.save(self.image, diagnostic, self.overlay, depth_sample=packet))

    def test_analyzer_missing_depth_saves_rgb_with_unavailable_evidence(self):
        session = self.session()
        packet, diagnostic = self.evidence(missing=True)
        diagnostic["depth"] = analyze_depth(packet, self.frame, self.camera,
                                             diagnostic["frontend"], diagnostic["pose"], self.geometry)
        sample = session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
        self.assertFalse(sample["depth"]["available"])
        self.assertIsNone(sample["depth"]["frame"])
        self.assertIn("missing_or_invalid_raw_depth", sample["depth"]["reasons"])
        self.assertTrue((session.path / sample["image"]).is_file())

    def test_mismatched_depth_evidence_rejected_before_rgb_save(self):
        session = self.session()
        for change in ("pixels", "shape", "dtype", "frame", "color_frame", "calibration", "diagnostic_calibration"):
            packet, diagnostic = self.evidence()
            if change == "pixels":
                packet["raw_z16"][0, 0] = 123
            elif change == "shape":
                packet["raw_z16"] = packet["raw_z16"].reshape(4, 3)
            elif change == "dtype":
                packet["raw_z16"] = packet["raw_z16"].astype(np.float32)
            elif change == "frame":
                packet["frame"]["frame_number"] += 1
            elif change == "color_frame":
                packet["color_frame"]["frame_number"] += 1
            elif change == "calibration":
                packet["calibration"]["depth_scale_m"] *= 10
            else:
                diagnostic["depth"]["calibration"]["depth_scale_m"] *= 10
            with self.subTest(change=change), self.assertRaises(ValueError):
                session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
        self.assertEqual(self.read(session, "manifest.json")["frames"], [])

    def test_same_rgb_frame_cannot_be_rebound_to_different_depth(self):
        session = self.session()
        packet, diagnostic = self.evidence()
        session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
        self.assertIsNone(session.save(self.image, diagnostic, self.overlay, depth_sample=packet))
        packet["raw_z16"][0, 0] = 12
        diagnostic["depth"]["raw_pixels_sha256"] = hashlib.sha256(packet["raw_z16"].tobytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "previous_frame_metadata"):
            session.save(self.image, diagnostic, self.overlay, depth_sample=packet)

    def test_depth_sidecar_failure_keeps_rgb_and_does_not_advertise_complete_pair(self):
        session = self.session()
        packet, diagnostic = self.evidence()

        def fail_depth(path, data):
            if Path(path).parent.name == "depth":
                raise OSError("depth disk full")
            _write_new(path, data)

        with patch("vision.preview_session._write_new", side_effect=fail_depth):
            with self.assertRaisesRegex(OSError, "depth disk full"):
                session.save(self.image, diagnostic, self.overlay, depth_sample=packet)
        capture = self.read(session, "manifest.json")
        preview = self.read(session, "preview_manifest.json")
        self.assertEqual(capture["status"], "failed")
        self.assertTrue((session.path / capture["frames"][0]["image"]).is_file())
        self.assertEqual(preview["status"], "failed")
        self.assertEqual(preview["samples"], [])
        self.assertEqual(session.saved_count, 0)

    def test_calibration_mismatch_is_rejected_before_session_created(self):
        for change in ("color", "scale", "rotation", "profile"):
            calibration = copy.deepcopy(self.calibration)
            if change == "color":
                calibration["color_intrinsics_sha256"] = "different"
            elif change == "scale":
                calibration["depth_scale_m"] = 0.
            elif change == "rotation":
                calibration["depth_to_color"]["rotation"][0][0] = -1.
            else:
                calibration["depth_profile"]["width"] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                PreviewSession(self.root, self.camera, "depth-test", pipeline_config={},
                               geometry=self.geometry, depth_calibration=calibration)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
