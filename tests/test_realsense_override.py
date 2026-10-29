"""Calibration override acceptance against a fake SDK; never opens a real camera."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from tests.test_data_capture import config, sdk_fixture
from vision.contracts import intrinsics_signature
from vision.realsense import RealSenseCamera


ROOT = Path(__file__).resolve().parents[1]


class CalibrationOverrideTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("camera-override-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.settings = {**config(), "max_frame_age_s": .5}
        sdk, *_ = sdk_fixture("Intel RealSense D435")
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), RealSenseCamera(self.settings) as camera:
            self.original = copy.deepcopy(camera.camera)
        self.fitted = copy.deepcopy(self.original)
        self.fitted["camera_matrix"][0][0] += 1
        self.fitted.update(version="synthetic-accepted-v1", validated=True, source="synthetic_fit",
                           source_sdk_intrinsics_sha256=intrinsics_signature(self.original))
        self.fitted["intrinsics_sha256"] = intrinsics_signature(self.fitted)
        self.report = {"kind": "rgb_intrinsics_check_v1", "version": self.fitted["version"],
                       "source_camera_sha256": intrinsics_signature(self.original),
                       "fitted_camera_sha256": intrinsics_signature(self.fitted),
                       "summary": {"heldout": {"view_count": 10, "fitted": {"pooled_corner_rms_px": .3}}}}
        self.report_path = self.root / "report.json"
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        self.fitted["validation_record"] = {"reviewer": "synthetic_test", "reviewed_at": "2026-10-29T01:00:00Z",
            "report_path": "report.json", "report_sha256": hashlib.sha256(self.report_path.read_bytes()).hexdigest(),
            "max_heldout_rms_px": .5}
        self.path = self.root / "accepted.json"
        self.settings.update(intrinsics_override_file=str(self.path), intrinsics_validated=True,
                             calibration_version=self.fitted["version"], intrinsics_sha256=self.fitted["intrinsics_sha256"])

    def cleanup(self):
        if self.root.resolve().parent != (ROOT / "artifacts/test-tmp").resolve():
            raise ValueError("invalid_test_cleanup_target")
        shutil.rmtree(self.root)

    def write(self, camera=None):
        self.path.write_text(json.dumps(camera or self.fitted), encoding="utf-8")

    def test_validated_matching_override_is_used_and_original_default_remains_sdk(self):
        self.write()
        sdk, *_ = sdk_fixture("Intel RealSense D435")
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), RealSenseCamera(self.settings) as camera:
            self.assertEqual(camera.camera["camera_matrix"][0][0], 101.)
            self.assertEqual(camera.camera["source_sdk_intrinsics_sha256"], intrinsics_signature(self.original))
        sdk.pipeline.return_value.stop.assert_called_once()
        sdk, *_ = sdk_fixture("Intel RealSense D435")
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), RealSenseCamera({**config(), "max_frame_age_s": .5}) as camera:
            self.assertEqual(camera.camera["camera_matrix"][0][0], 100.)
            self.assertFalse(camera.camera["validated"])

    def test_unvalidated_wrong_profile_wrong_version_or_changed_sdk_is_rejected(self):
        for field, value in (("validated", False), ("serial", "wrong"), ("device_name", "Intel RealSense D415"),
                             ("fps", 15), ("image_width", 12), ("version", "wrong"),
                             ("source_sdk_intrinsics_sha256", "0" * 64), ("validation_record", {})):
            bad = copy.deepcopy(self.fitted)
            bad[field] = value
            self.write(bad)
            sdk, *_ = sdk_fixture("Intel RealSense D435")
            with self.subTest(field=field), patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaises(ValueError):
                with RealSenseCamera(self.settings):
                    pass
            sdk.pipeline.return_value.stop.assert_called_once()

    def test_changed_report_and_failed_heldout_gate_are_rejected(self):
        self.write()
        self.report["summary"]["heldout"]["fitted"]["pooled_corner_rms_px"] = 1.2
        self.report_path.write_text(json.dumps(self.report), encoding="utf-8")
        for rehash in (False, True):
            if rehash:
                self.fitted["validation_record"]["report_sha256"] = hashlib.sha256(self.report_path.read_bytes()).hexdigest()
                self.write()
            sdk, *_ = sdk_fixture("Intel RealSense D435")
            with self.subTest(rehash=rehash), patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaises(ValueError):
                with RealSenseCamera(self.settings):
                    pass
            sdk.pipeline.return_value.stop.assert_called_once()

    def test_mismatched_actual_sdk_profile_stops_before_override(self):
        self.write()
        sdk, _, video, _ = sdk_fixture("Intel RealSense D435")
        video.fps.return_value = 15
        with patch.dict("sys.modules", {"pyrealsense2": sdk}), self.assertRaisesRegex(ValueError, "actual_D435_color_profile"):
            with RealSenseCamera(self.settings):
                pass
        sdk.pipeline.return_value.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
