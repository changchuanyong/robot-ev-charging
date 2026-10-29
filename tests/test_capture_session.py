from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np

from vision.capture_session import CaptureSession
from vision.contracts import atomic_json, intrinsics_signature, validate_camera


ROOT = Path(__file__).resolve().parents[1]


def camera_config(serial="capture_test"):
    return {
        "serial": serial, "device_name": "Intel RealSense D415", "stream": "color",
        "image_width": 8, "image_height": 6,
        "camera_matrix": [[8., 0., 4.], [0., 8., 3.], [0., 0., 1.]],
        "dist_coeffs": [0.] * 5, "distortion_model": "none", "validated": False,
        "source": "device_color_stream", "is_placeholder": False, "is_approximate": False,
    }


class CaptureSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("RGB采集-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_directory_outside_workspace_artifacts")
        shutil.rmtree(self.root)

    def session(self, **kwargs):
        return CaptureSession(self.root, kwargs.pop("camera", camera_config()), "batch-a", **kwargs)

    def save(self, session, number=1, image=None):
        return session.save(self.image if image is None else image, number, 1000. + number,
                            "hardware_clock", "2026-10-28T01:02:03+00:00")

    def manifest(self, session):
        return json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))

    def test_full_BGR_png_metadata_hashes_and_finish(self):
        session = self.session(notes="完整原图")
        self.assertEqual(self.manifest(session)["status"], "recording")
        record = self.save(session)
        payload = (session.path / record["image"]).read_bytes()
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, self.image)
        self.assertEqual(record["image_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual((record["width"], record["height"]), (8, 6))
        self.assertEqual(record["image"], "images/frame_000001.png")
        self.assertEqual(record["frame_number"], 1)
        manifest = self.manifest(session)
        self.assertEqual(manifest["frames"], [record])
        self.assertEqual(manifest["camera_sha256"], hashlib.sha256((session.path / "camera.json").read_bytes()).hexdigest())
        session.finish()
        self.assertEqual(self.manifest(session)["status"], "completed")
        with self.assertRaisesRegex(RuntimeError, "finished"):
            self.save(session, 2)

    def test_repeated_device_frames_and_identical_pixels_are_skipped(self):
        session = self.session()
        self.save(session)
        self.assertIsNone(self.save(session, 1, self.image + 1))
        self.assertIsNone(self.save(session, 2))
        self.assertIsNone(self.save(session, 2, self.image + 1))
        second = self.save(session, 3, self.image + 1)
        self.assertEqual(second["image"], "images/frame_000002.png")
        self.assertEqual(len(self.manifest(session)["frames"]), 2)

    def test_sessions_keep_their_own_camera_and_reject_tampering(self):
        camera = camera_config("first")
        first = self.session(camera=camera)
        camera["serial"] = "mutated"
        camera["camera_matrix"][0][0] = 99
        second = self.session(camera=camera_config("second"))
        self.assertNotEqual(first.path, second.path)
        recorded = json.loads((first.path / "camera.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["serial"], "first")
        self.assertEqual(recorded["camera_matrix"][0][0], 8.)
        validate_camera(recorded, self.image.shape)
        self.save(first)
        atomic_json(first.path / "camera.json", camera_config("wrong_camera"))
        with self.assertRaisesRegex(ValueError, "camera_file_changed"):
            self.save(first, 2, self.image + 1)
        self.assertEqual(len(self.manifest(first)["frames"]), 1)
        self.assertEqual(json.loads((second.path / "camera.json").read_text())["serial"], "second")

    def test_rejects_wrong_stream_fingerprint_and_factory_validation(self):
        camera = camera_config()
        camera["intrinsics_sha256"] = intrinsics_signature(camera)
        camera["serial"] = "other_camera"
        for bad in (camera, dict(camera_config(), stream="depth"),
                    dict(camera_config(), validated=True), dict(camera_config(), serial="")):
            with self.subTest(camera=bad), self.assertRaises(ValueError):
                self.session(camera=bad)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_one_batch_cannot_cross_splits_but_other_batches_can(self):
        first = self.session()
        with self.assertRaisesRegex(ValueError, "cross_dataset_splits"):
            self.session(split="test")
        second = CaptureSession(self.root, camera_config(), "batch-b", "test")
        self.assertEqual(self.manifest(first)["split"], "train")
        self.assertEqual(self.manifest(second)["split"], "test")

    def test_unknown_split_is_preserved_and_still_grouped_by_batch(self):
        session = self.session(split="unknown")
        self.assertEqual(self.manifest(session)["split"], "unknown")
        self.assertIsNotNone(self.save(session))
        with self.assertRaisesRegex(ValueError, "cross_dataset_splits"):
            self.session(split="train")

    def test_png_failure_does_not_publish_a_frame(self):
        session = self.session()
        with patch("vision.capture_session.cv2.imencode", return_value=(False, None)):
            with self.assertRaisesRegex(OSError, "PNG_encoding_failed"):
                self.save(session)
        self.assertEqual(self.manifest(session)["frames"], [])
        self.assertEqual(list((session.path / "images").iterdir()), [])
        self.assertIsNotNone(self.save(session))

    def test_manifest_failure_removes_only_new_frame_and_allows_retry(self):
        session = self.session()
        first = self.save(session)
        before = (session.path / "manifest.json").read_bytes()
        with patch("vision.contracts.os.replace", side_effect=OSError("disk_full")):
            with self.assertRaisesRegex(OSError, "disk_full"):
                self.save(session, 2, self.image + 1)
        self.assertEqual((session.path / "manifest.json").read_bytes(), before)
        self.assertTrue((session.path / first["image"]).is_file())
        self.assertFalse((session.path / "images/frame_000002.png").exists())
        self.assertEqual(list(session.path.glob("*.tmp")), [])
        self.assertIsNotNone(self.save(session, 2, self.image + 1))

    def test_constructor_failure_preserves_existing_session_and_removes_new_directory(self):
        previous = self.session()
        before = (previous.path / "manifest.json").read_bytes()

        def fail_manifest(path, data):
            if path.name == "manifest.json":
                self.assertTrue((path.parent / "camera.json").is_file())
                raise OSError("disk_full")
            atomic_json(path, data)

        with patch("vision.capture_session.atomic_json", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "disk_full"):
                self.session()
        self.assertEqual(list(self.root.iterdir()), [previous.path])
        self.assertEqual((previous.path / "manifest.json").read_bytes(), before)

    def test_keyboard_interrupt_during_constructor_removes_only_new_session(self):
        previous = self.session()
        before = (previous.path / "manifest.json").read_bytes()

        def interrupt_manifest(path, data):
            if path.name == "manifest.json":
                self.assertTrue((path.parent / "camera.json").is_file())
                raise KeyboardInterrupt()
            atomic_json(path, data)

        with patch("vision.capture_session.atomic_json", side_effect=interrupt_manifest):
            with self.assertRaises(KeyboardInterrupt):
                self.session()
        self.assertEqual(list(self.root.iterdir()), [previous.path])
        self.assertEqual((previous.path / "manifest.json").read_bytes(), before)

    def test_keyboard_interrupt_before_manifest_removes_only_new_PNG(self):
        session = self.session()
        first = self.save(session)
        before = (session.path / "manifest.json").read_bytes()
        with patch("vision.capture_session.atomic_json", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.save(session, 2, self.image + 1)
        self.assertEqual((session.path / "manifest.json").read_bytes(), before)
        self.assertTrue((session.path / first["image"]).is_file())
        self.assertFalse((session.path / "images/frame_000002.png").exists())
        self.assertIsNotNone(self.save(session, 2, self.image + 1))

    def test_conflicting_batch_published_during_creation_is_preserved_and_rejected(self):
        other = self.root / "concurrent_session"

        def publish_other_session(path, data):
            atomic_json(path, data)
            if path.name == "manifest.json":
                other.mkdir()
                atomic_json(other / "manifest.json", dict(data, split="test"))

        with patch("vision.capture_session.atomic_json", side_effect=publish_other_session):
            with self.assertRaisesRegex(ValueError, "cross_dataset_splits"):
                self.session()
        self.assertEqual(list(self.root.iterdir()), [other])
        self.assertEqual(json.loads((other / "manifest.json").read_text())["split"], "test")

    def test_existing_image_is_never_overwritten_or_removed(self):
        session = self.session()
        orphan = session.path / "images/frame_000001.png"
        orphan.write_bytes(b"existing_orphan")
        with self.assertRaises(FileExistsError):
            self.save(session)
        self.assertEqual(orphan.read_bytes(), b"existing_orphan")
        self.assertEqual(self.manifest(session)["frames"], [])

    def test_wrong_image_shape_and_non_UTC_metadata_are_rejected(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, "resolution"):
            self.save(session, image=self.image[:3])
        with self.assertRaisesRegex(ValueError, "BGR"):
            self.save(session, image=self.image[:, :, 0])
        with self.assertRaisesRegex(ValueError, "UTC"):
            session.save(self.image, 1, 1., "hardware_clock", "2026-10-28T09:00:00+08:00")
        self.assertEqual(self.manifest(session)["frames"], [])
        session.finish("failed", "capture_error")
        self.assertEqual(self.manifest(session)["error"], "capture_error")


if __name__ == "__main__":
    unittest.main()
