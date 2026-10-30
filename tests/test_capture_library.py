"""Raw capture registration preserves provenance and survives library failures."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from vision import collect, preview, training_library
from vision.capture_session import CaptureSession


class CaptureLibraryTests(unittest.TestCase):
    def setUp(self):
        self.test_root = collect.ROOT / "artifacts/test-tmp"
        self.root = self.test_root / ("capture-library-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.library = self.root / "library"
        self.camera = {
            "serial": "synthetic-d415", "device_name": "Intel RealSense D415", "stream": "color",
            "image_width": 80, "image_height": 60, "validated": False,
            "camera_matrix": [[70., 0., 40.], [0., 70., 30.], [0., 0., 1.]],
            "dist_coeffs": [0.] * 5, "distortion_model": "brown_conrady",
            "source": "device_color_stream", "is_placeholder": False, "is_approximate": False,
        }
        self.session = CaptureSession(self.root / "captures", self.camera, "batch-one", metadata={
            "specimen_id": "printed-one", "source_kind": "printed_connector"})
        self.image = np.full((60, 80, 3), 91, dtype=np.uint8)
        self.frame = {"frame_number": 3, "device_timestamp_ms": 99.,
                      "timestamp_domain": "hardware_clock", "received_at_utc": "2026-10-30T08:00:00+00:00"}

    def cleanup(self):
        if not self.root.resolve().is_relative_to(self.test_root.resolve()):
            raise ValueError("Test cleanup outside intended directory")
        shutil.rmtree(self.root)

    def test_registration_uses_only_saved_original_and_records_source(self):
        saved = self.session.save(self.image, **self.frame)
        original = (self.session.path / saved["image"]).read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertTrue(collect.register_saved_rgb(self.session.path, saved, self.library))
            self.assertTrue(collect.register_saved_rgb(self.session.path, saved, self.library))
        entries = training_library.read_entries(self.library)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "pending")
        self.assertEqual(entries[0]["date"], "2026-10-30")
        self.assertEqual(len(entries[0]["sources"]), 1)
        source = entries[0]["sources"][0]
        self.assertEqual(source["camera_serial"], "synthetic-d415")
        self.assertEqual(source["metadata"]["specimen_id"], "printed-one")
        self.assertEqual(source["session_manifest"], str(self.session.path / "manifest.json"))
        self.assertEqual(source["image"], saved["image"])
        self.assertEqual((self.session.path / saved["image"]).read_bytes(), original)
        self.assertEqual(list(self.library.rglob("*.txt")), [])

    def test_library_failure_keeps_committed_session_and_reports_pending_sync(self):
        saved = self.session.save(self.image, **self.frame)
        before = (self.session.path / "manifest.json").read_bytes()
        with (patch.object(training_library, "register_image", side_effect=OSError("disk full")),
              redirect_stdout(io.StringIO()) as output):
            self.assertFalse(collect.register_saved_rgb(self.session.path, saved, self.library))
        self.assertIn("待重新同步", output.getvalue())
        self.assertIn("disk full", output.getvalue())
        self.assertEqual((self.session.path / "manifest.json").read_bytes(), before)
        self.assertTrue((self.session.path / saved["image"]).is_file())
        self.session.finish()

    def test_disabled_registration_and_uncommitted_or_derived_image_never_import(self):
        saved = self.session.save(self.image, **self.frame)
        with patch.object(training_library, "register_image") as register, redirect_stdout(io.StringIO()):
            self.assertTrue(collect.register_saved_rgb(self.session.path, saved, None))
            self.assertFalse(collect.register_saved_rgb(
                self.session.path, {**saved, "image": "overlays/frame_000001.png"}, self.library))
            self.assertFalse(collect.register_saved_rgb(
                self.session.path, {**saved, "image_sha256": "0" * 64}, self.library))
            path = self.session.path / saved["image"]
            path.write_bytes(path.read_bytes() + b"changed")
            self.assertFalse(collect.register_saved_rgb(self.session.path, saved, self.library))
        register.assert_not_called()
        self.assertFalse(self.library.exists())

    def test_collect_save_automatically_registers_only_original_pixels(self):
        camera = SimpleNamespace(read=lambda: (self.image, self.frame))
        with ExitStack() as stack:
            for name in ("namedWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch.object(collect.cv2, name))
            stack.enter_context(patch.object(collect.cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch.object(collect.cv2, "waitKey", return_value=ord("s")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(collect.collect(camera, self.session, max_images=1, library_root=self.library), 1)
        entry = training_library.read_entries(self.library)[0]
        self.assertEqual(entry["status"], "pending")
        self.assertTrue(np.all(self.image == 91))

    def test_both_clis_default_custom_and_disabled_library(self):
        camera = SimpleNamespace(camera={"image_width": 80, "image_height": 60, "fps": 30}, rgb_controls=None)
        for flags, expected in (([], collect.ROOT / "dataset/vision"),
                                (["--library", str(self.library)], self.library),
                                (["--no-library"], None)):
            with self.subTest(entry="collect", flags=flags), ExitStack() as stack:
                factory = stack.enter_context(patch.object(collect, "CaptureColorCamera"))
                factory.return_value.__enter__.return_value = camera
                session = stack.enter_context(patch("vision.capture_session.CaptureSession"))
                run = stack.enter_context(patch.object(collect, "collect", return_value=0))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(collect.main(["--batch", "cli-test", *flags]), 0)
                self.assertEqual(run.call_args.kwargs["library_root"], expected)
                metadata = session.call_args.kwargs["metadata"]
                self.assertEqual(metadata["training_library_enabled"], expected is not None)
                self.assertEqual(metadata["training_library_path"], str(expected) if expected is not None else None)
            with self.subTest(entry="preview", flags=flags), ExitStack() as stack:
                factory = stack.enter_context(patch.object(preview, "CaptureColorCamera"))
                factory.return_value.__enter__.return_value = camera
                stack.enter_context(patch.object(preview, "DiagnosticFrontend"))
                session = stack.enter_context(patch.object(preview, "PreviewSession"))
                stack.enter_context(patch.object(preview, "PreviewCapture"))
                run = stack.enter_context(patch.object(preview, "run_preview", return_value={"saved_frames": 0}))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(preview.main(flags), 0)
                self.assertEqual(run.call_args.kwargs["library_root"], expected)
                metadata = session.call_args.kwargs["metadata"]
                self.assertEqual(metadata["training_library_enabled"], expected is not None)
                self.assertEqual(metadata["training_library_path"], str(expected) if expected is not None else None)


if __name__ == "__main__":
    unittest.main()
