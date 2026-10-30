from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import call, patch
import uuid

import numpy as np

from vision.capture_files import atomic_capture_json
from vision.capture_session import CaptureSession
from vision.contracts import atomic_json
from vision.preview_session import PreviewSession


ROOT = Path(__file__).resolve().parents[1]
DELAYS = [.025, .05, .1, .2, .4]


def sharing_error(code=5):
    error = PermissionError("Windows replacement refused")
    error.winerror = code
    return error


class CaptureFileRetryTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "artifacts/test-tmp" / ("capture-retry-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        self.camera = {
            "serial": "retry-test", "device_name": "RealSense D415", "stream": "color",
            "image_width": 8, "image_height": 6,
            "camera_matrix": [[8., 0., 4.], [0., 8., 3.], [0., 0., 1.]],
            "dist_coeffs": [0.] * 5, "distortion_model": "none", "validated": False,
        }
        self.image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.geometry = {"version": "nominal-test", "validated": False}

    def cleanup(self):
        if not self.root.resolve().is_relative_to((ROOT / "artifacts/test-tmp").resolve()):
            raise ValueError("test_directory_outside_workspace_artifacts")
        shutil.rmtree(self.root)

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def save_capture(self, session, number):
        return session.save(self.image + number, number, 1000. + number, "hardware_clock",
                            "2026-10-30T01:02:03+00:00")

    def save_preview(self, session, number):
        image = self.image + number
        diagnostic = {
            "source": "live_diagnostic", "authorizes_robot_motion": False,
            "image_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "camera": self.camera, "geometry": self.geometry,
            "frame": {"frame_number": number, "device_timestamp_ms": 1000. + number,
                      "timestamp_domain": "hardware_clock", "received_at_utc": "2026-10-30T01:02:03+00:00"},
        }
        return session.save(image, diagnostic, image)

    def test_transient_refusal_preserves_old_json_then_publishes_once(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                path = self.root / f"metadata-{code}.json"
                atomic_json(path, {"old": True})
                original = path.read_bytes()
                replace = os.replace
                attempts = []

                def temporarily_locked(source, destination):
                    attempts.append(source)
                    self.assertEqual(path.read_bytes(), original)
                    if len(attempts) < 3:
                        raise sharing_error(code)
                    replace(source, destination)

                with patch("vision.contracts.os.replace", side_effect=temporarily_locked), \
                        patch("vision.capture_files.time.sleep") as sleep:
                    atomic_capture_json(path, {"new": True})
                self.assertEqual(self.read(path), {"new": True})
                self.assertEqual(len(attempts), 3)
                self.assertEqual(sleep.call_args_list, [call(.025), call(.05)])
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_preview_retries_each_manifest_without_duplicate_sample_or_stopping(self):
        for target in ("manifest.json", "preview_manifest.json"):
            with self.subTest(target=target):
                session = PreviewSession(self.root, self.camera, "retry-test",
                                         pipeline_config={}, geometry=self.geometry)
                replace = os.replace
                attempts = []
                original = (session.path / target).read_bytes()

                def temporarily_locked(source, destination):
                    if Path(destination).name == target:
                        attempts.append(source)
                        self.assertEqual((session.path / target).read_bytes(), original)
                        if len(attempts) < 3:
                            raise sharing_error()
                    replace(source, destination)

                with patch("vision.contracts.os.replace", side_effect=temporarily_locked), \
                        patch("vision.capture_files.time.sleep") as sleep:
                    first = self.save_preview(session, 1)
                self.assertEqual(len(attempts), 3)
                self.assertEqual(sleep.call_args_list, [call(.025), call(.05)])
                preview = self.read(session.path / "preview_manifest.json")
                self.assertEqual(preview["status"], "recording")
                self.assertEqual(preview["samples"], [first])
                self.assertEqual(len(list((session.path / "images").iterdir())), 1)
                second = self.save_preview(session, 2)
                self.assertEqual(session.saved_count, 2)
                self.assertEqual(self.read(session.path / "preview_manifest.json")["samples"], [first, second])
                self.assertEqual(list(session.path.rglob("*.tmp")), [])

    def test_persistent_capture_refusal_preserves_manifest_and_earlier_image(self):
        session = CaptureSession(self.root, self.camera, "retry-test")
        first = self.save_capture(session, 1)
        original_manifest = (session.path / "manifest.json").read_bytes()
        original_image = (session.path / first["image"]).read_bytes()
        with patch("vision.contracts.os.replace", side_effect=sharing_error()) as replace, \
                patch("vision.capture_files.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                self.save_capture(session, 2)
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(sleep.call_args_list, [call(delay) for delay in DELAYS])
        self.assertEqual((session.path / "manifest.json").read_bytes(), original_manifest)
        self.assertEqual((session.path / first["image"]).read_bytes(), original_image)
        self.assertEqual(list((session.path / "images").iterdir()), [session.path / first["image"]])
        self.assertEqual(list(session.path.glob("*.tmp")), [])
        self.assertIsNotNone(self.save_capture(session, 2))

    def test_persistent_preview_refusal_keeps_earlier_complete_sample(self):
        session = PreviewSession(self.root, self.camera, "retry-test",
                                 pipeline_config={}, geometry=self.geometry)
        first = self.save_preview(session, 1)
        original = (session.path / "preview_manifest.json").read_bytes()
        files = {key: (session.path / first[key]).read_bytes() for key in ("image", "overlay", "diagnostic")}
        replace = os.replace
        refusals = []

        def locked_preview(source, destination):
            if Path(destination).name == "preview_manifest.json":
                refusals.append(source)
                raise sharing_error()
            replace(source, destination)

        with patch("vision.contracts.os.replace", side_effect=locked_preview), \
                patch("vision.capture_files.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                self.save_preview(session, 2)
        # Publishing the sample and persisting failure status are separately bounded.
        self.assertEqual(len(refusals), 12)
        self.assertEqual(sleep.call_args_list, [call(delay) for delay in DELAYS] * 2)
        self.assertEqual((session.path / "preview_manifest.json").read_bytes(), original)
        self.assertEqual(session.saved_count, 1)
        self.assertEqual(list(session.path.rglob("*.tmp")), [])
        for key, content in files.items():
            self.assertEqual((session.path / first[key]).read_bytes(), content)
        with self.assertRaisesRegex(RuntimeError, "already_finished"):
            self.save_preview(session, 3)
        session.finish()
        preview = self.read(session.path / "preview_manifest.json")
        self.assertEqual(preview["status"], "failed")
        self.assertEqual(preview["samples"], [first])

    def test_unrelated_errors_and_interrupts_do_not_retry(self):
        for error in (PermissionError("not Windows"), sharing_error(19),
                      OSError("disk full"), KeyboardInterrupt()):
            with self.subTest(error=repr(error)):
                path = self.root / "unrelated.json"
                atomic_json(path, {"old": True})
                with patch("vision.contracts.os.replace", side_effect=error) as replace, \
                        patch("vision.capture_files.time.sleep") as sleep:
                    with self.assertRaises(type(error)):
                        atomic_capture_json(path, {"new": True})
                self.assertEqual(replace.call_count, 1)
                sleep.assert_not_called()
                self.assertEqual(self.read(path), {"old": True})
                self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_control_atomic_write_still_fails_immediately(self):
        path = self.root / "control.json"
        atomic_json(path, {"old": True})
        with patch("vision.contracts.os.replace", side_effect=sharing_error()) as replace, \
                patch("vision.capture_files.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                atomic_json(path, {"new": True})
        self.assertEqual(replace.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(self.read(path), {"old": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing semantics")
    def test_real_windows_handle_without_delete_sharing_recovers_when_released(self):
        from ctypes import wintypes

        path = self.root / "windows-handle.json"
        atomic_json(path, {"old": True})
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        # Read/write sharing deliberately excludes FILE_SHARE_DELETE on this test file.
        handle = kernel32.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

        def release(_delay):
            nonlocal handle
            self.assertEqual(self.read(path), {"old": True})
            if not kernel32.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            handle = None

        try:
            with patch("vision.capture_files.time.sleep", side_effect=release) as sleep, \
                    patch("vision.contracts.os.replace", wraps=os.replace) as replace:
                atomic_capture_json(path, {"new": True})
            self.assertEqual(replace.call_count, 2)
            sleep.assert_called_once_with(.025)
            self.assertEqual(self.read(path), {"new": True})
            self.assertEqual(list(self.root.glob("*.tmp")), [])
        finally:
            if handle is not None:
                kernel32.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
