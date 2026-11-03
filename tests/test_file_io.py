"""Check persistence guarantees independently of review and training callers."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ev_charging import file_io


class FileIOTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "record.json"
        self.path.write_bytes(b'{"original": true}')

    def test_atomic_json_rejects_nonfinite_values_without_losing_original(self):
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            file_io.atomic_json(self.path, {"invalid": float("nan")})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_replace_failure_preserves_original_and_cleans_temporary(self):
        original = self.path.read_bytes()
        with patch.object(file_io.os, "replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                file_io.atomic_json(self.path, {"new": True})
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_capture_retries_only_supported_windows_errors(self):
        sharing_error = PermissionError("sharing violation")
        sharing_error.winerror = 32
        replace = file_io.os.replace
        with patch.object(file_io.os, "replace") as mocked:

            def attempt(source, destination):
                if mocked.call_count == 1:
                    raise sharing_error
                return replace(source, destination)

            mocked.side_effect = attempt
            with patch.object(file_io.time, "sleep") as sleep:
                file_io.atomic_capture_json(self.path, {"new": True}, retry_delays=(0.01,))
        self.assertEqual(json.loads(self.path.read_bytes()), {"new": True})
        self.assertEqual(mocked.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_sha256_hashes_original_bytes_for_path_and_string(self):
        content = b"first\r\nsecond\n\x00\xff"
        self.path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(file_io.sha256_file(self.path), expected)
        self.assertEqual(file_io.sha256_file(str(self.path)), expected)

    def test_other_os_errors_are_not_retried(self):
        original = self.path.read_bytes()
        with patch.object(file_io.os, "replace", side_effect=OSError("disk error")) as replace:
            with patch.object(file_io.time, "sleep") as sleep:
                with self.assertRaises(OSError):
                    file_io.atomic_capture_json(self.path, {"new": True}, retry_delays=(0.01,))
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(self.path.read_bytes(), original)

    def test_fallback_requires_explicit_opt_in(self):
        error = PermissionError("sharing violation")
        error.winerror = 32
        with patch.object(file_io.os, "replace", side_effect=error):
            with self.assertRaises(PermissionError):
                file_io.atomic_capture_json(self.path, {"new": True}, retry_delays=())
            self.assertEqual(json.loads(self.path.read_bytes()), {"original": True})
            file_io.atomic_capture_json(self.path, {"new": True}, retry_delays=(), in_place_fallback=True)
        self.assertEqual(json.loads(self.path.read_bytes()), {"new": True})


if __name__ == "__main__":
    unittest.main()
