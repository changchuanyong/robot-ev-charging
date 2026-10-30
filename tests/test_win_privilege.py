from __future__ import annotations

import ctypes
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

from vision import launch, win_privilege


class FakeApi:
    """Stands in for WindowsPrivilegeApi so no real token is touched."""

    def __init__(self, error: int = 0):
        self.error = error
        self.calls = 0

    def enable(self, name: str = win_privilege.PRIVILEGE) -> int:
        self.calls += 1
        return self.error


_SHELL32 = ctypes.WinDLL("shell32", use_last_error=True)
_SHELL32.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
_SHELL32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_KERNEL32.LocalFree.argtypes = [ctypes.c_void_p]
_KERNEL32.LocalFree.restype = ctypes.c_void_p


def split_windows_command_line(text: str) -> list[str]:
    argc = ctypes.c_int(0)
    argv = _SHELL32.CommandLineToArgvW(ctypes.c_wchar_p(text), ctypes.byref(argc))
    if not argv:
        raise OSError("CommandLineToArgvW failed")
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        _KERNEL32.LocalFree(ctypes.cast(argv, ctypes.c_void_p))


class NonWindowsTests(unittest.TestCase):
    """Ubuntu runs must not touch advapi32 at all."""

    def test_ensure_is_a_noop(self):
        api = FakeApi()
        with patch.object(sys, "platform", "linux"), patch.object(win_privilege, "_enabled", False):
            win_privilege.ensure_camera_privilege(api)
        self.assertEqual(api.calls, 0)

    def test_enable_reports_false(self):
        with patch.object(sys, "platform", "linux"), patch.object(win_privilege, "_enabled", False):
            self.assertFalse(win_privilege.enable_security_privilege(FakeApi()))


class PrivilegeTests(unittest.TestCase):
    def test_success_enables_once(self):
        api = FakeApi(0)
        with patch.object(sys, "platform", "win32"), patch.object(win_privilege, "_enabled", False):
            win_privilege.ensure_camera_privilege(api)
            win_privilege.ensure_camera_privilege(api)
            self.assertTrue(win_privilege._enabled)
        self.assertEqual(api.calls, 1)

    def test_not_assigned_is_actionable(self):
        api = FakeApi(win_privilege.ERROR_NOT_ALL_ASSIGNED)
        with patch.object(sys, "platform", "win32"), patch.object(win_privilege, "_enabled", False):
            with self.assertRaises(RuntimeError) as caught:
                win_privilege.ensure_camera_privilege(api)
        message = str(caught.exception)
        self.assertIn("1300", message)
        self.assertIn("vision.launch", message)
        self.assertFalse(win_privilege._enabled)

    def test_other_error_code_is_reported(self):
        api = FakeApi(5)
        with patch.object(sys, "platform", "win32"), patch.object(win_privilege, "_enabled", False):
            with self.assertRaises(RuntimeError) as caught:
                win_privilege.ensure_camera_privilege(api)
        self.assertIn("5", str(caught.exception))

    def test_test_double_sdk_is_skipped(self):
        """Tests inject MagicMock as pyrealsense2; those must never touch a real token."""
        api = FakeApi(win_privilege.ERROR_NOT_ALL_ASSIGNED)
        with patch.object(sys, "platform", "win32"), patch.object(win_privilege, "_enabled", False):
            win_privilege.ensure_camera_privilege(api, sdk=MagicMock())
        self.assertEqual(api.calls, 0)


class LauncherTests(unittest.TestCase):
    def test_no_arguments_prints_usage(self):
        stderr = io.StringIO()
        with patch.object(sys, "stderr", stderr):
            self.assertEqual(launch.main([]), 2)
        self.assertIn("vision.launch", stderr.getvalue())

    def test_non_windows_runs_the_module_with_its_arguments(self):
        seen = {}

        def fake_run_module(name, run_name=None, alter_sys=False):
            seen["name"] = name
            seen["argv"] = list(sys.argv)

        with patch.object(sys, "platform", "linux"), \
                patch.object(launch.runpy, "run_module", fake_run_module):
            self.assertEqual(launch.main(["vision.preview", "--depth", "--notes", "有 空格"]), 0)
        self.assertEqual(seen["name"], "vision.preview")
        self.assertEqual(seen["argv"], ["vision.preview", "--depth", "--notes", "有 空格"])


@unittest.skipUnless(sys.platform == "win32", "CommandLineToArgvW is Windows only")
class ArgumentForwardingTests(unittest.TestCase):
    """UAC re-launch must survive spaces and non-ASCII in paths and arguments."""

    def test_self_command_round_trips(self):
        executable = r"C:\Users\admin\Desktop\doc\毕业设 计\.venv\Scripts\python.exe"
        argv = ["vision.preview", "--depth", "--notes", "有 空格 的备注"]
        command = launch._self_command(argv, executable=executable)
        self.assertEqual(split_windows_command_line(command),
                         [executable, "-m", "vision.launch", *argv])

    def test_elevated_parameters_carry_every_argument(self):
        argv = ["vision.preview", "--depth", "--output-root", r"C:\tmp\my captures"]
        self.assertEqual(split_windows_command_line(launch._parameters(argv)),
                         ["-m", "vision.launch", *argv])


if __name__ == "__main__":
    unittest.main()
