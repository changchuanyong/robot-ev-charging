"""Run a vision entry point with SeSecurityPrivilege enabled (Windows).

Usage:
    .venv\\Scripts\\python.exe -m vision.launch vision.preview --depth

The launcher re-launches itself through the UAC prompt when the current token is not
elevated, enables the privilege inside the new process, and then runs the requested
module exactly as ``python -m <module> <args...>`` would. Nothing is stored: the privilege
lives only for that process. See README.md for the background.

Using ``.venv\\Scripts\\python.exe`` directly also sidesteps the PowerShell execution-policy
issue with Activate.ps1, which is unrelated to the camera privilege.
"""
from __future__ import annotations

import os
import ctypes
from ctypes import wintypes
import runpy
import subprocess
import sys
from types import ModuleType

LAUNCHER = "vision.launch"
SW_SHOWNORMAL = 1
MODULES = ("vision.preview", "vision.collect", "vision.run")
USAGE = ("用法: python -m vision.launch <相机模块> [参数...]\n"
         "支持: " + ", ".join(MODULES) + "\n"
         "例如: python -m vision.launch vision.preview --depth\n"
         "--help 与 vision.run --image 离线处理不申请管理员权限。")


TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY = 0x0008
SE_PRIVILEGE_ENABLED = 0x0002
ERROR_NOT_ALL_ASSIGNED = 1300
PRIVILEGE = "SeSecurityPrivilege"

HINT = (r"请用 .venv\Scripts\python.exe -m vision.launch <相机模块> [原参数] 重新启动"
        r"（相机模块为 vision.preview、vision.collect 或 vision.run），"
        r"或先打开管理员终端；两者都不修改系统安全策略。")

_enabled = False
_api = None


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class LUIDAndAttributes(ctypes.Structure):
    _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]


class TokenPrivileges(ctypes.Structure):
    _fields_ = [("PrivilegeCount", wintypes.DWORD),
                ("Privileges", LUIDAndAttributes * 1)]


class WindowsPrivilegeApi:
    """Thin binding over the three advapi32 calls needed to enable one privilege."""

    def __init__(self) -> None:
        self.advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentProcess.argtypes = []
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                 ctypes.POINTER(wintypes.HANDLE)]
        self.advapi.OpenProcessToken.restype = wintypes.BOOL
        self.advapi.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR,
                                                      ctypes.POINTER(LUID)]
        self.advapi.LookupPrivilegeValueW.restype = wintypes.BOOL
        self.advapi.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL,
                                                      ctypes.POINTER(TokenPrivileges),
                                                      wintypes.DWORD, ctypes.c_void_p,
                                                      ctypes.c_void_p]
        self.advapi.AdjustTokenPrivileges.restype = wintypes.BOOL

    def enable(self, name: str = PRIVILEGE) -> int:
        """Return 0 when the privilege is enabled, otherwise the Win32 error code."""
        token = wintypes.HANDLE()
        if not self.advapi.OpenProcessToken(self.kernel.GetCurrentProcess(),
                                            TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                                            ctypes.byref(token)):
            return ctypes.get_last_error() or 1
        try:
            luid = LUID()
            if not self.advapi.LookupPrivilegeValueW(None, name, ctypes.byref(luid)):
                return ctypes.get_last_error() or 1
            privileges = TokenPrivileges()
            privileges.PrivilegeCount = 1
            privileges.Privileges[0].Luid = luid
            privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
            ctypes.set_last_error(0)
            ok = self.advapi.AdjustTokenPrivileges(token, False, ctypes.byref(privileges),
                                                   0, None, None)
            error = ctypes.get_last_error()
            if not ok:
                return error or 1
            # AdjustTokenPrivileges succeeds even when the token lacks the privilege.
            return error
        finally:
            self.kernel.CloseHandle(token)


def _load_api():
    global _api
    if _api is None:
        _api = WindowsPrivilegeApi()
    return _api


def _message(error: int) -> str:
    if error == ERROR_NOT_ALL_ASSIGNED:
        return f"当前令牌未分配 {PRIVILEGE}（Win32 错误码 1300，普通 PowerShell 常见）。{HINT}"
    return f"启用 {PRIVILEGE} 失败，Win32 错误码 {error}。{HINT}"


def enable_security_privilege(apis=None) -> bool:
    """True when the privilege is active in this process; False elsewhere or on failure."""
    try:
        ensure_camera_privilege(apis)
    except RuntimeError:
        return False
    return sys.platform == "win32"


def _is_fake_sdk(sdk) -> bool:
    """Tests inject MagicMock/SimpleNamespace as pyrealsense2; those never need privileges."""
    return sdk is not None and not isinstance(sdk, ModuleType)


def ensure_camera_privilege(apis=None, sdk=None) -> None:
    """Enable SeSecurityPrivilege before the first RealSense device access.

    No-op on non-Windows platforms, and no-op when ``sdk`` is a test double. Raises
    RuntimeError with an actionable message when the privilege cannot be enabled, so
    callers report the real cause instead of letting the SDK fail with
    "CreateNamedMutex returned Mutex_TotalFailure".
    """
    global _enabled
    if sys.platform != "win32":
        return
    if _is_fake_sdk(sdk):
        return
    if _enabled:
        return
    error = (apis or _load_api()).enable(PRIVILEGE)
    if error != 0:
        raise RuntimeError(_message(error))
    _enabled = True


def _parameters(argv) -> str:
    """switches passed to the elevated child, i.e. everything after python.exe."""
    return subprocess.list2cmdline(["-m", LAUNCHER, *argv])


def _self_command(argv, executable=None) -> str:
    """Complete command line of the elevated child, quoted for ShellExecuteW."""
    return subprocess.list2cmdline([executable or sys.executable, "-m", LAUNCHER, *argv])


def is_elevated() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except OSError:
        return False


def _relaunch(argv) -> int:
    import ctypes

    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, _parameters(argv),
                                                 os.getcwd(), SW_SHOWNORMAL)
    if result <= 32:
        print(f"管理员相机进程未启动（ShellExecute 错误码 {result}）；请检查 UAC 授权。",
              file=sys.stderr)
        return 1
    print("已请求在管理员窗口启动；相机是否启动成功请查看新窗口。")
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(USAGE, file=sys.stderr)
        return 2
    if args[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    module, passthrough = args[0], args[1:]
    if module not in MODULES:
        print(f"不支持的相机模块: {module}\n{USAGE}", file=sys.stderr)
        return 2
    offline = any(arg in ("-h", "--help") for arg in passthrough) or (
        module == "vision.run" and any(
            arg == "--image" or arg.startswith("--image=") for arg in passthrough))
    if not offline and sys.platform == "win32" and not is_elevated():
        return _relaunch(args)

    if not offline:
        try:
            ensure_camera_privilege()
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    original_argv = sys.argv
    sys.argv = [module, *passthrough]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    sys.modules[LAUNCHER] = sys.modules[__name__]
    raise SystemExit(main())
