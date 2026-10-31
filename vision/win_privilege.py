"""Enable SeSecurityPrivilege for librealsense's global WinUSB mutex (Windows only).

librealsense 2.59.1 creates ``Global\\IVCAM_DLL_WINUSB_MUTEX<usb-instance>`` with a security
descriptor that carries SE_SACL_PRESENT but an empty SACL. Creating a kernel object that
has an SACL requires SeSecurityPrivilege, so a normal token gets
ERROR_PRIVILEGE_NOT_HELD (1314) and the SDK reports

    RuntimeError: CreateNamedMutex returned Mutex_TotalFailure

Enabling the privilege inside this process makes the device usable again. Nothing is
persisted: no user right is granted, no security policy is edited, and the privilege
disappears when the process exits.

Being an administrator is not the same as having the privilege enabled, so the code
always calls AdjustTokenPrivileges and inspects GetLastError instead of inferring
anything from the token's group membership.

Every entry point returns immediately on non-Windows platforms, so ROS/Ubuntu runs are
untouched by this module.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from types import ModuleType

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
