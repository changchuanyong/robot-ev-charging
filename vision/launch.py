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
import runpy
import subprocess
import sys

LAUNCHER = "vision.launch"
SW_SHOWNORMAL = 1
MODULES = ("vision.preview", "vision.collect", "vision.run")
USAGE = ("用法: python -m vision.launch <相机模块> [参数...]\n"
         "支持: " + ", ".join(MODULES) + "\n"
         "例如: python -m vision.launch vision.preview --depth\n"
         "--help 与 vision.run --image 离线处理不申请管理员权限。")


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
        from .win_privilege import ensure_camera_privilege
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
    raise SystemExit(main())
