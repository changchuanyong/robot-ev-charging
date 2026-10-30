# Windows 下 D415 打不开：SeSecurityPrivilege 与 SDK 的全局互斥体

## 现象

`python -m vision.preview --depth` 在设备加载完之后直接失败：

```
检测模型: runs\detect_split_...\weights\best.pt
RuntimeError: CreateNamedMutex returned Mutex_TotalFailure
```

此时相机在设备管理器里是正常的（`Intel(R) RealSense(TM) Depth Camera 415  Depth/RGB`，驱动 `usbvideo.inf`，状态 Started），YOLO 与 CUDA 也完全正常，`python -c "import torch; torch.cuda.is_available()"` 返回 True。所以这不是相机、显卡或虚拟环境的问题。

## 原因

librealsense 2.59.1 在创建设备前会建一个按 USB 实例编号命名的全局互斥体：

```
Global\IVCAM_DLL_WINUSB_MUTEX<usb-instance>
```

创建时传入的安全描述符是 `D:(A;;GA;;;WD)(A;;GA;;;AN)S:`。末尾的 `S:` 让描述符带上 `SE_SACL_PRESENT` 标志（SACL 本身是空的）。给内核对象设置 SACL 需要 `SeSecurityPrivilege`，普通令牌没有该特权，`CreateMutex` 返回 `ERROR_PRIVILEGE_NOT_HELD`（1314），SDK 把这一失败笼统地报成 `CreateNamedMutex returned Mutex_TotalFailure`。

同一进程内的对照测试可以确认这一点：

| 安全描述符 | 控制标志 | `CreateMutex` 结果 |
|---|---|---|
| 带 `SE_SACL_PRESENT`、SACL 为空 | `0x001c` | NULL，错误 1314 |
| 不带 SACL | `0x000c` | 成功 |

## 注意：管理员身份不等于特权已启用

即使令牌里存在 `SeSecurityPrivilege`，它默认也是 **Disabled**。判断依据只能是 `AdjustTokenPrivileges` 的返回值与随后 `GetLastError()` 是否为 1300，不能用"是不是管理员"来推断。`vision/win_privilege.py` 只检查后者。

## 两种启动方式

1. 普通 PowerShell，走 UAC 提权：

   ```powershell
   .\.venv\Scripts\python.exe -m vision.launch vision.preview --depth
   ```

   启动器用 `subprocess.list2cmdline` 组装参数、经 `ShellExecuteW(..., "runas", ...)` 重新拉起自身，再在进程内启用特权；参数原样透传，路径含空格或中文也不会被拆坏。

2. 已经是管理员终端时，直接按原来的方式运行即可，`vision.*` 的相机入口会在首次访问设备前自行启用：

   ```powershell
   .\.venv\Scripts\python.exe -m vision.preview --depth
   ```

两种方式都**不修改** `secpol.msc`、组策略或账户权限分配；特权只存在于当前进程，退出即失效。

## 自检

```powershell
whoami /priv | findstr /i SeSecurity
```

能看到条目只代表特权在令牌里；是否处于 Enabled 要以程序的实际调用结果为准。

## 安全边界

UAC 自提权会让**整个 Python 进程**（含 YOLO 推理与全部依赖）运行在管理员上下文里，不只是 RealSense SDK。该启动器只用于本仓库受信任的采集/预览代码，不要用它运行来路不明的脚本。

## 什么时候可以删掉这段代码

升级 pyrealsense2 之后，先在普通 PowerShell 里直接跑一次 `python -m vision.preview --depth`：

- 能正常出图 → 新版 SDK 已不再带这个 SACL，可以移除 `vision/launch.py`、`vision/win_privilege.py` 以及各入口的 `ensure_camera_privilege()` 调用；
- 仍报 `Mutex_TotalFailure` → 保留。
