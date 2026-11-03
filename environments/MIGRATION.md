# Conda 迁移记录

本项目默认环境为 `ev-vision`，由 `C:\Users\admin\anaconda3` 管理，Python 为 3.12.16。原有 43 个业务包的版本全部保留，CUDA PyTorch 仍为 `2.14.1+cu132`；版本清单见 `vision-pip.lock.txt`，Conda 基础包见 `ev-vision-win64.conda.lock.txt`。两份项目副本共用依赖环境，各自仍使用自己的代码和数据。

依赖检查通过；视觉测试 26 项中有 25 项在正常用户权限下通过，包括复核页的 HTTP 读写与冲突处理。1 项在导入 PyTorch 时被 Windows 应用控制策略 `VerifiedAndReputableDesktop` 阻止，报 `WinError 4551`。旧 `.venv` 导入 PyTorch 也受该策略阻止；新旧 CUDA DLL 的哈希相同。GPU 推理和训练尚未完成验收，需要系统管理方处理策略拦截或提供获准的签名构建后继续验证。

没有修改全局 PATH、Python 注册或 Windows 安全策略。旧 `.venv` 未修改、未删除。通过 `run_vision.cmd` 运行模块，或双击 `start_keypoint_review.cmd` 启动复核页；自动候选依赖 PyTorch，当前可能受上述拦截影响。

策略问题解决后运行：

```powershell
.\run_vision.cmd -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\run_vision.cmd -m unittest discover -s train/tests -p "test_*.py" -v
```

Conda 环境创建时 pip 的网络请求曾遇到 TLS 中断。本次通过已有 uv 向指定 Conda 解释器补齐锁定包；PyTorch wheel 从官方地址分段下载并通过原 SHA256 校验后安装。环境文件保留官方依赖来源，没有另建虚拟环境。
