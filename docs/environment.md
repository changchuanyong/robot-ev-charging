# 统一开发环境

使用 Conda 环境 `ev-charging`，Python 3.12.16。它覆盖原 `ev-vision` 的视觉、CUDA 训练和设备 SDK，以及原 `ev-sim` 的 MuJoCo 依赖。

原环境保留用于回退，不再作为本工程文档的默认入口。此次未修改全局 PATH、Windows 安全策略或备份工程。

## 创建与安装

在工程根目录运行：

```powershell
conda env create --format environment-yaml -f environment.yml
conda activate ev-charging
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

`--no-deps` 保留锁定清单安装的版本，避免包安装时重新选择 CUDA 构建。`--no-build-isolation` 使用环境内已有的 setuptools 与 wheel；这些构建工具由 Conda 环境提供。修改源码不需要重复安装；修改包配置或新增命令入口后重新执行可编辑安装。

`pyproject.toml` 描述包的直接依赖范围和入口；`environments/pip.lock.txt` 固定整套业务依赖及 CUDA wheel 的官方地址、SHA256。Conda 管理 Python、Tk、pip 等基础组件。

## 精确重建 Windows 基础组件

`environment.yml` 适合通常的环境创建；如需还原本次 Conda 基础组件的具体构建，可执行：

```powershell
conda create -n ev-charging --file environments/win64.conda.lock.txt
conda activate ev-charging
python -m pip install -r environments/pip.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

显式 Conda 清单不包含 pip 业务包，两部分都需要安装。仅对不存在的目标环境执行创建命令。

## 合并取舍

- Python：两套原环境均为 3.12.16。
- NumPy：保留视觉环境的 2.4.3，替代原仿真环境中的 2.5.4。
- PyTorch / TorchVision：保留 2.14.1+cu132 / 0.29.1+cu132。
- MuJoCo：保留原仿真环境的 3.17.0，并补齐其依赖。
- 本次不升级相机或机械臂 SDK。

## 检查

```powershell
python -m pip check
python -B -m unittest discover -s tests -v
python -B -m ev_charging.vision.check_project
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

测试中的 MuJoCo 场景仅验证引擎与统一 NumPy/PyTorch 环境共存；它不验证充电插装场景、实物传感器或硬件控制效果。

## 本次迁移验证（2026-11-04）

- 6 项包结构回归测试通过，涵盖模块导入、命令入口、复核页面、工作目录切换与 MuJoCo/PyTorch 共存。
- 工程完整性检查通过；原有相机参数来源缺失警告仍保留。
- 13 个控制软件仿真场景完成，日志回放均复现原决策。
- RTX 5080 上的 CUDA 前向/反向计算、当前检测器对合成图像的推理通过。
- MuJoCo 物理步进及 64×64 离屏渲染通过。
- 普通 wheel 构建及独立解包后的资源、自检通过，源码指纹与可编辑安装一致。
- 2,763 个配置、数据及训练结果文件与迁移前的 SHA256 一致。

详细结果保存在本机 `artifacts/structure_migration_20261104/verification.json` 和 `wheel_verification.json`。未连接相机或机器人，未执行正式训练或改写当前模型。
