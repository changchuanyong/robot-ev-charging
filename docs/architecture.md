# 工程结构

可复用源码统一位于 `src/ev_charging/`。工程根目录保存配置、实际数据和运行产物，数据处理代码位于包内的 `datasets/`，两者用途不同。

| 原位置 | 新位置（相对于 `src/ev_charging/`） |
| --- | --- |
| `vision/algorithms`、`vision/calibration`、`vision/evaluation`、`vision/locate` | 保留对应的 `vision/` 子包 |
| `vision/capture` | `capture/`，其中 `realsense.py` 迁入 `hardware/` |
| `vision/review` | `review/`，HTML 页面作为包资源一起分发 |
| `vision/library` | `datasets/vision/` |
| `control/hardware.py` | `hardware/realman.py` |
| `control/force_dataset.py` | `datasets/force.py` |
| `control` 其他模块 | `control/` |
| `train/python` 训练与拟合脚本 | `training/` |
| `train/python/runtime_env.py` | `runtime.py` |
| `vision/contracts.py` 的 JSON 写入函数 | `file_io.py`；观测数据协议仍在 `vision/contracts.py` |

设备适配模块在显式运行设备流程时连接硬件。采集与控制入口组合设备和算法；算法、数据管理与离线仿真可单独测试。导入设备模块不代表授权连接或运动。

`paths.py` 分别定位已安装代码和工程数据：可编辑安装默认使用其对应工程；wheel 安装可从工程目录启动，或通过进程环境变量 `EV_CHARGING_ROOT` 指定包含 `config/` 和 `dataset/` 的根目录。HTML 资源和源码哈希读取实际导入的包，配置、图库与权重读取工程数据目录。

数据、标注、模型配置、训练报告和权重保留原始位置与字节。新生成报告记录迁移后的源码路径与哈希；历史记录不改写。历史评估中绑定旧源码哈希的结果不能直接冒充当前版本的结果。

## 开发检查

在 `ev-charging` 环境、工程根目录运行：

```powershell
python -B -m unittest discover -s tests -v
python -B -m ev_charging.vision.check_project
python -B -m ev_charging.control.simulate --scenario nominal
```

`control.simulate` 使用合成数据检查控制软件。统一环境中的 MuJoCo 是运行依赖，本工程此次没有引入其他目录的仿真场景。
