# D435 同帧定位与眼在手标定

`vision.run` 在内存中顺序复用现有 YOLO、增强、Canny、孔位筛选、椭圆中心和 PnP 算法。每次调用只处理一张图像，不读取 `latest_roi.jpg` 等共享中间文件。旧视觉程序仍可独立使用；机器人应消费新的 `latest_control_pose.json`。

当前检测权重统一由 `config/vision_model.json` 指定，D435 的 `config/realsense_config.json` 与 D415 预览均引用这份记录。启动核对权重文件与 SHA256；训练完整成功后更新记录，已运行的进程需重启才能加载新模型。`config/observed_layout_prior.json` 根据人工复核的 train 图学习孔名关联，仅输出实际内孔口中心；PnP 仍使用原三维模型。共享检测权重不代表实际 D435 图像、几何或位姿已验收。

使用 D415 采集训练图片时，运行独立的 `python -m vision.collect`；D435 定位入口不消费这些采集清单，也不复用 D415 的标定参数。

## 启动及输出

在项目根目录运行：

```powershell
.venv\Scripts\python.exe -m vision.run --config config/realsense_config.json
```

首次接设备前，填写 D435 完整序列号。`intrinsics_validated` 初始为 `false`；可在获准采集后用 `--export-intrinsics artifacts/d435_color_intrinsics.json` 保存实际彩色流的参数和 `intrinsics_sha256`。完成该分辨率、设备和工作距离下的独立图像验证后，再填写校准版本、匹配的参数指纹并设置验证状态。改变序列号、分辨率、内参或畸变参数会使旧指纹失效。SDK 工厂内参不自动等同于任务标定已验收。

离线单张图像：

```powershell
.venv\Scripts\python.exe -m vision.run --image path/to/image.jpg --camera-config path/to/that_camera.json --output artifacts/offline_pose.json
```

离线结果始终标记 `source=offline`，不能作为真机运动输入。缺少 YOLO 权重、目标丢失、畸变模型不支持、图像尺寸不符、过期或未验证标定会产生明确无效结果。新消息以原子替换发布，失败不会保留上一帧的有效标记。实时采集退出后也会发布无效状态。

主要输出字段：

| 字段 | 含义 |
| --- | --- |
| `frame_id` | 采集进程随机 ID 与相机帧编号，跨重启不会重复 |
| `captured_at`、`received_at`、`processed_at` | 本机 monotonic 秒；跨设备时钟必须另外映射 |
| `camera_time_ms`、`timestamp_domain` | SDK 原始时间戳及定义域 |
| `time_sync_valid` | SDK global/system 时间已映射并通过时效检查；不代表零延迟 |
| `T_camera_target` | 目标坐标转换至 RGB 光学坐标的 4×4 变换，平移为 m |
| `quality.reprojection_rms_px` | 原始彩色图像像素中的全部匹配点 RMS，不与平均欧氏误差混用 |
| `quality.camera_calibrated` | 当前彩色流内参验证状态 |
| `calibration_version` | 内参版本；与手眼外参版本分别记录 |
| `quality.visual_budget` | `finite-trial-v2` 的四维视觉纠偏预算 `{x,y,Rx,Ry}`，单位 m、m、rad、rad；未标定为 null |
| `quality.geometry_reference` | 显式几何文件的版本、状态、验收标记及 SHA256；旧 CSV 为 null |

D435 的 modified/inverse Brown 与 OpenCV Brown 模型不直接混用：孔位先按 SDK 投影模型反求理想像素，再使用零畸变 PnP；最终残差重新投影回原始彩色图像计算。测试将这套投影与已安装 RealSense SDK 的投影结果比较。尚未实现的畸变模型直接拒绝。

全图仅允许一个候选完整接口。多接口场景应为每个目标配置独立 `target_id` 与 `target_window=[x1,y1,x2,y2]`（0–1 归一化）。窗口内仍出现多个候选时拒绝选择，避免仅因置信度改变就切换目标。接触阶段的目标冻结由控制状态机负责。

## 协同方法 F 的视觉预算

`realsense_config.json` 中的 `visual_budget` 默认是 `null`。要启用 F 组，需要独立位姿验证得到的配置对象，字段为：

```text
validated: true
rules_version: finite-trial-v2
axes: [x, y, Rx, Ry]
version: 预算校准版本
source: 独立验证记录的位置或批次
camera_intrinsics_sha256: 对应彩色流参数指纹
budget_m_rad: 四个正值，顺序为 x、y、Rx、Ry，单位为 m、m、rad、rad
depth_range_m: 验证覆盖的正深度范围 [min, max]
max_reprojection_rms_px: 验证域内允许的像素 RMS 上界
```

只有参数指纹、深度范围与残差范围都匹配，输出中才附带预算。程序不把像素误差直接换算为位置误差，也不把这些预算称为 PnP 统计协方差。实际预算如何由独立测量确定，需要随版本保留实验依据。

`geometry_reference` 可指向 `geometry_reference.template.json` 填写后的实测副本，优先于旧 `model_points` CSV；配置为标称或未验收几何时保留诊断位姿但不放行。正式实验控制器进一步核对该几何与当前样件冻结的版本和哈希。

## 手眼标定

```powershell
.venv\Scripts\python.exe -m vision.calibrate --samples artifacts/handeye_pairs.json --output artifacts/handeye_result.json
```

输入 JSON 顶层包含：

- `translation_unit="m"`、`source="measured"` 或 `"synthetic"`、`camera_serial`。
- `pairing_verified`：是否已核验采集时钟对应关系和静止采样条件。
- `max_pair_skew_s`：事先确定的配对时间偏差上限。
- `acceptance`：独立验证用的 `max_translation_m` 与 `max_rotation_rad`，未提供时只求解，不给出通过结论。
- `pairs`：每项含唯一 `id`、`split="train"` 或 `"validation"`、`camera_time`、`robot_time`、`T_base_flange`、`T_camera_board`。

变换将右侧坐标系中的点变换到左侧坐标系。至少需要三个非退化求解姿态及两个独立验证姿态；实际采集应覆盖工作距离、位置和多方向转动。同一机器人姿态不能重复计为独立观测，也不能跨集合重复使用。若机器人反馈 TCP，应先用独立工具外参恢复法兰位姿。

也可以用每项 `image` 路径替代 `T_camera_board`，同时在顶层提供该图像流的 `camera` 参数和 `board={columns, rows, square_size_m}`。行列是棋盘内部角点数，方格边长必须实测；`image` 路径相对样本 JSON 所在目录。

程序调用 OpenCV 4 的 Tsai 手眼求解，并仅由训练姿态建立固定板参考。验证姿态不参与求解或板参考拟合。输出 `T_flange_camera`、每次观测的独立残差及验收状态；合成样本即使解析测试通过也不会获得 `validated=true`。固定板一致性不自动证明外部绝对定位精度。

`T_flange_tool` 和 `T_tool_sensor` 继续为 `null`，需要独立测量，不能用手眼结果代替。`config/calibration.template.json` 展示这些独立项。OpenCV 5 当前 Python 包缺少 `calibrateHandEye`，本项目使用 `<5` 的依赖约束。

## 离线验证

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_vision.py -v
```

测试覆盖 SDK 畸变一致性、同帧 PnP、单位转换、过期与重复帧、无效覆盖、目标歧义、预算校准域、合成手眼恢复及独立验证失败。相机采集、手眼实测与机器人运动均不由这些测试执行。
