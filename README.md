# robot-ev-charging

基于 **RealSense、YOLOv8n 和 RM75 六维力机械臂**的充电口检测、九孔定位与插装研究项目。

**D415 采集 → 人工复核 → 模型训练 → 九孔 / PnP 诊断 → D435 定位与控制**

[快速开始](#快速开始) · [设备与配置](#设备与配置) · [目录结构](#目录结构) · [进阶用法](#进阶用法) · [项目约定](#项目约定)

## 快速开始

环境：**Windows · Python 3.12 · Conda**。以下命令均在项目根目录运行。

```powershell
conda activate ev-vision
```

<details>
<summary>首次使用：创建环境</summary>

在 Anaconda Prompt 中运行：

```powershell
conda env create --format environment-yaml -f environment-vision.yml
conda activate ev-vision
```

环境定义见 [environment-vision.yml](environment-vision.yml)，依赖版本见 [vision-pip.lock.txt](environments/vision-pip.lock.txt)。

</details>

按顺序操作，退出当前采集或复核窗口后再进入下一步：

| 步骤 | 命令 | 完成标志 |
| --- | --- | --- |
| ① 预览与拍照 | `python -m vision.launch vision.capture.preview --depth` | F5 保存，Esc 退出；查看本批摘要 |
| ② 复核检测框 | `python -m vision.review.review_library` | 人工确认待标注图片 |
| ③ 训练检测器 | `python train/python/train_port.py` | 训练完成，发布当前模型 |
| ④ 离线自检 | `python -m vision.check_project` | 查看错误、警告和数据待办 |

- 需要标注九孔中心时，运行 `python -m vision.review.review_keypoints`；孔位标注与检测框标签分别保存。
- Windows 相机入口通过 `vision.launch` 申请进程权限，可能弹出 UAC；是否启动成功以新窗口为准。
- 自检不连接相机或机器人；程序默认不驱动机器人。

### 采集快捷键

| 按键 | 操作 |
| --- | --- |
| **F5** / 空格 / S | 保存原图并入库；深度模式同时保存原始深度 |
| **F6** / P | 暂停或继续 |
| **Esc** / Q | 等待保存完成并退出 |
| **Z** | 切换局部放大，原图仍完整保存 |
| **A** | 切换 RGB 自动 / 手动曝光 |
| **[ / ]**、**- / =** | 调整曝光、增益；转入手动模式 |

字母键使用英文输入法。排队不等于保存完成；入库失败的帧可补同步。

## 设备与配置

| 用途 | 配置 | 说明 |
| --- | --- | --- |
| D415 采集与预览 | [d415.json](config/d415.json) | 型号、序列号、RGB / 深度流、视觉参数 |
| D435 数据采集 | [data_capture_d435.json](config/data_capture_d435.json) | 独立 RGB 采集，启动时填写序列号 |
| D435 定位 | [realsense_config.json](config/realsense_config.json) | 定位参数与标定信息 |
| 模型训练 | [train_config.json](config/train_config.json) | 默认 40 轮、batch 32、workers 4、AMP |
| 当前检测器 | [vision_model.json](config/vision_model.json) | 权重、训练报告及 SHA256 |
| 数据分组 | [port_dataset_split.json](config/port_dataset_split.json) | 固定 train / val 分组 |

D415 默认预览为 **1920×1080 / 30 fps**，纯 RGB 采集为 **1280×720 / 15 fps**。换相机时修改配置或传入 `--serial`。

**D435 采集与定位分别启动**，使用前填写对应设备序列号和配置：

```powershell
# RGB 数据采集
python -m vision.launch vision.capture.collect --config config/data_capture_d435.json --serial "实际完整序列号" --batch d435_session01

# 定位入口
python -m vision.launch vision.locate.run --config config/realsense_config.json
```

## 目录结构

```text
vision/
├── capture/       # 采集、预览、会话保存、补入库、相机
├── algorithms/    # 增强、轮廓、中心拟合、孔位关联、PnP
├── review/        # 检测框 / 九孔复核与页面
├── library/       # 图库、来源映射、同步与分组
├── evaluation/    # 批量评估、人工点对比、图库诊断
├── locate/        # D435 定位与诊断
└── calibration/   # 标定与深度诊断
train/python/      # 检测器 / 力觉模型训练、布局拟合
control/           # 插装控制、实验管理、软件仿真
config/            # 设备、模型、训练与实验配置
dataset/vision/    # 原图、深度、标签、相机参数、来源记录
runs/              # 训练快照、权重、指标与报告
artifacts/         # 原始采集会话、诊断与验证输出
```

## 进阶用法

每个命令可加 `--help` 查看参数；下面按任务展开。

<details>
<summary>采集参数与保存内容</summary>

```powershell
# 查看设备型号与序列号
python -m vision.launch vision.capture.collect --list-devices

# 记录验证集批次和样件
python -m vision.launch vision.capture.preview --depth --batch d415_val01 --specimen print_val01 --split val

# 仅采 RGB
python -m vision.launch vision.capture.collect --batch d415_session01 --split train

# 补同步未完成入库的会话
python -m vision.capture.intake
```

| 选项 | 用途 |
| --- | --- |
| `--zoom` | 启动时放大目标；也可按 Z 切换 |
| 省略 `--depth` | 仅预览彩色图 |
| `--frames 3 --save-first` | 分析 3 帧，保存首个完成结果 |
| `--batch`、`--specimen`、`--notes` | 记录批次、样件和拍摄条件 |
| `--split test --no-library` | 独立 test 采集，避免进入训练图库 |
| intake 的 `--session "会话目录"` | 只补同步指定会话 |

管理员终端可直接运行 `python -m vision.capture.preview --depth`。未检测到目标或 PnP 失败仍可保存；新图待人工复核，不自动成为负样本。

预览会话保存在 `artifacts/d415_preview/<会话>/`，纯 RGB 会话在 `artifacts/captures/<会话>/`：

| 会话内容 | 文件 |
| --- | --- |
| 原始 RGB / 叠加图 | `images/`、`overlays/` |
| 原始深度 | `depth/`，uint16 Z16，未对齐、未补洞 |
| 同帧诊断 | `diagnostics/*.json` |
| 参数与清单 | `camera.json`、`depth_camera.json`、`manifest.json` |

深度米值 = 原始像素值 × 会话的 `depth_scale_m`；伪彩色不能反算。入库保留配对深度和参数，检测器训练只使用 RGB。

</details>

<details>
<summary>复核与图库维护</summary>

| 操作 | 命令 |
| --- | --- |
| 同步已有图片 | `python -m vision.library.prepare_port_library` |
| 查看图库状态 | `python -m vision.library.training_library status` |
| 校验图片、标签和哈希 | `python -m vision.library.training_library validate` |
| 检查深度绑定 | `python -m vision.library.training_library depth-status` |
| 检查 / 保存相机参数 | `python -m vision.library.training_library calibration-status` / `preserve-calibrations` |
| 只复核一批检测框 | `python -m vision.review.review_library --session "会话目录"` |
| 只打印复核页地址 | 复核命令追加 `--no-browser` |

- 检测框需人工确认；草稿可恢复，版本冲突时先刷新。
- 九孔修改自动保存到 `dataset/vision/keypoint_annotations.json`；填写全部孔位状态及复核人后确认。
- 自动孔位使用当前模型生成，仍是待复核建议；更新保留人工修改和已确认图片，证据不足时不补造孔位。
- 参数 `verified` 只代表文件哈希一致；缺失的会话参数不能用其他相机或会话参数代替。

离线检测框页面的导出与回写：

```powershell
python -m vision.library.prepare_port_library --review artifacts/port_library_review.html
Start-Process artifacts/port_library_review.html
python -m vision.library.training_library annotate --review-json "已导出的框标注.json"
```

九孔离线页面：传 `--manifest` 绑定已结束的采集会话；历史图片改用 `--selection` 与 `--batch-report`。可加 `--annotations` 恢复已有复核内容，两种来源不可混用。

```powershell
python -m vision.review.prepare_keypoint_review --manifest "会话目录/manifest.json"
python -m vision.review.prepare_keypoint_review --selection config/vision_annotation_pilot.json --batch-report "历史批处理报告.json" --output-dir artifacts/keypoint_review
```

</details>

<details>
<summary>数据分组、训练与模型复现</summary>

- 同会话、批次、样件或打印批次保持分组；已有 train / val 不重新洗牌。
- 未分组开发数据按关联组继承用途；无关联且未声明用途的新组默认进入 train。
- 待标注、分组冲突、test 来源或哈希不符会阻止训练；val 仍是开发验证集。
- 训练完成并通过覆盖检查后才发布模型；权重和指标以 `vision_model.json` 关联的报告为准。

```powershell
# 查看待分组项；需要时明确指定用途
python -m vision.library.dataset_split --list-pending
python -m vision.library.dataset_split --assign-group "实际group_id" --split train

# 只准备数据快照，不训练
python train/python/train_port.py --prepare-only

# 指定训练参数
python train/python/train_port.py --epochs 40 --batch 32 --workers 4 --device 0

# 使用某轮训练导出的模型配置
python -m vision.launch vision.capture.preview --depth --vision-config "runs/本轮目录/preview_config.json"
```

训练支持 `--no-amp` 和 `--workers 0`。AMP 自检失败时使用 FP32；环境不支持多进程时回退到单进程。实际参数、逐轮覆盖和指标保存在训练报告中；缓存写入项目内 `.cache/`。

</details>

<details>
<summary>关键点评估与历史诊断</summary>

```powershell
# 评估已确认的图库孔位
python -m vision.evaluation.evaluate_library_keypoints --annotations "已复核孔位.json" --output-dir artifacts/keypoint_eval_first

# 使用相同标注和权重比较算法版本
python -m vision.evaluation.evaluate_library_keypoints --annotations artifacts/keypoint_eval_first/annotations.json --output-dir artifacts/keypoint_eval_second --baseline artifacts/keypoint_eval_first/report.json

# 历史图片批量诊断
python -m vision.evaluation.batch_pipeline_eval --max-images 50
```

- 输出目录必须是新目录；报告含逐孔诊断、叠加图和参考来源，对比模式另存比较报告。
- 默认 `--tolerance-px 5` 是诊断邻域，不是精度验收指标。
- 历史批量诊断使用 Kinect v2 近似内参，不能用于评价 D415 / D435 的重投影误差。

| 其他检查 | 查看参数 |
| --- | --- |
| 检测框是否覆盖人工孔位 | `python -m vision.evaluation.check_reviewed_detection_labels --help` |
| 人工点求解 / 带标注比较 / 批量结果比较 | `python -m vision.evaluation.evaluate_keypoint_annotations --help` |
| 重复标注一致性 | `python -m vision.review.keypoint_annotations repeatability --help` |

</details>

<details>
<summary>软件仿真、实验与力觉工具</summary>

```powershell
python -m control.simulate --scenario nominal
python -m control.simulate --scenario overload --output artifacts/sim_overload.jsonl
python -m control.simulate replay --input "任务日志.jsonl"
python -m control.simulate admittance --output artifacts/admittance_report.json
```

仿真使用合成数据，不访问硬件。场景覆盖阻塞、过载、过期反馈等；完整场景及 `--method`、`--model` 参数见 `python -m control.simulate --help`。

| 工具 | 入口 |
| --- | --- |
| 力觉数据准备 | `python -m control.force_dataset --help` |
| GRU 训练、评价、验收与演示 | `python train/python/train_force_gru.py --help` |
| 实验配置、排程与统计 | `python -m control.experiments --help` |
| 风险约束 | `python -m control.risk --help` |

正式实验需冻结当前配置和源码指纹；训练完成或合成仿真通过不代表真机验收。

</details>

<details>
<summary>常见问题与环境检查</summary>

| 现象 | 检查方式 |
| --- | --- |
| 相机权限错误 | 使用 `vision.launch`，确认 UAC；启动结果查看新窗口 |
| 保存排队 / 入库待同步 | 等待退出摘要，必要时运行 `vision.capture.intake` |
| 训练被阻止 | 检查待标注图片、分组冲突及哈希；不要强行跳过 |
| 相机参数缺失 | 运行 `calibration-status` 检查来源；RGB 框训练与几何验收分别判断 |
| CUDA DLL 被系统拦截 | 检查 Windows 应用控制策略及获准的依赖构建 |

GPU 检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

</details>

## 项目约定

- **数据与权重**：源码、配置、`dataset/vision/`、`yolov8n.pt` 和训练权重纳入 Git；诊断输出、缓存及其他训练产物忽略，规则见 [.gitignore](.gitignore)。
- **记录与复现**：保留原图、标注、相机参数和来源哈希；训练结果查看当前模型关联的报告。历史图片映射与近似内参继续用于历史诊断。
- **验证范围**：深度仅作诊断；开发验证指标、自检和软件仿真不代表真实九孔精度或插装效果。D435 标定、手眼外参及真机控制仍需独立验收。
