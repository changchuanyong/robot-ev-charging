# robot-ev-charging

充电口视觉检测与位姿估计项目，包含三条主线：
- 数据标注转换：`Labelme -> YOLO Detect`
- 目标检测训练：Ultralytics YOLO
- 实时流水线：采集 -> ROI -> 增强 -> 边缘 -> 候选点 -> PnP

## 1. 项目结构

```text
robot-ev-charging/
├─ train/python/
│  ├─ convert_labelme_to_yolo_detect.py
│  └─ train_port.py
├─ live/python/
│  ├─ launch_pipeline.py
│  ├─ roi.py
│  ├─ enhance.py
│  ├─ Canny.py
│  ├─ latest_candidate_contours.py
│  ├─ pnp2.py
│  └─ post_vis_watch.py
├─ live/cpp/
│  └─ kinect_live_capture_atomic.cpp
├─ yolo_port/
│  ├─ images/train, images/val
│  ├─ labels/train, labels/val
│  └─ dataset.yaml
├─ config/
│  └─ charging_port_model.csv
├─ dataset/live/
└─ runs/
```

## 2. 环境准备

推荐 Python 3.9+，Windows 下运行（当前脚本按 Windows 路径与进程管理实现）。

### 2.1 已验证依赖版本（来自本仓库 `.yolo_env`）

- `ultralytics==8.4.31`
- `torch==2.11.0+cu126`
- `opencv-python==4.13.0.92`
- `numpy==2.4.3`

### 2.2 安装

```powershell
python -m venv .yolo_env
.yolo_env\Scripts\activate
pip install -r requirements.txt
```

## 3. 从 0 到跑通（最短路径）

### 步骤 A：准备数据

将 Labelme 标注数据放到：
- `yolo_port/images/train`
- `yolo_port/images/val`

每张图对应一个同名 `.json`（Labelme）。

### 步骤 B：转换标注

```powershell
.yolo_env\Scripts\python.exe train\python\convert_labelme_to_yolo_detect.py
```

期望结果：
- 生成 `yolo_port/labels/train/*.txt`、`yolo_port/labels/val/*.txt`
- 更新 `yolo_port/dataset.yaml`

### 步骤 C：训练模型

```powershell
.yolo_env\Scripts\python.exe train\python\train_port.py
```

期望结果：
- 训练目录：`runs/detect_retrain`
- 权重：`runs/detect_retrain/weights/best.pt`

### 步骤 D：实时流水线

先确保：
- `live/cpp/kinect_live_capture_atomic.exe` 已存在
- `runs/detect_retrain/weights/best.pt` 已存在

启动：

```powershell
.yolo_env\Scripts\python.exe live\python\launch_pipeline.py
```

## 4. 流水线输入输出说明

### 4.1 关键输入

- 采集图：`dataset/live/latest.jpg`
- 模型：`runs/detect_retrain/weights/best.pt`
- PnP 模型点：`config/charging_port_model.csv`

### 4.2 关键中间产物（按顺序）

- `roi.py`
  - `dataset/live/latest_roi.jpg`
  - `dataset/live/latest_roi_meta.json`
  - `dataset/live/latest_vis.jpg`
- `enhance.py`
  - `dataset/live/latest_roi_enhanced.jpg`
- `Canny.py`
  - `dataset/live/latest_edges.jpg`
  - `dataset/live/latest_edges_vis.jpg`
- `latest_candidate_contours.py`
  - `dataset/live/latest_candidate_contours.jpg`
  - `dataset/live/latest_fitted_centers.jpg`
  - `dataset/live/latest_fitted_centers.json`
- `pnp2.py`
  - `dataset/live/latest_pose.json`
  - `dataset/live/latest_pose_vis.jpg`

## 5. 关键配置项（建议先看）

### 5.1 `train/python/train_port.py`

- `MODEL_FILE`：初始权重（默认 `yolov8n.pt`）
- `DATA_FILE`：数据集配置（默认 `yolo_port/dataset.yaml`）
- `EPOCHS`、`IMGSZ`、`BATCH`
- `DEVICE`：`0` 表示首张 GPU，`cpu` 表示 CPU，`None` 自动
- `RUN_NAME`：训练输出目录名

### 5.2 `live/python/launch_pipeline.py`

- `KINECT_EXE`：采集程序路径
- `PYTHON_EXE`：Python 解释器路径
- `ALLOW_NO_CAMERA_DURING_DEBUG`：无相机时是否继续调试
- `SHOW_POST_WINDOWS_IN_PIPELINE`：是否显示后处理窗口

### 5.3 `live/python/roi.py`

- `MODEL_PATH`：实时检测模型路径
- `CONF_THRES`：检测置信度阈值
- `MIN_PADDING` / `PAD_RATIO`：ROI 外扩策略

### 5.4 `live/python/pnp2.py`

- `K`、`DIST_COEFFS`：相机标定参数（必须替换成你的真实标定结果）
- `MODEL_CSV_PATH`：3D 模型点 CSV
- `LAYOUT_TO_MODEL_LABEL`：2D 点名到 3D 点名映射
- `POINTS_ARE_IN_ROI`：输入点是否在 ROI 坐标系

## 6. `charging_port_model.csv` 说明

支持的列名组合（至少要有点名、x、y）：
- 点名列：`name` / `point_name` / `id` / `label`
- X 列：`x` / `x_mm` / `model_x`
- Y 列：`y` / `y_mm` / `model_y`
- Z 列：`z` / `z_mm` / `model_z`（可选，不填默认 `0`）

建议统一使用：`label,x_mm,y_mm,z_mm`

## 7. 常见问题（FAQ）

### 7.1 找不到 `best.pt`

先训练，或检查 `roi.py` 的 `MODEL_PATH` 是否和训练输出一致。

### 7.2 找不到 `kinect_live_capture_atomic.exe`

先编译 `live/cpp/kinect_live_capture_atomic.cpp`，并把产物放到 `live/cpp/`。

### 7.3 PnP 结果漂移/翻转

- 优先检查 `pnp2.py` 的 `K` 和 `DIST_COEFFS`
- 检查 `LAYOUT_TO_MODEL_LABEL` 的左右极性是否正确
- 检查 `charging_port_model.csv` 的点名是否和 `latest_fitted_centers.json` 对齐

### 7.4 流水线运行但无更新

按顺序检查是否有文件持续更新：
- `latest.jpg` -> `latest_roi.jpg` -> `latest_roi_enhanced.jpg` -> `latest_edges.jpg`

## 8. 版本与忽略规则

- 运行产物（如 `runs/`、`*.pt`、`dataset/live/latest*.jpg/json`）已在 `.gitignore` 忽略。
- 如果你希望多人复现一致环境，优先使用 `requirements.txt`。
