# robot-ev-charging

电动车充电口视觉检测与位姿估计项目。

项目包含三条主线：

- Labelme 标注转换为 YOLO 检测数据集
- 使用 Ultralytics YOLO 训练充电口检测模型
- 实时视觉流水线：采集 -> ROI -> 增强 -> 边缘 -> 孔位中心 -> PnP 位姿

## 项目结构

```text
robot-ev-charging/
├─ config/
│  ├─ camera_intrinsics.json
│  ├─ charging_port_model.csv
│  ├─ live_pipeline_config.json
│  └─ train_config.json
├─ train/
│  ├─ python/
│  │  ├─ convert_labelme_to_yolo_detect.py
│  │  ├─ check_yolo_dataset.py
│  │  └─ train_port.py
│  └─ cpp/
├─ live/
│  ├─ cpp/
│  │  └─ kinect_live_capture_atomic.cpp
│  └─ python/
│     ├─ launch_pipeline.py
│     ├─ roi.py
│     ├─ enhance.py
│     ├─ Canny.py
│     ├─ latest_candidate_contours.py
│     ├─ candidate_types.py
│     ├─ layout_prior.py
│     ├─ center_fit.py
│     ├─ pnp2.py
│     └─ post_vis_watch.py
├─ yolo_port/
│  ├─ images/train, images/val
│  ├─ labels/train, labels/val
│  └─ dataset.yaml
├─ dataset/live/        # 运行时输入输出，Git 忽略
├─ artifacts/           # 本地验证图和临时产物，Git 忽略
├─ runs/                # YOLO 训练输出，Git 忽略
└─ docs/
```

## 环境准备

推荐 Windows + Python 3.9+。当前项目主要在 `.yolo_env` 虚拟环境中运行。

```powershell
python -m venv .yolo_env
.yolo_env\Scripts\activate
pip install -r requirements.txt
```

已验证过的关键依赖包括：

- `ultralytics`
- `torch`
- `opencv-python`
- `numpy`

## 数据集与训练

Labelme 标注文件和图片放在：

```text
yolo_port/images/train
yolo_port/images/val
```

每张图片对应一个同名 `.json` 标注文件。

转换为 YOLO 标签：

```powershell
.yolo_env\Scripts\python.exe train\python\convert_labelme_to_yolo_detect.py
```

检查数据集：

```powershell
.yolo_env\Scripts\python.exe train\python\check_yolo_dataset.py
```

训练模型：

```powershell
.yolo_env\Scripts\python.exe train\python\train_port.py
```

训练输出默认在：

```text
runs/detect_retrain/weights/best.pt
```

## 实时流水线

启动整条实时流水线：

```powershell
.yolo_env\Scripts\python.exe live\python\launch_pipeline.py
```

流水线配置在：

```text
config/live_pipeline_config.json
```

启动后主要步骤如下：

1. `roi.py`
   使用 YOLO 检测充电口并裁剪 ROI。

2. `enhance.py`
   对 ROI 做灰度增强、滤波和轻锐化。

3. `Canny.py`
   生成边缘图。默认使用稳定的 Canny；可通过环境变量 `VISION_EDGE_METHOD=hybrid` 试验形态学补边。

4. `latest_candidate_contours.py`
   读取边缘图，筛选孔位候选，应用布局先验，输出孔位中心。

5. `pnp2.py`
   将 ROI 坐标映射回原图，结合 3D 模型点计算 PnP 位姿。

## 候选点模块拆分

孔位候选处理原本集中在 `latest_candidate_contours.py`，现在拆为几块：

- `candidate_types.py`
  定义 `CandidateContour` 和 `LayoutModel`。

- `layout_prior.py`
  负责标准孔位布局、主孔锚点、layout 匹配、面积先验和候选排序。

- `center_fit.py`
  负责椭圆拟合、拟合中心漂移保护和中心 JSON 导出。

- `latest_candidate_contours.py`
  保留入口、基础轮廓过滤、可视化和文件读写。

这样拆分后，后续调孔位规则时主要改 `layout_prior.py`，调中心拟合时主要改 `center_fit.py`。

## 运行时输出

关键运行时文件都在 `dataset/live/` 下：

```text
latest.jpg                       # 原始采集帧
latest_vis.jpg                   # YOLO 检测可视化
latest_roi.jpg                   # 裁剪 ROI
latest_roi_meta.json             # ROI 元数据
latest_roi_enhanced.jpg          # 增强后的 ROI
latest_edges.jpg                 # 边缘图
latest_edges_vis.jpg             # 边缘叠加可视化
latest_candidate_contours.jpg    # 候选轮廓可视化
latest_fitted_centers.jpg        # 孔位中心拟合可视化
latest_fitted_centers.json       # 孔位中心数据
latest_pose.json                 # PnP 位姿数据
latest_pose_vis.jpg              # PnP 位姿可视化
```

这些文件属于运行产物，不提交到 Git。

## 关键配置

### `config/train_config.json`

控制训练模型、数据集、epoch、batch、device 和 run name。

### `config/live_pipeline_config.json`

控制实时流水线中的 Kinect exe、Python 解释器、窗口显示和等待超时。

### `config/charging_port_model.csv`

定义充电口 3D 模型点。当前推荐字段：

```csv
label,x_mm,y_mm,z_mm
```

### `config/camera_intrinsics.json`

相机内参配置。当前文件如果 `is_placeholder` 为 `true`，说明还不是实际标定结果。PnP 的绝对位姿精度会受这个限制。

## 当前已知限制

- `camera_intrinsics.json` 仍需要替换为真实相机标定结果。
- 右侧主大孔在某些图像中可能和外圈边缘粘连，目前中心基本可用，但后续还可以继续做内圈分离。
- 当前验证主要基于已有 live 样张；新角度、新光照下建议再做实拍回归。

## 常用检查命令

语法检查：

```powershell
.yolo_env\Scripts\python.exe -m py_compile live\python\latest_candidate_contours.py live\python\layout_prior.py live\python\center_fit.py live\python\pnp2.py
```

单独跑候选点与 PnP：

```powershell
$env:VISION_PIPELINE_MODE='1'
.yolo_env\Scripts\python.exe live\python\latest_candidate_contours.py
.yolo_env\Scripts\python.exe live\python\pnp2.py
```

## Git 忽略策略

以下内容不提交：

- `dataset/live/`
- `artifacts/`
- `runs/`
- `*.pt`
- Python 缓存和 YOLO cache

源码、配置、数据集图片和标签按需提交。
