# robot-ev-charging

充电口检测、九孔定位与 PnP，以及面向 **RM75 六维力机械臂 + D435 眼在手 + Windows/Python** 的力觉–视觉插装程序。主线：**D415 采集 → YOLOv8n 检测 → 九孔/PnP 诊断 → D435 定位与控制**；默认配置不连接、不驱动机器人。

- **默认模型**：`config/vision_model.json` → `runs/detect_split_20261030T110042Z_105be6be/weights/best.pt`，启动校验 SHA256。
- **图库**：`dataset/vision/` 241 条记录（233 ready、1 pending、7 排除、154 条带深度），随 Git 跟踪。
- **权重**：`yolov8n.pt`、`runs/*/weights/best.pt` 和 `last.pt` 随 Git 跟踪。
- **最近训练**：40 轮，197 train / 36 dev-val；P 0.970 / R 0.972 / mAP50 0.971 / mAP50-95 0.913。

命令都在仓库根目录运行，直接用 `.venv\Scripts\python.exe`，不要激活虚拟环境。

## 环境

`.venv` 已存在（Python 3.12.14），由 `uv` 创建、**没有 pip**：

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-control.txt
uv pip install --python .venv\Scripts\python.exe -r requirements-gpu.txt   # NVIDIA 机器追加 CUDA torch
```

GPU 自检（本机输出 `2.14.1+cu132 True` / `NVIDIA GeForce RTX 5080`）：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## D415 预览与采集

相机入口需要 `SeSecurityPrivilege`。**非管理员终端**用 `vision.launch`，首次弹一次 UAC：

```powershell
.\.venv\Scripts\python.exe -m vision.launch vision.preview --depth
```

**管理员终端**可以省掉启动器：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth
```

所有相机命令同理：非管理员时把 `-m vision.preview` / `-m vision.collect` 换成 `-m vision.launch ...`。设备序列号（默认绑定 D415 `349222063070`，换机加 `--serial`）：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.collect --list-devices
```

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth --zoom     # 放大唯一目标
.\.venv\Scripts\python.exe -B -X utf8 -m vision.preview                     # 只看彩色
.\.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth --frames 3 --save-first
.\.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth --batch d415_session01 --specimen print_dev01 --notes "打印件，正视"
```

### 按键

| 按键 | 操作 |
| --- | --- |
| **F5** | 保存当前帧；深度模式同时保存配对原始深度 |
| **F6** | 暂停/继续 |
| **Esc** | 结束会话、释放相机 |
| **Z** | 放大检测目标（保存仍是全分辨率原图） |
| **A** | 切换 RGB 自动/手动曝光 |
| **[ / ]**、**- / =** | 曝光、增益增减（转入手动） |

字母键需英文输入法；也支持空格/S 保存、P 暂停、Q 退出。没有检出目标也能保存。

### 保存内容

`artifacts/d415_preview/<会话>/` 下有：`images/` 原图、`overlays/` 叠加图、`depth/` 原始 uint16 Z16（640×480，未对齐、未补洞）、`diagnostics/*.json`（含 `depth` / `depth_capture` 的帧号、哈希、时差）、`camera.json` / `depth_camera.json` / `manifest.json`。

米值 = 像素值 × `depth_camera.json` 的 `depth_scale_m`（本机 0.001）；伪彩色只用于显示，不能反算。可用深度会按图像 SHA 复制进 `dataset/vision/depth/`，训练仍只读 RGB。

### 只采 RGB

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.collect --batch d415_session01 --split unknown --notes "打印件，多角度"
```

空格/S 保存，Q/Esc 退出，输出在 `artifacts/captures/`。

## 图库与标注

`dataset/vision/` 存原始 RGB、配对深度、标签和 `records/`，训练按 SHA256 去重；新图默认 `pending`。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 train/python/prepare_port_library.py       # 同步入库
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library status          # 库状态
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library depth-status    # 深度绑定
```

补标注：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 train/python/prepare_port_library.py --review artifacts/port_library_review.html
Start-Process artifacts/port_library_review.html
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library annotate --review-json "$env:USERPROFILE\Downloads\reviewed_port_boxes.json"
```

分组（固定划分 `config/port_dataset_split.json`：197 train / 36 dev-val，val 是开发验证集）：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m train.python.dataset_split --list-pending
.\.venv\Scripts\python.exe -B -X utf8 -m train.python.dataset_split --assign-group "实际group_id" --split train
```

## 训练

```powershell
.\.venv\Scripts\python.exe -B -X utf8 train/python/train_port.py --prepare-only
.\.venv\Scripts\python.exe -B -X utf8 train/python/train_port.py
.\.venv\Scripts\python.exe -B -X utf8 train/python/train_port.py --epochs 40 --batch 16 --device 0
```

训练前同步并检查全库，待标注、未分组、冲突、哈希不符都会报错；全部轮数与覆盖通过后才原子更新 `config/vision_model.json`。输出含 `training_manifest.json`、`training_report.json`、`epoch_coverage.jsonl`、`validation_coverage.jsonl`、`weights/`。

当前模型：40 轮、197/197 覆盖、36/36 验证；P 0.97011 / R 0.97222 / mAP50 0.97111 / mAP50-95 0.91302（目标框，不是九孔或插装精度）。复现旧轮次：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth --vision-config "runs/本轮目录/preview_config.json"
```

## 离线控制与批量评估

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m control.simulate --scenario nominal
.\.venv\Scripts\python.exe -B -X utf8 live\python\batch_pipeline_eval.py --max-images 50
```

仿真输出 `success: true`、393 周期、`stop_confirmed: true`，其他场景见 `--help`。批量评估默认跑 `dataset/vision/images/historical`，用 Kinect v2 近似内参（`config/camera_intrinsics.json`），结果在 `artifacts/batch_eval/`；不要用它评价 D415/D435 的重投影误差。

## Git 跟踪

- **跟踪**：源码、`config/`、`dataset/vision/`、`yolov8n.pt`、`runs/*/weights/{best,last}.pt`。
- **忽略**：`artifacts/`（采集、诊断与验证输出）、`dataset/live/`、`runs/` 其他文件、缓存与 C/C++ 编译产物。

## 已知限制

- RGB 内参需替换为当前设备标定；D415/D435 内参记录在各自会话。
- 深度只作诊断证据，不进入训练，也不驱动机械臂。
- val 是开发验证集，最终泛化需要独立 test。
- 真机手眼标定、载荷限值、力觉模型和插装效果仍需实验验收。

## 历史 Kinect 流程

`live/cpp/`、`live/python/`、`train/cpp/` 和 `config/camera_intrinsics.json` 是 2026-10 之前的 Kinect v2 流程，需要 Kinect SDK v2、设备和自行编译的采集程序（仓库不含 exe）；当前 D415 流程不依赖它们。
