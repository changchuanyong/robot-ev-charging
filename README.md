# robot-ev-charging

充电口检测、九孔定位与 PnP，以及面向 **RM75 六维力机械臂 + D435 眼在手 + Windows/Python** 的力觉–视觉插装程序。主线：**D415 采集 → YOLOv8n 检测 → 九孔/PnP 诊断 → D435 定位与控制**；默认配置不连接、不驱动机器人。

- **默认模型**：`config/vision_model.json` → `runs/detect_split_20261030T110042Z_105be6be/weights/best.pt`，启动校验 SHA256。
- **图库**：`dataset/vision/` 241 条记录（234 ready、0 pending、7 排除、154 条带深度），随 Git 跟踪；其中 1 张新会话图片尚待 train/val 分组。
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

Windows 相机入口需要进程内的 `SeSecurityPrivilege`，统一使用 `vision.launch`；每次从非管理员终端启动都可能弹出 UAC：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.preview --depth
```

日常流程：**F5 拍照并入库 → 退出后复核检测框 → 准备训练**。新会话默认 `train`；需要开发验证数据时，在采集前指定 `--split val`。建议填写 `--batch` 和 `--specimen`，同批次、同样件、同打印批次沿用一致分组，冲突会在打开相机前提示。

**管理员终端**可以省掉启动器：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth
```

启动器返回成功只表示已发起子进程，是否正常采集需查看新窗口；离线工具直接运行，无需提权。查看设备序列号（默认绑定 D415 `349222063070`，换机加 `--serial`）：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.collect --list-devices
```

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.preview --depth --zoom     # 放大唯一目标
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.preview                     # 只看彩色
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.preview --depth --frames 3 --save-first
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.preview --depth --batch d415_session01 --specimen print_dev01 --notes "打印件，正视"
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.preview --depth --batch d415_val01 --specimen print_val01 --split val
```

### 按键

| 按键 | 操作 |
| --- | --- |
| **F5** | 当前帧加入后台保存队列并入库；深度模式同时保存配对原始深度 |
| **F6** | 暂停/继续 |
| **Esc** | 结束会话、释放相机 |
| **Z** | 放大检测目标（保存仍是全分辨率原图） |
| **A** | 切换 RGB 自动/手动曝光 |
| **[ / ]**、**- / =** | 曝光、增益增减（转入手动） |

字母键需英文输入法；也支持空格/S 保存、P 暂停、Q 退出。没有检出目标也能保存。

窗口分别显示已保存、已入库、待同步、待标注、待分组和排队数；“已排队”尚不代表写盘完成。队列最多容纳两帧，繁忙时提示稍后重试；退出会等待已提交任务完成并打印本批摘要。入库失败会保留原图和 `intake.json` 待办，下次启动同一输出目录的采集时自动补同步。也可以手动恢复：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.intake
```

补同步遇到损坏文件或分组冲突会保留错误并返回非零退出码；可加 `--session "artifacts/d415_preview/实际会话目录"` 只处理一批。

### 保存内容

`artifacts/d415_preview/<会话>/` 下有：`images/` 原图、`overlays/` 叠加图、`depth/` 原始 uint16 Z16（640×480，未对齐、未补洞）、`diagnostics/*.json`（含 `depth` / `depth_capture` 的帧号、哈希、时差）、`camera.json` / `depth_camera.json` / `manifest.json`。

米值 = 像素值 × 本次会话的 `depth_scale_m`；伪彩色不能反算。入库时配对深度保存到 `dataset/vision/depth/`，RGB 和深度参数按文件 SHA 保存到 `dataset/vision/calibrations/`，训练仍只读 RGB。

### 只采 RGB

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.collect --batch d415_session01 --split train --notes "打印件，多角度"
```

空格/S 保存，Q/Esc 退出，输出在 `artifacts/captures/`。纯 RGB 采集同样后台入库，复核时手动画框；省略 `--batch` 会生成唯一批次名。独立 test 采集使用 `--split test --no-library`。

## 图库与标注

`dataset/vision/` 存原始 RGB、配对深度、相机参数、标签和 `records/`，训练按 SHA256 去重。检测预览会保留拍摄当帧的 YOLO 框与模型哈希作为建议，新图仍为 `pending`；PnP 失败不阻止入库，未检测到目标也不会自动写成负样本。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 train/python/prepare_port_library.py       # 同步入库
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library status          # 库状态
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library depth-status    # 深度绑定
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library calibration-status # 参数完整性
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library preserve-calibrations # 保存现有参数或标记缺失
```

参数状态 `verified` 只表示图库内文件与记录中的哈希一致，不代表标定验收。2026-10-31 检查时，RGB 参数按每图相同来源去重后有 161 条引用，深度参数有 154 条引用，文件均缺失；另有 80 条记录未保存 RGB 参数引用。154 张原始深度仍完整，但无法据此确认与 RGB 的几何对齐。缺失参数不阻止 RGB 框训练，也不能用其他会话参数替代；参数检查遇到缺失时返回非零退出码。

拍完后运行本机复核页；会自动打开浏览器，查看或拖动修改框，点击“保存框并下一张”直接写回图库。草稿可以恢复，只有明确确认才成为正式标注；保存冲突会提示刷新。完成后点击页面的结束按钮或在终端按 Ctrl+C：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.review_library
```

默认显示全库待标注图片；加 `--session "artifacts/d415_preview/实际会话目录"` 只看一批。采集结束会打印带真实路径的完整命令。该命令不需要 `vision.launch`；`--no-browser` 可只打印地址。退出采集时生成的 `review.html` 是离线副本，离线复核继续使用导出/导入：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 train/python/prepare_port_library.py --review artifacts/port_library_review.html
Start-Process artifacts/port_library_review.html
.\.venv\Scripts\python.exe -B -X utf8 -m vision.training_library annotate --review-json "$env:USERPROFILE\Downloads\reviewed_port_boxes.json"
```

新采集的 `--split train/val` 会自动登记到 `config/port_dataset_split.json`，不按单张图片随机划分。旧的 197 train / 36 dev-val 划分保留；val 是开发验证集。主动指定 `--split unknown` 或历史未分组的会话仍需人工指定用途：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m train.python.dataset_split --list-pending
.\.venv\Scripts\python.exe -B -X utf8 -m train.python.dataset_split --assign-group "实际group_id" --split train
```

九孔位置的人工复核统一使用 `prepare_keypoint_review.py`；它生成孔名/中心复核页，与上面的 YOLO 目标框复核用途不同。新采集传 `--manifest`，使用该会话保存的相机参数；历史图片传 `--selection` 和 `--batch-report`，保留原历史参数：

```powershell
# 已结束的采集会话，输出默认在会话目录的 review/ 下
.\.venv\Scripts\python.exe -B -X utf8 live/python/prepare_keypoint_review.py --manifest "artifacts/d415_preview/实际会话目录/manifest.json"
# 历史图片，明确指定已有的图片选择清单和批量诊断报告
.\.venv\Scripts\python.exe -B -X utf8 live/python/prepare_keypoint_review.py --selection "config/vision_annotation_pilot.json" --batch-report "artifacts/实际诊断目录/latest_details.json" --output-dir "artifacts/keypoint_review"
```

打开命令输出的 `review.html`，复核后导出 `reviewed_annotations.json`。加 `--annotations "已复核文件.json"` 可继续复核，工具不会覆盖原标注；`--manifest` 不能与历史数据参数混用。

## 训练

```powershell
.\.venv\Scripts\python.exe -B -X utf8 train/python/train_port.py --prepare-only
.\.venv\Scripts\python.exe -B -X utf8 train/python/train_port.py
.\.venv\Scripts\python.exe -B -X utf8 train/python/train_port.py --epochs 40 --batch 16 --device 0
```

训练前同步并检查全库，待标注、未分组、冲突、哈希不符都会报错；全部轮数与覆盖通过后才原子更新 `config/vision_model.json`。输出含 `training_manifest.json`、`training_report.json`、`epoch_coverage.jsonl`、`validation_coverage.jsonl`、`weights/`。

当前模型：40 轮、197/197 覆盖、36/36 验证；P 0.97011 / R 0.97222 / mAP50 0.97111 / mAP50-95 0.91302（目标框，不是九孔或插装精度）。复现旧轮次：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.launch vision.preview --depth --vision-config "runs/本轮目录/preview_config.json"
```

## 离线控制与批量评估

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m vision.check_project
.\.venv\Scripts\python.exe -B -X utf8 -m control.simulate --scenario nominal
.\.venv\Scripts\python.exe -B -X utf8 live\python\batch_pipeline_eval.py --max-images 50
```

`vision.check_project` 检查项目完整性并报告数据待办，不启动相机或机器人，不能替代算法与硬件验证。仿真其他场景见 `--help`。批量评估默认跑 `dataset/vision/images/historical`，用 Kinect v2 近似内参（`config/camera_intrinsics.json`），结果在 `artifacts/batch_eval/`；不要用它评价 D415/D435 的重投影误差。

## Git 跟踪

- **跟踪**：源码、`config/`、`dataset/vision/`、`yolov8n.pt`、`runs/*/weights/{best,last}.pt`。
- **忽略**：`artifacts/`（采集、诊断与验证输出）、`dataset/live/`、`runs/` 其他文件、缓存与 C/C++ 编译产物。
- 图库的相机参数和标签通过 `.gitattributes` 保持原始字节，避免 Git 换行转换破坏 SHA 校验。

## 已知限制

- RGB 内参需替换为当前设备标定；D415/D435 内参记录在各自会话。
- 深度只作诊断证据，不进入训练，也不驱动机械臂。
- val 是开发验证集，最终泛化需要独立 test。
- 真机手眼标定、载荷限值、力觉模型和插装效果仍需实验验收。

## 历史 Kinect 流程

`live/cpp/`、`train/cpp/` 中的 Kinect 采集入口需要 Kinect SDK v2、设备和自行编译的程序（仓库不含 exe）；`config/camera_intrinsics.json` 仅用于历史数据。`live/python/` 中的增强、轮廓、孔位和 PnP 模块仍由 `vision.pipeline` 复用，属于当前视觉流程的依赖。
