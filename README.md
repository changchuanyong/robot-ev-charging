# robot-ev-charging

## 1. 项目概览

充电口检测、九孔定位与 PnP，以及面向 **RM75 六维力机械臂 + D435 眼在手 + Windows/Python** 的力觉–视觉插装程序。主线：**D415 采集 → YOLOv8n 检测 → 九孔/PnP 诊断 → D435 定位与控制**；默认配置不连接、不驱动机器人。

- **默认模型**：`config/vision_model.json` → `runs/detect_split_20261031T105817Z_14d8271b/weights/best.pt`，启动校验 SHA256。
- **图库**：`dataset/vision/` 260 条记录（253 ready、0 pending、7 排除、173 条带深度），随 Git 跟踪；2026-10-31 的 19 张新图已人工复核并写入标签。
- **权重**：`yolov8n.pt`、`runs/*/weights/best.pt` 和 `last.pt` 随 Git 跟踪。
- **最近训练**：40 轮，217 train / 36 dev-val；P 0.97068 / R 0.97222 / mAP50 0.97250 / mAP50-95 0.90472（best epoch 21）。

日常命令在各自项目根目录运行，使用前先执行 `conda activate ev-vision`，再直接运行 `python` 命令。日常只走 **采集 → 复核 → 训练 → 离线检查**；不常用的带参数命令和维护命令集中在 [进阶命令](#4-进阶命令不常用参数)，正常流程不需要逐个了解。

## 2. 环境

默认使用 Anaconda 的 `ev-vision` 环境（Python 3.12），使用前先执行 `conda activate ev-vision`。环境声明为 `environment-vision.yml`，现有工作版本及 CUDA PyTorch 下载来源锁定在 `environments/vision-pip.lock.txt`。两份项目副本共用此依赖环境，但各自使用自己的代码和数据。

换机或首次创建环境时，在本项目目录执行一次：

```powershell
& "$env:USERPROFILE\anaconda3\Scripts\conda.exe" env create --format environment-yaml -f environment-vision.yml
```

本机 Anaconda 安装在 `C:\Users\admin\anaconda3`。可在 Navigator 的 Environments 中查看 `ev-vision`，或在 Anaconda Prompt 中执行 `conda activate ev-vision` 后使用 `python`。

GPU 自检（历史输出为 `2.14.1+cu132 True` / `NVIDIA GeForce RTX 5080`；当前 Windows 应用控制拦截 CUDA DLL，新旧环境均受影响，GPU 验收暂未通过，见 [迁移记录](environments/MIGRATION.md)）：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

九孔复核运行：

```powershell
python -m vision.review_keypoints
```

## 3. 正常流程

按采集 → 复核 → 训练 → 离线检查的顺序执行；默认命令即可，需要额外参数时再查 [进阶命令](#4-进阶命令不常用参数)。

### 3.1 D415 预览与采集

Windows 相机入口需要进程内的 `SeSecurityPrivilege`，统一使用 `vision.launch`；每次从非管理员终端启动都可能弹出 UAC：

```powershell
python -m vision.launch vision.preview --depth
```

日常流程：**F5 拍照并入库 → 退出后复核检测框 → 准备训练**。新会话默认 `train`；需要开发验证数据时在采集前指定 `--split val`。建议填写 `--batch` 和 `--specimen`，同批次、同样件、同打印批次沿用一致分组，冲突会在打开相机前提示：

```powershell
python -m vision.launch vision.preview --depth --batch d415_session01 --specimen print_dev01 --notes "打印件，正视"
```

D415 参数统一在 `config/d415.json`：`device` 设置型号和序列号，`profiles.preview` 设置检测预览（默认 1920×1080 / 30 fps），`profiles.collect` 设置纯 RGB 采集（默认 1280×720 / 15 fps），`depth` 设置深度流，`vision` 设置检测与 PnP。预览和纯采集都可用 `--config` 指定这份配置；`--vision-config` 仍可单独加载某次训练导出的检测配置。D435 继续使用自己的设备与定位配置。

**管理员终端**可以省掉启动器，直接运行 `vision.preview`；具体命令见 [采集与设备](#41-采集与设备)。

启动器返回成功只表示已发起子进程，是否正常采集需查看新窗口；离线工具直接运行，无需提权。默认绑定 D415 `349222063070`，换机加 `--serial`；查看设备序列号、纯 RGB 采集等命令见 [采集与设备](#41-采集与设备)。

#### 3.1.1 按键

| 按键 | 操作 |
| --- | --- |
| **F5** | 当前帧加入后台保存队列并入库；深度模式同时保存配对原始深度 |
| **F6** | 暂停/继续 |
| **Esc** | 结束会话、释放相机 |
| **Z** | 放大检测目标（保存仍是全分辨率原图） |
| **A** | 切换 RGB 自动/手动曝光 |
| **[ / ]**、**- / =** | 曝光、增益增减（转入手动） |

字母键需英文输入法；也支持空格/S 保存、P 暂停、Q 退出。没有检出目标也能保存。

窗口分别显示已保存、已入库、待同步、待标注、待分组和排队数；“已排队”尚不代表写盘完成。队列最多容纳两帧，繁忙时提示稍后重试；退出会等待已提交任务完成并打印本批摘要。入库失败会保留原图和 `intake.json` 待办，下次启动同一输出目录的采集时自动补同步；手动补同步见 [采集与设备](#41-采集与设备)。

#### 3.1.2 保存内容

`artifacts/d415_preview/<会话>/` 下有：`images/` 原图、`overlays/` 叠加图、`depth/` 原始 uint16 Z16（640×480，未对齐、未补洞）、`diagnostics/*.json`（含 `depth` / `depth_capture` 的帧号、哈希、时差）、`camera.json` / `depth_camera.json` / `manifest.json`。

米值 = 像素值 × 本次会话的 `depth_scale_m`；伪彩色不能反算。入库时配对深度保存到 `dataset/vision/depth/`，RGB 和深度参数按文件 SHA 保存到 `dataset/vision/calibrations/`，训练仍只读 RGB。

### 3.2 图库与标注

`dataset/vision/` 存原始 RGB、配对深度、相机参数、标签和 `records/`，训练按 SHA256 去重。检测预览会保留拍摄当帧的 YOLO 框与模型哈希作为建议，新图仍为 `pending`；PnP 失败不阻止入库，未检测到目标也不会自动写成负样本。

正常入库与检查：

```powershell
python train/python/prepare_port_library.py       # 同步入库
python -m vision.training_library status          # 库状态
```

图片与标签校验、深度绑定、参数完整性等维护命令见 [图库维护](#42-图库维护)。

参数状态 `verified` 只表示图库内文件与记录中的哈希一致，不代表标定验收。2026-10-31 检查时，RGB 参数按每图相同来源去重后有 161 条引用，深度参数有 154 条引用，文件均缺失；另有 80 条记录未保存 RGB 参数引用。154 张原始深度仍完整，但无法据此确认与 RGB 的几何对齐。缺失参数不阻止 RGB 框训练，也不能用其他会话参数替代；参数检查遇到缺失时返回非零退出码。

拍完后运行本机复核页；会自动打开浏览器，查看或拖动修改框，点击“保存框并下一张”直接写回图库。草稿可以恢复，只有明确确认才成为正式标注；保存冲突会提示刷新。完成后点击页面的结束按钮或在终端按 Ctrl+C：

```powershell
python -m vision.review_library
```

默认显示全库待标注图片；采集结束会打印带真实路径的完整命令。该命令不需要 `vision.launch`。只看一批、只打印地址，以及退出采集后离线 `review.html` 的导出/导入方式见 [复核](#43-复核)。

新采集的 `--split train/val` 会自动登记到 `config/port_dataset_split.json`。训练前会自动补齐 `unknown` 或历史未分组的开发数据：同会话、批次、样件或打印批次关联的组继承已有用途；无关联且未声明用途的新组默认进入 train。新组明确声明的 train/val 会被尊重，跨用途关联或 test 来源会拒绝训练，且不会只写入一部分自动决定。

已有分组保持固定，不为凑 80:20 拆开会话或重新洗牌；val 仍是开发验证集。自动决定及原因会保存在分组配置和本轮训练清单中，正常训练不再需要手动填写会话编号。需要查看待办或提前明确新组用途时，可用 `dataset_split`，见 [数据分组](#44-数据分组)。

九孔位置的人工复核统一使用 `prepare_keypoint_review.py`；它生成孔名/中心复核页，与上面的 YOLO 目标框复核用途不同。命令与参数见 [九孔复核](#45-九孔复核)。

### 3.3 训练

```powershell
python train/python/train_port.py
```

训练前依次同步图库、检查图片和标注、自动补齐分组、冻结本轮数据。待标注、分组冲突、来源无法判断或哈希不符仍会报错。`--prepare-only` 同样会自动补齐分组并生成数据快照，但不会训练；显式轮数、批大小与设备选择见 [训练与复现](#46-训练与复现)。训练完成全部轮数并通过覆盖检查后才原子更新 `config/vision_model.json`。输出含 `training_manifest.json`、`training_report.json`、`epoch_coverage.jsonl`、`validation_coverage.jsonl`、`weights/`。

Ultralytics 的设置与字体、Matplotlib 字体缓存和临时文件都写在项目内 `.cache/`（首次运行把系统的 `arial.ttf` 复制过去），因此训练不写用户目录、也不联网下载字体；`.cache/` 不纳入版本控制。

训练报告和清单的写入对 Windows 文件占用做重试（最长约 13 s，杀毒软件或文件监视器偶尔会短暂占住 JSON）；`training_report.json` 每轮都会重写，重试仍失败时改为就地写入，不会让几十轮训练白跑。

当前模型：40 轮、217/217 覆盖、36/36 验证；P 0.97068 / R 0.97222 / mAP50 0.97250 / mAP50-95 0.90472（best epoch 21；目标框，不是九孔或插装精度）。复现旧轮次用 `--vision-config`，命令见 [训练与复现](#46-训练与复现)。

### 3.4 离线检查

```powershell
python -m vision.check_project
python -m control.simulate --scenario nominal
```

`vision.check_project` 检查项目完整性并报告数据待办，不启动相机或机器人，不能替代算法与硬件验证。批量评估、其他仿真场景与力觉/实验工具见 [评估、控制与力觉](#47-评估控制与力觉)。

## 4. 进阶命令（不常用参数）

以下命令正常流程不需要，按需查阅；每项都可用 `--help` 查看完整参数。

### 4.1 采集与设备

```powershell
# 查看设备序列号（默认绑定 D415 349222063070，换机加 --serial）
python -m vision.launch vision.collect --list-devices
# 管理员终端可省掉启动器，直接运行相机入口
python -m vision.preview --depth
# 放大唯一目标（保存仍是全分辨率原图）
python -m vision.launch vision.preview --depth --zoom
# 只看彩色
python -m vision.launch vision.preview
# 分析 3 帧后退出；保存首个完成结果（含失败）
python -m vision.launch vision.preview --depth --frames 3 --save-first
# 开发验证数据
python -m vision.launch vision.preview --depth --batch d415_val01 --specimen print_val01 --split val
```

只采 RGB：

```powershell
python -m vision.launch vision.collect --batch d415_session01 --split train --notes "打印件，多角度"
```

空格/S 保存，Q/Esc 退出，输出在 `artifacts/captures/`。纯 RGB 采集同样后台入库，复核时手动画框；省略 `--batch` 会生成唯一批次名。独立 test 采集使用 `--split test --no-library`。

补同步入库：

```powershell
python -m vision.intake
```

补同步遇到损坏文件或分组冲突会保留错误并返回非零退出码；可加 `--session "artifacts/d415_preview/实际会话目录"` 只处理一批。

### 4.2 图库维护

```powershell
python -m vision.training_library validate        # 图片、标签和哈希检查
python -m vision.training_library depth-status    # 深度绑定
python -m vision.training_library calibration-status # 参数完整性
python -m vision.training_library preserve-calibrations # 保存现有参数或标记缺失
```

### 4.3 复核

```powershell
# 只看某一次采集会话
python -m vision.review_library --session "artifacts/d415_preview/实际会话目录"
# 只打印地址，不自动打开浏览器
python -m vision.review_library --no-browser
```

退出采集时生成的 `review.html` 是离线副本，离线复核继续使用导出/导入：

```powershell
python train/python/prepare_port_library.py --review artifacts/port_library_review.html
Start-Process artifacts/port_library_review.html
python -m vision.training_library annotate --review-json "$env:USERPROFILE\Downloads\reviewed_port_boxes.json"
```

### 4.4 数据分组

```powershell
python -m train.python.dataset_split --list-pending
python -m train.python.dataset_split --assign-group "实际group_id" --split train
```

### 4.5 九孔复核

日常直接打开全图库孔位标注页（与检测框复核独立）：

```powershell
python -m vision.review_keypoints
```

页面按需加载原图，包含检测框已排除的图片。选择孔名后点击孔中心，支持缩放和平移；不可见孔选择不确定、遮挡或出画。修改自动保存到 `dataset/vision/keypoint_annotations.json`，九孔状态齐全并填写复核人后点击确认。重启可继续，导出 JSON 可用于 `check_reviewed_detection_labels.py --annotations` 检查；孔位结果不会修改检测框标签。多页面编辑同一张图发生冲突时会阻止覆盖，提示先导出修改再刷新。`--no-browser` 只打印地址；`prepare_keypoint_review.py` 不带参数也会打开此页面。

全图库孔位文件记录原图像素坐标，不补填缺失的相机参数，也不直接作为位姿精度验收依据。下面的离线流程继续用于绑定特定会话或历史诊断报告：

每次打开或切换到未确认图片，都会读取 `config/vision_model.json` 指向的最近一次完成并发布的权重，重新检测充电口框，再用轮廓与二维布局算法提取孔位；不复用旧诊断点或人工检测框来代替当前模型推理。权重变化后自动重载模型并使旧候选缓存失效，无需重启服务。页面显示本图使用的训练轮次，自动点记录权重路径与 SHA256。橙色自动点仍是待复核草稿；更新会替换旧自动点，但保留人工点、人工清除状态和已确认图片。也可点击“用当前权重更新候选”立即刷新当前未确认图；识别不足时不会补造九个点。

孔位助手会对已通过椭圆拟合的 Canny 候选补做原图暗内孔证据检查，使用与原暗区分支相同的内外亮度差阈值；不会直接把所有外壁中心当作内孔，也不投影补造缺失点。该补充仅用于复核预标注，实时相机/PnP 的控制筛选逻辑仍独立。

复核预标注另外检查主孔间距相对图像区域的尺度、两个主孔的孔径比例及 PE 的相对孔径。补充 Canny 点不会放宽小孔匹配范围；存在接近等分的匹配时，只保留各方案一致的孔位，主孔或朝向不一致则不输出孔名。信号孔、中心孔和辅助孔还需符合实测 DC—PE—CC2 的相对位置。低证据图片因此可能比旧版本少显示候选，需要手工标注；已确认图与人工编辑不会被这些规则覆盖。

已确认的全图库孔位 JSON 可直接做逐孔诊断，不再需要历史批处理报告：

```powershell
python train/python/evaluate_library_keypoints.py --annotations "$env:USERPROFILE\Downloads\reviewed_annotations.json" --output-dir artifacts/keypoint_eval_first
# 修改算法后，使用冻结的相同标注；权重必须与基线一致
python train/python/evaluate_library_keypoints.py --annotations artifacts/keypoint_eval_first/annotations.json --output-dir artifacts/keypoint_eval_second --baseline artifacts/keypoint_eval_first/report.json
```

输出目录必须是新目录。报告包含阶段诊断、逐图叠加图和参考来源分类；对比模式另外生成 `comparison.html` 与 `comparison.json`。默认 5 原像素是诊断邻域，可用 `--tolerance-px` 明确指定，不能当作精度验收标准。孔名拓扑矛盾只提示人工检查，不修改标注。

新采集传 `--manifest`，使用该会话保存的相机参数；历史图片传 `--selection` 和 `--batch-report`，保留原历史参数：

```powershell
# 已结束的采集会话，输出默认在会话目录的 review/ 下
python live/python/prepare_keypoint_review.py --manifest "artifacts/d415_preview/实际会话目录/manifest.json"
# 历史图片，明确指定已有的图片选择清单和批量诊断报告
python live/python/prepare_keypoint_review.py --selection "config/vision_annotation_pilot.json" --batch-report "artifacts/实际诊断目录/latest_details.json" --output-dir "artifacts/keypoint_review"
```

打开命令输出的 `review.html`，复核后导出 `reviewed_annotations.json`。加 `--annotations "已复核文件.json"` 可继续复核，工具不会覆盖原标注；`--manifest` 不能与历史数据参数混用。

### 4.6 训练与复现

默认使用 `batch=32`、`workers=4` 和混合精度（AMP），仍逐轮检查实际完成批次中的训练图和验证图覆盖。AMP 使用本轮冻结的本地权重自检，不下载额外模型；自检不通过时使用 FP32。图片全量缓存保持关闭，避免额外占满内存。实际加载进程数和 AMP 启用状态写入 `training_report.json`；环境禁止多进程管道时会提示并回退到 `workers=0`。

```powershell
# 只同步、检查、分组并生成快照，不训练
python train/python/train_port.py --prepare-only
# 显式指定轮数、批大小、加载进程数和 GPU
python train/python/train_port.py --epochs 40 --batch 32 --workers 4 --device 0
# 使用原来的单进程、FP32 设置作速度对比
python train/python/train_port.py --batch 16 --workers 0 --no-amp
# 用某轮训练导出的配置复现旧轮次
python -m vision.launch vision.preview --depth --vision-config "runs/本轮目录/preview_config.json"
```

### 4.7 评估、控制与力觉

```powershell
python live\python\batch_pipeline_eval.py --max-images 50
```

批量评估默认跑 `dataset/vision/images/historical`，用 Kinect v2 近似内参（`config/camera_intrinsics.json`），结果在 `artifacts/batch_eval/`；不要用它评价 D415/D435 的重投影误差。

合并后的工具按子命令选择用途，以下命令显示各自完整参数：

```powershell
python live/python/evaluate_keypoint_annotations.py annotations --help
python live/python/evaluate_keypoint_annotations.py labeled --help
python live/python/evaluate_keypoint_annotations.py batch --help
python live/python/keypoint_annotations.py repeatability --help
python -m control.force_dataset --help
python train/python/train_force_gru.py train --help
python train/python/train_force_gru.py evaluate --help
python train/python/train_force_gru.py accept --help
python train/python/train_force_gru.py demo --help
python -m control.simulate simulate --help
python -m control.simulate replay --help
python -m control.simulate admittance --help
python -m control.experiments evaluate --help
python -m control.risk --help
```

视觉评价的 `annotations`、`labeled`、`batch` 分别对应人工点求解、带标注回归比较和批量结果比较，各自保留原统计口径。`control.force_dataset` 用 `--review-logs` 准备人工标签清单，用 `--annotations` 导出确认后的数据。力觉模型的训练、评价、验收和合成数据演示分别执行；训练不会自动通过真机验收。实验配置冻结、排程及统计统一由 `control.experiments` 提供。源码合并改变了代码指纹，新正式实验需按当前版本重新冻结，历史记录保留原指纹。

#### 4.7.1 离线仿真

`control.simulate` 用合成数据驱动完整控制状态机，不访问相机、机械臂或真实力传感器。默认场景 `nominal`：

```powershell
python -m control.simulate
```

指定场景或写任务日志：

```powershell
python -m control.simulate --scenario blocked
python -m control.simulate --scenario overload --output artifacts/sim_overload.jsonl
```

可选场景（13 个）：`nominal`、`blocked`、`overload`、`stale`、`disconnect`、`no_contact`、`bad_vision`、`risk_outside`、`feedback_lag`、`period_jitter`、`trial_timeout`、`observation_drift`、`confirmation_blocked`。

参数：`--config` 加载指定配置（默认合成配置，要求 `simulation_only`）；`--method` 切换控制方法；`--model` 指定 GRU 权重；`--output` 写 JSONL 任务日志。

成功时输出 JSON 摘要，例如 `nominal` 的结果为 `success: true`、`reason: mechanical_insertion_complete`、`states: [PREALIGN, APPROACH, CONTACT, INSERT, COMPLETE]`。

其他子命令：

```powershell
python -m control.simulate replay --input 任务日志.jsonl   # 重放记录，不访问硬件
python -m control.simulate admittance --output 报告.json   # 导纳离散化检查
```

仿真是离线合成夹具，不是经过验证的连接器仿真器；结果只用于检查状态机与异常分支，不代表真实插装效果。

## 5. 工程信息

### 5.1 Git 跟踪

- **跟踪**：源码、`config/`、`dataset/vision/`、`yolov8n.pt`、`runs/*/weights/{best,last}.pt`。
- **忽略**：`artifacts/`（采集、诊断与验证输出）、`dataset/live/`、`runs/` 其他文件、缓存与 C/C++ 编译产物。
- 图库的相机参数和标签通过 `.gitattributes` 保持原始字节，避免 Git 换行转换破坏 SHA 校验。

### 5.2 已知限制

- RGB 内参需替换为当前设备标定；D415/D435 内参记录在各自会话。
- 深度只作诊断证据，不进入训练，也不驱动机械臂。
- val 是开发验证集，最终泛化需要独立 test。
- 真机手眼标定、载荷限值、力觉模型和插装效果仍需实验验收。

### 5.3 历史数据与算法

Kinect 采集程序、旧磁盘流水线和一次性图库迁移脚本已移除。`config/camera_intrinsics.json` 仍供历史图片分析使用；`config/legacy_photo_index.json` 保留旧图片路径到统一图库的映射，历史标注继续可读。`live/python/enhance.py` 集中图像增强与边缘提取，`center_fit.py` 集中候选类型与中心拟合；这些算法和孔位/PnP 模块仍由当前视觉流程复用。
