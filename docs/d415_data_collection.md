# D415 采集训练数据，D435 用于机械臂定位

`vision.collect` 是独立的 D415 彩色图片采集入口。采集原始完整 RGB 画面，以无损 PNG 保存；不运行检测、PnP 或机械臂控制。默认绑定序列号 `349222063070`，分辨率为 1280×720、15 fps，启动后丢弃前 30 帧以等待成像稳定。参数在 `config/data_capture_d415.json` 中。

上述纯 RGB 采集配置保留 2026-10-29 USB 2.1 时使用的 15 fps。2026-10-30 已确认更换连接后实际为 USB 3.2；下方带检测和深度的 `vision.preview` 改用独立 USB3 配置。每次采集均从实际彩色流重新读取内参，随图像保存，不使用旧 Kinect 参数。

机械臂上的 D435 使用 `vision.run` 和 `config/realsense_config.json`，D415 预览使用 `config/d415_preview.json`；两者通过 `config/vision_model.json` 共用当前检测权重，各自保留设备身份、内参及控制门槛。D415 的图片可以用于训练二维检测/孔位模型，后续应以独立拍摄的 D435 数据验证迁移效果。

## 实时检测、孔位与残差预览

在仓库根目录运行：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.preview
```

窗口使用 D415 实际 RGB 内参，显示检测框、ROI、候选点、孔名中心、重投影点、逐孔残差及 mean/RMS/max。当前默认几何为 `config/geometry_nominal.json`；SDK 内参和名义几何均保留未独立验证标记，但可以进行诊断估计。`位姿估计`、`质量告警`、`暂无位姿` 只表示该帧的诊断情况，不输出机器人控制消息。

- **F5**：保存屏幕上已分析帧的原图、叠加图和诊断 JSON；未检测到目标或 PnP 失败也可保存。
- **F6**：暂停/继续显示，暂停时可反复查看和保存同一帧；后台仍读取相机，恢复后继续分析。
- **Esc**：结束会话并释放相机；点击关闭窗口同样有效。
- 英文输入状态下也支持空格/S 保存、P 暂停、Q 退出。中文输入法下优先使用 F5/F6。

相机由独立线程持续采集，仅保留最新一组完整 RGB/深度；分析使用单个后台线程，不累积待分析队列。图像、孔位和残差始终属于同一帧。相同结果缓存绘制，窗口键盘事件持续处理。面板分别显示配置 FPS、实测采集/分析/新结果绘制 FPS，以及当前显示帧的年龄；分析速度受 CPU 和目标复杂程度影响，配置 30 fps 不表示完整视觉分析达到 30 fps。首次结果产生前保存键只提示等待。

### USB3 清晰度和曝光调节

默认彩色为 **1920×1080 / BGR8 / 30 fps**（`config/d415_preview_usb3.json`），深度为 **640×480 / Z16 / 30 fps**（`config/d415_depth_usb3.json`）。启动预热 60 帧，RGB 自动曝光开启，自动曝光优先级设为 0。设备不支持所请求的模式时明确报错，不静默降分辨率。

使用当前模型，启动时放大唯一检测目标：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth --zoom
```

如果希望降低 RGB 数据量，使用 **1280×720 / 30 fps** 档，深度仍为 640×480 / 30 fps：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth --quality hd --zoom
```

`--quality full-hd` 显式选择 1080p/30，`--quality hd` 选择 720p/30；不传该参数时遵循 `--capture-config`。原图始终按实际采集分辨率无损保存，预览主图区按 1280 宽显示，侧栏独立绘制；每个新会话保存该分辨率实际内参，不能把旧 720p 的 `camera.json` 用于 1080p 图片。

| 按键 | 操作 |
| --- | --- |
| `Z` | 完整画面 / 唯一检测目标局部放大；没有唯一目标时保留全图 |
| `A` | 切换 RGB 自动/手动曝光 |
| `[` / `]` | 减少/增加 RGB 曝光，同时切换到手动曝光 |
| `-` / `=` | 减少/增加 RGB 增益，同时切换到手动曝光 |

这些字母和符号键请切换到英文输入状态。暂停时先按 F6 恢复再调成像参数；调整在采集线程中执行，只影响之后采集的帧。参数范围读取自真实 RGB 传感器，窗口消息报告实际读回值。曝光增加可能使运动拖影更明显或降低实际帧率，增益增加可能引入噪声；以目标孔口细节和面板实测 FPS 为准。

命令行可用 `--auto-exposure`，或用 `--exposure 数值 --gain 数值` 启动手动模式；数值采用 SDK 选项单位，程序校验设备范围。自动模式下显示的曝光/增益是带时间戳的 SDK 选项缓存，不能当成逐帧实际曝光。逐帧诊断另存 `frame.rgb_settings.frame_metadata` 中设备支持的实际帧元数据；不支持的字段留空。程序正常退出和异常清理时恢复本次修改前的 RGB 选项。

Windows RGB 曝光采用离散档位，例如本机 156→312；快捷键按这些档位调整，其他平台使用设备范围内的步进。请求值与读回值分别记录在 `sensor_options.last_changes`，不会把被量化的请求值当成实际应用值；读回有最多 0.62 秒的短暂确认等待。依据是 [librealsense v2.59.1 Windows 曝光换算实现](https://github.com/realsenseai/librealsense/blob/v2.59.1/src/mf/mf-uvc.cpp#L459)。本次调整不安装驱动或改系统设置。

局部放大仅帮助观察，不补造孔口细节。`images/` 保留完整无叠加原图，`overlays/` 是显示尺寸的叠加图，`diagnostics/` 中孔位仍使用原始完整图像坐标。F5 保存的是正在显示的已完成分析帧及其配对深度；同步落盘期间窗口可能短暂停顿，采集线程继续取帧，终端结果中 `save_ms` 记录保存耗时。

2026-10-30 在 D415 `349222063070`、固件 5.17.0.10、USB 3.2 下完成两档实测。每档先读 150 组 frameset，再运行 80 帧分析并保存一组 RGB/深度/诊断；重复 RGB 帧不计入实际采集 FPS。

| 彩色档位（深度均为 640×480/30） | 仅采集 FPS | 带窗口、分析及保存的采集 FPS | 完整分析 FPS | 首次保存耗时 |
| --- | ---: | ---: | ---: | ---: |
| 1920×1080/30 | 29.68 | 29.51 | 4.90 | 78 ms |
| 1280×720/30 | 29.76 | 29.55 | 7.12 | 47 ms |

上述实测发生在切换 CUDA 环境之前，使用 CPU，且是两次短时、实际桌面场景的结果；目标和遮挡有所变化，不是固定图像的算法速度对照。当前环境已支持 RTX 5080，不能把此表当作当前 GPU 的速度测试。1080p 提供更多原始像素；需要更快的完整诊断刷新时可试用 `--quality hd` 并检查实测 FPS。曝光从 156 调到 312、增益从 64 调到 70 后，实拍黑色孔口轮廓更明显，但白色背景过曝，不能作为统一最佳曝光。两次结束后均读回原始 RGB 选项并确认恢复。孔名关联和位姿检查仍有失败，该历史检查不宣称九孔识别率或位姿精度提高。

完整报告及原图保存在：

- `artifacts/d415_preview_upgrade/20261030/full-hd_20261030T085917526196Z/report.json`
- `artifacts/d415_preview_upgrade/20261030/hd_20261030T090020389602Z/report.json`

同目录 `run_validation.py --quality full-hd --preview` / `--quality hd --preview` 可复现本机检查，会短暂改变曝光/增益并在退出时恢复。该目录为本地诊断产物，不随 Git 分发，图片未自动加入训练集。

每次启动在 `artifacts/d415_preview/` 下新建会话。`images/` 保存不带叠加的完整原图，`overlays/` 保存预览图，`diagnostics/` 保存同帧原始候选、自动孔名、PnP、逐孔残差向量、错误原因和文件哈希；`preview_manifest.json` 记录模型、配置和几何来源。自动孔位不作为人工真值。相同帧/完全相同图像会跳过重复保存。

采集清单遇到 Windows `WinError 5/32/33` 时，会以 25、50、100、200、400 ms 的间隔重试原子写入，最多 6 次，每次清单写入累计等待不超过 0.775 秒。短暂占用解除后继续保存，不重复添加样本；持续拒绝访问仍报错并结束，保留此前完整记录。该重试仅用于采集元数据，不增加控制循环的写入等待。遇到持续错误时保留终端错误和会话目录，勿删除已有清单或将失败会话改成成功；重新运行原命令会创建新会话。

诊断会话默认 `split=unknown`；可用 `--batch`、`--specimen`、`--printing-batch` 和 `--notes` 记录实际采集条件。需按训练/验证/测试预分组的大批量采集仍使用下方 `vision.collect`。退出预览后，用该会话的 `manifest.json` 接入本文的九孔人工复核流程。

切换几何使用 `--geometry 路径.json`；实测几何须同时提供匹配的样件与打印批次。更换相机用 `--serial`，流配置用 `--capture-config`。诊断算法默认读取 `config/d415_preview.json`，可通过 `--vision-config` 显式选择固定轮次的预览配置；诊断不写入机器人控制输出路径。训练成功后更新共享模型记录，重启预览才能加载新模型；旧实验权重需显式选择，不会自动回退使用。

有限次启动检查（分析三帧，自动保存第一帧后退出）：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.preview --frames 3 --save-first
```

## 可选深度辅助诊断

在项目根目录运行带深度的预览：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth
```

保留原 RGB 诊断窗口，同时打开 `D415 Depth Diagnostics` 中文深度面板。F5 保存当前已分析的同一组 RGB/深度，F6 同时暂停两窗口的显示，Esc 或关闭任一窗口结束会话。默认不加 `--depth` 时仍为 RGB 模式；`vision.collect` 的原有采图入口不变。

默认使用上述 USB3 双流配置。旧 `config/d415_depth_diagnostics.json` 保留深度 640×480 / 15 fps，与旧 `config/data_capture_d415.json` 的 1280×720 / 15 fps 对应；2026-10-29 曾在 USB 2.1 下验证这组模式。需要复现旧模式时同时传入 `--capture-config config/data_capture_d415.json --depth-config config/d415_depth_diagnostics.json`，此旧配置默认不启用新的 RGB 曝光控制。可通过 `--depth-config` 指定其他经过核对的深度配置。

深度面板显示原生深度图、非零深度比例、场景深度中位数、双流时间差，以及检测目标 ROI 的测量覆盖和距离。黑色为无效测量；颜色按 0.15–1.50 m 固定范围显示，超界只截断颜色，不修改原始数据。此比例只统计原生 Z16 非零值，不等于测量精度或目标可用率。

目标距离和候选平面使用 SDK 点云及深度到彩色的外参转换到**彩色相机坐标系**。ROI 覆盖率是收到深度投影的不同 RGB 像素数 / 内缩 ROI 像素数，受两流分辨率影响，不能与原生深度有效比例混用。未检测到目标时不拟合目标平面；有目标时排除已命名孔附近和深度突变，检查平面支撑范围、内点比例及残差。结果只称为“候选表面平面”：检测框不能确认哪个台阶就是建模孔口截面。多层表面可能被拒绝，法向、距离或 PnP 一致性无结果时显示 `--`。

PnP 一致性仅在模型孔点共面且位姿可解时报告法向夹角与模型参考点到候选平面的距离，不用深度替换 PnP，也不据此宣称重投影误差改善。拍显示屏时深度测量来自屏幕实际表面，不会重建屏幕中的虚拟三维模型。

保存仍保留 `images/`、`overlays/`、`diagnostics/` 和原来的 RGB `manifest.json`，可继续进入人工复核；额外保存：

- `depth/frame_*.png`：未对齐、未滤波、未补洞的原始 uint16 Z16 深度；米值为原值 × `depth_scale_m`。
- `depth_camera.json`：实际深度内参、畸变、深度尺度、到彩色的外参（旋转矩阵按行保存、平移单位米）、双流配置与设备信息。彩色内参仍在 `camera.json`。
- `preview_manifest.json` 与逐帧诊断升级为 schema 2，绑定 RGB/深度帧、独立时间戳及时间域、标定、文件和像素哈希。彩图只用于显示，不作为测量保存。

默认时间差阈值 40 ms 是诊断配对设置；同一 frameset 不代表零时间差，两个流的帧号也不要求相等。深度缺失、重复、倒序、时钟域变化或时间差超限时保留 RGB 诊断并明确拒绝该次深度结果；存在的原始失败深度仍可保存。重复/倒序 RGB 不重新提交分析，避免同一张 RGB 换绑新的深度。异常元数据中的非有限数在 JSON 中记录为 `null` 并保留失败原因。所有平面筛选阈值随诊断保存，均非机器人安全限值。

两台相机必须分别读取自己的参数；这里仍使用未独立验证的 D415 SDK 参数。此功能没有接入 D435 控制入口或机械臂运动。

有限次双流检查：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.preview --depth --frames 3 --save-first
```

## RGB 数据采集

在项目根目录打开终端，使用现有虚拟环境。先列出型号和完整序列号，此命令不启动图像流：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --list-devices
```

当前这台 D415 已写入默认配置，可直接开始采集；更换相机时用 `--serial` 指定列表中的完整序列号：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --batch d415_session01 --split unknown --notes "室内，正面与倾斜视角"
```

点击预览窗口后操作：

- **空格或 S**：保存当前原图；窗口上的提示文字不会写进原图。
- **Q 或 Esc**：结束并完成批次清单；关闭窗口也会正常结束。
- **Ctrl+C**：中断采集，保留已经完整保存的图片。

每两秒保存一张，达到 50 张后结束：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --batch d415_session02 --split unknown --interval 2 --max-images 50
```

自动采集仍显示预览，可随时按 Q 停止。默认 `--interval 0` 为手动保存，`--max-images 0` 为按键结束。程序跳过重复设备帧和完全相同的 PNG 内容，但不自动判断相似视角；不要把定时采集当作近重复数据清理。

序列号也可填写到采集配置的 `serial` 中。程序会核对连接设备和实际启动的流，不接受 D435 冒充 D415，也不会静默换用其他分辨率。未检测到设备时，`--list-devices` 输出 `[]`。

## 输出与批次划分

默认写入被 Git 忽略的 `artifacts/captures/`；每次启动创建新的唯一会话目录，终端会打印其完整路径：

```text
artifacts/captures/<UTC时间_唯一标识>/
  camera.json
  manifest.json
  images/
    frame_000001.png
    frame_000002.png
```

`camera.json` 保存实际 D415 彩色流的序列号、型号、分辨率、帧率、内参矩阵、畸变模型和参数，以及参数指纹。设备提供的参数记录为 `validated=false`，不等于任务标定通过，也不会覆盖 D435 的标定文件。

`manifest.json` 记录拍摄批次 `batch_id`、分区 `split`、会话状态和采集说明。每张图片有内容哈希、设备帧号、设备时间戳、时间域和主机接收 UTC 时间。设备时间戳与接收时间只是记录，不代表已与机器人同步。文件先保存成功，再加入清单；异常时保留此前完整记录。

同一次拍摄、同一连续场景尽量使用相同 `--batch`，即使中途重启采集也保留该批次名。在**同一个输出根目录**内，同名批次不能跨 `train/val/test/unknown` 分区；`unknown` 表示尚未确定用途，不会进入当前二维先验的 train 拟合。不要通过更换输出根目录或随意改批次名，把近似图片拆进不同集合。

默认采集根目录适合持续使用；可用 `--output-root` 指定项目内另一个目录。一个会话的分区固定，人工复核不会修改分区、相机或图片身份。

Windows 上访问 D415 前需要 `SeSecurityPrivilege`（librealsense 2.59.1 的全局互斥体带 SACL），启动方式与排障见 [windows_realsense_privilege.md](windows_realsense_privilege.md)。

## 先使用 SDK 内参做视觉分析

当前先使用设备 SDK 内参，不要求先拍标定板。`config/d415_camera_intrinsics.json`（受版本管理，含 `intrinsics_sha256` 参数指纹）保存本机于 2026-10-29 从该 D415 的 1280×720 / 15 fps 彩色流实际读取的参数。它保留 `validated=false`，可以计算检测点、PnP 位姿及重投影残差；不将设备参数标记为独立验收结果。

内参、标定和验收类记录以版本管理内的 `config/` 为准。原始读数快照保存在 `config/d415_intrinsics/`：`readout.json` 记录设备序列号、固件版本、USB 速率、彩色流配置及连续 15 帧的设备时间戳，`camera.json` 是其 `camera_file_sha256` 所指向的参数文件（sha256 `09293ac2…b16cc7`），参数与 `config/d415_camera_intrinsics.json` 一致。`artifacts/` 只存放可再生成的运行产物，不承担长期保存职责。

单张同相机、同分辨率的原始图片可以这样分析（替换实际图片路径）：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.run --image "实际D415图片.png" --camera-config config/d415_camera_intrinsics.json --output artifacts/d415_pose.json
```

新采集图片优先将 `--camera-config` 指向该会话的 `camera.json`。不要用于缩放、裁剪后的图片或其他相机。未独立验证时结果仍可能包含位姿与残差，但 `valid=false`，原因包含 `intrinsics_not_independently_validated`，退出码为 2；另有漏孔、目标未检测或残差超限时也会明确记录。单图分析务必指定独立 `--output`，避免写入控制器使用的输出路径。

新会话复核完成后，`vision.diagnose` 自动使用标注绑定的会话内参，无需把这份 `version=null` 的 SDK 快照传给 `--camera-override`。后续需要更高精度时，再按 [相机与几何诊断](camera_geometry_diagnostics.md) 做内参独立检查。

## 接入九孔复核

采集正常结束或中断后，将实际清单路径填入以下命令：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/prepare_capture_review.py --manifest artifacts/captures/实际会话目录/manifest.json
```

工具生成同会话目录下的：

```text
review/annotations.json
review/review.html
```

从文件夹打开 `review.html`，沿用现有九孔标签：`S_neg、CC2、S_pos、CC1、DC_neg、DC_pos、A_neg、PE、A_pos`。每张图的九个点初始均为未标注；标出可见内孔中心，并为其余孔选择遮挡、出画或不确定状态，再填写复核人、确认图片并导出 JSON。工具不会把模板预测点填成真实观测。

页面包含原图，缩放显示不会改变保存的原始像素坐标。采集标注使用 schema 2，绑定该会话的图片、相机和清单哈希；旧 24 图的 schema 1 仍可按原流程使用。

恢复人工导出文件时，可用 `--annotations`，该 JSON 只读且不会被覆盖：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/prepare_capture_review.py --manifest artifacts/captures/实际会话目录/manifest.json --annotations "C:\Users\admin\Downloads\reviewed_annotations.json"
```

只接受已结束且有完整图片的会话；录制中、空会话、内参或原图变动、路径越界均拒绝。失败/中断会话中已成功保存的图片也可复核。

现有 `fit_observed_layout.py` 可读取人工确认后的 schema 2，并只使用 train 图片；先积累足够的不同视角再拟合。九孔复核不会自动重训或替换检测器。采集原始 RGB 会自动加入统一图库，目标框标注、按会话分组及训练方法见 [统一图库与训练](../README.md#数据集与训练)。旧 `evaluate_keypoint_annotations.py --batch-report` 是 Kinect 开发基线流程，不能直接用于这些新采集标注。

## 拍什么更有用

优先覆盖目前容易漏孔的情形：倾斜视角、大小孔之间的遮挡、不同工作距离、暗部与反光。每次改变视角后等待画面稳定再保存，并在 `--notes` 中写明条件。避免大量几乎相同的正面图片。

D415 训练图和 D435 实际作业画面应尽量覆盖相近的目标尺寸与视角。另留一批新的 D435 实拍数据做验证；用于反复调参的图应归为开发数据，不能再当独立测试集。

## 验证范围

离线测试覆盖真实图片编码保存、清单到九孔复核的衔接、相机/图片绑定、批次隔离、SDK 与显示替身、错误型号拒绝、原图不含提示文字、自动采集及异常退出。D435 原采集代码、视觉契约和配置经哈希核对保持不变。

全量 270 项测试通过，其中新增采集/保存/复核测试 34 项。日志为 `artifacts/d415_capture_smoke/tests.log`；合成会话已完成 PNG 保存、重复帧跳过、独立内参记录和复核页生成的完整离线检查。

2026-10-28 已在 D415 固件 5.15.1.55、USB 2.1、1280×720 / 15 fps 下验证连续彩色帧、实际内参读取、单张诊断 PNG 保存及手动采集预览。`artifacts/d415_live_check/` 中的诊断会话为 `split=unknown`，不是充电口训练数据；当次手动采集会话正常结束，但保存图片为 0 张。2026-10-29 固件升级至 5.17.0.10 后只复查了设备信息和支持模式，仍为 USB 2.1；未将该复查计为升级后的连续采集验收。

2026-10-29 本次应用 SDK 内参后，11 项采集相关测试通过；使用默认配置在固件 5.17.0.10、USB 2.1 下完成预热和单帧原图保存，实际流内参指纹与 `config/d415_camera_intrinsics.json` 一致。记录位于 `artifacts/d415_sdk_apply/20261029T084744Z_b340274aa21f4397a1cdcba130685985/`，分区为 `unknown`。该帧已使用新配置运行离线视觉入口，返回 `target_not_detected`，因此本次只确认采图和参数接入，尚未验证该帧的孔位、PnP 或重投影精度。

复核页面的导入逻辑已通过 JavaScript 自动检查，尚未做本轮实际浏览器点击验收。`artifacts/d415_capture_smoke/` 内仅为明确标注的合成离线测试数据，不属于采集训练集。
