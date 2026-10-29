# D415 采集训练数据，D435 用于机械臂定位

`vision.collect` 是独立的 D415 彩色图片采集入口。采集原始完整 RGB 画面，以无损 PNG 保存；不运行检测、PnP 或机械臂控制。默认分辨率为 1280×720、30 fps，启动后丢弃前 30 帧以等待成像稳定。参数在 `config/data_capture_d415.json` 中。

若设备当前枚举为 USB 2.1，本机实测的 1280×720 彩色流最高只支持 15 fps，默认 30 fps 请求会失败。请复制采集配置，将副本的 `fps` 改为 `15`，通过 `--config 配置副本路径` 启动。程序不会自动降低帧率；是否支持 30 fps 应以当前连接实际提供的模式为准。

机械臂上的 D435 仍使用 `vision.run` 和 `config/realsense_config.json`，其型号限制、内参验证及控制门槛保持原样。D415 的图片可以用于训练二维检测/孔位模型，后续应以独立拍摄的 D435 数据验证迁移效果。

## 采集

在项目根目录打开终端，使用现有虚拟环境。先列出型号和完整序列号，此命令不启动图像流：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --list-devices
```

将下面的 `你的D415序列号` 替换为列表中 D415 的完整序列号：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --serial 你的D415序列号 --batch d415_session01 --split train --notes "室内，正面与倾斜视角"
```

点击预览窗口后操作：

- **空格或 S**：保存当前原图；窗口上的提示文字不会写进原图。
- **Q 或 Esc**：结束并完成批次清单；关闭窗口也会正常结束。
- **Ctrl+C**：中断采集，保留已经完整保存的图片。

每两秒保存一张，达到 50 张后结束：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --serial 你的D415序列号 --batch d415_session02 --split train --interval 2 --max-images 50
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

现有 `fit_observed_layout.py` 可读取人工确认后的 schema 2，并只使用 train 图片；先积累足够的不同视角再拟合。此次没有自动导入 YOLO 数据集、重训或替换运行权重，也没有给原图补充未经人工确认的检测框。旧 `evaluate_keypoint_annotations.py --batch-report` 是 Kinect 开发基线流程，不能直接用于这些新采集标注。

## 拍什么更有用

优先覆盖目前容易漏孔的情形：倾斜视角、大小孔之间的遮挡、不同工作距离、暗部与反光。每次改变视角后等待画面稳定再保存，并在 `--notes` 中写明条件。避免大量几乎相同的正面图片。

D415 训练图和 D435 实际作业画面应尽量覆盖相近的目标尺寸与视角。另留一批新的 D435 实拍数据做验证；用于反复调参的图应归为开发数据，不能再当独立测试集。

## 验证范围

离线测试覆盖真实图片编码保存、清单到九孔复核的衔接、相机/图片绑定、批次隔离、SDK 与显示替身、错误型号拒绝、原图不含提示文字、自动采集及异常退出。D435 原采集代码、视觉契约和配置经哈希核对保持不变。

全量 270 项测试通过，其中新增采集/保存/复核测试 34 项。日志为 `artifacts/d415_capture_smoke/tests.log`；合成会话已完成 PNG 保存、重复帧跳过、独立内参记录和复核页生成的完整离线检查。

2026-10-28 已在 D415 固件 5.15.1.55、USB 2.1、1280×720 / 15 fps 下验证连续彩色帧、实际内参读取、单张诊断 PNG 保存及手动采集预览。`artifacts/d415_live_check/` 中的诊断会话为 `split=unknown`，不是充电口训练数据；当次手动采集会话正常结束，但保存图片为 0 张。2026-10-29 固件升级至 5.17.0.10 后只复查了设备信息和支持模式，仍为 USB 2.1；未将该复查计为升级后的连续采集验收。

复核页面的导入逻辑已通过 JavaScript 自动检查，尚未做本轮实际浏览器点击验收。`artifacts/d415_capture_smoke/` 内仅为明确标注的合成离线测试数据，不属于采集训练集。
