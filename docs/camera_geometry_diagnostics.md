# 新相机与打印件几何诊断

这些工具实现数据绑定、比较和统计；仓库尚无新的 60 张人工复核诊断图或打印件实测几何，因此不能给出“新内参已降低实测重投影误差”的结论。旧 Kinect 图和人工复核开发图保留在原评估入口，不混入新相机独立测试结论。

## 1. 采图与元数据

在项目目录使用 `.venv\Scripts\python.exe`。采集入口支持 D415、D435；机器人控制入口仍只接受实际 D435 和完整序列号。以下设备编号、批次和样件名称必须替换成实际记录。

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --list-devices
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --config config/data_capture_d435.json --device-model D435 --serial "实际D435完整序列号" --batch d435_dev_session01 --split train --specimen printed_dev01 --printing-batch print_dev01 --geometry-version "实测几何版本" --source-kind printed_connector
.venv\Scripts\python.exe -B -X utf8 -m vision.collect --config config/data_capture_d415.json --device-model D415 --serial "实际D415完整序列号" --batch d415_dev_session01 --split train --specimen printed_dev01 --printing-batch print_dev01 --geometry-version "实测几何版本" --source-kind printed_connector
```

每次采集保存原始全幅 PNG、相机型号/序列号、实际 RGB 内参/畸变/分辨率/帧率、设备与接收时间戳、图像哈希和会话清单。没有悄悄降分辨率或切换型号的回退。按空格或 S 保存，Q/ESC 结束；原图不包含预览叠字。

用 `--conditions 路径.json` 记录同会话工况，例如：

```json
{"distance":"near","tilt":"oblique","image_position":"edge","lighting":"diffuse"}
```

字段值是实测或预先定义的工况标签；不要从低残差反推其真值。会话内改变距离/角度/样件时应结束会话并另建会话。相同采集批次、样件或打印批次不能跨 train/val/test；采集工具检查同一输出根目录，汇总诊断再次检查输入各会话。`unknown` 不会自动升级为测试集。真实接口照片用 `--source-kind real_connector`，不把它当成打印件位姿真值。

使用现有复核页面导出 schema 2：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/prepare_capture_review.py --manifest "artifacts/captures/实际会话/manifest.json"
```

图像、相机、清单和样件元数据绑定哈希。导出的 reviewed JSON 可以另存；不要直接修改这些绑定字段。所有九孔都须确认可见/不确定/遮挡/出视野状态，确认后填写复核人和时间。像素容差应先通过 20 张图两次盲标估计重复性，再冻结；不应为提高正确率而逐次放宽。

## 2. RGB 内参拟合与独立姿态检查

`vision.calibrate` 仍负责手眼标定，不能替代这里的 RGB 内参诊断。

复制 `docs/templates/intrinsics_check.template.json` 为新的实验配置。填写实际标定板内角点列数/行数、实测方格边长（mm）、量具、测量人/时间和板编号。`capture_manifest` 相对于该配置文件；`fit_images`、`validation_images` 相对于采集会话目录，例如 `images/frame_000001.png`。从同一相机、同一固定流配置采集约 30 张拟合/检查图，另留 10 张独立姿态图；覆盖画面中心/边缘、不同倾斜和距离。近邻重复帧不算独立姿态。

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.intrinsics --spec "artifacts/calibration/d435_spec.json" --version d435_rgb_v1 --output-dir "artifacts/calibration/d435_rgb_v1"
```

工具核对原图哈希与实际流参数，提取棋盘格角点，用拟合集拟合 OpenCV Brown 模型，独立姿态只估计每图板位姿，不参与内参拟合。输出 `camera.fitted.json` 与 `intrinsics_report.json`，同时保存 SDK/拟合内参下的逐图、逐角点残差向量和 mean/RMS/max。失败检出的图保留在 `rejected_views`。程序最低要求 6 张拟合、3 张独立图，仅为算法运行下限，不等于采集预算或验收标准。

拟合产物始终 `validated=false`。需人工检查板尺寸、视场覆盖、独立姿态残差和冻结验收阈值后另存已验收版本。共享相机标定的板位姿不是独立绝对位姿真值。

如果独立检查证明 SDK 内参满足需要，保留默认 SDK 路径，用已有的 `intrinsics_validated`、`calibration_version`、`intrinsics_sha256` 绑定当前流。

如果改用拟合内参，另存验收文件，保留 `source_sdk_intrinsics_sha256` 和拟合 `intrinsics_sha256`，添加 `validation_record`：

```json
{
  "reviewer": "实际复核人",
  "reviewed_at": "实际ISO时间并带时区",
  "report_path": "intrinsics_report.json",
  "report_sha256": "实际报告文件SHA256",
  "max_heldout_rms_px": "填写预先冻结的数值阈值"
}
```

验收文件设 `validated=true`；在控制视觉配置中填写 `intrinsics_override_file`、同一 `calibration_version` 和拟合 `intrinsics_sha256`，并设 `intrinsics_validated=true`。上面带说明的字符串不是可运行数值。实时入口检查 D435 型号、完整序列号、color 流、分辨率、帧率、格式、原 SDK 指纹、拟合指纹、验收人/时间、报告哈希和独立集 RMS；任何不符直接拒绝。无 override 时保持 SDK 路径。软件测试使用假 SDK，未开流或移动机械臂。

## 3. 固定点集的相机/几何对照

`vision.diagnose` 接受多个 schema 2 复核文件、多个几何版本以及按相机序列号匹配的内参候选。几何由 `vision.geometry_reference.load_geometry_reference` 核对。`config/geometry_nominal.json` 只提供历史平面名义基准；必须另建指定孔口截面的实测版本，不可把模板填零当测量值。

先运行人工点比较，不需要 YOLO 权重：

```powershell
.venv\Scripts\python.exe -B -X utf8 -m vision.diagnose --annotations "artifacts/reviews/d435_reviewed.json" --geometry config/geometry_nominal.json --geometry "artifacts/geometry/printed_dev01.json" --camera-override "artifacts/calibration/d435_rgb_v1/camera.fitted.json" --tolerance-px 2 --manual-only --output "artifacts/diagnosis/manual_camera_geometry_v1.json"
```

这里 `2` 仅示范参数位置，必须换成盲标研究后冻结的像素容差。去掉 `--manual-only` 即运行现有 detector/轮廓/小孔/孔名关联流程；可重复 `--annotations` 汇总 60 张诊断图。自动前端用第一个几何和各会话 SDK 参数运行一次，然后冻结原始像素点，所有内参/几何变体重用同一组点。若后续要测量“改几何也改变前端关联”的端到端影响，应另跑独立报告，不与这组隔离变量的诊断混算。

报告保存：

- `frontend.candidate_points_raw_px`：PnP 前的所有候选，用一对一匹配计算候选召回。
- `frontend.named_points_raw_px`：孔名关联结果，求解失败也保留。统计正确、错名、歧义、未定位、缺失；正确表示在冻结容差内唯一匹配该孔，不代表绝对位姿真值。
- `comparisons.*.branches`：人工全部、自动全部、同孔名人工、同孔名自动四分支。每孔保存投影点、观测点、`projected_minus_observed` 残差向量和距离。
- `summary`：求解数、最终诊断放行数、失败数；逐图平均距离的统计和全体孔残差统计分开。`paired_human_automatic` 只比较同图同孔且两边均可解的结果。
- `paired_variants_against_baseline`：内参/几何变体与基准使用共同可解图片，避免各方法只挑自己成功图。失败仍在原始行中，不被删除。
- `strata`：按 distance/tilt/image_position、相机型号和序列号分层；未记录工况会明确显示 `unrecorded`。跨相机汇总只用于整体诊断，各机结论查看独立分层。

最终诊断放行另外要求内参和对应打印件几何已验收，且关联/残差/点数满足配置；它不授权机器人运动。实测几何只用于匹配的样件及打印批次，其他样件的几何被明确列为不适用。报告保存图像、标注、几何、内参、配置、源代码、权重和关联先验哈希；已有输出文件拒绝覆盖。

新工具始终标记 `independent_test_set=false`，避免仅凭图像行里的 test 字段宣称独立测试。正式 200 张 D435 测试图还需要打印批次/样件/会话完整隔离与冻结模型，不能混入本轮 60 张开发诊断。低重投影误差不证明孔名正确，更不证明机器人基坐标定位准确；手眼/TCP/力参考点验收仍单独进行。
