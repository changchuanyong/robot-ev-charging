# 人工复核后的视觉改进

2026-10-27：使用用户返回的 24 张人工复核标注改进检测与孔名关联，默认入口已启用。原有 79 张图和这 24 张标注都是开发数据，不是独立测试集。

## 实际改动

- 修正 `train_port_0018、0043、0054` 三张图的接口检测框，按原参数在 CPU 上训练 100 epoch。新权重为 `runs/detect_reviewed/weights/best.pt`，旧权重保留。
- `observed_layout.py` 仅用已复核 train 图学习二维外观布局；16 张 train 中 14 张满足 DC、PE、CC2 可见及点数条件，8 张 val 不参与模板拟合。val 已用于开发检查和检测器选模，仍不能称为独立验证。
- 双 DC 孔、PE 和 CC2 的孔径及相对位置确定孔名，一对一分配实际候选。外圈轮廓可支持布局假设，但没有内孔口证据的中心不作为测量输出。模板不会补造缺失孔。
- 先确定孔名，再用原相机参数和原三维 CSV 求解 PnP；不会通过重投影残差重排孔名。相机参数、三维几何和 PnP 求解器均未修改。
- 批处理、D435 同帧入口和原文件式候选提取入口复用新关联。原文件式质量检查会拒绝歧义/不合格关联。D435 的原有标定、时效和机器人执行条件继续生效。
- 增加检测框覆盖检查、直接标注回归评估，以及 Windows 中文路径下的批处理图片读写支持。

## 24 张人工复核图的结果

按固定 10 px 开发诊断容差判定正确点。每行的正确、错点、缺失之和均为 190 个可见参考点；这里沿用旧报告的“返回位姿时保存的命名点”口径，包含质量门槛未通过的位姿。

| 方案 | 正确点 | 错点 | 缺失 | 遮挡孔误输出 | 与标注明确冲突却放行的图片 |
|---|---:|---:|---:|---:|---:|
| 原检测器、原关联 | 106 | 48 | 36 | 11 | 6 |
| 原检测器、新关联 | 119 | 0 | 71 | 0 | 0 |
| 新检测器、新关联（默认） | 118 | 0 | 72 | 0 | 0 |

默认版本 train 为 74 正确 / 0 错 / 49 缺失，val 为 44 / 0 / 23。双方共同输出的 114 个同图同名可见点，平均中心偏差从 27.684954 px 降至 2.124018 px。新返回的 118 点平均偏差为 2.142503 px，中位数 1.852779 px，最大 6.596188 px；这些像素偏差不是三维定位精度。

质量门槛通过图从 9 降至 5：原 9 张中 6 张与标注明确冲突；新 5 张无此冲突，其中 4 张覆盖全部可见孔，1 张仍缺孔。位姿可求解图从 20 降至 16。主要收益是减少误认和错误放行，召回仍不足，不能只用“零错点”概括结果。

新检测框覆盖全部 190 个可见点，实际 ROI 完整覆盖图从 23/24 增至 24/24，修复 `0043` 的 DC_pos 被裁掉问题。不过 ROI 改变也影响后续轮廓，因此新检测器比“旧检测器＋新关联”少返回 1 个正确点、少 1 张门槛通过图；没有把这次重训描述成所有指标都提高。

标注中 167 个原 AI 目视中心经用户确认保留，另补充了 23 个可见中心。它们是人工复核的图像参考，不是独立高精度物理测量。

## 79 张开发图

默认版本：51/79 返回位姿，10/79 通过离线几何门槛，综合状态 6 正常、4 警告、69 失败。失败包含主动拒绝，不能把“可求解”或“重投影小”当成定位正确。原优化版为 65/79 返回位姿、14/79 通过门槛。

默认版所有返回位姿的平均/中位/最大重投影残差为 10.285975 / 10.430807 / 17.075445 px。原优化版平均为 12.011638 px；两者返回图片和点数不同，这不是同样本精度改善证据。只有 24 张有人工复核，其他 55 张不能据此宣称无错点。原 Kinect 占位内参仍不具备真机验收效力。

## 运行与复现

在项目根目录使用现有虚拟环境，默认选择新权重和新关联：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/batch_pipeline_eval.py --max-images 79 --output-dir artifacts/batch_eval_reviewed --save-visualizations
```

与冻结基线比较人工孔名和中心，而非仅比较残差：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/evaluate_labeled_regression.py --annotations artifacts/keypoint_review_pilot/human_20261027_849095b2/reviewed_annotations.json --before artifacts/batch_eval_optimized/latest_details.json --after artifacts/batch_eval_reviewed/latest_details.json --output artifacts/batch_eval_reviewed/labeled_comparison.json
```

消融对照可用 `--model runs/detect_retrain/weights/best.pt` 保留旧检测器；再加 `--legacy-layout` 使用原关联。D435 使用 `config/realsense_config.json` 中的 `weights` 和 `observed_layout_prior`，将后者设为 `null` 可显式使用原关联。没有连接或移动硬件。

原 Kinect 多进程启动器未做本轮整链路运行验收；其上游文件读写脚本仍存在中文路径兼容问题。本轮修复并验证的是批处理和候选提取入口的图片读写，日常离线检查使用上方批处理命令。

重新拟合二维先验时应使用新文件名，工具拒绝覆盖已有先验：

```powershell
.venv\Scripts\python.exe -B -X utf8 train/python/fit_observed_layout.py --annotations artifacts/keypoint_review_pilot/human_20261027_849095b2/reviewed_annotations.json --output artifacts/observed_layout_prior_refit.json
```

检测器训练参数保存在 `detector_training_manifest.json`。下面使用相同参数和另一个输出名复现；原 `train_port.py` 仍读取旧 `config/train_config.json`，不会自动更新新推理权重：

```powershell
.venv\Scripts\python.exe -B -X utf8 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='yolo_port/dataset.yaml', epochs=100, imgsz=640, batch=8, device='cpu', workers=0, project='runs', name='detect_reviewed_repeat', exist_ok=False, seed=0, deterministic=True)"
```

本地证据目录 `artifacts/supervised_vision_20261027/` 包含训练/检测框/ROI 覆盖报告、两组消融报告与可视化、人工回归对照及测试日志。`final_comparison.md` 是最终人工对照，`final_default/latest_details.json` 是最终默认配置重跑的 79 图报告，已核对源码与先验哈希。原标注 SHA256 为 `849095b2ae9ce1a9c1d0199495b7655843ce7fb475c8b09e4039abeda01bb601`，新权重 SHA256 为 `9fb0889076f1e85485ca9bb6158f5cb559deed1457ff1889f7fd6938f3906264`。

236 项测试通过，含实际开发照片的批处理/同帧/文件式孔名与坐标一致性、文件式 main 的图片编解码、歧义拒绝、遮挡不补点、仅 train 拟合、固定输入与标注回归检查。默认配置的批处理单图冒烟也通过。`runs/` 和 `artifacts/` 仍被 Git 忽略；复制程序到别处时需一并复制本地新权重。

下一步主要针对剩余漏点：补充被遮挡、倾斜、模糊条件下的人工中心，改善孔内口提取；随后用独立新拍图评估泛化，用实际 D435 内参和实测三维孔口基准验证位姿。
