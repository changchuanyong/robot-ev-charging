# RGB 曝光标定

## 为什么要动曝光

相机默认自动曝光对整幅画面测光。黑色接头放在亮桌面上时，背景把测光结果拉高，接头内部的孔口只剩 2–10 灰阶差。实测同一批图：有孔位输出的图片孔口对比度中位 33 灰阶，零输出图片中位 9 灰阶；115 张剩余失败图里有 22 张黑场已经被压到 0–2，信息不可恢复。

librealsense 的彩色传感器**没有测光区域（ROI）选项**（v2.59.1 的 `rs.option` 枚举里没有任何 ROI 项），所以"让自动曝光只对准充电口"做不到。可选的三条路：

1. 关闭自动曝光，固定曝光 + 固定增益（最可控）；
2. 保留自动曝光但限幅（`auto_exposure_limit`，需传感器支持，运行时用 `supports()` 判断）；
3. 关闭自动白平衡并固定 `white_balance`，修蓝偏（`06d863ab` 那类图）。

## 工具

```powershell
python -m ev_charging.capture.exposure_calibration        # 或 ev-calibrate-exposure
```

工具复用采集链路的 `RGBControls`：读取每个选项的实际范围与步进、写入后回读、结束或异常时还原原值（还原失败会报错，不会静默留在手动曝光）。

- 扫描 `exposure × gain`，每个组合取 `--frames` 帧的中值合成，压掉时域噪声；
- 打分：对 ROI 做尺度归一的环形匹配滤波，取最强的 5 个峰，去掉低于最大值 25% 的纹理峰后求均值，再乘截断惩罚 `max(0, 1 − 2·clip_low − clip_high)`；
- 结果写 `artifacts/exposure_calibration/report.json`，终端打印可直接粘贴的 `rgb_options`。

打分只看"孔径在这个曝光下是否可测"，不需要标注，也不依赖检测模型。

## 现场步骤

1. 固定相机与接头的工作距离、角度和环境光，和正式采集时保持一致。
2. 确认序列号：`python -m ev_charging.capture.exposure_calibration --list-devices`
3. 扫描（`--box` 圈住接头；不确定就先不传，默认取画面中央）：

   ```powershell
   python -m ev_charging.capture.exposure_calibration --serial <SN> --box 400,150,900,600 --save-frames
   ```

4. 打开 `artifacts/exposure_calibration/frames/*.png`，挑孔口清晰、背景不过曝的一张，与终端表格里的 `score`、`clip_low`、`clip_high` 对照确认。
5. 把终端输出的块粘贴到 `config/d415.json` 的 `profiles.collect.rgb_options`：

   ```json
   "rgb_options": {
     "enable_auto_exposure": 0,
     "exposure": 150.0,
     "gain": 16.0
   }
   ```

   采集工具会自动校验范围与步进、写入后回读，并在结束时还原；采集记录里带 options 快照。
6. 采集后抽查首张图：孔口对比度应从个位数灰阶升到两位数。

## 边界

- **只影响之后采集的图片**。已有 326 张已定型，其中 22 张黑场截断的无法恢复。
- 固定曝光对环境光敏感；换场地、换距离要重标。这也是为什么默认配置仍是自动曝光。
- 改曝光后新图与历史图分布不同：训练、评估、图库统计不要混用；建议升采集版本并在记录里标注。
- 本工具不改变在线识别路径。`RealSenseCamera` 目前既不应用 `rgb_options`，也不把生效曝光写进 camera 元数据；实时识别看到的仍是设备默认曝光。要统一需单独接，属于在线采集行为变更。
- `profiles.collect` 现在带 `rgb_options`（默认 `enable_auto_exposure: 1`，与改动前行为相同）。代价是采集启动时会实例化 `RGBControls`：若彩色传感器报告不支持其中任一选项，采集会直接拒绝启动并报 `unsupported_RGB_option`，不会静默降级。

## 相关

- 已有的图库失败分诊：`artifacts/vision_improvement_20261104/triage_v3/triage.html`（112 张零输出按曝光成因分类）
- 离线后期的四级回退（不能替代采集侧修正）：`docs/vision_improvement_round2.md`
