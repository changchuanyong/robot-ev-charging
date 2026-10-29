# RM75 打印件插装推进与验收

本轮软件规则为 `schema_version=2 / rules_version=finite-trial-v2`。对象是 RM75 操作打印枪座、D435 眼在手、D415 辅助采图。结果仅用于打印件机械插装，不推断真实车辆接口的电气、锁止或耐久性能。第一阶段目标是实测几何与新相机误差诊断；当前只交付软件和待采集模板，不能把旧照片或合成记录计为新实验。

## 当前可运行的入口

所有命令在仓库根目录运行。输出使用新路径，保留旧报告。

| 工作 | 入口与配置 | 产物 |
| --- | --- | --- |
| 相机采集 | `vision.collect`，`config/data_capture_d415.json` 或 `data_capture_d435.json` | 原始 PNG、实际 RGB 内参、会话/样件/批次/工况与哈希 |
| 内参拟合和独立图像检查 | `vision.intrinsics` | SDK/拟合内参逐角点残差，拟合图与保留图分开 |
| 新图误差诊断 | `vision.diagnose` | schema 2 人工标注，冻结同图同孔点，比较相机/几何/自动点 |
| 几何检查与 CAD 截面 | `vision.geometry_reference` | 几何版本与源文件哈希、未分配孔名的截面候选 |
| 两次盲标重复性 | `vision.annotation_repeatability` | 可见状态分歧与点差分布，不自动冻结评价容差 |
| 有限试探与回放 | `control.state_machine`、`control.finite_trial`、`control.replay` | 试探、观察、推进确认及异常结果 |
| 力觉标注、训练 | `train/python/prepare_force_dataset.py` 与现有 GRU 入口 | 试探前输入窗与独立实测反馈对应，按完整轨迹分组 |
| 520 次编排与统计 | `control.experiments`、`control.evaluate` | 待执行任务、冻结协议、分层成功率与失败记录 |
| 导纳解析对照 | `control.validate_admittance` | 三种阻尼响应、步长收敛；不是接触稳定性验收 |

采集、内参与诊断的完整格式见 [相机与几何诊断](camera_geometry_diagnostics.md)。实验与力觉日志的格式见 [实验工具](printed_experiment_tools.md)。真机接入顺序见 [控制说明](control_usage.md)。

## 第 1 周：建立真实基准

1. 填写 `config/specimen_registry.template.json` 的副本。3 套开发、1 套验证、2 套正式测试；测试件 ID 与实验协议保持一致，另备替换件。打印批次不跨数据集合，记录材料、层向、CAD 版本、重新装夹与累计插拔次数。
2. 填写 `config/print_measurements.template.json` 的副本，保留原始重复测量。逐孔明确“可见内孔口”的截面、三维中心、孔径/针径、台阶、有效配合长度和实际插入深度。标称孔距不等于修改后的打印件实测值。
3. 将确认后的三维中心写入 `config/geometry_reference.template.json` 的副本。模板故意包含 null，不能通过加载检查。`config/geometry_nominal.json` 只用于与旧模型对照，`validated=false`。正式几何使用 `status=measured_print`，附样件、批次、测量人与证据哈希。
4. D415、D435 各采约 30 张标定拟合/检查图和 10 张独立姿态图。棋盘内角点数和方格边长依据实物填写；工作距离、视野边缘和倾斜方向均需覆盖。SDK 参数先保存，再检查，不能复制另一台相机的内参。
5. 新拍并复核 60 张打印接口诊断图。建议按 3 个距离层 × 2 个倾斜层 × 2 个图像位置层 × 5 张分配，并保存各层实际距离/角度。距离和角度取值由工作空间和清晰度确定，本文件不替代测量。至少包含 D435；D415 结果单独分层。
6. 20 张开发图做两次盲标，以明确的复核记录确认没有查看前一遍坐标；先检查可见状态分歧，再冻结像素容差。不要用独立测试图调容差。

可立即离线检查旧几何并记录 CAD 源：

```powershell
.venv\Scripts\python.exe -m vision.geometry_reference check config/geometry_nominal.json
.venv\Scripts\python.exe -m vision.geometry_reference audit --cad-dir '..\模型' --output artifacts/stage1_new/cad_sources.json
```

截面提取例子（这里只演示明确选择 CAD 的 y=44.25 mm 平面，不代表已经选定实际孔口）：

```powershell
.venv\Scripts\python.exe -m vision.geometry_reference section --mesh '..\模型\插座改.STL' --axis y --level 44.25 --unit mm --output artifacts/stage1_new/section_y44_25.json
```

STL 不含单位；`--unit mm` 是对导出单位的显式声明。截面可能同时包含外轮廓、台阶和孔壁，工具保留全部闭合候选及圆拟合残差，不自动分配九孔身份。通过顶点的歧义平面会被拒绝；需要选内侧截面并记录其与实际孔口平面的偏移。CAD 圆心也不证明打印尺寸。

盲标比较命令中输入路径和记录号需要替换为实际文件：

```powershell
.venv\Scripts\python.exe -m vision.annotation_repeatability --first artifacts/blind_a.json --second artifacts/blind_b.json --blind-record '实际盲标流程记录号' --output artifacts/repeatability.json
```

## 第 2 周：诊断与软件验证

固定 60 张图及孔名，分别比较 SDK/独立检查后的内参、旧几何/实测几何、人工/自动点。各相机变体只应用于同序列号和同分辨率，实测几何只能应用于匹配样件。自动提取点在相机/几何对照中冻结，避免改动多个因素。报告逐孔投影减观测的残差向量、每图 mean/RMS/max，以及明确注明口径的聚合值。

候选召回在孔名关联前统计；命名正确、错误、缺失在 PnP 前统计；PnP 求解与最终放行分别统计。低残差不证明孔名正确，少量通过样本不代表总体改善。正式优化优先针对经诊断确认的小孔漏检、遮挡拒绝和对应不稳定。本次没有新建大型关键点网络，也没有在缺少新图时宣称残差已降低。倾斜条件的残差规律若支持圆孔透视偏心假设，再做专项补偿验证。

手眼误差不会修正相机坐标内的 PnP 残差。RGB 内参检查、三维孔口模型和手眼/TCP 标定分别记录版本。

```powershell
.venv\Scripts\python.exe -m control.validate_admittance --output artifacts/admittance_response_new.json
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m control.simulate --scenario feedback_lag --output artifacts/feedback_lag_new.jsonl
.venv\Scripts\python.exe -m control.simulate --scenario period_jitter --output artifacts/period_jitter_new.jsonl
.venv\Scripts\python.exe -m control.replay --input artifacts/feedback_lag_new.jsonl
```

`trial_timeout`、`observation_drift`、`confirmation_blocked` 用学习方法和明确标记的合成 GRU/受控测试策略覆盖。不能把一个从不触发纠偏的仿真视为已验证试探流程。合成演示只检查控制分支和记录一致性，不代表接触物理真实性。

## 第 3–4 周：实测标定与方向数据

约 20 个静止配对姿态求解手眼，另留 10 个姿态检查。记录实际法兰/TCP 定义、图像/反馈时钟、工具安装与力矩参考点；普通标定板的一致性不等于外部绝对真值。完成无接触运行、受控停止、多姿态静止力、轻触及固定导纳基线后，才开展有限试探。

C/E/F 共用机械、键位、运动和当前风险准入。F 仅增加 `{x,y,Rx,Ry}` 四分量视觉修正域及无效先验处理。旧的动作后六维载荷预测不再作为 F 准入。新试探锁定动作、起点、终点和截止时间；试探及观察阶段冻结名义轴向推进，不能重新选动作或重置计时。载荷降低仅作释放线索，允许载荷下的实际推进才支持有效标签。失败不自动反向纠偏。

每个方向类首轮安排 30 条训练、10 条验证、10 条测试的独立记录；Hold 同样需要经复核的适当推进证据。重叠时间窗、整条轨迹、会话、样件和打印批次不跨集合。故障、歧义、无效试探不塞入 Hold。力输入来自控制循环的同一处理链和 `cycle` 日志。

首批方向数据不依赖已有真实 GRU：使用 `control.run --probe-action` 显式指定单次方向，仍经过共同有限试探执行器、实际反馈确认与停止检查；这些日志标记为标定采数，排除在 520 次统计之外。完整配置与命令见实验工具说明。该入口已作为软件实现，不代表本次已启动真机采数。

视觉总预算为 400 训练、100 验证、200 独立 D435 测试图；这些是采集目标。历史 79 张含训练图，既不是新 60 张诊断图，也不能计作独立测试。真实接口照片只能按来源标记补充外观训练，没有对应几何时不能验证打印件位姿。

## 第 5–6 周：正式 520 次任务

预试验独立完成后冻结偏差幅值、速度、步长、观察时间、载荷边界与深度/姿态/保持完成判据。`config/experiment_protocol.template.json` 故意不填物理数值；模板可生成编排，不能冻结或执行正式任务。

```powershell
.venv\Scripts\python.exe -m control.experiments schedule --protocol config/experiment_protocol.template.json --output artifacts/schedule_520_new.json
```

编排为 A/B/C/F × 13 工况 × 10 次，共 520 个唯一任务，E 复用 C。每个工况和重复编号形成四方法区组，方法顺序交错；两套测试件各承担每格 5 次。运行前核对初始偏差、样件、会话、磨损次数和装夹记录。重试不能覆盖原任务，超时、停止、未完整结束的记录都保留。

成功只按实际插入深度、姿态和持续保持确认。报告每组成功数/总数及 Wilson 区间、力/力矩峰值、完成时间、试探有效率、视觉拒绝率，并按工况、样件、配置和代码版本分层。每格 10 次仍有较大不确定性，不能只列百分比。普通量具不足以支持预设的“亚毫米绝对定位”结论。

软件仿真与回放是必需项；完整 MuJoCo/Gazebo 接触模型本轮不进入关键路径。若后续补充，使用实测几何和摩擦/刚度范围作敏感性分析，其结果不替代真实力觉标签和 520 次任务。

## 第一阶段交付的边界

`artifacts/printed_stage1/` 保留本次 CAD 源清单、截面候选、解析导纳检查、520 条待执行编排和待采集状态报告。实测几何、60 张新复核图、两机内参验收、手眼/工具验收、真实力觉模型与 520 次任务尚未完成。这些缺项应在采集后用新的报告版本关闭，不以模板中的数量或合成结果代替。
