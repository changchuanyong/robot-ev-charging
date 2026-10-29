# 打印件实验与力觉数据工具

这些工具组织完整任务和记录证据，不会执行打印件测量，也不会凭空生成实验结果。正式设计为 A/B/C/F × 13 工况 × 10 次，共 520 次。E 复用 C；两套测试件在每格各承担 5 次。所有初始偏差、速度、有限步长、观察时间、载荷和成功阈值必须由独立预试验确定。

## 1. 生成待办任务

在项目根目录使用现有虚拟环境：

```powershell
.venv\Scripts\python.exe -B -m control.experiments schedule --protocol config/experiment_protocol.template.json --output artifacts/experiment_plan/new_draft_schedule.json
```

已交付 `artifacts/printed_stage1/schedule_520_v2_pending.json`，其中 520 项全部为 `pending`。随机种子固定；每个工况/重复号形成包含四种方法的配对区组，区组内顺序随机，各轮工况顺序随机。任务 ID 绑定协议内容；改变协议后必须生成新任务表。工具不覆盖已有输出。

`config/experiment_protocol.template.json` 中的物理偏差和容差为 `null`。填入实测值、打印批次、几何版本和预试验证据后，使用四份硬件配置冻结协议：

```powershell
.venv\Scripts\python.exe -B -m control.experiments freeze --protocol artifacts/experiment_plan/measured_protocol.json --config-a artifacts/experiment_plan/control_A.json --config-b artifacts/experiment_plan/control_B.json --config-c artifacts/experiment_plan/control_C.json --config-f artifacts/experiment_plan/control_F.json --geometry-references artifacts/geometry/test01_measured.json artifacts/geometry/test02_measured.json --output artifacts/experiment_plan/frozen_protocol.json
.venv\Scripts\python.exe -B -m control.experiments schedule --protocol artifacts/experiment_plan/frozen_protocol.json --output artifacts/experiment_plan/frozen_schedule.json
```

上述实测协议、两套样件各自的几何文件和四份配置尚需实验者准备。冻结会验证硬件配置，并要求除 `method` 外参数一致；消融差异由控制方法开关产生。冻结保存配置、代码、GRU 和两套实测几何文件哈希，并检查几何的样件/打印批次对应。缺少校准的模板、合成配置、不同规则版本不能冻结为正式实验。

## 2. 单次完整任务的追溯

复制 `config/experiment_setup.template.json`，记录任务 ID、会话、重新装夹/换件、样件累计插拔次数、测量依据和实际初始 `{x,y,Rx,Ry}`。只有真实复核后才设置 `verified=true`。工具核对实际初始偏差是否位于冻结容差内；它不会自动把机械臂移动到试验偏差。

```powershell
.venv\Scripts\python.exe -B -m control.run --check --config artifacts/experiment_plan/control_C.json --schedule artifacts/experiment_plan/frozen_schedule.json --scheduled-task TASK_ID --setup-record artifacts/experiment_plan/setup_TASK_ID.json
```

`--check` 不连接机械臂。正式执行仍须显式使用现有 `--execute-motion --output 新日志路径`；本次实现未执行真机动作。日志 schema 2 保存唯一任务、规则、条件、样件、配对区组、配置/源代码/模型哈希，以及每周期指令与实际反馈。修改冻结代码或模型后必须建立新协议。重复任务文件或 E/C 同区组重复样本在汇总时拒绝，失败、停止、超时和没有收到反馈的已启动任务保留。

```powershell
.venv\Scripts\python.exe -B -m control.evaluate artifacts/tasks/task1.jsonl artifacts/tasks/task2.jsonl --output artifacts/reports/new_summary.json
```

报告同时给出方法总体和工况/样件分层结果，保留规则、配置、代码、模型、协议和数据来源隔离。指标包括成功数/总数、Wilson 95% 区间、时长、轴向/横向力及合力矩峰值、有限试探有效率、视觉检查拒绝率。有效率分母包含已开始但未确认的试探；视觉拒绝率是实际视觉检查数的比例，不是图像数据集检测准确率。每格只有 10 次，不能忽略区间的不确定性。

## 3. 力觉数据与独立标签

有限试探使用控制循环原有 `WrenchProcessor` 和 `cycle.details.filtered_wrench`，包含工具参考点、补偿和版本。需要无动作多姿态力采集时，准备已校准的 `config/force_collection_frame.template.json` 副本：

```powershell
.venv\Scripts\python.exe -B -m control.run --collect --config artifacts/calibrated/control.json --collection-frame artifacts/calibrated/force_collection_frame.json --duration 10 --output artifacts/force/new_static_pose.jsonl
```

此模式不发运动指令；除原始 `sample` 外写入 `COLLECT` 周期和同一处理器的力。它不是插装任务，不能计入 520 次。无任务坐标标定时仍可使用原始采集，但原始力不能直接进入 GRU 训练。

首批方向数据尚无真实 GRU 时，使用显式 `--probe-action` 选择一个方向。准备方法 C 的实测配置，并在 `experiment` 中填写 `session_id`、`specimen_id`、`printing_batch_id`；它不能绑定正式任务或协议。此模式只替换方向选择来源，继续使用同一力处理、机械/当前风险约束、有限试探、实际推进确认和停止反馈。开始试探前暂停名义推进并积累动作前历史，首次试探有结果或发生异常后立即停止，不继续完整插装，也不自动反向。

```powershell
# 先检查配置，不访问硬件；下列本地配置需要根据实测记录准备
.venv\Scripts\python.exe -B -m control.run --config artifacts/calibrated/probe_C.json --probe-action +x --check
# 仅现场验收并明确启用运动时执行一次；每次使用新输出路径
.venv\Scripts\python.exe -B -m control.run --config artifacts/calibrated/probe_C.json --probe-action +x --execute-motion --output artifacts/force/probe_x_001.jsonl
```

方向取 `+x/-x/+y/-y/+Rx/-Rx/+Ry/-Ry`，负方向命令建议写为 `--probe-action=-x`。日志标记 `PROBE` / `force_direction_calibration`，单热权重仅代表操作者请求，不是 GRU 置信度。普通 C/F 入口仍要求已独立验收的真实 GRU；配置文件中的试探标记不能代替显式参数。试探记录支持回放及下面的标注流程，但被正式任务统计拒绝。`Hold` 从经复核的适当正常推进记录取得。

从有限试探日志生成待人工复核的预动作区间：

```powershell
.venv\Scripts\python.exe -B train/python/prepare_force_dataset.py --review-logs artifacts/force/task.jsonl --output artifacts/force/review.json
.venv\Scripts\python.exe -B train/python/prepare_force_dataset.py --annotations artifacts/force/reviewed.json --output artifacts/force/dataset.jsonl
```

第一条命令仅写 `invalid` 标签和建议动作，绝不自动将 GRU 决策变成真值。复核者填写打印批次、样件、会话、轨迹、身份、依据和标签。schema 2 的方向标签必须关联日志中的同方向 `effective` 试探，真实推进达到冻结阈值，且载荷允许；载荷下降或 `unconfirmed` 都不足以支持正标签。方向训练窗口在试探开始前结束，避免把动作后的结果泄露给分类输入。`Hold` 必须有允许载荷下的正常推进证据；故障使用 `fault`，歧义使用 `ambiguous`，都不作为 Hold。

训练前按完整轨迹划分 `train/validation/test`，校验打印批次、样件、采集会话与 acquisition batch 不跨集合。重叠窗口在同一轨迹内可以存在，但不能跨集合。旧 schema 1 仍可读，保留旧来源和版本，不能声称具备新数据的批次/样件证据。

目前交付的是工具及合成回归验证，没有采集新的真实力轨迹、完成 GRU 硬件验收或执行正式 520 次任务。
