# RM75 力觉视觉协同插装程序

运行平台是 Windows、Python 3.12、D435 眼在手和 RM75 内置六维力。程序将视觉预对准、接触监督、风险调度、GRU 有限方向和导纳连接到同一状态机。默认入口只检查本地配置；模拟参数不被真实设备入口接受。

代码实现与离线验证不代表实际接口已能完成插装。真实设备尚需确定控制器代际、固件、TCP、传感器参考点、手眼标定、工具负载、接口尺寸、允许载荷、实测周期和力觉模型。论文中的 UR5e 硬件接口已改为 RM75 适配，控制算法仍按已确认的第2—5章方案实现。

## 环境与快速开始

所有命令从仓库根目录执行。依赖安装到项目 `.venv`：

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-control.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m control.simulate --scenario nominal --output artifacts/nominal.jsonl
.venv\Scripts\python.exe -m control.simulate --scenario blocked --output artifacts/blocked.jsonl
.venv\Scripts\python.exe -m control.replay --input artifacts/nominal.jsonl
.venv\Scripts\python.exe -m control.evaluate artifacts/nominal.jsonl artifacts/blocked.jsonl --output artifacts/task_metrics.json
```

日志和报告使用新文件名，拒绝覆盖旧实验。`.venv`、`artifacts`、`runs` 和模型权重已经由项目 Git 忽略规则排除。OpenCV 固定4.13，因为当前5.0 Python包缺少 `calibrateHandEye`；`requirements.txt` 仍对应原来的 Kinect 环境。

`control.simulate` 是确定性软件夹具，载荷规律和数值均明确为合成参数。可选异常包括 `blocked`、`overload`、`stale`、`disconnect`、`no_contact`、`bad_vision` 和 `risk_outside`。它验证状态、门控、命令限制和停止路径，不是连接器物理精度模型。回放使用记录的实际反馈重新计算决策，不预测换一种控制器后真实系统会如何运动。

## 论文与代码对应

| 论文内容 | 实现 | 验证重点 |
| --- | --- | --- |
| 第2章视觉定位、2.4.6手眼 | `vision/pipeline.py`、`vision/calibrate.py` | 同一帧、有效内参、Tsai求解、独立姿态验证、米制转换 |
| 3.4—3.5工程域和六维风险 | `control/risk.py` | 双向裕度、耦合边界、严格内部参考、域外和无效标志 |
| 4.2力觉统一 | `control/wrench.py`、`control/geometry.py` | 参考点平移、坐标旋转、补偿语义、冻结零偏、双通道 |
| 4.3时序学习 | `control/learning.py`、`control/force_dataset.py` | reset-before GRU、训练集归一化、轨迹/批次隔离、置信度与动作稳定 |
| 4.4导纳与推进 | `control/admittance.py`、`control/state_machine.py` | 真实周期、正阻尼、轴向门控、限幅与内部状态同步 |
| 4.5.3新增协同机制 | `control/synergy.py` | 视觉SI预算、有限长度针位几何、局部动作响应、允许动作集合 |
| 4.5多阶段插装 | `control/state_machine.py` | 实际到达、持续接触、连续接管、实际深度、稳定完成、退出 |
| 第5章实验复算 | `control/logging.py`、`control/replay.py`、`control/evaluate.py` | 任务身份、代码/配置哈希、时间、指令与反馈分开、Wilson区间 |

固定正定质量/刚度与风险调度阻尼使用论文方向：风险增大使阻尼由 `damping_max` 降至正的 `damping_min`；轴向推进另由风险和载荷监督降低或暂停。半隐式导纳每次使用实际间隔，并检查各阻尼端点的保守离散界限。该局部条件不替代机器人内环、时延和真实接触的稳定性验证。

## 视觉和坐标

```powershell
.venv\Scripts\python.exe -m vision.run --config config/realsense_config.json
.venv\Scripts\python.exe -m vision.calibrate --samples artifacts/handeye_pairs.json --output artifacts/handeye_result.json
```

详细配置、离线图像、标定样本和流内参导出格式见 [视觉说明](../vision/README.md)。实际相机运行会访问 D435，应在设备采集获准后执行。新流水线在内存中串行复用原视觉算法，不受旧 `latest_roi` 等文件的更新顺序影响。

视觉消息输出 `T_camera_target`，平移为米。机器人反馈是当前 TCP；程序用独立 `T_flange_tool` 恢复法兰，再计算：

```text
T_base_target = T_base_tool @ inverse(T_flange_tool) @ T_flange_camera @ T_camera_target
T_base_plug_target = T_base_target @ T_target_plug
```

`T_target_plug` 定义真实配合方向、键位和参考点；`R_target_task` 定义任务轴在目标坐标中的方向。任务 +z 必须经实物核验为插入方向，平面九点本身不能确认正负号。针位及纠偏预算都表达在该任务系。手眼、工具安装、RGB内参和视觉预算分别有版本，不混用。机器人与图像按同一主机时基配对；UDP传输时延上界需要独立测量，配对检查覆盖整个可能采样区间。真实入口只接受本次启动后新发布的新观测。

目标是一个完整充电接口，九孔用于内部几何定位；完整九触头充电枪不会逐个插入九个小孔。同一尝试中冻结目标与视觉先验；接触后不把新视觉位姿直接叠加到运动参考。多目标通过独立身份和图像窗口选择，存在歧义就拒绝。

## GRU 数据与训练

先用 A 组固定导纳等经过验收的采集流程记录实际轨迹，再依据独立有限动作试验标注有效方向。不能把GRU自己选择的动作当作正确标签，也不能把故障塞进Hold类别。

```powershell
.venv\Scripts\python.exe train\python\prepare_force_dataset.py --annotations artifacts/force_annotations.json --output artifacts/force.jsonl
.venv\Scripts\python.exe train\python\train_force_gru.py --data artifacts/force.jsonl --split artifacts/force_split.json --output artifacts/force_gru.pt --model-version measured-v1
.venv\Scripts\python.exe train\python\evaluate_force_gru.py --data artifacts/force.jsonl --split artifacts/force_split.json --model artifacts/force_gru.pt --output artifacts/force_test.json
```

标注JSON的结构：

```json
{"schema_version":1,"logs":[{"path":"attempt.jsonl","trajectory_id":"t1","batch_id":"b1","intervals":[{"start_s":0.1,"end_s":0.5,"label":"+x","basis":"独立有限动作试验记录编号"}]}]}
```

起止时间相对于该日志第一个控制周期。每条训练记录有 `trajectory_id`、`batch_id`、`timestamp`、`wrench`、`label`、`frame="task"` 和单位数组 `['N','N','N','Nm','Nm','Nm']`。动作固定为 `+x,-x,+y,-y,+Rx,-Rx,+Ry,-Ry,Hold`；`ambiguous` 和 `invalid` 窗口会排除。划分文件包含 `train`、`validation`、`test` 三个轨迹ID列表，同一采集批次不跨集合。

训练只用训练集拟合标准化参数，用验证集选权重，测试集另行评价。模型保存窗口时长、采样周期、类别顺序、标准化、方程定义、数据哈希和版本。推理检查这些元信息，缺模型/窗口不足/过期/低置信度时不产生学习偏置，持续Hold超时进入停止路径。训练结果默认 `validated_for_hardware=false`；实际使用需独立验证后生成有依据的验收记录和相应模型版本，不能由训练准确率自动批准。

`train/python/accept_force_model.py --model REAL.pt --record acceptance.json --output ACCEPTED.pt` 可将有签署信息的独立验收记录、原权重哈希与实际硬件日志哈希绑定到新模型。它检查证据身份和完整性，不替代人工实验判断；合成模型始终不能通过。记录结构和必填项见该脚本的参数校验。

仅验证软件管线的合成演示：

```powershell
.venv\Scripts\python.exe train\python\create_demo_force_dataset.py --output-dir artifacts/force_demo
.venv\Scripts\python.exe train\python\train_force_gru.py --data artifacts/force_demo/force.jsonl --split artifacts/force_demo/split.json --output artifacts/force_demo/gru.pt --model-version synthetic-pipeline-v1 --window-size 8 --max-gap .035 --epochs 10 --synthetic
.venv\Scripts\python.exe -m control.simulate --method F --model artifacts/force_demo/gru.pt --output artifacts/synthetic_F.jsonl
```

合成模型会被真实设备入口拒绝；合成分类得分不属于真实力觉性能。

## A—F 方法和协同约束

| 方法 | GRU | 风险调度 | 接触阶段视觉约束 |
| --- | --- | --- | --- |
| A | 关闭 | 关闭 | 关闭 |
| B | 关闭 | 开启 | 关闭 |
| C | 开启 | 开启 | 关闭 |
| D | 开启 | 关闭 | 关闭 |
| E | 开启 | 开启 | 关闭，分阶段对照 |
| F | 在允许集合内开启 | 开启 | 开启 |

C/E共享控制关系，E是与F配对的实验身份。全部组保留相同的独立载荷保护、数据有效性、运动约束和完成判据。比较需要冻结共同参数与同一GRU版本，按完整任务统计，不能把采样点或多次尝试充当独立任务。

论文的 `g_v/g_g/g_f` 给出语义而未给全套数值规则。当前工程实现为：`g_v` 检查独立校准的SI纠偏包络，`g_g` 检查有限长度直针两端的径向间隙和键位转角，`g_f` 检查经标定的单次动作载荷增量对工程风险的影响。动作响应必须匹配工程域版本和动作步长。它是局部工程近似，不是一般多点接触预测或卡滞概率。未有实际校准依据时F组不能上机。

工程受力域可用以下命令从有来源的正负轴向限值和耦合约束生成：

```powershell
.venv\Scripts\python.exe -m control.build_risk_domain --spec artifacts/risk_spec.json --output artifacts/risk_domain.json --records artifacts/risk_validation.jsonl
```

`--records` 可省略。spec包含六通道 `limits`（每通道 `lower/upper/lower_source/upper_source`）、`sources`（证据ID、版本、文件）、物理单位的 `reference_wrench`、坐标/样件/工具/力觉版本，及可选归一化空间耦合行。完整格式见 `control/build_risk_domain.py`。工具验证证据哈希并报告独立记录的域内/域外比例，不根据观测峰值推断允许载荷，输出始终 `calibrated=false`；正式校准验收须另外完成。

## RM75 接入与验收顺序

先从 `config/control.template.json` 建立本次实验配置，所有null都需要真实来源。`config/simulation.json` 的数值不得复制为实机限值。准备好后可先做完全离线的检查：

```powershell
.venv\Scripts\python.exe -m control.run --config config/control.template.json --check
```

该命令默认返回 `ready=false` 并列出缺项，不连接设备。控制器代际、SDK/固件、六维力类型、当前TCP和工作系都必须核对；适配器不自动更改TCP、重力标定或零点。连接时临时设置UDP反馈，退出恢复原UDP配置。

后续完成现场确认与授权后，分别使用两个明确入口：

```powershell
# 读取和记录反馈，无运动命令；仍会建立连接并临时配置UDP
.venv\Scripts\python.exe -m control.run --config config/control.local.json --collect --duration 10 --output artifacts/feedback_check.jsonl
# 使用已经验收的配置执行单次真实机械插装
.venv\Scripts\python.exe -m control.run --config config/control.local.json --execute-motion --output artifacts/real_attempt_001.jsonl
```

控制程序按显式配置读取sensor坐标的系统外力，再通过传感器到插头的完整力旋量变换处理力矩参考点。不会把“工具轴表达”自动解释为“力矩已平移到插头尖端”。已补偿外力不重复扣重力；原始力只有显式负载、质心和重力参数时才补偿。

真实发送周期与UDP周期分别验收。Windows/Python不保证实时性，反馈过期、计算超时、发送失步和追踪偏差都触发停止。API停止应答与实际静止反馈分别记录；软件停止不等同于硬件急停。

本版每次运行最多一次尝试，错误后停止且锁存发送禁止。状态机提供显式条件退离接口用于受控验证；真实入口不自动解除停止锁存或重试。只有解除接触、检查状态并重新视觉定位后才能开始新任务。未接入锁止或电气反馈时，成功仅称为机械插装完成。

首次实机顺序：设备/坐标/载荷核验 → 无接触低速参考与明确停止 → 固定导纳 → 接触阈值与限幅 → 工程域独立校准 → 力觉样本和GRU独立验证 → A—F完整插装比较。每个阶段保存原始反馈和验收配置。
