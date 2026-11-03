# 待核实项目与保留边界

以下代码未因本轮静态搜索无调用而删除。外部脚本、历史复现需求和真实设备行为尚未完整核实。

| 位置 | 保留内容 | 阻塞原因与下一步 |
| --- | --- | --- |
| `vision/algorithms/layout_prior.py` | `layout_reject_log`、`find_main_big_hole_pair`、`match_candidate_to_layout_zone`、`draw_layout_prior` | 未找到当前仓库调用；需确认外部调试与历史复现入口后再决定退役 |
| 同上 | `MISSING_ZONE_RECOVERY_ACCEPT_SCALE`、`MISSING_ZONE_RECOVERY_NAMES`、`ENABLE_TOP_BACKUP`、五个 `TOP_BACKUP_*` 阈值 | 当前实现没有读取这些开关，暂不应宣称改动它们能改变算法；外部使用待核实 |
| `vision/algorithms/enhance.py` | `build_vis` | 可能是手工调试入口，待确认 |
| `vision/algorithms/pnp2.py` | `build_per_point_errors` | 外部评价脚本使用情况待确认 |
| `vision/launch.py` | `enable_security_privilege`、`_self_command` | 旧启动器与外部调用情况待确认；不影响当前相机权限控制 |
| `control/config.py` | `hardware_template` | 可能用于外部生成硬件配置；不可仅凭仓库内无调用删除 |
| `vision/pipeline.py` 与对应点选择 | 最终 PnP 重算 | 当前真实图基线缺少足够成功位姿、歧义对应和校正畸变样本；保留求解时机及误差口径 |
| `control/run.py`、`learning.py` | 检查点再次加载、部分重复安全校验 | 缺少完整 C/F 硬件验收模型夹具；合并前必须证明实际加载模型与验收元数据仍严格绑定 |
| `runtime.py` | `serialise_label_scan` 直接调用契约 | 项目训练入口已使用可恢复作用域；旧直接调用函数暂保留，外部调用方需迁移后再退役 |
| `capture/intake.py` | 全 manifest 遍历与记录 stat | 当前 JSON 格式及跨进程外部修改可见性要求仍需要核对；已减少重复解析与未变化写入，不宣称完全线性化会话累计成本 |
| 旧照片映射、标注 schema 1/2/3、控制日志 schema 1/2、E→C 别名 | 历史兼容读取 | 有明确现存读取路径，属于支持的复现契约 |
| `config/print_measurements.template.json`、`specimen_registry.template.json` | 人工实验模板 | 没有直接程序读取不等于无用途，需确认人工记录流程 |
| `capture/preview.py`、两份复核 HTML | 较长显示函数、在线/离线保存分支 | 表达的显示/保存行为不同，没有充分理由合并成新框架；缺少跨字体、窗口及交互验收，不机械拆分或改写 |

本次没有删除原始数据、标注、权重和历史记录，没有真实硬件动作。相机底层共享只经过 SDK 替身测试，不视为真实设备验收。

## 防止继续膨胀

1. 新兼容分支注明触发条件、支持版本、复现测试和退出条件；不要只写“兼容旧逻辑”。
2. 算法入口应复用已有前处理，明确裁剪边界、目标选择、孔位证据和验收差异。
3. 新输出字段必须说明坐标系、单位及是否用于控制；不要让诊断建议隐式变成运动许可。
4. 改变模块归属后运行依赖图检查；公共导出可以保留符号别名，不增加无行为的转发函数。
5. 比较同一格式下的代码量与复杂函数数量，同时报告真实调用次数、磁盘操作和测试成本。
6. 重构先固定行为，再修改实现；基线输出有差异时分析原因，禁止直接重新生成期望结果掩盖差异。
