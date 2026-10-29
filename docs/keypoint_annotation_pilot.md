# 九孔预标注与同条件 PnP 诊断

日期：2026-10-27。此阶段提供人工复核入口和对照工具，不修改现有检测、对应、PnP 或控制算法。

后续状态：用户已返回 24 张人工复核标注，并已据此完成 [视觉程序改进与回归](supervised_vision_improvement.md)。下文保留最初 AI 草稿阶段的过程和数字，不代表当前人工复核数量或程序版本。

## 当前交付与数据边界

- 从现有开发图定向选择 24 张：16 train、8 val，原图内容 SHA256 全部不同。清单为 `config/vision_annotation_pilot.json`，覆盖错孔、偏点、裁切、无解、倾斜及较好样例。
- 原有 51 个 LabelMe JSON 只有 `port` 多边形，79 个 YOLO 标签只有检测框；没有可复用的九孔中心真值。
- 已查看 24 张原图，制作 **167 个可见孔中心的 AI 目视近似预标注、22 个不确定状态、27 个遮挡状态**。非可见点不保存坐标，也不进入 PnP。
- 预标注没有复制自动点或用投影反推中心；但此前开发中已经查看过部分自动结果，因此不是严格盲标。所有图片均保持 `review_status=draft`、`source=ai_visual`，**人工复核数为 0**。
- 中心暂指可见内孔口的图像中心。实体极性、孔口三维基准和各孔高度未核验；图像中心也未认证为三维圆心的精确投影。
- `train_port_0059/0060` 中一个可见小孔的 S_pos/CC1 身份经第二次视觉审查仍无法独立确认，两标签均标不确定，没有强行互换。
- 选样针对已知问题，不能估计总体准确率。现有 train/val 都是开发数据，本批不是独立测试集。

当前标注文件：`artifacts/keypoint_review_pilot/annotations.json`。

初始交付标注 SHA256：`43a23fac1a8d8014153190a9ca05c06f1af6de042219b23378cdd3426b46251b`。人工编辑后哈希正常变化，评估报告会记录实际输入版本。

## 如何复核

1. 在文件夹中双击 `artifacts/keypoint_review_pilot/review.html`，用浏览器打开。页面嵌入原图，无需安装依赖、联网或启动服务。
2. 先确认接口方向，再逐孔检查名称及内孔口中心。数字 1–9 选择孔；点击原图修订坐标；滚轮缩放；Shift + 拖动平移。需要时打开孔名参考。
3. 孔口遮挡或身份/中心无法确认时，选择对应状态，不补预测点。自动点和检测框默认隐藏，可在完成独立观察后打开作比较。
4. 填写复核人，点击“确认本图已人工复核”。确认后再修改坐标、状态或备注，会撤销该图的复核状态。
5. 点击“导出 JSON”，保存为 `reviewed_annotations.json`，建议放在 `artifacts/keypoint_review_pilot/`。可以分批复核和导出；未确认图片仍保留为草稿，不进入正式人工对照。

浏览器本地草稿不等于文件备份，务必导出。导入 JSON 可续标；导入必须与本页的原图、相机、模型、自动报告绑定一致。重建页面读取已有标注文件，不覆盖其中的编辑；嵌入预标注版本变化时使用不同的浏览器草稿键，旧导出文件仍可手动导入。

## 对照方法与初步结果

评估器固定当前 Kinect 占位内参、原三维 CSV 和原 PnP 实现，核验原图、报告及这些输入的 SHA256。旧图不换成 D435 内参。分别计算：

1. 标注可见点全体。
2. 自动报告最终选中点全体。
3. 双方共同孔名子集的标注点。
4. 同一共同孔名子集的自动点。

同图同点名配对统计只包含第 3、4 分支均可解的图；同时保留所有图的点数、缺失、失败和原质量门槛结果。残差包含该分支全部对应点，不隐藏离群点。相同孔名不等于自动程序实际选中了正确物理孔。

当前结果仅属于 **AI 草稿诊断**：

| 项目 | 当前结果 |
|---|---:|
| 定向选择图数 | 24 |
| 人工已复核图数 | 0 |
| 可见预标注点数 | 167 |
| 自动最终选中点数 | 165 |
| 双方可比较同名点数（含不可解图） | 137 |
| 同图同孔名双方均可解 | 18 图 / 132 点 |
| 配对自动点平均残差 | 10.720775 px |
| 配对 AI 预标注平均残差 | 10.762444 px |
| 自动全点分支可解 | 20 / 24 |
| 预标注可见全点分支可解 | 22 / 24 |

配对平均残差没有整体下降。该结果不能证明几何模型有误，也不能证明自动点正确：AI 中心仍有近似误差，孔名和物理基准尚待人工核验；错误对应也可能得到较低几何残差。它说明目前没有证据支持“仅更换孔位前端即可消除残差”，人工复核后仍需检查相机及三维基准的一致性。

137 个同名点的坐标分歧中位数为 2.375149 px、均值为 43.251276 px、最大为 522.051877 px（`train_port_0055/CC2`）。这是自动点与 AI 草稿之间的分歧，不是真实中心误差或孔名错误率。两套全点分支的点集不同，不能直接用其均值比较精度。

默认模式报告 `reviewed_only_comparison.json` 的人工图片数为 0，残差为 `null`；不会把这次预标注自动算作人工真值。报告不输出未经物理参考验证的毫米/角度精度或插装成功率。

## 复现命令

以下命令在项目根目录运行。

生成或重建复核页面（保留已有标注 JSON）：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/prepare_keypoint_review.py
```

读取人工导出的文件，只评估已经确认的图片：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/evaluate_keypoint_annotations.py --annotations artifacts/keypoint_review_pilot/reviewed_annotations.json --batch-report artifacts/batch_eval_optimized/latest_details.json --output artifacts/keypoint_review_pilot/human_comparison.json
```

复现本次单独标明草稿的诊断：

```powershell
.venv\Scripts\python.exe -B -X utf8 live/python/evaluate_keypoint_annotations.py --annotations artifacts/keypoint_review_pilot/annotations.json --batch-report artifacts/batch_eval_optimized/latest_details.json --output artifacts/keypoint_review_pilot/draft_comparison.json --include-drafts
```

输入绑定变化时评估器拒绝运行；不要手改哈希绕过检查。新的相机、几何或视觉算法对照应另行保存输入和基线。输出路径不能覆盖标注、报告、相机参数、模型、原图或相关源码。

## 验证记录

- 21 项针对性 Python 测试通过：草稿排除、同点名配对、遮挡排除、坏坐标/重复/来源/哈希拒绝、已有标注保护及输入覆盖保护。
- HTML 内嵌数据与最终标注 JSON 一致，24 张原图均嵌入页面；JS 语法和独立内存 DOM 交互检查通过，覆盖改点、确认、撤销确认、保存、导入和导出。
- 内置浏览器因禁止 `file:` URL 拒绝打开该页面，因此**尚未完成实际浏览器视觉与点击验收**。用户可从文件夹双击本地页面复核；没有改用其他方式绕过浏览器限制。
- 未训练新模型，未修改原视觉运行链，未连接或移动硬件。

本地预标注、原图嵌入页面及对照 JSON 位于被 Git 忽略的 `artifacts/` 下；源码、测试、选样清单和本文档可纳入后续提交。本次未执行提交或推送。
