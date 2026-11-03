# 代码风格与精简原则

本工程参考 LeRobot 的 Python/Ruff 配置、Ultralytics 的小步精简原则及 Google Python 风格指南，采用适合当前 Python 3.12 工程的规则。

参考资料：

- [LeRobot 工具配置](https://github.com/huggingface/lerobot/blob/main/pyproject.toml)
- [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)
- [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)
- [Ruff 格式化与导入排序](https://docs.astral.sh/ruff/formatter/)

## 工具与日常检查

Ruff 版本固定为 0.17.3，在 `pyproject.toml` 的 `dev` 可选依赖中声明。现有 Conda 环境已经安装；新环境在工程根目录执行：

```powershell
conda activate ev-charging
python -m pip install --no-deps ruff==0.17.3
python -m ruff check src tests
python -m ruff format --check --output-format concise src tests
python -B -m unittest discover -s tests -v
```

需要更新格式时使用 `python -m ruff format src tests`。导入排序由 `ruff check` 的 `I` 规则负责，formatter 本身不排序导入。自动修复前应查看差异；不默认启用 `--unsafe-fixes`。

## 规则

- 使用 4 空格缩进、双引号和 110 字符目标行宽。保留工程现有的 CRLF 换行约定；无法合理拆分的字符串等允许超过目标行宽。
- 导入按标准库、第三方库、本工程分组。项目模块使用 `ev_charging.*` 包名，函数内 SDK 延迟导入保留。
- `training/train_port.py` 必须先配置缓存再导入相关业务模块，因此保留有说明的 `E402` 文件级例外。不得为了排序把缓存初始化或设备初始化移到错误阶段。
- 普通变量名称表达业务含义；数学代码可保留 `R`、`T` 等约定记号，并说明坐标系、单位或维度。避免使用容易与数字混淆的 `l`。
- 函数按明确职责组织。函数行数是审查线索，不作为机械拆分的硬门槛；拆分应减少重复或让业务阶段更清楚。
- 新增或修改的公共接口补齐输入、返回类型；类型注解不替代输入校验。简单函数使用简短说明，复杂接口采用 Google 风格的 `Args`、`Returns`、`Raises` 段落。
- 注释解释算法依据、单位、约束、顺序要求和特殊处理原因。保留设备回读、数据来源、校验与控制边界的必要说明。
- 公共实现按明确用途归属，例如 `file_io.py` 和 `review/http.py`。只有实际重复且语义一致的行为才合并；调用者的错误状态码和业务规则继续由调用者决定。
- 文件 SHA256 使用 `file_io.sha256_file`，计算原始字节；内存数据哈希与 SHA256 字符串格式校验是不同操作。
- JSON 文件写入优先复用 `file_io`。保留 `allow_nan=False`、原子替换、Windows 文件占用重试及显式选择的就地回退语义。追加日志和图片原始字节保存继续使用各自对应实现。

静态规则覆盖导入规范、未使用或未定义名称及基础语法风格。类型标注与文档质量仍需人工检查，本工程没有宣称通过全量静态类型检查。

## 精简结果如何统计

格式化可能把紧凑的多项表达式展开成多行，所以分别记录原始物理行数、同一 Ruff 配置下的格式化行数和语法语句数量。新增测试单独统计，不混入生产代码净减少量。

本次重构将复核服务的响应、鉴权、请求解析与启动收尾归入 `review/http.py`，将六处文件哈希实现和批量评估的 JSON 保存归入 `file_io.py`。两套复核服务原有的错误状态码、标注确认和冲突拒绝行为保留。

## 本轮结果（2026-11-04）

| 统计口径 | 调整前 | 调整后 |
| --- | ---: | ---: |
| 生产 Python 文件数 | 72 | 73 |
| 生产代码原始物理行数（含空行、注释） | 16,986 | 21,044 |
| 使用同一 Ruff 配置后的生产代码行数 | 21,104 | 21,044 |
| 生产代码 AST 语句数量 | 12,131 | 12,120 |
| 测试物理行数 | 90 | 369 |

物理行数增加主要来自长表达式和字典展开；在相同排版口径下，生产代码净减少 60 行。新增的公共 HTTP 模块集中两套复核服务的传输逻辑。

静态检查还发现并修复了两处缺失导入：`empty_observation` 使用的 `time`，以及轮廓函数返回类型中的 `LayoutModel`；均先用测试复现再修复。

验证结果：Ruff 检查与格式检查通过，19 项回归测试通过；32 个控制、采集、设备适配和视觉算法模块除导入外的语法树保持一致；13 个软件仿真场景结果与此前记录一致，旧日志决策全部复现；2,763 个配置、数据和训练结果文件哈希不变，运行依赖版本未改变。详细结果位于本机 `artifacts/style_refactor_20261104/verification.json`。

格式调整会改变源码指纹，新实验应重新冻结当前源码。历史报告、人工标注和权重未改写，未进行硬件连接或真机验收。
