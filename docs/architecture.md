# 工程结构

> 本文档合并了原「工程结构」「代码风格与精简原则」「统一开发环境」三篇，原文件已删除，历史版本见 Git 提交。

可复用源码统一位于 `src/ev_charging/`。工程根目录保存配置、实际数据和运行产物，数据处理代码位于包内的 `datasets/`，两者用途不同。

| 原位置 | 新位置（相对于 `src/ev_charging/`） |
| --- | --- |
| `vision/algorithms`、`vision/calibration`、`vision/evaluation`、`vision/locate` | 保留对应的 `vision/` 子包 |
| `vision/capture` | `capture/`，其中 `realsense.py` 迁入 `hardware/` |
| `vision/review` | `review/`，HTML 页面作为包资源一起分发 |
| `vision/library` | `datasets/vision/` |
| `control/hardware.py` | `hardware/realman.py` |
| `control/force_dataset.py` | `datasets/force.py` |
| `control` 其他模块 | `control/` |
| `train/python` 训练与拟合脚本 | `training/` |
| `train/python/runtime_env.py` | `runtime.py` |
| `vision/contracts.py` 的 JSON 写入函数 | `file_io.py`；观测数据协议仍在 `vision/contracts.py` |

设备适配模块在显式运行设备流程时连接硬件。采集与控制入口组合设备和算法；算法、数据管理与离线仿真可单独测试。导入设备模块不代表授权连接或运动。

`paths.py` 分别定位已安装代码和工程数据：可编辑安装默认使用其对应工程；wheel 安装可从工程目录启动，或通过进程环境变量 `EV_CHARGING_ROOT` 指定包含 `config/` 和 `dataset/` 的根目录。HTML 资源和源码哈希读取实际导入的包，配置、图库与权重读取工程数据目录。

定位与数据完整性校验分开：导入时允许已知源码工程或显式指定的现有目录暂缺数据，使纯软件模块与帮助入口可用；`find_project_root()` 默认仍严格要求配置和图库。不会在未知目录中自动猜测或创建数据根目录。

数据、标注、模型配置、训练报告和权重保留原始位置与字节。新生成报告记录迁移后的源码路径与哈希；历史记录不改写。历史评估中绑定旧源码哈希的结果不能直接冒充当前版本的结果。

## 开发检查

格式、命名、公共接口与精简原则见 [代码风格](#代码风格与精简原则)。复核 HTTP 公共层位于 `review/http.py`；文件哈希与 JSON 持久化统一位于 `file_io.py`。

在 `ev-charging` 环境、工程根目录运行：

```powershell
python -m ruff check src tests
python -m ruff format --check --output-format concise src tests
python -B -m unittest discover -s tests -v
python -B -m ev_charging.vision.check_project
python -B -m ev_charging.control.simulate --scenario nominal
```

`control.simulate` 使用合成数据检查控制软件。统一环境中的 MuJoCo 是运行依赖，本工程此次没有引入其他目录的仿真场景。

## 代码风格与精简原则

本工程参考 LeRobot 的 Python/Ruff 配置、Ultralytics 的小步精简原则及 Google Python 风格指南，采用适合当前 Python 3.12 工程的规则。

参考资料：

- [LeRobot 工具配置](https://github.com/huggingface/lerobot/blob/main/pyproject.toml)
- [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing/)
- [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)
- [Ruff 格式化与导入排序](https://docs.astral.sh/ruff/formatter/)

### 工具与日常检查

Ruff 版本固定为 0.17.3，在 `pyproject.toml` 的 `dev` 可选依赖中声明。现有 Conda 环境已经安装；新环境在工程根目录执行：

```powershell
conda activate ev-charging
python -m pip install --no-deps ruff==0.17.3
python -m ruff check src tests
python -m ruff format --check --output-format concise src tests
python -B -m unittest discover -s tests -v
```

需要更新格式时使用 `python -m ruff format src tests`。导入排序由 `ruff check` 的 `I` 规则负责，formatter 本身不排序导入。自动修复前应查看差异；不默认启用 `--unsafe-fixes`。

### 规则

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

### 精简结果如何统计

格式化可能把紧凑的多项表达式展开成多行，所以分别记录原始物理行数、同一 Ruff 配置下的格式化行数和语法语句数量。新增测试单独统计，不混入生产代码净减少量。

本次重构将复核服务的响应、鉴权、请求解析与启动收尾归入 `review/http.py`，将六处文件哈希实现和批量评估的 JSON 保存归入 `file_io.py`。两套复核服务原有的错误状态码、标注确认和冲突拒绝行为保留。

### 本轮结果（2026-11-04）

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

## 统一开发环境

使用 Conda 环境 `ev-charging`，Python 3.12.16。它覆盖原 `ev-vision` 的视觉、CUDA 训练和设备 SDK，以及原 `ev-sim` 的 MuJoCo 依赖。

原环境保留用于回退，不再作为本工程文档的默认入口。此次未修改全局 PATH、Windows 安全策略或备份工程。

### 创建与安装

在工程根目录运行：

```powershell
conda env create --format environment-yaml -f environment.yml
conda activate ev-charging
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

`--no-deps` 保留锁定清单安装的版本，避免包安装时重新选择 CUDA 构建。`--no-build-isolation` 使用环境内已有的 setuptools 与 wheel；这些构建工具由 Conda 环境提供。修改源码不需要重复安装；修改包配置或新增命令入口后重新执行可编辑安装。

`pyproject.toml` 描述包的直接依赖范围和入口；`environments/pip.lock.txt` 固定整套业务依赖及 CUDA wheel 的官方地址、SHA256。Conda 管理 Python、Tk、pip 等基础组件。

开发工具 Ruff 独立固定为 0.17.3，声明在 `pyproject.toml` 的 `dev` 可选依赖中，不改变业务依赖锁。当前环境已安装；新环境执行 `python -m pip install --no-deps ruff==0.17.3`。检查命令见 [代码风格](#代码风格与精简原则)。

### 精确重建 Windows 基础组件

`environment.yml` 适合通常的环境创建；如需还原本次 Conda 基础组件的具体构建，可执行：

```powershell
conda create -n ev-charging --file environments/win64.conda.lock.txt
conda activate ev-charging
python -m pip install -r environments/pip.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
```

显式 Conda 清单不包含 pip 业务包，两部分都需要安装。仅对不存在的目标环境执行创建命令。

### 合并取舍

- Python：两套原环境均为 3.12.16。
- NumPy：保留视觉环境的 2.4.3，替代原仿真环境中的 2.5.4。
- PyTorch / TorchVision：保留 2.14.1+cu132 / 0.29.1+cu132。
- MuJoCo：保留原仿真环境的 3.17.0，并补齐其依赖。
- 本次不升级相机或机械臂 SDK。

### 检查

```powershell
python -m pip check
python -B -m unittest discover -s tests -v
python -B -m ev_charging.vision.check_project
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

测试中的 MuJoCo 场景仅验证引擎与统一 NumPy/PyTorch 环境共存；它不验证充电插装场景、实物传感器或硬件控制效果。

### 本次迁移验证（2026-11-04）

- 6 项包结构回归测试通过，涵盖模块导入、命令入口、复核页面、工作目录切换与 MuJoCo/PyTorch 共存。
- 工程完整性检查通过；原有相机参数来源缺失警告仍保留。
- 13 个控制软件仿真场景完成，日志回放均复现原决策。
- RTX 5080 上的 CUDA 前向/反向计算、当前检测器对合成图像的推理通过。
- MuJoCo 物理步进及 64×64 离屏渲染通过。
- 普通 wheel 构建及独立解包后的资源、自检通过，源码指纹与可编辑安装一致。
- 2,763 个配置、数据及训练结果文件与迁移前的 SHA256 一致。

详细结果保存在本机 `artifacts/structure_migration_20261104/verification.json` 和 `wheel_verification.json`。未连接相机或机器人，未执行正式训练或改写当前模型。
