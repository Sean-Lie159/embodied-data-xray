# 具身智能数据分析 Agent — 系统架构设计

> 版本：v1.1（2026-08-19 更新工具层结构，经用户确认）
> 关联文件：`AGENTS.md`（行为准则）、`README.md`（使用说明）
> 技术栈（不可变更）：Python 3.12 / openai-agents / pandas / numpy / pyarrow / h5py / matplotlib / plotly / Streamlit / pytest

---

## 1. 项目定位与设计目标

本项目是一个**本地运行、面向具身智能（Embodied AI）领域的数据分析 Agent**。用户通过自然语言对话，让 Agent 完成：加载机器人数据集 → 统计分析 → 可视化 → 生成解读报告。支持的数据格式：LeRobot 数据集、HDF5、Parquet/CSV。

三个核心设计目标：

1. **业务逻辑与 UI 完全解耦**：`services/`、`tools/`、`agent/`、`llm/` 均不 import streamlit，确保同一套业务代码未来可以同时服务于 Streamlit 界面、CLI、FastAPI 或未来某个 Web 服务，无需改动。
2. **模型提供方可插拔**：默认提供方（DeepSeek / Kimi / OpenAI）可切换，密钥与 base_url 从 `.env` 读取，绝不硬编码。
3. **确定性计算优先、LLM 叙述其次**：统计和绘图用 pandas/matplotlib 算出真实结果，LLM 只负责围绕这些真实数字写解读，杜绝 LLM 自行编造算数（这是纯 LLM dataframe agent 最常见的失败模式）。

> 该设计深度借鉴 `reference/csv-ai/` 参考项目的分层思想（业务与 UI 解耦、确定性统计优先、pydantic-settings 配置），但实现框架不同：本项目的工具编排层改用 **openai-agents SDK**，且模型接入收敛为单一工厂函数而非手写 provider 类。

---

## 2. 整体分层图

```
┌─────────────────────────────────────────────────────────────────┐
│                     streamlit_app.py（薄入口）                      │
│                  streamlit run streamlit_app.py                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │  仅在此处 import streamlit
                  ┌─────────────▼─────────────┐
                  │        app/ui/             │
                  │  对话区 + 图表/表格展示区     │
                  └─────────────┬─────────────┘
                                │
                  ┌─────────────▼─────────────┐
                  │       app/services/        │  编排层：纯 Python，
                  │  串联 Agent 与工具、LLM 输出  │  无 UI 依赖
                  └─────────────┬─────────────┘
                                │
                  ┌─────────────▼─────────────┐
                  │        app/agent/          │  Agent 定义：
                  │   system prompt、工具注册、   │  RunContext 持有 DataFrame
                  │     单 Agent 运行入口         │
                  └─────────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌────────────────┐      ┌────────────────┐
│   app/tools/   │      │    app/llm/     │      │   app/config/   │
│ 质检层 + 统计层  │      │  单一工厂函数     │      │ pydantic-settings│
│ 能力标签驱动    │      │  构造 Model      │      │ 从 .env 读密钥   │
└───────────────┘      └────────────────┘      └────────────────┘
```

**关键约束（硬性规则）**：`app/services/`、`app/tools/`、`app/agent/`、`app/llm/`、`app/config/` 中**禁止 import streamlit**。只有 `app/ui/` 与 `streamlit_app.py` 可以。

---

## 3. 各模块职责

### 3.1 `app/config/` — 配置集中管理

用 `pydantic-settings` 读取 `.env` 与 OS 环境变量，作为全项目配置的唯一事实来源。任何需要 API 密钥或默认值的地方，一律调用 `get_settings()`，绝不直接 `os.environ`。

关键配置项（草案，最终以 `.env.example` 为准）：

| 配置项 | 说明 |
|---|---|
| `OPENAI_API_KEY` | 模型 API 密钥（DeepSeek / Kimi / OpenAI 任一兼容服务的密钥） |
| `OPENAI_BASE_URL` | 模型服务端点（如 DeepSeek/Kimi 的 OpenAI 兼容 base_url） |
| `DEFAULT_MODEL` | 默认模型名（如 `deepseek-chat`） |
| `DEFAULT_TEMPERATURE` | 默认温度（校验 0.0–2.0） |
| `output_dir` | 图表与报告的保存目录（默认 `outputs/`） |
| `max_rows_in_context` | 注入到 LLM 上下文的最大样本行数（默认如 200） |
| `max_turns` | Agent 单次运行最大循环轮数（防死循环，默认 15） |

**质检阈值配置**（质检工具判定 pass/warn/fail 的可调阈值，集中在此，便于按数据集类型调整）：

| 配置项 | 说明 | 默认建议 |
|---|---|---|
| `sync.max_skew_ms` | 各流时间戳最大允许偏差（超过则 fail） | 10.0 |
| `sync.max_drift_ppm` | 允许的时钟漂移率（ppm） | 1000 |
| `sensor.missing_ratio_warn` | 传感器缺失率触发 warn 的阈值 | 0.05 |
| `sensor.missing_ratio_fail` | 传感器缺失率触发 fail 的阈值 | 0.20 |
| `sensor.outlier_zscore` | 传感器离群判定的 z-score 阈值 | 5.0 |
| `sensor.silent_ms` | 判定"静默段"的最小持续时长 | 500 |

参考实现要点（来自 csv-ai）：`Settings(BaseSettings)` + `SettingsConfigDict(env_file=".env", extra="ignore")` + `@lru_cache get_settings()`。

### 3.2 `app/llm/` — 模型接口层（单一工厂函数）

目标：**为上层（agent/services/tools）产出一个 openai-agents 可用的 `Model` 对象，屏蔽具体的 API 端点与密钥来源**。不设抽象基类、也不拆分 provider 文件，全部逻辑收敛到一个工厂函数：

- `app/llm/factory.py`：`build_model(settings: Settings) -> Model`。从 `settings` 读取 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`DEFAULT_MODEL`，构造 OpenAI 兼容的 `Model` 对象返回。上层只依赖这一个工厂，不关心底层走的是 DeepSeek 还是 Kimi 端点——切换服务商只需改 `.env`。

**接入 openai-agents 的关键点**：openai-agents 原生支持第三方 OpenAI 兼容端点。因为 DeepSeek/Kimi 只支持 Chat Completions（不支持 OpenAI Responses API），需用 `OpenAIChatCompletionsModel` + 自定义 `AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)`，并把该 Model 实例传给 Agent。**同时必须调用 `set_tracing_disabled(disabled=True)`**，否则未配置 OpenAI 官方 key 时 tracing 会 401。这一工厂即 5.3 所述实现。

### 3.3 `app/tools/` — 领域工具（Agent 的"手"）

每个工具一个模块，是**普通、带类型标注、可独立运行和测试**的 Python 函数，用 `@tool` 装饰器注册给 Agent。工具只对 DataFrame / 元信息操作，返回**精简结果**（统计值、判定、图表文件路径），绝不把整个 DataFrame 塞进对话上下文。

工具层按职责分**两层**：

- **质检层**（面向原始采集数据的"流"）：`inspect_streams`、`check_temporal_sync`、`check_sensor_sanity`。面向原始信号流（传感器/视频时间戳、IMU/力觉等），关注数据采集质量与时间同步，产出结构化判定（pass/warn/fail）。
- **统计层**（面向状态/动作数据表）：`compute_stats` 职责收窄为**任务级统计**（成功率、轨迹长度、episode 分布等，作用在已规整为表的维度），不再承担传感器级信号检查。

每个工具声明**前置条件**（`requires`，基于数据集能力标签）。能力标签不满足时返回结构化"不适用"结果，并推荐该数据集可用的分析，而不是抛异常或硬跑出无意义结果。

### 3.4 `app/agent/` — Agent 定义与运行入口

- 组装：读取 `get_settings()` → 通过 `app/llm/factory.build_model` 构建 Model → 构建 `Agent`（含 `name`、`instructions`、`model`、`tools`）。
- 定义 `RunContext`（dataclass，位于 `app/agent/context.py`）：持有当前已加载的 `pd.DataFrame`、数据集标识 `dataset_id`、数据集元信息 `meta`（含**能力标签 `capabilities`** 与 **推测类型名 `guessed_type` / 置信度**）、输出目录 `output_dir`，以及**累积各工具分析结果摘要的 `findings: list` 字段**。通过 `Runner.run(context=...)` 注入，供各工具经 `RunContextWrapper` 访问。**DataFrame 不会序列化给 LLM**。
- **单数据集语义**：任一时刻 `RunContext` 只持有**一个**当前数据集，`load_dataset` 新加载会覆盖旧的 `df` / `dataset_id` / `meta`。旧数据集不再可被工具操作，其数字只能来自对话历史中工具真实返回过的结果。因此工具返回必须**自带数据来源标注**（`load_dataset` 返回 `dataset_id` 与 `replaced_previous`，`profile_data` 等分析工具返回 `dataset` 字段），避免模型把历史记忆误当当前数据。
- `findings` 的用途：`profile_data`、`compute_stats`、`plot_chart` 等工具在产出结果时，把关键结论以简短摘要 append 到 `RunContext.findings`；`generate_report` 汇总时遍历该列表作为报告正文素材。
- 暴露运行入口 `run_agent()`，封装 `Runner.run` / `Runner.run_streamed`，处理 `max_turns`、错误兜底。

### 3.5 `app/services/` — 编排层（串联 Agent 与 UI）

纯 Python，负责把"用户一句话"翻译成一次完整的 Agent 执行：构造 `RunContext`、调用 agent 运行入口、把结果（含图表路径、表格摘要）整理成 UI 可直接展示的结构。**不 import streamlit**，因此未来可被 FastAPI/CLI 复用。

### 3.6 `app/ui/` — Streamlit 界面

全项目唯一 import streamlit 的地方。职责：对话输入区、Agent 执行进度的展示、结果表格与图表的渲染。会话状态集中在单一模块管理，key 名保持一致。UI 不直接调用工具，只调用 `services`。

### 3.7 `outputs/`、`data/`、`tests/`、`docs/`

- `outputs/`：图表与生成报告的落盘目录（应加入 `.gitignore`）。
- `data/`：示例数据集（`.gitignore` 已排除，仅用小的公开样本或合成数据）。
- `tests/`：pytest，与 `app/` 目录结构对应；每个工具至少一个用例。
- `docs/`：设计文档，本文件为总纲。

---

## 4. 单 Agent + 工具循环方案

本项目坚持**单 Agent，不做多 Agent 拆分**（除非用户明确要求）。核心循环如下：

```
用户输入 ──▶ Runner.run(agent, input, context=..., max_turns=...)
                │
                ▼
          ┌─────────────┐   工具调用需求     ┌──────────────┐
          │  Agent/LLM   │ ───────────────▶  │  app/tools/   │
          │  决定下一步   │                    │  (工具函数)    │
          └─────────────┘ ◀─────────────────  └──────────────┘
                │      工具精简结果（统计值/图路径/摘要）
                ▼
          产出最终回答（final_output）
```

要点：

1. **工具即壁垒**：Agent 不直接碰 DataFrame，所有数据操作都通过工具完成。工具返回给模型的是精简结果（数字、摘要、文件路径），原始数据留在 Python 进程内的 `RunContext.df` 里。
2. **上下文注入**：把 `RunContext`（内含 DataFrame）通过 `Runner.run(context=...)` 传入，工具用 `RunContextWrapper` 访问 `wrapper.context.df`。SDK 保证上下文对象不发给 LLM。
3. **防死循环**：显式设置 `max_turns`（如 15），超过则抛 `MaxTurnsExceeded`，由 services 层兜底转成友好错误提示。
4. **错误可恢复**：工具失败时返回结构化错误信息（出了什么错、建议怎么办），而不是抛异常中断整个循环；SDK 的 `@tool` 默认会把异常转成错误结果回传给模型，让 Agent 可尝试换一种方式继续。
5. **流式可选**：需要打字机效果时用 `Runner.run_streamed` + `result.stream_events()`；Streamlit 下用 `st.empty()` + `st.markdown` 尾部光标实现。

---

## 5. 与 openai-agents 的对接设计

### 5.1 工具注册（`@tool`）

openai-agents 当前推荐用 `@tool`（`@function_tool` 的别名）装饰普通函数。SDK 会用 `inspect` 提取函数签名、`griffe` 解析 docstring、`pydantic` 生成 JSON Schema。因此：

- 工具函数必须**带完整类型标注**（`str`/`int`/`float`/`list`/`dict` 等可 JSON 序列化类型）。
- **中文 docstring 必须写清楚用途、参数、返回值**——docstring 会被当作工具描述注入给 LLM，直接影响模型的调用判断。
- 第一个参数可放 `RunContextWrapper`（不会发给 LLM），用于访问 `RunContext.df`。

### 5.2 Runner 与结果读取

- 异步：`await Runner.run(agent, input, context=..., max_turns=...)`；同步环境用 `Runner.run_sync`。
- 最终答案：`result.final_output`（str）。
- 中间工具调用：`result.new_items` 里的 `tool_call_item` / `tool_call_output_item`，可用于在 UI 里展示"Agent 正在执行 xxx 工具"。

### 5.3 模型接入的实现落地（单一工厂函数）

在 openai-agents 体系下，模型接入收敛为一个**产 Model 的工厂函数**，定义于 `app/llm/factory.py`：

```python
def build_model(settings: Settings) -> Model:
    # 从 settings 读取 OPENAI_API_KEY / OPENAI_BASE_URL / DEFAULT_MODEL
    # 构造 AsyncOpenAI(api_key=..., base_url=...) 客户端，
    # 并返回 OpenAIChatCompletionsModel(model=DEFAULT_MODEL, openai_client=client)
    ...
```

DeepSeek / Kimi / OpenAI 均走 OpenAI 兼容的 `OpenAIChatCompletionsModel` + 自定义 `AsyncOpenAI`。切换服务商只需改 `.env` 里的 `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `DEFAULT_MODEL`，代码无需变动。工厂内部**必须先调用 `set_tracing_disabled(disabled=True)`**（见 3.2），再构建客户端与 Model。

> 注意：具体 openai-agents 的 Model 构造接口细节在编码阶段以实际安装版本为准，本设计只确定"工厂产 Model"这一方向。

---

## 6. 工具层设计

### 6.0 能力标签与前置条件机制

**数据类型不做硬编码枚举**。Ego / UMI / 遥操 / 动捕等类型的组合会持续膨胀，穷举枚举不可维护。改为：`load_dataset` 加载时**嗅探**数据结构，在 `RunContext.meta` 记录能力标签，并给出**推测类型名（含置信度）**，供模型与工具决策。

**能力标签**（`RunContext.meta.capabilities`，布尔或枚举）示例：

| 能力标签 | 含义 |
|---|---|
| `has_video_streams` | 含视频/图像流（各摄像头） |
| `has_imu` | 含 IMU 数据 |
| `imu_axes` | IMU 轴数（如 3/6/9），无 IMU 时缺省 |
| `has_force` | 含力觉/力矩数据 |
| `has_calibration` | 含相机/手眼标定信息 |
| `has_actions` | 含动作指令/控制信号 |
| `action_space` | 动作空间描述（如 joint / ee / cartesian） |

**推测类型名**（`RunContext.meta.guessed_type`）：如 `"Ego"` / `"UMI"` / `"遥操"` / `"动捕"` 或 `"unknown"`，并附 `guessed_type_confidence`（0~1）。类型名仅供交互提示，**不作为逻辑分支依据**——逻辑只依赖能力标签。

**前置条件（requires）**：每个工具声明其所需能力标签。入口处校验，不满足时返回结构化"不适用"结果：

```python
{
  "success": False,
  "error": "not_applicable",
  "reason": "缺少能力标签 has_imu",
  "user_message": "当前数据集不含 IMU 数据，check_sensor_sanity 不适用。",
  "suggested_tools": ["inspect_streams", "compute_stats"]   # 推荐该数据集可用的分析
}
```

统一约定：

- 均为**普通 Python 函数**，带 `@tool` 装饰器；首个参数为 `RunContextWrapper`（`wrapper.context` 是 `RunContext`）。
- 返回值为**精简、可 JSON 序列化**的类型（dict / str / 路径），便于回传给模型。
- docstring 用中文，采用 Google 风格（`Args:` / `Returns:`），因为 openai-agents 会自动解析它作为工具描述。

> 说明：下方仅给出**函数签名与 docstring 草案**，供评审确认；最终实现允许在编码阶段做微小调整，但对外暴露的参数与返回结构保持稳定。

```python
# app/tools/load_dataset.py
from agents import RunContextWrapper
from app.agent.context import RunContext

@tool
def load_dataset(
    wrapper: RunContextWrapper[RunContext],
    path: str,
    fmt: str | None = None,
) -> dict:
    """加载机器人数据集为 DataFrame，并返回元信息。

    Args:
        path: 数据集路径，支持 .csv/.parquet/.h5/.hdf5 或 LeRobot 数据集目录。
        fmt: 可选，显式指定数据格式；省略时根据扩展名自动推断。

    Returns:
        dict，成功时含 dataset_id（当前数据集名）、source、n_rows、n_cols、
        columns 列表、capabilities（能力标签字典，如 has_imu / has_video_streams
        / imu_axes / action_space 等）、guessed_type（推测类型名）与
        guessed_type_confidence（0~1）、user_message；若覆盖了旧数据集还含
        replaced_previous。失败时返回含 error/reason/user_message 的结构化错误。

    Raises:
        不直接抛出；错误以结构化 dict 的 error 字段返回，便于 Agent 恢复。
    """
```

```python
# app/tools/profile_data.py
@tool
def profile_data(
    wrapper: RunContextWrapper[RunContext],
    max_unique: int = 20,
) -> dict:
    """生成当前已加载数据集的概况（episode 数、帧数、字段、缺失等）。

    Args:
        max_unique: 每个字段最多展示的唯一值数量，防止上下文过大。

    Returns:
        dict，包含 dataset（本次结果产自的数据集名）、行数/帧数、字段数、各列
        dtype、缺失比例、唯一值数量、样例值；未加载数据时返回结构化错误。
    """
```

```python
# app/tools/compute_stats.py   【统计层】职责收窄为任务级统计
@tool
def compute_stats(
    wrapper: RunContextWrapper[RunContext],
    metric: str,
    group_by: str | None = None,
    episode_filter: list[int] | None = None,
) -> dict:
    """计算任务级统计指标（成功率、轨迹长度、episode 分布等）。

    Args:
        metric: 统计指标，如 success_rate / trajectory_length / duration_stats /
            action_stats；具体可选值由实现阶段给出。
        group_by: 可选，按某列分组后统计（如按 condition 分组）。
        episode_filter: 可选，仅统计指定 episode。

    Returns:
        dict，包含 dataset（来源数据集名）、metric、结果数值与结果摘要，
        便于模型直接引用数字撰写解读；能力标签不满足或参数非法时返回
        结构化"不适用/错误"结果。
    """
```

### 6.1 质检层工具统一返回格式

所有质检工具（inspect_streams / check_temporal_sync / check_sensor_sanity）返回统一的
"测量值 + 阈值 + 判定"结构，阈值从 config 读取（可调）：

```python
{
  "success": True,
  "dataset": "xxx",                 # 来源数据集
  "check": "check_temporal_sync",   # 质检项
  "result": "pass",                 # pass / warn / fail
  "measurements": {                 # 测量值（实际计算出的指标）
    "max_skew_ms": 3.2,
    "num_unsynced_episodes": 0,
  },
  "thresholds": {                   # 判定所用的阈值（来自 config）
    "max_skew_ms": 10.0,
  },
  "affected_episodes": [],          # 受影响的 episode 清单（fail/warn 时非空）
  "user_message": "……",            # 可直接转达给用户的中文说明
}
```

判定规则：测量值在安全阈值内 → `pass`；接近阈值 → `warn`；越界 → `fail`。
`affected_episodes` 列出触发 warn/fail 的 episode 编号，供后续定位。

### 6.2 质检层工具签名

```python
# app/tools/inspect_streams.py   【质检层】面向原始流的探测
@tool
def inspect_streams(
    wrapper: RunContextWrapper[RunContext],
) -> dict:
    """探测数据集的流与能力（视频/IMU/力觉/动作/标定），返回能力标签与时钟来源。

    Args:
        无（基于 RunContext.meta.capabilities）。

    Returns:
        dict，包含 capabilities（能力标签字典）、clock_source（unified /
        per-device / unknown，默认 unknown）、n_streams、各流简要信息；无法
        识别时钟来源时 clock_source 记为 unknown。
    """
```

> **流登记表与按需读取（2026-08-19 更新）**：`load_dataset` 目录加载时会建立
> 流登记表 `RunContext.meta["streams"]`（每条流含 {path, format, kind, channels,
> role}）。`inspect_streams` 基于该登记表**按需读取**各流时间戳列实测采样率
> （csv 用 usecols、parquet 用 pyarrow 列裁剪），计算后立即释放，**不装入
> RunContext.df**（df 仍只装单主表）。实测结果回写
> `meta["streams"][i]["measured_rate"]` 缓存，重复调用不重复读盘。单条流读取
> 失败（文件缺失/格式损坏/无时间戳列）时该流标 unknown 并注明原因，不影响其他流。
> 角色判断（`infer_role`）收集全部命中语义线索并组合输出（如 `left_wrist_cam` →
> "腕部相机（左）"），无命中时标 unknown。

```python
# app/tools/check_temporal_sync.py   【质检层】时间同步检查
@tool
def check_temporal_sync(
    wrapper: RunContextWrapper[RunContext],
) -> dict:
    """检查各流之间的时间同步与漂移。

    前置条件：requires has_video_streams 或 has_imu 等多流能力。

    Args:
        无。

    Returns:
        dict，统一质检返回格式（measurements + thresholds + result +
        affected_episodes），含漂移检测结果；报告区分两个置信级别：
        - confidence="timestamp"：基于时间戳一致性推断（默认，较快速）；
        - confidence="physical"：基于物理实测的互相关对齐（需额外数据，v2）。
    """
```

> **v1 范围说明（2026-08-19 更新）**：`check_temporal_sync` v1 仅做**可对齐的表格流**
> 之间的时间戳一致性检查。**视频流不参与 v1 帧级对齐**（容器时间戳不可靠，内容级
> 对齐属 v2），因此视频流不计入"可对齐流数"；纯视频数据集返回"不适用"。返回中
> `streams_status` 逐条标注各流"参与对齐 / 未参与 + 原因"；因缺信息跳过的检查以
> `status: "skipped" + reason` 独立呈现，不计入判定依据。

```python
# app/tools/check_sensor_sanity.py   【质检层】传感器数据合理性
@tool
def check_sensor_sanity(
    wrapper: RunContextWrapper[RunContext],
    sensor: str | None = None,
) -> dict:
    """检查传感器数据合理性（范围、缺失率、突变/离群、静默段等）。

    前置条件：requires 对应能力标签（如 has_imu / has_force）。

    Args:
        sensor: 可选，指定要检查的传感器；省略时检查所有可用传感器。

    Returns:
        dict，统一质检返回格式；包含每路传感器的测量值与判定、受影响 episode。
    """
```

```python
# app/tools/plot_chart.py
@tool
def plot_chart(
    wrapper: RunContextWrapper[RunContext],
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str | None = None,
    save_name: str | None = None,
) -> dict:
    """绘制图表（轨迹图、时序图、分布图），保存到 outputs/ 并返回路径。

    Args:
        chart_type: 图表类型，如 line / scatter / histogram / trajectory。
        x: 可选，X 轴列名；省略时按需自动选取。
        y: 可选，Y 轴列名；省略时按图表类型自动选取。
        color: 可选，用于分组的列名。
        title: 可选，图表标题。
        save_name: 可选，保存文件名（不含扩展名）；省略时自动生成。

    Returns:
        dict，包含 file_path（相对 outputs/ 的路径）、chart_type、title、
        success；绘制失败时返回结构化错误。
    """
```

```python
# app/tools/generate_report.py
@tool
def generate_report(
    wrapper: RunContextWrapper[RunContext],
    report_type: str = "analysis",
    title: str | None = None,
) -> dict:
    """汇总当前分析结果为 Markdown 报告，保存到 outputs/ 并返回路径。

    Args:
        report_type: 报告类型，如 analysis（分析）/ summary（总结）。
        title: 可选，报告标题；省略时按时间自动生成。

    Returns:
        dict，包含 file_path（相对 outputs/ 的路径）、title、
        word_count；未完成任何分析时返回结构化错误。
    """
```

> 说明：`RunContext` 定义于 `app/agent/context.py`，字段含 `df: pd.DataFrame | None`、`dataset_id`、`meta: dict`（含 `capabilities` 能力标签与 `guessed_type` 推测类型）、`output_dir: str`、`findings: list`（各工具累积的分析结果摘要，供 `generate_report` 汇总）。此签名设计为定稿。

---

## 7. 技术取舍与理由

| 取舍 | 选择 | 理由 |
|---|---|---|
| 数据分析方式 | 确定性 pandas 计算 + LLM 叙述 | 避免 LLM 编造算数（csv-ai 同款理念） |
| 工具编排 | openai-agents 单 Agent + `@tool` | 与需求指定框架一致，成熟度高 |
| 上下文节俭 | 只把统计值/图路径给模型 | 防 token 爆炸，减少幻觉 |
| 模型接入 | 单一工厂 `build_model` 产 `OpenAIChatCompletionsModel` + `AsyncOpenAI` | 第三方兼容端点统一走 Chat Completions，切换服务商只改 `.env` |
| 界面 | Streamlit，UI 与业务严格分层 | 未来可复用为 API/CLI |
| 错误处理 | 工具返回结构化错误而非抛异常 | 保证 Agent 循环可恢复 |
| 工具分层 | 质检层（原始流）+ 统计层（任务级表） | 面向对象不同，职责清晰、互不混淆 |
| 数据类型识别 | 能力标签 + 推测类型名，不做枚举 | 组合持续膨胀，枚举不可维护；逻辑只依赖能力标签 |
| 质检判定 | 测量值 + 阈值 + pass/warn/fail + affected_episodes | 阈值可调，结果可定位到具体 episode |

---

## 8. 定稿决议记录

以下为已确认的决议：

1. **provider 接入方式**：不设抽象基类、不拆 provider 文件，收敛为 `app/llm/factory.build_model` 单一工厂函数。
2. **`RunContext` 位置**：放 `app/agent/context.py`。
3. **max_turns 默认值**：定为 15。
4. **工具层结构（2026-08-19 更新）**：工具分两层——【质检层】面向原始采集数据流
   （`inspect_streams` / `check_temporal_sync` / `check_sensor_sanity`），【统计层】
   面向状态/动作数据表（`compute_stats` 职责收窄为任务级统计）。开发顺序建议：
   load → profile → compute_stats → 质检层三件（inspect_streams → check_sensor_sanity
   → check_temporal_sync）→ plot_chart → generate_report。
5. **数据类型不做枚举（2026-08-19 更新）**：改为 `load_dataset` 嗅探后在
   `RunContext.meta` 记录能力标签（`capabilities`）+ 推测类型名（`guessed_type` 含
   置信度）；工具通过 `requires` 前置条件声明所需能力，不满足时返回结构化"不适用"结果。
6. **质检返回统一格式（2026-08-19 更新）**：测量值 + 阈值 + 判定（pass/warn/fail）+
   受影响 episode 清单；阈值写入 config 可调。
7. **时钟语义（2026-08-19 更新）**：`inspect_streams` 输出 `clock_source`
   （unified / per-device / unknown，默认 unknown）；`check_temporal_sync` 含漂移
   检测，报告区分"基于时间戳一致性"（timestamp）与"基于物理实测"（physical）两个
   置信级别。

---

*本文档已定稿，遵循"先文档后代码"规则，按决议顺序进入编码阶段。*

---

## 9. 未来演进清单（暂不实现，由真实需求驱动）

1. **RunContext 支持多数据集（命名管理）**：具身智能数据分析中"对比不同批次/版本
   数据"是常见需求。若真实需求出现，将 `RunContext` 从"单数据集、覆盖式"升级为
   "多数据集命名管理"，各工具增加 `dataset` 参数以指定操作目标。当前 MVP 阶段保持
   单数据集 + 明确语义（数据来源自带标注）即可，等 `compute_stats`、`plot_chart`
   稳定后再评估。
2. **互相关物理对齐实测（v2）**：`check_temporal_sync` 的漂移检测目前基于时间戳
   一致性（confidence="timestamp"）。v2 引入基于物理实测的互相关对齐
   （confidence="physical"），对信号做互相关以实测各流时间偏移，用于时间戳不可信
   或缺失时的兜底与校验。
3. **UMI 专用指标**：UMI 数据集特有分析，如 **SLAM 跟踪丢失率**（跟踪质量随时间
   变化、丢失段定位）等，作为 `check_sensor_sanity` 或独立工具的扩展。
4. **Ego 视频专用分析**：Ego 视频流的专用分析（如视觉里程计/感知相关的质量评估、
   帧率稳定性、曝光/运动模糊信号等），作为质检层或独立分析工具扩展。
