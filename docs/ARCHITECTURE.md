# 具身智能数据分析 Agent — 系统架构设计

> 版本：v1.0（定稿，经用户确认）
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
│   Agent 的"手"  │      │  单一工厂函数     │      │ pydantic-settings│
│  5 个领域工具   │      │  构造 Model      │      │ 从 .env 读密钥   │
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

参考实现要点（来自 csv-ai）：`Settings(BaseSettings)` + `SettingsConfigDict(env_file=".env", extra="ignore")` + `@lru_cache get_settings()`。

### 3.2 `app/llm/` — 模型接口层（单一工厂函数）

目标：**为上层（agent/services/tools）产出一个 openai-agents 可用的 `Model` 对象，屏蔽具体的 API 端点与密钥来源**。不设抽象基类、也不拆分 provider 文件，全部逻辑收敛到一个工厂函数：

- `app/llm/factory.py`：`build_model(settings: Settings) -> Model`。从 `settings` 读取 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`DEFAULT_MODEL`，构造 OpenAI 兼容的 `Model` 对象返回。上层只依赖这一个工厂，不关心底层走的是 DeepSeek 还是 Kimi 端点——切换服务商只需改 `.env`。

**接入 openai-agents 的关键点**：openai-agents 原生支持第三方 OpenAI 兼容端点。因为 DeepSeek/Kimi 只支持 Chat Completions（不支持 OpenAI Responses API），需用 `OpenAIChatCompletionsModel` + 自定义 `AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)`，并把该 Model 实例传给 Agent。**同时必须调用 `set_tracing_disabled(disabled=True)`**，否则未配置 OpenAI 官方 key 时 tracing 会 401。这一工厂即 5.3 所述实现。

### 3.3 `app/tools/` — 领域工具（Agent 的"手"）

每个工具一个模块，是**普通、带类型标注、可独立运行和测试**的 Python 函数，用 `@tool` 装饰器注册给 Agent。第一批 5 个工具在 §6 详细设计。工具只对 DataFrame / 元信息操作，返回**精简结果**（统计值、摘要、图表文件路径），绝不把整个 DataFrame 塞进对话上下文。

### 3.4 `app/agent/` — Agent 定义与运行入口

- 组装：读取 `get_settings()` → 通过 `app/llm/factory.build_model` 构建 Model → 构建 `Agent`（含 `name`、`instructions`、`model`、`tools`）。
- 定义 `RunContext`（dataclass，位于 `app/agent/context.py`）：持有当前已加载的 `pd.DataFrame`、数据集元信息 `meta`、输出目录 `output_dir`，以及**累积各工具分析结果摘要的 `findings: list` 字段**。通过 `Runner.run(context=...)` 注入，供各工具经 `RunContextWrapper` 访问。**DataFrame 不会序列化给 LLM**。
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

## 6. 第一批工具的函数签名设计

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
        dict，包含成功与否、source、n_rows、n_cols、columns 列表；
        失败时返回含 error 与建议的结构化错误信息。

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
        dict，包含行数/帧数、字段数、各列 dtype、缺失比例、唯一值数量、
        样例值；未加载数据时返回结构化错误。
    """
```

```python
# app/tools/compute_stats.py
@tool
def compute_stats(
    wrapper: RunContextWrapper[RunContext],
    column: str | None = None,
    group_by: str | None = None,
) -> dict:
    """对已加载数据集执行统计计算（成功率、轨迹长度、分布等）。

    Args:
        column: 可选，指定要统计的列；省略时给出整体概况统计。
        group_by: 可选，按某列分组后统计（如按 episode 分组）。

    Returns:
        dict，包含各统计项（count/mean/std/min/max/中位数等）与结果摘要，
        便于模型直接引用数字撰写解读。
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

> 说明：`RunContext` 定义于 `app/agent/context.py`，字段含 `df: pd.DataFrame | None`、`meta: dict`、`output_dir: str`、`findings: list`（各工具累积的分析结果摘要，供 `generate_report` 汇总）。此签名设计为定稿。

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

---

## 8. 定稿决议记录

以下为编码前已确认的决议：

1. **provider 接入方式**：不设抽象基类、不拆 provider 文件，收敛为 `app/llm/factory.build_model` 单一工厂函数。
2. **`RunContext` 位置**：放 `app/agent/context.py`。
3. **`compute_stats` 粒度**：保持单一工具 + 参数，不拆分。
4. **max_turns 默认值**：定为 15。
5. **首批 5 个工具**：维持 `load_dataset` / `profile_data` / `compute_stats` / `plot_chart` / `generate_report` 不变；开发顺序为 load → profile → stats → plot → report。

---

*本文档已定稿，遵循"先文档后代码"规则，按决议顺序进入编码阶段。*
