# AGENTS.md — 具身智能数据分析 Agent

> 本文件是 AI 编程助手（Code Buddy）在本仓库工作时的最高行为准则。
> 每次开始工作前必须先读本文件，并严格遵守其中的硬性规则。

---

## 1. 项目定位

一个**本地运行的、面向具身智能（Embodied AI）领域的数据分析 Agent**。
用户通过自然语言对话，让 Agent 完成：加载机器人数据集 → 统计分析 → 可视化 → 生成解读报告。
目标数据格式：LeRobot 数据集、HDF5、Parquet/CSV（后续可能扩展 rosbag）。
最终形态：个人本地工具，将以 MIT 协议开源到 GitHub，他人克隆后填入自己的模型 API 密钥即可使用。

## 2. 技术栈（未经允许不得变更）

- 语言：Python 3.12
- Agent 框架：OpenAI Agents SDK（`openai-agents`）
- 模型接入：标准 OpenAI 兼容接口，密钥与 base_url 从 `.env` 读取（pydantic-settings 管理），**默认模型提供方可切换（DeepSeek / Kimi / OpenAI）**
- 数据处理：pandas、numpy、pyarrow、h5py、lerobot（按需）
- 可视化：matplotlib / plotly
- 界面：Streamlit
- 测试：pytest

## 3. 硬性规则（违反即返工）

1. `reference/` 目录是只读参考资料，**禁止修改、移动、删除其中任何文件**，也禁止把其中的代码直接复制进项目（只可借鉴结构与设计）。
2. **禁止把 API 密钥、本地绝对路径写进任何代码或 git 提交**。密钥一律从 `.env` 读取；`.env` 已在 `.gitignore` 中。
3. **分层纪律**：`app/services/`、`app/tools/`、`app/agent/`、`app/llm/` 中**禁止 import streamlit**。UI 逻辑只出现在 `app/ui/` 和 `streamlit_app.py`。
4. **先文档后代码**：任何新模块动工前，先在 `docs/` 中有对应设计说明（或更新 ARCHITECTURE.md），经用户确认后再写代码。
5. **小步交付**：一次只实现一个工具或一个模块，完成后运行测试确认可用，再进入下一个。不要一次生成整个项目。
6. 数据集文件不进 git（`.gitignore` 已排除 `data/`）；示例数据使用小的公开样本或合成数据。
7. **指令与文档冲突时，停下并指出，由用户裁决。**

## 4. 目录结构与职责

```
app/
├── config/      # pydantic-settings：读取 .env，集中管理配置
├── llm/         # 模型接口层：OpenAI 兼容客户端，provider 可插拔
├── agent/       # Agent 定义：system prompt、工具注册、运行入口
├── tools/       # 领域工具（Agent 的"手"）：每个工具一个模块
│                #   - load_dataset   加载 LeRobot/HDF5/CSV 为 DataFrame + 元信息
│                #   - profile_data   数据集概况（episode 数、帧数、字段、缺失）
│                #   - compute_stats  统计计算（成功率、轨迹长度、分布等）
│                #   - plot_chart     画图（轨迹图、时序图、分布图），保存到 outputs/ 并返回路径
│                #   - generate_report 汇总分析结果为 Markdown 报告，保存到 outputs/
├── services/    # 编排层：串联工具与 Agent，纯 Python，不依赖 UI
└── ui/          # Streamlit 界面：对话区 + 图表/表格展示区
tests/           # pytest，与 app/ 结构对应
docs/            # 设计文档，ARCHITECTURE.md 为总纲
reference/       # 只读参考项目（不进 git）
streamlit_app.py # 唯一入口：streamlit run streamlit_app.py
```

## 5. Agent 设计约定

- **单 Agent + 工具循环**：不做多 Agent 拆分，除非用户明确要求。
- **工具即壁垒**：工具函数是普通 Python 函数（带类型标注和 docstring），先能独立运行和被测试，再注册给 Agent。
- **上下文节俭**：工具返回给模型的必须是**精简结果**（统计值、摘要、图表文件路径），禁止把整个 DataFrame 塞进对话上下文。原始数据留在 Python 进程内，通过工具按需查询。
- **错误可恢复**：工具失败时返回结构化错误信息（出了什么错、建议怎么办），而不是抛异常中断整个循环。

## 6. 代码规范

- 全部函数带类型标注；公开函数写中文 docstring（说明用途、参数、返回值）。
- 配置项集中在 `app/config/`，新增配置同步更新 `.env.example` 和 README。
- 每个工具至少配一个 pytest 用例（可用小型合成数据）。
- 提交信息用中文，一句话说明本次改动。

## 7. Git 提交规则

- **里程碑式提交**：每完成一个经用户审批确认的阶段（设计文档、模块、工具），主动执行 `git add .` 和 `git commit`；提交信息用中文，一句话说明本次改动。
- **提交前自查**：先 `git status` 确认暂存区里没有 `.env`、数据集、`reference/`、生成的图表与报告等不应入库的文件；若有，先修正 `.gitignore` 再提交。
- **中间状态不提交**：代码跑不通、测试没过时不提交，先修到可用状态。
- **只提交，不推送**：`git push`、`commit --amend`、`reset --hard` 等改写历史或影响远端的操作，必须经用户明确指示后才可执行。

## 8. 与用户的协作方式

- 用户是 Agent 开发新手但具备领域知识。涉及架构取舍时，用"术语 + 通俗解释"的方式说明选项与代价，让用户拍板。
- 每完成一个阶段，简要汇报：完成了什么、如何验证、下一步是什么。
