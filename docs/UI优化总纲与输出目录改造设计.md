# UI 优化总纲与输出目录改造设计

> 状态：**草案，等待 owner 逐份确认后进入编码阶段**。
> 遵循 AGENTS.md 第 4 条"先文档后代码"、第 5 条"小步交付"。
> 本文是六项 UI 优化的**总纲**（索引 + 决策 + 验收 + commit 拆分），并包含
> 其中"输出目录改造"的完整设计。其余五项各有独立文档（见第 2 节）。
> 关联：`docs/UI快速上手与模型配置设计.md`、`docs/多会话设计.md`、`streamlit_app.py`、
> `app/ui/*`、`app/tools/plot_chart.py`、`app/tools/generate_report.py`、`app/tools/profile_store.py`

---

## 1. 背景

2026-09-11 对现有 UI 做了一次全面审视（覆盖 `streamlit_app.py`、`app/ui/` 全部模块、
`ChatService`、工具落盘点与全部相关设计文档）。结论：**功能骨架已完整且有回归保护**
（多会话隔离、输入框钉底、编辑重发、引导页、数据面板、token/耗时可见均已落地），
但存在 6 个明确的"交互质感 / 长期可用性"提升点。

owner 决策（2026-09-11）：**六项全部实施**；推进方式为**先出全部设计文档、逐份确认后再写码**；
输出目录采用 **A 方案**（保持集中 `outputs/`，但按数据集名建子文件夹做到结构分明）。

---

## 2. 六项优化索引

| # | 文档 | 核心问题 | 价值 | 依赖 |
|---|---|---|---|---|
| 0 | **本文** | 输出散落 `outputs/` 根目录，与 Agent 产物混杂 | 结构分明、易查找、易清理 | 无（可最先做） |
| 1 | `docs/流式输出设计.md` | 无流式，长等待黑箱（实测单轮十几秒起） | 体感提升最大 | 改 `run_turn` 契约（本文第 6 节定接口） |
| 2 | `docs/右栏信息架构重构设计.md` | findings 只增不减，图表全宽平铺成流水账 | 多轮分析不退化 | 依赖 0（路径结构） |
| 3 | `docs/多会话标签页升级设计.md` | 会话多了标签挤成豆腐块 | 横向可扩展 | 独立 |
| 4 | `docs/数据集状态面板设计.md` | 语义确认画像等已有资产未显性化 | 减少重复追问 | 依赖 0（profile 路径） |
| 5 | `docs/UI工程质量与配置化设计.md` | CSS hack 脆弱、魔法数字、累加逻辑重复三处 | 可维护性底座 | 独立 |
| 6 | `docs/交互增强重试与错误恢复设计.md` | 错误后只能手动重输，无重试入口 | 小改动高回报 | 依赖 1（流式重试路径） |

**实施顺序建议**：0 → 5 → 2 → 1 → 6 → 3 → 4。
理由：0 是其它项的路径基础；5 抽出公共函数（含 turn 累加）后，1/2/6 改动更安全；
2 改动集中在 `components.py` 风险最低；1 是最大改动，放在公共底座就位后；
6 依赖 1 的流式重试路径；3/4 相互独立可最后做。

> 注：owner 亦可按"价值优先"改为 1 先做。第 6 节给出两种排序的 commit 依赖关系。

---

## 3. 【第 0 项】输出目录改造 —— 完整设计

### 3.1 现状（已核实代码）

所有产物**平铺在 `outputs/` 根目录**，仅靠文件名区分：

```
outputs/
  20260821_144044_histogram_20260825_155613.png      ← dataset_id + chart_type + ts
  20260821_144044_multi_stream_overlay_20260825_144223.png
  20260821_144044_report_20260825_155616.md
  lerobot_histogram_20260831_133149.png
  worldcode_ISuit-V3-01350F01_2026-09-03-17-12-47_0_vlta_reorg_sample_1-0_line_20260909_161624.png
  .dataset_profile.json                               ← 语义确认画像（隐藏文件）
  uploads/                                            ← 上传的单文件
  sample_dataset/                                     ← 示例数据集
```

问题（实拍可见）：
1. **无分组**：19 张 png + 11 个 md 混在一起，文件名长（dataset_id 可达 50+ 字符），
   用户找"某数据集的报告"只能靠肉眼扫；
2. **与 Agent 项目文件夹混杂**：`outputs/` 就在项目根目录下，与 `app/`、`docs/`、`tests/`
   平级，用户会困惑"哪些是项目代码、哪些是我分析产生的"；
3. **清理困难**：想清掉某数据集的全部产物需按前缀手工挑；
4. **多数据集长名截断**：`worldcode_ISuit-V3-...sample_1-0_line_...` 已接近可读极限。

落盘点（已核实全部 4 处 + 1 处夹带）：

| 落盘内容 | 代码位置 | 当前路径 |
|---|---|---|
| 图表 png | `plot_chart.py:53-62` `_output_path` | `outputs/<prefix><dataset>_<type>_<ts>.png` |
| 报告 md | `generate_report.py:23-30` `_output_report_path` | `outputs/<prefix><dataset>_report_<ts>.md` |
| 语义画像 json | `profile_store.py:62-73` `_profile_path` | `outputs/.dataset_profile.json`（**单个共享文件**） |
| MCAP 解包 | `mcap_reader.unpack_mcap_to_dir`（调用方给目录） | 由 `unpack_mcap` 工具指定 `outputs/mcap_unpack/<id>` |
| 上传文件 | `data_loader_panel.py:102` | `outputs/uploads/` |
| 示例数据集 | `sample_dataset.py:95` | `outputs/sample_dataset/` |

### 3.2 目标结构

按数据集名建子文件夹，**每数据集一个目录**，画像文件也随之**按数据集拆分**：

```
outputs/
  by_dataset/
    lerobot/                          ← 数据集名（dataset_id 净化后）
      charts/  lerobot_histogram_20260831_133149.png
               s-1a2b_lerobot_line_20260909_161624.png   ← session_tag 前缀保留
      reports/ lerobot_report_20260901_111606.md
      profile.json                    ← 该数据集的语义确认画像（原 .dataset_profile.json 的分片）
    worldcode_ISuit-V3-01350F01.../   ← 长名截断到 60 字符 + 短哈希后缀防碰撞
      charts/
      reports/
      profile.json
  _misc/                              ← 未加载数据集时产生的产物（如纯对话生成的图）
    charts/
    reports/
  uploads/                            ← 上传原文（与数据集无关，保持顶层）
  sample_dataset/                     ← 示例数据集（保持顶层）
  mcap_unpack/                        ← MCAP 解包产物（保持顶层）
```

设计要点：

1. **`by_dataset/<数据集名>/`** 下再分 `charts/` 与 `reports/`，避免 png 与 md 混放；
2. **数据集名净化**：`dataset_id` 可能含 `:`（如 `outputs_test-2`）、超长名。净化规则
   见 3.4，与 `upload_store.sanitize_upload_filename` 同源思路（不信任输入、去危险字符）；
3. **画像按数据集拆分**（这是本项最实质的改动）：`outputs/.dataset_profile.json` 是
   **所有数据集共享的单文件**（`profile_store._profile_path` 只取 `output_dir`），
   多会话并发写要靠锁+合并保护（`多会话设计.md` 3.4）。改为
   `by_dataset/<数据集名>/profile.json` 后，**每个数据集一个文件，跨数据集不再竞争**，
   锁的粒度也从"全局单文件"降到"每数据集"——这是顺带的架构改善，但也**影响兼容性**
   （见 3.5 迁移）；
4. **文件名规则不变**：`<session_tag 前缀><dataset_id>_<type>_<ts>` 全部保留原样，
   `多会话设计.md` 的多会话隔离契约**零改动**（前缀仍在，只是落在子目录内）；
5. **`dataset_id` 仍是完整值**：子目录名是净化后的显示名，**不改 `dataset_id` 本身**
   （避免影响工具内的 `dataset` 来源标注、画像 key、报告内容）；
6. **`RunContext.output_dir` 保持"outputs 根"语义**，新增派生方法定位子目录（见 3.3）。

### 3.3 代码改动设计

**新增纯函数模块 `app/tools/output_paths.py`**（不 import streamlit，可单测）：

```python
def sanitize_dataset_dir_name(dataset_id: str | None) -> str:
    """数据集名 → 安全的子目录名（净化 + 截断 + 短哈希防碰撞）。"""

def dataset_output_dir(output_dir: str, dataset_id: str | None) -> Path:
    """返回 outputs/by_dataset/<净名>/，不存在则创建。dataset_id 为 None → _misc/。"""

def chart_path(context, chart_type: str) -> Path:
    """outputs/by_dataset/<ds>/charts/<prefix><ds>_<type>_<ts>.png"""

def report_path(context) -> Path:
    """outputs/by_dataset/<ds>/reports/<prefix><ds>_report_<ts>.md"""
```

**`RunContext` 新增派生方法**（`app/agent/context.py`）：

```python
def dataset_output_dir(self) -> Path:
    """当前数据集的产物目录（outputs/by_dataset/<净名>/）；未数据集时 _misc/。"""
    from app.tools.output_paths import dataset_output_dir
    return dataset_output_dir(self.output_dir, self.dataset_id)
```

改动点：
- `plot_chart._output_path` → 改调 `output_paths.chart_path(context, chart_type)`；
- `generate_report._output_report_path` → 改调 `output_paths.report_path(context)`；
- `profile_store._profile_path(output_dir, dataset_id)` → **签名加 dataset_id**，
  路径改为 `by_dataset/<净名>/profile.json`。调用方 `load_dataset.py` /
  `propose_semantics.py` 需同步传 dataset_id（已核实现有调用处均已持有 dataset_id）；
- `sample_dataset.ensure_sample_dataset` 不变（示例数据不是 Agent 产物）；
- `data_loader_panel` 的 `uploads/` 不变。

**为什么把新逻辑放独立模块而不是各自内联**：4 处落盘点需要**同一套净化与目录规则**
（否则报告和图表可能落到不同目录）；集中后单测只需覆盖一个模块，且未来新增
产物类型（如导出 csv）有唯一入口。

### 3.4 目录名净化规则（防长名 / 防碰撞 / 防危险字符）

输入：`dataset_id`（如 `worldcode_ISuit-V3-01350F01_2026-09-03-17-12-47_0_vlta_reorg_sample_1-0`）。

规则（顺序执行，幂等）：

1. **去危险字符**：路径分隔符 `/ \`、控制字符、Windows 保留字符 `<>:"|?*` → `_`；
   空白 → `_`。中文与常规符号保留（与 `sanitize_upload_filename` 一致）；
2. **截断**：超过 **60 字符**截断到 60；
3. **防碰撞**：**无条件**追加 `-<sha1(dataset_id)前 6 位>`（如 `...sample_1-0-9f3a2c`）。
   为什么不只在"截断时"才加哈希：两个不同的长 dataset_id 截断到 60 字符后前 60 位
   可能相同（真实可能：同批次带不同时间戳后缀），**只截断不加哈希会串目录**；
   无条件加哈希使映射稳定、可逆查（哈希可重算），代价是目录名略长；
4. **空名兜底**：dataset_id 为 None/空 → `_misc`（不参与净化，固定名）。

> 取舍说明：`dataset_id` 保留完整值（3.2 要点 5），子目录名只用于"人类查找"，
> 程序定位产物仍以 `context.dataset_id` 与 findings 里的 `file_path` 为准——
> 因此目录名即便截断也**不影响正确性**。

### 3.5 兼容与迁移（关键）

现有 `outputs/.dataset_profile.json` 存有**用户已确认的流语义画像**（跨会话生效、
`user_confirmed` 来源，SYSTEM_PROMPT 纪律 12 明确告知用户"存于
outputs/.dataset_profile.json"）。改路径不能让这些确认丢失。

迁移策略（`profile_store` 内实现，首次读取时惰性触发）：

1. `load_dataset_profile(output_dir, dataset_id)` 先读新路径
   `by_dataset/<净名>/profile.json`；
2. 新路径不存在 → 回退读**旧全局文件** `output_dir/.dataset_profile.json`，
   取出 `datasets[dataset_id]` 分片；
3. 若命中旧数据 → 写入新路径（**迁移落盘**）并在返回中标注
   `migrated_from_legacy: true`（仅日志/调试用，不打扰用户）；
4. 旧文件**保留不删**（保守起见；可后续在文档中说明"可安全删除"）；
5. 旧文件不存在 → 返回空画像（与现状一致）。

- SYSTEM_PROMPT 纪律 12 中"存于 `outputs/.dataset_profile.json`"的措辞需**同步更新**
  为新路径说明（或改为不写死路径的表述："存于 outputs 下的持久化确认画像"）——
  这是文档/文案改动，不涉及逻辑；
- 回归保障：新增"旧格式画像可迁移读取"用例（造一份旧结构文件 → 断言能读到）。

### 3.6 `.gitignore` 与"与项目混杂"的治理

`outputs/` 已在 `.gitignore`（第 12 行），所以**git 层面产物从不入库**，不存在误提交风险。
owner 提的"混在一起不好"主要影响的是**本地文件浏览体验**。除子目录化外，追加两件事：

1. **`outputs/README.md`（纳入 git）**：在 outputs 目录内放一份说明——
   "本目录是 Agent 运行产物（已 gitignore，可随时删除）；结构为
   `by_dataset/<数据集>/{charts,reports}/`；`uploads/` 是上传原文；
   `sample_dataset/` 是示例数据"。**为什么放目录内**：用户打开资源管理器时第一眼
   就看到说明，而不是要去翻项目 README；
2. **项目 README 增补一节**"产物在哪里"：说明上述结构 + "想清理产物直接删
   `outputs/` 即可（示例数据集与上传文件也会一并删除，需重传）"。

> 是否把 `outputs/` 移出项目目录（如用户主目录 `.embodied-xray/`）？**不做**：
> 违反本项目"本地工具、路径可预期"的取向，且会让 README/文档的路径说明全部失准。
> A 方案（集中 + 子目录）已解决 owner 的实际痛点。

### 3.7 测试计划

- `output_paths.sanitize_dataset_dir_name`：危险字符、超长截断、**不同长名不碰撞**、
  空名兜底、幂等；
- `chart_path` / `report_path`：落在 `by_dataset/<净名>/charts|reports/`，
  文件名仍含 `session_tag` 前缀与 dataset_id（多会话契约不回退）；
- `profile_store`：新路径读写、**旧文件迁移**、并发合并（现有用例改路径后仍过）；
- AppTest：加载数据集后右栏图表仍能渲染（路径变更的端到端冒烟）。

---

## 4. 全局验收标准（六项共同遵守）

1. **零回归**：`python -m pytest` 全绿；`tests/test_multi_session.py`、
   `tests/test_streamlit_app.py`、`tests/test_chat_edit.py` 必须显式重跑通过；
2. **分层纪律**：新增代码遵守 `app/services|tools|agent|llm/config` 不 import streamlit；
3. **行为回归**：每项在 `docs/行为测试.md` 追加一条终身回归用例（背景 + 用例 + 期望）；
4. **配置集中**：新增可调项进 `app/config/`，同步 `.env.example` 与 README；
5. **小步提交**：每项拆成独立 commit（见第 5 节），提交信息中文一句话；
6. **人工验收**：每项都需真实对话验收（AppTest 只保证不崩，不保证体验）。

---

## 5. Commit 拆分（总览）

> 每行一个 commit；同一文档内的多处改动允许合并为一个 commit（保持"小步"但不碎片化）。

| 序 | Commit | 内容 | 依赖 |
|---|---|---|---|
| C1 | 输出目录子目录化 | `output_paths.py`（新）+ plot/report 路径改造 + 单测 | — |
| C2 | 画像按数据集拆分 + 旧文件迁移 | `profile_store` 改路径 + 迁移 + 并发用例改路径 + SYSTEM_PROMPT 措辞 | C1 |
| C3 | outputs README + 项目 README 产物章节 | 文档 | C1 |
| C4 | UI 工程质量：常量集中 + turn 累加抽纯函数 | `app/ui/constants.py`（新）+ `_record_turn()` + 三处去重 | — |
| C5 | 右栏信息架构重构 | `components.py` 分组/倒序/缩略图+对话框放大 | C1 |
| C6 | 数据集状态面板 | `components.py` 增画像/单位告警面板 | C1/C2 |
| C7 | 流式输出 | `chat_service.reply_stream` + `streamlit_app` 消费 + `run_turn` 流式分支 | C4 |
| C8 | 错误重试入口 | 错误消息旁"重试本轮"按钮（复用 C7 路径） | C7 |
| C9 | 多会话标签条升级 | 滚动标签条 + 独立关闭按钮 | C4 |

**可并行性**：C4 与 C1/C2 无依赖，可先做；C5/C6/C9 相互独立；
C1→C2→C3 与 C4→C7→C8 是两条独立链。**建议两条链交叉推进**（先 C1+C4，
再 C2+C5，再 C7，最后收尾）。

---

## 6. 对其它文档的接口约定（本文先行定下，避免后续冲突）

1. **`run_turn` 流式分支接口**（供第 1 项）：新增 `stream_turn()` 异步生成器，
   **不改** 现有 `run_turn` 的签名与返回契约（零回归）；两者共用历史压缩、
   错误兜底与 metrics 回填逻辑（抽公共函数）；
2. **findings 结构**（供第 2/4 项）：**不改**现有 finding 的字段名与语义
   （`type` / `tool` / `file_path` / `title` / `summary` 等）。右栏重构只改
   **渲染方式**（分组、排序、缩略图），不改数据来源。若需"轮次锚点"，
   以**附加字段**（如 `turn_index`）实现，且允许缺省（老 finding 无该字段时降级）；
3. **画像读取入口**（供第 4 项）：面板展示统一走 `profile_store.load_dataset_profile`
   （已含迁移逻辑），不自行拼路径。

---

## 7. 风险

| 风险 | 应对 |
|---|---|
| 画像路径变更导致用户已确认语义"丢失" | 3.5 惰性迁移 + 保留旧文件 + 专门迁移用例 |
| 目录名净化碰撞 | 无条件追加短哈希（3.4） |
| 长路径超 Windows 260 上限 | 数据集名截断 60 + 固定层级深度仅 3 级，实测远低于上限 |
| 文件名规则被误改（多会话隔离回退） | 保留 `output_paths` 单测断言前缀存在 |
| 六项一起改导致排查困难 | 严格 commit 拆分 + 每项独立验收 |

---

_状态：草案。请 owner 确认第 3 节（输出目录结构 3.2、净化规则 3.4、迁移策略 3.5）
与第 5 节 commit 拆分；六个子文档（第 2 节 #1–#6）将随后逐份提交确认。_
