# UI 快速上手与模型配置设计说明（2026-09-07 决策已确认，进入实施）

> 本文遵循 AGENTS.md 第 4 条"先文档后代码"。第 8 节五项决策已于 2026-09-07
> 经 owner 拍板（见第 8 节决策记录），据此修订后进入编码阶段。
> 分两阶段：阶段 A（本地版增强，本次设计）→ 阶段 B（独立部署模块，A 落地后单独出设计）。

---

## 1. 背景与目标

owner 拍板的产品方向（2026-09-07）：

- **阶段 A（主航道）**：本地运行的 Streamlit UI 增强——在 UI 内配置模型 API
  （无需手动编辑 `.env`）、首次运行引导、数据加载面板（粘贴绝对路径为主、
  单文件上传为辅）、合成示例数据一键加载。目标是开源用户"克隆 → 打开网页 →
  填表单 → 30 秒看到全链路跑起来"。
- **阶段 B（自然延伸）**：独立于开源项目本身的部署模块。部署机是 owner 的电脑，
  同事经局域网访问；API key 配置在部署实例上，同事无需填 key，只面向特定人群使用。

**默认模型**：HY3 免费模型。**key 仅 owner 持有**（2026-09-07 决策 4）：阶段 B
部署实例的本地 `.env` 预配 hy3 的 base_url/key/model；**开源仓库的 `.env.example`
不预填 hy3**（已核实其现状即通用模板：默认示例值 DeepSeek，注释说明 Kimi /
OpenAI 可选），阶段 A 的克隆用户按各自主用的 OpenAI 兼容服务商自行配置。

## 2. 现状与缺口（已核实代码）

已有：Streamlit 双栏 UI（`streamlit_app.py`，本地启动后浏览器访问
`http://localhost:8501`）；配置经 pydantic-settings 从 `.env` 读取
（`openai_api_key / openai_base_url / default_model / default_temperature`）；
`load_dataset` 按路径加载数据，工具可脱离 Agent 独立运行。

三个缺口，其中第一个是硬阻塞：

1. **缺 key 时 UI 直接崩溃**：`Settings._validate_required`（`settings.py:117`）在
   缺 key 时抛 `ConfigError`，`get_settings()` 失败 → `ChatService()` 构造失败 →
   `_get_service()`（`streamlit_app.py:51`）在页面渲染早期抛异常，用户看到的是
   traceback 而不是引导。首次引导必须解决"未配置时 UI 可打开"。
2. 无 UI 配置入口：改模型要手动编辑 `.env` 文件。
3. 无数据加载面板与示例数据：路径靠在对话框里打字，没有"快速看到效果"的通道。

另有两个影响设计的技术事实：

- `get_settings()` 带 `@lru_cache(maxsize=1)`（`settings.py:143`）——回写 `.env` 后
  必须显式 `cache_clear()`，否则 UI 读到的还是旧配置。
- `ChatService.__init__` 构造时即 `build_model(settings)`（`chat_service.py:125`）——
  模型客户端持有旧 key，**配置变更后必须重建整个 ChatService**，仅清配置缓存不够。

## 3. 阶段 A 设计

### 3.1 模型设置表单（`.env` 回写闭环）

- **入口**：侧栏"模型设置"expander；首次引导页内复用同一组件。
- **字段**：`OPENAI_BASE_URL`（文本）、`OPENAI_API_KEY`（`type="password"` 遮蔽）、
  `DEFAULT_MODEL`（文本）、`DEFAULT_TEMPERATURE`（滑条 0~2）。
- **已配置态**：key 不回显，只显示掩码（如 `sk-***abc3f`）与"重新填写"开关；
  base_url 与 model 明文显示（非密钥）。
- **保存逻辑**（新模块 `app/config/env_io.py`，与 Streamlit 无关、可单测）：
  1. 读项目根 `.env`（不存在则创建）；
  2. 逐行扫描，仅替换目标键所在行（**保留其余行原样，包括注释与用户手改的其他键**）；
  3. 目标键不存在则追加到文件末尾；
  4. 原子写回（临时文件 + replace），避免写一半损坏配置。
- **生效**：`get_settings.cache_clear()` → `st.session_state.pop("service")`。
  下一轮渲染时 ChatService 用新配置重建；对话历史（`messages`）保留，
  但提示用户"模型已切换，历史对话仍保留，继续提问即可"。
- **规则关系**：密钥仍只存在于 `.env`、仍由 pydantic-settings 读取，表单只是
  替代"手编文本文件"，完全符合 AGENTS.md 硬性规则 2；`.env` 已在 `.gitignore`
  （`.gitignore` 第 2 行），不会入库。

### 3.2 首次运行引导（未配置不崩）

- `_main()` 顶部：`try: get_settings() except ConfigError` → 渲染**引导页**并
  `st.stop()`，不构造 ChatService。UI 从"崩溃"变为"可打开的引导页"。
- 引导页**两块并列、非强制步骤流**（2026-09-07 决策 5：示例是可选项，不要求
  每个用户必须走一遍）：
  1. **配置模型**：复用 3.1 的表单组件（通用 placeholder 提示 DeepSeek / Kimi /
     OpenAI 端点写法，key 留空待填——见第 5.2 节，不预填 hy3）；
  2. **说明区**（非交互步骤）：文字说明"配置保存后进入主界面，侧栏『数据加载』
     可粘贴数据集路径、上传单文件，或一键加载内置示例数据集（可选，用于快速
     看效果）"。
- 配置成功自动 `st.rerun()` 进入主界面；已配置状态下引导页不再出现，侧栏常驻
  "模型设置"可随时修改。示例加载入口常驻侧栏数据面板，**永远可选、非必经**。

### 3.3 数据加载面板

- **入口**：侧栏"数据加载"expander（模型设置下方）。
- **路径输入（主）**：文本框粘贴数据集绝对路径（文件或目录均可，`load_dataset`
  本就两者都支持）+ "加载"按钮。
- **单文件上传（辅）**：`st.file_uploader`（限 csv/parquet/json/jsonl 单文件，
  200MB 上限提示）→ 存 `outputs/uploads/<原名>`（outputs/ 不进 git）→ 按保存后
  路径走 `load_dataset`。目录型数据集不适用上传，界面注明。
- **调用方式**：直接调 `load_dataset_impl(service.context, path)`（工具即壁垒，
  先可独立运行——UI 复用同一实现，Agent 工具循环里也是它）。
- **与 Agent 的状态一致性**（设计关键）：`RunContext` 由 ChatService 持有、被所有
  工具共享，UI 加载后 Agent 下一轮的工具调用自然读到新数据集，无需注入历史。
  但对话可溯源要求（SYSTEM_PROMPT 纪律 5"引用数字注明所属数据集"）需要对话中
  有记录——加载成功后向 `messages` 追加一条 assistant 说明消息（含数据集名与
  概况一句话），注明"通过数据加载面板加载"；失败时如实转达 `user_message`
  （与纪律 4 一致），不吞错。
- **上传目录安全**：保存文件名做净化（去路径分隔符、防覆盖加时间戳后缀），
  不信任客户端文件名。

### 3.4 合成示例数据集（可选的一键演示）

- 新脚本 `scripts/make_sample_dataset.py`：固定随机种子**确定性**生成小型数据集
  到 `outputs/sample_dataset/`（2026-09-07 决策 5 定形态）：
  - `imu_glove.csv`：120 Hz IMU 周期流，**中段挖一段约 207 帧缺口**——
    用于演示时间同步检查（`check_temporal_sync`）与缺口定位（`locate_gaps=True`
    回答"缺口落在哪里"），即 2026-09 wujiGlove 改造成果的展示；
  - `force_sensor.csv`：同起点同时长的完整周期流（对照流，可作对齐基线）；
  - `tasks.csv`：20 个 episode 的任务表（success / 关节列），演示统计与绘图。
- **数据不进 git、脚本进 git**（规则 6：示例用合成数据；`.gitignore` 已排除 outputs/）。
- UI：示例按钮是**数据加载面板中的一个普通可选项**（侧栏常驻），不是引导必经
  步骤；点击 → 目录不存在时先调用生成函数 → 走 3.3 的加载路径。
- 示例自带一句引导话术（"试试问：这个数据集时间同步如何？缺口发生在哪？"）。

### 3.5 UI 测试策略

沿用 `AppTest` 先例（`test_streamlit_app.py`）。注意 AppTest 对 `file_uploader`
的模拟能力有限，故**逻辑与渲染分离**：`.env` 读写、路径净化、示例生成等全部放
可单测的纯函数；AppTest 只验证渲染不崩、引导页出现条件、配置保存后的状态流转。

---

## 4. 阶段 B 方向约定（本次只定原则，详细设计后置）

1. **独立模块**：B 的部署资产（启动脚本、部署配置、品牌入口）不进开源主仓库的
   代码路径——单独仓库或部署机本地目录，主仓库仅为 B 提供"实例模式"支持。
2. **实例模式**：部署机 `.env` 预配 key 并设 `INSTANCE_MODE=1`；UI 在实例模式下
   隐藏模型配置入口（同事不可见、不可改 key），只保留数据加载与对话。
   A 阶段的"已配置态折叠"设计为其留好了位。
3. **数据通道是 B 的核心问题**：部署机是 owner 的电脑，Agent 只能读部署机的
   文件系统——同事的数据必须先进部署机（上传 / 共享目录约定）。A 的"单文件
   上传"在 B 中升格为主路径之一，且需考虑压缩包整目录上传。此项留 B 设计。
4. **访问口令**：Streamlit 无鉴权，局域网内知道 IP 即可访问；key 由实例承担、
   同事消耗。B 设计中提供可选的简单口令门（`ACCESS_PASSWORD`）。
5. **触发条件**：A 四个 commit 全部落地并经真实使用后，再启动 B 设计。

---

## 5. 明确不做（防止范围蔓延）

1. **API key 不进仓库**：`.env.example` 保持通用模板现状（已核实：默认示例值
   DeepSeek、key 占位），**不预填 hy3**——hy3 的 key 仅 owner 持有（决策 4），
   其 base_url/key/model 只出现在 owner 部署实例的本地 `.env`（阶段 B）。
2. **不做公网部署（形态 C）**：真分析依赖本地数据集，公网实例读不到访问者的
   数据，仅适合挂合成示例演示，不在本轮。
3. **不做目录上传**：数据集是目录形态（多文件），浏览器上传整目录不可靠，
   路径输入 + 单文件上传已覆盖真实场景。
4. **不做多模型 profile 管理**（多套 key 切换）：单配置即够，B 也不需要。
5. **不改 Agent 结构**：仍是单 Agent + 工具循环；UI 加载走工具实现复用，
   不新增 Agent 工具。
6. **不做强制新手向导**：示例数据集是数据面板中的常驻可选项（决策 5），
   引导页仅"配置模型 + 说明"，无强制步骤流。

## 6. 改动范围与 commit 拆分

涉及文件：`app/config/env_io.py`（新）、`app/config/settings.py`（加
`is_configured()` 辅助）、`app/llm/connection_test.py`（新，测试连接）、
`app/ui/settings_panel.py`（新）、`app/ui/onboarding.py`（新）、
`app/ui/data_loader_panel.py`（新）、`app/ui/upload_store.py`（新，上传纯函数）、
`streamlit_app.py`（集成）、`scripts/make_sample_dataset.py`（新）、
`README.md`（快速上手章节）、新增 pytest。`.env.example` **不动**。

按小步交付拆四次提交：

- **Commit 1 — `.env` 读写模块**：`env_io.py`（读、逐键替换、保留注释、原子写、
  缺文件创建）+ 单测（含"已有注释与其他键不受影响""文件不存在时创建"）。
- **Commit 2 — 引导页 + 模型设置 + 测试连接**：未配置时渲染引导页不崩；模型
  设置表单（掩码、保存、缓存失效、service 重建、"测试连接"按钮——决策 2）。
- **Commit 3 — 数据加载面板**：路径加载 + 单文件上传（净化、outputs/uploads/）
  + 对话说明消息（决策 3：进对话流）；AppTest 冒烟。
- **Commit 4 — 示例数据集 + README**：生成脚本（确定性断言，含缺口流 + 任务表）
  + 一键加载 + README 快速上手章节。

## 7. 新增配置

无新增 `Settings` 字段（表单配置的就是现有四项）。上传目录复用 `output_dir`
（`outputs/uploads/`）；`INSTANCE_MODE` / `ACCESS_PASSWORD` 属阶段 B，届时再入。

## 8. 决策记录（2026-09-07 owner 拍板）

| # | 决策项 | 结论 |
|---|---|---|
| 1 | 模型设置入口 | **侧栏折叠面板**（expander，随时可改） |
| 2 | "测试连接"按钮 | **要**（保存前发一次轻量真实调用验证 key 可用） |
| 3 | UI 加载成功后的对话呈现 | **进对话流**（追加 assistant 说明消息，保可溯源） |
| 4 | hy3 key 边界 | **仅 owner 持有**（对应阶段 B 部署实例）；**阶段 A 克隆用户自行配置各自 API**，`.env.example` 保持通用模板、不预填 hy3 |
| 5 | 示例数据集形态 | **小型 IMU 流（含缺口，演示时间同步缺口定位）+ 任务表**；且**示例是 UI 中的可选项**，非打开 UI 后的必经步骤 |

据此修订：3.2 引导页改两块并列非强制流、3.4 示例含缺口流且常驻侧栏可选、
5.1 与 6 的 `.env.example` 结论改为"保持现状不动"。

## 9. 风险

1. **`.env` 回写破坏用户手改**：逐行替换策略保留注释与其他键，且原子写；
   保存前不做格式重排。首次保存前建议 UI 提示"将修改项目根目录 .env 文件"。
2. **配置热切换的会话一致性**：切换模型后旧 ChatService 重建，历史 `messages`
   保留但 RunContext 保留（数据集不丢）——需在 UI 文案中说明"数据集保留，
   模型已切换"。
3. **AppTest 与 file_uploader**：模拟上传能力有限，上传逻辑放纯函数单测兜底。
4. **引导页与既有测试兼容**：现有 AppTest 用例依赖测试环境 `.env` 已配置，
   引导页只在未配置时出现，不影响既有用例。

---

_状态：草案，等待 owner 确认第 8 节决策后进入编码阶段（预计 4 个 commit）。_
