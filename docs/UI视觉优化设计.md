# UI 视觉优化设计

> 状态：**决策已确认（2026-09-11），进入实施**。
> owner 拍板：**A 靛青配色** / **描边卡片** / **做深色模式** / **图表配色跟主色同族**。
> 遵循 AGENTS.md 第 2/4/5 条（技术栈不可变更、先文档后代码、小步交付）。
> 关联：`docs/UI工程质量与配置化设计.md` 第 6 节（CSS 技术债）、
> `docs/UI优化总纲与输出目录改造设计.md`
> 关联代码：`streamlit_app.py`、`app/ui/components.py`、`app/ui/constants.py`、
> `app/tools/plot_chart.py`、各 `app/ui/*_panel.py`

---

## 1. 背景与诊断

2026-09-11 审视结果：**九项功能/工程优化已完成，但视觉层面几乎未动**。
进一步核查发现一个此前未被注意的事实：

**项目根本没有 `.streamlit/config.toml`，全部使用 Streamlit 默认主题。**

```powershell
> Test-Path ".streamlit/config.toml"
False
```

### 1.1 诊断：不是缺图标和颜色，而是缺层级与节奏

当前 UI 的观感问题按重要性排序（这是本设计的优先级依据）：

| # | 问题 | 具体表现 | 根因 |
|---|---|---|---|
| 1 | **无视觉层级** | 标题、正文、说明、面板权重接近，页面没有"从哪开始读" | 主题未配置 `headingFontSizes`/`headingFontWeights`；全用 `st.caption` 与 `st.markdown` 平铺 |
| 2 | **无分区边界** | 侧栏与主区、各面板之间只靠 `st.divider()` 一条线 | 未用 `st.container(border=True)`；未用 `[theme.sidebar]` 制造明度差 |
| 3 | **状态表达不统一** | `✓ ✗`（纯文本）、`⚠️`（emoji）、`st.warning`（原生色块）三种风格混用 | 无统一的状态呈现约定 |
| 4 | **图标语言混乱** | 产品标 `🩻`、状态 `✓✗⚠️`、操作 `🔁✏️＋` 混搭，字号字重不一 | 无图标使用规则 |
| 5 | **图表观感割裂** | `plot_chart` 用 matplotlib 默认样式（灰底白网格），与页面配色无关 | 未设 `chartCategoricalColors`；matplotlib 未套主题 |
| 6 | **信息密度过高** | 侧栏模型设置/数据加载/token/上下文四块堆叠，缺留白 | 面板间只用 `st.divider()`，无卡片间距 |

### 1.2 关键判断：三层推进，以"官方机制"为主

**说明：本项目在 `docs/UI工程质量与配置化设计.md` 第 6 节已登记"自定义 CSS 依赖
Streamlit 内部 DOM、升级易失效"的技术债。因此本设计强制原则：**

> **能用官方机制实现的，不写 CSS。** 每一节都标注实现方式（配置 / 官方 API / CSS）。

核查 Streamlit 1.61.1 能力后确认（已实测 API 签名）：

| 视觉目标 | 官方机制 | 是否需要 CSS |
|---|---|---|
| 品牌主色、背景、字体、圆角 | `[theme]` 段 | 否 |
| 侧栏与主区明度差 | `[theme.sidebar]` 段 | 否 |
| 标题层级（字号/字重） | `headingFontSizes` / `headingFontWeights` | 否 |
| 卡片边界（描边） | `st.container(border=True)` | 否 |
| 状态色块 | `st.success/warning/info/error` + `st.badge` | 否 |
| 图表配色统一 | `chartCategoricalColors`（**须正好 10 色**） | 否 |
| matplotlib 图表风格 | 代码内 `rcParams`（属工具层，见第 5 节） | 否 |

**结论：三层优化可以做到零新增 CSS。** 现有 `_inject_scroll_css()` 保持不变
（它是布局必需，已在技术债登记），本次不扩大 CSS 面。

---

## 2. 第一层：主题配置（最高杠杆，改一处全局生效）

### 2.1 新增 `.streamlit/config.toml`（含双主题）

`.gitignore` 第 3 行已排除 `.streamlit/secrets.toml`，说明该目录是**预期存在**的，
新增 `config.toml` 符合既有结构，不违反任何纪律（且该文件不含密钥，应入库）。

**已核实的两条官方规则（决定写法）**：
1. **未在 `[theme.light]`/`[theme.dark]` 中设置的项，从 `[theme]` 继承**——故
   `[theme]` 放"两套主题共用"的值（主色、圆角、字体层级、图表配色），
   两个子表只放**必须分主题覆盖**的值（背景、正文色、边框色）；
2. **侧栏颜色有自动交换规则**：若未定义 `theme.sidebar.backgroundColor`，
   Streamlit 取 `theme.secondaryBackgroundColor`；反之亦然。故侧栏必须
   **显式**定义，否则浅/深两套主题下的侧栏会各自"交换"成意外颜色。

配色方案 **A（靛青 `#2E6F8E`）**，已 owner 确认：

```toml
[theme]
# ---- 两套主题共用（light/dark 未覆盖时继承此处）----
# 品牌主色：X 光/透视的冷色调（靛青）。主按钮、当前标签、滑块等强调元素。
primaryColor = "#2E6F8E"
# 基础圆角（卡片/按钮的统一观感）
baseRadius = "medium"
baseFontSize = 16
# 标题层级：建立"从哪开始读"的视觉引导（h1 明显大于 h2，h3 起收紧）
headingFontSizes = [30, 22, 18, 16, 15, 14]
headingFontWeights = [700, 600, 600, 600, 600, 600]
# 图表配色（Streamlit 硬性要求：正好 10 个颜色）。与主色同族。
# 同时被 app/visual_theme.py 引用（matplotlib 图表），两处必须同一份列表。
chartCategoricalColors = [
  "#2E6F8E", "#4F9BBF", "#7BC0D9", "#A8D5E2", "#C9E4EA",
  "#8FB8A8", "#B7C9A8", "#D9CBA8", "#C9A8B7", "#9E9E9E",
]

# ---- 浅色主题（主区白 / 侧栏浅灰蓝：制造"工作台 vs 控制面板"分区感）----
[theme.light]
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F4F7F9"
textColor = "#1B2A33"
borderColor = "#DCE4E8"

[theme.light.sidebar]
backgroundColor = "#F4F7F9"
secondaryBackgroundColor = "#FFFFFF"

# ---- 深色主题（深蓝灰底，非纯黑：长时间阅读不刺眼）----
[theme.dark]
backgroundColor = "#0E1A20"
secondaryBackgroundColor = "#16262E"
textColor = "#E3EDF2"
borderColor = "#2A3F4A"

[theme.dark.sidebar]
backgroundColor = "#16262E"
secondaryBackgroundColor = "#1E333D"
```

**为什么 light 下侧栏用略深灰蓝**：主区白、侧栏浅灰蓝，既有区分又不压迫。
**为什么 dark 下底不用纯黑**：`#0E1A20` 是带蓝调的深色，与主色同族且比纯黑
减少眩光——这是深色主题的常见做法。

### 2.2 深色模式的实现与切换（owner 已确认要做）

**实现方式：零代码，纯配置。** 已核实官方文档："配置好 light 与 dark 两套主题后，
用户可通过**设置菜单（settings menu）**切换"，**无需任何额外配置**。

即：
- 用户在 Streamlit 右上角菜单 → Settings → Theme 处切换浅/深；
- 也可"跟随系统"（Streamlit 默认行为）；
- 本设计**不新增应用内切换开关**（那需要写代码 + session_state，
  与"零代码"目标冲突，且 Streamlit 已原生提供入口）。

**随之产生的约束（必须遵守，否则深色下会出现看不清的元素）**：

| 约束 | 原因 | 本次如何保证 |
|---|---|---|
| 不硬编码颜色（如 `#FFFFFF`、`black`） | 深色下不可读 | 面板/组件只用 Streamlit 原生元素；无自绘 HTML |
| 自定义 HTML/CSS 不用固定色 | 同上 | 本次零新增 CSS；`_inject_scroll_css` 只含 `overflow`/`padding`（无颜色） |
| matplotlib 图表背景须**透明**或**适配主题** | 图表是 `st.image` 载入的 png，深色页面上白底图会"贴白块" | 见 5 节：`savefig(transparent=True)` + 文字用主题无关的中性色 |

> 最后一行是深色模式**唯一有真实技术难度**的地方，已在 5 节单独设计。
> 另注：`plot_chart.py` 现有注释说明"图表内文字一律用英文"（防中文字形方框），
> 该约束不变。

### 2.3 配色决策（已确认）

**方案 A：靛青 `#2E6F8E`**（owner 2026-09-11 确认）。备选方案（墨绿/紫罗兰）
不实施，仅记录于此以备将来调整。

### 2.4 现有 CSS 的处理

`_inject_scroll_css()` **不动**。它管的是"滚动行为"（布局必需），
与"配色/圆角/字体"（本次范围）正交；且其内容**无任何颜色声明**，
天然兼容深色模式（见 2.2 约束表）。技术债条目保留。

---

## 3. 第二层：视觉层级（"感觉变好看"的真正来源）

### 3.1 用描边容器给面板加边界（owner 已确认采用）

**这是本层最推荐的一项**——成本极低、观感提升明显。改动点：

侧栏（`streamlit_app.py` 的 sidebar 块）：
- "模型设置""数据加载""Token 统计""上下文管理"四块各自包进
  `st.container(border=True)`，替代当前的 `st.divider()` 串联；
- 每块加一个小标题（`st.markdown("**Token 统计**")` 已有，其余补齐）。

右栏（`app/ui/components.py`）：
- `render_dataset_overview` 的五段已用 `st.divider()` 分隔——**保留 divider
  （语义分段）**，但把"流清单"表格所在区块用 `container(border=True)` 包一层，
  使长表格有明确边界。

**为什么用 border 而非背景色**：`secondaryBackgroundColor` 已被 expander/输入框
占用，再叠背景色会与组件自身底色冲突；描边更克制、不干扰内部组件。
**且描边在深色模式下自动跟随 `borderColor`**（已在 2.1 为 light/dark 分别定义），
无需额外适配——这是选描边而非背景色的另一个理由。

### 3.2 标题层级与副标题弱化

- `st.title("🩻 Embodied-data-Xray")` 保持（主标题）；
- 其下 `st.caption(...)` 改为更克制的说明（现有文案已合适，仅依赖 2.1 的
  `headingFontSizes` 建立层级即可）；
- 各面板内的 `st.markdown("**xxx**")` 保持不变（属 h 之外的粗体小标题，
  与 heading 层级配合形成"页面标题 → 区块标题 → 粗体小标题"三级）。

### 3.3 状态呈现统一约定（新增 `app/ui/constants.py` 约定 + 组件改造）

**约定（写进常量模块的 docstring，作为团队规则）：**

| 语义 | 呈现 | 用法 |
|---|---|---|
| 通过 / 成功 | `st.success` 或绿色 badge | 检查通过、加载成功 |
| 警告 / 需注意 | `st.warning` 或黄色 badge | 单位未知、时钟矛盾、截断 |
| 未检查 / 未知 | `st.info` 或灰色 badge | **必须与"通过"区分**（沿用既有纪律） |
| 错误 / 失败 | `st.error` 或红色 badge | 加载失败、本轮未完成 |
| 布尔能力标签 | **保持 `✓ ✗`**（见下） | 能力标签行 |

**关键取舍：能力标签行的 `✓ ✗` 保留，不改成彩色图标。** 理由：
- 该行是**密集的能力矩阵**（5 项一行），换成彩色 badge 会让侧栏/面板变花；
- `✓ ✗` 是纯符号，**没有 emoji 的字体不一致问题**，渲染稳定；
- `components.py:268-281` 的注释已说明该行的排布经过考量（IMU 无轴数时不显示冗余）。

**改动**：`_render_semantic_progress` 与 `_render_quality_warnings` 里
`st.caption("⚠️ ...")` 改为 `st.warning(...)`（原生色块，醒目不刺眼）；
"已分类 ✓" 用 `st.success`；"尚未检查"用 `st.info`。
**注意**：`docs/数据集状态面板设计.md` 5 节要求"未检查"与"无问题"措辞区分——
改色块后该措辞约束不变（`st.info` vs `st.success` 恰好强化了这个区分）。

### 3.4 图标语言统一

**规则**：同一类用途只用一种风格，且**控制总量**。

| 用途 | 现状 | 统一后 |
|---|---|---|
| 产品标 | `🩻` | 保留（页面 favicon + 标题各一处） |
| 状态 | `✓ ✗` 与 `⚠️` 混用 | 状态交由色块承担（3.3），**面板内不再手写 ⚠️** |
| 操作按钮 | `🔁 ✏️ ＋` | 保留这三个（数量少、语义明确） |
| 会话标签 | `●` | **已在上一轮移除**（改 primary 填充色），保持 |

**明确不做**：不给每个工具/面板加图标。这个项目信息密度已高，
加图标会加剧视觉噪音。"少而一致"优于"多而丰富"。

---

## 4. 第三层：细节打磨

### 4.1 侧栏信息密度

四块面板用 `container(border=True)` 后，块间**不再需要 `st.divider()`**
（描边本身即边界），减少横线数量——这是"减噪"而非"加料"。

### 4.2 空状态的引导统一

现有空状态文案风格不一（`st.info("暂无图表。请先通过对话生成图表。")`、
`st.info("当前会话暂无分析结果。")` 等）。统一为**"现状 + 下一步"**两段式
（现状已基本符合，仅核对一致性，不做大改）。

### 4.3 主操作按钮的一致性

已用 `type="primary"` 的：加载路径数据集、保存配置、压缩历史并重试。
核对一遍：**每个面板最多一个**主按钮（避免同屏多个高亮争抢注意力）。
如"数据加载"面板内已有主按钮，其上传/示例按钮保持 secondary（现状正确）。

---

## 5. 图表观感与深色模式适配（跨层，属工具层改动）

`plot_chart.py` 用 matplotlib 默认样式，且 `_save_fig` 以**不透明白底**保存：

```68:70:app/tools/plot_chart.py
def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=100)
```

**这是"割裂感"的主要来源**（浅灰底白网格，与页面配色无关），
**也是深色模式下唯一有真实技术难度的地方**：图表是 png 经 `st.image` 载入，
白色不透明底在深色页面上会呈现为一块刺眼的"白贴片"。

### 5.1 方案：透明底 + 主题中性色文字 + 统一配色

```python
from app.visual_theme import CHART_COLORS, CHART_INK, CHART_GRID

matplotlib.rcParams.update({
    "figure.facecolor": "none",      # 透明：随页面底色走（深/浅都适配）
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "savefig.facecolor": "none",
    "axes.edgecolor": CHART_GRID,
    "axes.grid": True,
    "grid.color": CHART_GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,        # 去上右边框（减噪）
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "font.size": 10,
    # 文字/刻度用中性灰：在浅色底与深色底上都可读（不能纯黑也不能纯白）
    "text.color": CHART_INK,
    "axes.labelcolor": CHART_INK,
    "xtick.color": CHART_INK,
    "ytick.color": CHART_INK,
    "axes.prop_cycle": matplotlib.cycler(color=CHART_COLORS),
})
```

**关键设计判断：文字用中性灰而非黑/白。**
深色页面上深色文字不可读，浅色页面上白色文字不可读；图表 png 是**静态图片**，
无法随主题切换重绘。故文字与网格必须取**中间明度的中性色**
（如 `#6B7C86`，深底/浅底均可辨识），这是"一张图同时适配两种主题"的
唯一可行解（避免"生成两份图"的复杂度）。

**配色（与主色同族，owner 已确认）**：`CHART_COLORS` 与 `config.toml` 的
`chartCategoricalColors` **必须是同一份列表**（值相等，且有测试断言）。

### 5.2 配色的模块归属（已核实，结论确定）

现状 `app/ui/constants.py` **只含纯常量、不 import streamlit**（已读全文确认），
技术上可被工具层 import，但**不应这样做**：

> `app/ui/` 目录的语义是"Streamlit 界面层"（`ARCHITECTURE.md` 3.6：
> "全项目唯一 import streamlit 的地方"）。让 `app/tools/plot_chart.py`
> 去 import `app/ui/...`，会在**依赖方向上**造成"工具层依赖 UI 层"的错觉——
> 即便当前无循环依赖，也违背"业务逻辑与 UI 完全解耦"的设计目标
> （`ARCHITECTURE.md` 1 节目标 1），且将来任何人在 `constants.py` 里
> 加一行 `import streamlit` 就会真的引入违规依赖。

**决策**：新增跨层视觉常量模块 **`app/visual_theme.py`**（无 streamlit 依赖），
`app/ui/constants.py` 与 `app/tools/plot_chart.py` 都从它取值。
命名上不属 `ui/`、不属 `tools/`，明确表达"跨层的视觉定义"。

模块内容（草案）：

```python
# app/visual_theme.py（无 streamlit 依赖，UI 层与工具层共用的视觉常量）
CHART_COLORS: list[str] = [  # 10 色，与 .streamlit/config.toml 的
    "#2E6F8E", "#4F9BBF", ...  # chartCategoricalColors 必须一致
]
CHART_INK = "#6B7C86"    # 中性灰：深/浅底都可读
CHART_GRID = "#B8C4CB"   # 网格线（比文字更淡）
```

### 5.3 其它约束

- **不引入新依赖**（不装 seaborn 等）；
- 中文乱码防护不变（`_safe_title` 保留，图表文字仍走英文）；
- 去上右边框属"减噪"，与 1.1 诊断 #5 一致。

---

## 6. 改动范围与 commit 拆分

| 序 | Commit | 内容 | 依赖 |
|---|---|---|---|
| V1 | 主题配置（含双主题） | 新增 `.streamlit/config.toml`（2.1，含 light/dark 两套与各自 sidebar）；README 增"界面主题与深色模式"说明 | — |
| V2 | 跨层视觉常量 | 新增 `app/visual_theme.py`（配色/中性色常量，无 streamlit 依赖）；`app/ui/constants.py` 引用之 | V1 |
| V3 | 面板卡片化 | 侧栏四块 + 右栏流清单改 `st.container(border=True)`；清理冗余 `st.divider()` | V1 |
| V4 | 状态色块统一 | `components.py` 的 ⚠️/✓ 改原生色块；"未检查 vs 无问题"措辞约束保持 | V3 |
| V5 | 图表风格统一 + 深色适配 | `plot_chart.py` rcParams（透明底 + 中性色文字）+ 配色引用 `app/visual_theme`；单测断言 | V2 |
| V6 | 行为回归 + 文档 | `docs/行为测试.md` 追加视觉一致性回归；README 深色模式说明 | V1–V5 |

**改动文件**：
`.streamlit/config.toml`（新）、`app/visual_theme.py`（新）、
`app/ui/constants.py`、`app/ui/components.py`、`app/tools/plot_chart.py`、
`streamlit_app.py`、`README.md`、测试。

**明确不做**：
- 不自定义 CSS 调间距/圆角（走主题配置）；
- **不做应用内主题切换开关**（Streamlit 设置菜单原生提供，见 2.2）；
- 不做品牌 Logo 图片（保持 emoji `🩻`，避免引入二进制资源）；
- 不改 `_inject_scroll_css()`；
- 不给每个工具加图标。

---

## 7. 测试计划（视觉改动的可测部分）

视觉改动难做像素级断言，故采用"**可测的契约 + 人工验收**"组合：

**单元（可断言的部分）**
- `app/visual_theme.py`：配色列表长度为 10（Streamlit 对 `chartCategoricalColors`
  的硬要求）；同一列表被 UI 与图表两处引用（值相等）；
- `.streamlit/config.toml`：可被 `tomllib` 解析（合法 TOML）；
  - `[theme]` 段含 `chartCategoricalColors` 且长度 == 10；
  - **配色一致性**：`config.toml` 的 `chartCategoricalColors`
    与 `app/visual_theme.CHART_COLORS` **值完全相等**（防 UI/图表配色漂移）；
  - **双主题完整**：`[theme.light]`、`[theme.dark]` 均存在且各自含
    `backgroundColor` / `textColor`；`[theme.light.sidebar]`、
    `[theme.dark.sidebar]` 均**显式**定义 `backgroundColor`
    （防侧栏自动交换规则产生意外颜色，见 2.1）；
- `plot_chart.py`：`rcParams` 生效——生成一张图后断言
  `savefig.transparent` 为真 / 图片存在透明像素（`PIL` 读回检查 alpha 通道）；
  `axes.spines.top` 不可见；文字色为 `CHART_INK`；
- 状态色块：`_render_quality_warnings` 在有告警时产生 `st.warning`（AppTest
  断言 `at.warning` 非空），无告警且已检查时产生 `st.success`，
  未检查时 `st.info`（**同时断言三者互斥**，守住"未检查≠无问题"纪律）。

> 透明底断言建议：用 `PIL.Image.open(p).convert("RGBA")` 读回，
> 断言四角像素 alpha < 255（背景透明）。这是"深色模式不出现白贴片"的
> **可自动化的回归守护**——否则该问题只会在深色模式下肉眼可见。

**人工验收（视觉的唯一可信判据）**
1. 启动后整体观感：主色是否统一（按钮/当前标签/滑块同色）；
2. 侧栏与主区是否有明确分区感；
3. 标题是否有层级（h1 明显大于 h2）；
4. 生成一张 `multi_stream_overlay`：配色是否与页面同族、是否去了上右边框；
5. 触发一次单位告警：状态是否为原生黄块（而非 emoji 文本）；
6. **深色模式专项**（右上角菜单 → Settings → Theme 切到 Dark）：
   - 侧栏与主区在深色下仍有分区感；
   - **图表不出现白色贴片**（透明底生效，最关键的一项）；
   - 图表文字/网格在深色底上可读（中性灰是否足够）；
   - 无任何元素出现"看不清/黑字黑底"。

> **截图对比建议**：改动前 / 浅色后 / 深色后 各存一张整页截图（本地即可），
> 人工对比——这是视觉改动最有效的验收方式。

---

## 8. 风险

| 风险 | 应对 |
|---|---|
| 主题配置与用户本地全局主题冲突 | 项目级 `config.toml` 优先级高于全局（官方文档确认），行为可预期 |
| `chartCategoricalColors` 长度不为 10 导致启动报错 | 单测断言长度；TOML 解析测试 |
| UI 配色与图表配色漂移（改一处漏一处） | 单测断言 `config.toml` 与 `visual_theme.CHART_COLORS` 值相等 |
| **深色模式：图表白底"贴片"** | 透明底 `savefig` + `PIL` alpha 断言（见 7 节），这是唯一的自动化守护 |
| **深色模式：中性灰在深底上对比度不足** | 人工验收第 6 项专项确认；必要时调亮 `CHART_INK` |
| 侧栏未显式定义颜色 → 自动交换成意外色 | 两个 `[theme.*.sidebar]` 均显式定义 `backgroundColor`；TOML 测试断言存在 |
| 描边容器嵌套后视觉反而更重 | 人工验收；必要时只保留侧栏卡片、右栏回退 divider |
| 工具层反向依赖 UI 常量（分层违规） | 已定：配色常量放跨层的 `app/visual_theme.py`（无 streamlit 依赖），见 5.2 |
| matplotlib rcParams 是全局状态，影响其它图表 | 现有全部图表都在 `plot_chart.py`；rcParams 在同模块设置，影响面一致（且更统一） |
| 视觉改动无回归保护，后续易回退 | 可测契约（配色长度与一致性、透明底、状态色块互斥）进测试套件 |

---

## 9. 决策记录（2026-09-11 owner 拍板）

| # | 决策项 | 结论 |
|---|---|---|
| 1 | 配色方案（2.3） | **A 靛青 `#2E6F8E`**（墨绿/紫罗兰不实施） |
| 2 | 侧栏分区手段（3.1） | **描边卡片**（`st.container(border=True)`）+ `[theme.*.sidebar]` 明度差 |
| 3 | 深色模式（2.2） | **做**；纯配置实现（`[theme.dark]`），切换走 Streamlit 原生设置菜单，**不加应用内开关** |
| 4 | 图表配色（5 节） | **与主色同族**（`CHART_COLORS` 与 `chartCategoricalColors` 同一份列表） |

**据决策追加的设计要求**：深色模式引入"图表白底贴片"这一新问题，
故 5.1 定为**透明底 + 中性色文字**方案，并在 7 节加 `PIL` alpha 回归断言。

---

_状态：决策已确认，进入编码阶段（6 个 commit：V1–V6）。_
