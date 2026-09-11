# UI 工程质量与配置化设计（UI 优化第 5 项）

> 状态：**草案，等待 owner 确认后进入编码阶段**。
> 遵循 AGENTS.md 第 4/5/6 条；关联总纲 `docs/UI优化总纲与输出目录改造设计.md` 第 2 节 #5。
> 关联：`streamlit_app.py`、`app/ui/components.py`、`app/ui/data_loader_panel.py`、
> `app/config/settings.py`

---

## 1. 问题（已核实代码）

### 1.1 CSS hack 脆弱且含未解释的魔法数字

```43:57:streamlit_app.py
    st.markdown(
        """
        <style>
        /* 页面主体不滚动：左右栏各自在固定高度容器内滚动，避免双重滚动条 */
        .block-container { overflow: hidden; }
        /* 输入框钉底（fixed）会盖住容器底部内容：给主区底部预留输入框高度 */
        .block-container { padding-bottom: 130px; }
        /* 滚动锚定：聊天/面板容器内新内容追加时尽量保持贴底/原位置稳定 */
        [data-testid="stVerticalBlock"] > div {
            overflow-anchor: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
```

问题：

1. **`padding-bottom: 130px` 是未解释的魔法数字**——注释说了"预留输入框高度"
   但没说 130 从何而来。输入框高度、字体、浏览器缩放、Streamlit 版本任一变化
   都会导致遮挡或过多留白；
2. **依赖 Streamlit 内部 DOM 与类名**：`.block-container`、
   `[data-testid="stVerticalBlock"]` 是**实现细节**，Streamlit 升级可能改名，
   届时页面布局静默失效（不会报错，只是体验变差）；
3. **`overflow: hidden` 是全局副作用**：它锁掉了整个页面的滚动，一旦某处内容
   超出（如未来的长表格、对话框），用户无法滚动查看；
4. **无任何测试覆盖**：`tests/test_streamlit_app.py` 只断言了 `chat_input` 的
   缩进与容器顺序，**没有**任何针对 CSS 生效/不生效的检查。

### 1.2 魔法数字散落

| 值 | 位置 | 含义 |
|---|---|---|
| `_SCROLL_HEIGHT = 600` | `streamlit_app.py:31` | 左右栏滚动容器高度（已有模块常量，✅ 好） |
| `130px` | `streamlit_app.py:49` | 输入框预留高度（**未常量化、未解释**） |
| `_MAX_UPLOAD_MB = 200` | `data_loader_panel.py:20` | 上传上限（模块常量，但**属配置项却硬编码在 UI**） |
| `len(ds_id) > 18` | `streamlit_app.py:178` | 标签名截断长度（**未常量化**） |
| `[1, 1.2]` | `streamlit_app.py:215` | 左右栏宽度比（未常量化） |

AGENTS.md 第 6 条："配置项集中在 `app/config/`"——`_MAX_UPLOAD_MB` 属明确违反。

### 1.3 turn 累加逻辑重复三处（本项目最实质的隐患）

```254:262:streamlit_app.py
                            if turn.usage:
                                cumulative["input_tokens"] += turn.usage.get(
                                    "input_tokens", 0)
                                cumulative["output_tokens"] += turn.usage.get(
                                    "output_tokens", 0)
                                cumulative["total_tokens"] += turn.usage.get(
                                    "total_tokens", 0)
                            cumulative["rounds"] += 1
                            _accumulate_metrics(cumulative, turn)
```

```299:304:streamlit_app.py
            if turn.usage:
                cumulative["input_tokens"] += turn.usage.get("input_tokens", 0)
                cumulative["output_tokens"] += turn.usage.get("output_tokens", 0)
                cumulative["total_tokens"] += turn.usage.get("total_tokens", 0)
            cumulative["rounds"] += 1
            _accumulate_metrics(cumulative, turn)
```

```111:122:streamlit_app.py
def _accumulate_metrics(cumulative: dict, turn) -> None:
    ...
    cumulative["duration_ms"] = int(cumulative.get("duration_ms", 0)) + int(
        m.get("duration_ms", 0) or 0)
    cumulative["n_model_calls"] = int(cumulative.get("n_model_calls", 0)) + int(
        m.get("n_model_calls", 0) or 0)
```

即：**token 三段累加写了两遍**（编辑重发分支 + 正常分支），metrics 累加单独一个
函数但调用点仍要人工记得调。这在第 1 项（流式）与第 6 项（重试）要新增
**第三个/第四个调用点**时会直接放大风险——漏一处就出现"统计对不上"的怪象。

### 1.4 UI 层对 `ChatTurn` 的层层 `.get()` 取值

```182:186:streamlit_app.py
        last_usage = (
            messages[-1].get("turn").usage
            if messages and messages[-1].get("turn") is not None
            else None
        )
```

`messages[-1].get("turn").usage` 这类写法假设了消息 dict 的结构，
一旦结构变化（如流式改造后 `turn` 可能为 None 的时机不同）就会 AttributeError。
**本次不改消息结构**（避免范围蔓延），但把取值收敛到一个 helper 里。

---

## 2. 设计

### 2.1 新增 `app/ui/constants.py`（UI 层的集中常量）

```python
"""UI 层常量（仅 UI 关注的表现参数；业务阈值仍在 app/config/）。

为什么需要：此前魔法数字散落在 streamlit_app.py 与各 panel 中，
第 1/3/6 项改动都要复用这些值（滚动高度、标签截断、栏宽比），
集中后改一处即全局生效，且便于在文档中说明每个值的依据。
"""

# 左右栏滚动容器高度（px）。
# 依据：常见笔记本视口高度约 800~900px，减去标题栏/标签条/输入框后
# 留给内容区约 600px。过大会导致页面级滚动与容器滚动叠加。
SCROLL_HEIGHT = 600

# 输入框钉底时为页面底部预留的高度（px）。
# 依据：st.chat_input 钉底后约 90~110px（含内边距与安全间距），
# 取 130 留出余量，避免最后一条消息被遮挡。
CHAT_INPUT_RESERVE_PX = 130

# 会话标签名最大显示长度（字符）；超出以 … 结尾。
# 依据：标签条需并排显示多个会话，超过约 18 字符会挤压其它标签。
SESSION_LABEL_MAX_CHARS = 18

# 左右栏宽度比（左：对话，右：展示区）。
COLUMN_RATIO = (1, 1.2)
```

**关于 `_inject_scroll_css` 的 130px**：改为引用 `CHAT_INPUT_RESERVE_PX`
（f-string 注入），并在注释中写明依据。**不追求"精确自适应"**——那需要
JS 测量，超出本项目"不做前端 hack"的取向；改为"有依据的常量 + 注释"是可接受的
工程折中。

### 2.2 把 `_MAX_UPLOAD_MB` 提升为配置项

`app/config/settings.py`：

```python
# 单文件上传大小上限（MB）：仅辅助通道，目录型数据集请走路径输入。
upload_max_mb: int = Field(default=200, ge=1)
```

- `.env`: `UPLOAD_MAX_MB=200`；
- `data_loader_panel.py` 改为读 `get_settings().upload_max_mb`；
- 理由：AGENTS.md 第 6 条明确要求；且不同用户磁盘/网络条件不同，这是**真实可调项**。

### 2.3 抽出 `_record_turn()`（本项最重要的改动）

在 `streamlit_app.py` 抽出：

```python
def _record_turn(cumulative: dict, turn, messages: list[dict]) -> None:
    """把一轮结果记入会话统计与消息列表（token / 轮数 / 耗时 / 往返次数）。

    为什么集中：此前 token 累加在"正常输入"与"编辑重发"两处各写一遍，
    metrics 累加单列一个函数但调用点靠人工记得——任何新增交互入口
    （流式 / 重试 / 重新生成）都可能漏掉某处，导致统计静默偏差。
    集中后所有入口调用同一个函数，杜绝漏加。
    """
    if turn.usage:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            cumulative[key] = cumulative.get(key, 0) + turn.usage.get(key, 0)
    cumulative["rounds"] = cumulative.get("rounds", 0) + 1
    m = getattr(turn, "metrics", None) or {}
    cumulative["duration_ms"] = int(cumulative.get("duration_ms", 0)) + int(
        m.get("duration_ms", 0) or 0)
    cumulative["n_model_calls"] = int(cumulative.get("n_model_calls", 0)) + int(
        m.get("n_model_calls", 0) or 0)
    messages.append({"role": "assistant", "content": turn.reply, "turn": turn})
```

- **行为必须与现状逐字段一致**（现有三处语义合并，见 1.3 代码）；
- `_accumulate_metrics` 合并进本函数（原函数删除）；
- 三处调用点（正常输入、编辑重发、未来流式/重试）统一调用；
- **顺序契约**：`messages.append` 也在本函数内 —— 现状两处都是
  "先累加 usage → rounds → metrics → append 消息"，顺序保持。

**风险点**：`_record_turn` 内部 append 消息，与现有"先 `st.markdown` 渲染再
append"的顺序**不同**（现状 append 在渲染之后，`streamlit_app.py:305`）。
需确认无影响：`messages` 只用于**后续轮次**的渲染，当前轮已用局部变量渲染完，
append 时机不影响本轮显示。**实现时须验证**（见第 4 节用例）。

### 2.4 收敛 `messages[-1].get("turn")` 取值

```python
def _last_turn(messages: list[dict]):
    """返回最近一条 assistant 消息的 ChatTurn（无则 None）。"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("turn")
    return None
```

- 现状只取 `messages[-1]`；但编辑重发后 `messages[-1]` 一定为 assistant，
  正常输入后亦然 —— 用 `reversed` 查找更健壮（且语义正确：要"最近一次
  assistant 的 usage"）；
- 顺带修掉 `.get("turn").usage` 的潜在 AttributeError。

---

## 3. 改动范围

| 文件 | 改动 |
|---|---|
| `app/ui/constants.py` | 新增（UI 表现常量） |
| `streamlit_app.py` | 引用常量；新增 `_record_turn` / `_last_turn`；删除 `_accumulate_metrics`；三处调用点统一 |
| `app/config/settings.py` | 新增 `upload_max_mb` |
| `app/ui/data_loader_panel.py` | 读配置而非模块常量 |
| `.env.example` / `README.md` | 同步 `UPLOAD_MAX_MB` |

**明确不做**：
- 不用 JS 精确测量输入框高度（超出项目取向）；
- 不重写 CSS 方案（保留现有 hack，仅加注释与常量，并在第 6 节列为技术债）；
- 不动 `RunContext` / `ChatService`；
- 不改消息 dict 结构。

---

## 4. 测试计划

**单元/结构**
- `_record_turn`：给定 usage/metrics → cumulative 三个 token 字段 + rounds +
  duration_ms + n_model_calls 全部正确累加；`usage=None` 时不加 token 但
  rounds 仍 +1（**与现状一致**：现状 `if turn.usage` 外面才是 `rounds += 1`）；
- `_record_turn` 两次调用 → 累加而非覆盖；
- `_last_turn`：空列表 → None；末尾为 user 消息 → 返回更早的 assistant 的 turn；
- 常量引用：`_inject_scroll_css` 产出的 CSS 含 `padding-bottom: 130px`
  （断言常量被真正使用，防"定义了但没引用"）；
- 配置：`UPLOAD_MAX_MB` 生效（设不同值 → 上传校验阈值随之变化）。

**UI（AppTest）**
- 现有全部用例重跑通过（**本项是纯重构，零行为变化**）；
- 输入一轮后 cumulative 与改动前一致（快照对比）。

**回归关键**
- **流式（第 1 项）与重试（第 6 项）接入后，仍只调用 `_record_turn`**——
  在其代码审查中确认，并补一条静态断言防新增重复累加代码。

---

## 5. 风险

| 风险 | 应对 |
|---|---|
| 重构 `_record_turn` 时行为漂移 | 严格按 1.3 现状语义实现；单测逐字段断言；AppTest 全量重跑 |
| `messages.append` 时机变化影响渲染 | 第 4 节专门验证；assistant 消息渲染用局部变量、不读 messages |
| CSS 类名随 Streamlit 升级失效（未解决） | 记入第 6 节技术债；本项只做常量化+注释，不承诺解决 |
| `upload_max_mb` 新增配置未同步 `.env.example` | 写进 commit 检查清单 |

---

## 6. 遗留技术债（本项记录，不解决）

1. **`_inject_scroll_css` 依赖 Streamlit 内部 DOM**（`.block-container`、
   `data-testid`）——升级 Streamlit 时需重新验收布局；建议在
   `docs/技术债.md` 登记条目（含"验收方法：打开页面确认无双重滚动条、
   最后一条消息不被遮挡"）；
2. **`padding-bottom` 的像素值仍非精确**——若未来输入框高度变化明显，
   需重新测量并更新 `CHAT_INPUT_RESERVE_PX` 与其依据注释。

> 是否把这两条写进 `docs/技术债.md`？该文件在 `.gitignore` 中（第 32 行），
> 属内部记录；建议登记以便后续查询。

---

_状态：草案。请 owner 确认：2.2 新增 `UPLOAD_MAX_MB` 配置是否必要
（若不必要则保留模块常量但补注释）；6 节的两条技术债是否需要登记。_
