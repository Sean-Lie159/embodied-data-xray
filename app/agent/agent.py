"""主 Agent 定义与运行入口。

定义面向具身智能数据分析场景的中文主 Agent（system prompt、工具注册），
并封装单轮运行入口 ``run_turn``，处理 max_turns 与错误兜底。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, cast

from agents import Agent, Model, ModelSettings, Runner, RunResult, Tool
from agents.exceptions import MaxTurnsExceeded
from agents.items import RunItem, TResponseInputItem

from app.agent.context import RunContext

# 单轮运行的返回类型。契约：正常分支 result 为 RunResult（含本轮完整结果）；
# MaxTurnsExceeded 分支 result 为 None（此时 final_output 已由本函数生成友好提示）。
RunTurnResult = tuple[str, list[TResponseInputItem], RunResult | None]


@dataclass
class RunMetrics:
    """单轮运行的观测指标（供 UI 展示"这轮跑了多久、几次工具循环"）。

    为什么需要：502 排查时缺少"这一轮实际跑了几次模型往返、耗时多少"的事实，
    只能靠猜（是超时过载还是服务侧抖动？）。本结构把这两个量固定记录下来，
    使同类问题可直接肉眼判定，也是"批量纪律"是否生效的度量。

    注意：仅记录**已实现的调用轮数**（无论成功或失败），异常分支同样回填——
    失败轮没有指标恰恰是最需要看数据的情况。

    Attributes:
        duration_ms: 本轮端到端耗时（毫秒，含工具执行与全部模型往返）。
        n_model_calls: 本轮实际发生的模型往返次数（≈ 工具调用轮数 + 1）。
        n_tool_calls: 本轮 SDK 报告的工具调用次数（result 为 None 时无法获取，
            回填 0）。
        completed: 是否正常完成（False 表示撞 max_turns 或模型 API 异常）。
        error_kind: 失败类别（None 表示正常）；用于 UI 精确渲染与恢复入口——
            ``"context_overflow"`` 时提供"压缩历史并重试"，``"max_turns"`` 与
            ``"api"`` 提供"重试本轮"。新增可选字段，缺省 None（零回归）。
    """

    duration_ms: int = 0
    n_model_calls: int = 0
    n_tool_calls: int = 0
    completed: bool = True
    error_kind: str | None = None


def _extract_raw_usage(result: RunResult | None) -> Any:
    """从 RunResult 提取 SDK 原始 usage 对象（可能为 None）。

    仅用于读取 requests 计数等观测字段；任何结构不符都安全降级为 None。
    """
    if result is None:
        return None
    try:
        state = result.to_state()
        wrapper = getattr(state, "_context", None)
        return getattr(wrapper, "usage", None) if wrapper is not None else None
    except Exception:  # noqa: BLE001 - 观测字段取不到不影响主流程
        return None


def _count_tool_calls(result: RunResult | None) -> int:
    """统计本轮 SDK 报告的工具调用次数（取 RunResult 与 usage 两者的较大值）。"""
    if result is None:
        return 0
    by_items = 0
    try:
        by_items = len(result.tool_input_items or [])
    except Exception:  # noqa: BLE001
        by_items = 0
    usage = _extract_raw_usage(result)
    by_usage = 0
    try:
        by_usage = int(getattr(usage, "requests", 0) or 0)
    except Exception:  # noqa: BLE001
        by_usage = 0
    return max(by_items, by_usage)

# 历史压缩参数的**默认值**（供 CLI 等单会话路径使用）。
#
# 为什么不再用模块级可变全局：UI 多会话下每个会话应有独立的压缩预算——
# 模块级变量会被后配置的会话覆盖（A 会话的历史按 B 会话的预算压缩）。
# 现改为 run_turn 的显式参数，由各 ChatService 传入自己的配置。
_DEFAULT_HISTORY_BUDGET_TOKENS: int = 0
_DEFAULT_HISTORY_KEEP_RECENT_TURNS: int = 3


def configure_history_compaction(*, budget_tokens: int, keep_recent_turns: int) -> None:
    """设置历史压缩的**默认值**（兼容壳：仅 CLI 等单会话路径使用）。

    多会话场景请直接给 run_turn 传 history_budget_tokens / 
    history_keep_recent_turns，不要依赖本函数的全局默认值。

    Args:
        budget_tokens: 历史 token 阈值，超过即自动压缩；<=0 表示关闭自动压缩。
        keep_recent_turns: 压缩时保留最近若干轮的完整工具返回。
    """
    global _DEFAULT_HISTORY_BUDGET_TOKENS, _DEFAULT_HISTORY_KEEP_RECENT_TURNS
    _DEFAULT_HISTORY_BUDGET_TOKENS = max(0, int(budget_tokens))
    _DEFAULT_HISTORY_KEEP_RECENT_TURNS = max(1, int(keep_recent_turns))

# 主 Agent 的中文系统提示词。
SYSTEM_PROMPT: str = """\
你是 Embodied-data-Xray——一名具身智能数据结构透视助手，服务于机器人数据集\（LeRobot、HDF5、Parquet、CSV、JSONL 等）的分析场景。你的专长是像 X 光\一样透视数据的结构骨架：文件构成、时间戳伴随表、嵌套信封、标定信息。\
你通过调用工具完成数据处理，自己不具备直接读写数据的权限。

【工作方式】
1. 先了解数据，再分析数据。用户提出分析问题但尚未加载数据时，先询问数据文件路径，\
并在获得路径后调用 load_dataset 加载。
2. 数据路径不明确或用户需求含糊时，主动追问，不要盲目猜测路径或自行假设数据集内容。

【纪律】
1. 只能基于工具返回的真实结果进行分析和表述，不得假设、推测或编造任何数据内容。
2. 需要数据或统计结果而没有时，先调用对应工具获取，再作答。
3. 工具未提供的信息，明确告知"当前数据无法回答"，绝不臆造数字。在判定"无法回答"之前，\
先检查该工具是否带有可用参数（如基线流、流子集、时间列、定位缺口等）可收窄或指定范围，\
尝试后再下结论，不得因一次默认参数的调用不理想就断言工具能力不足。
4. 当工具的返回中 success 为 false 时，必须如实引用其 user_message 字段的内容告知用户，\
不得自行描述、改写或推测执行结果；若返回中还含有 supported_formats，应一并转达。
5. 任一时刻只有一个"当前数据集"（最后加载的），新加载会替换旧的。此前数据集的信息\
只能引用历史对话中工具真实返回过的数字，且必须说明"这是此前加载 X 时的结果"；\
用户要求分析或对比旧数据集时，提示其需要重新加载该数据集。工具返回中带 dataset / \
dataset_id 字段时，引用其数字须注明所属数据集。
6. 对用户输入中含义不明的字词或未明确提及的概念，简短澄清确认即可，不要展开多段推测性分析。
7. 转达漂移检测结果时必须保留其相对性说明：漂移是相对量，基于时间戳只能测出流间偏移趋势，\
无法判定哪条流是漂移源头；不得暗示某条流是漂移源。
8. 描述图表内容（坐标轴、曲线含义、分组方式）时，只能引用工具返回的 plot_spec 与 \
description，不得自行推测图表的绘制方式；plot_spec 未包含的信息，明确说"工具未提供该信息"。
9. 格式假设：当 load_dataset 返回的未知标签（unknown）占比高、且目录结构符合公开数据集\
格式特征（如 meta/info.json + data/chunk-* → LeRobot v2）时，允许（且鼓励）提出格式假设，\
但必须标注"假设"+ 证据 + "未经工具验证"，并主动请用户确认，提醒用户"确认前请核实，错误确认会被记录并影响后续加载"；用户确认后走既有\
confirm_stream_semantic 路径落盘。**假设不得伪装成结论**——不得在未确认时把假设当作事实陈述。
10. 单位纪律：引用工具返回的时长/间隔/采样率等数值时，以字段名后缀标注的单位为准\
（如 duration_ns 是纳秒、median_interval_ns 是纳秒、actual_rate_hz 是赫兹），禁止自行换算。\
返回中出现"兜底假设""未归一化""不可用"字样，或存在 unit_warnings 列表时，必须如实告知用户\
对应指标不可信及其原因，不得把其数值当作物理值转述。
11. 结果截断纪律（上下文预算）：工具返回中若出现 truncated/omitted 为 true，\
或含 truncation_note 字段，必须**主动告知用户本次结果因体积上限未能完整展示**，\
说明被省略的是哪一部分，并建议分批查看或缩小范围（如指定表名、指定流子集、\
限定时间范围或子目录）；**不得把截断后的部分结果当作完整结论陈述**。\
同理，若历史中出现"[上下文管理]"开头的说明，表明较早轮次的工具返回明细已被\
压缩为摘要——此时引用旧数字仍须标注来源；若具体细节已不可考，明确说\
"该细节已随历史压缩省略，如需请重新调用工具"，不得凭印象补数字。
12. 多时钟与语义假设（闸门）：工具返回含多个时间候选（time_candidates）或
clock_artifact_suspected/clock_conflicts 时，**必须并列呈现各时间口径的指标
与矛盾**，不得只引用其中之一下结论；可提出"传感器时间列应为 X"的假设，但须以
time_column=X 重算验证后再表述。流清单 unknown 占比高且你能从样本结构推断
分组时，应**主动**用 propose_stream_semantics 批量提交假设并转述验证结果；
向用户转述并获明确同意后，以 confirm=True 落盘（一次确认，永久生效）。
**判定变更必须经工具验证与用户确认**——未经确认不得断言"某流就是某类"，
也不得在对话里宣称已落盘。回答涉及流语义（数据概况、清单、模态、分组）时，
未确认的流只能如实说明"语义未分类"，或先经 propose_stream_semantics 验证；
不得凭文件名暗示语义而不加标注。inspect_streams 返回含 unclassified_hint 时，
应向用户转述并建议批量确认。用户用自然语言表达流的类别或命名（如"这些是
触觉数据""tf 是坐标变换""帮我把这些流归类"）时，**自动翻译**为
propose_stream_semantics 流程：自行从流内容/嵌套发现中推断 kind、语义标签、
time_column 与结构证据（不要求用户提供字段路径等内部细节），验证后转述结果
并请用户确认落盘。对用户的表述使用功能语言（如"我把这些流归类为触觉并
记住了"），不要求也不展示工具名与参数。引用 user_confirmed 来源的流标签时，须说明这是
**持久化的确认画像**（存于 outputs/by_dataset/<数据集名>/profile.json，跨会话生效）——
新会话中出现"此前已确认"的标签并非模型记忆，而是画像自动应用；用户可要求
重新确认覆盖，或直接编辑该文件撤销。
13. 容器与视频：h5 / mcap 是"单容器多子流"——问"各子流对齐如何 / 谁截断了"\
时用 align_container_streams（一次看全貌）；需要逐帧残差与漂移时再对单个\
子流用时间同步检查。确认某路相机画面内容（朝向 / 遮挡）时用 \
inspect_video_frame 抽单帧——**画面含义需用户判读，不得凭文件名臆断画面内容**。
14. 批量纪律（效率硬约束）：**一次工具调用能覆盖 N 条流的，绝不拆成 N 次调用**。\
逐流循环调用会让整轮耗时随流数线性增长（真实事故：检查 24 条流时逐流发问，\
整轮串行十余次模型往返，最终触发上游 502 超时）。因此：问"多条流/全部流"的\
丢包、对齐、采样率、异常等问题时，**必须**把范围一次性作为参数传给单个工具\
（如 check_temporal_sync 的 streams 传文件名子串列表，一次即可返回各流明细），\
不得对每条流各调用一次再自行汇总；只有单个工具返回被截断、或确需按流换用\
不同参数（如不同的 time_column / baseline_stream）时才追加调用，且不得重复\
调用已覆盖的流。回答多流问题时，优先引用同一次调用返回的汇总与逐流明细。
15. 耗时纪律（上下文预算之外的另一项硬约束）：工具调用有实际计算与 IO 成本，\
应按"先粗后细"推进——先用一次调用拿到全量概览（对齐全貌、逐流概况、缺失汇总），\
**仅在概览显示某条流确有疑点、或用户明确要求逐帧细节时**，才对该流做深入调用。\
不要为了"更稳妥"而对已通过的工具重复调用；不要把一次调用能回答的问题拆成\
多轮追问；用户问题范围含糊时，先按当前数据集全量执行一次再报告，而非逐条试探。

【表述】
1. 全程用中文回答。
2. 引用具体数字时，说明该数字来自哪个工具的哪次调用（例如"据 profile_data 的统计，……"）。\
3. 面向用户的回复中，优先用功能描述（如"时间对齐检查""数据集概况"）而非内部工具名；\
仅在引用具体数字溯源、或用户明确要求技术细节时，才附带工具名（格式：功能描述 + 括号内工具名）。\
"""


def build_agent(
    model: Model,
    tools: list[Tool],
    model_settings: ModelSettings | None = None,
) -> Agent[RunContext]:
    """构建主 Agent。

    Args:
        model: openai-agents 的 Model 实例。
        tools: 要注册给 Agent 的工具列表（应已过 :func:`guard_tools` 包装）。
        model_settings: 可选的 Agent 级模型设置（当前承载推理档位；
            由 :func:`app.llm.factory.build_model_settings` 构造）。
            None 时不设置，行为与改动前一致（零回归）。

    Returns:
        配置好的 ``Agent`` 实例。
    """
    # 注意：SDK 的 Agent 构造器**不接受 model_settings=None**（会抛
    # TypeError: must be a ModelSettings instance or a dict），因此未配置推理档位
    # 时必须省略该关键字参数，而不是传 None。此处用 dict 展开实现条件传参。
    optional: dict[str, Any] = (
        {"model_settings": model_settings} if model_settings is not None else {}
    )
    return Agent[RunContext](
        name="embodied-data-xray",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=list(tools),
        **optional,
    )


# 各工具"档 1 优先丢弃"的次要字段（按丢弃优先级排列）。
# 原则：先丢可再生的明细/清单，保留结论与计数；未列出的工具只走档 2/3 通用降级。
_TOOL_DROPPABLE: dict[str, tuple[str, ...]] = {
    "load_dataset": ("subdirs", "ext_dist", "streams"),
    "profile_data": ("sample_values", "columns"),
    "inspect_streams": ("streams", "video_streams", "table_streams"),
    "check_temporal_sync": ("streams_status", "per_stream", "gaps"),
    "check_sensor_sanity": ("checks", "skipped_checks"),
    "compute_stats": ("per_episode", "episodes"),
    "plot_chart": (),
    "generate_report": ("report_markdown", "content"),
    "propose_stream_semantics": ("results", "confirmed"),
    "unpack_mcap": ("skipped_topics",),
    # 容器对齐：streams 为逐子流明细（长列表）；warnings 是有损摘要，保留。
    "align_container_streams": ("streams",),
    # 视频抽帧：返回极小（路径 + 元数据），无需降级字段。
    "inspect_video_frame": (),
    # 数据集对比：matched 是同名流逐条对比（长列表）；只保留 summary 与计数。
    "compare_datasets": ("matched", "only_in_a", "only_in_b"),
}


def guard_tools(tools: list[Any], *, budget_tokens: int) -> list[Any]:
    """给工具套上"返回体积护栏"（第 2 层防御，安全网）。

    为什么在 agent 层统一做：即使某个工具自身漏做结构化降级，本层也保证返回
    绝不超出预算——宁可信息不全，不可撑爆上下文导致 HTTP 400。

    Args:
        tools: FunctionTool 列表。
        budget_tokens: 单次工具返回的 token 预算。

    Returns:
        包装后的工具列表（非 FunctionTool 原样透传，不改动其 schema）。
    """
    from agents.tool import FunctionTool

    from app.tools._output_guard import enforce_output_limit

    guarded: list[Any] = []
    for tool in tools:
        if not isinstance(tool, FunctionTool):
            guarded.append(tool)
            continue
        original_invoke = tool.on_invoke_tool
        droppable = _TOOL_DROPPABLE.get(tool.name, ())

        async def _invoke(ctx: Any, input_json: str, *,
                          _orig: Any = original_invoke,
                          _name: str = tool.name,
                          _droppable: tuple[str, ...] = droppable) -> Any:
            raw = await _orig(ctx, input_json)
            try:
                return enforce_output_limit(
                    raw, budget_tokens, droppable=_droppable, tool_name=_name,
                )
            except Exception:  # noqa: BLE001
                # 护栏自身出错不得吞掉工具结果——原样返回（由调用方/SDK 处理）。
                return raw

        guarded.append(
            FunctionTool(
                name=tool.name,
                description=tool.description,
                params_json_schema=tool.params_json_schema,
                on_invoke_tool=_invoke,
                strict_json_schema=tool.strict_json_schema,
                is_enabled=tool.is_enabled,
            )
        )
    return guarded


def classify_model_error(exc: BaseException) -> str:
    """把模型/运行异常归类，供 UI 精确渲染恢复入口（与 _describe_model_error 同判据）。

    Returns:
        "context_overflow"（上下文超限，建议压缩历史后重试）/ "api"（其它模型侧
        故障，建议直接重试）。
    """
    text = str(exc).lower()
    if "context" in text and ("length" in text or "too long" in text
                              or "maximum" in text):
        return "context_overflow"
    return "api"


def _describe_model_error(exc: BaseException) -> str:
    """把模型 API 异常转成**可读的中文提示**（含重试建议）。

    为什么需要：模型侧故障（502 上游错误 / 限流 / 超时 / 鉴权失败）不是本工具
    或数据的问题，但此前会以裸异常外抛、炸穿 UI 页面（用户看到 traceback 白屏，
    聊天记录还可能处于半截状态）。本函数把它们转成用户能理解并知道怎么做的
    提示——符合项目「错误可恢复 + 诚实降级」纪律。

    Args:
        exc: 捕获到的异常。

    Returns:
        面向用户的中文提示文本。
    """
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()

    # 按 HTTP 状态码/错误类型给出**可操作**的提示。
    if "502" in text or "bad gateway" in low or "upstream error" in low:
        return (
            "模型服务暂时不可用（502 上游错误）——这不是数据或分析本身的问题，"
            "通常是模型服务侧临时故障或过载。请稍后重试（直接重发这条消息即可）。"
        )
    if "503" in text or "service unavailable" in low:
        return (
            "模型服务暂时不可用（503）——服务侧过载或正在维护。"
            "请稍后重试（直接重发这条消息即可）。"
        )
    if "504" in text or "timeout" in low or name in ("APITimeoutError", "TimeoutError"):
        return (
            "模型请求超时（504/超时）——可能是本轮上下文较大或服务侧响应慢。"
            "可稍后重试；若反复超时，建议先压缩历史（侧栏「压缩历史」）或缩小问题范围。"
        )
    if "429" in text or "rate limit" in low:
        return (
            "模型请求过于频繁（429 限流）——请稍等片刻再重发；"
            "若持续出现，请检查所用服务商的配额与并发限制。"
        )
    if "401" in text or "403" in text or "invalid api key" in low or "unauthorized" in low:
        return (
            "模型鉴权失败（401/403）——请检查 .env 中的 OPENAI_API_KEY 与 "
            "OPENAI_BASE_URL 是否正确、密钥是否已过期。"
        )
    if "404" in text or "model_not_found" in low or "does not exist" in low:
        return (
            "模型不存在或不可用（404）——请检查 .env 中的 DEFAULT_MODEL "
            "是否为所用服务商支持的模型名。"
        )
    if "context" in low and ("length" in low or "too long" in low or "maximum" in low):
        return (
            "上下文超出模型上限——请先在侧栏点击「压缩历史」，"
            "或缩小本次问题的范围后重试。"
        )
    if "connection" in low or name in ("APIConnectionError",):
        return (
            "无法连接模型服务——请检查网络与 OPENAI_BASE_URL 是否可达"
            "（若使用了代理，确认代理正在运行）。稍后可直接重发本条消息。"
        )
    # 兜底：如实说明类型与摘要，不假装成功、也不裸露 traceback。
    brief = text.strip().splitlines()[0][:200] if text.strip() else "(无详细信息)"
    return (
        f"本轮模型调用失败（{name}）：{brief}\n"
        "这通常不是数据或分析本身的问题。请稍后重试（直接重发本条消息即可）；"
        "若反复出现，请检查 .env 的模型配置与服务商状态。"
    )


def _maybe_compact_history(
    context: RunContext,
    history_input: list[TResponseInputItem] | None,
    *,
    history_budget_tokens: int | None,
    history_keep_recent_turns: int | None,
) -> list[TResponseInputItem] | None:
    """第 3 层防御：历史压缩（在送给模型之前拦截）。

    run_turn 与 stream_turn 共用的前置步骤（抽公共函数避免两套实现漂移）。
    """
    budget = (
        _DEFAULT_HISTORY_BUDGET_TOKENS
        if history_budget_tokens is None else max(0, int(history_budget_tokens))
    )
    keep_recent = (
        _DEFAULT_HISTORY_KEEP_RECENT_TURNS
        if history_keep_recent_turns is None
        else max(1, int(history_keep_recent_turns))
    )
    if history_input is not None and budget > 0:
        from app.agent.history_compaction import (
            compact_history,
            estimate_history_tokens,
        )

        if estimate_history_tokens(history_input) > budget:
            history_input, _stats = compact_history(
                history_input, keep_recent_turns=keep_recent
            )
            context.last_compaction = _stats
    return history_input


def _build_input_items(
    history_input: list[TResponseInputItem] | None, user_input: str
) -> list[TResponseInputItem] | str:
    """组装本轮输入：有历史时拼接，否则仅用户消息。"""
    if history_input is not None:
        user_msg: TResponseInputItem = {"role": "user", "content": user_input}
        return [*history_input, user_msg]
    return user_input


def _fallback_input(
    history_input: list[TResponseInputItem] | None, user_input: str
) -> list[TResponseInputItem]:
    """异常/超限分支的历史：**必须含本轮用户消息**。

    否则用户"直接重发"时本轮问题不在上下文里（模型看不到上一句，回答脱节）。
    成功分支由 SDK 的 to_input_list 自然包含本轮消息。
    """
    return [*(history_input or []), {"role": "user", "content": user_input}]


async def run_turn(
    agent: Agent[RunContext],
    context: RunContext,
    user_input: str,
    history_input: list[TResponseInputItem] | None = None,
    max_turns: int = 15,
    *,
    history_budget_tokens: int | None = None,
    history_keep_recent_turns: int | None = None,
    metrics: RunMetrics | None = None,
) -> RunTurnResult:
    """执行单轮 Agent 运行。

    Args:
        agent: 主 Agent。
        context: 运行时上下文（跨轮共享）。
        user_input: 用户本轮输入。
        history_input: 上一轮返回的 input 列表（用于携带对话历史），首轮为 None。
        max_turns: 单轮最大循环轮数，防死循环。
        history_budget_tokens: 历史压缩阈值（**按会话传入**，多会话各自独立）；
            None 时用模块默认值（CLI 单会话路径）。<=0 关闭自动压缩。
        history_keep_recent_turns: 压缩保留的最近轮数；None 时用模块默认值。
        metrics: 可选，传入一个 :class:`RunMetrics` 实例用于回填本轮耗时与
            模型往返次数（**含异常分支**，失败轮的指标最需要观测）。传 None
            时不做记录，行为与改动前完全一致（零回归）。

    Returns:
        (final_output, next_input, result) 三元组：final_output 为最终回答文本，
        next_input 为可传给下一轮 run 的 input 列表，result 为完整 RunResult；
        当触发 MaxTurnsExceeded 或模型 API 异常（502/限流/超时/网络/鉴权等）时
        result 为 None，此时 final_output 已由本函数生成友好提示（含重试建议），
        调用方不得再对 result 解引用（需判空）。历史保持不变，用户可直接重发。

    Raises:
        ConfigError: 工具或模型配置异常。
    """
    _t0 = time.perf_counter()
    _m = metrics if metrics is not None else RunMetrics()

    def _finish(
        payload: RunTurnResult, *, n_tool_calls: int = 0, completed: bool = True
    ) -> RunTurnResult:
        """统一回填观测指标并返回结果（成功/异常分支共用，避免遗漏）。"""
        _m.duration_ms = int((time.perf_counter() - _t0) * 1000)
        _m.n_tool_calls = n_tool_calls
        _m.completed = completed
        return payload

    # 第 3 层防御：历史压缩。**在送给模型之前**拦截（不是爆了再压）——
    # 历史超阈值即把旧轮次的工具返回原文压缩为结论摘要。
    history_input = _maybe_compact_history(
        context, history_input,
        history_budget_tokens=history_budget_tokens,
        history_keep_recent_turns=history_keep_recent_turns,
    )

    # 组装本轮输入：有历史时，把历史与用户本轮消息拼接；否则仅用用户消息。
    input_items = _build_input_items(history_input, user_input)

    # 兜底范围覆盖**模型侧故障**：不只是 MaxTurnsExceeded，还包括 502/503/504
    # （服务端错误）、限流、超时、网络不可达、鉴权失败等——此前这些会以裸异常
    # 外抛，炸穿 UI 页面（用户看到 traceback 白屏）。现统一转结构化友好提示，
    # 并保留历史与已产生的对话记录（用户可直接重发）。
    try:
        result = await Runner.run(
            agent,
            input=input_items,
            context=context,
            max_turns=max_turns,
        )
    except MaxTurnsExceeded:
        # 用 + 显式拼接字符串（消除隐式拼接告警），文案与原实现一致。
        msg = (
            f"本轮工具调用次数已达上限（max_turns={max_turns}），为避免死循环已停止。"
            + "请尝试更明确地描述需求，或分步提问。"
        )
        # 撞上限说明轮数已到 max_turns：按已知轮数回填（供 UI 显示"跑了多久"）。
        _m.n_model_calls = max(_m.n_model_calls, max_turns)
        _m.error_kind = "max_turns"
        return _finish(
            (msg, _fallback_input(history_input, user_input), None),
            n_tool_calls=0, completed=False,
        )
    except BaseException as exc:  # noqa: BLE001
        # 键盘中断/系统退出不吞（用户主动中断应正常传播）。
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _m.error_kind = classify_model_error(exc)
        return _finish(
            (_describe_model_error(exc),
             _fallback_input(history_input, user_input), None),
            n_tool_calls=0,
            completed=False,
        )

    # final_output 在 SDK（RunResultBase）中类型标注为 Any | None，无法通过泛型收紧；
    # 这里用 str() 强制转成 str，并在类型检查层忽略 Any 告警（运行时行为不变）。
    final: str = str(result.final_output or "").strip()  # pyright: ignore[reportAny]
    next_input = result.to_input_list(mode="normalized")
    _m.n_model_calls = max(1, _count_tool_calls(result) + 1)
    return _finish(
        (final, next_input, result),
        n_tool_calls=_count_tool_calls(result),
        completed=True,
    )


def _extract_tool_name(item: RunItem) -> str:
    """宽容提取工具调用名，失败时降级为原始类型名。

    Args:
        item: 单个 RunResult item（RunItem union）。

    Returns:
        工具名；无法提取时返回 item 的类型名（不静默丢弃）。
    """
    for attr in ("name", "tool_name", "function_name", "call_id"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # 尝试从原始字段提取。raw_item 的类型未在 SDK 中标明，用 cast 显式声明
    # 为 dict，既避免 Unknown 告警又不改变运行时取值行为。
    raw_item = cast("dict[str, object] | None", getattr(item, "raw_item", None))
    if raw_item is not None:
        name = raw_item.get("name") or raw_item.get("tool_name") or raw_item.get("function")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return f"<{type(item).__name__}>"


def format_tool_activity(result: RunResult | None) -> str:
    """从 RunResult 中提取工具调用过程，格式化为可读文本。

    按类型宽容提取：只对工具调用类 item（type 含 tool_call、类名含 ToolCall
    或 function_call）提取工具名；提取失败时降级显示原始类型名，而不是静默丢弃。

    Args:
        result: 单轮运行的完整结果；为 None 时返回空字符串。

    Returns:
        工具调用过程摘要（工具名与调用顺序）。
    """
    if result is None:
        return ""
    lines: list[str] = []
    for item in result.new_items:
        item_type = getattr(item, "type", "") or ""
        class_name = type(item).__name__.lower()
        is_tool_call = (
            "tool_call" in item_type.lower()
            or "toolcall" in class_name
            or "function_call" in item_type.lower()
        )
        if not is_tool_call:
            continue
        name = _extract_tool_name(item)
        lines.append(f"调用工具: {name}")
    return " → ".join(lines)


# ---------------------------------------------------------------------------
# 流式输出（docs/流式输出设计.md）
# ---------------------------------------------------------------------------

# 工具名 → 面向用户的功能描述（播报用）。与 SYSTEM_PROMPT 表述纪律一致：
# 对普通用户用功能语言，不展示内部工具名与参数。
_TOOL_FUNCTION_TEXT: dict[str, str] = {
    "load_dataset": "正在加载数据集",
    "profile_data": "正在生成数据概况",
    "inspect_streams": "正在探测设备清单",
    "check_temporal_sync": "正在检查时间同步",
    "check_sensor_sanity": "正在检查传感器合理性",
    "compute_stats": "正在计算统计指标",
    "plot_chart": "正在绘图",
    "generate_report": "正在生成报告",
    "propose_stream_semantics": "正在验证流语义假设",
    "unpack_mcap": "正在解包容器",
    "align_container_streams": "正在对齐子流",
    "inspect_video_frame": "正在抽查视频帧",
    "compare_datasets": "正在对比数据集",
}


def describe_tool_call(tool_name: str) -> str:
    """把工具名转为面向用户的进行时播报文本（未知工具降级为工具名本身）。"""
    return _TOOL_FUNCTION_TEXT.get(tool_name, f"正在执行 {tool_name}")


@dataclass
class TurnEvent:
    """流式单轮的单个事件。

    kind 取值：
    - ``"delta"``：正文增量（text 为新增片段）；
    - ``"tool"``：工具调用播报（tool_name 为工具名，text 为功能描述）；
    - ``"done"``：收尾（final 为完整正文，next_input/result 供下一轮与统计）。
    """

    kind: str
    text: str = ""
    tool_name: str = ""
    final: str = ""
    next_input: list[TResponseInputItem] | None = None
    result: RunResult | None = None
    error: str | None = None


def _is_tool_call_item(item: Any) -> bool:
    """判断流事件中的 item 是否为工具调用（复用 format_tool_activity 的判据）。"""
    item_type = getattr(item, "type", "") or ""
    class_name = type(item).__name__.lower()
    return (
        "tool_call" in str(item_type).lower()
        or "toolcall" in class_name
        or "function_call" in str(item_type).lower()
    )


async def stream_turn(
    agent: Agent[RunContext],
    context: RunContext,
    user_input: str,
    history_input: list[TResponseInputItem] | None = None,
    max_turns: int = 15,
    *,
    history_budget_tokens: int | None = None,
    history_keep_recent_turns: int | None = None,
    metrics: RunMetrics | None = None,
) -> AsyncIterator[TurnEvent]:
    """流式执行单轮 Agent 运行（yield TurnEvent 序列，最后一项 kind="done"）。

    与 :func:`run_turn` 共用历史压缩、错误兜底与 metrics 回填逻辑（抽公共私有
    函数）；**不改 run_turn 的签名与返回契约**（零回归）。

    中途失败处理（docs/流式输出设计.md 3.4）：若异常发生在已产出若干 delta 之后，
    不再新 yield delta，转而 yield 一个 kind="done"、error 非空的事件——
    **已输出的部分正文保留**（不突然清空），error 为可操作的中文提示。

    Args:
        agent: 主 Agent。
        context: 运行时上下文（跨轮共享）。
        user_input: 用户本轮输入。
        history_input: 上一轮返回的 input 列表，首轮为 None。
        max_turns: 单轮最大循环轮数。
        history_budget_tokens: 历史压缩阈值（按会话传入）；None 用模块默认。
        history_keep_recent_turns: 压缩保留的最近轮数；None 用模块默认。
        metrics: 可选，回填本轮耗时与模型往返次数（含异常分支）。

    Yields:
        TurnEvent：若干 delta/tool 事件，最后一个为 done。
    """
    _t0 = time.perf_counter()
    _m = metrics if metrics is not None else RunMetrics()

    history_input = _maybe_compact_history(
        context, history_input,
        history_budget_tokens=history_budget_tokens,
        history_keep_recent_turns=history_keep_recent_turns,
    )
    input_items = _build_input_items(history_input, user_input)

    acc = ""
    try:
        result = Runner.run_streamed(
            agent, input=input_items, context=context, max_turns=max_turns,
        )
        async for ev in result.stream_events():
            etype = getattr(ev, "type", "") or ""
            # 正文增量。
            if etype == "raw_response_event":
                data = getattr(ev, "data", None)
                delta = getattr(data, "delta", None) if data is not None else None
                dtype = getattr(data, "type", "") or ""
                if isinstance(delta, str) and delta and "output_text" in str(dtype):
                    acc += delta
                    yield TurnEvent(kind="delta", text=delta)
                continue
            # 工具调用播报。
            if etype == "run_item_stream_event":
                item = getattr(ev, "item", None)
                if item is not None and _is_tool_call_item(item):
                    name = _extract_tool_name(item)
                    yield TurnEvent(kind="tool", tool_name=name,
                                    text=describe_tool_call(name))
    except MaxTurnsExceeded:
        _m.duration_ms = int((time.perf_counter() - _t0) * 1000)
        _m.n_model_calls = max(_m.n_model_calls, max_turns)
        _m.n_tool_calls = 0
        _m.completed = False
        _m.error_kind = "max_turns"
        msg = (
            f"本轮工具调用次数已达上限（max_turns={max_turns}），为避免死循环已停止。"
            + "请尝试更明确地描述需求，或分步提问。"
        )
        yield TurnEvent(kind="done", final=acc, error=msg,
                        next_input=_fallback_input(history_input, user_input))
        return
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        # 注意：流式下异常可能发生在已 yield 若干 delta 之后——保留 acc（部分正文），
        # 不丢已展示内容（诚实降级）。
        _m.duration_ms = int((time.perf_counter() - _t0) * 1000)
        _m.completed = False
        _m.error_kind = classify_model_error(exc)
        yield TurnEvent(kind="done", final=acc, error=_describe_model_error(exc),
                        next_input=_fallback_input(history_input, user_input))
        return

    # 正常结束：从 result 取完整正文（可能比 acc 更完整，作为权威值）。
    final: str = str(getattr(result, "final_output", "") or "").strip()
    if not final:
        final = acc
    next_input = result.to_input_list(mode="normalized")
    _m.duration_ms = int((time.perf_counter() - _t0) * 1000)
    _m.n_tool_calls = _count_tool_calls(result)
    _m.n_model_calls = max(1, _m.n_tool_calls + 1)
    _m.completed = True
    yield TurnEvent(kind="done", final=final, next_input=next_input, result=result)
