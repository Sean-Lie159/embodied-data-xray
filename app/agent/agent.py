"""主 Agent 定义与运行入口。

定义面向具身智能数据分析场景的中文主 Agent（system prompt、工具注册），
并封装单轮运行入口 ``run_turn``，处理 max_turns 与错误兜底。
"""

from __future__ import annotations

from typing import cast

from agents import Agent, Model, Runner, RunResult, Tool
from agents.exceptions import MaxTurnsExceeded
from agents.items import RunItem, TResponseInputItem

from app.agent.context import RunContext

# 单轮运行的返回类型。契约：正常分支 result 为 RunResult（含本轮完整结果）；
# MaxTurnsExceeded 分支 result 为 None（此时 final_output 已由本函数生成友好提示）。
RunTurnResult = tuple[str, list[TResponseInputItem], RunResult | None]

# 历史压缩阈值与保留轮数（由服务层按配置注入）。
# 默认 0 = 不自动压缩（未注入配置时保持既有行为，不引入意外副作用）。
_history_budget_tokens: int = 0
_history_keep_recent_turns: int = 3


def configure_history_compaction(*, budget_tokens: int, keep_recent_turns: int) -> None:
    """注入历史压缩参数（由服务层按配置调用一次）。

    Args:
        budget_tokens: 历史 token 阈值，超过即自动压缩；<=0 表示关闭自动压缩。
        keep_recent_turns: 压缩时保留最近若干轮的完整工具返回。
    """
    global _history_budget_tokens, _history_keep_recent_turns
    _history_budget_tokens = max(0, int(budget_tokens))
    _history_keep_recent_turns = max(1, int(keep_recent_turns))

# 主 Agent 的中文系统提示词。
SYSTEM_PROMPT: str = """\
你是一名具身智能数据分析助手，服务于机器人数据集（LeRobot、HDF5、Parquet、CSV 等）的分析场景。\
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

【表述】
1. 全程用中文回答。
2. 引用具体数字时，说明该数字来自哪个工具的哪次调用（例如"据 profile_data 的统计，……"）。\
3. 面向用户的回复中，优先用功能描述（如"时间对齐检查""数据集概况"）而非内部工具名；\
仅在引用具体数字溯源、或用户明确要求技术细节时，才附带工具名（格式：功能描述 + 括号内工具名）。\
"""


def build_agent(model: Model, tools: list[Tool]) -> Agent[RunContext]:
    """构建主 Agent。

    Args:
        model: openai-agents 的 Model 实例。
        tools: 要注册给 Agent 的工具列表（应已过 :func:`guard_tools` 包装）。

    Returns:
        配置好的 ``Agent`` 实例。
    """
    return Agent[RunContext](
        name="embodied-data-agent",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=list(tools),
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


async def run_turn(
    agent: Agent[RunContext],
    context: RunContext,
    user_input: str,
    history_input: list[TResponseInputItem] | None = None,
    max_turns: int = 15,
) -> RunTurnResult:
    """执行单轮 Agent 运行。

    Args:
        agent: 主 Agent。
        context: 运行时上下文（跨轮共享）。
        user_input: 用户本轮输入。
        history_input: 上一轮返回的 input 列表（用于携带对话历史），首轮为 None。
        max_turns: 单轮最大循环轮数，防死循环。

    Returns:
        (final_output, next_input, result) 三元组：final_output 为最终回答文本，
        next_input 为可传给下一轮 run 的 input 列表，result 为完整 RunResult；
        当触发 MaxTurnsExceeded 时 result 为 None，此时 final_output 已由本函数
        生成友好的超限提示，调用方不得再对 result 解引用（需判空）。

    Raises:
        ConfigError: 工具或模型配置异常。
    """
    # 第 3 层防御：历史压缩。**在送给模型之前**拦截（不是爆了再压）——
    # 历史超阈值即把旧轮次的工具返回原文压缩为结论摘要。
    if history_input is not None and _history_budget_tokens > 0:
        from app.agent.history_compaction import (
            compact_history,
            estimate_history_tokens,
        )

        if estimate_history_tokens(history_input) > _history_budget_tokens:
            history_input, _stats = compact_history(
                history_input, keep_recent_turns=_history_keep_recent_turns
            )
            context.last_compaction = _stats

    # 组装本轮输入：有历史时，把历史与用户本轮消息拼接；否则仅用用户消息。
    if history_input is not None:
        user_msg: TResponseInputItem = {"role": "user", "content": user_input}
        input_items: list[TResponseInputItem] | str = [*history_input, user_msg]
    else:
        input_items = user_input

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
        # 保持与原实现相同的 `or` 语义：history 为 None 或空列表时都用仅含
        # 用户消息的列表作为 fallback，避免改变运行时行为。
        fallback_input: list[TResponseInputItem] = history_input or [
            {"role": "user", "content": user_input}
        ]
        return (msg, fallback_input, None)

    # final_output 在 SDK（RunResultBase）中类型标注为 Any | None，无法通过泛型收紧；
    # 这里用 str() 强制转成 str，并在类型检查层忽略 Any 告警（运行时行为不变）。
    final: str = str(result.final_output or "").strip()  # pyright: ignore[reportAny]
    next_input = result.to_input_list(mode="normalized")
    return final, next_input, result


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
