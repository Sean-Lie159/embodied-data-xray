"""主 Agent 定义与运行入口。

定义面向具身智能数据分析场景的中文主 Agent（system prompt、工具注册），
并封装单轮运行入口 ``run_turn``，处理 max_turns 与错误兜底。
"""

from __future__ import annotations

from typing import Any

from agents import Agent, Model, RunContextWrapper, Runner, RunResult
from agents.exceptions import MaxTurnsExceeded

from app.agent.context import RunContext

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
3. 工具未提供的信息，明确告知"当前数据无法回答"，绝不臆造数字。
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

【表述】
1. 全程用中文回答。
2. 引用具体数字时，说明该数字来自哪个工具的哪次调用（例如"据 profile_data 的统计，……"）。\
"""


def build_agent(model: Model, tools: list[Any]) -> Agent[RunContext]:
    """构建主 Agent。

    Args:
        model: openai-agents 的 Model 实例。
        tools: 要注册给 Agent 的工具列表。

    Returns:
        配置好的 ``Agent`` 实例。
    """
    return Agent[RunContext](
        name="embodied-data-agent",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=list(tools),
    )


async def run_turn(
    agent: Agent[RunContext],
    context: RunContext,
    user_input: str,
    history_input: list[Any] | None = None,
    max_turns: int = 15,
) -> tuple[str, list[Any], RunResult]:
    """执行单轮 Agent 运行。

    Args:
        agent: 主 Agent。
        context: 运行时上下文（跨轮共享）。
        user_input: 用户本轮输入。
        history_input: 上一轮返回的 input 列表（用于携带对话历史），首轮为 None。
        max_turns: 单轮最大循环轮数，防死循环。

    Returns:
        (final_output, next_input, result) 三元组：final_output 为最终回答文本，
        next_input 为可传给下一轮 run 的 input 列表，result 为完整 RunResult。

    Raises:
        ConfigError: 工具或模型配置异常。
    """
    # 组装本轮输入：有历史时，把历史与用户本轮消息拼接；否则仅用用户消息。
    if history_input is not None:
        input_items: list[Any] | str = list(history_input) + [
            {"role": "user", "content": user_input}
        ]
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
        return (
            f"本轮工具调用次数已达上限（max_turns={max_turns}），为避免死循环已停止。"
            "请尝试更明确地描述需求，或分步提问。",
            history_input
            or [{"role": "user", "content": user_input}],
            None,  # type: ignore[return-value]
        )

    final = (result.final_output or "").strip()
    next_input = result.to_input_list(mode="normalized")
    return final, next_input, result


def _extract_tool_name(item: Any) -> str:
    """宽容提取工具调用名，失败时降级为原始类型名。

    Args:
        item: 单个 RunResult item。

    Returns:
        工具名；无法提取时返回 item 的类型名（不静默丢弃）。
    """
    for attr in ("name", "tool_name", "function_name", "call_id"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # 尝试从原始字段提取。
    raw = getattr(item, "raw_item", None)
    if isinstance(raw, dict):
        name = raw.get("name") or raw.get("tool_name") or raw.get("function")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return f"<{type(item).__name__}>"


def format_tool_activity(result: RunResult | None) -> str:
    """从 RunResult 中提取工具调用过程，格式化为可读文本。

    按类型宽容提取：只对工具调用类 item（type 含 tool_call 或类名含 ToolCall）
    提取工具名；提取失败时降级显示原始类型名，而不是静默丢弃。

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
