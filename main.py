"""命令行多轮对话入口。

在终端与主 Agent 对话，观察其自主调用工具（load_dataset / profile_data 等）。
跨轮共享同一个 RunContext，因此已加载的数据集在后续提问中持续可用。

用法：
    python main.py

退出：输入 exit / quit / 退出 即可。
上下文管理：输入 /compact 手动压缩历史，/history 查看当前历史体积。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from agents import Agent

from app.agent.agent import (
    build_agent,
    configure_history_compaction,
    format_tool_activity,
    guard_tools,
    run_turn,
)
from app.agent.context import RunContext
from app.agent.history_compaction import compact_history, estimate_history_tokens
from app.config import get_settings
from app.llm import build_model
from app.llm.context_window import derive_budget
from app.services.chat_service import extract_usage
from app.tools import (
    align_container_streams,
    annotate_task,
    check_annotation_qc,
    check_dataset_quality,
    check_sensor_sanity,
    compare_datasets,
    compare_table_columns,
    check_temporal_sync,
    compute_stats,
    generate_report,
    inspect_streams,
    inspect_video_frame,
    list_tables,
    load_dataset,
    plot_chart,
    profile_data,
    propose_stream_semantics,
    save_annotations,
    segment_actions,
    unpack_mcap,
)

_EXIT_COMMANDS = {"exit", "quit", "q", "退出", "再见"}
# 上下文管理指令：手动压缩历史 / 查看历史体积。
_COMPACT_COMMANDS = {"/compact", "/压缩", "压缩历史"}
_HISTORY_COMMANDS = {"/history", "/历史", "历史体积"}


def _format_tokens(usage: dict[str, int] | None, cumulative: dict[str, int]) -> str:
    """格式化单轮 + 会话累计的 token 用量与可选成本。

    Args:
        usage: 本轮用量 {input/output/total} 或 None。
        cumulative: 会话累计 {input/output/total/rounds}。

    Returns:
        "[tokens] 输入 X | 输出 Y | 合计 Z | 会话累计 ..." 一行文本；usage 为
        None 时显示"本次未获取到用量"，不显示 0 冒充。
    """
    cost_text = _format_cost(usage, cumulative)
    if usage is None:
        base = "本次未获取到用量"
    else:
        base = (
            f"输入 {usage.get('input_tokens', 0):,} | "
            f"输出 {usage.get('output_tokens', 0):,} | "
            f"合计 {usage.get('total_tokens', 0):,}"
        )
    return (
        f"[tokens] {base} | "
        + f"会话累计 {cumulative.get('input_tokens', 0):,}/{cumulative.get('output_tokens', 0):,}"
        + f"/{cumulative.get('total_tokens', 0):,}（{cumulative.get('rounds', 0)} 轮）"
        + (f" | {cost_text}" if cost_text else "")
    )


def _format_cost(usage: dict[str, int] | None, cumulative: dict[str, int]) -> str:
    """估算本轮与累计成本（美元）；未配置价格或用量缺失时返回空串。

    Args:
        usage: 本轮用量。
        cumulative: 会话累计用量。

    Returns:
        成本文本（如 "≈$0.0021 / 累计 $0.034"）；未启用返回空串。
    """
    settings = get_settings()
    p_in, p_out = settings.price_input_per_mtok, settings.price_output_per_mtok
    if not (p_in > 0 and p_out > 0):
        return ""
    if usage is None:
        return ""
    cost = usage.get("input_tokens", 0) / 1e6 * p_in + usage.get("output_tokens", 0) / 1e6 * p_out
    cost_acc = (
        cumulative.get("input_tokens", 0) / 1e6 * p_in
        + cumulative.get("output_tokens", 0) / 1e6 * p_out
    )
    return f"≈${cost:.4f} / 累计 ${cost_acc:.4f}"


# CLI 工具名的静态快照（仅作 _build_main_agent 不可用时的展示兜底；
# 真实注册以 _build_main_agent 的返回为准）。
_CLI_TOOL_NAMES = [
    "load_dataset", "profile_data", "inspect_streams", "check_temporal_sync",
    "check_sensor_sanity", "check_dataset_quality", "compute_stats",
    "plot_chart", "generate_report",
    "propose_stream_semantics", "unpack_mcap", "align_container_streams",
    "inspect_video_frame", "compare_datasets", "compare_table_columns",
    "list_tables",
    "segment_actions",
    "annotate_task", "save_annotations", "check_annotation_qc",
]


def _build_main_agent() -> tuple[Agent[RunContext], list[str]]:
    """构建 CLI 的主 Agent。

    Returns:
        (agent, 已注册工具名列表)。返回工具名是为了让 CLI 横幅**从实际注册
        结果生成**——此前横幅是硬编码字符串，与真实注册不同步（`unpack_mcap`
        长期缺失却无人察觉的根因）。
    """
    settings = get_settings()
    model = build_model(settings)
    tools = [
        load_dataset,
        profile_data,
        inspect_streams,
        check_temporal_sync,
        check_sensor_sanity,
        compute_stats,
        plot_chart,
        generate_report,
        propose_stream_semantics,
        # unpack_mcap 此前遗漏（CLI 仅 12 个工具、UI 13 个），导致 CLI 无法
        # 解包容器（行为不一致）。本次补齐——与 chat_service._ALL_TOOLS 对齐。
        unpack_mcap,
        align_container_streams,
        inspect_video_frame,
        compare_datasets,
        compare_table_columns,
        list_tables,
        check_dataset_quality,
        segment_actions,
        annotate_task,
        save_annotations,
        check_annotation_qc,
    ]
    # 与 chat_service 一致：套上工具返回体积护栏（第 2 层防御）。
    # CLI 自行组装工具（不经 chat_service），故此处必须补上，否则 CLI 路径
    # 既无返回护栏也无历史压缩。
    budget = derive_budget(
        settings.default_model,
        configured_window=settings.context_window_tokens,
        budget_ratio=settings.context_budget_ratio,
        history_ratio=settings.history_budget_ratio,
        tool_output_ratio=settings.tool_output_budget_ratio,
    )
    configure_history_compaction(
        budget_tokens=(
            budget.history_budget if settings.history_compaction_enabled else 0
        ),
        keep_recent_turns=settings.history_keep_recent_turns,
    )
    agent = build_agent(
        model, guard_tools(tools, budget_tokens=budget.tool_output_budget)
    )
    return agent, [t.name for t in tools]


async def chat_loop() -> None:
    # _build_main_agent 返回 (agent, 工具名列表)。工具名用 getattr 兜底取
    # （测试替身可能只给 agent），拿不到时退回静态清单，仅供横幅展示。
    built = _build_main_agent()
    agent, tool_names = built
    context = RunContext()
    history_input: list[Any] | None = None
    # 会话累计 token（CLI 进程内维护，不持久化）。
    cumulative: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "rounds": 0}

    print("=" * 56)
    print("🩻 Embodied-data-Xray — 具身智能数据结构透视")
    # 工具清单从实际注册结果生成（此前是硬编码字符串，与真实注册不同步——
    # 正是 unpack_mcap 长期缺失却无人发现的根因）。测试替身无此信息时退回
    # 静态清单（_CLI_TOOL_NAMES 有专门测试守护其与真实注册一致）。
    shown = tool_names or _CLI_TOOL_NAMES
    print(f"已加载工具({len(shown)}): " + ", ".join(shown))
    print("输入 exit / quit / 退出 结束对话；/compact 压缩历史，/history 查看历史体积。")
    print("=" * 56)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in _EXIT_COMMANDS or user_input in _EXIT_COMMANDS:
            print("再见！")
            break

        # 手动压缩历史：释放上下文并展示统计（对应"二者皆做"的手动入口）。
        if user_input.lower() in _COMPACT_COMMANDS or user_input in _COMPACT_COMMANDS:
            if not history_input:
                print("\n历史为空，无需压缩。")
                continue
            before = estimate_history_tokens(history_input)
            keep = get_settings().history_keep_recent_turns
            history_input, stats = compact_history(history_input, keep_recent_turns=keep)
            print(
                f"\n[压缩] 已压缩 {stats['compacted_outputs']} 条旧工具返回"
                f"（保留最近 {stats['kept_turns']} 轮完整）："
                f"{stats['before_tokens']:,} → {stats['after_tokens']:,} token"
                f"（省约 {stats['saved_tokens']:,}）"
            )
            print(f"[压缩] 用户提问与助手结论均完整保留；需明细请重新调用对应工具。")
            _ = before
            continue

        # 查看当前历史体积。
        if user_input.lower() in _HISTORY_COMMANDS or user_input in _HISTORY_COMMANDS:
            n = len([i for i in (history_input or []) if isinstance(i, dict)
                     and i.get("role") == "user"])
            print(
                f"\n[历史] 约 {n} 轮，估算 {estimate_history_tokens(history_input or []):,} token"
            )
            continue

        final, history_input, result = await run_turn(
            agent, context, user_input, history_input
        )

        # 展示工具调用过程，便于观察 Agent 自主调用。
        activity = format_tool_activity(result)
        if activity:
            print(f"\n[工具] {activity}")

        # Token 统计：本轮用量 + 会话累计（usage 为 None 时显示"未获取到用量"）。
        usage = extract_usage(result)
        if usage:
            cumulative["input_tokens"] += usage.get("input_tokens", 0)
            cumulative["output_tokens"] += usage.get("output_tokens", 0)
            cumulative["total_tokens"] += usage.get("total_tokens", 0)
        cumulative["rounds"] += 1
        print(f"\n{_format_tokens(usage, cumulative)}")

        print(f"\n助手: {final}")


def main() -> int:
    try:
        asyncio.run(chat_loop())
    except Exception as exc:  # noqa: BLE001 - 顶层兜底
        print(f"\n运行出错: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
