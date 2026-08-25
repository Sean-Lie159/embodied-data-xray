"""命令行多轮对话入口。

在终端与主 Agent 对话，观察其自主调用工具（load_dataset / profile_data 等）。
跨轮共享同一个 RunContext，因此已加载的数据集在后续提问中持续可用。

用法：
    python main.py

退出：输入 exit / quit / 退出 即可。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from app.agent.agent import build_agent, format_tool_activity, run_turn
from app.agent.context import RunContext
from app.config import get_settings
from app.llm import build_model
from app.services.chat_service import extract_usage
from app.tools import (
    check_sensor_sanity,
    check_temporal_sync,
    compute_stats,
    generate_report,
    inspect_streams,
    load_dataset,
    plot_chart,
    profile_data,
)

_EXIT_COMMANDS = {"exit", "quit", "q", "退出", "再见"}


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


def _build_main_agent():
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
    ]
    return build_agent(model, tools)


async def chat_loop() -> None:
    agent = _build_main_agent()
    context = RunContext()
    history_input: list[Any] | None = None
    # 会话累计 token（CLI 进程内维护，不持久化）。
    cumulative: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "rounds": 0}

    print("=" * 56)
    print("具身智能数据分析 Agent")
    print("已加载工具: load_dataset, profile_data, inspect_streams, check_temporal_sync, check_sensor_sanity, compute_stats, plot_chart, generate_report")
    print("输入 exit / quit / 退出 结束对话。")
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
