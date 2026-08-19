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
from app.tools import inspect_streams, load_dataset, profile_data

_EXIT_COMMANDS = {"exit", "quit", "q", "退出", "再见"}


def _build_main_agent():
    settings = get_settings()
    model = build_model(settings)
    tools = [load_dataset, profile_data, inspect_streams]
    return build_agent(model, tools)


async def chat_loop() -> None:
    agent = _build_main_agent()
    context = RunContext()
    history_input: list[Any] | None = None

    print("=" * 56)
    print("具身智能数据分析 Agent")
    print("已加载工具: load_dataset, profile_data, inspect_streams")
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
