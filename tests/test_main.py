"""main.py 命令行入口的健壮性测试。

聚焦 MaxTurnsExceeded 路径：run_turn 返回 result=None 时，chat_loop 应打印
友好提示而非抛出 AttributeError（不依赖真实模型与网络）。
"""

from __future__ import annotations

import asyncio
import builtins
from types import SimpleNamespace

import main as main_module


def _run_chat_loop_with_inputs(monkeypatch, capsys, inputs, fake_run_turn):
    """以给定输入序列驱动 chat_loop，返回捕获的 stdout。"""
    it = iter(inputs)

    def fake_input(prompt=""):
        return next(it)

    # input 是内置函数，需 patch builtins.input（chat_loop 通过全局名解析）。
    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)

    # 避免真实构建模型/访问网络，替换为假 agent。
    monkeypatch.setattr(
        main_module,
        "_build_main_agent",
        lambda: SimpleNamespace(name="fake_agent"),
    )

    asyncio.run(main_module.chat_loop())
    return capsys.readouterr().out


def test_chat_loop_handles_max_turns_exceeded(monkeypatch, capsys) -> None:
    """MaxTurnsExceeded 时 run_turn 返回 result=None，chat_loop 应打印友好提示不崩溃。"""
    async def fake_run_turn(agent, context, user_input, history_input=None, max_turns=15):
        return (
            "本轮工具调用次数已达上限（max_turns=15），为避免死循环已停止。请尝试更明确地描述需求，或分步提问。",
            [{"role": "user", "content": user_input}],
            None,  # MaxTurnsExceeded 分支：result 为 None
        )

    out = _run_chat_loop_with_inputs(
        monkeypatch, capsys, ["分析一下", "exit"], fake_run_turn
    )

    # 未抛异常且打印了友好提示；工具轨迹为空故不打印 [工具] 行。
    assert "已达上限" in out
    assert "[工具]" not in out
    assert "再见" in out


def test_chat_loop_prints_activity_when_result_present(monkeypatch, capsys) -> None:
    """正常路径：result 非 None 时打印工具轨迹与回复。"""
    async def fake_run_turn(agent, context, user_input, history_input=None, max_turns=15):
        fake_result = SimpleNamespace(
            new_items=[SimpleNamespace(type="tool_call_item", name="load_dataset")],
        )
        return ("已加载", [{"role": "user", "content": user_input}], fake_result)

    out = _run_chat_loop_with_inputs(
        monkeypatch, capsys, ["加载", "exit"], fake_run_turn
    )

    assert "[工具]" in out
    assert "已加载" in out
