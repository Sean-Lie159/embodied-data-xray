"""模型 API 异常兜底（502/限流/超时/鉴权等）的单元测试。

背景（用户实测）：模型服务返回 502（CodeBuddy: upstream error），异常以裸
traceback 外抛、炸穿 Streamlit 页面并丢失界面状态。run_turn 此前只兜
MaxTurnsExceeded，模型侧故障全部外抛——违反项目「错误可恢复 + 诚实降级」纪律。

覆盖：错误分类话术 + run_turn 端到端兜底（不抛、历史保留、result 为 None）。
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.agent import _describe_model_error, run_turn
from app.agent.context import RunContext


# --- 1. 错误分类话术 --------------------------------------------------------


@pytest.mark.parametrize("raw,expect_key", [
    ("Error code: 502 - {'error': {'message': 'CodeBuddy: upstream error (HTTP 502)'}}",
     "502"),
    ("Error code: 503 - Service Unavailable", "503"),
    ("Error code: 504 - gateway timeout", "504"),
    ("Error code: 429 - rate limit exceeded", "429"),
    ("Error code: 401 - invalid api key", "401"),
    ("Error code: 404 - model_not_found", "404"),
    ("APIConnectionError: connection error", "网络"),
])
def test_error_messages_are_actionable(raw: str, expect_key: str) -> None:
    """各类模型故障都给出**可操作**的中文提示（含重试建议）。"""
    msg = _describe_model_error(Exception(raw))
    assert msg
    # 提示里应包含关键定位信息与"重试/检查"类建议。
    assert ("重试" in msg or "检查" in msg), msg
    if expect_key == "网络":
        assert "无法连接" in msg or "网络" in msg
    else:
        assert expect_key in msg


def test_context_length_error_gives_compaction_hint() -> None:
    """上下文超限 → 提示先去压缩历史（复用已有手动入口）。"""
    msg = _describe_model_error(
        Exception("This model's maximum context length is 65536 tokens")
    )
    assert "压缩历史" in msg


def test_unknown_error_degrades_honestly() -> None:
    """未知异常：如实说明类型与摘要，不含糊、不裸露 traceback。"""
    msg = _describe_model_error(ValueError("something odd happened"))
    assert "ValueError" in msg
    assert "something odd happened" in msg
    assert "重试" in msg
    assert "Traceback" not in msg


def test_error_message_truncates_long_text() -> None:
    """超长错误文本被截断（防上下文被灌爆）。"""
    msg = _describe_model_error(Exception("x" * 5000))
    assert len(msg) < 1000


# --- 2. run_turn 端到端兜底 -------------------------------------------------


def test_run_turn_swallows_model_error(monkeypatch) -> None:
    """模型抛 502 → run_turn **不抛**，返回友好提示且 result 为 None。"""
    import agents

    class _Boom:
        async def run(self, *a, **k):
            raise RuntimeError(
                "Error code: 502 - {'error': {'message': 'upstream error (HTTP 502)'}}"
            )

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)

    history = [{"role": "user", "content": "之前的问题"}]
    final, next_input, result = asyncio.run(
        run_turn(object(), RunContext(), "本轮问题", history)
    )
    assert result is None              # 契约：异常分支 result 为 None
    assert "502" in final              # 友好提示
    assert "重试" in final
    # 历史保留（用户可重发），且未吞掉此前历史。
    assert next_input[0]["content"] == "之前的问题"
    assert next_input[-1]["content"] == "本轮问题"


def test_run_turn_error_keeps_history_intact(monkeypatch) -> None:
    """异常时历史原样返回（不含半截轮次），用户重发不会污染上下文。"""
    import agents

    class _Boom:
        async def run(self, *a, **k):
            raise TimeoutError("request timeout")

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)

    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    _final, next_input, result = asyncio.run(
        run_turn(object(), RunContext(), "Q2", history)
    )
    assert result is None
    assert next_input[:2] == history   # 原历史一字不动
    assert next_input[-1] == {"role": "user", "content": "Q2"}


def test_run_turn_propagates_keyboard_interrupt(monkeypatch) -> None:
    """用户主动中断（KeyboardInterrupt）不被吞——正常传播。"""
    import agents

    class _Boom:
        async def run(self, *a, **k):
            raise KeyboardInterrupt

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_turn(object(), RunContext(), "x", None))


# --- 3. UI 层不崩（两条 reply 路径都不再外抛）-------------------------------


def test_chat_service_reply_does_not_raise_on_model_error(monkeypatch) -> None:
    """ChatService.reply 在模型故障时返回正常 ChatTurn（不抛异常）。"""
    from app.services.chat_service import ChatService

    svc = ChatService.__new__(ChatService)
    svc.agent = object()          # run_turn 会把它传给被 mock 的 Runner.run
    svc.context = RunContext()
    svc.history_input = None
    svc._pending_notes = []
    svc._history_budget = 0
    svc._keep_recent_turns = 3

    import agents

    class _Boom:
        async def run(self, *a, **k):
            raise RuntimeError("Error code: 502 - upstream error (HTTP 502)")

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)
    turn = svc.reply("你好")           # 不抛
    assert turn.reply and "502" in turn.reply
    assert turn.usage is None          # 失败轮无用量（不显示 0 冒充）
