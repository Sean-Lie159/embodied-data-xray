"""流式输出的单测（docs/流式输出设计.md 第 5 节）。

覆盖：stream_turn 事件序列、中途异常保留已输出正文、异常分支 fallback 含本轮
用户消息、metrics 回填、reply_stream 的 final 块与 reply() 结构同构、
配置开关回退路径。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agent.agent import RunMetrics, TurnEvent, describe_tool_call


# ---- 替身：模拟 openai-agents 的流式结果 ------------------------------------


class _FakeRawDelta:
    def __init__(self, delta: str) -> None:
        self.delta = delta
        self.type = "response.output_text.delta"


class _FakeRawEvent:
    type = "raw_response_event"

    def __init__(self, delta: str) -> None:
        self.data = _FakeRawDelta(delta)


class _FakeToolItem:
    type = "tool_call_item"

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeItemEvent:
    type = "run_item_stream_event"

    def __init__(self, item: Any) -> None:
        self.item = item


class _FakeResult:
    """模拟 Runner.run_streamed 的返回值。"""

    def __init__(self, events: list[Any], final: str = "", *, raise_after=None):
        self._events = events
        self.final_output = final
        self._raise_after = raise_after
        self.new_items: list[Any] = []
        self.tool_input_items: list[Any] = []

    async def stream_events(self):
        for i, ev in enumerate(self._events):
            if self._raise_after is not None and i >= self._raise_after:
                raise self._raise_after
            yield ev

    def to_input_list(self, mode: str = "normalized"):
        return [{"role": "user", "content": "x"}, {"role": "assistant", "content": self.final_output}]


def _patch_runner(monkeypatch, fake: _FakeResult) -> None:
    import app.agent.agent as agent_mod

    class _Runner:
        @staticmethod
        def run_streamed(*a, **k):
            return fake

    monkeypatch.setattr(agent_mod, "Runner", _Runner)


def _collect(agen) -> list[TurnEvent]:
    async def _drain():
        return [ev async for ev in agen]

    return asyncio.run(_drain())


# ---- stream_turn 事件序列 --------------------------------------------------


def test_stream_yields_deltas_and_done(monkeypatch) -> None:
    """正常轮：多个 delta + 一个 done；done.final 等于完整正文。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult(
        [_FakeRawEvent("你好"), _FakeRawEvent("，世界")],
        final="你好，世界",
    )
    _patch_runner(monkeypatch, fake)
    ctx = _ctx()
    events = _collect(stream_turn(None, ctx, "hi", None, metrics=RunMetrics()))

    deltas = [e for e in events if e.kind == "delta"]
    dones = [e for e in events if e.kind == "done"]
    assert "".join(e.text for e in deltas) == "你好，世界"
    assert len(dones) == 1
    assert dones[0].final == "你好，世界"
    assert dones[0].error is None


def test_stream_emits_tool_broadcast(monkeypatch) -> None:
    """工具调用事件被转为 tool 播报（功能描述，不含参数）。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult(
        [_FakeItemEvent(_FakeToolItem("check_temporal_sync")),
         _FakeRawEvent("结论")],
        final="结论",
    )
    _patch_runner(monkeypatch, fake)
    events = _collect(stream_turn(None, _ctx(), "hi", None, metrics=RunMetrics()))

    tools = [e for e in events if e.kind == "tool"]
    assert len(tools) == 1
    assert tools[0].tool_name == "check_temporal_sync"
    assert "时间同步" in tools[0].text  # 功能描述而非工具名


def test_stream_preserves_partial_output_on_mid_stream_error(monkeypatch) -> None:
    """**回归关键**：中途异常时，已输出的正文保留，done.error 非空。"""
    from app.agent.agent import stream_turn

    # 第 0 个事件正常，第 1 个事件（索引 1）前抛错。
    fake = _FakeResult(
        [_FakeRawEvent("第一部分"), _FakeRawEvent("第二部分")],
        final="第一部分第二部分",
        raise_after=1,
    )
    _patch_runner(monkeypatch, fake)
    metrics = RunMetrics()
    events = _collect(stream_turn(None, _ctx(), "hi", None, metrics=metrics))

    deltas = [e for e in events if e.kind == "delta"]
    dones = [e for e in events if e.kind == "done"]
    assert "".join(e.text for e in deltas) == "第一部分"  # 已输出部分保留
    assert len(dones) == 1
    assert dones[0].error  # 有可操作提示
    assert dones[0].final == "第一部分"  # final 保留部分正文，不丢
    assert metrics.completed is False


def test_stream_error_fallback_contains_user_message(monkeypatch) -> None:
    """异常分支 next_input 必须含本轮用户消息（用户可直接重发）。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult([_FakeRawEvent("x")], raise_after=0)
    _patch_runner(monkeypatch, fake)
    events = _collect(stream_turn(None, _ctx(), "我的问题", None,
                                  metrics=RunMetrics()))
    done = [e for e in events if e.kind == "done"][0]
    assert done.next_input is not None
    assert any(
        m.get("content") == "我的问题" for m in done.next_input
        if isinstance(m, dict)
    )


def test_stream_metrics_backfilled_on_success(monkeypatch) -> None:
    """成功轮 metrics 回填 completed=True 且耗时非负。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult([_FakeRawEvent("ok")], final="ok")
    _patch_runner(monkeypatch, fake)
    m = RunMetrics()
    _collect(stream_turn(None, _ctx(), "hi", None, metrics=m))
    assert m.completed is True
    assert m.duration_ms >= 0


def test_describe_tool_call_known_and_unknown() -> None:
    """已知工具给功能描述（不含工具名）；未知工具降级为工具名（不静默丢弃）。"""
    sync_text = describe_tool_call("check_temporal_sync")
    assert "时间同步" in sync_text
    assert "check_temporal_sync" not in sync_text  # 不暴露工具名
    # 已知工具：功能描述，不含工具名本身。
    assert describe_tool_call("plot_chart") == "正在绘图"
    # 未知工具：降级为工具名（可见，不静默丢弃）。
    assert "mystery_tool" in describe_tool_call("mystery_tool")


# ---- reply_stream 与 reply 的结构一致性 -------------------------------------


class _FakeService:
    """最小替身：复用 ChatService.reply_stream 但不构造 agent。"""


def _ctx():
    from app.agent.context import RunContext

    return RunContext(dataset_id="ds")


def test_reply_stream_final_turn_shape(monkeypatch) -> None:
    """reply_stream 的 final 块 turn 与 reply() 同构（关键字段齐全）。"""
    from app.services.chat_service import ChatService, ChatTurn

    svc = ChatService.__new__(ChatService)
    svc.context = _ctx()
    svc.history_input = None
    svc._pending_notes = []
    svc._history_budget = 0
    svc._keep_recent_turns = 3
    svc.agent = None

    fake = _FakeResult([_FakeRawEvent("流式正文")], final="流式正文")
    _patch_runner(monkeypatch, fake)

    chunks = list(svc.reply_stream("hi"))
    assert chunks[-1].kind == "final"
    turn = chunks[-1].turn
    assert isinstance(turn, ChatTurn)
    assert turn.reply == "流式正文"
    assert turn.findings == []
    assert turn.metrics is not None
    assert "completed" in turn.metrics


def test_stream_output_config_default_and_override() -> None:
    """流式开关是配置项，默认开启，可关闭（回退路径）。"""
    from app.config.settings import Settings

    s = Settings(_env_file=None, openai_api_key="k", openai_base_url="http://x",
                 default_model="m")
    assert s.stream_output_enabled is True
    s2 = Settings(_env_file=None, openai_api_key="k", openai_base_url="http://x",
                  default_model="m", stream_output_enabled=False)
    assert s2.stream_output_enabled is False


def test_ui_has_non_stream_fallback_path() -> None:
    """UI 保留非流式回退路径（配置关闭时有用例覆盖）。"""
    import inspect

    import streamlit_app

    src = inspect.getsource(streamlit_app._run_agent_turn)
    assert "stream_output_enabled" in src
    assert "service.reply(" in src  # 回退分支仍在
    assert "reply_stream" in src
