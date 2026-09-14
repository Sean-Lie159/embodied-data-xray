"""思考过程展示与流式过程持久化的单测（docs/思考过程展示与流式过程持久化设计.md 第 7 节）。

覆盖：reasoning 增量采集与分段、steps 时间线顺序与合并、无 reasoning 降级、
中途失败保留过程、旧数据兼容、UI 双占位符（过程不被正文覆盖）、
过程区渲染降级。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from app.agent.agent import RunMetrics, StreamStep, TurnEvent, _append_step


# ---- 事件替身（复用 test_stream_turn 的思路，但覆盖 reasoning）-------------


class _FakeRawDelta:
    def __init__(self, delta: str, dtype: str) -> None:
        self.delta = delta
        self.type = dtype


class _FakeRawEvent:
    type = "raw_response_event"

    def __init__(self, delta: str, dtype: str) -> None:
        self.data = _FakeRawDelta(delta, dtype)


class _FakeToolItem:
    type = "tool_call_item"

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeItemEvent:
    type = "run_item_stream_event"

    def __init__(self, item: Any) -> None:
        self.item = item


class _FakeResult:
    def __init__(self, events: list[Any], final: str = "", *, raise_after=None):
        self._events = events
        self.final_output = final
        self._raise_after = raise_after
        self.new_items: list[Any] = []

    async def stream_events(self):
        for i, ev in enumerate(self._events):
            if self._raise_after is not None and i >= self._raise_after:
                raise self._raise_after
            yield ev

    def to_input_list(self, mode: str = "normalized"):
        return [{"role": "user", "content": "x"}]


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


def _ctx():
    from app.agent.context import RunContext

    return RunContext(dataset_id="ds")


RT = "response.reasoning_summary_text.delta"
RP = "response.reasoning_summary_part.added"
OT = "response.output_text.delta"


# ---- 采集：reasoning 事件 ---------------------------------------------------


def test_reasoning_deltas_are_captured(monkeypatch) -> None:
    """思考摘要增量被识别为 reasoning 事件，且 done.reasoning 为全文。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult(
        [_FakeRawEvent("先看问题", RT), _FakeRawEvent("再作答", RT),
         _FakeRawEvent("答案", OT)],
        final="答案",
    )
    _patch_runner(monkeypatch, fake)
    events = _collect(stream_turn(None, _ctx(), "hi", None, metrics=RunMetrics()))

    reasons = [e for e in events if e.kind == "reasoning"]
    assert "".join(e.text for e in reasons) == "先看问题再作答"
    done = [e for e in events if e.kind == "done"][0]
    assert done.reasoning == "先看问题再作答"


def test_reasoning_part_boundary_inserts_blank_line(monkeypatch) -> None:
    """新分段（part.added）插入空行，使长摘要分段可读。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult(
        [_FakeRawEvent("第一段", RT),
         _FakeRawEvent("", RP),        # 分段边界
         _FakeRawEvent("第二段", RT),
         _FakeRawEvent("答案", OT)],
        final="答案",
    )
    _patch_runner(monkeypatch, fake)
    events = _collect(stream_turn(None, _ctx(), "hi", None, metrics=RunMetrics()))
    done = [e for e in events if e.kind == "done"][0]
    assert "第一段\n\n第二段" in done.reasoning


def test_steps_order_interleaves_reasoning_and_tools(monkeypatch) -> None:
    """steps 按发生顺序交错记录思考与工具，且相邻同类合并。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult(
        [_FakeRawEvent("思考A", RT), _FakeRawEvent("续A", RT),
         _FakeItemEvent(_FakeToolItem("check_temporal_sync")),
         _FakeRawEvent("思考B", RT),
         _FakeRawEvent("正文", OT)],
        final="正文",
    )
    _patch_runner(monkeypatch, fake)
    events = _collect(stream_turn(None, _ctx(), "hi", None, metrics=RunMetrics()))
    steps = [e for e in events if e.kind == "done"][0].steps
    kinds = [s.kind for s in steps]
    # 相邻同类合并：两段 reasoning 合并为一条。
    assert kinds == ["reasoning", "tool", "reasoning"]
    assert steps[0].text == "思考A续A"
    assert steps[1].tool_name == "check_temporal_sync"
    assert steps[2].text == "思考B"


def test_no_reasoning_degrades_gracefully(monkeypatch) -> None:
    """**降级**：无 reasoning 事件时 reasoning 为空、steps 不含 reasoning 项。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult([_FakeRawEvent("正文", OT)], final="正文")
    _patch_runner(monkeypatch, fake)
    events = _collect(stream_turn(None, _ctx(), "hi", None, metrics=RunMetrics()))
    done = [e for e in events if e.kind == "done"][0]
    assert done.reasoning == ""
    assert all(s.kind != "reasoning" for s in (done.steps or []))


def test_mid_stream_error_preserves_steps(monkeypatch) -> None:
    """**回归关键**：中途失败时已收到的思考与工具过程**保留**在 steps 中。"""
    from app.agent.agent import stream_turn

    fake = _FakeResult(
        [_FakeRawEvent("已思考", RT),
         _FakeItemEvent(_FakeToolItem("plot_chart")),
         _FakeRawEvent("后续", RT)],
        raise_after=2,  # 处理完前 2 个事件后抛错
    )
    _patch_runner(monkeypatch, fake)
    events = _collect(stream_turn(None, _ctx(), "hi", None, metrics=RunMetrics()))
    done = [e for e in events if e.kind == "done"][0]
    assert done.error  # 有失败提示
    assert "已思考" in done.reasoning
    kinds = [s.kind for s in (done.steps or [])]
    assert "reasoning" in kinds and "tool" in kinds


def test_append_step_merges_adjacent_same_kind() -> None:
    """_append_step 合并相邻同类、跳过空文本。"""
    steps: list[StreamStep] = []
    _append_step(steps, "reasoning", "a")
    _append_step(steps, "reasoning", "b")
    _append_step(steps, "tool", "t")
    _append_step(steps, "tool", "t2")
    _append_step(steps, "reasoning", "")
    assert len(steps) == 2
    assert steps[0].text == "ab"
    assert steps[1].text == "tt2"


# ---- 服务层：ChatTurn 透传与兼容 --------------------------------------------


def test_reply_stream_final_turn_has_steps(monkeypatch) -> None:
    """reply_stream 的 final 块 turn 携带 steps/reasoning。"""
    from app.services.chat_service import ChatService

    svc = ChatService.__new__(ChatService)
    svc.context = _ctx()
    svc.history_input = None
    svc._pending_notes = []
    svc._history_budget = 0
    svc._keep_recent_turns = 3
    svc.agent = None

    fake = _FakeResult([_FakeRawEvent("思考", RT), _FakeRawEvent("正文", OT)],
                       final="正文")
    _patch_runner(monkeypatch, fake)
    chunks = list(svc.reply_stream("hi"))
    turn = chunks[-1].turn
    assert turn.reasoning == "思考"
    assert [s.kind for s in turn.steps] == ["reasoning"]
    # final 块也携带 steps 快照。
    assert chunks[-1].steps


def test_chat_turn_defaults_empty_steps() -> None:
    """旧结构（未传 steps）不报错——向后兼容。"""
    from app.services.chat_service import ChatTurn

    t = ChatTurn(reply="x", tool_activity="")
    assert t.steps == []
    assert t.reasoning == ""


# ---- UI：双占位符（过程不被正文覆盖）----------------------------------------


def test_run_agent_turn_uses_separate_placeholders() -> None:
    """**静态断言**：过程区与正文区使用**不同**占位符（防回退到单一会被覆写的格子）。"""
    import streamlit_app

    src = inspect.getsource(streamlit_app._run_agent_turn)
    assert "body_placeholder" in src
    assert "process_placeholder" in src
    # 过程不应写入正文占位符。
    assert "body_placeholder.markdown(acc" in src


def test_main_creates_two_placeholders() -> None:
    """静态：主输入分支为过程与正文各建一个占位符。"""
    import streamlit_app

    src = inspect.getsource(streamlit_app._main)
    assert "process_ph = st.empty()" in src
    assert "body_ph = st.empty()" in src


def test_history_renders_process_area() -> None:
    """静态：历史轮次的 assistant 消息也渲染过程区（默认收起）。"""
    import streamlit_app

    src = inspect.getsource(streamlit_app._main)
    assert "render_reasoning_and_steps" in src
    assert 'getattr(msg["turn"], "steps", []) or []' in src
