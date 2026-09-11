"""单轮运行观测指标（耗时 / 模型往返次数）与"批量纪律"提示词的测试。

背景（用户实测 502）：检查 24 条流时工具循环串行十余次模型往返，整轮耗时
触及上游超时阈值返回 502。当时**没有任何事实可查**——不知道跑了几轮、多久，
只能靠猜。本次改动做两件事：
  1. 架构侧收敛：SYSTEM_PROMPT 明确"批量纪律"（一次调用覆盖 N 条流的绝不
     拆成 N 次），工具描述同步声明批量语义；
  2. run_turn 记录轮数与耗时并经 ChatTurn 透出到 UI（**失败轮也记录**——
     失败轮的指标才是排查 502 最需要的）。

覆盖：指标回填（成功/失败两分支）+ 提示词纪律存在性 + UI 摘要格式。
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.agent import (
    SYSTEM_PROMPT,
    RunMetrics,
    _count_tool_calls,
    run_turn,
)
from app.agent.context import RunContext


# --- 1. RunMetrics 回填（成功分支）------------------------------------------


def test_metrics_filled_on_success(monkeypatch) -> None:
    """正常完成：耗时 > 0、往返次数 >= 1、completed 为 True。"""
    import agents

    class _OkResult:
        final_output = "答完了"
        tool_input_items: list = []

        def to_input_list(self, mode: str = "normalized"):  # noqa: ANN001
            return [{"role": "assistant", "content": "答完了"}]

        def to_state(self):  # noqa: ANN001
            raise RuntimeError("no usage")  # usage 取不到也必须安全降级

    class _Ok:
        async def run(self, *a, **k):
            return _OkResult()

    monkeypatch.setattr(agents.Runner, "run", _Ok().run)

    m = RunMetrics()
    final, _next, result = asyncio.run(
        run_turn(object(), RunContext(), "问题", None, metrics=m)
    )
    assert final == "答完了"
    assert result is not None
    assert m.completed is True
    assert m.duration_ms >= 0        # 计时已回填（不要求 > 0，避免快机器取整为 0）
    assert m.n_model_calls >= 1      # 至少一次模型往返


def test_metrics_filled_on_model_error(monkeypatch) -> None:
    """**关键**：502 等异常分支同样回填指标（失败轮才最需要看数据）。"""
    import agents

    class _Boom:
        async def run(self, *a, **k):
            raise RuntimeError("Error code: 502 - upstream error (HTTP 502)")

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)

    m = RunMetrics()
    final, _next, result = asyncio.run(
        run_turn(object(), RunContext(), "问题", None, metrics=m)
    )
    assert result is None
    assert "502" in final
    assert m.completed is False      # 明确标注未正常完成
    assert m.duration_ms >= 0        # 失败也有耗时事实


def test_metrics_filled_on_max_turns(monkeypatch) -> None:
    """撞 max_turns：completed 为 False，且往返次数回填为已知的 max_turns。"""
    import agents
    from agents.exceptions import MaxTurnsExceeded

    class _Boom:
        async def run(self, *a, **k):
            raise MaxTurnsExceeded("too many turns")

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)

    m = RunMetrics()
    final, _next, result = asyncio.run(
        run_turn(object(), RunContext(), "问题", None, 7, metrics=m)
    )
    assert result is None
    assert "max_turns=7" in final
    assert m.completed is False
    assert m.n_model_calls == 7


def test_metrics_optional_zero_regression(monkeypatch) -> None:
    """不传 metrics：行为与改动前完全一致（不抛、正常返回三元组）。"""
    import agents

    class _Boom:
        async def run(self, *a, **k):
            raise RuntimeError("Error code: 502 - upstream error")

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)
    final, next_input, result = asyncio.run(
        run_turn(object(), RunContext(), "本轮问题", [{"role": "user", "content": "旧"}])
    )
    assert result is None
    assert next_input[-1]["content"] == "本轮问题"


def test_keyboard_interrupt_still_propagates(monkeypatch) -> None:
    """用户主动中断仍正常传播（指标改造不得吞掉中断）。"""
    import agents

    class _Boom:
        async def run(self, *a, **k):
            raise KeyboardInterrupt

    monkeypatch.setattr(agents.Runner, "run", _Boom().run)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_turn(object(), RunContext(), "x", None, metrics=RunMetrics()))


# --- 2. 工具调用计数容错 ----------------------------------------------------


def test_count_tool_calls_handles_none() -> None:
    """result 为 None（异常分支）→ 计数为 0，不抛异常。"""
    assert _count_tool_calls(None) == 0


def test_count_tool_calls_handles_broken_structure() -> None:
    """结构不符（to_state 抛错）→ 安全降级为 0，不连锁炸掉主流程。"""

    class _Broken:
        tool_input_items: list = []

        def to_state(self):  # noqa: ANN001
            raise RuntimeError("bad structure")

    assert _count_tool_calls(_Broken()) == 0  # type: ignore[arg-type]


# --- 3. 架构侧收敛：提示词与工具描述的批量纪律 ------------------------------


def test_system_prompt_has_batch_discipline() -> None:
    """SYSTEM_PROMPT 含批量纪律：一次调用覆盖 N 条流，禁止逐流循环。"""
    assert "批量纪律" in SYSTEM_PROMPT
    assert "绝不拆成 N 次" in SYSTEM_PROMPT
    # 必须点名 streams 参数这一具体落点（否则模型不知道怎么做）。
    assert "streams" in SYSTEM_PROMPT


def test_system_prompt_has_time_discipline() -> None:
    """SYSTEM_PROMPT 含耗时纪律：先粗后细，不重复调用已通过的流。"""
    assert "耗时纪律" in SYSTEM_PROMPT
    assert "先粗后细" in SYSTEM_PROMPT


def test_temporal_sync_tool_declares_batch_semantics() -> None:
    """check_temporal_sync 的工具描述声明批量语义（省略 streams 即全部流）。"""
    from app.tools.check_temporal_sync import check_temporal_sync

    desc = check_temporal_sync.description or ""
    assert "批量语义" in desc
    assert "不要对每条流各调用一次" in desc


def test_sensor_sanity_tool_declares_batch_semantics() -> None:
    """check_sensor_sanity 的工具描述声明批量语义。"""
    from app.tools.check_sensor_sanity import check_sensor_sanity

    desc = check_sensor_sanity.description or ""
    assert "批量语义" in desc


def test_align_container_tool_declares_batch_semantics() -> None:
    """align_container_streams 的工具描述声明"只调用一次"的全量口径。"""
    from app.tools.align_container import align_container_streams

    desc = align_container_streams.description or ""
    assert "批量语义" in desc
    assert "只调用一次" in desc


# --- 4. UI 摘要格式（纯函数，不需要 Streamlit 运行时）----------------------


def test_format_duration_units() -> None:
    """耗时格式化：<1s 用毫秒，>=1s 用秒；非法值降级为"未知"。"""
    from app.ui.components import _format_duration

    assert _format_duration(350) == "350 ms"
    assert _format_duration(1500) == "1.5 s"
    assert _format_duration(None) == "未知"
    assert _format_duration(0) == "未知"


def test_loop_summary_reports_round_trips() -> None:
    """摘要行展示模型往返次数（判断"是否逐流循环"的关键量）。"""
    from app.ui.components import _loop_summary

    assert "模型往返 3 次" in _loop_summary({"n_model_calls": 3, "n_tool_calls": 2})
    # 未正常完成时明确标注（502 轮一眼可见）。
    assert "未正常完成" in _loop_summary({"n_model_calls": 5, "completed": False})
    assert _loop_summary({}) == "过程未知"


def test_speed_text_divides_by_round_trips() -> None:
    """速度摘要给出平均每次往返耗时（区分"往返多"与"单次慢"）。"""
    from app.ui.components import _speed_text

    text = _speed_text({"duration_ms": 30_000, "n_model_calls": 10})
    assert "30.0 s" in text
    assert "10 次" in text
    assert "3.0 s/次" in text
    assert _speed_text({}) == "速度：暂无数据"


# --- 5. ChatService 透出（不连真实模型）-------------------------------------


def test_chat_service_turn_exposes_metrics(monkeypatch) -> None:
    """ChatTurn.metrics 含四个字段，且失败轮同样透出（completed=False）。"""
    from app.services.chat_service import ChatService

    svc = ChatService.__new__(ChatService)
    svc.agent = object()
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
    turn = svc.reply("你好")
    assert turn.metrics is not None
    assert set(turn.metrics) == {
        "duration_ms", "n_model_calls", "n_tool_calls", "completed",
        "error_kind"}  # error_kind 为 2026-09-11 新增（供 UI 精确重试入口）
    assert turn.metrics["completed"] is False
    assert turn.metrics["error_kind"] == "api"  # 502 归类为 api
    assert turn.usage is None
