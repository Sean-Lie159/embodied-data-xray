"""流式渲染节流测试（**真实事故回归**，2026-09-22）。

事故：单轮提问含 4 项分析 → 思考链极长 → 后台产出 **8501 个事件**，
而 UI 对**每个** reasoning 增量都执行"新建容器 + 全量重渲染"，
形成平方级开销 → 脚本跑 **196 秒**后被 Streamlit 执行控制中断
（``StopException``）→ 整轮内容丢失。

修复：
1. **节流**（时间 + 字符双阈值）——把渲染次数与思考速度解耦；
2. **渲染上限**——过程区只渲染尾部若干条；
3. **中断恢复**——异常时仍写入 messages，内容不丢、可重试。

详见 ``docs/流式渲染节流与中断恢复设计.md``。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.ui.constants import (
    PROCESS_RENDER_INTERVAL_S,
    PROCESS_RENDER_MAX_BLOCKS,
    PROCESS_RENDER_MIN_CHARS,
)


# ---------------------------------------------------------------------------
# 节流策略：模拟 _run_agent_turn 的节流逻辑（与实现同构）
# ---------------------------------------------------------------------------


class _Throttle:
    """复刻 streamlit_app._run_agent_turn 的节流逻辑（供纯逻辑测试）。

    为什么复刻而不直接调用：真实实现在 UI 层、依赖 streamlit 运行时；
    这里按同一策略实现，用于验证**策略性质**（渲染次数与事件数解耦）。
    若实现改动导致策略变化，本测试的断言会失败，提示同步更新。
    """

    def __init__(self) -> None:
        self.dirty = False
        self.pending_chars = 0
        self.last_render = 0.0
        self.n_renders = 0
        self.now = 0.0

    def on_delta(self, text: str, *, dt: float = 0.0) -> None:
        """收到一个过程增量（dt 为距上次调用的时间推进）。"""
        self.now += dt
        self.dirty = True
        self.pending_chars += len(text)
        self.flush()

    def flush(self, *, force: bool = False) -> None:
        if not self.dirty:
            return
        if not force:
            elapsed = self.now - self.last_render
            if (elapsed < PROCESS_RENDER_INTERVAL_S
                    and self.pending_chars < PROCESS_RENDER_MIN_CHARS):
                return
        self.n_renders += 1
        self.dirty = False
        self.last_render = self.now
        self.pending_chars = 0


def test_throttle_decouples_render_count_from_event_count() -> None:
    """**核心性质**：渲染次数与事件数解耦（不是一对一）。

    事故中 8501 个事件 → 8501 次渲染。节流后同样的事件数应产生
    远少于事件数的渲染次数。
    """
    t = _Throttle()
    n_events = 8501
    # 模拟逐 token 到达：每个 token 约 2 字符，间隔约 5ms。
    for _ in range(n_events):
        t.on_delta("ab", dt=0.005)

    assert t.n_renders < n_events / 10, (
        f"节流未生效：{n_events} 个事件产生了 {t.n_renders} 次渲染"
    )
    # 8501 个事件 × 5ms ≈ 42.5 秒。渲染由两个阈值**任一**触发：
    #   时间阈值：42.5 / 0.4 ≈ 106 次；
    #   字符阈值：每事件 2 字符，60 事件（0.3s）即达 120 字符 → 更频繁。
    # 取字符阈值主导：42.5 / 0.3 ≈ 141 次。留 20% 余量断言上界。
    assert t.n_renders <= 180, f"渲染次数 {t.n_renders} 超出预期上限"
    # 关键对比：从 8501 次降到约 140 次，降幅 ~60 倍。
    assert t.n_renders < 200


def test_throttle_renders_on_char_threshold_before_time() -> None:
    """字符阈值可**先于**时间阈值触发（避免慢速思考时界面迟滞）。"""
    t = _Throttle()
    # 一次性灌入超过阈值的字符（时间未推进）。
    t.on_delta("x" * (PROCESS_RENDER_MIN_CHARS + 1), dt=0.0)
    assert t.n_renders == 1, "字符阈值未触发渲染"


def test_throttle_renders_on_time_threshold() -> None:
    """时间阈值触发（字符不足但已过间隔）。"""
    t = _Throttle()
    t.on_delta("x", dt=0.0)
    assert t.n_renders == 0, "首次不应立即渲染（字符与时间都不足）"
    t.on_delta("y", dt=PROCESS_RENDER_INTERVAL_S + 0.01)
    assert t.n_renders == 1, "时间阈值未触发渲染"


def test_throttle_skips_render_when_below_both_thresholds() -> None:
    """双阈值都不满足时不渲染（这正是省下大量开销的关键）。"""
    t = _Throttle()
    for _ in range(50):
        t.on_delta("a", dt=0.001)  # 50ms 内 50 字符
    # 时间 50ms < 400ms；字符 50 < 120 → 可能恰好都不满足，渲染极少。
    assert t.n_renders <= 1


def test_force_flush_always_renders() -> None:
    """收尾必须能强制渲染（保证最终内容完整，不被节流吞掉）。"""
    t = _Throttle()
    t.on_delta("x", dt=0.0)
    assert t.n_renders == 0
    t.flush(force=True)
    assert t.n_renders == 1, "强制渲染未生效——收尾内容会丢失"


def test_force_flush_noop_when_nothing_pending() -> None:
    """无待渲染内容时强制渲染不产生多余渲染（避免无效开销）。"""
    t = _Throttle()
    t.flush(force=True)
    assert t.n_renders == 0


# ---------------------------------------------------------------------------
# 渲染上限
# ---------------------------------------------------------------------------


def test_render_max_blocks_is_bounded() -> None:
    """渲染上限必须是有限正值（防超长文本反复进渲染管线）。"""
    assert PROCESS_RENDER_MAX_BLOCKS > 0
    assert PROCESS_RENDER_MAX_BLOCKS <= 200


def test_harmonize_then_truncate_keeps_tail() -> None:
    """尾部截断应保留**最后** N 条（用户关心"当前在做什么"）。"""
    from app.agent.agent import StreamStep
    from app.ui.components import _harmonize_steps

    steps = [
        StreamStep(kind="reasoning", text=f"step-{i}") for i in range(100)
    ]
    blocks = _harmonize_steps(steps)
    assert len(blocks) == 100

    kept = blocks[-PROCESS_RENDER_MAX_BLOCKS:]
    assert kept[-1] == "step-99", "未保留尾部（最新过程）"
    assert kept[0] == f"step-{100 - PROCESS_RENDER_MAX_BLOCKS}"


# ---------------------------------------------------------------------------
# 中断恢复
# ---------------------------------------------------------------------------


def test_build_interrupted_turn_recognizes_stop_exception() -> None:
    """``StopException`` 应被识别为"执行时间过长被中断"，并给出拆分建议。"""
    import streamlit_app as app_mod

    class StopException(Exception):  # 模拟同名异常
        pass

    turn = app_mod._build_interrupted_turn(StopException())
    assert turn.metrics["completed"] is False
    assert turn.metrics["error_kind"] == "interrupted"
    # 必须给出可操作建议（而非只说"失败了"）。
    assert "拆分" in turn.reply or "拆成几次提问" in turn.reply
    assert "过多" in turn.reply


def test_build_interrupted_turn_handles_generic_exception() -> None:
    """其它异常走通用失败轮（可重试）。"""
    import streamlit_app as app_mod

    turn = app_mod._build_interrupted_turn(ValueError("boom"))
    assert turn.metrics["completed"] is False
    assert turn.metrics["error_kind"] == "ui_exception"
    assert "重试" in turn.reply
    assert turn.error


def test_interrupted_turn_is_recordable() -> None:
    """失败轮必须可被 `_record_turn` 记录（这是"内容不丢"的关键）。"""
    import streamlit_app as app_mod

    turn = app_mod._build_interrupted_turn(ValueError("x"))
    cumulative: dict = {}
    messages: list[dict] = [{"role": "user", "content": "问题"}]
    app_mod._record_turn(cumulative, turn, messages)

    assert len(messages) == 2, "assistant 消息未被写入（内容仍会丢失）"
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == turn.reply


# ---------------------------------------------------------------------------
# 实现一致性（防节流被误删）
# ---------------------------------------------------------------------------


def test_run_agent_turn_uses_throttle_constants() -> None:
    """实现必须引用节流常量（防将来重构时把节流去掉）。"""
    src = (
        Path(__file__).resolve().parent.parent / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    assert "PROCESS_RENDER_INTERVAL_S" in src
    assert "PROCESS_RENDER_MIN_CHARS" in src
    # 必须存在强制渲染的收尾调用。
    assert "_flush_process(force=True)" in src


def test_ui_catches_exception_and_records_turn() -> None:
    """主循环必须捕获异常并转成可记录的失败轮（防"整轮静默丢失"回归）。"""
    src = (
        Path(__file__).resolve().parent.parent / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    assert "_build_interrupted_turn" in src
    # 异常分支必须仍然调用 _record_turn（在 try 之外，故必然执行）。
    assert "_record_turn(cumulative, turn, messages)" in src
