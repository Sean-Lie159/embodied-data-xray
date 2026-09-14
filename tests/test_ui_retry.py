"""重试与错误恢复的单测（docs/交互增强重试与错误恢复设计.md 第 4 节）。

**回归关键**：重试不产生重复用户消息、agent 历史不重复、统计不翻倍。
"""

from __future__ import annotations

import inspect

from app.services.chat_service import ChatTurn


def _turn(reply: str, *, completed: bool = True, error_kind=None,
          usage=None) -> ChatTurn:
    return ChatTurn(
        reply=reply, tool_activity="",
        usage=usage,
        metrics={"duration_ms": 10, "n_model_calls": 1, "n_tool_calls": 0,
                 "completed": completed, "error_kind": error_kind},
        error=None if completed else "本轮未正常完成。",
    )


def _cumulative() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "rounds": 0, "duration_ms": 0, "n_model_calls": 0}


class _FakeService:
    """替身：记录 truncate 调用与 reply 结果，验证重试语义。"""

    def __init__(self, reply_turn: ChatTurn) -> None:
        self._reply_turn = reply_turn
        self.truncate_calls: list[int] = []
        self.reply_inputs: list[str] = []
        self.compact_calls = 0

    def truncate_history_to_turn(self, idx: int) -> dict:
        self.truncate_calls.append(idx)
        return {"before_turns": idx, "after_turns": idx}

    def reply(self, text: str) -> ChatTurn:
        self.reply_inputs.append(text)
        return self._reply_turn

    def compact_now(self) -> dict:
        self.compact_calls += 1
        return {}


def _patch_ui_run(monkeypatch, turn: ChatTurn) -> None:
    """让 streamlit_app._run_agent_turn 不真的跑 agent，直接返回给定 turn。

    签名需与现实现一致（body_placeholder + process_placeholder）；
    process 占位符为可选参数，故用默认值兼容旧调用。
    """
    import streamlit_app

    monkeypatch.setattr(
        streamlit_app, "_run_agent_turn",
        lambda service, prompt, body_ph, process_ph=None: turn,
    )


def test_is_failed_detection() -> None:
    """失败判定依据 metrics.completed。"""
    import streamlit_app

    assert streamlit_app._is_failed(_turn("x", completed=False)) is True
    assert streamlit_app._is_failed(_turn("x", completed=True)) is False
    assert streamlit_app._is_failed(ChatTurn(reply="x", tool_activity="")) is False


def test_error_kind_extraction() -> None:
    """error_kind 从 metrics 取出。"""
    import streamlit_app

    assert streamlit_app._error_kind(
        _turn("x", completed=False, error_kind="context_overflow")
    ) == "context_overflow"
    assert streamlit_app._error_kind(_turn("x")) is None


def test_retry_does_not_duplicate_user_message(monkeypatch) -> None:
    """**回归关键**：重试后 messages 中该用户消息只有一条。"""
    import streamlit_app

    svc = _FakeService(_turn("新回答"))
    _patch_ui_run(monkeypatch, _turn("新回答"))

    msgs = [
        {"role": "user", "content": "我的问题"},
        {"role": "assistant", "content": "失败提示", "turn": _turn("失败提示", completed=False)},
    ]
    cum = _cumulative()
    streamlit_app._retry_turn(svc, msgs, cum, 0)

    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert len(user_msgs) == 1, f"用户消息重复：{user_msgs}"
    assert user_msgs[0]["content"] == "我的问题"
    # 失败的 assistant 消息已被替换。
    assert msgs[-1]["content"] == "新回答"


def test_retry_truncates_history_before_turn(monkeypatch) -> None:
    """重试先把 agent 历史回退到本轮之前（防同输入在历史中重复）。"""
    import streamlit_app

    svc = _FakeService(_turn("新回答"))
    _patch_ui_run(monkeypatch, _turn("新回答"))
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "fail",
             "turn": _turn("fail", completed=False)}]
    streamlit_app._retry_turn(svc, msgs, _cumulative(), 0)

    assert svc.truncate_calls == [0], "未按预期回退历史"


def test_retry_records_stats_once(monkeypatch) -> None:
    """统计只累加一次（不因重试翻倍）。"""
    import streamlit_app

    usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    new_turn = _turn("新回答", usage=usage)
    svc = _FakeService(new_turn)
    _patch_ui_run(monkeypatch, new_turn)

    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "fail",
             "turn": _turn("fail", completed=False)}]
    cum = _cumulative()
    streamlit_app._retry_turn(svc, msgs, cum, 0)

    assert cum["input_tokens"] == 10
    assert cum["total_tokens"] == 15
    assert cum["rounds"] == 1, "轮数应只加一次"
    assert len(msgs) == 2  # user + 新 assistant


def test_retry_first_turn_idx_zero(monkeypatch) -> None:
    """首轮失败（idx=0）时历史清空且行为正确。"""
    import streamlit_app

    svc = _FakeService(_turn("好的"))
    _patch_ui_run(monkeypatch, _turn("好的"))
    msgs = [{"role": "user", "content": "首个问题"},
            {"role": "assistant", "content": "fail",
             "turn": _turn("fail", completed=False)}]
    streamlit_app._retry_turn(svc, msgs, _cumulative(), 0)
    assert svc.truncate_calls == [0]
    assert len(msgs) == 2
    assert msgs[0]["content"] == "首个问题"


def test_retry_wired_into_ui_last_turn_only() -> None:
    """静态：重试入口仅在最后一轮失败时渲染（其后无对话）。"""
    import streamlit_app

    src = inspect.getsource(streamlit_app._main)
    assert "_is_failed(msg[\"turn\"])" in src
    assert "i == len(messages) - 1" in src
    assert "_render_failure_actions" in src


def test_failure_actions_offer_compact_for_context_overflow() -> None:
    """上下文超限时提供"压缩历史并重试"入口。"""
    import streamlit_app

    src = inspect.getsource(streamlit_app._render_failure_actions)
    assert "context_overflow" in src
    assert "compact_now" in src
    assert "重试本轮" in src


def test_classify_model_error() -> None:
    """错误分类：上下文超限单独识别，其余归 api。"""
    from app.agent.agent import classify_model_error

    assert classify_model_error(
        RuntimeError("This model's maximum context length is 128000 tokens")
    ) == "context_overflow"
    assert classify_model_error(RuntimeError("502 Bad Gateway")) == "api"
