"""消息编辑重发功能的单元测试。

语义核心（用户需求：像网页 ChatBot 一样编辑已发消息并让 bot 重新回答）：
编辑第 n 条用户消息后，其**之后**的一切轮次必须从 agent 历史中移除——
否则 bot 会"记得"已被编辑掉的旧内容，重新回答不干净。
"""

from __future__ import annotations

from app.services.chat_service import ChatService, _split_turns


def _svc_with_turns(n: int) -> ChatService:
    """构造带 n 轮假历史的服务实例（不连真实模型）。"""
    svc = ChatService.__new__(ChatService)
    svc.context = None  # truncate 不触碰 context
    history: list[dict] = []
    for i in range(n):
        history.append({"role": "user", "content": f"问题{i}"})
        history.append({"type": "function_call_output", "call_id": f"c{i}",
                        "output": f"结果{i}"})
        history.append({"role": "assistant", "content": f"回答{i}"})
    svc.history_input = history
    return svc


def test_truncate_to_turn_zero_keeps_nothing() -> None:
    """截断到第 0 轮之前 → 历史清空（首条消息编辑场景）。"""
    svc = _svc_with_turns(3)
    stats = svc.truncate_history_to_turn(0)
    assert svc.history_input == []
    assert stats["before_turns"] == 3
    assert stats["after_turns"] == 0


def test_truncate_mid_keeps_prior_turns_only() -> None:
    """编辑第 2 条（0-based 1）→ 保留第 0 轮，丢弃其后一切。"""
    svc = _svc_with_turns(3)
    svc.truncate_history_to_turn(1)
    turns = _split_turns(svc.history_input)
    assert len(turns) == 1
    assert turns[0][0]["content"] == "问题0"
    # 旧的第 1/2 轮内容（含工具返回）不得残留。
    text = str(svc.history_input)
    assert "问题1" not in text and "回答1" not in text
    assert "问题2" not in text and "回答2" not in text


def test_truncate_out_of_range_is_noop() -> None:
    """越界下标（防御）→ 不变。"""
    svc = _svc_with_turns(2)
    stats = svc.truncate_history_to_turn(99)
    assert stats["before_turns"] == stats["after_turns"] == 2


def test_truncate_empty_history_safe() -> None:
    """空历史不崩。"""
    svc = _svc_with_turns(0)
    stats = svc.truncate_history_to_turn(0)
    assert stats["after_turns"] == 0


def test_edited_tool_outputs_gone_from_history() -> None:
    """被截断轮次里的工具返回（含大 JSON）不残留——重新回答干净的前提。"""
    svc = _svc_with_turns(3)
    svc.truncate_history_to_turn(2)
    text = str(svc.history_input)
    assert "结果0" in text and "结果1" in text
    assert "结果2" not in text


def test_streamlit_app_has_edit_ui() -> None:
    """streamlit_app 含编辑入口与保存/取消控件（静态断言）。"""
    import inspect

    import streamlit_app

    src = inspect.getsource(streamlit_app)
    for needle in ("edit_btn_", "edit_area_", "edit_save_",
                   "truncate_history_to_turn"):
        assert needle in src, f"缺少编辑 UI 要素：{needle}"
