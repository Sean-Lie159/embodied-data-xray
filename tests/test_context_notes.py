"""面板事件便签测试（2026-09-07 缺陷修复：侧栏加载后模型反向索要路径）。

缺陷复盘：侧栏加载成功只更新了 RunContext（工具层状态），而模型只读对话
输入流——模型凭历史判断"没有任何已加载数据集"，要求用户提供路径。工具
状态与模型认知脱节的根因是面板事件从未进入模型输入（UI messages 只是
渲染列表，history_input 才是模型输入流）。

修复：ChatService.add_context_note 投递便签，reply 时以"[侧栏面板事件]"
前缀拼入本轮输入。本文件守护：

- _compose_user_input：便签为空原样返回；有便签时前缀含加载事实与原问题；
- add_context_note：登记后由 reply 消费并清空（经 monkeypatch run_turn
  验证模型真正收到的输入）；
- _load_path 集成：面板加载成功即投递便签。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import app.config.settings as settings_mod
import app.services.chat_service as cs_mod
from app.services.chat_service import ChatService, _compose_user_input
from app.ui import data_loader_panel as dlp


# --- _compose_user_input（纯函数） --------------------------------------------


def test_compose_no_notes_passthrough() -> None:
    """无便签：输入原样返回（CLI 场景不受影响）。"""
    assert _compose_user_input("概况如何？", []) == "概况如何？"


def test_compose_single_note() -> None:
    """单便签：含面板事件前缀、数据集事实与原问题。"""
    out = _compose_user_input("概况如何？", ["用户刚通过面板加载数据集 sample_dataset"])
    assert out.startswith("[侧栏面板事件]")
    assert "sample_dataset" in out
    assert "无需再向用户索要数据集路径" in out
    assert out.endswith("用户问题：概况如何？")


def test_compose_multiple_notes_ordered() -> None:
    """多便签（连续加载两次）：按登记顺序全部列出。"""
    out = _compose_user_input("q", ["便签A", "便签B"])
    assert "- 便签A" in out and "- 便签B" in out
    assert out.index("便签A") < out.index("便签B")


# --- ChatService 状态管理（monkeypatch run_turn 验证真实输入） ------------------


@pytest.fixture()
def service() -> ChatService:
    settings_mod.get_settings.cache_clear()
    return ChatService()


def test_reply_consumes_notes(service: ChatService, monkeypatch: pytest.MonkeyPatch) -> None:
    """reply 把便签拼入模型输入，并清空待投递队列。

    经 monkeypatch run_turn 捕获模型实际收到的 user_input（async fake，
    由 reply 内部的 asyncio.run 驱动，无需 pytest-asyncio）。
    """
    captured: dict = {}

    async def _fake_run_turn(agent, context, user_input, history_input, **kwargs):  # noqa: ANN001
        captured["user_input"] = user_input
        # result=None：format_tool_activity/_extract_tool_names/extract_usage 均兼容。
        return "回复", None, None

    monkeypatch.setattr(cs_mod, "run_turn", _fake_run_turn)

    service.add_context_note("已加载数据集 demo（路径 /x/demo）")
    service.reply("概况如何？")

    assert "已加载数据集 demo" in captured["user_input"]
    assert captured["user_input"].endswith("用户问题：概况如何？")
    # 便签消费后清空：下一轮输入不再携带。
    service.reply("第二轮")
    assert captured["user_input"] == "第二轮"


def test_reply_without_notes_passthrough(
    service: ChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """无便签：模型收到原始输入（不引入多余前缀）。"""
    captured: dict = {}

    async def _fake_run_turn(agent, context, user_input, history_input, **kwargs):  # noqa: ANN001
        captured["user_input"] = user_input
        return "回复", None, None

    monkeypatch.setattr(cs_mod, "run_turn", _fake_run_turn)
    ChatService().reply("直接问题")
    assert captured["user_input"] == "直接问题"


# --- _load_path 集成：面板加载成功即投递便签 ------------------------------------


def test_load_path_success_delivers_note(service: ChatService, tmp_path: Path) -> None:
    """面板加载成功 → service._pending_notes 含数据集名与路径。"""
    csv_path = tmp_path / "panel_ds.csv"
    pd.DataFrame({
        "episode": [1, 2], "success": [1, 0], "j1": [0.1, 0.2],
    }).to_csv(csv_path, index=False)
    messages: list[dict] = []
    ok = dlp._load_path(service, messages, str(csv_path))
    assert ok is True
    assert len(service._pending_notes) == 1
    note = service._pending_notes[0]
    assert "panel_ds.csv" in note and str(csv_path) in note
    assert "success" in note
    # UI 对话流说明仍然保留（用户可见层不变）。
    assert any("panel_ds.csv" in m["content"] for m in messages)


def test_load_path_failure_no_note(service: ChatService) -> None:
    """加载失败：不投递便签（模型无需知道失败事件，UI 说明足够）。"""
    messages: list[dict] = []
    ok = dlp._load_path(service, messages, "Z:/no/such/ds")
    assert ok is False
    assert service._pending_notes == []
