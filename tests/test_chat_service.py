"""app/services/chat_service 的单元测试。

用 mock 替换 chat_service 模块内的 run_turn，验证 ChatService 的接口结构
（回复、工具轨迹、findings），不依赖真实模型调用。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services import chat_service
from app.services.chat_service import ChatService, ChatTurn


def _fake_result():
    """构造一个假 RunResult：含一次工具调用与回复。"""
    return SimpleNamespace(
        final_output="已加载数据集 demo",
        new_items=[
            SimpleNamespace(type="tool_call_item", name="load_dataset"),
            SimpleNamespace(type="tool_call_output_item"),
        ],
        to_input_list=lambda mode="normalized": [{"role": "user", "content": "x"}],
    )


def test_chat_service_builds_agent() -> None:
    """ChatService 应能初始化并构建 agent。"""
    service = ChatService()
    assert service.agent is not None
    assert service.context is not None
    assert service.history_input is None


def test_reply_returns_chat_turn(monkeypatch) -> None:
    """reply 应返回 ChatTurn（回复/工具轨迹/findings）。"""
    async def fake_run_turn(agent, context, user_input, history_input=None, max_turns=15):
        return ("已加载数据集 demo", [{"role": "user", "content": user_input}], _fake_result())

    # patch chat_service 模块内的 run_turn。
    monkeypatch.setattr(chat_service, "run_turn", fake_run_turn)

    service = ChatService()
    turn = service.reply("请加载数据")

    assert isinstance(turn, ChatTurn)
    assert turn.reply == "已加载数据集 demo"
    # 工具轨迹与工具名列表含 load_dataset。
    assert "load_dataset" in turn.tool_activity
    assert "load_dataset" in turn.tool_calls


def test_areply_async_returns_chat_turn(monkeypatch) -> None:
    """异步 areply 也应返回 ChatTurn。"""
    async def fake_run_turn(agent, context, user_input, history_input=None, max_turns=15):
        return ("回复", [{"role": "user", "content": user_input}], _fake_result())

    monkeypatch.setattr(chat_service, "run_turn", fake_run_turn)

    import asyncio

    service = ChatService()
    turn = asyncio.run(service.areply("你好"))
    assert turn.reply == "回复"


def test_areply_handles_max_turns_exceeded(monkeypatch) -> None:
    """MaxTurnsExceeded 时 run_turn 返回 result=None，areply 应返回友好提示而不崩溃。

    契约：run_turn 触发超限时返回 (友好提示, fallback_input, None)。调用方不得
    对 result 解引用，应判空后返回空工具轨迹与友好提示。
    """
    async def fake_run_turn(agent, context, user_input, history_input=None, max_turns=15):
        return (
            "本轮工具调用次数已达上限（max_turns=15），为避免死循环已停止。请尝试更明确地描述需求，或分步提问。",
            [{"role": "user", "content": user_input}],
            None,
        )

    monkeypatch.setattr(chat_service, "run_turn", fake_run_turn)

    import asyncio

    service = ChatService()
    turn = asyncio.run(service.areply("分析一下"))
    assert "已达上限" in turn.reply
    assert turn.tool_activity == ""
    assert turn.tool_calls == []
    assert isinstance(turn.findings, list)


def test_dataset_summary_shape() -> None:
    """dataset_summary 应返回能力标签与流清单摘要。"""
    service = ChatService()
    s = service.dataset_summary()
    assert "dataset_id" in s
    assert "capabilities" in s
    assert "streams" in s
