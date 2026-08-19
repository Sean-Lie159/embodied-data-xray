"""app/agent/agent.format_tool_activity 的单元测试。

验证工具调用轨迹展示的稳健性：按类型宽容提取工具名，提取失败时降级显示
原始类型名而非静默丢弃。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.agent.agent import format_tool_activity


def _tool_item(type_: str, **attrs) -> SimpleNamespace:
    """构造一个模拟的 tool_call item。"""
    return SimpleNamespace(type=type_, **attrs)


def test_none_result_returns_empty() -> None:
    assert format_tool_activity(None) == ""


def test_extracts_name() -> None:
    result = SimpleNamespace(
        new_items=[_tool_item("tool_call_item", name="load_dataset")]
    )
    out = format_tool_activity(result)
    assert "load_dataset" in out


def test_falls_back_to_tool_name() -> None:
    # 无 name 属性，但有 tool_name。
    result = SimpleNamespace(
        new_items=[_tool_item("tool_call_item", tool_name="profile_data")]
    )
    out = format_tool_activity(result)
    assert "profile_data" in out


def test_degrades_to_raw_dict_name() -> None:
    # name 属性均缺失，但 raw_item 字典里有 name。
    item = SimpleNamespace(type="tool_call_item", raw_item={"name": "check_temporal_sync"})
    result = SimpleNamespace(new_items=[item])
    out = format_tool_activity(result)
    assert "check_temporal_sync" in out


def test_degrades_to_type_name_when_unresolvable() -> None:
    # 无任何可提取的工具名 → 降级显示原始类型名，不静默丢弃。
    item = SimpleNamespace(type="tool_call_item")
    result = SimpleNamespace(new_items=[item])
    out = format_tool_activity(result)
    assert out != ""
    assert "<" in out  # 降级显示类型名


def test_ignores_non_tool_call_items() -> None:
    result = SimpleNamespace(
        new_items=[
            _tool_item("message", content="hi"),
            _tool_item("tool_call_item", name="load_dataset"),
        ]
    )
    out = format_tool_activity(result)
    assert "load_dataset" in out
    assert "message" not in out
