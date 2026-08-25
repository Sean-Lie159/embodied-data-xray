"""第 8 步 token 消耗统计的单元测试。

覆盖：extract_usage 从带 usage 的 result 透传正确、result=None 不崩、
无 usage 结构安全降级、CLI 格式化函数与成本估算。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.chat_service import ChatTurn, extract_usage
from app.config import get_settings


def _fake_result_with_usage(input_tokens: int, output_tokens: int) -> SimpleNamespace:
    """构造带 usage 的假 RunResult（含 to_state()._context.usage）。"""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    wrapper = SimpleNamespace(usage=usage)
    state = SimpleNamespace(_context=wrapper)
    result = SimpleNamespace(to_state=lambda: state)
    return result


def test_extract_usage_passthrough() -> None:
    """带 usage 的 result → extract_usage 正确返回 input/output/total。"""
    result = _fake_result_with_usage(1200, 300)
    usage = extract_usage(result)
    assert usage == {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500}


def test_extract_usage_none_result_returns_none() -> None:
    """result=None（MaxTurnsExceeded 等路径）→ usage None，不崩。"""
    assert extract_usage(None) is None


def test_extract_usage_no_usage_structure_returns_none() -> None:
    """result 无 usage 结构 → 返回 None，不抛异常（结构不符安全降级）。"""
    # to_state() 返回不含 _context 的对象。
    result = SimpleNamespace(to_state=lambda: SimpleNamespace())
    assert extract_usage(result) is None
    # to_state() 本身抛异常也应降级。
    result2 = SimpleNamespace(to_state=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert extract_usage(result2) is None


def test_chatturn_usage_field_default_none() -> None:
    """ChatTurn.usage 缺省为 None（未获取用量时不为 0 冒充）。"""
    turn = ChatTurn(reply="hi", tool_activity="")
    assert turn.usage is None


def test_format_tokens_usage_none_shows_not_available() -> None:
    """CLI 格式化：usage=None 显示"本次未获取到用量"，不显示 0。"""
    from main import _format_tokens

    line = _format_tokens(None, {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "rounds": 1})
    assert "本次未获取到用量" in line
    assert "会话累计 100/50/150" in line


def test_format_tokens_usage_present() -> None:
    """CLI 格式化：正常显示输入/输出/合计 + 会话累计。"""
    from main import _format_tokens

    usage = {"input_tokens": 1234, "output_tokens": 567, "total_tokens": 1801}
    cumulative = {"input_tokens": 1234, "output_tokens": 567, "total_tokens": 1801, "rounds": 1}
    line = _format_tokens(usage, cumulative)
    assert "输入 1,234" in line
    assert "输出 567" in line
    assert "合计 1,801" in line
    assert "会话累计 1,234/567/1,801" in line


def test_format_cost_not_configured_returns_empty() -> None:
    """未配置价格（默认 0）→ 成本为空串，不显示。"""
    from main import _format_cost

    assert get_settings().price_input_per_mtok == 0.0
    assert _format_cost({"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
                        {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500, "rounds": 1}) == ""


def test_format_cost_configured(monkeypatch) -> None:
    """配置价格后 → 返回成本估算文本。"""
    from main import _format_cost
    import app.config as config_mod
    import app.config.settings as settings_mod

    settings = settings_mod.get_settings()
    monkeypatch.setattr(settings, "price_input_per_mtok", 0.14)
    monkeypatch.setattr(settings, "price_output_per_mtok", 0.28)
    monkeypatch.setattr(config_mod, "get_settings", lambda: settings)
    # 输入 1000 + 输出 500 万 token 折算：1000/1e6*0.14 + 500/1e6*0.28 = 0.00028
    text = _format_cost({"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
                        {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500, "rounds": 1})
    assert text.startswith("≈$")
    assert "累计" in text
