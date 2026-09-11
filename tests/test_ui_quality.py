"""UI 工程质量改动的单测（docs/UI工程质量与配置化设计.md 第 4 节）。

覆盖：_record_turn 累加语义（与改动前逐字段一致）、_last_turn 取值健壮性、
UI 常量被真正引用（防"定义了但没用"）、UPLOAD_MAX_MB 配置生效。
"""

from __future__ import annotations

import inspect
from pathlib import Path

from app.services.chat_service import ChatTurn


def _turn(reply: str = "ok", *, usage=None, metrics=None) -> ChatTurn:
    return ChatTurn(reply=reply, tool_activity="", usage=usage, metrics=metrics)


def _cumulative() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "rounds": 0, "duration_ms": 0, "n_model_calls": 0}


def test_record_turn_accumulates_all_fields() -> None:
    """token 三段 + rounds + 耗时 + 往返次数全部累加，且追加 assistant 消息。"""
    import streamlit_app

    cum = _cumulative()
    msgs: list[dict] = []
    turn = _turn("回答", usage={"input_tokens": 100, "output_tokens": 20,
                                "total_tokens": 120},
                 metrics={"duration_ms": 1500, "n_model_calls": 3,
                          "completed": True})
    streamlit_app._record_turn(cum, turn, msgs)

    assert cum["input_tokens"] == 100
    assert cum["output_tokens"] == 20
    assert cum["total_tokens"] == 120
    assert cum["rounds"] == 1
    assert cum["duration_ms"] == 1500
    assert cum["n_model_calls"] == 3
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["turn"] is turn


def test_record_turn_accumulates_not_overwrites() -> None:
    """两次调用是累加而非覆盖（此前三处重复实现易漏加）。"""
    import streamlit_app

    cum = _cumulative()
    msgs: list[dict] = []
    streamlit_app._record_turn(
        cum, _turn(usage={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
                   metrics={"duration_ms": 100, "n_model_calls": 1}), msgs)
    streamlit_app._record_turn(
        cum, _turn(usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                   metrics={"duration_ms": 200, "n_model_calls": 2}), msgs)

    assert cum["input_tokens"] == 15
    assert cum["output_tokens"] == 3
    assert cum["total_tokens"] == 18
    assert cum["rounds"] == 2
    assert cum["duration_ms"] == 300
    assert cum["n_model_calls"] == 3
    assert len(msgs) == 2


def test_record_turn_usage_none_still_counts_round() -> None:
    """usage 为 None 时不加 token，但轮数仍 +1（与改动前语义一致）。"""
    import streamlit_app

    cum = _cumulative()
    msgs: list[dict] = []
    streamlit_app._record_turn(cum, _turn(usage=None), msgs)

    assert cum["input_tokens"] == 0
    assert cum["rounds"] == 1
    assert len(msgs) == 1


def test_last_turn_returns_none_for_empty() -> None:
    """空消息列表 → None（不抛异常）。"""
    import streamlit_app

    assert streamlit_app._last_turn([]) is None


def test_last_turn_skips_trailing_user_message() -> None:
    """末尾是用户消息（编辑态）时，仍能取到更早的真实回复。"""
    import streamlit_app

    turn = _turn("旧回复", usage={"input_tokens": 1, "output_tokens": 1,
                                  "total_tokens": 2})
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "旧回复", "turn": turn},
            {"role": "user", "content": "新问题"}]
    assert streamlit_app._last_turn(msgs) is turn


def test_last_turn_handles_missing_turn_key() -> None:
    """assistant 消息缺 turn 字段时不抛 AttributeError（此前 `.get('turn').usage` 会）。"""
    import streamlit_app

    msgs = [{"role": "assistant", "content": "面板加载说明"}]  # 无 turn
    assert streamlit_app._last_turn(msgs) is None


def test_scroll_css_uses_reserve_constant() -> None:
    """CSS 注入真正引用了 CHAT_INPUT_RESERVE_PX（防"定义了但没用"）。"""
    import streamlit_app
    from app.ui.constants import CHAT_INPUT_RESERVE_PX

    src = inspect.getsource(streamlit_app._inject_scroll_css)
    assert "CHAT_INPUT_RESERVE_PX" in src
    assert CHAT_INPUT_RESERVE_PX > 0


def test_constants_are_imported_and_used() -> None:
    """SCROLL_HEIGHT / SESSION_LABEL_MAX_CHARS / COLUMN_RATIO 被 streamlit_app 引用。"""
    import streamlit_app

    src = inspect.getsource(streamlit_app)
    for name in ("SCROLL_HEIGHT", "SESSION_LABEL_MAX_CHARS", "COLUMN_RATIO"):
        assert name in src, f"常量 {name} 未被引用"


def test_upload_max_mb_config(tmp_path: Path) -> None:
    """UPLOAD_MAX_MB 是配置项且可覆盖（AGENTS.md 第 6 条：配置集中在 app/config）。"""
    from app.config.settings import Settings

    s = Settings(_env_file=None, openai_api_key="k", openai_base_url="http://x",
                 default_model="m")
    assert s.upload_max_mb == 200  # 默认值

    s2 = Settings(_env_file=None, openai_api_key="k", openai_base_url="http://x",
                  default_model="m", upload_max_mb=50)
    assert s2.upload_max_mb == 50


def test_upload_panel_reads_config_not_module_constant() -> None:
    """数据加载面板读配置而非模块常量（防回退到硬编码）。"""
    import app.ui.data_loader_panel as panel

    src = inspect.getsource(panel)
    assert "upload_max_mb" in src
    assert "_MAX_UPLOAD_MB" not in src
