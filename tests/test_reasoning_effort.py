"""推理档位（REASONING_EFFORT）可配置项的单元测试。

背景（2026-09-11 实测）：本项目基准链路是 CodeBuddy 网关 → 混元 hy3，而该网关
在请求**未带** ``reasoning_effort`` 时会自动注入 "high"（见网关 upstream.py 的
"Reasoning / chain-of-thought injection"）——即推理模式由网关替客户端开启。
实测 hy3 默认档思考 token 占总输出 91%~96%，一次 300 字分析问答烧约 900 个
看不见的思考 token，按 13 ms/token 折算，等待时间几乎全在思考上。

本改动把它做成配置项（默认留空 = 不干预，即保持网关默认 high，零回归）。

覆盖：值映射边界、ModelSettings 构造、Agent 挂载（含 None 不传参的 SDK 约束）、
以及未配置时的零回归。
"""

from __future__ import annotations

from app.agent.agent import build_agent
from app.config.settings import Settings
from app.llm.factory import build_model_settings, build_reasoning_extra_body


def _settings(reasoning_effort: str = "") -> Settings:
    """构造测试用配置（必填三项给占位值，不联网）。"""
    return Settings(
        openai_api_key="test-key",
        openai_base_url="http://127.0.0.1:1/v1",
        default_model="hy3",
        reasoning_effort=reasoning_effort,
    )


# --- 1. 值映射边界 ----------------------------------------------------------


def test_empty_means_no_intervention() -> None:
    """留空 = 不注入任何字段（把选择权留给服务端/网关，零回归的语义基础）。"""
    assert build_reasoning_extra_body("") == {}
    assert build_reasoning_extra_body("   ") == {}


def test_valid_levels_are_passed_through() -> None:
    """三个合法档位原样透传（high/low/off）。"""
    assert build_reasoning_extra_body("high") == {"reasoning_effort": "high"}
    assert build_reasoning_extra_body("low") == {"reasoning_effort": "low"}
    assert build_reasoning_extra_body("off") == {"reasoning_effort": "off"}


def test_case_and_whitespace_insensitive() -> None:
    """大小写与首尾空白不敏感（.env 手写常见）。"""
    assert build_reasoning_extra_body("LOW") == {"reasoning_effort": "low"}
    assert build_reasoning_extra_body("  High  ") == {"reasoning_effort": "high"}


def test_unknown_values_are_ignored_not_forwarded() -> None:
    """不认识的取值不转发（防拼写错误被当成无效值送给上游）。

    实测教训：上游只认 "low"/"high"，其它值（如 "none"/"medium"）送到上游
    会导致思考链行为不确定；本项目对未知值一律"不干预"。
    """
    for bad in ("none", "medium", "disable", "garbage", "1", "true"):
        assert build_reasoning_extra_body(bad) == {}, bad


# --- 2. ModelSettings 构造 --------------------------------------------------


def test_model_settings_none_when_unconfigured() -> None:
    """未配置档位 → 返回 None（表示"不注入"，而非"注入空设置"）。"""
    assert build_model_settings(_settings("")) is None
    assert build_model_settings(_settings("garbage")) is None


def test_model_settings_carries_extra_body() -> None:
    """配置档位 → ModelSettings.extra_body 携带该字段。"""
    ms = build_model_settings(_settings("low"))
    assert ms is not None
    assert ms.extra_body == {"reasoning_effort": "low"}


def test_model_settings_leaves_other_fields_untouched() -> None:
    """只设 extra_body，不污染其它字段（不覆盖 SDK 默认行为）。"""
    ms = build_model_settings(_settings("high"))
    assert ms is not None
    assert ms.temperature is None
    assert ms.max_tokens is None
    assert ms.tool_choice is None


# --- 3. Agent 挂载（含 SDK 对 None 的硬约束）-------------------------------


def test_agent_accepts_none_without_passing_it() -> None:
    """**关键回归**：未配置时不得把 None 传给 SDK。

    SDK 的 Agent 构造器对 model_settings=None 会抛
    ``TypeError: must be a ModelSettings instance or a dict``——若实现写成
    ``Agent(..., model_settings=None)``，未配置推理档位的用户会**直接崩溃**
    （改动前所有用户都属于这种情况）。本测试锁定"条件传参"这一实现。
    """
    agent = build_agent("hy3", [], model_settings=None)
    assert agent is not None


def test_agent_carries_configured_settings() -> None:
    """配置档位时 Agent 持有对应 settings。"""
    agent = build_agent("hy3", [], model_settings=build_model_settings(_settings("low")))
    assert agent.model_settings is not None
    assert agent.model_settings.extra_body == {"reasoning_effort": "low"}


def test_agent_without_settings_has_empty_extra_body() -> None:
    """未配置时 Agent 的 extra_body 为空（不注入字段）。"""
    agent = build_agent("hy3", [])
    assert agent.model_settings is None or not agent.model_settings.extra_body


# --- 4. 默认值（零回归语义）------------------------------------------------


def test_default_is_empty_not_high() -> None:
    """默认留空 = 不干预。

    为什么默认不是 "high" 而是空：本参数的作用是**覆盖网关默认值**。留空即
    "客户端不发表意见"，网关仍按自己的默认（当前为 high）处理——效果上等价于
    high，但把决定权留在网关侧：将来网关改了默认值，本项目自动跟随，不需要
    用户改 .env。而写死 "high" 会反过来锁住网关的默认。
    """
    assert Settings(
        openai_api_key="k", openai_base_url="http://x", default_model="m"
    ).reasoning_effort == ""


def test_end_to_end_unconfigured_is_zero_regression() -> None:
    """端到端：默认配置下 build_model_settings 返回 None → 不注入任何字段。"""
    s = _settings()
    assert build_model_settings(s) is None
    agent = build_agent("hy3", [], model_settings=build_model_settings(s))
    assert agent.model_settings is None or not agent.model_settings.extra_body
