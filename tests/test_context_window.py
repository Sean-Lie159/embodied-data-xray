"""模型上下文窗口推定与上下文预算派生的单元测试。

覆盖：三级兜底优先级、最长前缀匹配（防 deepseek V3/V4 互相误判）、
token 估算的保守方向、预算派生各个比例。
"""

from __future__ import annotations

from app.config.settings import Settings
from app.llm.context_window import (
    _FALLBACK_CONTEXT_WINDOW,
    _MODEL_CONTEXT_WINDOWS,
    ContextBudget,
    derive_budget,
    estimate_tokens,
    resolve_context_window,
)

# --- 1. 三级兜底优先级 ------------------------------------------------------


def test_configured_window_wins_over_table() -> None:
    """配置显式指定 > 内置表。"""
    window, source = resolve_context_window("hy3", configured=32_000)
    assert window == 32_000
    assert source == "configured"


def test_model_table_hit() -> None:
    """表命中：hy3 → 256K。"""
    window, source = resolve_context_window("hy3")
    assert window == 262_144
    assert source == "model_table"


def test_unknown_model_falls_back() -> None:
    """认不出 → 按 256K 兜底（以 HY3 为基准）。"""
    window, source = resolve_context_window("某个未收录的模型-v1")
    assert window == _FALLBACK_CONTEXT_WINDOW == 262_144
    assert source == "fallback"


def test_empty_model_name_falls_back() -> None:
    """空模型名不崩，走兜底。"""
    window, source = resolve_context_window("")
    assert window == _FALLBACK_CONTEXT_WINDOW
    assert source == "fallback"


# --- 2. 最长前缀匹配（防低估/高估 8 倍）------------------------------------


def test_deepseek_v4_not_matched_by_generic_deepseek() -> None:
    """deepseek-v4-flash 必须命中 v4（1M），不能被 deepseek 兜底（128K）命中。"""
    window, _ = resolve_context_window("deepseek-v4-flash")
    assert window == 1_048_576, "V4 是 1M；若被通用 deepseek 命中会低估 8 倍"


def test_deepseek_v3_stays_128k() -> None:
    """deepseek-v3 / deepseek-chat 保持 128K，不得被高估为 1M。"""
    assert resolve_context_window("deepseek-v3-0324")[0] == 131_072
    assert resolve_context_window("deepseek-chat")[0] == 131_072
    assert resolve_context_window("deepseek-reasoner")[0] == 131_072


def test_table_is_sorted_by_prefix_length_desc() -> None:
    """表必须按前缀长度降序——这是最长前缀优先的前提（防回归）。"""
    lengths = [len(prefix) for prefix, _ in _MODEL_CONTEXT_WINDOWS]
    assert lengths == sorted(lengths, reverse=True)


def test_case_and_space_insensitive() -> None:
    """模型名大小写与首尾空格不影响匹配。"""
    assert resolve_context_window("  Hy3  ")[0] == 262_144
    assert resolve_context_window("DeepSeek-V4-Flash")[0] == 1_048_576


# --- 3. token 估算（保守方向）----------------------------------------------


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_cjk_is_conservative() -> None:
    """中文按 1 token/字：估算不得低于实际量级方向。"""
    n = estimate_tokens("中" * 100)
    assert n >= 100, "中文应至少按 1 token/字估算（保守）"


def test_estimate_tokens_latin_ratio() -> None:
    """英文约 3 字符/token，比实际的 4 更保守（估高）。"""
    n = estimate_tokens("a" * 300)
    # 300/3 = 100（+1 兜底）；实际英文约 75 token → 估算偏高，方向安全。
    assert n >= 100


def test_estimate_tokens_mixed() -> None:
    """中英混合不崩，且量级合理。"""
    n = estimate_tokens("数据集 analysis 结果 2026")
    assert 1 <= n <= 100


# --- 4. 预算派生 ------------------------------------------------------------


def test_derive_budget_defaults() -> None:
    """默认比例：总预算 = 窗口×0.6，历史 = 总×0.75，单次返回 = 总×0.25。"""
    b = derive_budget("hy3")
    assert isinstance(b, ContextBudget)
    assert b.context_window == 262_144
    assert b.total_budget == int(262_144 * 0.6)
    assert b.history_budget == int(int(262_144 * 0.6) * 0.75)
    assert b.tool_output_budget == int(int(262_144 * 0.6) * 0.25)


def test_derive_budget_respects_configured_window() -> None:
    """显式配置的窗口参与派生。"""
    b = derive_budget("hy3", configured_window=1_048_576)
    assert b.context_window == 1_048_576
    assert b.total_budget == int(1_048_576 * 0.6)


def test_derive_budget_custom_ratios() -> None:
    """自定义比例生效。"""
    b = derive_budget("hy3", budget_ratio=0.5, history_ratio=0.8, tool_output_ratio=0.2)
    assert b.total_budget == int(262_144 * 0.5)
    assert b.history_budget == int(int(262_144 * 0.5) * 0.8)
    assert b.tool_output_budget == int(int(262_144 * 0.5) * 0.2)


def test_derive_budget_never_zero() -> None:
    """极端小窗口也保证各预算 ≥1（防除零/无意义阈值）。"""
    b = derive_budget("x", configured_window=1)
    assert b.total_budget >= 1
    assert b.history_budget >= 1
    assert b.tool_output_budget >= 1


# --- 5. 与 Settings 集成 ----------------------------------------------------


def test_settings_has_context_fields_with_defaults() -> None:
    """新增配置项存在且默认值符合设计。"""
    s = Settings(
        _env_file=None,
        openai_api_key="k",
        openai_base_url="https://api.example.com/v1",
        default_model="example-model",
    )
    assert s.context_window_tokens == 0          # 0 = 走内置表
    assert s.context_budget_ratio == 0.6
    assert s.history_budget_ratio == 0.75
    assert s.tool_output_budget_ratio == 0.25
    assert s.history_compaction_enabled is True
    assert s.history_keep_recent_turns == 3
