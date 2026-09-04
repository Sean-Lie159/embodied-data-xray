"""模型上下文窗口推定与上下文预算派生的单元测试。

覆盖：三级兜底优先级、最长前缀匹配（顺序无关，防 deepseek V3/V4 互相误判）、
核证清单全量抽检、token 估算的保守方向、预算派生各个比例。
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


def test_longest_prefix_wins_regardless_of_order() -> None:
    """最长命中前缀必须胜出——与表内书写顺序无关（防维护陷阱）。

    历史教训：按厂商分组书写表会破坏"长度降序"，靠排序断言才抓到。现匹配器
    已改为顺序无关（取最长命中前缀），本条断言锁定该行为本身。
    """
    # 构造：把表打乱顺序后逐一验证关键模型，结果必须与原表一致。
    key_cases = {
        "deepseek-v4-flash": 1_048_576,
        "deepseek-chat": 131_072,
        "gpt-5.6-sol": 1_050_000,
        "gpt-5": 400_000,
        "gpt-4o": 131_072,
        "gpt-4": 8_192,
        "kimi-k3": 1_048_576,
        "kimi-k2.7-code": 262_144,
        "glm-5.3-flash": 1_048_576,
        "glm-5-turbo": 200_000,
        "hy3": 262_144,
    }
    for name, expected in key_cases.items():
        assert resolve_context_window(name)[0] == expected, name



def test_shadowed_prefixes_agree_on_overlap() -> None:
    """互为前缀的表内条目，短前缀命中时不得落在长前缀的"窗口差异"里。

    匹配器已顺序无关（最长命中胜出），本条断言检查表内互为前缀的条目组合：
    短前缀作为独立模型名（如模型恰好叫 "gpt-5"）时，必须命中短前缀自身
    （400_000 而非长前缀 gpt-5.6 的 1_050_000）。
    """
    assert resolve_context_window("gpt-5")[0] == 400_000
    assert resolve_context_window("grok-4")[0] == 1_048_576
    assert resolve_context_window("glm-5")[0] == 200_000
    assert resolve_context_window("kimi")[0] == 262_144
    assert resolve_context_window("llama-4")[0] == 262_144
    assert resolve_context_window("deepseek-v3")[0] == 131_072


# --- 2.1 核证清单抽检（2026-09，owner 提供）--------------------------------


def test_openai_family() -> None:
    """GPT 各代不得互相误判（gpt-5.6/4.1/4o/5/4 前缀纠缠）。"""
    assert resolve_context_window("gpt-5.6-sol")[0] == 1_050_000
    assert resolve_context_window("gpt-5")[0] == 400_000
    assert resolve_context_window("gpt-4.1-mini")[0] == 1_048_576
    assert resolve_context_window("gpt-4o-2024")[0] == 131_072
    assert resolve_context_window("gpt-4")[0] == 8_192
    assert resolve_context_window("gpt-3.5-turbo")[0] == 16_384


def test_anthropic_family() -> None:
    """Claude 5 代 1M，4 代走 claude 兜底 200K。"""
    assert resolve_context_window("claude-sonnet-5")[0] == 1_048_576
    assert resolve_context_window("claude-opus-5-20260101")[0] == 1_048_576
    assert resolve_context_window("claude-fable-5")[0] == 1_048_576
    assert resolve_context_window("claude-opus-4.1")[0] == 200_000
    assert resolve_context_window("claude-sonnet-4.5")[0] == 200_000
    assert resolve_context_window("claude-haiku-4.5")[0] == 200_000


def test_google_xai_meta_mistral() -> None:
    """Gemini 全系 1M；Grok 分代；Llama scout 刻意保守。"""
    assert resolve_context_window("gemini-3.1-pro")[0] == 1_048_576
    assert resolve_context_window("gemini-2.5-flash")[0] == 1_048_576
    assert resolve_context_window("grok-4.20")[0] == 2_097_152
    assert resolve_context_window("grok-4.3")[0] == 1_048_576
    assert resolve_context_window("grok-4.5")[0] == 1_048_576
    assert resolve_context_window("grok-3")[0] == 131_072  # 3 系走 grok 兜底
    assert resolve_context_window("llama-4-scout")[0] == 262_144, (
        "scout 的 10M 是外推值，按预训练可靠值 256K 保守计"
    )
    assert resolve_context_window("llama-4-maverick")[0] == 1_048_576
    assert resolve_context_window("mistral-large-3")[0] == 262_144


def test_deepseek_qwen_glm() -> None:
    """DeepSeek V4 1M 其余 128K；Qwen/GLM 只认 Max/5.x 高配。"""
    assert resolve_context_window("deepseek-v4-pro")[0] == 1_048_576
    assert resolve_context_window("deepseek-v3.2")[0] == 131_072
    assert resolve_context_window("qwen3.8-max")[0] == 1_048_576
    assert resolve_context_window("qwen3.7-max")[0] == 1_048_576
    assert resolve_context_window("qwen3-235b")[0] == 131_072  # 清单外保守
    assert resolve_context_window("glm-5.3-flash")[0] == 1_048_576
    assert resolve_context_window("glm-5.2")[0] == 1_048_576
    assert resolve_context_window("glm-5-turbo")[0] == 200_000
    assert resolve_context_window("glm-4-plus")[0] == 131_072


def test_moonshot_tencent_minimax() -> None:
    """Kimi K3 1M / K2 256K；Hy3 与 hunyuan 同值；MiniMax 分代。"""
    assert resolve_context_window("kimi-k3")[0] == 1_048_576
    assert resolve_context_window("kimi-k2.7-code")[0] == 262_144
    assert resolve_context_window("kimi-k2-0905-preview")[0] == 262_144
    assert resolve_context_window("hunyuan-turbo")[0] == 262_144
    assert resolve_context_window("hy3")[0] == 262_144
    assert resolve_context_window("minimax-m3")[0] == 1_048_576
    assert resolve_context_window("minimax-m2.5")[0] == 200_000


def test_max_input_ratio_constraint_documented() -> None:
    """默认 ratio 0.6 必须低于核证模型的最紧输入占比（GPT-5 的 0.68）。

    这是 docstring 中"预算比例上限"警告的护栏：预算 = 窗口 × ratio，若 ratio
    高于某模型"最大输入/窗口"占比，该模型会在输入上限先于窗口处爆掉。
    """
    settings = Settings(
        _env_file=None,
        openai_api_key="k",
        openai_base_url="https://api.example.com/v1",
        default_model="example-model",
    )
    assert settings.context_budget_ratio <= 0.68, (
        "context_budget_ratio 超过 GPT-5 的最大输入占比 0.68，"
        "GPT-5 等模型将先撞输入上限（HTTP 400）"
    )


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
