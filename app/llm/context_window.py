"""模型上下文窗口推定与上下文预算派生。

为什么需要本模块：本项目走 OpenAI 兼容的 Chat Completions 端点，而
``/v1/models`` 接口**不返回** ``context_window`` 字段（OpenAI 官方亦无标准的
上下文长度查询 API），因此"模型能吃多少 input token"无法自动获知，必须显式提供。

三级兜底（优先级从高到低）：
    1. 配置显式指定（``context_window_tokens > 0``）——最准，换模型时推荐显式配；
    2. 内置模型表按**最长前缀**匹配——覆盖常见模型（见下表，由项目 owner
       于 2026-09 自行检索核证后提供）；
    3. 认不出时按 256K 兜底——以本项目基准模型 HY3 为准。

为什么兜底取 256K 而不是更小值：按项目 owner 指定以 HY3（256K）为基准。
已知代价：若使用内置表未覆盖的**小上下文**模型（如 32K/64K），会按 256K 估算
从而仍可能撞 HTTP 400；缓解方式是换模型时显式配置 ``CONTEXT_WINDOW_TOKENS``。

内置模型表（项目 owner 核证，2026-09；前缀 → 上下文窗口 token）：

    厂商        前缀                        上下文      备注
    ── OpenAI
    gpt-5.6    1_050_000   Sol/Terra/Luna；最大输入约 922K
    gpt-5        400_000   原版；最大输入 272K（占比 0.68，全表最紧）
    gpt-4.1    1_048_576   最大输出仅 32K
    gpt-4o       131_072
    gpt-4          8_192   老版本
    gpt-3.5      16_384
    ── Anthropic（claude-*；5 代 1M，4 代 200K）
    claude-sonnet-5 / claude-opus-5 / claude-fable   1_048_576
    claude（兜底）                                   200_000   含 opus-4.1/sonnet-4.5/haiku-4.5
    ── Google（gemini-* 全系 1M）
    gemini                                           1_048_576   3.1 Pro / 3.5 Flash / 2.5 Pro
    ── xAI
    grok-4.20    2_097_152   推理/非推理同窗口
    grok-4.3 / grok-4.5 / grok-4（4 系兜底）          1_048_576   grok-4 原版未核证，取 4 系最小
    grok（兜底，3 系及更早）                          131_072   保守
    ── Meta
    llama-4-scout    262_144   **刻意保守**：官方宣称 10M 为外推值（预训练仅到
                               256K），按可靠值计；估高方向才是危险方向
    llama-4-maverick 1_048_576
    llama-4 / llama（兜底）      262_144 / 131_072
    ── Mistral
    mistral-large-3  262_144
    mistral（兜底）  131_072   Large 2 为 128K
    ── DeepSeek
    deepseek-v4  1_048_576   Pro/Flash；1M 输入输出共享，最大输出 384K
    deepseek-v3 / deepseek-chat / deepseek-reasoner / deepseek（兜底）   131_072
                             V3.2 注意：最大输入仅 96K（128K 窗口的 0.75）
    ── 阿里
    qwen3.8-max / qwen3.7-max  1_048_576   最大输入 991,808（标准）
    qwen（兜底）               131_072   清单外（Qwen3 标准系 128K）保守
    ── 智谱
    glm-5.3 / glm-5.3-flash / glm-5.2  1_048_576
    glm-5（兜底 5/5-Turbo/5.1）        200_000
    glm（兜底）                        131_072   GLM-4 为 128K
    ── Moonshot
    kimi-k3    1_048_576   最大输出可设 1M（常规 128K）
    kimi（兜底）262_144   K2.7 Code / K2.6 均为 256K
    ── 腾讯
    hy3 / hunyuan    262_144   **本项目基准**；注意最大输入仅 192K（占比 0.75）
    ── MiniMax
    minimax-m3  1_048_576
    minimax（兜底）200_000   M2.5 / M2.7

**必须最长前缀优先**：``deepseek-v4-flash`` 若被 ``deepseek`` 命中会按 128K 算
（低估 8 倍，过早压缩）；反之若 ``deepseek`` 兜底设成 1M，则 V3 会被高估 8 倍
而撑爆。匹配器实现为"取最长命中前缀"，**与表内书写顺序无关**——表条目可按
厂商分组阅读，排序维护陷阱从根上消除。

**预算比例上限**：预算 = 窗口 × ratio，而部分模型"最大输入"小于窗口
（GPT-5 占比 0.68、Hy3 与 V3.2 为 0.75）。默认 ratio=0.6 低于所有核证模型的
输入占比，安全；**若自行调高 ``context_budget_ratio``，不要超过 0.68**，
否则 GPT-5 这类模型会在"输入上限"先于"上下文窗口"处爆掉。
"""

from __future__ import annotations

from dataclasses import dataclass

# 内置模型表（前缀 → 上下文 token）。匹配器取**最长命中前缀**，与书写顺序
# 无关（历史教训：按厂商分组书写曾破坏"长度降序"要求，靠断言才抓到，遂将
# 顺序无关性做进实现而非依赖约定）。tests/test_context_window.py 锁定该行为。
# 维护提示：模型更新会过时，需随模型更新维护；用户可随时用配置覆盖。
_MODEL_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    # --- OpenAI ---
    ("gpt-5.6", 1_050_000),          # Sol/Terra/Luna；最大输入约 922K
    ("gpt-4.1", 1_048_576),
    ("gpt-3.5", 16_384),
    ("gpt-4o", 131_072),
    ("gpt-5", 400_000),              # 原版；最大输入 272K（全表最紧，占比 0.68）
    ("gpt-4", 8_192),                # 老版本；须排在 "gpt-4o"/"gpt-4.1" 之后
    # --- Anthropic（5 代 1M，4 代 200K）---
    ("claude-sonnet-5", 1_048_576),
    ("claude-opus-5", 1_048_576),
    ("claude-fable", 1_048_576),
    ("claude", 200_000),             # 兜底：opus-4.1 / sonnet-4.5 / haiku-4.5 均 200K
    # --- Google（核证清单内全系 1M）---
    ("gemini", 1_048_576),
    # --- xAI ---
    ("grok-4.20", 2_097_152),        # 推理/非推理同窗口
    ("grok-4.3", 1_048_576),
    ("grok-4.5", 1_048_576),
    ("grok-4", 1_048_576),           # 4 系兜底（grok-4 原版未核证，取 4 系最小）
    ("grok", 131_072),               # 3 系及更早，保守
    # --- Meta ---
    ("llama-4-maverick", 1_048_576),
    ("llama-4-scout", 262_144),      # 刻意保守：10M 为外推值，预训练仅到 256K
    ("llama-4", 262_144),
    ("llama", 131_072),              # llama-3 系 128K
    # --- Mistral ---
    ("mistral-large-3", 262_144),
    ("mistral", 131_072),            # Large 2 为 128K
    # --- DeepSeek ---
    ("deepseek-v4", 1_048_576),      # Pro/Flash；1M 输入输出共享，最大输出 384K
    ("deepseek-chat", 131_072),
    ("deepseek-reasoner", 131_072),
    ("deepseek-v3", 131_072),        # V3.2 最大输入仅 96K（窗口的 0.75）
    ("deepseek", 131_072),           # 兜底：保守，不假设是 V4
    # --- 阿里 ---
    ("qwen3.8-max", 1_048_576),
    ("qwen3.7-max", 1_048_576),
    ("qwen", 131_072),               # 清单外（Qwen3 标准系 128K）保守
    # --- 智谱 ---
    ("glm-5.3", 1_048_576),          # 含 5.3-flash
    ("glm-5.2", 1_048_576),
    ("glm-5", 200_000),              # 5 / 5-Turbo / 5.1
    ("glm", 131_072),                # GLM-4 为 128K
    # --- Moonshot ---
    ("kimi-k3", 1_048_576),          # 最大输出可设 1M（常规 128K 用）
    ("kimi", 262_144),               # K2.7 Code / K2.6 均为 256K
    # --- 腾讯 ---
    ("hunyuan", 262_144),            # 最大输入仅 192K（占比 0.75）
    ("hy3", 262_144),                # 本项目基准
    # --- MiniMax ---
    ("minimax-m3", 1_048_576),
    ("minimax", 200_000),            # M2.5 / M2.7
)

# 认不出模型时的兜底值：以 HY3（256K）为基准（项目 owner 指定）。
_FALLBACK_CONTEXT_WINDOW = 262_144


@dataclass(frozen=True)
class ContextBudget:
    """上下文预算派生结果（单位：token，估算值）。"""

    context_window: int
    """模型上下文窗口（token）。"""
    source: str
    """窗口来源："configured"（配置显式指定）/ "model_table"（表命中）/ "fallback"（兜底）。"""
    total_budget: int
    """总可用预算 = context_window × context_budget_ratio。"""
    history_budget: int
    """历史压缩触发阈值。"""
    tool_output_budget: int
    """单次工具返回硬上限。"""


def resolve_context_window(model_name: str, configured: int = 0) -> tuple[int, str]:
    """确定模型的上下文窗口（token）。

    Args:
        model_name: 模型名（如 "hy3"、"deepseek-v4-flash"）。
        configured: 配置显式指定的值；>0 时直接采用（最高优先级）。

    Returns:
        (上下文窗口 token, 来源标记)。
    """
    if configured and configured > 0:
        return int(configured), "configured"

    name = (model_name or "").strip().lower()
    if name:
        # 取**最长命中前缀**（最具体条目优先）。与表顺序无关——这从根上消除
        # "插行破坏降序导致静默误判"的维护陷阱（真实发生过：按厂商分组书写
        # 会破坏长度排序，靠测试断言才抓到）。
        best: tuple[str, int] | None = None
        for prefix, window in _MODEL_CONTEXT_WINDOWS:
            if name.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
                best = (prefix, window)
        if best is not None:
            return best[1], "model_table"
    return _FALLBACK_CONTEXT_WINDOW, "fallback"


def derive_budget(
    model_name: str,
    *,
    configured_window: int = 0,
    budget_ratio: float = 0.6,
    history_ratio: float = 0.75,
    tool_output_ratio: float = 0.25,
) -> ContextBudget:
    """由模型上下文窗口派生各层预算。

    Args:
        model_name: 模型名。
        configured_window: 配置显式指定的上下文窗口（>0 优先）。
        budget_ratio: 上下文可用比例（其余留给 system prompt / 本轮输入 / 模型输出）。
        history_ratio: 历史压缩阈值占总预算的比例。
        tool_output_ratio: 单次工具返回上限占总预算的比例。

    Returns:
        派生后的 :class:`ContextBudget`。
    """
    window, source = resolve_context_window(model_name, configured_window)
    total = max(1, int(window * budget_ratio))
    return ContextBudget(
        context_window=window,
        source=source,
        total_budget=total,
        history_budget=max(1, int(total * history_ratio)),
        tool_output_budget=max(1, int(total * tool_output_ratio)),
    )


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数（**保守启发式**，宁可估高也不估低）。

    为什么是估算而非精确计数：模型按 token 限流，但我们手上只有字符数；且
    DeepSeek / Kimi / 混元的分词器与 OpenAI 不同，引入 tiktoken 会增加依赖且对
    第三方模型并不准确。

    保守系数（偏差方向是安全的——估高只会提前压缩，绝不会估低撑爆）：
        - CJK 字符：1.0 token/字（实际约 0.6~1，取上限）
        - 非 CJK  ：1 token / 3 字符（英文实际约 4 字符/token，取 3 更保守）

    Args:
        text: 任意文本。

    Returns:
        估算的 token 数（≥0）。
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        # CJK 统一表意文字 + 扩展区 + 兼容表意文字。
        if (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3400" <= ch <= "\u4dbf"
            or "\uf900" <= ch <= "\ufaff"
        ):
            cjk += 1
        elif ch.isspace():
            continue  # 空白按 0 计（略微低估，但被其它项的保守系数覆盖）
        else:
            other += 1
    return int(cjk * 1.0) + int(other / 3) + 1
