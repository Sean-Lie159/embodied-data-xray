"""模型上下文窗口推定与上下文预算派生。

为什么需要本模块：本项目走 OpenAI 兼容的 Chat Completions 端点，而
``/v1/models`` 接口**不返回** ``context_window`` 字段（OpenAI 官方亦无标准的
上下文长度查询 API），因此"模型能吃多少 input token"无法自动获知，必须显式提供。

三级兜底（优先级从高到低）：
    1. 配置显式指定（``context_window_tokens > 0``）——最准，换模型时推荐显式配；
    2. 内置模型表按**最长前缀**匹配——覆盖常见模型，已联网核证（见下表注）；
    3. 认不出时按 256K 兜底——以本项目基准模型 HY3 为准。

为什么兜底取 256K 而不是更小值：按项目 owner 指定以 HY3（256K）为基准。
已知代价：若使用内置表未覆盖的**小上下文**模型（如 32K/64K），会按 256K 估算
从而仍可能撞 HTTP 400；缓解方式是换模型时显式配置 ``CONTEXT_WINDOW_TOKENS``。

内置模型表（已联网核证，2026-09）：

    前缀                 上下文    说明
    hy3                  262144    腾讯混元 Hy3（官方页/新华社/澎湃一致）
    deepseek-v4          1048576   V4 Flash/Pro；1M 为输入+输出共享，最大输出 384K
    deepseek-v3/chat/...  131072   V3/V3.2 官方 128K
    deepseek（兜底）       131072   保守：不假设是 V4
    kimi                 131072    保守：K2 七月版 128K，0905 起 256K（新版请覆盖）
    gpt-4o               131072
    gpt-4                  8192    老版本

**必须最长前缀优先**：``deepseek-v4-flash`` 若被 ``deepseek`` 命中会按 128K 算
（低估 8 倍，过早压缩）；反之若 ``deepseek`` 兜底设成 1M，则 V3 会被高估 8 倍
而撑爆。故按前缀长度降序排列后逐个匹配。
"""

from __future__ import annotations

from dataclasses import dataclass

# 内置模型表（前缀 → 上下文 token）。**必须按前缀长度降序**书写，匹配时取首个
# 命中即为最长前缀。反例警示：若 "gpt-4"（5）排在 "gpt-4o"（6）之前，"gpt-4o"
# 会被误判为 8K；若 "deepseek"（8）排在 "deepseek-v4"（11）之前，V4 会被误判
# 为 128K（低估 8 倍）。tests/test_context_window.py 有该顺序的回归断言。
# 维护提示：模型更新会过时，需随模型更新维护；用户可随时用配置覆盖。
_MODEL_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("deepseek-reasoner", 131_072),
    ("deepseek-chat", 131_072),      # V3 系列默认名
    ("deepseek-v4", 1_048_576),      # 1M（输入+输出共享，最大输出 384K）
    ("deepseek-v3", 131_072),        # 128K
    ("deepseek", 131_072),           # 兜底：保守，不假设是 V4
    ("hunyuan", 262_144),
    ("gpt-3.5", 16_384),
    ("gpt-4o", 131_072),
    ("gpt-4", 8_192),                # 老版本；须排在 "gpt-4o" 之后
    ("kimi", 131_072),               # 保守；K2-0905 起为 256K，用新版请显式配置
    ("hy3", 262_144),                # 腾讯混元 Hy3 —— 本项目基准
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
        # 表已按前缀长度降序排列，取首个命中即为最长前缀。
        for prefix, window in _MODEL_CONTEXT_WINDOWS:
            if name.startswith(prefix):
                return window, "model_table"
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
