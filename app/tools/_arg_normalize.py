"""工具入参归一（模型传参防御层）。

## 为什么需要（真实事故，2026-09-22）

模型经 OpenAI 兼容接口调用工具时，会把 **JSON 的 `null` 传成字符串 `"null"`**
（上游序列化/网关透传的常见偏差）。工具拿到 `"null"` 后不会报错，而是
**当作一个合法的字符串参数去匹配**：

- ``check_temporal_sync(baseline_stream="null")`` → 拿 "null" 去匹配文件名子串
  → 必然找不到 → 返回 ``baseline_no_match`` → 模型误以为"参数没生效"而
  **重试整轮**（白耗一次工具循环 + 一次模型往返）。当时正值长思考链，
  这次多余重试加剧了 UI 卡死。

同类风险还有：本应是列表的参数被传成单个字符串（会被逐字符拆开）、
空字符串、`"undefined"`/`"none"` 等前端常见空值表示。

## 设计原则

这些值在**语义上等于"未指定"**，因此在**任何匹配/判断之前**归一为 None。
归一必须发生在工具实现的**入口**，而不是散落在各匹配点——后者必然遗漏。
"""

from __future__ import annotations

from typing import Any

# "空值"字面量：语义上等于"未指定"。
# 覆盖：JSON null 的各语言写法、前端常见空值、以及中文口语（"自动"/"无"）。
# 注意：**不包含** `"all"`/`"全部"`——它们表达"全部流"而非"未指定"，
# 语义不同（前者是明确要求全量，后者是缺省行为），不可混为一谈。
NULLISH_LITERALS: frozenset[str] = frozenset({
    "null", "none", "undefined", "nil", "", "-", "n/a", "na",
    "auto", "default", "自动", "无", "空",
})


def normalize_optional_str(value: Any) -> str | None:
    """把可能为"空值字面量"的字符串参数归一为 None。

    Args:
        value: 原始参数值。

    Returns:
        去掉首尾空白后的字符串；空值字面量或 None 返回 None（表示"未指定"）。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    s = value.strip()
    return None if s.lower() in NULLISH_LITERALS else s


def normalize_optional_list(value: Any) -> list[str] | None:
    """把列表参数归一，并容忍"误传单个字符串"的情形。

    **为什么不能直接把字符串当可迭代**：``["a.csv"]`` 与 ``"a.csv"`` 在
    Python 里都能迭代，后者会被**逐字符**处理（"a"、"."、"c"…），
    产生大量无意义的子串匹配。故此处把单字符串包成单元素列表。

    Args:
        value: 原始参数值（None / str / list / tuple）。

    Returns:
        归一后的非空字符串列表；空值或全为空值时返回 None。
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in NULLISH_LITERALS:
            return None
        return [s]
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
        items = [v for v in items if v.lower() not in NULLISH_LITERALS]
        return items or None
    return None


def normalize_optional_int(value: Any) -> int | None:
    """把可能为"空值字面量"的整数参数归一为 None（容忍数字字符串）。

    Args:
        value: 原始参数值。

    Returns:
        整数；空值/无法解析时返回 None。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if s.lower() in NULLISH_LITERALS:
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None
