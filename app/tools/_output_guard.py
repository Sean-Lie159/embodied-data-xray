"""工具返回体积护栏（防单次返回撑爆上下文）。

为什么需要：工具返回中的无界字段（流明细、逐列统计、逐 episode 明细）在数据集
大时可单次即超过模型上下文上限，直接导致 ``input length too long (HTTP 400)``。

设计沿用项目既有护栏 ``load_dataset._compress_file_survey`` 的成功模式，并推广为
通用能力：

- **渐进降级**：先砍次要细节，达标即停，不是一刀切；
- **计数守恒**：任何降级都保留总数与已展示数，绝不静默抽样；
- **失败兜底**：序列化异常时视为超限（fail-safe，强制压缩）；
- **可恢复指引**：note 写明"缩小范围 / 指定表名 / 分批查看"等自救路径；
- **显式标注**：发生降级必带 ``truncated: True`` 与 ``truncation_note``。

三档降级（顺序执行，达标即停）：
    档 1 丢弃次要字段   —— 按 droppable 顺序逐个丢弃（各工具自行声明）
    档 2 截断长列表     —— 列表保留前 N 条，但**保留完整计数**
    档 3 压为结论摘要   —— 极端情况只留关键结论字段与计数
"""

from __future__ import annotations

import copy
import json
from typing import Any

from app.llm.context_window import estimate_tokens

# 档 2（长列表截断）每条列表最多保留的条目数。
_MAX_LIST_ITEMS = 50

# 档 3（结论摘要）保留的字段白名单：这些是"结论"，任何情况下都必须留下。
_CONCLUSION_KEYS: tuple[str, ...] = (
    "success", "error", "reason", "user_message", "dataset", "dataset_id",
    "result", "n_rows", "n_cols", "table_name", "main_table", "file_path",
    "check", "measurements", "thresholds", "affected_episodes",
    "suggested_tools", "supported_formats",
)

# 档 3 保留的嵌套字段（位于 measurements 等容器内）：判定的核心数字。
_CONCLUSION_NESTED_KEYS: tuple[str, ...] = (
    "status", "result", "value", "ratio", "count", "rate", "max", "min",
    "mean", "median", "note", "reason",
)


def measure_tokens(obj: Any) -> int:
    """测量对象序列化后的估算 token 数。

    序列化失败时返回 ``sys.maxsize``（fail-safe），使调用方必然走压缩分支——
    宁可给出降级结果，也不能把测不出体积的内容原样送进上下文。

    Args:
        obj: 任意可 JSON 序列化的对象。

    Returns:
        估算 token 数；序列化失败返回极大值。
    """
    import sys

    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return sys.maxsize
    return estimate_tokens(text)


def _measure_data(obj: Any) -> int:
    """测量**数据体积**（排除护栏自身追加的 truncation_note 元信息）。

    为什么排除 note：note 是护栏加的说明文本（固定几十 token）。若计入达标判定，
    会出现"加了说明反而超预算"的降级螺旋——本已达标的结果因 note 超限而被迫
    多降一档，信息损失更大却只为容纳一句说明。

    Args:
        obj: 任意对象。

    Returns:
        数据部分的估算 token 数。
    """
    if isinstance(obj, dict) and "truncation_note" in obj:
        return measure_tokens({k: v for k, v in obj.items() if k != "truncation_note"})
    return measure_tokens(obj)


def _set_note(result: dict, note: str) -> None:
    """把降级说明写入 result（累积多条，不覆盖）。"""
    existing = result.get("truncation_note")
    if existing:
        result["truncation_note"] = f"{existing}；{note}"
    else:
        result["truncation_note"] = note
    result["truncated"] = True


def _truncate_long_lists(
    obj: Any, max_items: int = _MAX_LIST_ITEMS, depth: int = 0,
) -> tuple[Any, int]:
    """递归截断过长的列表（保留前 max_items 条 + 完整计数）。

    Args:
        obj: 任意对象。
        max_items: 每条列表最多保留的条目数（档 2 会自适应下调此值）。
        depth: 当前递归深度（防止过深递归）。

    Returns:
        (处理后的对象, 被截断的条目总数)。
    """
    dropped = 0
    if depth > 6:  # 防御：结构过深时不再深入
        return obj, 0
    if isinstance(obj, list):
        if len(obj) > max_items:
            dropped += len(obj) - max_items
            kept = [
                _truncate_long_lists(v, max_items, depth + 1)[0]
                for v in obj[:max_items]
            ]
            return {
                "items": kept,
                "total": len(obj),
                "shown": max_items,
                "truncated": True,
            }, dropped
        return [_truncate_long_lists(v, max_items, depth + 1)[0] for v in obj], 0
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            new_v, d = _truncate_long_lists(v, max_items, depth + 1)
            dropped += d
            out[k] = new_v
        return out, dropped
    return obj, dropped


def _adaptive_trim(out: dict, budget_tokens: int) -> tuple[dict, int]:
    """档 2 自适应截断：从宽到紧尝试保留条数，取**能达标的最大条数**。

    为什么自适应：固定截到 _MAX_LIST_ITEMS 在预算紧时可能仍超标而被迫跳到
    档 3（只剩结论），信息损失过大。递减尝试可"尽可能多给信息"（用户要求），
    且达标即停。

    Args:
        out: 当前结果 dict。
        budget_tokens: token 预算。

    Returns:
        (处理后的 dict, 被截断的条目总数)。
    """
    last: tuple[Any, int] = (out, 0)
    for limit in (_MAX_LIST_ITEMS, 20, 10, 5, 2, 1):
        trimmed, dropped = _truncate_long_lists(out, max_items=limit)
        last = (trimmed if isinstance(trimmed, dict) else out, dropped)
        if _measure_data(trimmed) <= budget_tokens:
            return last
    return last


def _to_conclusion_only(result: dict) -> dict:
    """档 3：压缩为结论摘要（保留白名单字段与计数）。"""
    out: dict[str, Any] = {}
    for key in _CONCLUSION_KEYS:
        if key not in result:
            continue
        value = result[key]
        # measurements 等容器：只留嵌套结论键，避免长序列。
        if isinstance(value, dict):
            filtered = {
                k: v for k, v in value.items()
                if k in _CONCLUSION_NESTED_KEYS or not isinstance(v, (list, dict))
            }
            out[key] = filtered
        elif isinstance(value, list) and len(value) > _MAX_LIST_ITEMS:
            out[key] = {
                "items": value[:_MAX_LIST_ITEMS],
                "total": len(value),
                "shown": _MAX_LIST_ITEMS,
                "truncated": True,
            }
        else:
            out[key] = value
    return out


def enforce_output_limit(
    result: Any,
    budget_tokens: int,
    *,
    droppable: tuple[str, ...] = (),
    tool_name: str = "",
) -> Any:
    """把工具返回压到预算内的渐进降级（三档，达标即停）。

    Args:
        result: 工具返回的 dict（非 dict 原样返回——护栏只处理结构化返回）。
        budget_tokens: token 预算（来自 derive_budget().tool_output_budget）。
        droppable: 档 1 的可丢弃字段名，按"优先丢弃"顺序排列。
        tool_name: 工具名（用于 note 文案）。

    Returns:
        降级后的 dict；未超预算时**原样返回**（不改动、不加字段）。
    """
    if not isinstance(result, dict) or budget_tokens <= 0:
        return result
    if _measure_data(result) <= budget_tokens:
        return result  # 未超限：不动原值，不加噪声字段

    # 深拷贝后再改，避免污染调用方持有的原对象。
    out = copy.deepcopy(result)
    name = tool_name or "该工具"

    # 档 1：按 droppable 顺序逐个丢弃次要字段（保留计数，不静默删细节）。
    dropped_fields: list[str] = []
    for field in droppable:
        if _measure_data(out) <= budget_tokens:
            break
        if field in out:
            out.pop(field)
            dropped_fields.append(field)
    if dropped_fields:
        _set_note(
            out,
            f"因体积超限已省略字段：{'、'.join(dropped_fields)}"
            "（结论与计数完整；如需完整信息请缩小范围后重试）",
        )
    if _measure_data(out) <= budget_tokens:
        return out

    # 档 2：自适应截断长列表（保留完整计数，尽可能多给信息）。
    out, dropped_items = _adaptive_trim(out, budget_tokens)
    if dropped_items:
        _set_note(
            out,
            f"因体积超限已截断长列表（共省略 {dropped_items} 条明细，"
            "各列表的 total 仍为完整总数）",
        )
    if _measure_data(out) <= budget_tokens:
        return out

    # 档 3：压为结论摘要（最后防线，保证绝不超限）。
    out = _to_conclusion_only(out)
    _set_note(
        out,
        f"{name} 的返回因体积过大已压缩为结论摘要：仅保留判定与关键数字，"
        "明细已省略；如需完整结果，请缩小范围（指定表名/流子集/时间范围）后重试。",
    )
    if _measure_data(out) > budget_tokens:
        # 极端兜底：连结论都超标时，只留最小可用信息（宁可少，不可爆）。
        minimal = {
            "success": out.get("success"),
            "error": "output_too_large",
            "reason": "工具返回体积超出上下文预算，已压缩至最小信息",
            "user_message": (
                f"{name} 的返回结果体积超出当前上下文预算，已省略全部明细。"
                "请缩小分析范围（指定表名、流子集或时间范围）后重试。"
            ),
            "truncated": True,
            # 任何降级都必须带说明（纪律：绝不静默失真）。
            "truncation_note": (
                f"{name} 返回体积超出预算，已压缩至最小信息（仅保留成功/失败与提示）；"
                "如需完整结果请缩小分析范围后重试。"
            ),
        }
        return minimal
    return out
