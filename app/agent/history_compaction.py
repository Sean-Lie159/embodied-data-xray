"""对话历史压缩（第 3 层防御，治"历史无界累积"根因）。

背景：``app/agent/agent.py`` 每轮以 ``next_input = result.to_input_list()`` 累积
历史，递推公式为 ``H_n = H_{n-1} + [本轮用户消息] + 本轮全部工具调用与返回``，
SDK 不做任何裁剪——每轮都把此前所有轮次的工具返回原文重发一遍。一次目录加载
约数十 KB，固化后每轮重发，几轮即撞 ``input length too long (HTTP 400)``。

压缩策略（**只丢可再生的工具返回原文，绝不丢结论**）：

    保留（一条不丢）：
      - 全部 user / assistant 文本消息（用户原始意图与 agent 结论）
      - 最近 keep_recent_turns 轮的完整工具调用与返回（保持连贯）
      - function_call 调用参数（体积很小，且便于溯源"当时调了什么"）
    压缩（仅旧轮次）：
      - function_call_output（工具返回原文）→ 结构化摘要：
        工具名 + 关键结论字段（success/result/n_rows/main_table/判定等）

为什么压缩代价低（本项目天然优势）：数据始终留在 Python 进程内
（``RunContext.df`` / ``meta`` / ``findings``），压缩掉的只是"发给模型看的历史
副本"。agent 需要细节时**重新调工具即可取回**。

绝不静默：压缩后向历史追加一条说明，告知用户与模型"哪些细节已被省略"。
"""

from __future__ import annotations

import json
from typing import Any

from app.llm.context_window import estimate_tokens

# 摘要保留的结论字段白名单（这些是"结论"，必须留下）。
_CONCLUSION_KEYS: tuple[str, ...] = (
    "success", "error", "result", "dataset", "dataset_id", "n_rows", "n_cols",
    "table_name", "main_table", "file_path", "check", "user_message",
    "suggested_tools", "supported_formats", "truncated", "truncation_note",
)

# 摘要中 user_message 最多保留的字符数（避免长提示撑爆摘要）。
_USER_MESSAGE_MAX_CHARS = 120

# 压缩说明（追加到历史末尾，对模型与用户均可见）。
_COMPACTION_NOTICE_PREFIX = "[上下文管理]"


def estimate_history_tokens(history: list[Any]) -> int:
    """估算历史的 token 总量。

    Args:
        history: input item 列表。

    Returns:
        估算 token 数；空历史返回 0。
    """
    if not history:
        return 0
    total = 0
    for item in history:
        total += _item_tokens(item)
    return total


def _item_tokens(item: Any) -> int:
    """估算单个历史 item 的 token 数。"""
    if isinstance(item, dict):
        # 工具返回原文是体积主体，直接测 output/arguments 等大字段。
        parts: list[str] = []
        for key in ("content", "output", "arguments"):
            value = item.get(key)
            if value is None:
                continue
            parts.append(value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, default=str))
        if not parts:
            parts.append(json.dumps(item, ensure_ascii=False, default=str))
        return estimate_tokens(" ".join(parts))
    return estimate_tokens(str(item))


def _is_user_turn_start(item: Any) -> bool:
    """判断 item 是否为新一轮的起点（user 消息）。"""
    return isinstance(item, dict) and item.get("role") == "user"


def _split_turns(history: list[Any]) -> list[list[Any]]:
    """按 user 消息把历史切分为轮次。

    首个 user 之前的 item（如 system/历史遗留）归入第 0 轮。

    Args:
        history: input item 列表。

    Returns:
        轮次列表（每轮是 item 列表）。
    """
    turns: list[list[Any]] = []
    current: list[Any] = []
    for item in history:
        if _is_user_turn_start(item) and current:
            turns.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        turns.append(current)
    return turns


def _summarize_output(output: str) -> str:
    """把工具返回原文压缩为结构化摘要。

    Args:
        output: 工具返回的原始字符串（通常为 JSON）。

    Returns:
        摘要文本；非 JSON 或解析失败时退化为截断的原文本。
    """
    raw = output if isinstance(output, str) else str(output)
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        # 非 JSON：退化为截断（保留首尾，便于识别内容）。
        if len(raw) <= 200:
            return raw
        return f"{raw[:150]}…（已省略 {len(raw) - 200} 字符）"

    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)[:200]

    parts: list[str] = []
    for key in _CONCLUSION_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if key == "user_message" and isinstance(value, str):
            value = value[:_USER_MESSAGE_MAX_CHARS] + (
                "…" if len(value) > _USER_MESSAGE_MAX_CHARS else ""
            )
        parts.append(f"{key}={json.dumps(value, ensure_ascii=False, default=str)}")
    # measurements 常含判定数字，纳入摘要（只取标量项）。
    measurements = obj.get("measurements")
    if isinstance(measurements, dict):
        scalars = {
            k: v for k, v in measurements.items()
            if isinstance(v, (int, float, str, bool))
        }
        if scalars:
            parts.append(f"measurements={json.dumps(scalars, ensure_ascii=False)}")

    summary = "；".join(parts) if parts else f"keys={list(obj.keys())[:10]}"
    omitted = estimate_tokens(raw) - estimate_tokens(summary)
    return (
        f"{summary}"
        f"（原返回已省略，约省 {max(0, omitted)} token；"
        "如需明细请重新调用该工具）"
    )


def compact_history(
    history: list[Any],
    *,
    keep_recent_turns: int = 3,
) -> tuple[list[Any], dict[str, Any]]:
    """压缩历史：把旧轮次的工具返回原文替换为结构化摘要。

    Args:
        history: input item 列表。
        keep_recent_turns: 保留最近若干轮的完整工具返回（保持连贯）。

    Returns:
        (压缩后历史, 压缩统计)。统计含 compacted_outputs（压缩条数）、
        before_tokens / after_tokens / saved_tokens、总轮数与保留轮数。
    """
    if not history:
        return list(history), {
            "compacted_outputs": 0, "before_tokens": 0, "after_tokens": 0,
            "saved_tokens": 0, "total_turns": 0, "kept_turns": 0,
        }

    before = estimate_history_tokens(history)
    turns = _split_turns(history)
    # 至少保留 1 轮，避免把当前轮也压掉。
    keep = max(1, min(int(keep_recent_turns), len(turns)))
    cut = len(turns) - keep  # 前 cut 轮为"旧轮次"，需要压缩

    compacted = 0
    out: list[Any] = []
    for idx, turn in enumerate(turns):
        if idx < cut:
            for item in turn:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    new_item = dict(item)
                    new_item["output"] = _summarize_output(item.get("output", ""))
                    out.append(new_item)
                    compacted += 1
                else:
                    out.append(item)
        else:
            out.extend(turn)

    after = estimate_history_tokens(out)
    stats: dict[str, Any] = {
        "compacted_outputs": compacted,
        "before_tokens": before,
        "after_tokens": after,
        "saved_tokens": max(0, before - after),
        "total_turns": len(turns),
        "kept_turns": keep,
    }

    if compacted:
        # 绝不静默：向历史追加一条压缩说明（对模型与用户均可见）。
        out.append({
            "role": "user",
            "content": (
                f"{_COMPACTION_NOTICE_PREFIX} 为控制上下文长度，已把前 {cut} 轮中 "
                f"{compacted} 条工具返回的明细压缩为结论摘要（节省约 "
                f"{stats['saved_tokens']} token）。用户提问与助手结论均完整保留；"
                "若需查看某项明细，请重新调用对应工具。"
            ),
        })
        after = estimate_history_tokens(out)
        stats["after_tokens"] = after
        stats["saved_tokens"] = max(0, before - after)

    return out, stats
