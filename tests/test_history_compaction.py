"""对话历史压缩（history_compaction）的单元测试。

覆盖核心不变量：
  - **user/assistant 文本一条不丢**（用户意图与 agent 结论）
  - 最近 N 轮的完整工具返回被保留（连贯性）
  - 旧轮次的 function_call_output 被替换为摘要，体积显著下降
  - 压缩统计字段正确；空历史/单轮历史不误压
"""

from __future__ import annotations

import json

from app.agent.history_compaction import (
    _split_turns,
    compact_history,
    estimate_history_tokens,
)


def _user(text: str = "分析一下") -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str = "好的") -> dict:
    return {"role": "assistant", "content": text}


def _call(name: str = "load_dataset", args: str = '{"path":"x"}') -> dict:
    return {
        "type": "function_call", "call_id": f"c_{name}", "name": name,
        "arguments": args,
    }


def _output(payload: dict) -> dict:
    return {
        "type": "function_call_output", "call_id": "c_x",
        "output": json.dumps(payload, ensure_ascii=False),
    }


def _big_output(rows: int = 200) -> dict:
    """构造体积很大的工具返回（模拟真实目录加载）。"""
    return _output({
        "success": True,
        "dataset": "demo",
        "n_rows": rows,
        "result": "pass",
        "user_message": "已加载数据集，共 %d 行" % rows,
        "streams": [{"path": f"stream_{i}.csv", "kind": "imu",
                     "measured_rate": 100.0 + i} for i in range(rows)],
    })


def _build_history(turns: int = 5) -> list[dict]:
    """构造多轮历史：每轮 = 1 user + 1 call + 1 大 output + 1 assistant。"""
    h: list[dict] = []
    for i in range(turns):
        h.append(_user(f"第 {i} 轮的问题"))
        h.append(_call(f"tool_{i}"))
        h.append(_big_output())
        h.append(_assistant(f"第 {i} 轮的回答"))
    return h


# --- 1. 切分与估算 ----------------------------------------------------------


def test_split_turns_by_user_message() -> None:
    """按 user 消息切分轮次。"""
    h = [_user("a"), _call(), _output({"x": 1}), _user("b"), _call()]
    turns = _split_turns(h)
    assert len(turns) == 2
    assert turns[0][0]["content"] == "a"
    assert turns[1][0]["content"] == "b"


def test_split_turns_empty() -> None:
    assert _split_turns([]) == []


def test_estimate_history_tokens_monotonic() -> None:
    """历史越长，估算 token 越多（基本单调性）。"""
    a = estimate_history_tokens([_user("短")])
    b = estimate_history_tokens([_user("短"), _big_output()])
    assert b > a > 0


def test_estimate_empty_history() -> None:
    assert estimate_history_tokens([]) == 0


# --- 2. 核心不变量：不丢结论 ------------------------------------------------


def test_user_and_assistant_messages_never_dropped() -> None:
    """压缩后 user/assistant 文本一条不丢（关键不变量）。"""
    h = _build_history(turns=5)
    original_users = [i["content"] for i in h if i.get("role") == "user"]
    original_assistants = [i["content"] for i in h if i.get("role") == "assistant"]

    compressed, stats = compact_history(h, keep_recent_turns=2)

    assert stats["compacted_outputs"] > 0, "应当发生了压缩"
    # 比对时不计入压缩说明那条追加消息。
    users = [
        i["content"] for i in compressed
        if i.get("role") == "user" and not str(i["content"]).startswith("[上下文管理]")
    ]
    assistants = [i["content"] for i in compressed if i.get("role") == "assistant"]
    assert users == original_users, "用户提问不得丢失"
    assert assistants == original_assistants, "助手结论不得丢失"


def test_recent_turns_outputs_preserved() -> None:
    """最近 N 轮的工具返回保持原样（连贯性）。"""
    h = _build_history(turns=5)
    compressed, _ = compact_history(h, keep_recent_turns=2)
    outputs = [i for i in compressed if i.get("type") == "function_call_output"]
    # 最近 2 轮的返回必须完整（仍是可解析的原始 JSON，且含 streams 明细）
    intact = [
        o for o in outputs
        if '"streams"' in (o.get("output") or "")
    ]
    assert len(intact) == 2, f"最近 2 轮返回应完整保留，实际 {len(intact)}"


def test_old_outputs_replaced_by_summary() -> None:
    """旧轮次的返回被替换为摘要（含结论字段、不含明细）。"""
    h = _build_history(turns=5)
    compressed, stats = compact_history(h, keep_recent_turns=1)
    outputs = [i for i in compressed if i.get("type") == "function_call_output"]
    old = [o for o in outputs if "原返回已省略" in (o.get("output") or "")]
    assert len(old) == 4, f"前 4 轮应被压缩，实际 {len(old)}"
    # 摘要保留结论字段
    assert "success" in old[0]["output"]
    assert "重新调用该工具" in old[0]["output"]


# --- 3. 体积下降与统计 ------------------------------------------------------


def test_compaction_reduces_tokens() -> None:
    """压缩后体积显著下降，统计字段自洽。"""
    h = _build_history(turns=6)
    compressed, stats = compact_history(h, keep_recent_turns=2)
    assert stats["saved_tokens"] > 0
    assert stats["after_tokens"] < stats["before_tokens"]
    assert stats["after_tokens"] >= 0
    assert estimate_history_tokens(compressed) == stats["after_tokens"]


def test_stats_fields_present() -> None:
    """统计字段齐全（UI/CLI 依赖）。"""
    _, stats = compact_history(_build_history(turns=4), keep_recent_turns=1)
    for key in ("compacted_outputs", "before_tokens", "after_tokens",
                "saved_tokens", "total_turns", "kept_turns"):
        assert key in stats, f"统计缺字段 {key}"
    assert stats["total_turns"] == 4
    assert stats["kept_turns"] == 1


def test_compaction_notice_appended() -> None:
    """压缩后追加可见说明（绝不静默）。"""
    compressed, _ = compact_history(_build_history(turns=4), keep_recent_turns=1)
    notices = [
        i for i in compressed
        if i.get("role") == "user"
        and str(i["content"]).startswith("[上下文管理]")
    ]
    assert len(notices) == 1
    assert "已把前" in notices[0]["content"]
    assert "重新调用对应工具" in notices[0]["content"]


# --- 4. 边界 ----------------------------------------------------------------


def test_empty_history_no_crash() -> None:
    compressed, stats = compact_history([], keep_recent_turns=3)
    assert compressed == []
    assert stats["compacted_outputs"] == 0


def test_single_turn_not_compacted() -> None:
    """单轮历史不压缩（keep>=1 保证当前轮完整）。"""
    h = _build_history(turns=1)
    compressed, stats = compact_history(h, keep_recent_turns=3)
    assert stats["compacted_outputs"] == 0
    # 内容应完全不变（不追加说明）
    assert len(compressed) == len(h)


def test_keep_recent_turns_at_least_one() -> None:
    """keep_recent_turns=0 时至少保留 1 轮，避免把当前轮也压掉。"""
    h = _build_history(turns=3)
    compressed, stats = compact_history(h, keep_recent_turns=0)
    assert stats["kept_turns"] >= 1
    intact = [
        o for o in compressed
        if o.get("type") == "function_call_output"
        and '"streams"' in (o.get("output") or "")
    ]
    assert len(intact) >= 1


def test_non_json_output_degrades_gracefully() -> None:
    """非 JSON 的工具返回：退化为截断而非崩溃。"""
    h = [
        _user("q1"), _call(), {"type": "function_call_output",
                               "call_id": "c1", "output": "纯文本返回" * 100},
        _user("q2"), _call(), {"type": "function_call_output",
                               "call_id": "c2", "output": "另一个文本" * 50},
    ]
    compressed, stats = compact_history(h, keep_recent_turns=1)
    assert stats["compacted_outputs"] == 1
    old = [i for i in compressed if i.get("call_id") == "c1"][0]
    assert "已省略" in old["output"] or len(old["output"]) <= 200


def test_summary_keeps_conclusion_fields() -> None:
    """摘要保留关键结论字段（success/result/n_rows 等）。"""
    h = [
        _user("q1"), _call(), _output({
            "success": True, "result": "fail", "n_rows": 999,
            "dataset": "demo", "user_message": "判定为失败",
            "streams": [{"a": i} for i in range(100)],
        }),
        _user("q2"), _call(), _output({"success": True}),
    ]
    compressed, _ = compact_history(h, keep_recent_turns=1)
    old = [i for i in compressed if i.get("type") == "function_call_output"][0]
    out_text = old["output"]
    assert "success" in out_text
    assert "999" in out_text or "n_rows" in out_text
