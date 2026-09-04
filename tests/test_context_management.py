"""上下文管理"手动入口 + 纪律"的单元测试（第 4 层）。

覆盖：SYSTEM_PROMPT 第 11 条截断纪律、RunContext.last_compaction、
ChatService 的 history_stats / compact_now、CLI 指令常量、以及 CLI 与
ChatService 两条路径都接入了护栏（防"CLI 绕过防御"的回归）。
"""

from __future__ import annotations

import json

from app.agent.agent import SYSTEM_PROMPT
from app.agent.context import RunContext
from app.agent.history_compaction import compact_history, estimate_history_tokens


# --- 1. SYSTEM_PROMPT 纪律 --------------------------------------------------


def test_prompt_has_truncation_discipline() -> None:
    """第 11 条：工具返回被截断时必须主动告知，不得当完整结论陈述。"""
    assert "结果截断纪律" in SYSTEM_PROMPT
    assert "truncated" in SYSTEM_PROMPT
    assert "truncation_note" in SYSTEM_PROMPT
    assert "不得把截断后的部分结果当作完整结论陈述" in SYSTEM_PROMPT


def test_prompt_mentions_compaction_notice() -> None:
    """历史压缩说明的识别标记须在 prompt 中给出（否则模型不知如何解释）。"""
    assert "[上下文管理]" in SYSTEM_PROMPT


def test_prompt_tells_how_to_recover_from_compaction() -> None:
    """压缩后引用旧数字的规则：不可考时不得凭印象补数字。"""
    assert "已随历史压缩省略" in SYSTEM_PROMPT


def test_prompt_suggests_batching_when_truncated() -> None:
    """截断时应建议分批查看或缩小范围。"""
    assert "分批查看" in SYSTEM_PROMPT or "缩小范围" in SYSTEM_PROMPT


# --- 2. RunContext ----------------------------------------------------------


def test_run_context_has_last_compaction_field() -> None:
    """RunContext 新增 last_compaction 字段（默认 None）。"""
    ctx = RunContext()
    assert hasattr(ctx, "last_compaction")
    assert ctx.last_compaction is None


# --- 3. ChatService 手动接口 ------------------------------------------------


def _fake_service() -> object:
    """构造一个不连真实模型的 ChatService 实例（跳过 agent 构建）。"""
    from app.services.chat_service import ChatService

    svc = ChatService.__new__(ChatService)  # 不触发 __init__（避免连模型）
    svc.context = RunContext()
    svc.history_input = None
    return svc


def test_history_stats_empty() -> None:
    """无历史时统计为 0，不崩。"""
    svc = _fake_service()
    stats = svc.history_stats()
    assert stats["turns"] == 0
    assert stats["estimated_tokens"] == 0
    assert stats["last_compaction"] is None


def test_history_stats_counts_turns() -> None:
    """统计轮数与体积。"""
    svc = _fake_service()
    svc.history_input = [
        {"role": "user", "content": "q1"},
        {"type": "function_call_output", "call_id": "c1",
         "output": json.dumps({"a": "x" * 100}, ensure_ascii=False)},
        {"role": "user", "content": "q2"},
    ]
    stats = svc.history_stats()
    assert stats["turns"] == 2
    assert stats["estimated_tokens"] > 0


def test_compact_now_reduces_and_records() -> None:
    """手动压缩：体积下降，并写入 last_compaction。"""
    svc = _fake_service()
    big = json.dumps({"success": True,
                      "detail": [{"i": i, "pad": "y" * 80} for i in range(200)]},
                     ensure_ascii=False)
    history = []
    for i in range(4):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"type": "function_call_output",
                        "call_id": f"c{i}", "output": big})
    svc.history_input = history

    before = estimate_history_tokens(history)
    stats = svc.compact_now()

    assert stats["compacted_outputs"] > 0
    assert stats["saved_tokens"] > 0
    assert svc.context.last_compaction is not None
    assert estimate_history_tokens(svc.history_input) < before


def test_compact_now_on_empty_history_is_safe() -> None:
    """空历史手动压缩不崩，返回零值统计。"""
    svc = _fake_service()
    stats = svc.compact_now()
    assert stats["compacted_outputs"] == 0
    assert stats["saved_tokens"] == 0


def test_compact_now_keeps_user_messages() -> None:
    """手动压缩同样不得丢用户提问（与自动压缩同一不变量）。"""
    svc = _fake_service()
    big = json.dumps({"success": True, "detail": ["z" * 80] * 100},
                     ensure_ascii=False)
    history = []
    for i in range(4):
        history.append({"role": "user", "content": f"问题{i}"})
        history.append({"type": "function_call_output",
                        "call_id": f"c{i}", "output": big})
    svc.history_input = history
    svc.compact_now()

    users = [
        i["content"] for i in svc.history_input
        if i.get("role") == "user" and not str(i["content"]).startswith("[上下文管理]")
    ]
    assert users == [f"问题{i}" for i in range(4)], "用户提问一条不丢"


# --- 4. CLI 指令与两条路径都接护栏 ------------------------------------------


def test_cli_compaction_commands_defined() -> None:
    """CLI 的压缩/查看指令常量存在且可用。"""
    import main as cli

    assert hasattr(cli, "_COMPACT_COMMANDS")
    assert "/compact" in cli._COMPACT_COMMANDS
    assert hasattr(cli, "_HISTORY_COMMANDS")
    assert "/history" in cli._HISTORY_COMMANDS
    # 指令不得与退出指令冲突
    assert not (set(cli._COMPACT_COMMANDS) & set(cli._EXIT_COMMANDS))


def test_cli_builds_agent_with_guard() -> None:
    """CLI 自行组装工具（不经 chat_service），必须同样接入护栏与压缩配置。

    这是关键回归：CLI 路径若漏接，则既无返回护栏也无历史压缩。
    """
    import main as cli

    src = cli.__file__
    with open(src, encoding="utf-8") as f:
        code = f.read()
    assert "guard_tools" in code, "CLI 必须套上工具返回护栏"
    assert "configure_history_compaction" in code, "CLI 必须注入历史压缩配置"
