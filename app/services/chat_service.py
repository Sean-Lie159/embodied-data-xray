"""对话编排服务（纯 Python，无 UI 依赖）。

封装"初始化 agent + 管理 RunContext + 执行单轮对话"为同步接口，供 Streamlit（或
未来 CLI / FastAPI）复用。不 import streamlit。

asyncio 衔接说明：
openai-agents 的 Runner 是异步接口（await），而 Streamlit 脚本是同步模型。本服务
在每个 ``reply()`` 调用内用 ``asyncio.run()`` 启动一次性事件循环执行单轮对话。
注意：若调用方自身已运行在事件循环中（如 Jupyter / FastAPI），``asyncio.run()``
会报错，需改用 ``areply()`` 异步方法。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from agents import RunResult
from agents.usage import Usage

from app.agent.agent import (
    build_agent,
    configure_history_compaction,
    format_tool_activity,
    guard_tools,
    run_turn,
)
from app.agent.history_compaction import (
    _split_turns,
    compact_history,
    estimate_history_tokens,
)
from app.agent.context import RunContext
from app.config import get_settings
from app.llm import build_model
from app.llm.context_window import derive_budget
from app.tools import (
    check_sensor_sanity,
    check_temporal_sync,
    compute_stats,
    generate_report,
    inspect_streams,
    load_dataset,
    plot_chart,
    profile_data,
    propose_stream_semantics,
)

# 注册给 agent 的全部工具。
_ALL_TOOLS = [
    load_dataset,
    profile_data,
    inspect_streams,
    check_temporal_sync,
    check_sensor_sanity,
    compute_stats,
    plot_chart,
    generate_report,
    propose_stream_semantics,
]


def _compose_user_input(user_input: str, pending_notes: list[str]) -> str:
    """把待投递的面板事件便签与用户本轮输入合并为模型输入。

    便签以"[侧栏面板事件]"前缀逐条列出后接原问题——模型由此得知"数据集
    已由面板加载成功"，直接基于该数据集回答，不再反向索要路径。便签为空
    时输入原样返回（CLI 等无面板场景不受影响）。

    Args:
        user_input: 用户本轮原始输入。
        pending_notes: 面板事件便签列表（按登记顺序）。

    Returns:
        合并后的本轮输入文本。
    """
    if not pending_notes:
        return user_input
    notes = "\n".join(f"- {n}" for n in pending_notes)
    return (
        "[侧栏面板事件] 以下状态变化已发生（工具执行成功，非用户手输路径）：\n"
        f"{notes}\n\n"
        "请基于上述已加载数据直接回答，无需再向用户索要数据集路径。\n\n"
        f"用户问题：{user_input}"
    )


@dataclass
class ChatTurn:
    """单轮对话的返回结果。"""

    reply: str  # 最终回复文本
    tool_activity: str  # 工具调用轨迹（可折叠展示）
    tool_calls: list[str] = field(default_factory=list)  # 本轮调用的工具名列表
    findings: list[dict] = field(default_factory=list)  # 截止本轮的最新 findings
    usage: dict[str, int] | None = None  # 本轮 token 用量（input/output/total），获取不到为 None


def extract_usage(result: RunResult | None) -> dict[str, int] | None:
    """从 RunResult 提取本轮 token 用量（input/output/total）。

    真实 SDK 结构：``RunResult`` 不暴露公开的 ``context_wrapper`` 属性，usage 位于
    ``RunContextWrapper.usage``（``result.to_state()._context.usage``），``Usage`` 含
    ``input_tokens / output_tokens / total_tokens``，且为整轮（含全部内部模型调用）的
    累计值。result 为 None（MaxTurnsExceeded 等）或结构不符时返回 None，不抛异常。

    Args:
        result: 单轮运行的完整结果；可能为 None。

    Returns:
        dict{input_tokens, output_tokens, total_tokens}；获取不到返回 None。
    """
    if result is None:
        return None
    try:
        state = result.to_state()
        wrapper = getattr(state, "_context", None)
        usage: Usage | None = getattr(wrapper, "usage", None) if wrapper is not None else None
        if usage is None:
            return None
        return {
            "input_tokens": int(usage.input_tokens or 0),
            "output_tokens": int(usage.output_tokens or 0),
            "total_tokens": int(usage.total_tokens or 0),
        }
    except Exception:  # noqa: BLE001 - 结构不符时安全降级，不中断对话
        return None


def _count_turns(history: list[Any]) -> int:
    """统计历史中的轮数（按 user 消息切分）。"""
    return len(_split_turns(history)) if history else 0


class ChatService:
    """管理 agent、RunContext 与对话历史的对话服务。

    用法（同步，适合 Streamlit）：
        service = ChatService()
        turn = service.reply("请加载数据集 data/xxx")
        print(turn.reply)
    """

    def __init__(self) -> None:
        self.agent = self._build_agent()
        self.context = RunContext()
        self.history_input: list[Any] | None = None
        # 面板事件便签：UI 侧栏加载等状态变化，下一轮 reply 时拼入模型输入
        # （工具层状态与模型认知的桥梁，见 add_context_note docstring）。
        self._pending_notes: list[str] = []
        # 注入历史压缩参数（自动压缩开关在配置中控制）。
        self._configure_compaction()

    def _build_agent(self):
        settings = get_settings()
        model = build_model(settings)
        # 套上工具返回体积护栏（第 2 层防御）：单工具返回绝不超预算。
        budget = derive_budget(
            settings.default_model,
            configured_window=settings.context_window_tokens,
            budget_ratio=settings.context_budget_ratio,
            history_ratio=settings.history_budget_ratio,
            tool_output_ratio=settings.tool_output_budget_ratio,
        )
        return build_agent(
            model, guard_tools(_ALL_TOOLS, budget_tokens=budget.tool_output_budget)
        )

    def _configure_compaction(self) -> None:
        """按配置注入历史压缩参数（自动压缩开关在此生效）。"""
        settings = get_settings()
        budget = derive_budget(
            settings.default_model,
            configured_window=settings.context_window_tokens,
            budget_ratio=settings.context_budget_ratio,
            history_ratio=settings.history_budget_ratio,
            tool_output_ratio=settings.tool_output_budget_ratio,
        )
        configure_history_compaction(
            budget_tokens=(
                budget.history_budget if settings.history_compaction_enabled else 0
            ),
            keep_recent_turns=settings.history_keep_recent_turns,
        )
        self._compaction_settings = settings

    def history_stats(self) -> dict[str, Any]:
        """返回当前历史的体积统计（供 UI/CLI 展示）。

        Returns:
            dict，含 turns（轮数）、estimated_tokens（估算 token）、
            last_compaction（最近一次压缩统计或 None）。
        """
        history = self.history_input or []
        return {
            "turns": _count_turns(history),
            "estimated_tokens": estimate_history_tokens(history),
            "last_compaction": self.context.last_compaction,
        }

    def compact_now(self) -> dict[str, Any]:
        """手动压缩历史（不经过阈值判断，立即执行）。

        Returns:
            压缩统计（compact_history 的 stats；无历史时返回零值统计）。
        """
        if not self.history_input:
            return {
                "compacted_outputs": 0, "before_tokens": 0, "after_tokens": 0,
                "saved_tokens": 0, "total_turns": 0, "kept_turns": 0,
            }
        settings = getattr(self, "_compaction_settings", None)
        keep = settings.history_keep_recent_turns if settings else 3
        self.history_input, stats = compact_history(
            self.history_input, keep_recent_turns=keep
        )
        self.context.last_compaction = stats
        return stats

    def add_context_note(self, note: str) -> None:
        """登记一条"面板事件便签"，下一轮 reply 时拼入模型输入（随后清空）。

        背景（2026-09-07 真实缺陷）：侧栏数据面板加载成功后只更新了
        RunContext（工具层可见），而模型只读对话输入流——模型凭历史判断
        "没有任何已加载数据集"，反向要求用户给路径。工具状态与模型认知
        脱节，根因是面板事件从未进入模型输入。本方法让 UI 把这类"模型
        应知道的状态变化"投递进来，reply 时以前缀形式注入本轮输入。

        Args:
            note: 面板事件说明（中文，含数据集名/路径/来源，供模型引用）。
        """
        self._pending_notes.append(note)

    def reply(self, user_input: str) -> ChatTurn:
        """同步执行单轮对话（内部用 asyncio.run 启动事件循环）。

        Args:
            user_input: 用户本轮输入。

        Returns:
            ChatTurn：回复文本、工具轨迹、本轮工具名列表、最新 findings。
        """
        return asyncio.run(self.areply(user_input))

    async def areply(self, user_input: str) -> ChatTurn:
        """异步执行单轮对话（供已有事件循环的调用方使用）。"""
        composed = _compose_user_input(user_input, self._pending_notes)
        self._pending_notes.clear()
        final, self.history_input, result = await run_turn(
            self.agent, self.context, composed, self.history_input
        )
        tool_activity = format_tool_activity(result)
        tool_calls = _extract_tool_names(result)
        return ChatTurn(
            reply=final,
            tool_activity=tool_activity,
            tool_calls=tool_calls,
            findings=list(self.context.findings),
            usage=extract_usage(result),
        )

    def dataset_summary(self) -> dict[str, Any]:
        """返回当前数据集的能力标签与流清单摘要（供 UI 展示）。

        含 video_fps_by_file（视频文件 → fps，来自 ffprobe），供流清单表格展示
        视频帧率（否则视频流采样率显示"未知"，但 fps 实际可得）。
        """
        caps = self.context.meta.get("capabilities", {})
        streams = self.context.meta.get("streams", [])
        video_fps_by_file: dict[str, Any] = {}
        for v in self.context.meta.get("video_meta", []):
            fps = v.get("fps")
            if fps is not None:
                video_fps_by_file[v.get("file", "")] = fps
        return {
            "dataset_id": self.context.dataset_id,
            "capabilities": caps,
            "streams": streams,
            "guessed_type": self.context.meta.get("guessed_type"),
            "video_fps_by_file": video_fps_by_file,
        }


def _extract_tool_names(result: RunResult | None) -> list[str]:
    """从 RunResult 中提取本轮调用的工具名列表。

    Args:
        result: 单轮运行的完整结果；MaxTurnsExceeded 时 run_turn 返回 None。

    Returns:
        工具名列表；result 为 None 时返回空列表（不抛 AttributeError）。
    """
    if result is None:
        return []
    names: list[str] = []
    for item in result.new_items:
        it = getattr(item, "type", "") or ""
        if "tool_call" not in it.lower():
            continue
        name = None
        for attr in ("name", "tool_name", "function_name"):
            v = getattr(item, attr, None)
            if isinstance(v, str) and v.strip():
                name = v.strip()
                break
        if name:
            names.append(name)
    return names
