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

from app.agent.agent import build_agent, format_tool_activity, run_turn
from app.agent.context import RunContext
from app.config import get_settings
from app.llm import build_model
from app.tools import (
    check_sensor_sanity,
    check_temporal_sync,
    compute_stats,
    generate_report,
    inspect_streams,
    load_dataset,
    plot_chart,
    profile_data,
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
]


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

    def _build_agent(self):
        settings = get_settings()
        model = build_model(settings)
        return build_agent(model, _ALL_TOOLS)

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
        final, self.history_input, result = await run_turn(
            self.agent, self.context, user_input, self.history_input
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
