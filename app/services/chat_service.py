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
from typing import Any, Iterator

from agents import RunResult
from agents.usage import Usage

from app.agent.agent import (
    RunMetrics,
    build_agent,
    configure_history_compaction,
    format_tool_activity,
    guard_tools,
    run_turn,
    stream_turn,
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
from app.llm.factory import build_model_settings
from app.tools import (
    align_container_streams,
    check_sensor_sanity,
    compare_datasets,
    check_temporal_sync,
    compute_stats,
    generate_report,
    inspect_streams,
    inspect_video_frame,
    load_dataset,
    plot_chart,
    profile_data,
    propose_stream_semantics,
    unpack_mcap,
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
    unpack_mcap,
    align_container_streams,
    inspect_video_frame,
    compare_datasets,
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
    # 本轮观测指标（耗时 / 模型往返次数 / 工具调用次数 / 是否正常完成）。
    # 用于回答"这轮为什么慢/是不是工具循环太多"——502 类问题的第一手事实。
    metrics: dict[str, Any] | None = None
    # 流式轮的错误提示（未正常完成时非空）；非流式（reply）路径为 None。
    # 与 metrics["completed"] 配合，供 UI 渲染"本轮未完成"与重试入口。
    error: str | None = None
    # 过程时间线（思考摘要片段与工具播报按发生顺序交错）；供 UI **持久**展示
    # 过程（此前只有一个被覆写的占位符，留不住过程）。
    # 缺省空列表——**旧会话消息无此字段时必须容忍缺失**（向后兼容）。
    steps: list = field(default_factory=list)
    # 思考摘要全文（与 steps 中 reasoning 片段一致，便于整体取用/测试）；不可得为 ""。
    reasoning: str = ""


@dataclass
class StreamChunk:
    """流式输出的单个块（供 UI 消费）。

    kind 取值：
    - ``"delta"``：正文增量（text 为新增片段）；
    - ``"tool"``：工具调用播报（text 为功能描述）；
    - ``"reasoning"``：思考摘要增量（text 为新增片段）；
    - ``"final"``：收尾块（turn 为完整 ChatTurn，含 findings/usage/metrics/
      steps/reasoning）。
    """

    kind: str
    text: str = ""
    turn: "ChatTurn | None" = None
    # 流式进行中的过程快照（kind 为 tool/reasoning 时可选携带）：
    # 让 UI 能"边生成边累积渲染"过程时间线，而不是自己重新拼接。
    steps: "list | None" = None


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


def _new_session_tag() -> str:
    """生成短会话标识（用于输出文件名隔离，形如 "s-1a2b"）。"""
    import uuid

    return f"s-{uuid.uuid4().hex[:4]}"


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

    def __init__(self, session_tag: str | None = None) -> None:
        self.agent = self._build_agent()
        # 会话标识：UI 多会话时用于隔离输出文件名（缺省自动生成短标）；
        # 单会话场景可传 "" 关闭前缀（文件名与历史行为完全一致）。
        self.context = RunContext(
            session_tag=(session_tag if session_tag is not None
                         else _new_session_tag())
        )
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
        # 推理档位（REASONING_EFFORT）经 Agent 级 model_settings 注入；未配置时为
        # None，不干预网关默认值（零回归）。
        return build_agent(
            model,
            guard_tools(_ALL_TOOLS, budget_tokens=budget.tool_output_budget),
            model_settings=build_model_settings(settings),
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
        # **本会话**的压缩预算（不写全局——多会话各有各的预算）。
        self._history_budget: int = (
            budget.history_budget if settings.history_compaction_enabled else 0
        )
        self._keep_recent_turns: int = settings.history_keep_recent_turns
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
        # 观测指标容器：由 run_turn 回填（**含失败轮**——502 排查最需要失败轮的
        # 耗时与往返次数，这决定了是"工具循环太多"还是"服务侧单次抖动"）。
        metrics = RunMetrics()
        final, self.history_input, result = await run_turn(
            self.agent, self.context, composed, self.history_input,
            history_budget_tokens=self._history_budget,
            history_keep_recent_turns=self._keep_recent_turns,
            metrics=metrics,
        )
        tool_activity = format_tool_activity(result)
        tool_calls = _extract_tool_names(result)
        return ChatTurn(
            reply=final,
            tool_activity=tool_activity,
            tool_calls=tool_calls,
            findings=list(self.context.findings),
            usage=extract_usage(result),
            metrics={
                "duration_ms": metrics.duration_ms,
                "n_model_calls": metrics.n_model_calls,
                "n_tool_calls": metrics.n_tool_calls,
                "completed": metrics.completed,
                "error_kind": metrics.error_kind,
            },
            # 非流式路径：失败时最终正文本身即为友好提示，error 只做标记
            # （UI 据此渲染"本轮未完成"与重试入口）。
            error=None if metrics.completed else "本轮未正常完成。",
        )

    def reply_stream(self, user_input: str) -> "Iterator[StreamChunk]":
        """同步流式执行单轮对话：逐块 yield 正文增量/工具播报，最后 yield 完整结果。

        实现（docs/流式输出设计.md 3.2 方案甲）：在后台线程里新建事件循环消费
        :func:`stream_turn` 的异步生成器，经队列桥接回同步调用方——把 asyncio
        完整封在服务层内，UI 只面对同步生成器（与 reply() 的调用风格一致，
        UI 不碰 asyncio）。

        最后一个块 kind="final"，其 turn 为完整 :class:`ChatTurn`（含
        findings/usage/metrics），**与 reply() 返回的结构同构**，使 UI 侧的
        收尾逻辑（写 messages / 累计统计）可完全复用。

        Args:
            user_input: 用户本轮输入。

        Yields:
            StreamChunk：若干 delta/tool 块，最后一个 final 块。
        """
        import queue
        import threading

        composed = _compose_user_input(user_input, self._pending_notes)
        self._pending_notes.clear()
        metrics = RunMetrics()
        q: "queue.Queue[Any]" = queue.Queue()
        _DONE = object()

        async def _pump() -> None:
            try:
                async for ev in stream_turn(
                    self.agent, self.context, composed, self.history_input,
                    history_budget_tokens=self._history_budget,
                    history_keep_recent_turns=self._keep_recent_turns,
                    metrics=metrics,
                ):
                    q.put(ev)
            except BaseException as exc:  # noqa: BLE001 - 异常经队列送达消费端
                q.put(exc)
            finally:
                q.put(_DONE)

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_pump())
            finally:
                loop.close()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        done_event = None
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    # 兜底：_pump 自身异常（理论上 stream_turn 已吞）。转成
                    # 收尾事件，保留已产出的正文（见下方 acc 处理）。
                    if isinstance(item, (KeyboardInterrupt, SystemExit)):
                        raise item
                    from app.agent.agent import TurnEvent, _describe_model_error
                    done_event = TurnEvent(kind="done", final="",
                                           error=_describe_model_error(item),
                                           next_input=None)
                    break
                kind = getattr(item, "kind", "")
                if kind == "delta":
                    yield StreamChunk(kind="delta", text=item.text)
                elif kind == "tool":
                    yield StreamChunk(kind="tool", text=item.text)
                elif kind == "reasoning":
                    yield StreamChunk(kind="reasoning", text=item.text)
                elif kind == "done":
                    done_event = item
        finally:
            thread.join(timeout=5)

        turn = self._finalize_stream_turn(done_event, metrics)
        yield StreamChunk(kind="final", text=turn.reply, turn=turn,
                          steps=turn.steps)

    def _finalize_stream_turn(self, done_event, metrics: RunMetrics) -> ChatTurn:
        """把 stream_turn 的收尾事件整理为 ChatTurn（并写回历史）。"""
        from app.agent.agent import TurnEvent

        if done_event is None:
            done_event = TurnEvent(kind="done", final="", error="本轮未产生任何结果。")
        result = done_event.result
        # 历史写回：正常分支用 next_input；异常分支的 fallback_input 已含本轮
        # 用户消息（与 run_turn 一致，保证"直接重发"能接上）。
        if done_event.next_input is not None:
            self.history_input = done_event.next_input
        tool_activity = format_tool_activity(result) if result is not None else ""
        tool_calls = _extract_tool_names(result)
        return ChatTurn(
            reply=done_event.final,
            tool_activity=tool_activity,
            tool_calls=tool_calls,
            findings=list(self.context.findings),
            usage=extract_usage(result),
            metrics={
                "duration_ms": metrics.duration_ms,
                "n_model_calls": metrics.n_model_calls,
                "n_tool_calls": metrics.n_tool_calls,
                "completed": metrics.completed,
                "error_kind": metrics.error_kind,
            },
            error=done_event.error,
            steps=list(done_event.steps or []),
            reasoning=done_event.reasoning or "",
        )

    def truncate_history_to_turn(self, turn_index: int) -> dict[str, Any]:
        """把 agent 对话历史截断到第 turn_index 轮之前（编辑重发用）。

        消息编辑的语义核心：编辑第 n 条用户消息后，其**之后**的一切轮次
        （含该轮的旧回答与工具轨迹）必须从 agent 上下文中移除——否则 bot
        会"记得"已被编辑掉的旧内容，重新回答不干净。

        Args:
            turn_index: 目标用户消息在 messages 中的下标（0-based），
                也是 history_input 中的轮次序号（每轮由一条用户输入触发）。

        Returns:
            dict，含 before_turns / after_turns（截断前后轮数）。
        """
        history = self.history_input or []
        turns = _split_turns(history)
        if turn_index >= len(turns):
            return {"before_turns": len(turns), "after_turns": len(turns)}
        kept: list[Any] = []
        for turn in turns[:turn_index]:
            kept.extend(turn)
        self.history_input = kept
        return {"before_turns": len(turns), "after_turns": turn_index}

    def dataset_summary(self) -> dict[str, Any]:
        """返回当前数据集的状态摘要（供 UI 数据集状态面板展示）。

        含 video_fps_by_file（视频文件 → fps，来自 ffprobe），供流清单表格展示
        视频帧率（否则视频流采样率显示"未知"，但 fps 实际可得）。

        另含数据集状态面板所需字段（docs/数据集状态面板设计.md 2.2）：
        - ``main_table``：主表行/列数与截断信息（数据概况段）；
        - ``qc_state``：inspect_streams 回写的质检状态（语义确认进度段）；
        - ``qc``：check_temporal_sync 写的质量明细（单位告警/时钟矛盾段）。
        缺失时 UI 降级为"尚未检查"，**不误报为"无问题"**。
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
            "main_table": self.context.meta.get("main_table", {}),
            "qc_state": self.context.meta.get("qc_state"),
            "qc": self.context.meta.get("qc", {}),
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
