"""Streamlit 界面入口（唯一入口：streamlit run streamlit_app.py）。

布局：顶部**多会话标签页** + 左侧对话区 + 右侧展示区（图表 / Findings·报告 /
数据概况 tabs）。**每个会话独立**持有 ChatService（含 RunContext / 对话历史 /
数据集 / token 累计），可分析同一或不同数据集，互不干扰。

会话状态存于 st.session_state（刷新页面重置属正常，不持久化）；切换标签页不
丢状态、也不触发 agent 执行——只有新的用户输入才调用 agent。
"""

from __future__ import annotations

import streamlit as st

from app.config.settings import is_configured
from app.services.chat_service import ChatService, ChatTurn
from app.ui.components import (
    render_charts,
    render_dataset_overview,
    render_findings_and_report,
    render_token_stats,
    render_tool_activity,
)
from app.ui.constants import (
    CHAT_INPUT_RESERVE_PX,
    COLUMN_RATIO,
    SCROLL_HEIGHT,
    SESSION_LABEL_MAX_CHARS,
)
from app.ui.data_loader_panel import render_data_loader
from app.ui.onboarding import render_onboarding
from app.ui.settings_panel import render_model_settings

st.set_page_config(page_title="Embodied-data-Xray", page_icon="🩻", layout="wide")

# 左右栏可滚动容器高度（px）——集中在 app/ui/constants.py，见其依据注释。
_SCROLL_HEIGHT = SCROLL_HEIGHT


def _inject_scroll_css() -> None:
    """注入最小 CSS（仅 overflow/滚动锚定相关，不做自定义布局 hack）。

    目的：
    1. 让页面主体不产生页面级滚动（body overflow hidden），左右栏各自在
       ``st.container(height=...)`` 内独立滚动，避免"页面 + 容器"双重滚动条的别扭体验。
    2. 给聊天容器开启 ``overflow-anchor``（滚动锚定），使新消息到达时自动贴底，
       而不是把滚动位置留在旧消息处。

    注意（技术债）：本函数依赖 Streamlit 内部 DOM 与类名（``.block-container``、
    ``data-testid="stVerticalBlock"``），升级 Streamlit 时需重新验收布局
    （打开页面确认无双重滚动条、最后一条消息不被输入框遮挡）。
    """
    st.markdown(
        f"""
        <style>
        /* 页面主体不滚动：左右栏各自在固定高度容器内滚动，避免双重滚动条 */
        .block-container {{ overflow: hidden; }}
        /* 输入框钉底（fixed）会盖住容器底部内容：给主区底部预留输入框高度。
           像素值依据见 app/ui/constants.py 的 CHAT_INPUT_RESERVE_PX。 */
        .block-container {{ padding-bottom: {CHAT_INPUT_RESERVE_PX}px; }}
        /* 滚动锚定：聊天/面板容器内新内容追加时尽量保持贴底/原位置稳定 */
        [data-testid="stVerticalBlock"] > div {{
            overflow-anchor: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sessions() -> dict[str, dict]:
    """会话字典：{会话ID: {service, messages, cumulative, editing_index, name}}。

    首次访问时自动创建一个会话（保证始终有 active 会话）。
    """
    if "sessions" not in st.session_state or not st.session_state.sessions:
        st.session_state.sessions = {}
        _create_session()
    return st.session_state.sessions


def _create_session(name: str | None = None) -> str:
    """新建会话并返回其 ID（不切换 active）。"""
    from app.services.chat_service import _new_session_tag

    tag = _new_session_tag()
    st.session_state.sessions[tag] = {
        "service": ChatService(session_tag=tag),
        "messages": [],
        "cumulative": {"input_tokens": 0, "output_tokens": 0,
                       "total_tokens": 0, "rounds": 0,
                       "duration_ms": 0, "n_model_calls": 0},
        "editing_index": None,
        "name": name or f"对话 {len(st.session_state.sessions) + 1}",
    }
    if "active_session" not in st.session_state:
        st.session_state.active_session = tag
    return tag


def _active() -> dict:
    """当前 active 会话的状态 dict（自动兜底：active 失效时取会话列表首个）。"""
    sessions = _sessions()
    active = st.session_state.get("active_session")
    if active not in sessions:
        active = next(iter(sessions))
        st.session_state.active_session = active
    return sessions[active]


def _close_session(tag: str) -> None:
    """关闭会话；关到最后一个时自动新建（保证始终有会话）。"""
    sessions = _sessions()
    sessions.pop(tag, None)
    if not sessions:
        _create_session()
        return
    if st.session_state.get("active_session") == tag:
        st.session_state.active_session = next(iter(sessions))


def _open_chart_dialog() -> None:
    """打开图表放大对话框（@st.dialog 装饰器需在模块级定义）。

    逻辑本身在 ``components._chart_dialog``（UI 组件层）；此处只提供
    Streamlit 对话框壳。关闭对话框时清掉选中下标，避免下次重跑又弹出。
    """
    from app.ui.components import _chart_dialog

    @st.dialog("图表详情", width="large")
    def _dialog() -> None:
        _chart_dialog()
        if st.button("关闭", key="close_chart_dialog"):
            st.session_state["zoom_chart"] = None
            st.rerun()

    _dialog()


def _run_agent_turn(service, prompt: str, placeholder) -> ChatTurn:
    """执行一轮 agent（流式或非流式），把正文渲染进 placeholder。

    统一入口：流式开启时逐块更新 placeholder（首字节可见 + 工具播报），
    关闭或异常时回退到非流式 reply()。两条路径返回的 ChatTurn 同构，
    调用方的收尾逻辑（_record_turn）无需区分。

    Args:
        service: 当前会话的 ChatService。
        prompt: 用户本轮输入。
        placeholder: ``st.empty()`` 占位符（流式逐块写入正文）。

    Returns:
        本轮 ChatTurn（含 usage/metrics/error）。
    """
    from app.config.settings import get_settings
    from app.ui.constants import STREAM_CURSOR

    if not getattr(get_settings(), "stream_output_enabled", True):
        # 非流式回退路径（配置关闭，或流式实现出问题时的一键回退）。
        with st.spinner("分析中……"):
            turn = service.reply(prompt)
        placeholder.markdown(turn.reply)
        return turn

    acc = ""
    final_turn = None
    for chunk in service.reply_stream(prompt):
        if chunk.kind == "tool":
            # 工具播报：斜体 + 与正文区分（不阻塞后续正文写入）。
            placeholder.markdown(f"_{chunk.text}…_")
        elif chunk.kind == "delta":
            acc += chunk.text
            placeholder.markdown(acc + STREAM_CURSOR)
        elif chunk.kind == "final":
            final_turn = chunk.turn
    if final_turn is None:
        # 理论上不会发生（reply_stream 必 yield final）；兜底为空轮。
        final_turn = service.reply(prompt)
    # 收尾：去掉光标；失败轮保留已产出的部分正文（诚实降级）。
    placeholder.markdown(final_turn.reply)
    return final_turn


def _is_failed(turn) -> bool:
    """该轮是否未正常完成（模型 API 异常或撞 max_turns）。"""
    m = getattr(turn, "metrics", None) or {}
    return m.get("completed") is False


def _error_kind(turn) -> str | None:
    """该轮失败类别（None 表示正常）。"""
    m = getattr(turn, "metrics", None) or {}
    return m.get("error_kind")


def _retry_turn(service, messages: list[dict], cumulative: dict, user_idx: int) -> None:
    """重试第 user_idx 条用户消息对应的一轮（**替换而非追加**，见设计 2.3）。

    语义（避免消息与历史重复的关键）：
    1. agent 历史回退到本轮之前（同输入不在历史中重复）——复用编辑重发的方法；
    2. messages 截断到该用户消息之后（丢弃失败的 assistant 消息）；
    3. 用原文本重新执行；成功后经 _record_turn 追加新的 assistant 消息。

    Args:
        service: 当前会话 ChatService。
        messages: 会话消息列表（就地修改）。
        cumulative: 会话语义统计（就地修改）。
        user_idx: 该用户消息在 messages 中的下标。
    """
    user_text = messages[user_idx]["content"]
    service.truncate_history_to_turn(user_idx)
    del messages[user_idx + 1:]
    placeholder = st.empty()
    turn = _run_agent_turn(service, user_text, placeholder)
    _record_turn(cumulative, turn, messages)


def _render_failure_actions(service, messages: list[dict], cumulative: dict,
                            user_idx: int, turn) -> None:
    """失败轮的操作区：重试本轮（+ 上下文超限时"压缩历史并重试"）。

    仅在**最后一轮**失败时由调用方渲染（重试语义要求其后无对话）。
    """
    kind = _error_kind(turn)
    st.warning("本轮未完成，可直接重试。")
    cols = st.columns(2)
    if kind == "context_overflow":
        if cols[0].button("压缩历史并重试", key=f"retry_compact_{user_idx}",
                          type="primary"):
            service.compact_now()
            _retry_turn(service, messages, cumulative, user_idx)
            st.rerun()
    if cols[1].button("🔁 重试本轮", key=f"retry_{user_idx}"):
        _retry_turn(service, messages, cumulative, user_idx)
        st.rerun()


def _record_turn(cumulative: dict, turn, messages: list[dict]) -> None:
    """把一轮结果记入会话统计与消息列表（token / 轮数 / 耗时 / 往返次数）。

    为什么集中（docs/UI工程质量与配置化设计.md 2.3）：此前 token 累加在
    "正常输入"与"编辑重发"两处各写一遍，metrics 累加单列一个函数但调用点靠
    人工记得——任何新增交互入口（流式 / 重试 / 重新生成）都可能漏掉某处，
    导致统计静默偏差。集中后所有入口调用同一函数，杜绝漏加。

    累计量放 UI 侧而不放 ChatService：累计是**展示语义**（会话级统计），
    服务层只负责单轮事实；两处口径不同。

    Args:
        cumulative: 会话累计统计 dict（就地修改）。
        turn: 本轮 ChatTurn。
        messages: 会话消息列表（就地追加 assistant 消息）。
    """
    if turn.usage:
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            cumulative[key] = cumulative.get(key, 0) + turn.usage.get(key, 0)
    cumulative["rounds"] = cumulative.get("rounds", 0) + 1
    m = getattr(turn, "metrics", None) or {}
    cumulative["duration_ms"] = int(cumulative.get("duration_ms", 0)) + int(
        m.get("duration_ms", 0) or 0)
    cumulative["n_model_calls"] = int(cumulative.get("n_model_calls", 0)) + int(
        m.get("n_model_calls", 0) or 0)
    messages.append({"role": "assistant", "content": turn.reply, "turn": turn})


def _last_turn(messages: list[dict]):
    """返回最近一条 assistant 消息携带的 ChatTurn（无则 None）。

    比 ``messages[-1].get("turn")`` 更健壮：即便末尾是用户消息（如编辑态）
    也能取到最近一次真实回答的用量，且不会因 turn 缺失抛 AttributeError。
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("turn")
    return None


def _render_session_tabs() -> None:
    """顶部会话标签条：切换 / 关闭当前 / 新建。

    设计（docs/多会话标签页升级设计.md 2.1–2.3）：
    - 标签条用**横向滚动容器**（不再按会话数均分列宽），标签多时不挤压变形；
    - 每个标签只占**一个按钮**，当前标签用 primary 填充色区分（取代 `● ` 前缀，
      可多显示名字）；
    - 关闭入口收敛为**一个**"关闭当前"按钮（控件数由 2N 降到 N+2）；
      只剩 1 个会话时该按钮禁用（避免"关了又自动新建"的隐晦行为）；
    - 不做"关闭非当前会话"（需要每标签一个关闭控件，与前一条取舍冲突）。
    """
    sessions = _sessions()
    active = st.session_state.active_session
    names = list(sessions.keys())

    left, right = st.columns([4, 1])
    with left:
        # 横向滚动容器：标签不压缩宽度（需要时容器内滚动，而非变形）。
        with st.container(horizontal=True, horizontal_alignment="left"):
            for tag, stt in sessions.items():
                is_active = tag == active
                if st.button(stt["name"], key=f"tab_{tag}",
                             type="primary" if is_active else "secondary"):
                    if not is_active:
                        st.session_state.active_session = tag
                        st.rerun()
    with right:
        c_close, c_new = st.columns(2)
        only_one = len(sessions) <= 1
        if c_close.button("关闭当前", key="close_current", use_container_width=True,
                          disabled=only_one,
                          help="至少保留一个对话" if only_one else "关闭当前对话"):
            _close_session(active)
            st.rerun()
        if c_new.button("＋ 新建", key="new_session", use_container_width=True):
            _create_session()
            st.session_state.active_session = list(_sessions())[-1]
            st.rerun()


def _main() -> None:
    _inject_scroll_css()

    st.title("🩻 Embodied-data-Xray")
    st.caption("具身智能数据结构透视 · 全链路：加载 → 质检 → 统计 → 绘图 → 报告")

    # 未配置模型：渲染引导页（配置表单 + 说明），不构造 ChatService——
    # 修复此前"缺 key 时 UI 直接 traceback"的问题（设计文档 3.2）。
    if not is_configured():
        render_onboarding()
        return

    # 顶部：多会话标签页（新建 / 切换 / 关闭）；以下全部作用于 active 会话。
    _render_session_tabs()

    _stt = _active()
    service: ChatService = _stt["service"]
    messages: list[dict] = _stt["messages"]
    cumulative: dict = _stt["cumulative"]

    # 侧栏：模型设置（expander，随时改；保存后自动重建 service）。
    with st.sidebar:
        render_model_settings()
        st.divider()
        # 数据加载：路径（主）+ 单文件上传（辅）+ 示例数据集（可选）；
        # 加载结果进对话流（决策 3），故需传入 messages。
        render_data_loader(service, messages)
        # 该会话加载数据集后，标签名自动改为数据集名（多会话时便于辨识）。
        ds_id = service.context.dataset_id
        if ds_id and _stt.get("name", "").startswith("对话 "):
            _stt["name"] = (
                f"{ds_id[:SESSION_LABEL_MAX_CHARS]}…"
                if len(ds_id) > SESSION_LABEL_MAX_CHARS else ds_id
            )

    # 侧栏：Token 统计（本轮 + 会话累计；刷新页面重置属正常，不持久化）。
    with st.sidebar:
        last = _last_turn(messages)
        render_token_stats(last.usage if last is not None else None, cumulative)

        # 上下文管理：历史体积 + 手动压缩（对应"二者皆做"的手动入口）。
        st.divider()
        st.caption("上下文管理")
        stats = service.history_stats()
        st.write(
            f"历史：约 {stats['turns']} 轮 · 估算 {stats['estimated_tokens']:,} token"
        )
        last_c = stats.get("last_compaction")
        if last_c:
            st.caption(
                f"上次压缩：{last_c['compacted_outputs']} 条 · "
                f"{last_c['before_tokens']:,} → {last_c['after_tokens']:,} token"
                f"（省约 {last_c['saved_tokens']:,}）"
            )
        if st.button("压缩历史", help="把较早轮次的工具返回明细压缩为结论摘要"):
            if not service.history_input:
                st.info("历史为空，无需压缩。")
            else:
                r = service.compact_now()
                st.success(
                    f"已压缩 {r['compacted_outputs']} 条旧工具返回（保留最近 "
                    f"{r['kept_turns']} 轮完整）：{r['before_tokens']:,} → "
                    f"{r['after_tokens']:,} token（省约 {r['saved_tokens']:,}）"
                )
                st.rerun()

    left, right = st.columns(list(COLUMN_RATIO), gap="large")

    # ---- 左侧：对话区（聊天记录独立滚动 + 输入框钉底固定）----
    with left:
        st.subheader("对话")
        # 聊天记录独立滚动容器：回顾历史时输入框不跟随滚动。
        chat_container = st.container(height=_SCROLL_HEIGHT)
        editing_index = _stt.get("editing_index")
        with chat_container:
            # 渲染历史消息（新消息在容器底部，配合滚动锚定自动贴底）。
            for i, msg in enumerate(messages):
                if msg["role"] == "user" and i == editing_index:
                    # 编辑态：预填原文的文本框 + 保存重新生成 / 取消。
                    with st.chat_message("user"):
                        new_text = st.text_area(
                            "编辑消息（保存后此消息之后的对话将被重新生成）",
                            value=msg["content"],
                            key=f"edit_area_{i}",
                        )
                        c1, c2, _c3 = st.columns([1, 1, 2])
                        do_save = c1.button("保存并重新生成", key=f"edit_save_{i}",
                                            type="primary")
                        do_cancel = c2.button("取消", key=f"edit_cancel_{i}")
                        if do_cancel:
                            _stt["editing_index"] = None
                            st.rerun()
                        if do_save and new_text.strip():
                            # 语义核心：截断 agent 历史到该轮之前（bot 上下文
                            # 同步丢弃被编辑消息之后的一切），再重发新文本。
                            service.truncate_history_to_turn(i)
                            messages[:] = messages[:i] + [
                                {"role": "user", "content": new_text.strip()}
                            ]
                            _stt["editing_index"] = None
                            placeholder = st.empty()
                            turn = _run_agent_turn(service, new_text.strip(),
                                                   placeholder)
                            _record_turn(cumulative, turn, messages)
                            st.rerun()
                else:
                    with st.chat_message(msg["role"]):
                        # 失败轮先给醒目标记（滚历史时一眼可辨，见设计 2.1）。
                        if (msg["role"] == "assistant" and msg.get("turn")
                                and _is_failed(msg["turn"])):
                            st.caption("⚠️ 本轮未完成")
                        st.markdown(msg["content"])
                        # 用户消息旁的编辑入口（最近一条才显示，避免历史深处
                        # 编辑造成大面积重生成；与主流对话 UI 一致）。
                        if msg["role"] == "user" and i == len(messages) - 1:
                            if st.button("✏️ 编辑", key=f"edit_btn_{i}",
                                         help="修改并重新发送，之后的回答将重新生成"):
                                _stt["editing_index"] = i
                                st.rerun()
                        # 失败轮的恢复入口：仅最后一轮失败时显示（重试要求其后
                        # 无对话；该 assistant 消息的 user 消息在下标 i-1）。
                        if (msg["role"] == "assistant" and msg.get("turn")
                                and _is_failed(msg["turn"])
                                and i == len(messages) - 1):
                            _render_failure_actions(service, messages, cumulative,
                                                    i - 1, msg["turn"])
                        # 助手回复下方附工具轨迹（可折叠）。
                        if msg["role"] == "assistant" and msg.get("turn"):
                            render_tool_activity(msg["turn"])

        # 输入框放在滚动容器**之外**：Streamlit 会把它钉在视口底部（ChatGPT 式
        # 布局），滚动聊天记录时位置不变——可同时回顾历史与输入新对话。
        # （此前输入框在滚动容器内，会随聊天记录一起滚走——用户体验问题。）
        prompt = st.chat_input("输入你的问题……")
        if prompt:
            # 追加用户消息。
            messages.append({"role": "user", "content": prompt})

            # 新消息渲染进滚动容器（末尾追加，配合滚动锚定贴底）。
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                # 只有新输入才调用 agent（页面重跑时 prompt 为空，不触发）。
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    turn = _run_agent_turn(service, prompt, placeholder)
                    render_tool_activity(turn)

            # 累计本轮统计并追加 assistant 消息（集中入口，防漏加）。
            _record_turn(cumulative, turn, messages)

    # ---- 右侧：展示区（固定高度独立滚动容器）----
    with right:
        # 右栏面板放入固定高度容器：与左栏各自独立滚动，互不影响。
        with st.container(height=_SCROLL_HEIGHT):
            tab_charts, tab_findings, tab_overview = st.tabs(
                ["图表", "Findings/报告", "数据概况"]
            )
            findings = service.context.findings

            with tab_charts:
                render_charts(findings)
            with tab_findings:
                render_findings_and_report(findings)
            with tab_overview:
                render_dataset_overview(service.dataset_summary())

    # 图表放大对话框：放在页面末尾渲染（@st.dialog 的调用位置不影响弹出）。
    if hasattr(st, "dialog") and st.session_state.get("zoom_chart") is not None:
        _open_chart_dialog()


if __name__ == "__main__":
    _main()
