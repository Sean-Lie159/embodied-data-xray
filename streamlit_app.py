"""Streamlit 界面入口（唯一入口：streamlit run streamlit_app.py）。

布局：左侧对话区 + 右侧展示区（图表 / Findings·报告 / 数据概况 tabs）。
会话状态（ChatService、对话历史）存于 st.session_state，只有新的用户输入才触发
agent 执行；页面重跑（切换 tab、点击按钮）不重复调用 agent。
"""

from __future__ import annotations

import streamlit as st

from app.services.chat_service import ChatService
from app.ui.components import (
    render_charts,
    render_dataset_overview,
    render_findings_and_report,
    render_token_stats,
    render_tool_activity,
)

st.set_page_config(page_title="具身智能数据分析 Agent", layout="wide")

# 左右栏可滚动容器高度（像素）。按视口合理设定，避免与页面级滚动叠成双重滚动条。
_SCROLL_HEIGHT = 600


def _inject_scroll_css() -> None:
    """注入最小 CSS（仅 overflow/滚动锚定相关，不做自定义布局 hack）。

    目的：
    1. 让页面主体不产生页面级滚动（body overflow hidden），左右栏各自在
       ``st.container(height=...)`` 内独立滚动，避免"页面 + 容器"双重滚动条的别扭体验。
    2. 给聊天容器开启 ``overflow-anchor``（滚动锚定），使新消息到达时自动贴底，
       而不是把滚动位置留在旧消息处。
    """
    st.markdown(
        """
        <style>
        /* 页面主体不滚动：左右栏各自在固定高度容器内滚动，避免双重滚动条 */
        .block-container { overflow: hidden; }
        /* 滚动锚定：聊天/面板容器内新内容追加时尽量保持贴底/原位置稳定 */
        [data-testid="stVerticalBlock"] > div {
            overflow-anchor: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_service() -> ChatService:
    """惰性初始化并缓存 ChatService（含 agent / RunContext / 对话历史）。"""
    if "service" not in st.session_state:
        st.session_state.service = ChatService()
    return st.session_state.service


def _get_messages() -> list[dict]:
    """返回对话消息列表（含每轮的回复与工具轨迹）。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages


def _get_cumulative_usage() -> dict:
    """返回会话累计 token 用量（st.session_state 维护，刷新页面重置属正常）。"""
    if "cumulative_usage" not in st.session_state:
        st.session_state.cumulative_usage = {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "rounds": 0,
        }
    return st.session_state.cumulative_usage


def _main() -> None:
    service = _get_service()
    messages = _get_messages()
    cumulative = _get_cumulative_usage()

    _inject_scroll_css()

    st.title("具身智能数据分析 Agent")
    st.caption("全链路：加载 → 质检 → 统计 → 绘图 → 报告")

    # 侧栏：Token 统计（本轮 + 会话累计；刷新页面重置属正常，不持久化）。
    with st.sidebar:
        last_usage = (
            messages[-1].get("turn").usage
            if messages and messages[-1].get("turn") is not None
            else None
        )
        render_token_stats(last_usage, cumulative)

    left, right = st.columns([1, 1.2], gap="large")

    # ---- 左侧：对话区（固定高度独立滚动容器）----
    with left:
        st.subheader("对话")
        # 左栏对话放入固定高度容器：内容超限时在容器内独立滚动，不带动右栏。
        with st.container(height=_SCROLL_HEIGHT):
            # 渲染历史消息（新消息在容器底部，配合滚动锚定自动贴底）。
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # 助手回复下方附工具轨迹（可折叠）。
                    if msg["role"] == "assistant" and msg.get("turn"):
                        render_tool_activity(msg["turn"])

            # 输入框：只有非空输入才触发 agent 执行。
            prompt = st.chat_input("输入你的问题……")
            if prompt:
                # 追加用户消息。
                messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # 只有新输入才调用 agent（页面重跑时 prompt 为空，不触发）。
                with st.chat_message("assistant"):
                    with st.spinner("分析中……"):
                        turn = service.reply(prompt)
                    st.markdown(turn.reply)
                    render_tool_activity(turn)
                # 累计本轮 token 用量（usage 为 None 时不加，避免 0 冒充）。
                if turn.usage:
                    cumulative["input_tokens"] += turn.usage.get("input_tokens", 0)
                    cumulative["output_tokens"] += turn.usage.get("output_tokens", 0)
                    cumulative["total_tokens"] += turn.usage.get("total_tokens", 0)
                cumulative["rounds"] += 1
                messages.append({"role": "assistant", "content": turn.reply, "turn": turn})

    # ---- 右侧：展示区（固定高度独立滚动容器）----
    with right:
        # 右栏面板放入固定高度容器：与左栏各自独立滚动，互不影响。
        with st.container(height=_SCROLL_HEIGHT):
            tab_charts, tab_findings, tab_overview = st.tabs(
                ["图表", "Findings/报告", "数据概况"]
            )
            findings = service.context.findings if "service" in st.session_state else []

            with tab_charts:
                render_charts(findings)
            with tab_findings:
                render_findings_and_report(findings)
            with tab_overview:
                render_dataset_overview(service.dataset_summary())


if __name__ == "__main__":
    _main()
