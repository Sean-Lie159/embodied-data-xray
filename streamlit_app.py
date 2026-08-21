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
    render_tool_activity,
)

st.set_page_config(page_title="具身智能数据分析 Agent", layout="wide")


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


def _main() -> None:
    service = _get_service()
    messages = _get_messages()

    st.title("具身智能数据分析 Agent")
    st.caption("全链路：加载 → 质检 → 统计 → 绘图 → 报告")

    left, right = st.columns([1, 1.2], gap="large")

    # ---- 左侧：对话区 ----
    with left:
        st.subheader("对话")
        # 渲染历史消息。
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
            messages.append({"role": "assistant", "content": turn.reply, "turn": turn})

    # ---- 右侧：展示区 ----
    with right:
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
