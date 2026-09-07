"""首次运行引导页：未配置模型时替代主界面渲染（修复"打开即崩"）。

设计（2026-09-07 决策 5）：**两块并列、非强制步骤流**——配置模型表单 +
说明区，不设强制向导；配置保存后自动进入主界面。示例数据集是主界面侧栏
数据面板中的常驻可选项，不是引导必经步骤。

实例模式（阶段 B）：未配置时不渲染配置表单——模型配置由部署者预设，
访问者不得看到或代填部署者的 key，仅提示联系部署者。
"""

from __future__ import annotations

import streamlit as st

from app.config.settings import get_instance_mode
from app.ui.settings_panel import render_model_settings


def render_onboarding() -> None:
    """渲染首次运行引导页（调用方负责在此之前 st.title 等，并 st.stop()）。"""
    if get_instance_mode():
        st.warning(
            "本实例尚未完成模型配置（INSTANCE_MODE 已开启）。"
            "请联系部署者在服务器上完成 `.env` 配置并重启服务后使用。"
        )
        return
    st.subheader("快速上手")
    st.markdown(
        "首次使用需先配置模型 API（OpenAI 兼容接口）。填写后点击**保存配置**，"
        "页面将自动进入主界面。\n\n"
        "常用服务商接口地址：DeepSeek `https://api.deepseek.com` · "
        "Kimi `https://api.moonshot.cn/v1` · OpenAI `https://api.openai.com/v1`"
    )

    render_model_settings()

    st.divider()
    with st.expander("配置好之后能做什么？"):
        st.markdown(
            "1. **加载数据**：主界面左侧栏『数据加载』面板粘贴数据集绝对路径"
            "（文件或目录），或上传单个 csv/parquet 文件；\n"
            "2. **一键看效果**：『数据加载』面板内置**示例数据集**（小型 IMU 流"
            "含数据缺口 + 任务表），点击即可加载——可选，用于 30 秒看到"
            "全链路（概况 → 质检 → 统计 → 绘图）的效果；\n"
            "3. **对话分析**：在对话框直接提问，如『这个数据集概况如何？』"
            "『时间同步检查一下，缺口发生在哪？』。"
        )
