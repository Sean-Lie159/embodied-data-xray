"""访问口令门（阶段 B 可选门禁）：局域网部署时防止"路过点开"。

定位是简单门禁而非强鉴权（docs/实例部署模式设计.md 第 5 节）：防局域网内
无关人员误入，不防定向攻击；Streamlit 无 TLS，口令明文传输，仅限可信内网。

判定逻辑抽纯函数（``check_access_password``）可单测；``render_access_gate``
负责渲染输入表单与会话状态。
"""

from __future__ import annotations

import secrets

import streamlit as st


def check_access_password(expected: str | None, provided: str | None) -> bool:
    """校验访问口令（纯函数）。

    Args:
        expected: 部署机配置的口令（ACCESS_PASSWORD）；空 = 门禁未启用。
        provided: 访问者输入的口令。

    Returns:
        True 表示放行（未启用门禁时恒 True）。
    """
    if not expected:
        return True
    # compare_digest 恒时比较，避免按字符短路产生的时序差。
    return secrets.compare_digest(str(expected), str(provided or ""))


def render_access_gate() -> bool:
    """渲染口令门（调用方在页面渲染早期调用）。

    门禁未启用（ACCESS_PASSWORD 为空）时直接放行；已解锁会话直接放行；
    否则渲染口令输入表单并阻断后续渲染。

    Returns:
        True 表示**应停止渲染**（未解锁）；False 表示放行，继续渲染主界面。
    """
    from app.config.settings import get_access_password_lenient

    # lenient 读取：实例 key 未配好时口令门同样生效（防同事看到配置表单）。
    expected = get_access_password_lenient()
    if not expected:
        return False
    if st.session_state.get("access_granted"):
        return False

    st.title("访问口令")
    st.caption("本实例面向特定人员开放，请输入访问口令。")
    provided = st.text_input("访问口令", type="password", key="access_password_input")
    if st.button("进入", type="primary", key="access_gate_enter"):
        if check_access_password(expected, provided):
            st.session_state["access_granted"] = True
            st.rerun()
        else:
            st.error("口令不正确，请重试。")
    return True
