"""UI 可复用组件（唯一允许 import streamlit 的模块之一）。

渲染：图表（findings type=chart）、findings/报告、数据概况、工具轨迹面板。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.config import get_settings
from app.services.chat_service import ChatTurn


def render_tool_activity(turn: ChatTurn) -> None:
    """在回复下方渲染可折叠的工具调用轨迹面板。

    聊天正文保持干净，技术细节（工具调用）可展开查看。
    """
    if not turn.tool_activity:
        return
    with st.expander("查看执行过程"):
        st.caption(turn.tool_activity)
        if turn.tool_calls:
            st.caption("本轮调用工具：" + "、".join(turn.tool_calls))


def render_token_stats(usage: dict | None, cumulative: dict) -> None:
    """在侧栏渲染 Token 统计区块。

    Args:
        usage: 本轮 token 用量 {input/output/total} 或 None。
        cumulative: 会话累计 {input/output/total/rounds}。
    """
    st.markdown("**Token 统计**")
    if usage is None:
        st.caption("本轮：本次未获取到用量")
    else:
        st.caption(
            f"本轮：输入 {usage.get('input_tokens', 0):,} | "
            f"输出 {usage.get('output_tokens', 0):,} | "
            f"合计 {usage.get('total_tokens', 0):,}"
        )
    st.caption(
        f"累计：输入 {cumulative.get('input_tokens', 0):,} | "
        f"输出 {cumulative.get('output_tokens', 0):,} | "
        f"合计 {cumulative.get('total_tokens', 0):,} | "
        f"{cumulative.get('rounds', 0)} 轮"
    )
    cost = _estimate_cost_text(usage, cumulative)
    if cost:
        st.caption(f"成本估算：{cost}")


def _estimate_cost_text(usage: dict | None, cumulative: dict) -> str:
    """估算本轮与累计成本（美元）；未配置价格或用量缺失时返回空串。"""
    settings = get_settings()
    p_in, p_out = settings.price_input_per_mtok, settings.price_output_per_mtok
    if not (p_in > 0 and p_out > 0):
        return ""
    if usage is None:
        return "本次未获取到用量"
    cost = usage.get("input_tokens", 0) / 1e6 * p_in + usage.get("output_tokens", 0) / 1e6 * p_out
    cost_acc = (
        cumulative.get("input_tokens", 0) / 1e6 * p_in
        + cumulative.get("output_tokens", 0) / 1e6 * p_out
    )
    return f"本轮 ≈${cost:.4f} / 累计 ≈${cost_acc:.4f}"


def render_charts(findings: list[dict]) -> None:
    """渲染 findings 中 type=chart 的图片。

    图片路径来自 chart 条目的 file_path（相对 outputs/ 的相对路径）。
    文件缺失时降级显示提示而非报错。
    """
    charts = [f for f in findings if f.get("type") == "chart"]
    if not charts:
        st.info("暂无图表。请先通过对话生成图表。")
        return

    for i, f in enumerate(charts):
        fp = f.get("file_path", "")
        title = f.get("title", f"图表 {i + 1}")
        desc = f.get("description", "")
        spec = f.get("plot_spec", {})
        st.subheader(title)
        if desc:
            st.caption(desc)
        if fp:
            path = Path(fp)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"图表文件缺失：{fp}。该文件可能已被清理。")
        else:
            st.warning("该图表条目缺少文件路径。")
        if spec:
            st.caption(
                f"坐标：x={spec.get('x_axis', '?')}，y={spec.get('y_axis', [])}，"
                f"分组={spec.get('grouped_by', None)}，曲线数={spec.get('n_series', '?')}"
            )
        st.divider()


def render_findings_and_report(findings: list[dict]) -> None:
    """展示 findings 列表与报告下载按钮。"""
    if not findings:
        st.info("当前会话暂无分析结果。")
        return

    st.subheader("Findings")
    for f in findings:
        ftype = f.get("type", "?")
        tool = f.get("tool", "?")
        summary = f.get("summary") or f.get("description") or ""
        st.markdown(f"- **[{ftype}]** {tool}: {summary}")

    # 报告下载按钮。
    report_paths = [f.get("file_path") for f in findings if f.get("type") == "report"]
    if report_paths:
        for rp in report_paths:
            path = Path(rp)
            if path.exists():
                st.download_button(
                    label=f"下载报告：{path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/markdown",
                )
            else:
                st.warning(f"报告文件缺失：{rp}")


def render_dataset_overview(summary: dict[str, Any]) -> None:
    """展示当前数据集的能力标签与流清单摘要。"""
    dataset_id = summary.get("dataset_id")
    if not dataset_id:
        st.info("尚未加载数据集。请先在对话中提供数据路径。")
        return

    st.subheader(f"数据集：{dataset_id}")
    guessed = summary.get("guessed_type")
    if guessed:
        st.caption(f"推测类型：{guessed}")

    caps = summary.get("capabilities", {})
    st.markdown("**能力标签**")
    # IMU 轴数：None 或 "unknown" 时显示"未知"，避免侧栏出现 "None 轴"。
    imu_axes = caps.get("imu_axes")
    imu_axes_txt = "未知" if imu_axes is None or imu_axes == "unknown" else f"{imu_axes} 轴"
    cap_lines = [
        f"- 视频流: {'✓' if caps.get('has_video_streams') else '✗'}",
        f"- IMU: {'✓' if caps.get('has_imu') else '✗'}（{imu_axes_txt}）",
        f"- 力/力矩: {'✓' if caps.get('has_force') else '✗'}",
        f"- 标定: {'✓' if caps.get('has_calibration') else '✗'}",
        f"- 状态/动作: {'✓' if caps.get('has_actions') else '✗'}",
    ]
    st.markdown("\n".join(cap_lines))

    streams = summary.get("streams", [])
    if streams:
        st.markdown("**流清单**")
        rows = []
        for s in streams:
            name = Path(s.get("path", "")).name if s.get("path") else "(main)"
            role = (s.get("role") or {}).get("role", s.get("kind", "?"))
            mr = s.get("measured_rate")
            rate = (mr or {}).get("sample_rate_hz") if isinstance(mr, dict) else None
            rate_str = f"{rate} Hz" if rate is not None else "未知"
            rows.append({"流": name, "角色": role, "采样率": rate_str})
        st.table(rows)
