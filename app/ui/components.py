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

    聊天正文保持干净，技术细节（工具调用）可展开查看。耗时与工具循环次数
    一并展示（502 排查的第一手事实：逐流循环会表现为循环次数高 + 耗时随流数
    线性增长）。
    """
    metrics = turn.metrics or {}
    if not turn.tool_activity and not metrics:
        return
    with st.expander("查看执行过程"):
        # 摘要行：耗时 + 工具循环次数（始终可见，便于一眼判断慢在哪里）。
        st.caption(f"本轮耗时 {_format_duration(metrics.get('duration_ms'))}｜"
                   f"{_loop_summary(metrics)}")
        if not turn.tool_activity:
            return
        st.caption(turn.tool_activity)
        if turn.tool_calls:
            st.caption("本轮调用工具：" + "、".join(turn.tool_calls))


def _format_duration(duration_ms: Any) -> str:
    """把毫秒耗时格式化为可读文本（<1s 用毫秒，否则用秒）。"""
    if not isinstance(duration_ms, (int, float)) or duration_ms <= 0:
        return "未知"
    if duration_ms < 1000:
        return f"{int(duration_ms)} ms"
    return f"{duration_ms / 1000:.1f} s"


def _loop_summary(metrics: dict) -> str:
    """把工具循环指标格式化为一行摘要。

    模型往返次数（n_model_calls）≈ 工具调用轮数 + 1，是判断"是否逐流循环"的
    关键量：一次调用覆盖 N 条流应为个位数，逐流循环会接近 N。
    """
    calls = metrics.get("n_model_calls")
    tools = metrics.get("n_tool_calls")
    done = metrics.get("completed", True)
    if not isinstance(calls, (int, float)) or calls <= 0:
        return "过程未知"
    text = f"模型往返 {int(calls)} 次"
    if isinstance(tools, (int, float)) and tools > 0:
        text += f"｜工具调用 {int(tools)} 次"
    if not done:
        text += "｜未正常完成"
    return text


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
    st.caption(_speed_text(cumulative))


def _speed_text(cumulative: dict) -> str:
    """会话级速度摘要：累计耗时 + 总模型往返次数 + 平均每次往返耗时。

    平均单次往返耗时（duration / n_model_calls）是判断瓶颈位置的量：
    该值稳定在低位而总耗时高，说明是**往返次数多**（工具循环/逐流调用）——
    这正是可以靠"批量纪律"优化的那一类；该值本身很高则属单次请求慢
    （服务侧或上下文过大），优化方向不同。
    """
    total_ms = int(cumulative.get("duration_ms", 0) or 0)
    calls = int(cumulative.get("n_model_calls", 0) or 0)
    if total_ms <= 0:
        return "速度：暂无数据"
    text = f"速度：累计耗时 {_format_duration(total_ms)}"
    if calls > 0:
        text += (f"｜模型往返 {calls} 次"
                 f"｜平均 {total_ms / calls / 1000:.1f} s/次")
    return text


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


def _chart_dialog() -> None:
    """放大查看单个图表的对话框。

    设计（docs/右栏信息架构重构设计.md 2.1）：网格缩略图 + 点击放大，
    替代此前"每张图全宽铺开"的低密度呈现。被选中的图表下标存于
    session_state["zoom_chart"]，由缩略图按钮写入。
    """
    idx = st.session_state.get("zoom_chart")
    charts = st.session_state.get("_charts_cache", [])
    if idx is None or idx >= len(charts):
        return
    f = charts[idx]
    fp = f.get("file_path", "")
    title = f.get("title", "图表")
    st.subheader(title)
    desc = f.get("description", "")
    if desc:
        st.caption(desc)
    path = Path(fp) if fp else None
    if path is not None and path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"图表文件缺失：{fp or '（无路径）'}。该文件可能已被清理。")
    spec = f.get("plot_spec", {})
    if spec:
        st.caption(
            f"坐标：x={spec.get('x_axis', '?')}，y={spec.get('y_axis', [])}，"
            f"分组={spec.get('grouped_by', None)}，曲线数={spec.get('n_series', '?')}"
        )


def render_charts(findings: list[dict]) -> None:
    """渲染 findings 中 type=chart 的图片（两列缩略图 + 点击放大，最新在上）。

    设计见 docs/右栏信息架构重构设计.md 2.1：倒序（最新在最上）+ 缩略图网格
    + 点击 @st.dialog 放大。仅改渲染方式，finding 字段与语义不变。
    """
    charts = [f for f in findings if f.get("type") == "chart"]
    if not charts:
        st.info("暂无图表。请先通过对话生成图表。")
        return

    # 倒序：最新在最上（多轮分析时注意力在最近一轮）。
    charts = list(reversed(charts))
    # 供放大对话框取用（dialog 在页面底部渲染，需能访问到当前图表列表）。
    st.session_state["_charts_cache"] = charts

    for row_start in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, f in zip(cols, charts[row_start:row_start + 2]):
            with col:
                fp = f.get("file_path", "")
                title = f.get("title", "图表")
                path = Path(fp) if fp else None
                if path is not None and path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.warning("图片缺失")
                # 缩略图下方：标题 + 类型/表名（finding 已有字段，不新增数据）。
                st.caption(f"**{title}**")
                meta_bits = [str(f.get("chart_type", ""))]
                if f.get("table_name"):
                    meta_bits.append(f"表：{f['table_name']}")
                meta_txt = " · ".join(b for b in meta_bits if b)
                if meta_txt:
                    st.caption(meta_txt)
                if st.button("放大查看", key=f"zoom_{row_start}_{id(f)}",
                             use_container_width=True):
                    st.session_state["zoom_chart"] = charts.index(f)
                    st.rerun()


def _render_finding_group(title: str, items: list[dict]) -> None:
    """渲染一个 finding 分组（表头 + 条目列表）；空组不渲染（防噪声）。"""
    if not items:
        return
    with st.expander(f"{title}（{len(items)}）", expanded=True):
        for f in reversed(items):  # 组内同样最新在上
            tool = f.get("tool", "?")
            summary = f.get("summary") or f.get("description") or ""
            if summary:
                st.markdown(f"- **{tool}**：{summary}")
            else:
                st.markdown(f"- **{tool}**")


def render_findings_and_report(findings: list[dict]) -> None:
    """按类型分组展示 findings 与报告下载按钮（最新优先）。

    设计见 docs/右栏信息架构重构设计.md 2.2：此前所有类型混在一列扁平 bullet
    且最新在最后；改为按 type 分组（图表清单/统计结论/质检结果/报告），
    组内倒序，空组不渲染。
    """
    if not findings:
        st.info("当前会话暂无分析结果。")
        return

    # 按 type 分组（type 是稳定小集合；用户心智是"看图/看数字/看报告"）。
    charts = [f for f in findings if f.get("type") == "chart"]
    stats = [f for f in findings if f.get("type") == "stat"]
    reports = [f for f in findings if f.get("type") == "report"]
    others = [f for f in findings
              if f.get("type") not in ("chart", "stat", "report")]

    _render_finding_group("统计结论", stats)
    _render_finding_group("图表清单", charts)
    _render_finding_group("其它结果", others)

    # 报告下载按钮（保留原逻辑，仅位置归组）。
    if reports:
        st.subheader("报告")
        for f in reversed(reports):
            rp = f.get("file_path")
            if not rp:
                continue
            path = Path(rp)
            if path.exists():
                st.download_button(
                    label=f"下载报告：{path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/markdown",
                    key=f"dl_{path.name}",
                )
            else:
                st.warning(f"报告文件缺失：{rp}")


# 语义标签来源的显示名（profile_store 定义的常量 → 中文）。
_LABEL_SOURCE_TEXT = {
    "user_confirmed": "用户确认",
    "content_fingerprint": "内容指纹",
    "dictionary": "词典",
}


def _render_capabilities(caps: dict) -> None:
    """[1] 能力标签段（保持原有展示逻辑不变）。"""
    st.markdown("**能力标签**")
    # IMU 轴数：无 IMU（✗）时不显示轴数（避免"✗（未知轴）"的冗余）；有 IMU 但
    # 轴数未知时才显示"未知轴"。
    imu_axes = caps.get("imu_axes")
    if caps.get("has_imu"):
        imu_axes_txt = "未知轴" if imu_axes is None or imu_axes == "unknown" else f"{imu_axes} 轴"
        imu_line = f"- IMU: ✓（{imu_axes_txt}）"
    else:
        imu_line = "- IMU: ✗"
    cap_lines = [
        f"- 视频流: {'✓' if caps.get('has_video_streams') else '✗'}",
        imu_line,
        f"- 力/力矩: {'✓' if caps.get('has_force') else '✗'}",
        f"- 标定: {'✓' if caps.get('has_calibration') else '✗'}",
        f"- 状态/动作: {'✓' if caps.get('has_actions') else '✗'}",
    ]
    st.markdown("\n".join(cap_lines))


def _render_semantic_progress(streams: list[dict], qc_state: dict | None) -> None:
    """[2] 语义确认进度段（含已确认清单）。

    口径与 inspect_streams 一致（docs/数据集状态面板设计.md 2.2）：
    优先用工具回写的 qc_state（与工具同一次计算，绝不与工具口径打架）；
    工具未调用过时退化为 UI 自算（同一判据，见下方 _is_classified）。
    """
    st.markdown("**语义确认进度**")
    n = (qc_state or {}).get("n_streams")
    if n is None:
        n = len(streams)
    classified = (qc_state or {}).get("n_classified")
    if classified is None:
        classified = sum(1 for s in streams if _is_classified(s))
    unclassified = max(0, n - classified)

    # 状态用原生语义色块承载（docs/UI视觉优化设计.md 3.3）：
    # 全部已分类 → success（绿）；有未分类 → warning（黄）。不用 emoji 符号。
    if classified == n and n > 0:
        st.success(f"{classified}/{n} 条流已分类")
    else:
        st.warning(f"{classified}/{n} 条流已分类 · {unclassified} 条未分类")
    hint = (qc_state or {}).get("unclassified_hint")
    if hint:
        st.caption("建议：让 Agent 批量提交语义假设并确认（一次确认，跨会话生效）。")

    # 已确认清单（来自持久化画像，跨会话生效——非模型记忆）。
    confirmed = [s for s in streams
                 if s.get("label_source") == "user_confirmed"]
    if confirmed:
        with st.expander(f"已确认清单（{len(confirmed)}）", expanded=False):
            st.caption("来自持久化确认画像，跨会话生效（非模型记忆）。")
            rows = []
            for s in confirmed:
                path = s.get("path", "")
                rows.append({
                    "流": Path(path).name if path else "(main)",
                    "语义": s.get("semantic_label") or s.get("kind") or "?",
                })
            st.table(rows)


def _is_classified(s: dict) -> bool:
    """流是否已分类（与 inspect_streams 的判据一致，见其 classified 统计）。"""
    return (
        s.get("label_source") == "user_confirmed"
        or ((s.get("semantic_label") or "").find("未知") < 0
            and s.get("kind") not in (None, "unknown"))
    )


def _render_quality_warnings(qc: dict) -> None:
    """[3] 数据质量告警段（单位未知 / 时钟形态矛盾）。

    **措辞区分"未检查"与"无问题"**（docs/数据集状态面板设计.md 5 节）：
    三种状态互斥且用**不同的原生色块**承载（docs/UI视觉优化设计.md 3.3）——
    未检查 → ``st.info``（蓝，中性）；已检查无告警 → ``st.success``（绿）；
    有告警 → ``st.warning``（黄）。绝不把"没查"渲染成"没问题"。
    """
    sync = (qc or {}).get("check_temporal_sync")
    if not sync:
        st.markdown("**数据质量告警**")
        st.info("尚未执行时间同步检查（如需请让 Agent 检查时间同步）。")
        return
    detail = sync.get("detail", {})
    unit_warnings = detail.get("unit_warnings") or []
    clock_conflicts = detail.get("clock_conflicts") or []
    if not unit_warnings and not clock_conflicts:
        st.markdown("**数据质量告警**")
        st.success("已执行时间同步检查，无单位告警。")
        return
    st.markdown("**数据质量告警**")
    if unit_warnings:
        st.warning(f"{len(unit_warnings)} 条流时间戳单位未知（不参与跨流对齐）")
        with st.expander("查看单位未知的流", expanded=False):
            for w in unit_warnings:
                st.markdown(f"- {w}")
    if clock_conflicts:
        st.warning(f"{len(clock_conflicts)} 条流疑似时钟形态矛盾")
        with st.expander("查看时钟矛盾的流", expanded=False):
            for c in clock_conflicts:
                st.markdown(f"- {c}")


def _render_main_table(main_table: dict) -> None:
    """[5] 数据概况段（主表行/列；截断时明确提示）。"""
    if not main_table:
        return
    st.markdown("**数据概况**")
    rows_total = main_table.get("rows_total")
    rows_loaded = main_table.get("rows_loaded")
    n_cols = main_table.get("n_cols")
    bits = []
    if rows_loaded is not None:
        bits.append(f"行数 {rows_loaded:,}")
    if n_cols is not None:
        bits.append(f"列数 {n_cols}")
    if bits:
        st.caption(" · ".join(bits))
    if (rows_total is not None and rows_loaded is not None
            and rows_total != rows_loaded):
        st.warning(
            f"主表被截断装载：{rows_loaded:,} / {rows_total:,} 行"
            "（分析基于截断后数据，非全量）。"
        )


def render_dataset_overview(summary: dict[str, Any]) -> None:
    """展示当前数据集状态（五段式，见 docs/数据集状态面板设计.md 2.1）。

    段：[1] 能力标签 · [2] 语义确认进度 · [3] 数据质量告警 ·
    [4] 流清单 · [5] 数据概况。**只做展示，不提供语义编辑入口**
    （语义确认必须走 Agent + 工具验证，是项目核心纪律）。
    """
    dataset_id = summary.get("dataset_id")
    if not dataset_id:
        st.info("尚未加载数据集。请先在对话中提供数据路径。")
        return

    st.subheader(f"数据集：{dataset_id}")
    guessed = summary.get("guessed_type")
    if guessed:
        st.caption(f"推测类型：{guessed}")

    streams = summary.get("streams", [])
    _render_capabilities(summary.get("capabilities", {}))
    st.divider()
    _render_semantic_progress(streams, summary.get("qc_state"))
    st.divider()
    _render_quality_warnings(summary.get("qc", {}))

    if streams:
        st.divider()
        # 流清单是最长的内容块：用描边卡片包一层，使长表格有明确边界
        # （docs/UI视觉优化设计.md 3.1 右栏部分）。
        with st.container(border=True):
            st.markdown("**流清单**")
            # 视频 fps 映射（ffprobe 实测），供视频流展示帧率而非"未知"。
            fps_by_file = summary.get("video_fps_by_file") or {}
            rows = []
            for s in streams:
                path = s.get("path", "")
                name = Path(path).name if path else "(main)"
                role = (s.get("role") or {}).get("role", s.get("kind", "?"))
                mr = s.get("measured_rate")
                rate = (mr or {}).get("sample_rate_hz") if isinstance(mr, dict) else None
                if rate is not None:
                    rate_str = f"{rate} Hz"
                elif path in fps_by_file:
                    rate_str = f"{fps_by_file[path]} fps（视频）"
                else:
                    rate_str = "未知"
                # 语义标签与来源（新增两列，来源于已有流登记表字段）。
                label = s.get("semantic_label") or s.get("kind") or "未分类"
                source_raw = s.get("label_source")
                source = _LABEL_SOURCE_TEXT.get(source_raw, source_raw or "自动识别")
                rows.append({"流": name, "角色": role, "采样率": rate_str,
                             "语义标签": label, "来源": source})
            st.table(rows)

    main_table = summary.get("main_table") or {}
    if main_table:
        st.divider()
        _render_main_table(main_table)

