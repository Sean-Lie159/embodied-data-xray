"""报告生成工具（确定性组装，LLM 不参与正文生成）。

汇总当前会话的分析结果为 Markdown 报告，保存到 outputs/。报告由工具从三个权威
来源确定性拼装：RunContext.findings（统计结论 type=stat 与图表条目 type=chart）、
meta["qc"]（质检判定摘要）、meta 的流登记表与能力标签（数据集概况）。

原则：报告里的每个数字都来自上述来源，工具不得编造；与"确定性计算优先"的架构
原则一致。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools.capability_meta import build_capability_matrix


def _output_report_path(context: RunContext) -> Path:
    """报告路径：outputs/by_dataset/<数据集>/reports/<...>.md（子目录见总纲第 3 节）。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    from app.tools._data_access import output_prefix
    from app.tools.output_paths import report_dir

    name = (f"{output_prefix(context)}{context.dataset_id or 'dataset'}"
            f"_report_{ts}.md")
    return report_dir(context.output_dir or "outputs", context.dataset_id) / name


def _build_dataset_overview(context: RunContext) -> str:
    """数据集画像章节：文件普查摘要 + 流明细表 + 模态矩阵。"""
    caps = context.meta.get("capabilities", {})
    streams = context.meta.get("streams", [])
    #
    # **类型措辞修正（2026-09-21）**：区分"用户已确认"与"自动推测"两种来源，
    # 并在 unknown 时主动提示可确认——此前报告永远只显示 `推测类型: unknown`，
    # 即使用户已经做过语义确认也无处体现（用户质疑："做过语义确认，为什么
    # 还是 unknown？"）。根因是数据集类型缺少确认通道。
    guessed = context.meta.get("guessed_type", "unknown")
    type_source = context.meta.get("guessed_type_source")
    auto_type = context.meta.get("guessed_type_auto_detected")
    type_note = context.meta.get("guessed_type_note")

    if type_source == "user_confirmed":
        type_line = f"**数据集类型**: {guessed}（来源：用户确认）"
        if auto_type and auto_type != guessed:
            type_line += f"；自动识别曾为 `{auto_type}`"
    else:
        type_line = f"**推测类型**: {guessed}（自动识别，未经确认）"

    lines = [f"**dataset_id**: {context.dataset_id or 'unknown'}", type_line]

    if type_note:
        lines.append(f"**类型说明**: {type_note}")

    if type_source != "user_confirmed" and str(guessed) == "unknown":
        lines.append(
            "> 未匹配到已知的数据格式。**如需固定类型**，可确认后写入数据集画像"
            "（跨会话生效），此后报告与质检都会采用该确认。"
        )

    # ---- 文件普查摘要（os 层面读大小，不进 df）----
    #
    # **单文件数据集回落**（2026-09-20）：单文件加载（如 calibration.json）不产生
    # 流登记表，此前会显示"文件数 0 / 总大小 0.0 B"，看起来像报告出错。此时改用
    # meta 里的 source / format / n_rows 描述该文件本身。
    n_files = 0
    total_bytes = 0
    fmt_dist: dict[str, int] = {}
    for s in streams:
        path = s.get("path")
        if not path:
            continue
        n_files += 1
        fmt = s.get("format", "?")
        fmt_dist[fmt] = fmt_dist.get(fmt, 0) + 1
        try:
            total_bytes += Path(path).stat().st_size
        except OSError:
            pass
    if n_files == 0:
        # 单文件：用 meta 的 source 描述（存在则给出真实大小）。
        src = context.meta.get("source")
        if src:
            n_files = 1
            try:
                total_bytes = Path(str(src)).stat().st_size
            except OSError:
                total_bytes = 0
            fmt = context.meta.get("format", "?")
            fmt_dist[fmt] = 1
            n_rows, n_cols = context.meta.get("n_rows"), context.meta.get("n_cols")
            lines.append(f"- **来源**: `{src}`")
            if n_rows is not None:
                lines.append(f"- **表规模**: {n_rows} 行 × {n_cols} 列")
    lines.append(f"- **文件数**: {n_files}")
    lines.append(f"- **总大小**: {_fmt_size(total_bytes)}")
    if fmt_dist:
        fmt_desc = ", ".join(f"{k}: {v}" for k, v in sorted(fmt_dist.items()))
        lines.append(f"- **格式分布**: {fmt_desc}")

    # ---- 流明细表 ----
    lines.append("")
    lines.append("**流明细**")
    if streams:
        lines.append("| 流 | 角色 | 格式 | 采样率/帧率 | 来源文件 |")
        lines.append("|---|---|---|---|---|")
        for s in streams:
            name = Path(s.get("path", "")).name if s.get("path") else "(main)"
            role = (s.get("role") or {}).get("role", s.get("kind", "unknown"))
            fmt = s.get("format", "?")
            mr = s.get("measured_rate")
            rate = (mr or {}).get("sample_rate_hz") if isinstance(mr, dict) else None
            rate_str = f"{rate} Hz" if rate is not None else "未知"
            src = s.get("path", "N/A")
            lines.append(f"| {name} | {role} | {fmt} | {rate_str} | `{src}` |")
    else:
        # 单文件数据集没有流概念；说明清楚而非笼统的"无流登记表"。
        if context.meta.get("source"):
            lines.append(
                "（单文件数据集，无多流结构；列清单见 `columns` 字段）"
            )
        else:
            lines.append("（无流登记表）")

    # ---- 模态矩阵（能力标签有无对照表）----
    #
    # **动态生成，不再硬编码**（真实缺陷修正，2026-09-21）：
    # 初版是写死的六行，而 capabilities 实际有 10 个键——漏掉了 has_pose
    # （手套数据的核心模态）、has_audio、has_hand_tracking 等。后果是在一份
    # 位姿追踪数据上，矩阵显示出"IMU ✗ / 力 ✗ / 标定 ✗"却**没有"位姿"这一行**，
    # 用户看到的全是"没有"，最关键的能力反而不见了。
    #
    # 现在遍历 capabilities 的全部键（含未来新增），中文名取自注册表；
    # 未登记的键会用原键名并标注"未登记"——让遗漏立刻可见，而不是静默丢失。
    lines.append("")
    lines.append("**模态矩阵**")
    matrix = build_capability_matrix(caps)
    if matrix["rows"]:
        lines.append("| 模态 | 有无 | 说明 |")
        lines.append("|---|---|---|")
        for row in matrix["rows"]:
            if row["present"] is True:
                mark = "✓"
            elif row["present"] is False:
                mark = "✗"
            else:
                mark = "未探测"
            # 说明列：有值时给白话；无值时给"没有≠缺陷"的安抚说明（若有）。
            note = row["what"]
            if row["present"] is not True and row["absent_note"]:
                note = f"{note}；{row['absent_note']}"
            lines.append(f"| {row['label']} | {mark} | {note} |")
    else:
        lines.append("（无能力标签）")

    if matrix["details"]:
        lines.append("")
        detail_str = "；".join(
            f"{d['label']}：{d['value']}"
            for d in matrix["details"] if d["value"] is not None
        )
        if detail_str:
            lines.append(f"附注：{detail_str}")

    if matrix["unregistered"]:
        lines.append("")
        lines.append(
            "> 注意：以下能力标签未登记中文名，已按原键名列出："
            + "、".join(f"`{k}`" for k in matrix["unregistered"])
        )

    return "\n".join(lines)


def _fmt_size(nbytes: int) -> str:
    """格式化字节数为可读大小。"""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} GB"


def _build_qc_section(context: RunContext) -> tuple[str, int]:
    """质检结果章节（检查明细表）。返回 (内容, 条目数)。"""
    qc = context.meta.get("qc", {})
    if not qc:
        return "（未运行质检工具，无质检结果。）", 0

    lines = []
    count = 0

    if "check_temporal_sync" in qc:
        ts = qc["check_temporal_sync"]
        result = ts.get("result", "unknown")
        vlevel = ts.get("verification_level", "")
        note = "（基于时间戳一致性，物理级未验证）" if vlevel == "timestamp_consistency" else ""
        lines.append(f"- 时间对齐（check_temporal_sync）: **{result}**{note}")
        detail = ts.get("detail") or {}
        # 各流时间戳检查明细。
        sc = detail.get("stream_checks", {})
        if sc:
            lines.append("")
            lines.append("  各流时间戳检查：")
            lines.append("  | 流 | 乱序 | 重复 | 丢帧率 | 实测采样率(Hz) |")
            lines.append("  |---|---|---|---|---|")
            for name, v in sc.items():
                if v.get("status") == "skipped":
                    lines.append(f"  | {Path(name).name} | - | - | - | - (skipped: {v.get('reason','')}) |")
                else:
                    lines.append(f"  | {Path(name).name} | {v.get('disorder_count','?')} | {v.get('duplicate_count','?')} | {v.get('frame_loss_ratio','?')} | {v.get('actual_rate_hz','?')} |")
        # 残差与漂移。
        resid = detail.get("residuals", {})
        for name, v in resid.items():
            lines.append(f"  - 对齐残差（{Path(name).name} 相对基线）: max={v.get('residual_max_ms','?')} ms, mean={v.get('residual_mean_ms','?')} ms")
        for name, v in detail.get("drift", {}).items():
            lines.append(f"  - 漂移（{Path(name).name}）: 斜率 {v.get('drift_slope_ms_per_s','?')} ms/s, 检出={v.get('drift_detected','?')}")
        count += 1

    if "check_sensor_sanity" in qc:
        sn = qc["check_sensor_sanity"]
        result = sn.get("result", "unknown")
        const = sn.get("constant_channels", [])
        extra = f"，恒定通道 {len(const)} 个" if const else ""
        lines.append(f"- 传感器合理性（check_sensor_sanity）: **{result}**{extra}")
        detail = sn.get("detail") or {}
        streams_detail = detail.get("streams", {})
        if streams_detail:
            lines.append("")
            lines.append("  各流检查明细：")
            lines.append("  | 流 | 类型 | 单位 | 重力 | 陀螺仪 | 饱和 | 饱和比例 | NaN比例 |")
            lines.append("  |---|---|---|---|---|---|---|---|")
            for name, v in streams_detail.items():
                if v.get("type") == "imu":
                    lines.append(f"  | {Path(name).name} | imu | {v.get('accel_unit','?')} | {v.get('gravity_verdict','?')} | {v.get('gyro_verdict','?')} | {v.get('saturation_verdict','?')} | {v.get('gyro_saturation_ratio','?')} | {v.get('nan_ratio','?')} |")
                else:
                    lines.append(f"  | {Path(name).name} | force | - | - | - | {v.get('saturation_verdict','?')} | {v.get('saturation_ratio','?')} | {v.get('nan_ratio','?')} |")
        if const:
            lines.append(f"  - 恒定通道（疑似掉线）: {', '.join(const)}")
        count += 1

    return "\n".join(lines), count


def _build_stats_section(findings: list[dict]) -> tuple[str, int]:
    """任务级统计章节，保留推测标注与关键数字。返回 (内容, 条目数)。"""
    stat_items = [f for f in findings if f.get("type") == "stat"]
    if not stat_items:
        return "（无任务级统计。）", 0

    lines = []
    for f in stat_items:
        metric = f.get("metric", "task_level")
        summary = f.get("summary", "")
        m = f.get("metrics") or {}
        lines.append(f"- **{metric}**: {summary}")

        # 关键数字明细（全部来自 metrics 的真实计算）。
        n_ep = m.get("n_episodes")
        sr = m.get("success_rate")
        if n_ep is not None:
            lines.append(f"  - episode 数: {n_ep}")
        if sr is not None:
            lines.append(f"  - 成功率: {sr}")

        rom = m.get("joint_range_of_motion")
        if rom:
            lines.append("  - 关节活动范围:")
            for col, v in rom.items():
                lines.append(f"    - {col}: min={v.get('min')}, max={v.get('max')}, range={v.get('range')}")

        ed = m.get("episode_duration")
        if ed:
            lines.append(f"  - episode 时长分布: min={ed.get('min')}, 中位={ed.get('median')}, max={ed.get('max')}")

        oe = m.get("outlier_episodes")
        if oe:
            outlier_desc = ", ".join(f"episode {o.get('episode')}(时长 {o.get('duration')})" for o in oe.get("outliers", []))
            lines.append(f"  - 离群 episode ({oe.get('method','IQR')}, k={oe.get('k')}): {outlier_desc if outlier_desc else '无'}")

        # 该条目的推测/注意事项标注（如 success 聚合规则为推测）。
        for note in f.get("semantic_notes", []):
            lines.append(f"  - 标注（推测/注意事项）: {note}")
    return "\n".join(lines), len(stat_items)


def _build_charts_section(findings: list[dict]) -> tuple[str, int]:
    """图表章节，以相对路径插入图片。返回 (内容, 条目数)。"""
    chart_items = [f for f in findings if f.get("type") == "chart"]
    if not chart_items:
        return "（无图表。）", 0

    lines = []
    for f in chart_items:
        fp = f.get("file_path", "")
        basename = Path(fp).name if fp else ""
        title = f.get("title", "chart")
        desc = f.get("description", "")
        spec = f.get("plot_spec", {})
        # 图片引用用相对路径（../outputs/<basename>），保证 Markdown 可移植。
        img_ref = f"![](../outputs/{basename})" if basename else ""
        lines.append(f"### {title}\n")
        if img_ref:
            lines.append(img_ref)
        if desc:
            lines.append(f"\n*{desc}*")
        if spec:
            x = spec.get("x_axis", "?")
            y = spec.get("y_axis", [])
            grouped = spec.get("grouped_by")
            lines.append(f"\n- 坐标：x={x}，y={y}，分组={grouped}")
        lines.append("")
    return "\n".join(lines), len(chart_items)


def _build_limitations_section(context: RunContext, findings: list[dict]) -> tuple[str, int]:
    """局限性说明章节：自动汇总 skipped/unknown/推测项。返回 (内容, 条目数)。

    通用化：收集**所有** findings 条目的 semantic_notes 字段（推测/注意事项类），
    这是报告的诚实底线，不应依赖单个工具记得带。
    """
    items: list[str] = []

    # 通用化：汇总所有 findings 的 semantic_notes 字段（任意工具，不只 stat）。
    for f in findings:
        for note in f.get("semantic_notes", []):
            tool = f.get("tool", "工具")
            items.append(f"{tool} 标注：{note}")

    # 兜底：从 stat 条目 summary 中提取推测/无法确定/未检测标注（无 semantic_notes 时）。
    for f in findings:
        if f.get("type") != "stat" or f.get("semantic_notes"):
            continue
        summary = f.get("summary", "")
        for kw in ("推测", "无法确定", "未检测", "未质检", "不可信"):
            if kw in summary:
                items.append(f"统计结论含「{kw}」：{summary}")
                break

    # 从 qc 判定的 warn/fail 提取局限性。
    qc = context.meta.get("qc", {})
    for tool_name, entry in qc.items():
        if entry.get("result") in ("warn", "fail"):
            items.append(f"质检 {tool_name} 判定为 {entry.get('result')}，存在需人工确认的疑点")

    # 推测类型 unknown。
    if context.meta.get("guessed_type") == "unknown":
        items.append("数据集推测类型为 unknown，类型归属未确定")

    if not items:
        return "（无已知局限性。）", 0
    return "\n".join(f"- {i}" for i in items), len(items)


def generate_report_impl(context: RunContext, title: str | None = None) -> dict[str, Any]:
    """汇总当前会话的分析结果为 Markdown 报告。

    Args:
        context: 运行时上下文。
        title: 可选，报告标题。

    Returns:
        dict，含 success、file_path、title、section_counts、dataset、findings、
        user_message；无分析结果时返回结构化错误。
    """
    findings = context.findings
    dataset_id = context.dataset_id

    # **无 findings 时仍可出报告**（2026-09-20 修复）。
    #
    # 此前直接返回 no_findings 错误，但报告的数据集概况与质检章节本就来自
    # ``meta``（能力标签、流登记表、qc 明细），**不依赖 findings**——只要数据集
    # 已加载就有内容可写。真实事故：用户做完 load_dataset + profile_data +
    # inspect_streams + check_temporal_sync 后要求"生成 md 报告"，工具却回
    # "当前会话尚无分析结果"（因为这几个工具都不写 findings），用户只能自己
    # 手工整理文档。
    #
    # 现在只在**数据集也未加载**时才拒绝；否则照常出报告，缺失章节如实标注为空。
    #
    # 判据用 ``meta`` **内容**而非 ``dataset_id``：后者可能被手工置上（测试或
    # 旧会话残留）而 meta 仍为空，此时报告其实无任何可写内容——只看 id 会
    # 产出"全空报告"，比明确拒绝更糟。
    has_dataset = bool(context.meta) or context.df is not None
    if not findings and not has_dataset:
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载数据集，也无分析结果",
            "user_message": (
                "当前会话既未加载数据集、也无分析结果，无法生成报告。"
                "请先调用 load_dataset 加载数据。"
            ),
            "dataset": dataset_id,
        }

    title_safe = title if (title and title.isascii()) else f"{dataset_id or 'dataset'} 分析报告"

    # 确定性拼装各章节。
    overview = _build_dataset_overview(context)
    qc_section, n_qc = _build_qc_section(context)
    stats_section, n_stats = _build_stats_section(findings)
    charts_section, n_charts = _build_charts_section(findings)
    limits_section, n_limits = _build_limitations_section(context, findings)

    md = f"""# {title_safe}

## 1. 数据集概况

{overview}

## 2. 质检结果

{qc_section}

## 3. 任务级统计

{stats_section}

## 4. 图表

{charts_section}

## 5. 局限性说明

{limits_section}
"""

    path = _output_report_path(context)
    path.write_text(md, encoding="utf-8")

    section_counts = {
        "qc": n_qc,
        "stats": n_stats,
        "charts": n_charts,
        "limitations": n_limits,
    }

    report_finding = {
        "tool": "generate_report",
        "type": "report",
        "file_path": str(path),
        "title": title_safe,
        "section_counts": section_counts,
    }
    context.findings.append(report_finding)

    return {
        "success": True,
        "dataset": dataset_id,
        "file_path": str(path),
        "title": title_safe,
        "section_counts": section_counts,
        "findings": [report_finding],
        "user_message": (
            f"已生成 Markdown 报告，保存至 {path}。"
            f"章节条目：质检 {n_qc}、统计 {n_stats}、图表 {n_charts}、局限性 {n_limits}。"
        ),
    }


@tool
def generate_report(
    wrapper: RunContextWrapper[RunContext],
    title: str | None = None,
) -> dict:
    """汇总当前会话的分析结果为 Markdown 报告，保存到 outputs/。

    报告由工具确定性拼装（LLM 不参与正文生成），内容来自 findings、meta["qc"]、
    流登记表与能力标签。每个数字都来自这些来源，不编造。

    Args:
        title: 可选，报告标题。

    Returns:
        dict，含 success、file_path、title、section_counts、dataset；无分析结果时
        返回结构化错误。
    """
    return generate_report_impl(wrapper.context, title)
