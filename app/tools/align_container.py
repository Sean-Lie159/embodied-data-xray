"""容器内多子流对齐（h5 节点 / mcap topic 一次对齐全部子流）。

**为什么需要**：h5 / mcap 是"单容器多子流"——各子流的采样率、时间跨度、
缺口互不相同（真实案例：dataset.h5 的 68 个节点里，action 100Hz、相机
30Hz、位姿 30Hz 但尾部截断 96.8s）。既有 check_temporal_sync 能逐流转，
但要逐个子流传 table 或 streams 参数，没有"一次看完整容器对齐全貌"的入口。

本工具复用统一读取注册表的 ``candidates()`` 枚举子流（重构后已就绪），
逐子流取时间戳，产出**跨子流的对齐全貌表**：
- 采样率 / 时间跨度 / 样本数 / 缺口数（每子流一行）；
- 相对主时钟（跨度最大者）的首尾偏移（容器内各子流是否同起同止）；
- 明显的尾部/头部截断自动标注（真实案例：位姿尾部 96.8s 缺口）。

边界：本工具做**子流级概览**（谁对不齐、差多少），不做逐帧残差与漂移
（那属 check_temporal_sync 的职责，可对单个子流精算）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools._readers import ReadRequest, read_stream

# 单次对齐的子流上限（防容器子流过多时上下文爆炸）。
_MAX_SUBSTREAMS = 80
# 尾部/头部截断判定阈值：首尾偏移超过该比例即标注（相对主时钟跨度）。
_TRUNCATION_RATIO = 0.05


def _stream_timing(path_spec: str) -> dict[str, Any] | None:
    """取单个子流的时间戳统计（采样率 / 跨度 / 缺口）。

    Args:
        path_spec: 流路径（``"<file>::<sub>"``）。

    Returns:
        dict（start_ns/end_ns/span_s/n/rate_hz/n_gaps/max_gap_s），失败返回 None。
    """
    result = read_stream(ReadRequest(path_spec=path_spec, want="timestamp"))
    if not result.ok or result.timestamp is None:
        return None
    arr = np.sort(np.asarray(result.timestamp, dtype=float))
    # 时间戳单位：按列名/量级推断并归一化到纳秒（复用既有单位链路）。
    from app.tools.timestamp_units import TIME_UNITS, infer_unit, to_ns

    unit = infer_unit(arr, result.timestamp_column or "")["unit"]
    if unit in TIME_UNITS:
        arr = to_ns(arr, unit)
    if len(arr) < 2:
        return None
    span = float(arr[-1] - arr[0])
    diffs = np.diff(arr)
    med = float(np.median(diffs)) if len(diffs) else 0.0
    n_gaps = int((diffs > 5 * med).sum()) if med > 0 else 0
    max_gap = float(diffs.max()) if len(diffs) else 0.0
    return {
        "start_ns": float(arr[0]),
        "end_ns": float(arr[-1]),
        "span_s": round(span / 1e9, 3),
        "n": int(len(arr)),
        "rate_hz": round((len(arr) - 1) / (span / 1e9), 3) if span > 0 else None,
        "n_gaps": n_gaps,
        "max_gap_s": round(max_gap / 1e9, 3),
        "unit": unit,
    }


def align_container_streams_impl(
    context: RunContext,
    container: str | None = None,
    max_streams: int = _MAX_SUBSTREAMS,
) -> dict[str, Any]:
    """对齐容器内全部子流（h5 节点 / mcap topic），产出对齐全貌表。

    Args:
        context: 运行时上下文（取 meta.streams）。
        container: 可选，容器文件名子串（如 "dataset.h5"）；省略时自动选
            子流最多的容器。
        max_streams: 单次对齐的子流上限（超出则截断并标注）。

    Returns:
        dict，含 container、n_substreams、master（主时钟子流）、streams
        （逐子流统计 + 相对主时钟偏移 + 截断标注）、warnings。
    """
    streams = context.meta.get("streams", [])
    containers: dict[str, list[dict[str, Any]]] = {}
    for s in streams:
        fmt = s.get("format")
        if fmt in ("h5", "mcap"):
            file_part, sub = (s.get("path", "").partition("::")[0],
                              s.get("path", "").partition("::")[2])
            if sub:
                containers.setdefault(file_part, []).append(s)

    if not containers:
        return {
            "success": False,
            "error": "no_container",
            "user_message": (
                "当前数据集中没有可对齐的容器型流（h5 节点 / mcap topic）。"
                "目录型多文件数据集请直接用 check_temporal_sync。"
            ),
        }

    if container:
        hint = container.lower()
        matched = [(f, ss) for f, ss in containers.items()
                   if hint in Path(f).name.lower()]
        if not matched:
            return {
                "success": False,
                "error": "container_not_found",
                "user_message": (
                    f"未找到容器 {container!r}。当前可用容器："
                    f"{[Path(f).name for f in containers]}。"
                ),
            }
        target_file, subs = matched[0]
    else:
        target_file, subs = max(containers.items(), key=lambda kv: len(kv[1]))

    truncated = len(subs) > max_streams
    subs = subs[:max_streams]

    # 逐子流取时间统计（只读时间戳列，不读全量数据）。
    rows: list[dict[str, Any]] = []
    for s in subs:
        spec = s.get("path", "")
        timing = _stream_timing(spec)
        label = s.get("semantic_label") or s.get("kind") or ""
        sub_name = spec.partition("::")[2]
        if timing is None:
            rows.append({
                "sub": sub_name, "label": label, "status": "no_timestamp",
                "reason": "该子流无时间戳字段（或读取失败）",
            })
            continue
        rows.append({"sub": sub_name, "label": label, "status": "ok", **timing})

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        return {
            "success": False,
            "error": "no_timestamp",
            "user_message": (
                f"容器 {Path(target_file).name} 的 {len(rows)} 个子流均无可用"
                "时间戳字段，无法对齐。"
            ),
        }

    # 主时钟 = 跨度最大者（覆盖最全，作对齐参照）。
    master = max(ok_rows, key=lambda r: r["span_s"])
    master_start, master_span = master["start_ns"], master["span_s"]

    warnings: list[str] = []
    for r in ok_rows:
        head_off = (r["start_ns"] - master_start) / 1e9
        tail_off = (r["end_ns"] - (master_start + master_span * 1e9)) / 1e9
        r["head_offset_s"] = round(head_off, 3)
        r["tail_offset_s"] = round(tail_off, 3)
        # 截断标注：首/尾偏移超过主时钟跨度的阈值比例。
        thr = master_span * _TRUNCATION_RATIO
        notes = []
        if abs(head_off) > thr:
            notes.append(f"起始比主时钟{'晚' if head_off > 0 else '早'} "
                         f"{abs(head_off):.1f}s")
        if abs(tail_off) > thr:
            notes.append(f"结束比主时钟{'早' if tail_off < 0 else '晚'} "
                         f"{abs(tail_off):.1f}s（疑似截断）")
        if r["n_gaps"]:
            notes.append(f"{r['n_gaps']} 个缺口（最大 {r['max_gap_s']}s）")
        r["notes"] = notes
        if notes:
            warnings.append(f"{r['sub']}：{'；'.join(notes)}")

    return {
        "success": True,
        "dataset": context.dataset_id,
        "container": Path(target_file).name,
        "n_substreams": len(subs),
        "n_with_timestamp": len(ok_rows),
        "truncated": truncated,
        "master": {"sub": master["sub"], "span_s": master["span_s"],
                   "rate_hz": master["rate_hz"]},
        "streams": rows,
        "warnings": warnings,
        "user_message": (
            f"容器 {Path(target_file).name}：{len(ok_rows)}/{len(subs)} 个子流"
            f"有可用时间戳；主时钟为 {master['sub']}（跨度 {master['span_s']}s，"
            f"约 {master['rate_hz']}Hz）。"
            + (f" 发现 {len(warnings)} 条可疑：{'；'.join(warnings[:3])}"
               + ("…" if len(warnings) > 3 else "") if warnings else " 各子流首尾对齐良好。")
            + (" （子流数超上限，已截断）" if truncated else "")
        ),
    }


@tool
def align_container_streams(
    wrapper: RunContextWrapper[RunContext],
    container: str | None = None,
    max_streams: int = _MAX_SUBSTREAMS,
) -> dict:
    """对齐容器内全部子流（h5 节点 / mcap topic），产出对齐全貌表。

    适用：单文件多子流的容器（.h5 / .mcap）——各子流采样率与时间跨度不同，
    需一次看清"谁与谁对不齐、差多少、哪里有缺口/截断"。目录型多文件数据集
    请改用时间同步检查（check_temporal_sync）。

    Args:
        container: 可选，容器文件名子串（如 "dataset.h5"）；省略时自动选
            子流最多的容器。
        max_streams: 单次对齐的子流上限（默认 80，超出截断并标注）。

    Returns:
        dict，含 container、master（主时钟子流）、streams（逐子流 采样率/
        跨度/样本数/缺口数/相对主时钟首尾偏移/截断标注）、warnings、
        user_message。
    """
    return align_container_streams_impl(wrapper.context, container, max_streams)
