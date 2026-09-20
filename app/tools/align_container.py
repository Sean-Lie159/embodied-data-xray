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


def _timing_from_array(values: Any, unit: str) -> dict[str, Any] | None:
    """从**已知单位**的时间戳数组算统计（采样率 / 跨度 / 缺口 / 流形态）。

    与 :func:`_stream_timing` 的区别：单位由调用方给定（如 MCAP 容器时间恒为
    纳秒），不做单位推断——避免对已明确的数据再猜一次。

    **流形态辨别**（2026-09-14 真实缺陷）：突发型流（MCAP 的 /tf、IMU 突发内
    微秒级、突发间毫秒级静默）的"缺口"是**伪影**——按"间隔 > 5×中位"统计会
    把每个突发间静默都算成丢包（实测 /tf 报出 121620 个"缺口"，实为噪声）。
    故对 burst 流把缺口计数置 None 并附说明，改报突发段数（与
    check_temporal_sync 的既有口径一致）。

    Args:
        values: 时间戳序列（array-like，数值）。
        unit: 单位（须在 TIME_UNITS 内，如 "ns"）。

    Returns:
        dict（start_ns/end_ns/span_s/n/rate_hz/n_gaps/max_gap_s/shape）；
        不足 2 点返回 None。
    """
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return None
    from app.tools.timestamp_units import TIME_UNITS, to_ns

    if unit in TIME_UNITS:
        arr = to_ns(arr, unit)
    if len(arr) < 2:
        return None
    span = float(arr[-1] - arr[0])
    diffs = np.diff(arr)
    med = float(np.median(diffs)) if len(diffs) else 0.0
    n_gaps = int((diffs > 5 * med).sum()) if med > 0 else 0
    max_gap = float(diffs.max()) if len(diffs) else 0.0

    # 流形态：复用 check_temporal_sync 的唯一实现（阈值同源，避免两套口径）。
    shape = "periodic"
    try:
        from app.config import get_settings
        from app.tools.check_temporal_sync import _classify_stream_shape

        shape = _classify_stream_shape(arr, get_settings())
    except Exception:  # noqa: BLE001 - 形态判定失败不阻塞对齐
        shape = "periodic"

    out: dict[str, Any] = {
        "start_ns": float(arr[0]),
        "end_ns": float(arr[-1]),
        "span_s": round(span / 1e9, 3),
        "n": int(len(arr)),
        "rate_hz": round((len(arr) - 1) / (span / 1e9), 3) if span > 0 else None,
        "unit": unit,
        "shape": shape,
    }
    if shape == "burst":
        # 突发型：缺口计数不适用（突发间静默会被误报），改报突发段数。
        big = int((diffs > 2.0 * med).sum()) if med > 0 else 0
        out["n_gaps"] = None
        out["n_bursts"] = big + 1 if len(diffs) else 0
        out["gap_status"] = "not_applicable"
    else:
        out["n_gaps"] = n_gaps
        out["max_gap_s"] = round(max_gap / 1e9, 3)
    return out


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
    arr = np.asarray(result.timestamp, dtype=float)
    # 时间戳单位：按列名/量级推断（复用既有单位链路）。
    from app.tools.timestamp_units import infer_unit

    unit = infer_unit(arr, result.timestamp_column or "")["unit"]
    return _timing_from_array(arr, unit)


def _collect_camera_names(context: RunContext, container_file: str) -> dict[str, Any]:
    """比对容器内相机时间戳路数与**目录侧同名 txt** 的路数（如实标注不对称）。

    为什么需要（2026-09-20，用户确认的取舍 3）：真实数据集 ``2655849`` 存在
    一路**不对称**——h5 内 ``timestamp/camera/`` 只有 6 路（hand_left/right_color、
    head_color、head_depth、head_stereo_left/right），而 ``camera/`` 目录下有
    **9 路**同名 txt（另有 head_back_fisheye、head_left_fisheye、
    head_right_fisheye）。用户问"h5 时间戳与 camera/ 下每个同名 txt 的对齐"
    时，若工具只报 h5 侧的 6 路，会让人误以为这 6 路就是全部相机；若反过来
    按 9 路去找 h5 对应物，又会找不到 3 路。

    本函数**只标注、不补齐**（不臆造 h5 里不存在的相机时间戳），把"哪些相机
    只在一侧存在"如实列给用户，由用户判断是否属预期。

    Args:
        context: 运行时上下文（取 meta.streams）。
        container_file: 当前对齐的容器文件绝对路径。

    Returns:
        dict，含 h5_cameras / dir_cameras / only_in_h5 / only_in_dir / note；
        两侧任一为空或无不对称时 note 为 None。
    """
    h5_cams: set[str] = set()
    dir_cams: set[str] = set()
    for s in context.meta.get("streams", []):
        path = str(s.get("path", ""))
        file_part, _, sub = path.partition("::")
        # 容器侧：本 h5 的 timestamp/camera/<name> 节点。
        if (s.get("format") == "h5" and file_part == container_file
                and "timestamp/camera/" in sub):
            h5_cams.add(sub.rsplit("/", 1)[-1].lower())
        # 目录侧：camera/<相机名>/ 下的同名 txt（即 txt 的父目录名）。
        if s.get("format") in ("txt", "csv") and "camera" in Path(file_part).parts:
            parts = Path(file_part).parts
            try:
                idx = [p.lower() for p in parts].index("camera")
            except ValueError:
                continue
            if idx + 1 < len(parts) - 1:  # 还有更深一层 = 相机目录名
                dir_cams.add(parts[idx + 1].lower())

    only_h5 = sorted(h5_cams - dir_cams)
    only_dir = sorted(dir_cams - h5_cams)
    note: str | None = None
    if h5_cams and dir_cams and (only_h5 or only_dir):
        bits: list[str] = []
        if only_dir:
            bits.append(
                f"仅 camera/ 目录有、h5 内无对应时间戳的相机（{len(only_dir)} 路）："
                f"{only_dir}"
            )
        if only_h5:
            bits.append(
                f"仅 h5 内有时间戳、camera/ 目录无同名 txt 的相机（{len(only_h5)} 路）："
                f"{only_h5}"
            )
        note = (
            f"**相机路数不对称**：h5 内 timestamp/camera/ 有 {len(h5_cams)} 路，"
            f"camera/ 目录下有 {len(dir_cams)} 路同名 txt，并非一一对应。"
            + "；".join(bits)
            + "。工具仅如实标注、不补齐缺失侧——这些相机能否跨侧对齐需按上表"
            "实际情况判断。"
        )
    return {
        "h5_cameras": sorted(h5_cams),
        "dir_cameras": sorted(dir_cams),
        "only_in_h5": only_h5,
        "only_in_dir": only_dir,
        "note": note,
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
    #
    # **MCAP 走批量路径**（2026-09-14 性能事故）：mcap 库的
    # `iter_messages(topics=[t])` 过滤开销与文件总消息数成正比（实测 1.86GB /
    # 419 万条消息的容器：遍历全部 32.8s，而"只取单个 topic"也要 16s）——
    # 逐 topic 过滤是 O(全部消息) × topic 数，26 个 topic 实测 431 秒。
    # 故先一次遍历分桶取齐所有 topic 的时间戳，再逐个组装（总耗时 ≈ 33s）。
    mcap_paths: dict[str, str] = {}   # topic -> file（MCAP 子流）
    if any(s.get("format") == "mcap" for s in subs):
        for s in subs:
            if s.get("format") != "mcap":
                continue
            file_part, sub_name = (s.get("path", "").partition("::")[0],
                                   s.get("path", "").partition("::")[2])
            if sub_name:
                mcap_paths[sub_name] = file_part
    mcap_ts: dict[str, Any] = {}
    if mcap_paths:
        from app.tools.mcap_reader import read_mcap_all_timestamps

        # 同一容器文件（本工具的 target_file）一次性取齐。
        mcap_ts = read_mcap_all_timestamps(target_file, list(mcap_paths))

    rows: list[dict[str, Any]] = []
    for s in subs:
        spec = s.get("path", "")
        sub_name = spec.partition("::")[2]
        if s.get("format") == "mcap" and sub_name in mcap_ts:
            timing = _timing_from_array(
                mcap_ts[sub_name]["log_time_ns"], "ns")
        else:
            timing = _stream_timing(spec)
        label = s.get("semantic_label") or s.get("kind") or ""
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
    burst_subs: list[str] = []
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
        # 缺口只对周期型流报（突发型的"缺口"是突发间静默的伪影，见
        # _timing_from_array 的说明——此前误报 /tf 有 12 万个"缺口"）。
        if r.get("shape") == "burst":
            burst_subs.append(r["sub"])
            notes.append(
                f"突发型流（{r.get('n_bursts')} 个突发段）：丢包/缺口口径不适用，"
                "突发间静默不算丢包"
            )
        elif r.get("n_gaps"):
            notes.append(f"{r['n_gaps']} 个缺口（最大 {r['max_gap_s']}s）")
        r["notes"] = notes
        if notes:
            warnings.append(f"{r['sub']}：{'；'.join(notes)}")

    # 多时钟提示（**诚实降级**）：本工具取的是**容器时间戳**
    # （MCAP 的 log_time / publish_time）。若容器为突发型而消息体自带传感器
    # 时间戳（如 /tf 的 data.transforms[].timestamp_us），则容器时间可能只是
    # 批量写入时间，据此判定的"采样率/缺口"不代表传感器真实节拍——必须提示
    # 用户改用 time_column 指定传感器时间列复核，不得让结论被当作定论。
    clock_note: str | None = None
    if burst_subs:
        clock_note = (
            f"注意：{len(burst_subs)} 个子流（如 {burst_subs[0]}）在容器时间口径下"
            "呈突发型。若其消息体自带传感器时间戳（MCAP 常见 data.*_time_us 字段），"
            "则容器时间可能只是**批量写入时间**而非传感器采样节拍——"
            "此时上述采样率与突发段数不代表真实节拍。建议用 check_temporal_sync 的 "
            "time_column 参数指定传感器时间列复核后再下结论。"
        )

    # 相机路数不对称标注（用户确认的取舍 3，2026-09-20）：如实列出"哪些相机
    # 只在一侧存在"，不补齐缺失侧。真实案例 h5 6 路 vs 目录 9 路。
    camera_note = _collect_camera_names(context, target_file)

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
        "burst_streams": burst_subs,
        "clock_note": clock_note,
        "camera_coverage": camera_note,
        "user_message": (
            f"容器 {Path(target_file).name}：{len(ok_rows)}/{len(subs)} 个子流"
            f"有可用时间戳；主时钟为 {master['sub']}（跨度 {master['span_s']}s，"
            f"约 {master['rate_hz']}Hz）。"
            + (f" 发现 {len(warnings)} 条可疑：{'；'.join(warnings[:3])}"
               + ("…" if len(warnings) > 3 else "") if warnings else " 各子流首尾对齐良好。")
            + (" （子流数超上限，已截断）" if truncated else "")
            + (f" {clock_note}" if clock_note else "")
            + (f" {camera_note['note']}" if camera_note.get("note") else "")
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

    **批量语义（重要）**：本工具一次调用返回容器内**全部**子流的对齐全貌，
    问"各子流对齐如何 / 有没有截断"时**只调用一次**，不要逐子流反复调用
    （align_container_streams 已是全量口径，逐流调用纯属重复且拖长整轮耗时）。

    Args:
        container: 可选，容器文件名子串（如 "dataset.h5"）；省略时自动选
            子流最多的容器。
        max_streams: 单次对齐的子流上限（默认 80，超出截断并标注）。

    Returns:
        dict，含 container、master（主时钟子流）、streams（逐子流 采样率/
        跨度/样本数/缺口数/相对主时钟首尾偏移/截断标注）、warnings、
        camera_coverage（相机路数对称性：h5 内路数 vs camera/ 目录 txt 路数，
        含 only_in_h5 / only_in_dir / note）、user_message。
        **相机路数可能不对称**——本工具只如实标注存在哪侧，不补齐缺失侧。
    """
    return align_container_streams_impl(wrapper.context, container, max_streams)
