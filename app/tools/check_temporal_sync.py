"""时间同步检查工具（质检层，v1）。

仅基于时间戳一致性做流间同步与漂移检测（不做光流互相关）。数据来自流登记表，
逐条流按需读取时间戳列（不装入 context.df），测完释放。verification_level 明确
标注为 "timestamp_consistency"，物理级对齐需互相关实测（未来 v2）。

判定三档：pass（无漂移且残差小）/ warn（残差接近阈值或有疑点）/ fail（丢帧超阈
或检出漂移）。每项带测量值、阈值、受影响 episode 清单。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.config import get_settings
from app.tools.inspect_streams import _read_timestamp_only
from app.tools.timestamp_units import FRAME_UNIT, TIME_UNITS, to_ns, unit_to_ns_factor


def _read_stream_timestamps(stream: dict[str, Any]) -> np.ndarray | None:
    """按需读取单条流的原始时间戳序列（文件顺序，不排序）。

    Args:
        stream: 流登记项（含 path / format / kind）。

    Returns:
        原始时间戳 numpy 数组（数值化、去 NaN）；读取失败或非表格流返回 None。
    """
    if stream.get("kind") == "video":
        return None  # 视频流时间戳由 ffprobe 提供，见 _video_ideal_ts
    path = stream.get("path", "")
    fmt = stream.get("format", "")
    if not path or not Path(path).exists():
        return None
    try:
        ts = _read_timestamp_only(path, fmt)
        if ts is None:
            return None
        arr = np.asarray(pd_to_numeric(ts), dtype=float)
        arr = arr[~np.isnan(arr)]
        return arr if len(arr) > 0 else None
    except Exception:  # noqa: BLE001
        return None


def pd_to_numeric(series) -> np.ndarray:
    """将 Series 安全转数值数组。"""
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").to_numpy()


def _to_seconds(ts: np.ndarray, unit_info: dict[str, Any] | None) -> np.ndarray:
    """把时间戳换算为秒（供单流检查用）。

    已归一化到纳秒的（normalized=True）除以 1e9；未归一化的（unknown/frame_index）
    按原值当作秒（仅用于乱序/重复/丢帧等相对检查，不涉绝对时间）。

    Args:
        ts: 时间戳数组（可能为纳秒或原值）。
        unit_info: _normalize_to_ns 返回的单位说明。

    Returns:
        秒级时间戳数组。
    """
    if unit_info and unit_info.get("normalized"):
        return np.asarray(ts, dtype=float) / 1e9
    return np.asarray(ts, dtype=float)


def infer_metainfo_unit(arr: np.ndarray, col_name: str) -> str:
    """推断 metainfo 时间戳列的单位。

    规则：列名含真实时钟标记（utc / epoch / real / host）→ 用 infer_unit 按量级推断
    （如 exposure_start_utc_ns → ns）；列名是帧序号类（pts / frame / packet / index）→
    直接判为 frame_index（无物理时间，不参与跨流对齐）；否则退回 infer_unit。

    Args:
        arr: 时间戳数值数组。
        col_name: 时间戳列名。

    Returns:
        单位名（s/ms/us/ns/frame_index/unknown）。
    """
    from app.tools.timestamp_units import infer_unit

    lower = col_name.lower()
    if any(k in lower for k in ("utc", "epoch", "real", "host")):
        return infer_unit(arr)["unit"]
    if any(k in lower for k in ("pts", "frame", "packet", "index", "idx")):
        return FRAME_UNIT
    return infer_unit(arr)["unit"]


def _normalize_to_ns(ts: np.ndarray, unit: str) -> tuple[np.ndarray, dict[str, Any]]:
    """把原始时间戳归一化到纳秒基准，返回 (归一化数组, 单位说明)。

    时间单位（s/ms/us/ns）直接换算；frame_index（帧序号）无物理时间，不换算、
    不参与跨流对齐残差判定，仅用于单流检查（乱序/重复/丢帧）。单位未知时保持
    原值并注明"未归一化"。

    Args:
        ts: 原始时间戳数组。
        unit: 流的时间戳单位（来自嗅探推断）。

    Returns:
        (归一化数组, 说明 dict)，说明含 original_unit、normalized、basis。
    """
    if unit in TIME_UNITS:
        return to_ns(ts, unit), {
            "original_unit": unit,
            "normalized": True,
            "basis": f"原始单位 {unit}，已归一化到纳秒（×{unit_to_ns_factor(unit)}）",
        }
    if unit == FRAME_UNIT:
        return ts, {
            "original_unit": FRAME_UNIT,
            "normalized": False,
            "basis": "帧序号时间戳（无物理时间），不参与跨流对齐残差判定",
        }
    # 单位未知：不硬猜，但为兼容历史数据（未标注单位的秒级时间戳），按秒换算到纳秒。
    # 说明标注"默认按秒"，使下游可见这是兜底假设而非实测单位。
    return ts * 1e9, {
        "original_unit": unit or "unknown",
        "normalized": True,
        "basis": "单位未知，按秒换算到纳秒（兜底假设，非实测单位；真实采集应带 timestamp_unit）",
    }


def _nominal_rate(stream: dict[str, Any]) -> float | None:
    """从流登记表/meta 读取标称采样率；缺省返回 None。"""
    rate = stream.get("nominal_rate_hz")
    if rate is None:
        rate = stream.get("measured_rate", {}).get("sample_rate_hz")
    return float(rate) if isinstance(rate, (int, float)) and rate > 0 else None


def _single_stream_checks(
    ts: np.ndarray, nominal: float | None
) -> dict[str, Any]:
    """单流时间戳检查：单调性、重复、丢帧率、实际采样率、时长。

    Args:
        ts: 原始顺序时间戳数组（**秒**，调用方已统一换算为秒）。
        nominal: 标称采样率（Hz），None 时跳过实际 vs 标称对比。

    Returns:
        dict，含各项测量值与判定标记。
    """
    ts_sorted = np.sort(ts)
    diffs = np.diff(ts_sorted)

    # 单调性（乱序计数）：原始顺序中后项小于前项的次数。
    disorder = int(np.sum(np.diff(ts) < 0)) if len(ts) > 1 else 0

    # 重复时间戳计数。
    duplicates = int(np.sum(np.diff(ts_sorted) == 0)) if len(ts) > 1 else 0

    # 实际采样率（中位数间隔倒数）。
    actual_rate = None
    if len(diffs) > 0:
        med = float(np.median(diffs))
        if med > 0:
            actual_rate = round(1.0 / med, 3)

    # 时长（秒）。
    duration_s = None
    if len(ts_sorted) >= 2:
        duration_s = round(float(ts_sorted[-1] - ts_sorted[0]), 4)

    # 丢帧率：以"时长 × 实际采样率 + 1"推算期望帧数，与实际帧数比得缺失比例。
    # 相比"间隔 > 1.5×中位数"法，此法能正确反映整段缺失。
    frame_loss_ratio = 0.0
    if duration_s and actual_rate:
        expected = max(1, duration_s * actual_rate + 1)
        frame_loss_ratio = round(max(0.0, 1.0 - len(ts) / expected), 4)
        if expected <= len(ts) + 1:
            frame_loss_ratio = 0.0  # 无缺失

    # 实际 vs 标称偏差：标称缺失时该项标记为 skipped（不得静默消失）。
    rate_deviation = None
    nominal_check: dict[str, Any] = {"status": "done"}
    if nominal is None:
        nominal_check = {"status": "skipped", "reason": "标称采样率缺失（未配置 nominal_rate_hz）"}
    elif actual_rate:
        rate_deviation = round(abs(actual_rate - nominal) / nominal, 4)
        nominal_check["rate_deviation"] = rate_deviation
    else:
        nominal_check = {"status": "skipped", "reason": "无法实测采样率"}

    return {
        "n_samples": int(len(ts)),
        "disorder_count": disorder,
        "duplicate_count": duplicates,
        "frame_loss_ratio": frame_loss_ratio,
        "actual_rate_hz": actual_rate,
        "nominal_rate_hz": nominal,
        "rate_deviation": rate_deviation,
        "nominal_check": nominal_check,
        "duration_s": duration_s,
    }


def _align_residuals(base_ts: np.ndarray, other_ts: np.ndarray) -> dict[str, Any]:
    """以 base_ts 为基准，对 other_ts 做最近邻匹配，统计残差分布。

    Args:
        base_ts: 基准流时间戳（排序后）。
        other_ts: 待对齐流时间戳（排序后）。

    Returns:
        dict，含 n_match、residual_mean_ms、residual_max_ms、residual_p95_ms。
    """
    base_sorted = np.sort(base_ts)
    other_sorted = np.sort(other_ts)
    if len(base_sorted) == 0 or len(other_sorted) == 0:
        return {"n_match": 0, "residual_mean_ms": None,
                "residual_max_ms": None, "residual_p95_ms": None}
    # 最近邻：对每个 other 时间戳找 base 中最近点。
    residuals = []
    for t in other_sorted:
        idx = np.searchsorted(base_sorted, t)
        best = np.inf
        for j in (idx - 1, idx, min(idx, len(base_sorted) - 1)):
            if 0 <= j < len(base_sorted):
                best = min(best, abs(base_sorted[j] - t))
        residuals.append(best)
    res = np.asarray(residuals) / 1e6  # 纳秒 → 毫秒
    return {
        "n_match": int(len(res)),
        "residual_mean_ms": round(float(np.mean(res)), 3),
        "residual_max_ms": round(float(np.max(res)), 3),
        "residual_p95_ms": round(float(np.percentile(res, 95)), 3),
    }


def _detect_drift(
    base_ts: np.ndarray,
    other_ts: np.ndarray,
    n_windows: int,
    slope_threshold: float,
) -> dict[str, Any]:
    """按时间窗口检测流间漂移：窗口残差偏移的线性拟合斜率。

    Args:
        base_ts: 基准流时间戳。
        other_ts: 待对齐流时间戳。
        n_windows: 窗口数。
        slope_threshold: 漂移判定斜率阈值（ms/s）。

    Returns:
        dict，含 drift_slope_ms_per_s、drift_detected、per_window（各窗口偏移）。
    """
    if len(base_ts) < 2 or len(other_ts) < 2:
        return {"drift_detected": False, "drift_slope_ms_per_s": None,
                "per_window": []}
    base_sorted = np.sort(base_ts)
    other_sorted = np.sort(other_ts)

    # 对基准流每个时间戳，找待测流最近邻，得带符号偏移（ms）。
    # 最近邻只吸收采样离散误差（≤半采样间隔），只要漂移大于采样间隔，
    # 偏移随绝对时间单调增大，能被线性回归捕捉。
    base_times: list[float] = []
    offsets: list[float] = []
    for b in base_sorted:
        idx = int(np.searchsorted(other_sorted, b))
        best: tuple[float, float] | None = None  # (abs_diff, signed_diff)
        for j in (idx - 1, idx, min(idx, len(other_sorted) - 1)):
            if 0 <= j < len(other_sorted):
                d = other_sorted[j] - b
                if best is None or abs(d) < best[0]:
                    best = (abs(d), d)
        if best is not None:
            base_times.append(float(b) / 1e9)      # 纳秒 → 秒
            offsets.append(best[1] / 1e6)          # 纳秒 → 毫秒

    if len(offsets) < 3:
        return {"drift_detected": False, "drift_slope_ms_per_s": None,
                "offset_range_ms": None}

    # 线性回归：偏移 y(ms) vs 基准绝对时间 x(s)。斜率 = 漂移速率 ms/s。
    x = np.asarray(base_times, dtype=float)
    y = np.asarray(offsets, dtype=float)
    slope, _ = np.polyfit(x, y, 1)

    # 漂移判定：斜率超阈值 且 偏移累计跨度足够大（> 2×基准采样间隔），
    # 避免恒定小偏移/抖动被误判。
    bd = np.diff(base_sorted)
    med_b = float(np.median(bd)) if len(bd) > 0 else 0.0
    base_interval_ms = (med_b / 1e6) if med_b > 0 else 0.0  # 纳秒 → 毫秒
    offset_range = float(np.max(y) - np.min(y))
    drift_detected = bool(
        abs(slope) > slope_threshold
        and offset_range > 2.0 * base_interval_ms
    )

    return {
        "drift_detected": drift_detected,
        "drift_slope_ms_per_s": round(float(slope), 4),
        "offset_range_ms": round(offset_range, 3),
    }


def check_temporal_sync_impl(context: RunContext, settings=None) -> dict[str, Any]:
    """执行时间同步检查（v1，仅时间戳一致性）。

    Args:
        context: 运行时上下文（复用 meta.streams）。
        settings: 应用配置（阈值）；缺省时读取 get_settings()。

    Returns:
        统一质检返回格式：success、verification_level、result（pass/warn/fail）、
        measurements、thresholds、affected_episodes、user_message。

    Raises:
        不直接抛出异常；错误以结构化 dict 返回。
    """
    settings = settings or get_settings()
    streams = context.meta.get("streams", [])
    capabilities = context.meta.get("capabilities", {})

    # 视频 ↔ metainfo 配对映射（type=media_metainfo），用于视频流时间戳级对齐。
    video_metainfo: dict[str, str] = {}
    for pair in context.meta.get("stream_pairs", []):
        if pair.get("type") == "media_metainfo":
            video_metainfo[pair.get("media", "")] = pair.get("metainfo", "")

    # 逐流读取时间戳（表格流读文件；视频流若存在配对 metainfo 表，经该表参与
    # 时间戳级对齐，注明时间戳来自曝光元数据而非容器）。
    per_stream: dict[str, dict[str, Any]] = {}
    streams_status: dict[str, str] = {}
    for s in streams:
        key = s.get("path") or s.get("kind")
        if s.get("kind") == "video":
            meta_path = video_metainfo.get(s.get("path", ""))
            if meta_path:
                # 经配对 metainfo 表读取时间戳（exposure_start_utc_ns / pts_us 等）。
                meta_ts = _read_timestamp_only(meta_path, "csv")
                if meta_ts is not None:
                    arr = pd_to_numeric(meta_ts)  # 已为 ndarray（见 pd_to_numeric）
                    arr = arr[~np.isnan(arr)]
                    # 推断 metainfo 时间戳列的单位并归一化到纳秒；若为帧序号
                    # （如 pts 无配对真实时间戳列）则标 frame_index、不参与对齐残差。
                    col_name = str(meta_ts.name) if meta_ts.name is not None else ""
                    meta_unit = infer_metainfo_unit(arr, col_name)
                    arr_ns, unit_info = _normalize_to_ns(arr, meta_unit)
                    frame_indexed = meta_unit == FRAME_UNIT
                    per_stream[key] = {
                        "kind": "video",
                        "ts": arr_ns if len(arr) > 0 else None,
                        "source": meta_path,
                        "timestamp_origin": "exposure_metadata",
                        "unit_info": unit_info,
                        "timestamp_unit": meta_unit,
                        "frame_indexed": frame_indexed,
                    }
                    if frame_indexed:
                        streams_status[key] = (
                            "仅单流检查：metainfo 时间戳为帧序号（无配对真实时间戳列，"
                            "不参与跨流对齐残差判定）"
                        )
                    else:
                        streams_status[key] = (
                            "参与对齐：经配对 metainfo 表读取曝光时间戳"
                            "（时间戳来自曝光元数据而非容器）"
                        )
                else:
                    per_stream[key] = {"kind": "video", "ts": None,
                                       "source": s.get("path")}
                    streams_status[key] = "未参与：配对的 metainfo 表无法读取时间戳"
            else:
                per_stream[key] = {"kind": "video", "ts": None,
                                   "source": s.get("path")}
                streams_status[key] = "未参与：v1 不做视频帧级对齐（容器时间戳不可靠，内容级对齐属 v2）"
        else:
            ts = _read_stream_timestamps(s)
            unit = s.get("timestamp_unit", "unknown")
            ts_ns, unit_info = (
                _normalize_to_ns(ts, unit) if ts is not None else (None, None)
            )
            # 帧序号类时间戳（如 metainfo 的 pts）无物理时间：不参与跨流对齐残差，
            # 但保留用于单流检查（乱序/重复/丢帧）。
            frame_indexed = unit == FRAME_UNIT
            per_stream[key] = {
                "kind": s.get("kind"),
                "ts": ts_ns,
                "ts_raw": ts,
                "source": s.get("path"),
                "nominal": _nominal_rate(s),
                "unit_info": unit_info,
                "timestamp_unit": unit,
                "frame_indexed": frame_indexed,
            }
            if ts is None:
                streams_status[key] = "未参与：无法读取时间戳列"
            elif frame_indexed:
                streams_status[key] = "仅单流检查：帧序号时间戳（不参与跨流对齐残差判定）"
            else:
                streams_status[key] = "参与对齐"

    # 可对齐流数：统计能实际读到时间戳列、且为真实时间（非帧序号）的流。
    # 帧序号流（frame_indexed）仅做单流检查，不参与跨流对齐残差判定。
    alignable = [
        k for k, p in per_stream.items()
        if p["ts"] is not None and not p.get("frame_indexed", False)
    ]
    n_alignable = len(alignable)
    if n_alignable < 2:
        return {
            "success": False,
            "error": "not_applicable",
            "reason": f"需要至少两路可对齐流（能读到时间戳列的表格流或经 metainfo 配对的视频流），当前可对齐流数 {n_alignable}",
            "user_message": "check_temporal_sync 需要至少两路可对齐的数据流（能实际读取时间戳列的表格流，如 IMU 与力/力矩；或经配对 metainfo 表参与对齐的视频流）。"
            f"当前可对齐流数 {n_alignable}（无配对 metainfo 的视频流不计入，v1 不做视频帧级对齐），不适用。",
            "streams_status": streams_status,
        }

    # 单流检查。
    stream_checks: dict[str, Any] = {}
    valid_ts = []
    for key, p in per_stream.items():
        if p["ts"] is None:
            # 无法参与对齐的流：标注未参与原因，不进入判定依据。
            stream_checks[key] = {"present": False,
                                  "status": "skipped",
                                  "reason": streams_status.get(key, "无法读取时间戳")}
            continue
        # 单流检查用秒；p["ts"] 已统一为纳秒（normalized）或原值（unknown/frame）。
        ts_sec = _to_seconds(p["ts"], p.get("unit_info"))
        checks = _single_stream_checks(ts_sec, p.get("nominal"))
        checks["present"] = True
        # 透出该流时间戳的原始单位与换算说明（供核对归一化是否正确）。
        if p.get("unit_info"):
            checks["timestamp_unit"] = p.get("timestamp_unit")
            checks["timestamp_unit_basis"] = p["unit_info"].get("basis")
        stream_checks[key] = checks
        # 仅真实时间流进入跨流对齐残差 / 漂移判定（帧序号流已在上方排除）。
        if not p.get("frame_indexed", False):
            valid_ts.append((key, p["ts"]))

    # episode 口径：无 episode 划分时整段视为一个 episode。
    has_episodes = bool(capabilities.get("has_episodes")) or bool(
        context.meta.get("episode_ids")
    )
    episode_note = "未检测到 episode 划分，将整个录制视为单个 episode。"

    # 流间对齐残差 + 漂移：以帧率最低（采样间隔最大）的流为基准。
    align: dict[str, Any] = {"baseline": None, "residuals": {}}
    drift: dict[str, Any] = {"detected": False, "detail": {}}
    max_interval_hz = 0.0
    baseline_key = None
    for key, ts in valid_ts:
        if len(ts) > 1:
            med = float(np.median(np.diff(np.sort(ts))))
            if med > 0 and 1e9 / med > max_interval_hz:
                max_interval_hz = 1e9 / med  # 纳秒间隔 → Hz

    # 基准 = 帧率最低的流（采样间隔最大）。
    if valid_ts:
        min_rate = np.inf
        for key, ts in valid_ts:
            if len(ts) > 1:
                med = float(np.median(np.diff(np.sort(ts))))
                rate = 1e9 / med if med > 0 else np.inf  # 纳秒间隔 → Hz
                if rate < min_rate:
                    min_rate = rate
                    baseline_key = key

    if baseline_key is not None:
        base_ts = per_stream[baseline_key]["ts"]
        align["baseline"] = baseline_key
        for key, ts in valid_ts:
            if key == baseline_key:
                align["residuals"][key] = {"n_match": len(ts),
                                           "is_baseline": True}
            else:
                res = _align_residuals(base_ts, ts)
                align["residuals"][key] = {**res, "is_baseline": False}
                d = _detect_drift(
                    base_ts, ts, settings.sync_drift_windows,
                    settings.sync_drift_slope_ms_per_s,
                )
                drift["detail"][key] = d
                if d.get("drift_detected"):
                    drift["detected"] = True

    # 三档判定。
    frame_loss_bad = any(
        c.get("frame_loss_ratio", 0) > settings.sync_frame_loss_ratio
        for c in stream_checks.values() if c.get("present")
    )
    drift_flag = drift["detected"]
    residual_threshold_ms = (1000.0 / max_interval_hz * settings.sync_residual_ratio
                             if max_interval_hz > 0 else settings.sync_max_skew_ms)
    residual_near = any(
        r.get("residual_max_ms", 0) and r["residual_max_ms"] > residual_threshold_ms * 0.75
        for r in align["residuals"].values() if not r.get("is_baseline")
    )
    disorder_dup = any(
        c.get("disorder_count", 0) > 0 or c.get("duplicate_count", 0) > 0
        for c in stream_checks.values() if c.get("present")
    )

    if frame_loss_bad or drift_flag:
        result = "fail"
    elif residual_near or disorder_dup:
        result = "warn"
    else:
        result = "pass"

    affected = []  # 受影响 episode（无划分时为整个录制）
    if result != "pass" and not has_episodes:
        affected = ["whole_recording"]

    # 跳过的检查项：独立列出（skipped 不得计入 pass 依据，也不得静默消失）。
    skipped_checks = {
        k: {"status": "skipped", "reason": v.get("reason")}
        for k, v in stream_checks.items() if not v.get("present")
    }

    # 漂移相对性说明（检出漂移时固定附上）。
    drift_note = None
    if drift_flag:
        drift_note = (
            "漂移是相对量——基于时间戳只能测出流间偏移趋势，无法判定哪条流是漂移源头，"
            "需物理实测（v2）才能定位。"
        )

    user_message = (
        f"时间同步检查判定：{result}。verification_level=timestamp_consistency（仅基于"
        "时间戳一致性推断，物理级对齐需互相关实测，属未来 v2）。"
        + (f" {episode_note}" if not has_episodes else "")
    )
    if drift_flag:
        user_message += " 检出漂移，该数据集可能各设备独立打钟，建议物理对齐实测。"
    if drift_note:
        user_message += f" {drift_note}"

    # 质检结果写回 meta["qc"]，供 compute_stats / generate_report 读取质检明细。
    qc = context.meta.setdefault("qc", {})
    qc["check_temporal_sync"] = {
        "result": result,
        "verification_level": "timestamp_consistency",
        "drift_detected": drift_flag,
        "dataset": context.dataset_id,
        "detail": {
            "stream_checks": {
                k: {
                    "n_samples": v.get("n_samples"),
                    "disorder_count": v.get("disorder_count"),
                    "duplicate_count": v.get("duplicate_count"),
                    "frame_loss_ratio": v.get("frame_loss_ratio"),
                    "actual_rate_hz": v.get("actual_rate_hz"),
                } if v.get("present") else {"status": "skipped", "reason": v.get("reason")}
                for k, v in stream_checks.items()
            },
            "residuals": {
                k: {"residual_max_ms": v.get("residual_max_ms"),
                    "residual_mean_ms": v.get("residual_mean_ms")}
                for k, v in align["residuals"].items() if not v.get("is_baseline")
            },
            "drift": {
                k: {"drift_slope_ms_per_s": v.get("drift_slope_ms_per_s"),
                    "drift_detected": v.get("drift_detected")}
                for k, v in drift["detail"].items()
            },
            "thresholds": {
                "frame_loss_ratio": settings.sync_frame_loss_ratio,
                "residual_threshold_ms": round(residual_threshold_ms, 3),
                "drift_slope_ms_per_s": settings.sync_drift_slope_ms_per_s,
            },
        },
    }

    return {
        "success": True,
        "verification_level": "timestamp_consistency",
        "verification_note": "仅基于时间戳一致性推断；物理级对齐需互相关实测（未来 v2）。",
        "result": result,
        "episode_note": episode_note if not has_episodes else "存在 episode 划分。",
        "baseline_stream": align["baseline"],
        "streams_status": streams_status,
        "skipped_checks": skipped_checks,
        "measurements": {
            "stream_checks": stream_checks,
            "residuals": align["residuals"],
            "drift": drift["detail"],
        },
        "thresholds": {
            "frame_loss_ratio": settings.sync_frame_loss_ratio,
            "residual_threshold_ms": round(residual_threshold_ms, 3),
            "drift_slope_ms_per_s": settings.sync_drift_slope_ms_per_s,
            "n_drift_windows": settings.sync_drift_windows,
        },
        "affected_episodes": affected,
        "note": drift_note,  # 仅检出漂移时为非 None
        "user_message": user_message,
    }


@tool
def check_temporal_sync(
    wrapper: RunContextWrapper[RunContext],
) -> dict:
    """检查各流之间的时间同步与漂移（v1，仅时间戳一致性）。

    基于流登记表逐流读取时间戳，检查单调性、重复、丢帧率、采样率、时长，并做
    流间最近邻对齐残差与窗口漂移检测。verification_level=timestamp_consistency，
    物理级对齐需互相关实测（未来 v2）。

    Args:
        无（基于 RunContext.meta）。

    Returns:
        统一质检返回格式：result（pass/warn/fail）、measurements、thresholds、
        affected_episodes、user_message；流数不足时返回 not_applicable。
    """
    return check_temporal_sync_impl(wrapper.context)
