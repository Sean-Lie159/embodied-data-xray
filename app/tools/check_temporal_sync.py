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
from app.tools._readers import split_path_spec
from app.tools.inspect_streams import _read_timestamp_only
from app.tools.timestamp_units import (
    FRAME_UNIT,
    TIME_UNITS,
    infer_unit,
    self_correct_unit,
    to_ns,
    unit_to_ns_factor,
)


def _read_stream_timestamps(
    stream: dict[str, Any], column_hint: str | None = None
) -> tuple[np.ndarray | None, str]:
    """按需读取单条流的原始时间戳序列（文件顺序，不排序）。

    Args:
        stream: 流登记项（含 path / format / kind）。
        column_hint: 可选，指定时间戳列（列名或子串，如 "log_time"）。

    Returns:
        (原始时间戳 numpy 数组, 时间戳列名)。数组为数值化、去 NaN 结果；
        读取失败或非表格流返回 (None, "")。列名用于单位交叉校验——流登记表的
        timestamp_unit 可能判为 unknown，而实际列名（如 mcap_log_time_ns）能
        重新推断出单位。
    """
    if stream.get("kind") == "video":
        return None, ""  # 视频流时间戳由 ffprobe 提供，见 _video_ideal_ts
    path = stream.get("path", "")
    fmt = stream.get("format", "")
    file_part, sub = split_path_spec(path)
    if not path or not Path(file_part).exists():
        return None, ""
    # 容器子流（h5 节点 / mcap topic）：时间戳列经统一读取注册表取
    # （`"<file>::<node|topic>"` 的解析与分派收敛在 _readers，各工具不再特判）。
    if sub:
        from app.tools._readers import ReadRequest, read_stream

        result = read_stream(ReadRequest(
            path_spec=path, want="timestamp", fmt=fmt, column=column_hint))
        if not result.ok or result.timestamp is None:
            return None, ""
        arr = np.asarray(result.timestamp, dtype=float)
        arr = arr[~np.isnan(arr)]
        return (arr, result.timestamp_column or "") if len(arr) > 0 else (None, "")
    # 嵌套时间路径（含 "."，如 data.header.timestamp_us）：经点分路径逐行
    # 提取（仅 jsonl/json）。读取失败返回 None，由调用方注明"指定时间列不存在"。
    if column_hint and "." in column_hint:
        from app.tools._data_access import read_nested_time_column

        ts = read_nested_time_column(path, fmt, column_hint)
        if ts is None:
            return None, ""
        arr = np.asarray(pd_to_numeric(ts), dtype=float)
        arr = arr[~np.isnan(arr)]
        return (arr, column_hint) if len(arr) > 0 else (None, "")
    try:
        ts = _read_timestamp_only(path, fmt, column_hint)
        if ts is None:
            return None, ""
        col_name = str(ts.name) if ts.name is not None else ""
        arr = np.asarray(pd_to_numeric(ts), dtype=float)
        arr = arr[~np.isnan(arr)]
        return (arr, col_name) if len(arr) > 0 else (None, "")
    except Exception:  # noqa: BLE001
        return None, ""


def pd_to_numeric(series) -> np.ndarray:
    """将 Series 安全转数值数组。"""
    import pandas as pd

    return pd.to_numeric(series, errors="coerce").to_numpy()


def is_unit_known(unit_info: dict[str, Any] | None) -> bool:
    """判断时间戳是否已归一化到纳秒基准（单位已知且换算成功）。

    单位未知 / 帧序号的流未归一化，其时长、间隔、采样率等绝对量不可计算——
    此前这类流被"按秒兜底"换算后，产出的 duration_s=5.28e10（实为纳秒原值）
    等伪值会污染判定，故此处一律不参与跨流对齐与绝对量计算。

    Args:
        unit_info: _normalize_to_ns 返回的单位说明。

    Returns:
        True 表示数组已是纳秒基准，绝对量可信。
    """
    return bool(unit_info and unit_info.get("normalized"))


def infer_metainfo_unit(arr: np.ndarray, col_name: str) -> str:
    """推断 metainfo 时间戳列的单位。

    规则：列名含真实时钟标记（utc / epoch / real / host）→ 用 infer_unit 按量级推断
    （如 exposure_start_utc_ns → ns）；列名以时间单位后缀（_ns/_us/_ms/_s）结尾 →
    物理时间（用 infer_unit 量级确认，如 frame_timestamps_ns → ns）；列名是**裸**帧
    序号词（pts / frame_index / frame_id / packet_index 等，无时间单位后缀）→
    frame_index（无物理时间，不参与跨流对齐）；否则退回 infer_unit。

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
    # 时间单位后缀优先：frame_timestamps_ns 是物理时间，不是帧序号。
    if lower.endswith(("_ns", "_us", "_ms", "_s")):
        return infer_unit(arr)["unit"]
    # 裸帧序号词（无时间单位后缀）→ frame_index。
    if any(k in lower for k in ("pts", "packet", "frame_index", "frame_id", "_idx")):
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
    # 单位未知：不硬猜、不兜底换算，保持原值并标 normalized=False。
    # 说明（2026-09-04 变更）：此前按秒换算到纳秒，导致纳秒列被放大 1e9 倍后
    # 又被下游当秒除回，产出 duration_s=5.28e10、残差 2.43e11 等伪值。现改为
    # 显式不可用——该流不参与跨流对齐，其时长/间隔/采样率一律置 None。
    return np.asarray(ts, dtype=float), {
        "original_unit": unit or "unknown",
        "normalized": False,
        "basis": "单位未知，未归一化；该流不参与跨流对齐，时长/间隔/采样率不可计算",
    }


def _nominal_rate(stream: dict[str, Any]) -> float | None:
    """从流登记表/meta 读取标称采样率；缺省返回 None。"""
    rate = stream.get("nominal_rate_hz")
    if rate is None:
        rate = stream.get("measured_rate", {}).get("sample_rate_hz")
    return float(rate) if isinstance(rate, (int, float)) and rate > 0 else None


# 基线排除的文件名关键词：静态变换广播 / 标定 / 元数据（大小写不敏感，子串匹配）。
_BASELINE_NAME_EXCLUDE = (
    "tf_static", "static_tf", "_static", "static_",
    "calib", "metadata", "metainfo",
)


def _stream_baseline_exclude_reason(
    path: str,
    stream: dict[str, Any],
    n_samples: int,
    min_samples: int,
) -> str | None:
    """判断一条流是否不得作为对齐基线，返回排除原因（不排除返回 None）。

    判据（按语义角色优先原则，参照 detect_episode_mirrors / is_calibration_file 的
    既有排除先例）：静态变换广播（tf_static 等只发布一次、不随时间变化）、标定
    文件、元数据表、样本数过少的稀疏流。这些流参与"帧率最低"竞选时会胜出并污染
    残差（真实事故：以 105 点的 tf_static.jsonl 为基线）。

    Args:
        path: 流路径（用于文件名关键词判定）。
        stream: 流登记项（kind / semantic_label）。
        n_samples: 该流时间戳样本数。
        min_samples: 基线候选的最小样本数（config.sync_static_min_samples）。

    Returns:
        排除原因（中文，可直接转述）；不排除返回 None。
    """
    name = Path(path).name.lower()
    if n_samples < min_samples:
        return f"样本数 {n_samples} < {min_samples}，疑似静态/稀疏流"
    for kw in _BASELINE_NAME_EXCLUDE:
        if kw in name:
            return f"文件名含 {kw!r}，疑似静态/标定/元数据流（不随时间变化，不适合做对齐基准）"
    if str(stream.get("kind") or "") == "calibration":
        return "标定文件（不随时间变化）"
    if "标定" in str(stream.get("semantic_label") or ""):
        return "语义标签为标定类（不随时间变化）"
    return None


def _recommend_baseline(
    valid_ts: list[tuple[str, np.ndarray]],
    per_stream: dict[str, dict[str, Any]],
    streams: list[dict[str, Any]],
    settings: Any,
) -> tuple[str | None, dict[str, Any]]:
    """推荐对齐基线：先排除静态/标定/元数据流，再按覆盖×稳定×规模打分。

    打分公式：score = coverage_ratio × stability × log10(n_samples)
      coverage_ratio = 该流时间跨度 / 全部候选流的最大跨度（覆盖完整性）
      stability      = 1 / (1 + 差分变异系数)（间隔稳定性，抖动越小越高）
    即"覆盖整段录制、间隔稳定、样本充足"的周期型流优先。

    Args:
        valid_ts: 可对齐流 [(key, 纳秒时间戳数组)]（单位未知/帧序号流已被排除）。
        per_stream: 逐流检查数据（取样本数）。
        streams: 流登记表（取 kind / semantic_label）。
        settings: 应用配置（sync_static_min_samples / sync_baseline_min_coverage）。

    Returns:
        (baseline_key, recommendation)。recommendation 含 stream、reason（可直接
        转述的推荐理由）、score、excluded（被排除的流及原因）；无候选时
        baseline_key 为 None 并在 note 说明。
    """
    meta_by_path = {s.get("path") or "": s for s in streams}
    spans: dict[str, float] = {}
    cvs: dict[str, float] = {}
    for key, ts in valid_ts:
        if len(ts) < 2:
            continue
        t_sorted = np.sort(ts)
        d = np.diff(t_sorted)
        d_pos = d[d > 0]
        spans[key] = float(t_sorted[-1] - t_sorted[0])
        if len(d_pos) > 1 and float(d_pos.mean()) > 0:
            cvs[key] = float(d_pos.std() / d_pos.mean())
        else:
            cvs[key] = 0.0

    if not spans:
        return None, {
            "stream": None, "reason": "无可打分的候选流", "score": None,
            "excluded": [],
        }

    max_span = max(spans.values())
    excluded: list[dict[str, str]] = []
    candidates: list[tuple[str, float, float, float, int]] = []
    for key in spans:
        cov = spans[key] / max_span if max_span > 0 else 0.0
        reason = _stream_baseline_exclude_reason(
            key, meta_by_path.get(key, {}), len(per_stream[key]["ts"]),
            settings.sync_static_min_samples,
        )
        if reason:
            excluded.append({"stream": Path(key).name, "reason": reason})
            continue
        if cov < settings.sync_baseline_min_coverage:
            excluded.append({
                "stream": Path(key).name,
                "reason": f"时间覆盖率 {cov:.0%} 低于阈值 "
                          f"{settings.sync_baseline_min_coverage:.0%}（未覆盖整段录制）",
            })
            continue
        n = int(len(per_stream[key]["ts"]))
        score = cov * (1.0 / (1.0 + cvs[key])) * float(np.log10(max(n, 10)))
        candidates.append((key, score, cov, cvs[key], n))

    if not candidates:
        return None, {
            "stream": None,
            "reason": "所有候选流均被排除（静态/标定/元数据/覆盖率不足），"
                      "无法推荐基线，跳过流间残差与漂移检测",
            "score": None,
            "excluded": excluded,
        }

    # 确定性 tie-break：分数并列时按路径字母序取最小（不同会话/运行得到
    # 同一基线；真实案例：两轮对话分别推荐了不同的基线流）。
    candidates.sort(key=lambda x: (-x[1], x[0]))
    best_key, best_score, cov, cv, n = candidates[0]
    recommendation: dict[str, Any] = {
        "stream": best_key,
        "reason": (
            f"覆盖整段录制（{cov:.0%}）、间隔稳定（变异系数 {cv:.2f}）、"
            f"样本 {n}，为覆盖完整且间隔稳定的周期型流"
        ),
        "score": round(best_score, 4),
        # 前 3 名候选（含分数）：并列或接近时供用户判断/改用 baseline_stream 指定。
        "top_candidates": [
            {"stream": Path(k).name, "score": round(s, 4)}
            for k, s, *_ in candidates[:3]
        ],
        "excluded": excluded,
    }
    return best_key, recommendation


def _single_stream_checks(
    ts: np.ndarray,
    nominal: float | None,
    unit_known: bool = True,
    settings: Any = None,
) -> dict[str, Any]:
    """单流时间戳检查：单调性、重复、丢帧率、实际采样率、时长、形态分类。

    Args:
        ts: 原始顺序时间戳数组（**纳秒**基准；单位未知时为原值）。
        nominal: 标称采样率（Hz），None 时跳过实际 vs 标称对比。
        unit_known: 时间戳是否已归一化到纳秒基准。False 时绝对量（时长 / 中位
            间隔 / 采样率）一律置 None 并在 absolute_note 注明不可用，仅保留
            单位无关的检查项（乱序 / 重复 / 丢帧率——丢帧率是比值，与单位无关）。
        settings: 应用配置（形态分类阈值）；缺省时读取 get_settings()。

    Returns:
        dict，含各项测量值与判定标记（含 stream_shape；burst 流的
        frame_loss_ratio 为 None 并附 frame_loss_status="not_applicable"）。
    """
    ts_sorted = np.sort(ts)
    diffs = np.diff(ts_sorted)

    # 单调性（乱序计数）：原始顺序中后项小于前项的次数。
    disorder = int(np.sum(np.diff(ts) < 0)) if len(ts) > 1 else 0

    # 重复时间戳计数。
    duplicates = int(np.sum(np.diff(ts_sorted) == 0)) if len(ts) > 1 else 0

    # 实际采样率（**平均**间隔倒数）。守恒口径：平均采样率 × 时长 = 样本数，
    # 这是"期望帧数"计算唯一自洽的口径；中位差分代表"典型间隔"（受抖动影响的
    # 分布众数附近值），用它乘时长会在抖动存在时系统性虚报丢帧（真实案例：
    # 抖动 6.6ms 使中位 38.6ms < 平均 39.5ms，虚报 2.28% 丢帧并误判 FAIL）。
    med = float(np.median(diffs)) if len(diffs) > 0 else 0.0
    mean_d = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    # 绝对量（时长 / 中位间隔 / 采样率）：**仅单位已知时计算**。
    # 单位未知的流此前被"按秒兜底"换算，产出 duration_s=5.28e10（实为纳秒原值）
    # 等伪值并污染判定；现改为显式置 None 并注明原因。
    actual_rate = None
    median_interval_ns = None
    duration_ns = None
    absolute_note = None
    if not unit_known:
        absolute_note = "不可用：时间戳单位未知，未归一化（该流不参与跨流对齐）"
    else:
        if len(diffs) > 0:
            median_interval_ns = round(med, 3) if med > 0 else None
            if mean_d > 0:
                # 输入为纳秒基准：采样率 = 1e9 / 平均间隔(ns)。
                actual_rate = round(1e9 / mean_d, 3)
        if len(ts_sorted) >= 2:
            duration_ns = round(float(ts_sorted[-1] - ts_sorted[0]), 3)

    # 丢帧率：**异常间隔累加法**（对抖动鲁棒、对真实缺口敏感）。
    # 旧公式 expected = duration × (中位采样率) + 1 在有抖动时系统性虚报
    # （中位差分 < 平均差分 → expected 虚高 → 把抖动当丢帧）。
    # 新口径：间隔 > 5×中位差分 视为缺口，缺失时长 = Σ(缺口间隔 - 中位差分)，
    # 丢帧率 = 缺失时长 / 总时长。抖动（< 5×中位）不计入；整段缺失被正确累加。
    # 丢帧率是比值，与单位无关，故单位未知时仍可计算（用原值跨度作分母）。
    frame_loss_ratio = 0.0
    gap_count = 0
    span = float(ts_sorted[-1] - ts_sorted[0]) if len(ts_sorted) >= 2 else None
    if span and span > 0 and len(diffs) > 0 and med > 0:
        K = 5.0
        gaps = diffs[diffs > K * med]
        gap_count = int(len(gaps))
        missing_duration = float(np.sum(gaps - med)) if len(gaps) else 0.0
        frame_loss_ratio = round(min(1.0, missing_duration / span), 4)

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

    # 流形态分类与突发型流指标改报（2026-09-04 Commit D）。
    # 突发型流（MCAP 录制的 tf / IMU：突发内 1–2µs、突发间毫秒级静默）的
    # "丢帧率"没有物理意义——突发间静默全部超过 5×中位间隔，被系统性误报为
    # 丢帧（真实案例 0.9986）。对 burst 流改报有效速率与突发统计。
    shape = _classify_stream_shape(ts, settings or get_settings())
    result: dict[str, Any] = {
        "n_samples": int(len(ts)),
        "disorder_count": disorder,
        "duplicate_count": duplicates,
        "frame_loss_ratio": frame_loss_ratio,
        "gap_count": gap_count,
        "actual_rate_hz": actual_rate,
        # 字段名带单位后缀，杜绝被当作秒解读（duration_s=5.28e10 事故）。
        "median_interval_ns": median_interval_ns,
        "duration_ns": duration_ns,
        "absolute_note": absolute_note,  # 仅单位未知时非 None
        "stream_shape": shape,
        "nominal_rate_hz": nominal,
        "rate_deviation": rate_deviation,
        "nominal_check": nominal_check,
    }
    if shape == "burst":
        result["frame_loss_ratio"] = None
        result["frame_loss_status"] = "not_applicable"
        result["burst_note"] = (
            "突发型流：丢帧率与常规采样率口径不适用（突发间静默会被误报为丢帧），"
            "改用有效速率与突发次数刻画"
        )
        # 突发统计：大间隔（> 2×中位）的数量即突发段切换数，段数 = 大间隔数 + 1。
        big_gaps = int(np.sum(diffs > 2.0 * med)) if len(diffs) else 0
        result["n_bursts"] = big_gaps + 1 if len(diffs) else 0
        if unit_known:
            result["intra_burst_median_interval_ns"] = median_interval_ns
            if span and span > 0:
                # 有效速率 = 样本数 / 时长（对突发流唯一有意义的速率口径）。
                result["effective_rate_hz"] = round(len(ts) / (span / 1e9), 3)
    return result


def _classify_stream_shape(
    ts: np.ndarray, settings: Any
) -> str:
    """按差分分布对流做时间形态分类：periodic / burst / static。

    判据（确定性，见 docs/时间对齐能力改造设计.md 4.5；实现时修正）：
    - static：样本数 < sync_static_min_samples，或中位间隔 > 流自身跨度/10
      （点太稀，"平均多久一条"没有意义）；
    - burst：平均间隔 ≥ 中位间隔 × sync_burst_interval_ratio——突发型流
      （如 MCAP 录制的 tf / IMU）在突发内间隔极小、突发间静默，均值被静默
      段拉高而中位数不变（真实数据：IMU 均值 1.26ms / 中位 1.79µs ≈ 700）；
    - 其余为 periodic（周期型）。

    Args:
        ts: 纳秒基准时间戳数组。
        settings: 应用配置（sync_static_min_samples / sync_burst_interval_ratio）。

    Returns:
        形态名："periodic" / "burst" / "static"。
    """
    ts_sorted = np.sort(np.asarray(ts, dtype=float))
    if len(ts_sorted) < 2:
        return "static"
    diffs = np.diff(ts_sorted)
    med = float(np.median(diffs)) if len(diffs) else 0.0
    span = float(ts_sorted[-1] - ts_sorted[0])
    if len(ts_sorted) < settings.sync_static_min_samples:
        return "static"
    if med > 0 and span > 0 and med > span / 10:
        return "static"
    if med > 0 and float(diffs.mean()) >= settings.sync_burst_interval_ratio * med:
        return "burst"
    return "periodic"


def _locate_gaps(ts: np.ndarray, max_report: int) -> dict[str, Any]:
    """定位时间戳序列中的数据缺口（与丢帧判定同口径：间隔 > 5×中位差分）。

    Args:
        ts: 纳秒基准时间戳数组（单位未知时为原值——缺口位置是相对量，
            仍可用，但注明时间口径未知）。
        max_report: 最多返回的缺口条数（超出时返回前 N 条并注明总数）。

    Returns:
        dict，含 gaps（[{start_ns, end_ns, duration_ns, missing_frames_est}]）、
        total（缺口总数）、truncated（是否因上限截断）。
    """
    ts_sorted = np.sort(np.asarray(ts, dtype=float))
    if len(ts_sorted) < 2:
        return {"gaps": [], "total": 0, "truncated": False}
    diffs = np.diff(ts_sorted)
    med = float(np.median(diffs)) if len(diffs) else 0.0
    if med <= 0:
        return {"gaps": [], "total": 0, "truncated": False}
    K = 5.0
    gap_idx = np.where(diffs > K * med)[0]
    total = int(len(gap_idx))
    gaps: list[dict[str, Any]] = []
    for i in gap_idx[:max_report]:
        start = float(ts_sorted[i])
        end = float(ts_sorted[i + 1])
        duration = end - start
        gaps.append({
            "start_ns": start,
            "end_ns": end,
            "duration_ns": round(duration, 3),
            # 估算缺失帧数：缺口时长 / 中位间隔，向下取整。
            "missing_frames_est": int(duration / med) - 1 if med > 0 else None,
        })
    return {
        "gaps": gaps,
        "total": total,
        "truncated": total > len(gaps),
    }


def _align_residuals(base_ts: np.ndarray, other_ts: np.ndarray) -> dict[str, Any]:
    """以 base_ts 为基准，对 other_ts 做最近邻匹配，统计残差分布（向量化）。

    对 other 的每个时间戳，在 base 中找最近点（searchsorted 定位后取左右候选），
    残差 = 最近 base 时间戳 − 该 other 时间戳（带符号）。除绝对值统计外，输出
    带符号中位数与分位数——带符号中位数即"目标流相对基线的固定时延"估计量
    （正值 = 目标流比基线晚），此前只有绝对值口径，回答不了"A 比 B 晚多少"。

    Args:
        base_ts: 基准流时间戳（任意顺序，内部排序）。
        other_ts: 待对齐流时间戳（任意顺序，内部排序）。

    Returns:
        dict，含 n_match、residual_mean_ms、residual_max_ms、residual_p95_ms
        （绝对值口径）与 residual_median_signed_ms / residual_p05_ms /
        residual_p95_signed_ms（带符号口径）。
    """
    base_sorted = np.sort(base_ts)
    other_sorted = np.sort(other_ts)
    if len(base_sorted) == 0 or len(other_sorted) == 0:
        return {
            "n_match": 0, "residual_mean_ms": None, "residual_max_ms": None,
            "residual_p95_ms": None, "residual_median_signed_ms": None,
            "residual_p05_ms": None, "residual_p95_signed_ms": None,
        }
    # 向量化最近邻：searchsorted 一次定位全部 other 时间戳，取左右两个候选，
    # 保留绝对差更小的（等价于原逐点三候选逻辑，且修掉了 idx=0 时负索引的歧义）。
    # 带符号残差约定：other − nearest_base，正值 = 目标流比基线晚。
    n_base = len(base_sorted)
    idx = np.searchsorted(base_sorted, other_sorted)
    idx_r = np.clip(idx, 0, n_base - 1)
    idx_l = np.clip(idx - 1, 0, n_base - 1)
    d_r = other_sorted - base_sorted[idx_r]
    d_l = other_sorted - base_sorted[idx_l]
    signed = np.where(np.abs(d_r) <= np.abs(d_l), d_r, d_l)  # 纳秒，带符号
    res = signed / 1e6  # 纳秒 → 毫秒
    abs_res = np.abs(res)
    return {
        "n_match": int(len(res)),
        "residual_mean_ms": round(float(abs_res.mean()), 3),
        "residual_max_ms": round(float(abs_res.max()), 3),
        "residual_p95_ms": round(float(np.percentile(abs_res, 95)), 3),
        # 带符号口径：中位数 = 固定时延估计；p05/p95 给出方向与离散度。
        "residual_median_signed_ms": round(float(np.percentile(res, 50)), 3),
        "residual_p05_ms": round(float(np.percentile(res, 5)), 3),
        "residual_p95_signed_ms": round(float(np.percentile(res, 95)), 3),
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

    # 对基准流每个时间戳，找待测流最近邻，得带符号偏移（ms）。向量化：
    # searchsorted 一次定位全部基准时间戳，取左右候选中绝对差更小者。
    # 最近邻只吸收采样离散误差（≤半采样间隔），只要漂移大于采样间隔，
    # 偏移随绝对时间单调增大，能被线性回归捕捉。
    n_other = len(other_sorted)
    idx = np.searchsorted(other_sorted, base_sorted)
    idx_r = np.clip(idx, 0, n_other - 1)
    idx_l = np.clip(idx - 1, 0, n_other - 1)
    d_r = other_sorted[idx_r] - base_sorted
    d_l = other_sorted[idx_l] - base_sorted
    signed_ns = np.where(np.abs(d_r) <= np.abs(d_l), d_r, d_l)

    base_times = base_sorted / 1e9          # 纳秒 → 秒
    offsets = signed_ns / 1e6               # 纳秒 → 毫秒

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


def check_temporal_sync_impl(
    context: RunContext,
    settings=None,
    baseline_stream: str | None = None,
    streams: list[str] | None = None,
    time_column: str | None = None,
    locate_gaps: bool = False,
) -> dict[str, Any]:
    """执行时间同步检查（v1，仅时间戳一致性）。

    Args:
        context: 运行时上下文（复用 meta.streams）。
        settings: 应用配置（阈值）；缺省时读取 get_settings()。
        baseline_stream: 可选，指定对齐基线流（路径/文件名子串）。省略时自动
            推荐（排除静态/标定/元数据流后按覆盖×稳定×规模打分）。指定且无法
            匹配任何可对齐流时返回结构化错误并列出候选，不静默回退。
        streams: 可选，只检查这些流（文件名子串列表）。省略时检查全部流。
            指定且无任何流匹配时返回结构化错误并列出可用流名。
        time_column: 可选，指定时间戳列（列名或子串，如 "log_time"）。省略时
            自动识别（词表 + 指纹回退；多时间列时物理时间优先）。
        locate_gaps: 默认 False；置 True 时逐流定位数据缺口的起止时刻、持续
            时长与估算缺失帧数（条数受 sync_gap_report_limit 限制）。

    Returns:
        统一质检返回格式：success、verification_level、result（pass/warn/fail）、
        measurements、thresholds、affected_episodes、user_message、
        baseline_recommendation、unit_warnings。

    Raises:
        不直接抛出异常；错误以结构化 dict 返回。
    """
    settings = settings or get_settings()
    all_streams = context.meta.get("streams", [])
    capabilities = context.meta.get("capabilities", {})

    # 流子集过滤：按文件名子串匹配（大小写不敏感）。指定且无匹配时返回结构化
    # 错误并列出可用流名，不静默检查全部——静默会令模型误以为子集过滤生效。
    stream_filter = streams
    streams = all_streams
    if stream_filter:
        available_names = [Path(s.get("path") or "").name or str(s.get("kind") or "")
                           for s in all_streams]
        matched = [
            s for s in all_streams
            if any(p.lower() in Path(s.get("path") or "").name.lower()
                   for p in stream_filter)
        ]
        if not matched:
            return {
                "success": False,
                "error": "streams_no_match",
                "reason": f"streams 过滤条件 {stream_filter} 未匹配到任何流",
                "user_message": (
                    f"streams 过滤条件 {stream_filter} 未匹配到任何流。"
                    f"当前数据集共 {len(available_names)} 条流，可用流名示例："
                    f"{available_names[:10]}{' …' if len(available_names) > 10 else ''}。"
                    "请用文件名的子串（如 'left_glove'、'imu'）重新指定。"
                ),
            }
        streams = matched

    # 视频 ↔ metainfo 配对映射（type=media_metainfo），用于视频流时间戳级对齐。
    # 值为 (metainfo 路径, 格式)，格式来自配对登记（不再硬编码 csv）。
    video_metainfo: dict[str, tuple[str, str]] = {}
    for pair in context.meta.get("stream_pairs", []):
        if pair.get("type") == "media_metainfo":
            video_metainfo[pair.get("media", "")] = (
                pair.get("metainfo", ""),
                pair.get("metainfo_format", ""),
            )
    # 版本组去重：变体视频（variant_of 指向主版本）不参与跨流对齐，避免重复计入
    # 可对齐流数。仅主版本（无分辨率/_pre 后缀）参与对齐。
    video_variant_of: set[str] = set()
    for pair in context.meta.get("stream_pairs", []):
        if pair.get("type") != "video_version_group":
            continue
        for v in pair.get("variants", []):
            if v.get("path"):
                video_variant_of.add(v.get("path"))

    # 逐流读取时间戳（表格流读文件；视频流若存在配对 metainfo 表，经该表参与
    # 时间戳级对齐，注明时间戳来自曝光元数据而非容器）。
    per_stream: dict[str, dict[str, Any]] = {}
    streams_status: dict[str, str] = {}
    for s in streams:
        key = s.get("path") or s.get("kind")
        if s.get("kind") == "video":
            # 版本组去重：变体视频不参与对齐（仅主版本参与跨流对齐）。
            if s.get("path") in video_variant_of:
                per_stream[key] = {"kind": "video", "ts": None,
                                   "source": s.get("path")}
                streams_status[key] = (
                    "未参与：视频版本组的变体（非主版本），不重复计入可对齐流数"
                )
                continue
            metainfo = video_metainfo.get(s.get("path", ""))
            if metainfo:
                meta_path, meta_fmt = metainfo
                # 经配对 metainfo 表读取时间戳（csv/parquet/json，格式来自配对登记）。
                meta_ts = _read_timestamp_only(meta_path, meta_fmt or "csv")
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
                        "unit_known": is_unit_known(unit_info),
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
            # 每流 hint 优先级：用户确认的 time_column（登记表，可嵌套）>
            # 全局 time_column 参数 > None（词表自动）。确认列来自
            # propose_stream_semantics 落盘（真实案例：IMU 传感器时间
            # data.header.timestamp_us 优先于容器批量写入时间）。
            per_stream_hint = s.get("time_column") or time_column
            ts, col_name = _read_stream_timestamps(s, per_stream_hint)
            unit = s.get("timestamp_unit", "unknown")
            # 交叉校验：登记表的 timestamp_unit 可能判为 unknown（列名未命中词表
            # 时），用实际读到的时间戳列名重推断一次，纠正这类漏判（真实案例：
            # mcap_log_time_ns 不在 _TIMESTAMP_COLS 内，单位被判 unknown）。
            # hint 命中了与登记表不同的列（如嵌套的传感器时间列）时同样重推断：
            # 登记表的单位属于旧列，对新列不再适用（真实案例：信封流的
            # data.header.timestamp_us 为 µs，登记表单位 ns 属 mcap_log_time_ns）。
            if ts is not None and col_name and (
                unit not in TIME_UNITS or col_name != s.get("timestamp_column")
            ):
                name_unit = infer_unit(ts, col_name)["unit"]
                if name_unit in TIME_UNITS:
                    unit = name_unit
            # 单位自我纠正：判错的单位（如 parquet 时间戳被判 ns 实为 s）经
            # self_correct_unit 换候选重算，避免算出 2.5e10 Hz 这类非物理值；
            # unknown 亦进入纠正（按数值量级重推断，不再按秒兜底）。
            if ts is not None:
                correction = self_correct_unit(ts, unit)
                if correction.get("corrected"):
                    unit = correction["unit"]
            ts_ns, unit_info = (
                _normalize_to_ns(ts, unit) if ts is not None else (None, None)
            )
            # 帧序号类时间戳（如 metainfo 的 pts）无物理时间：不参与跨流对齐残差，
            # 但保留用于单流检查（乱序/重复/丢帧）。
            frame_indexed = unit == FRAME_UNIT
            # 单位未知（未归一化）的流同样不参与跨流对齐：其数值口径不确定，
            # 混入会产出无意义的残差（真实案例：残差 2.43e11 ms）。
            unit_known = is_unit_known(unit_info)
            per_stream[key] = {
                "kind": s.get("kind"),
                "ts": ts_ns,
                "ts_raw": ts,
                "source": s.get("path"),
                "nominal": _nominal_rate(s),
                "unit_info": unit_info,
                "timestamp_unit": unit,
                "timestamp_column": col_name or None,
                "frame_indexed": frame_indexed,
                "unit_known": unit_known,
            }
            if ts is None:
                if time_column:
                    streams_status[key] = (
                        f"未参与：指定时间列 {time_column!r} 无法匹配（读取失败或"
                        "该列不存在）"
                    )
                else:
                    streams_status[key] = "未参与：无法读取时间戳列"
            elif frame_indexed:
                streams_status[key] = "仅单流检查：帧序号时间戳（不参与跨流对齐残差判定）"
            elif not unit_known:
                streams_status[key] = (
                    "仅单流检查：时间戳单位未知，未归一化"
                    "（不参与跨流对齐，时长/采样率不可计算）"
                )
            else:
                streams_status[key] = "参与对齐"

    # 可对齐流数：统计能实际读到时间戳列、且已归一化到纳秒基准的真实时间流。
    # 帧序号流（frame_indexed）与单位未知流均只做单流检查，不参与跨流对齐残差。
    alignable = [
        k for k, p in per_stream.items()
        if p["ts"] is not None and not p.get("frame_indexed", False)
        and p.get("unit_known", False)
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
    unit_warnings: list[str] = []  # 单位未知/不可用的流，显式列出供模型转达
    for key, p in per_stream.items():
        if p["ts"] is None:
            # 无法参与对齐的流：标注未参与原因，不进入判定依据。
            stream_checks[key] = {"present": False,
                                  "status": "skipped",
                                  "reason": streams_status.get(key, "无法读取时间戳")}
            continue
        # 单流检查直接用纳秒基准数组；单位未知时绝对量由函数内置 None。
        unit_known = p.get("unit_known", is_unit_known(p.get("unit_info")))
        checks = _single_stream_checks(p["ts"], p.get("nominal"), unit_known, settings)
        checks["present"] = True
        # 透出该流时间戳的原始单位与换算说明（供核对归一化是否正确）。
        if p.get("unit_info"):
            checks["timestamp_unit"] = p.get("timestamp_unit")
            checks["timestamp_unit_basis"] = p["unit_info"].get("basis")
        # 实际采用的时间戳列名（time_column 参数指定或自动识别的结果）。
        checks["timestamp_column"] = p.get("timestamp_column")
        stream_checks[key] = checks
        # gap 定位（locate_gaps=True）：缺口位置是相对量，与单位无关，对单位
        # 未知的流同样可用（但注明时间口径未知）。burst 流的突发间静默不视为
        # 缺口，不输出 gap 定位。
        if locate_gaps and checks.get("stream_shape") == "burst":
            checks["gaps"] = {
                "gaps": [], "total": 0, "truncated": False,
                "note": "突发型流：突发间静默不视为缺口，不输出 gap 定位",
            }
        elif locate_gaps:
            gap_info = _locate_gaps(p["ts"], settings.sync_gap_report_limit)
            if not unit_known:
                gap_info["note"] = "时间戳单位未知：缺口位置为原值口径的相对量，非物理时刻"
            checks["gaps"] = gap_info
        # 仅"已归一化到纳秒"的真实时间流进入跨流对齐残差 / 漂移判定
        # （帧序号流与单位未知流均已在上方排除）。
        if not p.get("frame_indexed", False) and unit_known:
            valid_ts.append((key, p["ts"]))
        elif not unit_known and p["ts"] is not None:
            unit_warnings.append(
                f"流 {Path(key).name}：时间戳单位未知（列名 "
                f"{p.get('timestamp_column') or '未识别'} 未推断出时间单位），"
                "未归一化，不参与跨流对齐，其时长/采样率不可计算"
            )

    # 双口径交叉验证（多时钟共存）：流登记表存在 ≥2 个时间候选、且主口径判为
    # burst 时，用其余候选重算流形态。容器批量写入时间判 burst、传感器时间判
    # periodic 的组合是典型的写入伪影（真实案例：IMU 的 log_ns 口径 1.7 万个
    # 假 gap，header.timestamp_us 口径 0 gap 且规整周期）。矛盾必须透出，
    # 不静默选边——主判定保留，但可信度降级并给出重算建议。
    clock_conflicts: list[dict[str, Any]] = []
    registry_by_path = {s.get("path") or "": s for s in streams}
    for key, checks in stream_checks.items():
        if checks.get("stream_shape") != "burst":
            continue  # 只核主口径为 burst 的流（成本与误报双收敛）
        reg = registry_by_path.get(key) or {}
        candidates = reg.get("time_candidates") or []
        main_col = checks.get("timestamp_column")
        alternates = [
            c for c in candidates
            if c.get("path") != main_col and c.get("coverage", 0) >= 0.99
        ]
        if not alternates:
            continue
        clock_candidates: dict[str, Any] = {
            str(main_col): {
                "shape": checks.get("stream_shape"),
                "actual_rate_hz": checks.get("actual_rate_hz"),
            }
        }
        alt_notes: list[str] = []
        suspected = False
        for alt in alternates[:2]:
            alt_col = str(alt.get("path"))
            ts_alt, _ = _read_stream_timestamps(reg, alt_col)
            if ts_alt is None or len(ts_alt) < 3:
                continue
            alt_unit = alt.get("unit_hint")
            if alt_unit not in TIME_UNITS:
                alt_unit = infer_unit(ts_alt, alt_col)["unit"]
            if alt_unit not in TIME_UNITS:
                continue
            alt_ns, alt_info = _normalize_to_ns(ts_alt, alt_unit)
            if not is_unit_known(alt_info):
                continue
            shape = _classify_stream_shape(alt_ns, settings)
            n = len(alt_ns)
            span = float(alt_ns[-1] - alt_ns[0])
            rate = (n - 1) / span * 1e9 if span > 0 else None
            clock_candidates[alt_col] = {
                "shape": shape,
                "rate_hz": round(rate, 3) if rate else None,
            }
            if shape != checks.get("stream_shape"):
                suspected = True
                alt_notes.append(
                    f"{alt_col} 口径为 {shape}"
                    + (f"（约 {rate:.1f} Hz）" if rate else "")
                )
        if suspected:
            checks["clock_candidates"] = clock_candidates
            checks["clock_artifact_suspected"] = True
            checks["clock_note"] = (
                "多时间口径形态矛盾："
                + "；".join(alt_notes)
                + f"。主口径 {main_col} 疑似容器批量写入时间而非传感器采样节拍；"
                "建议用 time_column 指定传感器时间列重算后再下结论。"
            )
            clock_conflicts.append({
                "stream": Path(key).name,
                "main_column": main_col,
                "note": checks["clock_note"],
            })

    # episode 口径：无 episode 划分时整段视为一个 episode。
    has_episodes = bool(capabilities.get("has_episodes")) or bool(
        context.meta.get("episode_ids")
    )
    episode_note = "未检测到 episode 划分，将整个录制视为单个 episode。"

    # 流间对齐残差 + 漂移：基线经"排除静态/标定/元数据流 + 打分推荐"选出。
    align: dict[str, Any] = {"baseline": None, "residuals": {}}
    drift: dict[str, Any] = {"detected": False, "detail": {}}
    max_interval_hz = 0.0
    for key, ts in valid_ts:
        if len(ts) > 1:
            med = float(np.median(np.diff(np.sort(ts))))
            if med > 0 and 1e9 / med > max_interval_hz:
                max_interval_hz = 1e9 / med  # 纳秒间隔 → Hz

    baseline_key, baseline_rec = _recommend_baseline(valid_ts, per_stream, streams, settings)

    # 用户指定基线：按文件名子串匹配可对齐流。指定且无法匹配时返回结构化错误
    # 并列出候选，不静默回退到自动推荐——静默回退会令模型误以为指定已生效。
    if baseline_stream:
        hint = baseline_stream.lower()
        matched = [(k, ts) for k, ts in valid_ts if hint in Path(k).name.lower()]
        if not matched:
            return {
                "success": False,
                "error": "baseline_no_match",
                "reason": f"baseline_stream={baseline_stream!r} 未匹配到任何可对齐流",
                "user_message": (
                    f"指定的基线流 {baseline_stream!r} 未匹配到任何可对齐流。"
                    f"当前可对齐流（已读到纳秒基准时间戳）共 {len(valid_ts)} 条："
                    f"{[Path(k).name for k, _ in valid_ts][:10]}。"
                    "请用文件名子串重新指定，或省略该参数改用自动推荐。"
                ),
            }
        baseline_key = matched[0][0]
        baseline_rec = {
            "stream": baseline_key,
            "reason": f"由用户指定（参数 baseline_stream={baseline_stream!r}，"
                      f"匹配到 {Path(baseline_key).name}）",
            "score": None,
            "excluded": [],
        }
        if len(matched) > 1:
            baseline_rec["note"] = (
                f"子串匹配到 {len(matched)} 条流，取第一条 "
                f"{Path(baseline_key).name}；如需精确指定请提供更完整的文件名"
            )

    if baseline_key is not None:
        base_ts = per_stream[baseline_key]["ts"]
        align["baseline"] = baseline_key
        for key, ts in valid_ts:
            if key == baseline_key:
                align["residuals"][key] = {"n_match": len(ts),
                                           "is_baseline": True}
            else:
                # 突发型流不参与漂移检测：跨突发的最近邻偏移是纯噪声（突发间
                # 静默的相位差），线性回归会误报漂移。残差仍输出（供参考）。
                if stream_checks.get(key, {}).get("stream_shape") == "burst":
                    drift["detail"][key] = {
                        "drift_detected": False,
                        "drift_slope_ms_per_s": None,
                        "note": "突发型流：不参与漂移检测（最近邻跨突发，偏移噪声无意义）",
                    }
                    continue
                res = _align_residuals(base_ts, ts)
                align["residuals"][key] = {**res, "is_baseline": False}
                d = _detect_drift(
                    base_ts, ts, settings.sync_drift_windows,
                    settings.sync_drift_slope_ms_per_s,
                )
                drift["detail"][key] = d
                if d.get("drift_detected"):
                    drift["detected"] = True

    # 三档判定。burst 流的 frame_loss_ratio 为 None（口径不适用），不计入。
    frame_loss_bad = any(
        (c.get("frame_loss_ratio") or 0) > settings.sync_frame_loss_ratio
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
    if unit_warnings:
        user_message += (
            f" 有 {len(unit_warnings)} 条流的时间戳单位未能确定，已排除在跨流对齐之外，"
            "其时长与采样率不可计算（详见 unit_warnings）。"
        )

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
                    "duration_ns": v.get("duration_ns"),
                } if v.get("present") else {"status": "skipped", "reason": v.get("reason")}
                for k, v in stream_checks.items()
            },
            "unit_warnings": unit_warnings,
            "clock_conflicts": clock_conflicts,
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

    if clock_conflicts:
        user_message = (
            str(user_message)
            + f" 另有 {len(clock_conflicts)} 条流存在多时间口径形态矛盾"
            "（容器时间口径与传感器时间口径的流形态判定不一致，疑似批量写入"
            "伪影）：详见 measurements.stream_checks 各流的 clock_note，"
            "建议用 time_column 指定传感器时间列重算后再下结论。"
        )

    return {
        "success": True,
        "verification_level": "timestamp_consistency",
        "verification_note": "仅基于时间戳一致性推断；物理级对齐需互相关实测（未来 v2）。",
        "result": result,
        "episode_note": episode_note if not has_episodes else "存在 episode 划分。",
        "baseline_stream": align["baseline"],
        # 基线推荐说明（含被排除的静态/标定流及原因），模型可直接转述。
        "baseline_recommendation": baseline_rec,
        "streams_status": streams_status,
        "skipped_checks": skipped_checks,
        # 单位未知/不可用的流显式列出，避免模型把"未参与"误读为"已检查通过"。
        "unit_warnings": unit_warnings,
        # 多时间口径形态矛盾的流清单（容器 vs 传感器时间判定不一致）。
        "clock_conflicts": clock_conflicts,
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
    baseline_stream: str | None = None,
    streams: list[str] | None = None,
    time_column: str | None = None,
    locate_gaps: bool = False,
) -> dict:
    """检查各流之间的时间同步与漂移（v1，仅时间戳一致性）。

    逐流读取时间戳，检查单调性、重复、丢帧率、采样率、时长，并做流间最近邻
    对齐残差与窗口漂移检测。默认自动推荐基线并对全部可对齐流检查；可用参数
    收窄范围（如只比对左右手套、指定基线流、定位数据缺口）。

    Args:
        baseline_stream: 可选，指定对齐基线流（路径或文件名的子串，如
            "left_glove_emf_poses"）。省略时自动推荐（排除静态/标定/元数据流，
            按"覆盖全时长 + 间隔稳定 + 帧数充足"打分）。指定值无法匹配任何
            可对齐流时返回结构化错误并列出可用候选，不静默回退。
        streams: 可选，只检查这些流（文件名子串列表，如 ["left_glove",
            "right_glove"] 表示只比对左右手套）。省略时检查全部流；指定且无
            匹配时返回结构化错误并列出可用流名。
        time_column: 可选，指定用哪一列作为时间戳（列名或子串，如
            "log_time"）。数据集含多个时间列（如 MCAP 的 log_time 与
            publish_time）时用它明确指定；省略时自动识别。
        locate_gaps: 默认 False。置 True 时额外定位每条流的数据缺口（起止
            时刻、持续时长、估算缺失帧数），用于回答"缺口发生在什么时刻"。

    Returns:
        统一质检返回格式：result（pass/warn/fail）、measurements（stream_checks
        逐流指标，时长/间隔字段带 _ns 单位后缀；residuals 相对基线的最近邻
        残差）、thresholds、affected_episodes、user_message、
        baseline_recommendation（基线推荐理由与排除清单）、unit_warnings
        （单位未知的流清单）。流数不足时返回 not_applicable。
    """
    return check_temporal_sync_impl(
        wrapper.context,
        baseline_stream=baseline_stream,
        streams=streams,
        time_column=time_column,
        locate_gaps=locate_gaps,
    )
