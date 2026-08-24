"""流探测工具（质检层）。

对已加载数据集输出"设备清单"：视频流、IMU、力/力矩通道、标定文件、时钟来源。
优先复用 ``RunContext.meta`` 中已有的嗅探结果（能力标签 + 流登记表），本工具
负责补充需要读时间戳/元数据才能得到的运行时指标（实际采样率、帧数、时长等）。

采样率实测基于流登记表**按需读取**：对每条表格流只读时间戳列（usecols / pyarrow
列裁剪），计算后立即释放；结果回写 ``meta["streams"][...]["measured_rate"]`` 缓存，
重复调用不重复读盘。单条流失败标 unknown，不影响其他流。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools import _sniffing

# 常见时间戳列名（用于实测采样率）。直接复用 _sniffing 的权威集合，避免两份
# 不同步导致真实数据的时间戳列（timestamp_ns / exposure_start_utc_ns 等）漏匹配。
_TIMESTAMP_COLS = _sniffing._TIMESTAMP_COLS


def _read_timestamp_only(path: str, fmt: str) -> pd.Series | None:
    """按需读取文件的时间戳列（不读全量，立即释放）。

    Args:
        path: 文件路径。
        fmt: 格式（csv / parquet / json）。

    Returns:
        时间戳列 Series；无时间戳列或读取失败返回 None。
    """
    from app.tools.load_dataset import _detect_encoding

    try:
        if fmt == "csv":
            encoding = _detect_encoding(Path(path).read_bytes())
            # 先读全部列名，再只读时间戳列。
            df_head = pd.read_csv(path, encoding=encoding, nrows=0, engine="python")
            cols = [c for c in df_head.columns if str(c).lower().strip() in _TIMESTAMP_COLS]
            if not cols:
                return None
            ts = pd.read_csv(
                path, encoding=encoding, usecols=[cols[0]], engine="python"
            )[cols[0]]
            return ts
        if fmt == "parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            schema_cols = [c.lower().strip() for c in pf.schema.names]
            ts_col = next(
                (c for c in pf.schema.names if str(c).lower().strip() in _TIMESTAMP_COLS),
                None,
            )
            if ts_col is None:
                return None
            table = pf.read(columns=[ts_col])
            return pd.Series(table.column(ts_col).to_pylist())
        if fmt == "json":
            encoding = _detect_encoding(Path(path).read_bytes())
            df = pd.read_json(path, encoding=encoding)
            ts_col = _find_timestamp_column(df)
            return df[ts_col] if ts_col else None
        return None
    except Exception:  # noqa: BLE001
        return None


def _measure_rate_from_file(
    path: str, fmt: str, channels: list[str]
) -> dict[str, Any]:
    """从文件实测采样率（均值 + 抖动）。

    Args:
        path: 文件路径。
        fmt: 文件格式（csv/parquet/json）。
        channels: 该流的通道列（用于判断数据是否存在）。

    Returns:
        dict，含 present、sample_rate_hz、jitter_ms、n_samples；文件缺失、格式
        损坏、无时间戳列或通道缺失时 present=False 并注明原因。
    """
    if not Path(path).exists():
        return {"present": False, "reason": f"文件不存在：{path}"}
    ts = _read_timestamp_only(path, fmt)
    if ts is None:
        return {"present": False, "reason": "未找到时间戳列或读取失败"}
    try:
        ts = pd.to_numeric(ts, errors="coerce").dropna().sort_values()
        if len(ts) < 2:
            return {"present": False, "reason": "时间戳样本不足"}
        diffs = ts.diff().dropna()
        med = diffs.median()
        if med and med > 0:
            diffs = diffs[diffs <= med * 10]
        mean_interval = float(diffs.mean())
        if mean_interval <= 0:
            return {"present": False, "reason": "时间戳间隔非正"}
        jitter_ms = float(diffs.std()) * 1000.0
        return {
            "present": True,
            "sample_rate_hz": round(1.0 / mean_interval, 3) if mean_interval else None,
            "jitter_ms": round(jitter_ms, 3),
            "n_samples": int(len(ts)),
            "timestamp_column": str(ts.name) if ts.name else None,
        }
    except Exception:  # noqa: BLE001
        return {"present": False, "reason": "时间戳解析失败"}


def _find_timestamp_column(df: pd.DataFrame) -> str | None:
    """在 DataFrame 中查找时间戳列。"""
    for c in df.columns:
        if str(c).lower().strip() in _TIMESTAMP_COLS:
            return str(c)
    return None


def _measure_stream_rate(
    stream: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any]:
    """实测单条流的采样率，带缓存（回写 meta["streams"]）。

    Args:
        stream: 流登记项（含 path/format/kind）。
        meta: RunContext.meta（用于读写 measured_rate 缓存）。

    Returns:
        dict，含 present、sample_rate_hz、jitter_ms、n_samples 或 reason。
    """
    cached = stream.get("measured_rate")
    if isinstance(cached, dict) and cached.get("present") is not None:
        return cached  # 缓存命中，不重复读盘

    if stream.get("kind") == "video":
        result = {"present": False, "reason": "视频流采样率由 ffprobe 提供（见视频元数据）"}
    else:
        result = _measure_rate_from_file(
            stream.get("path", ""),
            stream.get("format", ""),
            stream.get("channels", []),
        )

    stream["measured_rate"] = result  # 回写缓存
    return result


def inspect_streams_impl(context: RunContext) -> dict[str, Any]:
    """探测已加载数据集的设备清单。

    Args:
        context: 运行时上下文（复用 meta.capabilities / streams，必要时按需读盘）。

    Returns:
        dict，含 success、dataset、clock_source、video_streams、imus、
        force_channels、calibration、summary。前置条件不满足（未加载数据集）时
        返回 success=False 的 not_applicable 结果。

    Raises:
        不直接抛出异常；错误以结构化 dict 返回。
    """
    capabilities = context.meta.get("capabilities", {})
    streams = context.meta.get("streams", [])
    if context.df is None and not capabilities:
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载任何数据集",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset 加载数据，再执行 inspect_streams。",
        }

    # --- 时钟来源：meta 声明 > 用户配置 > unknown ---------------------------
    clock_source = "unknown"
    declared = context.meta.get("clock_source")
    if declared in ("unified", "per-device"):
        clock_source = declared
    settings_cfg = getattr(context, "clock_source_override", None)
    if settings_cfg in ("unified", "per-device"):
        clock_source = settings_cfg

    # 若流登记表缺失（旧数据或异常），从 video_meta 兜底。
    if not streams and context.meta.get("video_meta"):
        streams = [
            {"path": v.get("file", "unknown"), "format": "video",
             "kind": "video", "channels": [], "role": _sniffing.infer_role(v.get("file", ""))}
            for v in context.meta["video_meta"]
        ]
        context.meta["streams"] = streams

    # --- 视频流 -------------------------------------------------------------
    video_streams: list[dict[str, Any]] = []
    video_rate_cache: dict[str, Any] = {}
    for vmeta in context.meta.get("video_meta", []):
        src = vmeta.get("file", "unknown")
        if vmeta.get("ffprobe_available"):
            nb_frames = vmeta.get("nb_frames")
            nb_source = vmeta.get("nb_frames_source", "probe")
            nb_basis = vmeta.get("nb_frames_basis", "")
            duration_s = vmeta.get("duration")
            # 估算值展示必须带"估算"字样，禁止伪装成实测。
            if nb_source == "estimated":
                nb_display = f"约{nb_frames}（估算）" if nb_frames is not None else "未知（无法估算）"
            else:
                nb_display = nb_frames
            video_streams.append({
                "source": src,
                "role": _sniffing.infer_role(src),
                "nominal_fps": vmeta.get("fps"),
                "actual_fps": vmeta.get("fps"),
                "nb_frames": nb_display,
                "nb_frames_source": nb_source,
                "nb_frames_basis": nb_basis,
                "duration_s": duration_s,
                "resolution": (
                    f"{vmeta.get('width')}x{vmeta.get('height')}"
                    if vmeta.get("width") and vmeta.get("height")
                    else None
                ),
                "codec": vmeta.get("codec"),
            })
        else:
            video_streams.append({
                "source": src,
                "role": _sniffing.infer_role(src),
                "status": "unknown",
                "reason": vmeta.get("user_message", "ffprobe 不可用"),
            })

    # --- IMU / 力 / 其他表格流：从流登记表按需实测采样率 -----------------------
    imu_streams: list[dict[str, Any]] = []
    force_stream: dict[str, Any] | None = None
    action_streams: list[dict[str, Any]] = []
    other_streams: list[dict[str, Any]] = []

    for i, s in enumerate(streams):
        kind = s.get("kind")
        rate = _measure_stream_rate(s, context.meta)
        entry = {
            "source": s.get("path"),
            "kind": kind,
            "role": s.get("role", {}),
            "channels": s.get("channels", []),
            "sample_rate": rate,
            # 第 2 层内容指纹透出：语义标签 + 依据 + 置信 + 状态 + 轴数 + 时间戳列。
            "semantic_label": s.get("semantic_label"),
            "label_evidence": s.get("label_evidence"),
            "label_confidence": s.get("label_confidence"),
            "label_source": s.get("label_source"),  # user_confirmed / content_fingerprint / dictionary / None
            "status": s.get("status", "active"),
            "imu_axes": s.get("imu_axes"),
            "timestamp_column": s.get("timestamp_column"),
        }
        if kind == "video":
            continue  # 视频已在上方 video_streams 处理
        if s.get("status") == "empty":
            # 空流（行数 ≤ 阈值）显式标注，不计入可对齐流数。
            other_streams.append(entry)
            continue
        if kind == "imu":
            imu_streams.append(entry)
        elif kind == "force":
            force_stream = entry
        elif kind == "actions":
            action_streams.append(entry)
        elif kind == "pose":
            other_streams.append(entry)
        else:
            other_streams.append(entry)

    # IMU 汇总。优先用流配对（accel+gyro=六轴）标注轴数，否则用能力标签。
    imu_6axis_pair = any(
        p.get("type") == "imu_6axis" for p in context.meta.get("stream_pairs", [])
    )
    imus: list[dict[str, Any]] = []
    if imu_streams:
        axes = 6 if imu_6axis_pair else capabilities.get("imu_axes")
        imu_entry: dict[str, Any] = {"axes": axes, "streams": imu_streams}
        if imu_6axis_pair:
            imu_entry["pairing"] = "accel + gyro 配对为一组六轴 IMU"
        imus.append(imu_entry)
    elif capabilities.get("has_imu"):
        # 有 IMU 能力但无登记流（旧数据），用能力标签兜底。
        imus.append({
            "axes": capabilities.get("imu_axes"),
            "channels": capabilities.get("imu_channels", []),
            "sample_rate": {"present": False, "reason": "流登记表未含 IMU 流"},
        })

    # 力/力矩汇总。
    if force_stream is not None:
        force = {
            "present": True,
            "channels": force_stream["channels"],
            "n_channels": len(force_stream["channels"]),
            "role": force_stream.get("role", {}),
            "sample_rate": force_stream["sample_rate"],
        }
    else:
        force = {"present": False, "channels": [], "n_channels": 0}

    # 其他流（动作/位姿/未知/空流）一并给出，便于模型理解完整设备清单。
    other_stream_list = [*action_streams, *other_streams]
    # 空流清单：单独列出，明确标注未使用。
    empty_streams = [
        {"source": s.get("path"), "semantic_label": s.get("semantic_label"),
         "label_evidence": s.get("label_evidence")}
        for s in streams if s.get("status") == "empty"
    ]

    # --- 标定 -------------------------------------------------------------
    has_calib = bool(capabilities.get("has_calibration"))
    calibration = {"present": has_calib, "parameters": "unknown"}
    calib_detail = context.meta.get("calibration_detail")
    if has_calib and calib_detail:
        names = [c.get("name") for c in calib_detail]
        calibration["parameters"] = (
            f"已检测到 {len(calib_detail)} 个标定文件：{names}；"
            f"依据 {calib_detail[0].get('evidence', '')}"
        )
        calibration["files"] = calib_detail
    elif has_calib and context.meta.get("source"):
        calibration["parameters"] = "已检测到标定文件（参数详见源目录 calib 文件）"

    # --- 汇总 -----------------------------------------------------------
    # 用户确认覆盖清单（第 4 层）：来自 meta.user_profile，标注哪些标签是
    # user_confirmed（覆盖第 1-3 层自动识别）。
    user_profile = context.meta.get("user_profile", {})
    user_confirmed_overrides = [
        {"filename": fname, **mapping}
        for fname, mapping in user_profile.get("streams", {}).items()
    ]

    summary = {
        "n_video_streams": len(video_streams),
        "n_imus": len(imus),
        "has_force": force["present"],
        "has_calibration": has_calib,
        "has_imu_6axis_pair": imu_6axis_pair,
        "has_media_metainfo_pair": any(
            p.get("type") == "media_metainfo" for p in context.meta.get("stream_pairs", [])
        ),
        "n_empty_streams": len(empty_streams),
        "n_user_confirmed": len(user_confirmed_overrides),
        "clock_source": clock_source,
        "n_table_streams": len([s for s in streams if s.get("kind") != "video"]),
    }

    return {
        "success": True,
        "dataset": context.dataset_id,
        "clock_source": clock_source,
        "video_streams": video_streams,
        "imus": imus,
        "force_channels": force,
        "calibration": calibration,
        "table_streams": other_stream_list,
        "empty_streams": empty_streams,
        "stream_pairs": context.meta.get("stream_pairs", []),
        "user_confirmed_overrides": user_confirmed_overrides,
        "summary": summary,
        "user_message": (
            f"已生成设备清单：{len(video_streams)} 路视频、{len(imus)} 个 IMU、"
            f"力通道 {'有' if force['present'] else '无'}、标定{'有' if has_calib else '无'}；"
            f"空流 {len(empty_streams)} 条（已标记未使用）；时钟来源 {clock_source}。"
        ),
    }


@tool
def inspect_streams(
    wrapper: RunContextWrapper[RunContext],
) -> dict:
    """探测当前已加载数据集的设备清单（视频/IMU/力/标定/时钟来源）。

    复用 load_dataset 已记录的流登记表与能力标签，按需读取各流时间戳列实测
    采样率（结果缓存于 meta，重复调用不重复读盘）。视频相关项依赖 ffprobe，
    不可用时对应项标 unknown 并说明原因。

    Args:
        无（基于 RunContext.meta）。

    Returns:
        dict，含 success、clock_source、video_streams、imus、force_channels、
        calibration、summary；未加载数据集时返回 not_applicable。
    """
    return inspect_streams_impl(wrapper.context)
