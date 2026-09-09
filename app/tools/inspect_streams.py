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

import numpy as np
import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools import _sniffing

# 常见时间戳列名（用于实测采样率）。直接复用 _sniffing 的权威集合，避免两份
# 不同步导致真实数据的时间戳列（timestamp_ns / exposure_start_utc_ns 等）漏匹配。
_TIMESTAMP_COLS = _sniffing._TIMESTAMP_COLS


def _resolve_timestamp_column(
    columns: list[str], main: str | None, column_hint: str | None
) -> str | None:
    """按 column_hint 解析时间戳列名（无 hint 时用自动识别主列）。

    Args:
        columns: 全部列名。
        main: find_timestamp_columns 自动识别的主列（可 None）。
        column_hint: 用户指定的时间列（列名或子串，大小写不敏感）。

    Returns:
        实际采用的时间戳列名；hint 指定了但无列可匹配时返回 None（调用方
        应向模型明确报"指定列不存在"，不得静默回退到主列）。
    """
    if not column_hint:
        return main
    hint = column_hint.lower()
    exact = [c for c in columns if str(c).lower() == hint]
    if exact:
        return exact[0]
    partial = [c for c in columns if hint in str(c).lower()]
    if partial:
        return partial[0]
    return None  # hint 明确但匹配不到：让调用方报错，不静默回退


def _read_timestamp_only(
    path: str, fmt: str, column_hint: str | None = None
) -> pd.Series | None:
    """按需读取文件的时间戳列（不读全量，立即释放）。

    列识别经 find_timestamp_columns：词表命中优先，未命中则内容指纹回退
    （单调递增 + 量级符合时间单位）；主列选择物理时间 > 帧序号。
    column_hint 指定时优先用该列（精确或子串匹配）；指定且匹配不到任何列时
    返回 None，由调用方在 streams_status 中注明"指定时间列不存在"。

    Args:
        path: 文件路径。
        fmt: 格式（csv / parquet / json / jsonl）。
        column_hint: 可选，指定时间戳列（列名或子串）。

    Returns:
        时间戳列 Series（其 name 为列名）；无时间戳列或读取失败返回 None。
    """
    from app.tools.load_dataset import _detect_encoding
    from app.tools._sniffing import find_timestamp_columns

    try:
        sample_rows = 20  # 仅读前若干行用于内容指纹回退
        # 嵌套时间路径（含 "."，如 data.header.timestamp_us）：经点分路径逐行
        # 提取（仅 jsonl/json）。用户确认的传感器时间列即此类嵌套路径。
        if column_hint and "." in column_hint:
            from app.tools._data_access import read_nested_time_column

            return read_nested_time_column(path, fmt, column_hint)
        if fmt == "csv":
            encoding = _detect_encoding(Path(path).read_bytes())
            df_head = pd.read_csv(path, encoding=encoding, nrows=0, engine="python")
            sample = pd.read_csv(path, encoding=encoding, nrows=sample_rows, engine="python")
            ts_info = find_timestamp_columns(list(df_head.columns), sample)
            col = _resolve_timestamp_column(
                list(df_head.columns), ts_info["main"], column_hint
            )
            if col is None:
                return None
            return pd.read_csv(path, encoding=encoding, usecols=[col], engine="python")[col]
        if fmt == "parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            sample = pf.read().slice(0, sample_rows).to_pandas()
            ts_info = find_timestamp_columns(list(pf.schema.names), sample)
            col = _resolve_timestamp_column(
                list(pf.schema.names), ts_info["main"], column_hint
            )
            if col is None:
                return None
            table = pf.read(columns=[col])
            return pd.Series(table.column(col).to_pylist(), name=col)
        if fmt == "json":
            # 经统一 reader 读取（JSON 顶层 dict 按行列表键 frames/data 展开，
            # 避免把标量键如 fps 当数据列）。
            from app.tools import _data_access

            df = _data_access.read_stream_full(path, fmt)
            if df is None:
                return None
            ts_info = find_timestamp_columns(list(df.columns), df.head(sample_rows))
            col = _resolve_timestamp_column(
                list(df.columns), ts_info["main"], column_hint
            )
            return df[col] if col else None
        if fmt == "jsonl":
            # JSONL：经统一 reader 逐行解析（lines=True），与 .json 严格区分。
            from app.tools import _data_access

            df = _data_access.read_stream_full(path, fmt)
            if df is None:
                return None
            ts_info = find_timestamp_columns(list(df.columns), df.head(sample_rows))
            col = _resolve_timestamp_column(
                list(df.columns), ts_info["main"], column_hint
            )
            return df[col] if col else None
        return None
    except Exception:  # noqa: BLE001
        return None


def _measure_rate_from_file(
    path: str,
    fmt: str,
    channels: list[str],
    timestamp_unit: str | None = None,
    stream: dict[str, Any] | None = None,
    column_hint: str | None = None,
) -> dict[str, Any]:
    """从文件实测采样率（均值 + 抖动），先把时间戳归一化到纳秒基准。

    多时钟交叉验证：主口径判 burst 且流登记表存在其它时间候选时，用候选
    （传感器时间列）重算形态——形态矛盾（burst vs periodic）即透出
    clock_artifact_suspected + clock_note + clock_candidates（真实案例：
    MCAP 信封流的容器批量写入时间把 IMU 算成 ~58 万 Hz 荒谬采样率，
    传感器时间口径下是规整 ~796 Hz 周期）。

    Args:
        path: 文件路径。
        fmt: 文件格式（csv/parquet/json）。
        channels: 该流的通道列（用于判断数据是否存在）。
        timestamp_unit: 时间戳单位（s/ms/us/ns，来自嗅探推断）。传入可用的时间单位
            时，先把差分换算到纳秒再算采样率，避免"微秒被当成秒"之类导致采样率
            误算成 10^-9；无法确定单位（None/unknown/frame_index）时按原值计算，
            并注明单位未知。
        stream: 流登记项（含 time_candidates），用于多时钟交叉验证。

    Returns:
        dict，含 present、sample_rate_hz、jitter_ms、n_samples、timestamp_unit、
        timestamp_unit_basis；文件缺失、格式损坏、无时间戳列或通道缺失时
        present=False 并注明原因。
    """
    from app.tools.timestamp_units import infer_unit, self_correct_unit, to_ns

    if not Path(path.split("::")[0]).exists():
        return {"present": False, "reason": f"文件不存在：{path}"}
    # h5 节点流（format="h5"，path 带 ::node）：时间戳为 compound 字段列。
    if fmt == "h5" and "::" in path:
        from app.tools._data_access import read_h5_node_field
        from app.tools.timestamp_units import infer_unit

        field = column_hint or "timestamp"
        ts = read_h5_node_field(path, field)
        if ts is None and field != "timestamp":
            field = "timestamp"
            ts = read_h5_node_field(path, field)
        if ts is None:
            return {"present": False,
                    "reason": f"h5 节点无时间戳字段（{field}）"}
        # 单位强制量级推断（登记表单位属别的节点/列；真实案例 ms 被当 s →
        # 0.1 Hz 千倍失真）。timestamp_unit 参数改为推断结果。
        timestamp_unit = infer_unit(
            pd.to_numeric(ts, errors="coerce").dropna().to_numpy(), field
        )["unit"]
    else:
        ts = _read_timestamp_only(path, fmt, column_hint)
    if ts is None:
        hint_note = f"（指定时间列 {column_hint} 不存在或读取失败）" if column_hint else ""
        return {"present": False, "reason": f"未找到时间戳列或读取失败{hint_note}"}
    try:
        ts = pd.to_numeric(ts, errors="coerce").dropna().sort_values()
        if len(ts) < 2:
            return {"present": False, "reason": "时间戳样本不足"}
        # 单位自我纠正：初始单位算出的采样率超物理区间时自动换候选单位重算。
        init_unit = timestamp_unit if timestamp_unit in ("s", "ms", "us", "ns") else "unknown"
        # 确认列（column_hint）与登记表单位所属列不同 → 旧单位不再适用，按
        # 新列名重推断（真实案例：确认 data.header.timestamp_us（µs）后仍按
        # 登记表 ns 算 → 1e6 Hz 千倍失真；与 check_temporal_sync 同款处理）。
        if column_hint and column_hint != (stream or {}).get("timestamp_column"):
            name_unit = infer_unit(ts.to_numpy(), column_hint)["unit"]
            if name_unit in ("s", "ms", "us", "ns"):
                init_unit = name_unit
        correction = self_correct_unit(ts.to_numpy(), init_unit)
        unit = correction["unit"] if correction["unit"] in ("s", "ms", "us", "ns") else None
        corrected = correction.get("corrected", False)
        # 归一化到纳秒基准（仅当单位可换算）；否则保留原值（未知单位按秒兜底）。
        normalized = unit is not None
        ts_arr = to_ns(ts.to_numpy(), unit) if normalized else ts.to_numpy()
        diffs = np.diff(ts_arr)
        med = float(np.median(diffs)) if len(diffs) > 0 else 0.0
        if med and med > 0:
            diffs = diffs[diffs <= med * 10]
        mean_interval = float(diffs.mean()) if len(diffs) > 0 else 0.0
        if mean_interval <= 0:
            return {"present": False, "reason": "时间戳间隔非正"}
        if not normalized:
            # 单位未知：不按秒兜底。此前会算出 1e-7 Hz 这类伪值，且与
            # check_temporal_sync 的兜底结果不一致；现统一为显式不可用。
            return {
                "present": True,
                "sample_rate_hz": None,
                "jitter_ms": None,
                "n_samples": int(len(ts)),
                "timestamp_column": str(ts.name) if ts.name else None,
                "timestamp_unit": init_unit,
                "timestamp_unit_basis": (
                    f"单位未知（{init_unit}），未归一化，采样率与抖动不可计算"
                    f"——{correction.get('basis', '无法推断时间单位')}"
                ),
                "unit_corrected": corrected,
                "rate_unavailable": True,
            }
        # 已归一化到纳秒：采样率 = 1e9 / 平均间隔(ns)。
        sample_rate = 1e9 / mean_interval
        jitter_ms = float(diffs.std()) / 1e6
        if corrected:
            # 发生纠正时只显示纠正后的连贯表述，避免与"原始单位 s"拼接自相矛盾。
            unit_note = (
                f"（{correction.get('basis', '单位经自我纠正')}；"
                f"当前按单位 {unit} 归一化到纳秒计算）"
            )
        else:
            unit_note = f"（原始单位 {unit}，已归一化到纳秒）"

        # 主口径形态判定：间隔分布与 burst 判据同 check_temporal_sync 口径
        #（mean ≥ 3×median，容忍不同实现的阈值常量）。
        diffs_all = np.diff(ts_arr)
        med_all = float(np.median(diffs_all)) if len(diffs_all) else 0.0
        shape = "periodic"
        if len(ts_arr) < 3 or med_all <= 0:
            shape = "static"
        elif float(diffs_all.mean()) >= 3.0 * med_all:
            shape = "burst"

        result = {
            "present": True,
            "sample_rate_hz": round(sample_rate, 3),
            "jitter_ms": round(jitter_ms, 3),
            "n_samples": int(len(ts)),
            "timestamp_column": str(ts.name) if ts.name else None,
            "timestamp_unit": unit or init_unit,
            "timestamp_unit_basis": unit_note,
            "unit_corrected": corrected,
            "stream_shape": shape,
        }

        # burst 形态：常规采样率口径失真（批量写入间隔 → ~58 万 Hz 荒谬值）。
        # 主值置空、另给有效速率（n/span 写盘平均口径），note 说明原因。
        if shape == "burst":
            span_s = float(ts_arr[-1] - ts_arr[0]) / 1e9 if len(ts_arr) > 1 else 0.0
            result["sample_rate_hz"] = None
            result["sample_rate_note"] = "突发型流：常规采样率口径不适用（批量写入间隔）"
            if span_s > 0:
                result["effective_rate_hz"] = round((len(ts_arr) - 1) / span_s, 3)

        # 双口径交叉验证：主口径 burst 且登记表有其它时间候选 → 用候选重算。
        # 候选为 periodic → 主值**替换为传感器口径真值**（用户明确要求 UI 显示
        # 正确数字；容器失真值保留在 clock_candidates 供审计），矛盾经
        # clock_note 透出，不静默。
        cands = (stream or {}).get("time_candidates") or []
        main_col = str(ts.name) if ts.name else None
        alternates = [
            c for c in cands
            if c.get("path") != main_col and c.get("coverage", 0) >= 0.99
            and c.get("monotonic")
        ]
        if shape == "burst" and alternates:
            clock_candidates: dict[str, Any] = {
                str(main_col): {"shape": shape,
                                "sample_rate_hz": round(sample_rate, 3)}
            }
            for alt in alternates[:2]:
                alt_col = str(alt.get("path"))
                from app.tools._data_access import read_nested_time_column

                alt_series = read_nested_time_column(path, fmt, alt_col)
                if alt_series is None:
                    continue
                alt_ts = pd.to_numeric(alt_series, errors="coerce")\
                    .dropna().sort_values()
                if len(alt_ts) < 3:
                    continue
                alt_arr = alt_ts.to_numpy(dtype=float)
                alt_unit = alt.get("unit_hint")
                if alt_unit not in ("s", "ms", "us", "ns"):
                    alt_unit = infer_unit(alt_arr, alt_col)["unit"]
                if alt_unit not in ("s", "ms", "us", "ns"):
                    continue
                alt_ns = to_ns(alt_arr, alt_unit)
                alt_diffs = np.diff(alt_ns)
                alt_med = float(np.median(alt_diffs)) if len(alt_diffs) else 0.0
                alt_span = float(alt_ns[-1] - alt_ns[0])
                if alt_span <= 0 or alt_med <= 0:
                    continue
                alt_shape = (
                    "burst" if float(alt_diffs.mean()) >= 3.0 * alt_med
                    else "periodic"
                )
                alt_rate = (len(alt_ns) - 1) / alt_span * 1e9
                clock_candidates[alt_col] = {
                    "shape": alt_shape,
                    "sample_rate_hz": round(alt_rate, 3),
                }
                if alt_shape != "burst":
                    result["clock_artifact_suspected"] = True
                    # 主值替换为传感器口径真值（UI 渲染主值字段——用户明确
                    # 要求显示正确数字）；容器失真值保留在 clock_candidates。
                    result["sample_rate_hz"] = round(alt_rate, 3)
                    result["sample_rate_note"] = (
                        f"传感器时间口径（{alt_col}）；容器批量写入口径曾算出 "
                        f"{round(sample_rate, 3)} Hz（失真，见 clock_candidates）"
                    )
                    result["clock_note"] = (
                        f"多时间口径形态矛盾：主口径 {main_col} 为 burst，"
                        f"而 {alt_col} 口径为 periodic（约 {alt_rate:.1f} Hz，"
                        "间隔规整）——主口径疑似容器批量写入时间而非传感器"
                        f"采样节拍；传感器时间口径的真实采样率约 {alt_rate:.1f} Hz。"
                        "check_temporal_sync 可用 time_column="
                        f"{alt_col} 重算对齐。"
                    )
            if len(clock_candidates) > 1:
                result["clock_candidates"] = clock_candidates
        return result
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
    confirmed_col = stream.get("time_column")  # 用户确认的传感器时间列（可嵌套）
    if isinstance(cached, dict) and cached.get("present") is not None:
        # 缓存失效：确认的时间列与缓存所用的列不同（确认前测的是旧口径，
        # 如容器批量写入时间算出的 ~58 万 Hz 荒谬采样率）→ 重测。
        cached_col = cached.get("timestamp_column")
        if not confirmed_col or cached_col == confirmed_col:
            return cached

    if stream.get("kind") == "video":
        result = {"present": False, "reason": "视频流采样率由 ffprobe 提供（见视频元数据）"}
    else:
        result = _measure_rate_from_file(
            stream.get("path", ""),
            stream.get("format", ""),
            stream.get("channels", []),
            stream.get("timestamp_unit"),
            stream=stream,
            column_hint=confirmed_col,
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
    # 多分辨率版本组映射：variant_path -> 主版本路径（登记级，标 variant_of）。
    video_variant_of: dict[str, str] = {}
    for pair in context.meta.get("stream_pairs", []):
        if pair.get("type") != "video_version_group":
            continue
        for v in pair.get("variants", []):
            if v.get("variant_of"):
                video_variant_of[v.get("path")] = v.get("variant_of")
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
                # 多分辨率版本组：非主版本标 variant_of（主版本基础名）。
                "variant_of": (
                    Path(video_variant_of[src]).stem if src in video_variant_of else None
                ),
            })
        else:
            video_streams.append({
                "source": src,
                "role": _sniffing.infer_role(src),
                "status": "unknown",
                "reason": vmeta.get("user_message", "ffprobe 不可用"),
                "variant_of": (
                    Path(video_variant_of[src]).stem if src in video_variant_of else None
                ),
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
            "timestamp_unit": s.get("timestamp_unit"),
            "timestamp_unit_basis": s.get("timestamp_unit_basis"),
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

    # 未分类流占比：unknown 为主时引导 agent 走 propose 确认（语义假设的
    # 结构化出口——绑定在工具返回上，任何对话路径都触达，不依赖 agent 自觉）。
    classified = sum(
        1 for s in streams
        if s.get("label_source") == "user_confirmed"
        or (s.get("semantic_label") or "").find("未知") < 0
        and s.get("kind") not in (None, "unknown")
    )
    n_streams = len(streams)
    unclassified_hint = None
    # 有未分类即提示（边界教训：恰半数确认时 13/26 < 0.5 为 False，恰好漏报
    # 用户实测场景）。比例措辞：全部未分类（强引导）vs 部分未分类（温和）。
    if n_streams >= 5 and classified < n_streams:
        unclassified = n_streams - classified
        unclassified_hint = (
            f"{unclassified}/{n_streams} 条流语义未分类（unknown/低置信）。"
            "回答涉及这些流的类别或分组时，请先用 propose_stream_semantics "
            "批量提交假设（工具会确定性验证），转述验证结果并请用户确认后"
            "落盘——一次确认，跨会话生效。"
        )

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
        "unclassified_hint": unclassified_hint,
        "user_message": (
            f"已生成设备清单：{len(video_streams)} 路视频、{len(imus)} 个 IMU、"
            f"力通道 {'有' if force['present'] else '无'}、标定{'有' if has_calib else '无'}；"
            f"空流 {len(empty_streams)} 条（已标记未使用）；时钟来源 {clock_source}。"
            + (f" {unclassified_hint}" if unclassified_hint else "")
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
