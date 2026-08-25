"""传感器数据合理性检查（质检层）。

把领域常识变成自动检查：单位推断、IMU 加速度/陀螺仪检查、力/力矩零漂与饱和、
通用 NaN/Inf 与恒定通道检测、静止段确定。数据来自流登记表，按需读取数值列，
不依赖 context.df 是否为主表。单条流失败标 unknown，不影响其他流。

判定三档 pass/warn/fail；因缺信息跳过的检查以 status=skipped + reason 独立可见；
无可检查流时返回"不适用"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.config import get_settings


def _read_columns(path: str, fmt: str, columns: list[str]) -> dict[str, np.ndarray] | None:
    """按需读取指定数值列（csv 用 usecols、parquet 用 pyarrow 列裁剪）。

    Args:
        path: 文件路径。
        fmt: 格式（csv / parquet / json）。
        columns: 需要读取的列名。

    Returns:
        dict {列名: numpy 数组}；读取失败返回 None。
    """
    from app.tools.load_dataset import _detect_encoding

    try:
        if fmt == "csv":
            encoding = _detect_encoding(Path(path).read_bytes())
            df = pd.read_csv(path, encoding=encoding, usecols=columns, engine="python")
            return {c: df[c].to_numpy(dtype=float) for c in columns if c in df.columns}
        if fmt == "parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            table = pf.read(columns=columns)
            return {c: np.asarray(table.column(c).to_pylist(), dtype=float) for c in columns}
        if fmt == "json":
            encoding = _detect_encoding(Path(path).read_bytes())
            df = pd.read_json(path, encoding=encoding)
            return {c: df[c].to_numpy(dtype=float) for c in columns if c in df.columns}
        return None
    except Exception:  # noqa: BLE001
        return None


def _split_imu_channels(
    channels: list[str], filename: str = ""
) -> tuple[list[str], list[str]]:
    """把 IMU 通道拆分为加速度计列与陀螺仪列。

    优先按列名命中（accel_* / gyro_*）；当列为通用 x/y/z（真实 accel.csv/gyro.csv
    用 x/y/z 而非 accel_x）时，用**文件名**区分——文件名含 accel 则该文件 x/y/z 全
    归加速度计，含 gyro/gyr 则全归陀螺仪。

    Args:
        channels: IMU 通道列名清单。
        filename: 文件路径或文件名（用于 x/y/z 通用列时的归属判断）。

    Returns:
        (accel_cols, gyro_cols)。
    """
    lower_name = filename.lower()
    accel = [c for c in channels if "accel" in c]
    gyro = [c for c in channels if ("gyro" in c or "gyr" in c)]
    # 通用列（x/y/z）按文件名归属。
    generic = [c for c in channels if str(c).lower().strip() in ("x", "y", "z")]
    if not accel and not gyro and generic:
        if "accel" in lower_name:
            accel = generic
        elif "gyro" in lower_name or "gyr" in lower_name:
            gyro = generic
    return accel, gyro


def _nan_inf_ratio(arr: np.ndarray) -> float:
    """计算 NaN/Inf 比例。"""
    if arr.size == 0:
        return 0.0
    bad = np.isnan(arr) | np.isinf(arr)
    return round(float(np.mean(bad)), 4)


def _constant_columns(
    data: dict[str, np.ndarray], threshold: float
) -> list[str]:
    """检测恒定通道（归一化方差低于阈值，疑似掉线）。

    Args:
        data: {列名: 数值数组}。
        threshold: 恒定判定阈值（归一化方差）。

    Returns:
        恒定列名列表。
    """
    constant: list[str] = []
    for c, arr in data.items():
        vc = arr[~np.isnan(arr)]
        if vc.size <= 10:
            continue
        span = float(np.max(vc)) - float(np.min(vc))
        norm_var = float(np.var(vc)) / (span * span) if span > 0 else 0.0
        if norm_var < threshold:
            constant.append(c)
    return constant


def _saturation_ratio(data: dict[str, np.ndarray]) -> float:
    """计算量程饱和削顶比例（连续重复出现的极值点比例，取各通道最大值）。

    Args:
        data: {列名: 数值数组}。

    Returns:
        饱和削顶比例（0~1）；无有效数据返回 0.0。
    """
    sat = 0.0
    for c, v in data.items():
        vc = v[~np.isnan(v)]
        if vc.size == 0:
            continue
        vmax, vmin = float(np.max(vc)), float(np.min(vc))
        if vmax > vmin:
            flat = (v == vmax) | (v == vmin)
            sat = max(sat, float(np.mean(flat)))
    return round(sat, 4)


def _stationary_mask(
    norms: np.ndarray, window: int, var_threshold: float
) -> tuple[np.ndarray, float]:
    """用方差阈值法判静止段（滑动窗口内归一化方差低于阈值）。

    用变异系数平方（var / mean²）做归一化，使阈值对 m/s² 系与 g 系统一适用。

    Args:
        norms: 加速度计模长序列。
        window: 滑动窗口样本数。
        var_threshold: 归一化静止方差阈值（相对均值的平方）。

    Returns:
        (mask, static_ratio)：静止掩码与静止占比。
    """
    n = len(norms)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0
    mask = np.zeros(n, dtype=bool)

    def _is_static(seg: np.ndarray) -> bool:
        seg = seg[~np.isnan(seg)]
        if len(seg) < 3:
            return False
        mean = float(np.mean(seg))
        if mean == 0:
            return float(np.var(seg)) < var_threshold
        cv2 = float(np.var(seg)) / (mean * mean)
        return cv2 < var_threshold

    if n < window:
        if _is_static(norms):
            mask[:] = True
    else:
        for i in range(0, n - window + 1, window):
            if _is_static(norms[i : i + window]):
                mask[i : i + window] = True
    ratio = round(float(np.mean(mask)), 4)
    return mask, ratio


def _infer_accel_unit(
    static_norms: np.ndarray,
) -> tuple[str, dict[str, Any]]:
    """推断加速度计单位（m/s² / g / raw / 无法确定）。

    Args:
        static_norms: 静止段加速度计模长（去除 NaN）。

    Returns:
        (unit, 依据 dict)。
    """
    if len(static_norms) < 3:
        return "无法确定", {"reason": "静止段样本不足"}
    med = float(np.median(static_norms))
    if 8.5 <= med <= 11.0:
        return "m/s2", {"median_norm": round(med, 3), "参考": "静止模长≈9.8"}
    if 0.8 <= med <= 1.2:
        return "g", {"median_norm": round(med, 3), "参考": "静止模长≈1.0"}
    if med > 100.0:
        return "raw", {"median_norm": round(med, 3), "参考": "模长为大整数，疑似原始 ADC 值"}
    return "无法确定", {"median_norm": round(med, 3), "reason": "模长不在已知单位量级"}


def _infer_gyro_unit(static_gyro_magnitudes: np.ndarray) -> tuple[str, dict[str, Any]]:
    """推断陀螺仪单位（rad/s / deg/s / 无法确定）。

    Args:
        static_gyro_magnitudes: 静止段陀螺仪角速度模长（去除 NaN）。

    Returns:
        (unit, 依据 dict)。
    """
    if len(static_gyro_magnitudes) < 3:
        return "无法确定", {"reason": "静止段样本不足"}
    med = float(np.median(static_gyro_magnitudes))
    # rad/s 静止偏置通常 < 0.5；°/s 静止偏置通常 < 100。
    if med < 0.5:
        return "rad/s", {"median_deg_equiv": round(np.degrees(med), 3), "依据": "静止偏置很小，疑似 rad/s"}
    if med < 100.0:
        return "deg/s", {"median": round(med, 3), "依据": "静止偏置为几十量级，疑似 °/s"}
    return "无法确定", {"median": round(med, 3), "reason": "静止偏置量级异常"}


def _imu_check_units(
    context: RunContext,
    imu_streams: list[dict[str, Any]],
    table: str | None,
) -> list[dict[str, Any]]:
    """构建 IMU 检查单元（单流或 accel+gyro 配对组），按流登记表惰性读数据。

    优先用第 3 步已建立的 accel+gyro 六轴配对（stream_pairs），把 accel 表与 gyro
    表合成一个检查单元（accel 列来自 accel 文件的 x/y/z，gyro 列来自 gyro 文件的
    x/y/z），不依赖主表身份。指定 table 时仅检查该表。

    Args:
        context: 运行时上下文。
        imu_streams: 流登记表中 kind=imu 的流。
        table: 可选，指定检查对象表名。

    Returns:
        list[dict]，每项含 key、table_name、data、accel_cols、gyro_cols、rate、
        source（"pair" / "stream" / "explicit"）。
    """
    from app.tools import _data_access

    streams = context.meta.get("streams", [])
    pairs = context.meta.get("stream_pairs", [])
    units: list[dict[str, Any]] = []

    # 显式指定表：按名解析，不要求 kind=imu（如直接检查 accel.csv）。
    if table is not None:
        resolved = _data_access.resolve_table_name(context, table)
        if resolved["success"]:
            s = next((x for x in streams if Path(x.get("path", "")).name.lower() == table.lower()), None)
            channels = (s or {}).get("channels", []) or list(resolved["df"].columns)
            accel_cols, gyro_cols = _split_imu_channels(list(resolved["df"].columns), resolved["table_name"])
            units.append({
                "key": resolved["table_name"],
                "table_name": resolved["table_name"],
                "data": {c: resolved["df"][c].to_numpy(dtype=float) for c in accel_cols + gyro_cols if c in resolved["df"].columns},
                "accel_cols": accel_cols,
                "gyro_cols": gyro_cols,
                "rate": (s or {}).get("measured_rate", {}).get("sample_rate_hz") if isinstance((s or {}).get("measured_rate"), dict) else None,
                "source": "explicit",
            })
        return units

    # 六轴配对：accel 表 + gyro 表合成一个单元。
    used = set()
    for p in pairs:
        if p.get("type") != "imu_6axis":
            continue
        accel_paths = [t for t in p.get("streams", []) if "accel" in Path(t).name.lower()]
        gyro_paths = [t for t in p.get("streams", []) if "gyro" in Path(t).name.lower() or "gyr" in Path(t).name.lower()]
        if not accel_paths or not gyro_paths:
            continue
        accel_s = next((x for x in imu_streams if x.get("path") == accel_paths[0]), None)
        gyro_s = next((x for x in imu_streams if x.get("path") == gyro_paths[0]), None)
        accel_df = _data_access.read_stream_full(accel_paths[0], (accel_s or {}).get("format", "csv"))
        gyro_df = _data_access.read_stream_full(gyro_paths[0], (gyro_s or {}).get("format", "csv"))
        accel_cols, _ = _split_imu_channels(list(accel_df.columns) if accel_df is not None else [], Path(accel_paths[0]).name)
        _, gyro_cols = _split_imu_channels(list(gyro_df.columns) if gyro_df is not None else [], Path(gyro_paths[0]).name)
        # accel 与 gyro 文件可能都用 x/y/z 通用列 → 用前缀命名避免 key 冲突。
        data: dict[str, np.ndarray] = {}
        if accel_df is not None:
            for c in accel_cols:
                if c in accel_df.columns:
                    data[f"accel_{c}"] = accel_df[c].to_numpy(dtype=float)
        if gyro_df is not None:
            for c in gyro_cols:
                if c in gyro_df.columns:
                    data[f"gyro_{c}"] = gyro_df[c].to_numpy(dtype=float)
        accel_cols = [f"accel_{c}" for c in accel_cols]
        gyro_cols = [f"gyro_{c}" for c in gyro_cols]
        rate = None
        for x in (accel_s, gyro_s):
            if x and isinstance(x.get("measured_rate"), dict):
                rate = x.get("measured_rate", {}).get("sample_rate_hz")
                break
        units.append({
            "key": "accel + gyro（六轴配对）",
            "table_name": Path(accel_paths[0]).name + " + " + Path(gyro_paths[0]).name,
            "data": data,
            "accel_cols": accel_cols,
            "gyro_cols": gyro_cols,
            "rate": rate,
            "source": "pair",
        })
        used.update(accel_paths)
        used.update(gyro_paths)

    # 其余单个 IMU 流（未被配对覆盖）。
    for imu in imu_streams:
        if imu.get("path") in used:
            continue
        key = imu.get("path") or "imu"
        accel_cols, gyro_cols = _split_imu_channels(imu.get("channels", []), key)
        path = imu.get("path", "")
        fmt = imu.get("format", "")
        data = _read_columns(path, fmt, accel_cols + gyro_cols) if path else {}
        rate = (imu.get("measured_rate") or {}).get("sample_rate_hz") if isinstance(imu.get("measured_rate"), dict) else None
        units.append({
            "key": key,
            "table_name": Path(path).name if path else None,
            "data": data or {},
            "accel_cols": accel_cols,
            "gyro_cols": gyro_cols,
            "rate": rate,
            "source": "stream",
        })
    return units


def check_sensor_sanity_impl(context: RunContext, settings=None, table: str | None = None) -> dict[str, Any]:
    """执行传感器数据合理性检查。

    Args:
        context: 运行时上下文（复用 meta.streams / capabilities）。
        settings: 应用配置（阈值）；缺省读取 get_settings()。
        table: 可选，指定检查对象表名（与 profile_data 对齐）；缺省自动定位。

    Returns:
        统一质检返回：result（pass/warn/fail）、units、静止段、各检查项、
        skipped_checks、affected_episodes、dataset、user_message。
    """
    settings = settings or get_settings()
    capabilities = context.meta.get("capabilities", {})
    streams = context.meta.get("streams", [])
    dataset_id = context.dataset_id

    imu_streams = [s for s in streams if s.get("kind") == "imu"]
    force_streams = [s for s in streams if s.get("kind") == "force"]
    has_episodes = bool(capabilities.get("has_episodes")) or bool(
        context.meta.get("episode_ids")
    )
    episode_note = "未检测到 episode 划分，将整个录制视为单个 episode。"

    # 无可检查流（既无 IMU 也无力）→ 不适用。
    if not imu_streams and not force_streams and table is None:
        return {
            "success": False,
            "error": "not_applicable",
            "reason": "数据集中无可检查的传感器流（IMU 或力/力矩）",
            "user_message": "check_sensor_sanity 需要 IMU 或力/力矩传感器流。当前数据集中没有可检查的传感器流，不适用。",
            "dataset": dataset_id,
        }

    checks: dict[str, Any] = {}
    skipped_checks: dict[str, Any] = {}
    failures: list[str] = []
    warns: list[str] = []

    # ---- IMU 检查单元（含配对与单流；指定 table 时仅该表）----
    imu_units = _imu_check_units(context, imu_streams, table)
    for imu in imu_units:
        key = imu["key"]
        accel_cols = imu["accel_cols"]
        gyro_cols = imu["gyro_cols"]
        data = imu["data"]
        check_table = imu["table_name"]

        if not data or not accel_cols:
            skipped_checks[key] = {"status": "skipped", "reason": "无法读取 IMU 数值列"}
            continue

        # 恒定通道检测前置：先识别恒定列，避免把故障恒定输出误当静止段。
        all_constant = _constant_columns(data, settings.sanity_constant_var)
        const_accel = [c for c in all_constant if c in accel_cols]
        const_gyro = [c for c in all_constant if c in gyro_cols]

        # 非恒定加速度列 → 用于模长与静止段判定。
        usable_accel = [c for c in accel_cols if c in data and c not in const_accel]
        accel_arr = np.column_stack([data[c] for c in usable_accel]) if usable_accel else np.zeros((0, 0))
        # NaN/Inf 比例：行层面（任一加速度列在该行含 NaN/Inf 的行占比），覆盖所有列。
        accel_full = np.column_stack([data[c] for c in accel_cols if c in data])
        if accel_full.size > 0:
            row_bad = np.any(np.isnan(accel_full) | np.isinf(accel_full), axis=1)
            nan_ratio = round(float(np.mean(row_bad)), 4)
        else:
            nan_ratio = 0.0

        # 静止段窗口自适应采样率（measured_rate 优先，回退 config 默认）。
        rate = imu.get("rate")
        if rate is None:
            rate = settings.sanity_static_window_rate
        window = max(2, int(round(float(rate) * 1.0)))
        window_note = f"窗口 {window} 样本（采样率 {rate}Hz）"

        # 静止段判定：仅当有足够的非恒定加速度列（≥2）才可靠；否则降级。
        static_unreliable = len(usable_accel) < 2
        if not static_unreliable and accel_arr.size > 0:
            norms = np.linalg.norm(accel_arr, axis=1)
            static_mask, static_ratio = _stationary_mask(
                norms, window, settings.sanity_static_var_threshold
            )
            static_norms = norms[static_mask & ~np.isnan(norms)]
        else:
            norms = np.full(len(next(iter(data.values()))), np.nan) if data else np.zeros(0)
            static_mask = np.zeros(len(norms), dtype=bool)
            static_ratio = 0.0
            static_norms = np.zeros(0)

        # 恒定加速度列影响静止段判定 → 交叉引用，降级。
        if const_accel:
            warns.append(
                f"{key}: 静止段判定不可信：加速度通道 {', '.join(const_accel)} 疑似故障恒定输出（见恒定通道检查项）"
            )

        # 单位推断（仅用非恒定列的静止段模长）。
        unit, unit_evidence = _infer_accel_unit(static_norms)

        # 加速度模长 vs 重力。
        gravity_check: dict[str, Any] = {"status": "skipped"}
        if const_accel:
            gravity_check = {
                "status": "skipped",
                "reason": f"静止段判定不可信：加速度通道 {', '.join(const_accel)} 疑似故障恒定输出（见恒定通道检查项）",
            }
        elif static_unreliable:
            gravity_check = {"status": "skipped", "reason": "可用非恒定加速度列不足，静止段判定不可信"}
        elif unit != "无法确定" and len(static_norms) >= 3:
            ref = 9.8 if unit == "m/s2" else 1.0
            med_norm = float(np.median(static_norms))
            dev = abs(med_norm - ref) / ref
            tol = settings.sanity_gravity_tolerance
            gravity_check = {
                "status": "done",
                "unit": unit,
                "static_median_norm": round(med_norm, 3),
                "reference": ref,
                "deviation": round(dev, 4),
                "threshold": tol,
            }
            if dev > tol:
                gravity_check["verdict"] = "fail"
                failures.append(f"{key}: 加速度静止模长 {round(med_norm,2)} 偏离重力参考 {ref}（偏差 {round(dev*100,1)}%）")
            else:
                gravity_check["verdict"] = "pass"
        else:
            gravity_check["reason"] = "单位无法确定或静止段样本不足，不硬套重力阈值"

        # 逐轴静止段中位数（加速度计）。
        per_axis_accel = {}
        for c in accel_cols:
            if c in data:
                arr = data[c][static_mask & ~np.isnan(data[c])]
                per_axis_accel[c] = round(float(np.median(arr)), 4) if len(arr) else None

        # 陀螺仪静止偏置（模长 + 逐轴）。
        gyro_check: dict[str, Any] = {"status": "skipped"}
        per_axis_gyro = {}
        if gyro_cols:
            usable_gyro = [c for c in gyro_cols if c in data and c not in const_gyro]
            gdata = np.column_stack([data[c] for c in usable_gyro]) if usable_gyro else np.zeros((0, 0))
            if gdata.size > 0:
                gmags = np.linalg.norm(gdata, axis=1)
                gnan = _nan_inf_ratio(gmags)
                # accel 与 gyro 行数可能不同（真实 accel 52354 / gyro 52347），
                # 当长度不一致时对 gyro 单独判静止段，避免广播越界。
                if len(gmags) == len(static_mask):
                    gstatic = gmags[static_mask & ~np.isnan(gmags)]
                else:
                    gmask, _ = _stationary_mask(
                        gmags, window, settings.sanity_static_var_threshold
                    )
                    gstatic = gmags[gmask & ~np.isnan(gmags)]
                gunit, gunit_evidence = _infer_gyro_unit(gstatic)
                for c in gyro_cols:
                    if c in data:
                        arr = data[c][~np.isnan(data[c])]
                        if len(arr) == len(static_mask):
                            arr = arr[static_mask]
                        else:
                            gmask2, _ = _stationary_mask(
                                arr, window, settings.sanity_static_var_threshold
                            )
                            arr = arr[gmask2]
                        per_axis_gyro[c] = round(float(np.degrees(np.median(arr))), 4) if len(arr) else None
                gyro_check = {
                    "status": "done",
                    "unit": gunit,
                    "static_median_deg": round(np.degrees(np.median(gstatic)) if len(gstatic) else 0, 3),
                    "per_axis_deg": per_axis_gyro,
                    "nan_ratio": gnan,
                    "unit_evidence": gunit_evidence,
                }
                if const_gyro:
                    gyro_check["constant_gyro_channels"] = const_gyro
                    warns.append(f"{key}: 陀螺仪通道 {', '.join(const_gyro)} 疑似故障恒定输出（见恒定通道检查项）")
                if len(gstatic) >= 3 and np.degrees(np.median(gstatic)) > 10.0:
                    gyro_check["verdict"] = "fail"
                    failures.append(f"{key}: 陀螺仪静止偏置过大（{round(np.degrees(np.median(gstatic)),1)}°/s）")
                else:
                    gyro_check["verdict"] = "pass"

        # IMU 通道量程饱和削顶检测（陀螺仪 + 加速度计，快速运动下陀螺仪削顶是高频问题）。
        accel_sat = _saturation_ratio({c: data[c] for c in accel_cols if c in data})
        gyro_sat = _saturation_ratio({c: data[c] for c in gyro_cols if c in data})
        imu_sat = max(accel_sat, gyro_sat)
        saturation_check: dict[str, Any] = {
            "accel_saturation_ratio": accel_sat,
            "gyro_saturation_ratio": gyro_sat,
            "threshold": settings.sanity_saturation_ratio,
        }
        if imu_sat > settings.sanity_saturation_ratio:
            saturation_check["verdict"] = "fail"
            failures.append(f"{key}: IMU 量程饱和削顶比例 {round(imu_sat*100,1)}% 超阈")
        else:
            saturation_check["verdict"] = "pass"

        checks[key] = {
            "type": "imu",
            "table_name": check_table,
            "data_source": imu.get("source"),
            "accel_unit": unit,
            "accel_unit_evidence": unit_evidence,
            "static_ratio": static_ratio,
            "static_window_note": window_note,
            "nan_ratio": nan_ratio,
            "gravity_check": gravity_check,
            "gyro_check": gyro_check,
            "saturation_check": saturation_check,
            "per_axis_accel": per_axis_accel,
        }
        if nan_ratio > settings.sanity_nan_ratio:
            failures.append(f"{key}: NaN/Inf 比例 {nan_ratio} 超阈")
        if not static_unreliable and static_ratio < settings.sanity_static_ratio_warn:
            warns.append(f"{key}: 静止段占比过低（{round(static_ratio*100,1)}%），数据可能全程在运动，重力/零漂检查降级为 warn")

    # ---- 力/力矩流检查 ----
    for fs in force_streams:
        key = fs.get("path") or "force"
        path = fs.get("path", "")
        fmt = fs.get("format", "")
        channels = fs.get("channels", [])
        data = _read_columns(path, fmt, channels) if path else None
        if data is None or not channels:
            skipped_checks[key] = {"status": "skipped", "reason": "无法读取力/力矩数值列"}
            continue
        cols = [c for c in channels if c in data]
        if not cols:
            skipped_checks[key] = {"status": "skipped", "reason": "力/力矩列为空"}
            continue
        arr = np.column_stack([data[c] for c in cols])
        nan_ratio = _nan_inf_ratio(arr)

        # 饱和削顶：连续重复出现的极值点比例。
        sat_ratio = _saturation_ratio({c: data[c] for c in cols})

        force_check = {
            "type": "force",
            "table_name": Path(path).name if path else None,
            "nan_ratio": nan_ratio,
            "saturation_ratio": sat_ratio,
            "threshold": settings.sanity_saturation_ratio,
        }
        if sat_ratio > settings.sanity_saturation_ratio:
            force_check["verdict"] = "fail"
            failures.append(f"{key}: 量程饱和削顶比例 {round(sat_ratio*100,1)}% 超阈")
        else:
            force_check["verdict"] = "pass"
        if nan_ratio > settings.sanity_nan_ratio:
            failures.append(f"{key}: NaN/Inf 比例 {nan_ratio} 超阈")
        checks[key] = force_check

    # ---- 通用：恒定通道检测（汇总，含 force 流）----
    constant_channels: list[str] = []
    for s in streams:
        if s.get("kind") == "video":
            continue
        data = _read_columns(s.get("path", ""), s.get("format", ""), s.get("channels", []))
        if not data:
            continue
        constant_channels.extend(_constant_columns(data, settings.sanity_constant_var))
    # 去重保序。
    constant_channels = list(dict.fromkeys(constant_channels))

    checks["_constant_channels"] = constant_channels
    if constant_channels:
        warns.append(f"检测到恒定通道（疑似掉线）：{', '.join(constant_channels)}")

    # ---- 三档判定 ----
    if failures:
        result = "fail"
    elif warns:
        result = "warn"
    else:
        result = "pass"

    affected = [] if result == "pass" else (
        ["whole_recording"] if not has_episodes else []
    )

    user_message = (
        f"传感器合理性检查判定：{result}。"
        + (f" {episode_note}" if not has_episodes else "")
    )
    if warns:
        user_message += " 存在疑点：" + "；".join(warns)

    # 质检结果写回 meta["qc"]，供 compute_stats / generate_report 读取质检明细。
    qc = context.meta.setdefault("qc", {})
    detail_streams: dict[str, Any] = {}
    for k, c in checks.items():
        if k == "_constant_channels" or c.get("type") not in ("imu", "force"):
            continue
        if c.get("type") == "imu":
            detail_streams[k] = {
                "type": "imu",
                "accel_unit": c.get("accel_unit"),
                "gravity_verdict": (c.get("gravity_check") or {}).get("verdict"),
                "gyro_verdict": (c.get("gyro_check") or {}).get("verdict"),
                "saturation_verdict": (c.get("saturation_check") or {}).get("verdict"),
                "gyro_saturation_ratio": (c.get("saturation_check") or {}).get("gyro_saturation_ratio"),
                "nan_ratio": c.get("nan_ratio"),
            }
        else:  # force
            detail_streams[k] = {
                "type": "force",
                "saturation_ratio": c.get("saturation_ratio"),
                "saturation_verdict": c.get("verdict"),
                "nan_ratio": c.get("nan_ratio"),
            }
    qc["check_sensor_sanity"] = {
        "result": result,
        "constant_channels": constant_channels,
        "dataset": dataset_id,
        "detail": {
            "streams": detail_streams,
            "thresholds": {
                "saturation_ratio": settings.sanity_saturation_ratio,
                "nan_ratio": settings.sanity_nan_ratio,
                "gravity_tolerance": settings.sanity_gravity_tolerance,
            },
        },
    }

    return {
        "success": True,
        "dataset": dataset_id,
        "result": result,
        "episode_note": episode_note if not has_episodes else "存在 episode 划分。",
        "checks": checks,
        "skipped_checks": skipped_checks,
        "constant_channels": constant_channels,
        "affected_episodes": affected,
        "user_message": user_message,
    }


@tool
def check_sensor_sanity(
    wrapper: RunContextWrapper[RunContext],
    sensor: str | None = None,
    table: str | None = None,
) -> dict:
    """检查传感器数据合理性（单位、重力、零漂、饱和、NaN、恒定通道）。

    基于流登记表按需读取数值列，执行单位推断、IMU/力/力矩检查与通用检查。
    IMU 检查优先用 accel+gyro 六轴配对（按配对从流注册表惰性读取对应表，不依赖
    主表身份）；也可显式指定 table 检查某张表。静止段用方差阈值启发式确定；
    判定三档 pass/warn/fail。每个检查项注明作用表名。

    Args:
        sensor: 可选，指定要检查的传感器；省略时检查所有可用传感器。
        table: 可选，指定检查对象表名（如 "accel.csv"）；缺省自动定位/全查。

    Returns:
        统一质检返回格式：result、checks、skipped_checks、dataset、user_message；
        无可检查流时返回 not_applicable。
    """
    return check_sensor_sanity_impl(wrapper.context, table=table)
