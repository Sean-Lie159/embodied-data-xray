"""时间戳单位推断与纳秒归一化（独立模块）。

背景：采集数据里同一时间列可能是秒 / 毫秒 / 微秒 / 纳秒 / 帧序号，混在一起会导致
采样率被算成 10^-9、时间对齐残差放大到 10^21 等失真。本模块负责：

1. 按时间戳**差分**与**绝对值**的量级推断单位（s / ms / us / ns / frame_index）；
2. 提供统一换算到纳秒基准的函数，供采样率 / 时长 / 对齐残差计算前调用；
3. 推断依据必须透出（如 ``unit="ns"``、``unit_basis="中位差分≈1e6"``）；
   无法推断时标 ``"unknown"``，**绝不硬猜**。

本模块为纯 Python（不 import streamlit），且不依赖 sniffing 内部，可独立复用。
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 单位 → 到纳秒的换算倍数（1 单位 = N 纳秒）。
_UNIT_TO_NS: dict[str, int] = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}
# 全部合法单位（含 frame_index：帧序号非时间，不参与纳秒换算）。
_UNITS = ("s", "ms", "us", "ns")
_FRAME_UNIT = "frame_index"

# 单位识别量级区间（以中位差分在"纳秒"下的数量级为准）。判定阈值取相邻单位的
# 几何中点（如 ns 与 us 的边界在 1e3 ns），给判定留天然裕度：
#   ns: 差分 < 1e3 ns；us: [1e3, 1e6) ns；ms: [1e6, 1e9) ns；s: >= 1e9 ns。
_MED_DIFF_NS_RANGES: list[tuple[float, float, str]] = [
    # (下限, 上限, 单位) —— 中位差分（纳秒）落在 [下限, 上限) 内即判为该单位。
    # us/ms 边界取 1e5 ns（0.1ms）：典型微秒级采样间隔 < 0.1ms，毫秒级（如 1kHz
    # IMU，周期 1ms≈1e6ns，含抖动后~9.87e5ns）≥ 0.1ms → 归 ms。
    (0.0, 1e3, "ns"),      # 纳秒：差分 < 1e3 ns
    (1e3, 1e5, "us"),      # 微秒：差分 1e3~1e5 ns
    (1e5, 1e9, "ms"),      # 毫秒：差分 1e5~1e9 ns
    (1e9, float("inf"), "s"),  # 秒：差分 >= 1e9 ns
]


def _median_positive_diff(arr: np.ndarray) -> float | None:
    """排序后相邻正差分的中位数（丢弃非正与重复，代表真实采样间隔）。

    Args:
        arr: 时间戳数组（数值型）。

    Returns:
        中位正差分；样本不足或无正差分返回 None。
    """
    if arr is None or len(arr) < 2:
        return None
    s = np.sort(np.asarray(arr, dtype=float))
    d = np.diff(s)
    d = d[d > 0]
    if len(d) == 0:
        return None
    return float(np.median(d))


def _unit_from_name(col_name: str) -> str | None:
    """从列名后缀推断单位（真实数据常用 timestamp_ns / pts_us 等命名）。

    Args:
        col_name: 时间戳列名。

    Returns:
        单位（s/ms/us/ns/frame_index）；无法从命名判断返回 None。
    """
    lower = (col_name or "").lower().strip()
    # 帧序号类命名（PTS / frame / packet / index）优先：即使带 us 后缀（如 pts_us），
    # 也是帧序号而非真实时间。
    if any(k in lower for k in ("pts", "packet")):
        return FRAME_UNIT
    if "frame_index" in lower or lower.endswith("_index") or lower.endswith("_idx"):
        return FRAME_UNIT
    # 显式单位后缀（去掉下划线/数字干扰，优先匹配最长后缀）。
    if lower.endswith("_ns") or lower.endswith("ns"):
        return "ns"
    if lower.endswith("_us") or lower.endswith("us"):
        return "us"
    if lower.endswith("_ms") or lower.endswith("ms"):
        return "ms"
    if lower.endswith("_s") or lower.endswith("s") and not lower.endswith("index"):
        return "s"
    return None


def infer_unit(
    ts: np.ndarray | list[float], col_name: str = ""
) -> dict[str, Any]:
    """推断时间戳单位（s / ms / us / ns / frame_index）。

    策略：优先用**列名单位后缀**（timestamp_ns→ns、pts_us→frame_index 等，真实数据
    命名即编码单位，可避免差分量级的歧义）；列名无明确单位时，再按**中位差分**量级
    推断；差分无法判断时回退看**绝对值量级**。都无法归属则标 ``unit="unknown"``
    并给出原因，不硬猜。

    Args:
        ts: 时间戳数组（数值型；可为 list）。
        col_name: 时间戳列名（用于按命名后缀推断单位）。

    Returns:
        dict，含 unit、unit_basis（推断依据）与内部中间量（med_diff_ns、med_diff）。
    """
    arr = np.asarray(ts, dtype=float)
    arr = arr[~np.isnan(arr)]

    # ① 列名单位后缀优先（最可靠，无差分歧义）。
    name_unit = _unit_from_name(col_name)
    if name_unit in ("s", "ms", "us", "ns", FRAME_UNIT):
        med = _median_positive_diff(arr)
        return {
            "unit": name_unit,
            "unit_basis": f"列名 {col_name} 后缀标明单位 {name_unit}",
            "med_diff": med,
            "med_diff_ns": med,
        }

    if len(arr) < 2:
        return {
            "unit": "unknown",
            "unit_basis": f"时间戳样本不足（{len(arr)}），无法推断单位",
            "med_diff": None,
            "med_diff_ns": None,
        }

    med = _median_positive_diff(arr)
    abs_med = float(np.median(np.abs(arr)))

    # ① 绝对量级 epoch 判定优先：值 ~1e15~1e19 且中位差分在 ~1e5~1e9 之间 → 纳秒
    # 时间戳（如 2026 年的 UTC 纳秒 epoch ≈1.7e18，即使差分 ~33ms 也是 ns 列）。
    if 1e15 <= abs_med <= 1e19 and med is not None and 1e4 <= med <= 1e9:
        return {
            "unit": "ns",
            "unit_basis": f"绝对值中位≈{abs_med:.3g}（纳秒 epoch 量级），列应为纳秒时间戳",
            "med_diff": med,
            "med_diff_ns": med,
        }

    # ② 差分判定：中位正差分落在某单位量级区间。
    if med is not None and med > 0:
        for lo, hi, unit in _MED_DIFF_NS_RANGES:
            if lo <= med < hi:
                return {
                    "unit": unit,
                    "unit_basis": f"中位差分≈{med:.3g}，落于{unit}典型量级区间",
                    "med_diff": med,
                    "med_diff_ns": med,
                }

    # ② 差分无法判定（过小/全重复）→ 回退看绝对值量级（常见于起点即时刻戳）。
    if med is None or med <= 0:
        abs_med = float(np.median(np.abs(arr)))
        if abs_med >= 1e12:
            return {"unit": "s", "unit_basis": f"时间戳绝对值中位≈{abs_med:.3g}，远超纳秒量级（疑似秒）",
                    "med_diff": med, "med_diff_ns": None}
        if 1e9 <= abs_med < 1e12:
            return {"unit": "ms", "unit_basis": f"时间戳绝对值中位≈{abs_med:.3g}（毫秒量级）",
                    "med_diff": med, "med_diff_ns": None}
        if 1e6 <= abs_med < 1e9:
            return {"unit": "us", "unit_basis": f"时间戳绝对值中位≈{abs_med:.3g}（微秒量级）",
                    "med_diff": med, "med_diff_ns": None}
        if 1e3 <= abs_med < 1e6:
            return {"unit": "us", "unit_basis": f"时间戳绝对值中位≈{abs_med:.3g}（千→百万量级）",
                    "med_diff": med, "med_diff_ns": None}

    # ③ 差分既非正整数又绝对值太小时，无法判断 → unknown。
    return {
        "unit": "unknown",
        "unit_basis": "差分与绝对值量级均无法归属到 s/ms/us/ns（疑似帧序号或全重复/缺失）",
        "med_diff": med,
        "med_diff_ns": None,
    }


def to_ns(ts: np.ndarray, unit: str) -> np.ndarray:
    """把给定单位的原始时间戳换算为纳秒。

    Args:
        ts: 原始时间戳数组。
        unit: 单位（s / ms / us / ns / frame_index）。

    Returns:
        纳秒时间戳数组。unit 为 frame_index 或 unknown 时，直接返回原值
        （帧序号无物理时间，由调用方决定不参与换算）。
    """
    factor = _UNIT_TO_NS.get(unit)
    if factor is None:
        return np.asarray(ts, dtype=float)
    return np.asarray(ts, dtype=float) * factor


def unit_to_ns_factor(unit: str) -> int | None:
    """返回某单位到纳秒的换算倍数；非时间单位（frame_index/unknown）返回 None。

    Args:
        unit: 单位名。

    Returns:
        int 换算倍数；无法换算返回 None。
    """
    return _UNIT_TO_NS.get(unit)


# 供其它模块引用的单位常量。
TIME_UNITS = _UNITS
FRAME_UNIT = _FRAME_UNIT
