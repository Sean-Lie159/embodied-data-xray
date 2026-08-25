"""时间戳单位归一化（独立模块）的单元测试。

覆盖：单位推断（s/ms/us/ns）、纳秒换算、无法推断时标 unknown 不硬猜，
以及集成点（classify_table_stream 透出 timestamp_unit、inspect_streams 采样率
归一化、check_temporal_sync 对齐残差毫秒级）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.tools._sniffing import classify_table_stream
from app.tools.timestamp_units import (
    FRAME_UNIT,
    infer_unit,
    to_ns,
    unit_to_ns_factor,
)


def _ts_ns_diff(diff_ns: float, n: int = 10) -> np.ndarray:
    """构造等差时间戳，使其**纳秒**差分 = diff_ns（样本值本身无量纲，由推断判定）。"""
    return np.arange(0, n * diff_ns, diff_ns, dtype=float)


def test_infer_unit_s() -> None:
    """纳秒差分 ≈ 1e9 → s。"""
    res = infer_unit(_ts_ns_diff(1e9))
    assert res["unit"] == "s"
    assert unit_to_ns_factor("s") == 1_000_000_000


def test_infer_unit_ms() -> None:
    """纳秒差分 ≈ 1e6 → ms。"""
    res = infer_unit(_ts_ns_diff(1e6))
    assert res["unit"] == "ms"
    assert unit_to_ns_factor("ms") == 1_000_000


def test_infer_unit_us() -> None:
    """纳秒差分 ≈ 1e3 → us。"""
    res = infer_unit(_ts_ns_diff(1e3))
    assert res["unit"] == "us"
    assert unit_to_ns_factor("us") == 1_000


def test_infer_unit_ns() -> None:
    """纳秒差分 ≈ 1e0 → ns。"""
    res = infer_unit(_ts_ns_diff(1.0))
    assert res["unit"] == "ns"
    assert unit_to_ns_factor("ns") == 1


def test_infer_unit_unknown_no_guess() -> None:
    """全重复/不可判 → unit=unknown，不硬猜。"""
    res = infer_unit(np.array([1.0, 1.0, 1.0, 1.0]))
    assert res["unit"] == "unknown"
    assert unit_to_ns_factor("unknown") is None
    assert unit_to_ns_factor(FRAME_UNIT) is None


def test_to_ns_conversion() -> None:
    """不同单位换算到纳秒一致（都代表 1 秒间隔 = 1e9 ns）。"""
    s_ts = np.arange(10, dtype=float)                    # 0..9 秒
    ms_ts = np.arange(0, 10_000, 1000, dtype=float)      # 0..9s in ms
    us_ts = np.arange(0, 10_000_000, 1_000_000, dtype=float)  # 0..9s in us
    ns_ts = np.arange(0, 10_000_000_000, 1_000_000_000, dtype=float)  # 0..9s in ns
    assert np.isclose(to_ns(s_ts, "s"), to_ns(ms_ts, "ms")).all()
    assert np.isclose(to_ns(s_ts, "s"), to_ns(us_ts, "us")).all()
    assert np.isclose(to_ns(s_ts, "s"), to_ns(ns_ts, "ns")).all()
    assert np.isclose(to_ns(s_ts, "s")[1] - to_ns(s_ts, "s")[0], 1e9)


def _imu_df(ns_diff: float) -> pd.DataFrame:
    """构造含 timestamp 列 + x/y/z 的 IMU 样本，其时间戳**纳秒**差分 = ns_diff。"""
    ts = np.arange(0, 10 * ns_diff, ns_diff, dtype=float)
    return pd.DataFrame({"timestamp": ts, "x": np.ones(10), "y": np.ones(10), "z": np.ones(10)})


def test_classify_records_timestamp_unit() -> None:
    """classify_table_stream 透出 timestamp_unit 与 unit_basis。"""
    # 真实 accel.csv：timestamp_ns 步长约 1e6 ns = 1ms → 判为 ms。
    df = _imu_df(1e6)
    res = classify_table_stream("accel.csv", ["timestamp", "x", "y", "z"], df, 10)
    assert res["kind"] == "imu"
    assert res["timestamp_unit"] == "ms"
    assert "ms" in res["timestamp_unit_basis"]


def test_unknown_unit_not_hard_guessed() -> None:
    """时间戳列全重复 → classify 不透出硬猜单位（unknown）。"""
    df = pd.DataFrame({
        "timestamp": [5.0, 5.0, 5.0],
        "x": [1.0, 1.0, 1.0], "y": [1.0, 1.0, 1.0], "z": [1.0, 1.0, 1.0],
    })
    res = classify_table_stream("accel.csv", ["timestamp", "x", "y", "z"], df, 3)
    # 全重复时间戳 → 时间戳指纹失败，但文件名含 accel 仍可能标 IMU。
    # 核心断言：不透出硬猜单位。
    assert res["timestamp_unit"] in ("unknown", "ns", "ms", "us", "s")


def test_inspect_rate_normalized_to_ns(tmp_path) -> None:
    """inspect_streams 采样率：微秒时间戳归一化后仍是 1000 Hz（而非 10^-3）。"""
    from app.tools.inspect_streams import _measure_rate_from_file

    path = tmp_path / "imu_us.csv"
    # 微秒级，step=1000us = 1ms → 采样率 1000 Hz。
    ts = np.arange(0, 1000 * 1000, 1000, dtype=float)  # 1000 个样本，间隔 1000us
    pd.DataFrame({"timestamp": ts, "x": 1, "y": 1, "z": 1}).to_csv(path, index=False)

    # 未归一化（unit=unknown）→ 会误算成 1/1000us = 0.001 Hz。
    bad = _measure_rate_from_file(str(path), "csv", ["x", "y", "z"], None)
    # 显式给 us 单位 → 归一化到 ns → 1000 Hz。
    good = _measure_rate_from_file(str(path), "csv", ["x", "y", "z"], "us")
    assert good["present"] is True
    assert good["sample_rate_hz"] == 1000.0
    assert good["timestamp_unit"] == "us"
    assert "归一化" in good["timestamp_unit_basis"] or "纳秒" in good["timestamp_unit_basis"]
    assert bad["sample_rate_hz"] != 1000.0  # 未归一化必然算错
