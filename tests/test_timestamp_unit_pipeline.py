"""时间戳单位链路回归测试（2026-09-04 单位链路修复）。

守护四类此前真实发生过的失效：

1. 列名带单位后缀但**不在词表内**（如 MCAP 导出的 ``mcap_log_time_ns``）→ 此前
   ``fingerprint_timestamp`` 只查白名单，单位落到 unknown，进而被"按秒兜底"换算，
   产出 duration_s≈5.28e10、残差 2.43e11 等伪值；
2. 流登记表判为 unknown 时，应能用**实际时间戳列名**交叉校验纠正；
3. 单位确实未知时，应显式置为不可用（normalized=False、绝对量 None），
   **绝不按秒兜底产出伪值**；
4. ``inspect_streams`` 与 ``check_temporal_sync`` 的兜底口径必须一致。

回归背景见 ``docs/时间对齐能力改造设计.md`` 第 3 节。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools.check_temporal_sync import (
    _normalize_to_ns,
    _single_stream_checks,
    check_temporal_sync_impl,
)
from app.tools.inspect_streams import _measure_rate_from_file
from app.tools.timestamp_units import self_correct_unit

# MCAP 导出的真实量级：2026 年 UTC 纳秒 epoch ≈ 1.78e18。
_NS_EPOCH_0 = 1_787_294_445_650_951_302
# 手套低频流：120 Hz → 间隔 8.33 ms = 8.33e6 ns。
_STEP_NS = 8_330_000.0


def _mcap_frame(n: int = 6339, step: float = _STEP_NS) -> pd.DataFrame:
    """构造 MCAP 导出形态的双时间列表（log_time / publish_time，纳秒）。"""
    ts = np.arange(n, dtype=float) * step + _NS_EPOCH_0
    return pd.DataFrame({
        "mcap_log_time_ns": ts,
        "mcap_publish_time_ns": ts,
        "sensor_value": np.arange(n, dtype=float),
    })


# --- 1. 词表外列名（带 _ns 后缀）经指纹回退识别 ----------------------------


def test_mcap_ns_column_recognized_via_fingerprint() -> None:
    """mcap_log_time_ns 不在 _TIMESTAMP_COLS 内，应经内容指纹回退识别为 ns。"""
    from app.tools._sniffing import _TIMESTAMP_COLS, fingerprint_timestamp

    # 前置条件：该列名确实不在词表内（否则本用例失去意义）。
    assert "mcap_log_time_ns" not in _TIMESTAMP_COLS

    df = _mcap_frame()
    res = fingerprint_timestamp(df, list(df.columns))
    assert res["present"] is True, res["evidence"]
    assert res["column"] == "mcap_log_time_ns"
    assert res["timestamp_unit"] == "ns", res["timestamp_unit_basis"]
    # 识别来源应标明是回退命中，便于排查。
    assert "内容指纹回退" in res["evidence"]


def test_end_to_end_ns_duration_is_physical() -> None:
    """端到端：登记表单位判为 unknown 时，经列名交叉校验得 ns，时长物理正确。

    修复前：单位 unknown → 按秒兜底 → duration_s≈5.28e10（字段名标秒，实为纳秒）。
    修复后：duration_ns≈5.28e10 纳秒（52.8 秒），采样率≈120 Hz。
    """
    root = Path(__file__).parent / "_tmp_mcap"
    root.mkdir(exist_ok=True)
    try:
        paths: list[str] = []
        for i in range(2):
            p = root / f"glove_{i}.csv"
            _mcap_frame().to_csv(p, index=False)
            paths.append(str(p))

        meta = {
            "capabilities": {},
            # 显式把单位写成 unknown，模拟修复前流登记表的漏判。
            "streams": [
                {"path": paths[0], "format": "csv", "kind": "unknown",
                 "timestamp_unit": "unknown"},
                {"path": paths[1], "format": "csv", "kind": "unknown",
                 "timestamp_unit": "unknown"},
            ],
        }
        ctx = RunContext(dataset_id="wujiGlove_data", df=None, meta=meta)
        result = check_temporal_sync_impl(ctx)

        assert result["success"] is True, result.get("user_message")
        checks = result["measurements"]["stream_checks"]
        assert len(checks) == 2

        for key, c in checks.items():
            assert c["present"] is True
            # 关键：单位经交叉校验修正为 ns，不再停留在 unknown。
            assert c["timestamp_unit"] == "ns", (
                f"{key} 单位应为 ns，实为 {c.get('timestamp_unit')}"
            )
            # 时长约 52.8 秒（纳秒口径），而非被误读为"528 亿秒"。
            assert c["duration_ns"] is not None
            duration_s = c["duration_ns"] / 1e9
            assert 52.0 < duration_s < 53.5, f"时长应约 52.8s，实为 {duration_s}s"
            # 采样率约 120 Hz。
            assert c["actual_rate_hz"] is not None
            assert 118.0 < c["actual_rate_hz"] < 122.0, (
                f"采样率应约 120Hz，实为 {c['actual_rate_hz']}"
            )
            # 无缺口、无乱序、无重复。
            assert c["gap_count"] == 0
            assert c["disorder_count"] == 0

        # 单位全部可用 → 无 unit_warnings。
        assert result["unit_warnings"] == []
    finally:
        for f in root.glob("*.csv"):
            f.unlink()
        root.rmdir()


# --- 2. 单位未知时显式不可用，绝不按秒兜底 --------------------------------


def test_unknown_unit_not_normalized() -> None:
    """单位未知：normalized=False 且数值保持原值（此前会 ×1e9 造成放大 1e9 倍）。"""
    ts = np.array([1.0, 2.0, 3.0])
    out, info = _normalize_to_ns(ts, "unknown")
    assert info["normalized"] is False
    assert np.allclose(out, ts), "未知单位不得做兜底换算"
    assert "未归一化" in info["basis"]


def test_unknown_unit_absolute_measures_are_none() -> None:
    """单位未知：时长/间隔/采样率置 None，但单位无关的丢帧率仍可计算。"""
    # 构造一段含真实缺口的时间序列（单位未知，按原值口径）。
    ts = np.arange(0, 100, dtype=float)
    ts = np.delete(ts, np.arange(40, 60))  # 挖掉 20 个样本
    r = _single_stream_checks(ts, nominal=None, unit_known=False)

    # 绝对量不可用。
    assert r["duration_ns"] is None
    assert r["median_interval_ns"] is None
    assert r["actual_rate_hz"] is None
    assert r["absolute_note"] is not None
    # 单位无关项仍有效：丢帧率是比值，与单位无关。
    assert r["frame_loss_ratio"] > 0.0
    assert r["gap_count"] >= 1


def test_known_unit_absolute_measures_present() -> None:
    """对照：单位已知（纳秒）时绝对量正常计算。"""
    ts = np.arange(0, 100, dtype=float) * 1_000_000.0  # 1ms 间隔 → 1000 Hz
    r = _single_stream_checks(ts, nominal=None, unit_known=True)
    assert r["duration_ns"] == 99_000_000.0
    assert r["median_interval_ns"] == 1_000_000.0
    assert r["actual_rate_hz"] == 1000.0
    assert r["absolute_note"] is None


# --- 3. 自我纠正覆盖 unknown，且两工具口径一致 ----------------------------


def test_self_correct_unit_accepts_unknown() -> None:
    """unknown 作为初始单位时，应经数值量级推断纠正，而非直接放弃。"""
    # 微秒时间戳：间隔 1000us → 1000 Hz。
    ts = np.arange(0, 1000 * 1000, 1000, dtype=float)
    res = self_correct_unit(ts, "unknown")
    assert res["corrected"] is True
    assert res["unit"] == "us"
    assert res["sample_rate_hz"] == 1000.0


def test_self_correct_unit_rejects_frame_index() -> None:
    """帧序号不得被纠正成时间单位（它是整数计数序列，无量级语义）。"""
    ts = np.arange(0, 1000, dtype=float)  # 0,1,2,...
    res = self_correct_unit(ts, "frame_index")
    assert res["corrected"] is False
    assert res["unit"] == "frame_index"
    assert res["sample_rate_hz"] is None


def test_inspect_rate_unknown_no_fake_value(tmp_path) -> None:
    """inspect_streams：单位无法推断时不产出伪值（与 check_temporal_sync 口径一致）。"""
    # 全相同时间戳 → 差分无效 → 无法推断单位，也无法计算采样率。
    path = tmp_path / "flat.csv"
    pd.DataFrame({"timestamp": [5.0] * 20, "x": 1.0}).to_csv(path, index=False)
    res = _measure_rate_from_file(str(path), "csv", ["x"], None)
    # 差分全零 → 间隔非正 → 明确报不可用，而非给出 0 或 inf。
    assert res["present"] is False
    assert "间隔" in res["reason"]
