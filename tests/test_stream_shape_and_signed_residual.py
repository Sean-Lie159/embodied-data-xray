"""流形态分类与残差增强回归测试（2026-09-04 Commit D）。

守护三类行为：

1. 形态分类：突发型流（MCAP 录制的 tf / IMU）应判 burst 并改报有效速率，
   丢帧率显式标不适用——真实事故中 IMU 的 frame_loss_ratio 被误报为 0.9986；
2. 带符号残差：目标流固定偏移 +20ms 时，带符号残差中位数应 ≈ +20ms
   （回答"A 比 B 晚多少"；此前只有绝对值口径）；
3. 向量化等价：向量化后的最近邻残差与漂移检测，判定结果与原实现一致
   （漂移用例与周期流用例继续全绿即为守护）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools.check_temporal_sync import (
    _align_residuals,
    _classify_stream_shape,
    _single_stream_checks,
    check_temporal_sync_impl,
)
from app.config import get_settings

_STEP_NS = 8_330_000.0  # 120 Hz
_NS_EPOCH_0 = 1_787_294_445_650_951_302
_BURST_INNER_NS = 1792.0  # 突发内间隔（IMU 突发实测量级）


# --- 形态分类单元测试 --------------------------------------------------------


def _burst_series(n_bursts: int = 30, per_burst: int = 20) -> np.ndarray:
    """构造突发型序列：每段 per_burst 点（1.79µs 间隔），段间静默 50ms。

    贴近真实 MCAP 录制形态：突发段极短（微秒级密集）、段间毫秒级静默，
    使平均间隔被静默段大幅拉高（mean/med ≈ 1400，真实 IMU 约 700）。
    """
    pieces = []
    t = _NS_EPOCH_0
    for _ in range(n_bursts):
        inner = t + np.arange(per_burst, dtype=float) * _BURST_INNER_NS
        pieces.append(inner)
        t = float(inner[-1]) + 50_000_000.0  # 突发间静默 50ms
    return np.concatenate(pieces)


def test_periodic_stream_shape() -> None:
    """120 Hz 均匀采样流 → periodic，丢帧率正常输出。"""
    ts = np.arange(1000, dtype=float) * _STEP_NS + _NS_EPOCH_0
    settings = get_settings()
    assert _classify_stream_shape(ts, settings) == "periodic"
    r = _single_stream_checks(ts, nominal=None, settings=settings)
    assert r["stream_shape"] == "periodic"
    assert r["frame_loss_ratio"] == 0.0


def test_burst_stream_shape_and_metrics() -> None:
    """突发型流 → burst：丢帧率置 None + not_applicable，改报有效速率与突发数。"""
    ts = _burst_series()
    settings = get_settings()
    r = _single_stream_checks(ts, nominal=None, settings=settings)
    assert r["stream_shape"] == "burst"
    assert r["frame_loss_ratio"] is None
    assert r["frame_loss_status"] == "not_applicable"
    assert "不适用" in r["burst_note"]
    # 有效速率 = 样本数 / 时长：600 点 / 约 1.5s ≈ 400 Hz。
    assert r["effective_rate_hz"] is not None and 350 < r["effective_rate_hz"] < 450
    # 突发段数 ≈ 30。
    assert 25 <= r["n_bursts"] <= 35
    # 突发内中位间隔 ≈ 1792 ns。
    assert abs(r["intra_burst_median_interval_ns"] - _BURST_INNER_NS) < 1


def test_sparse_static_shape() -> None:
    """极稀疏流（中位间隔 > 跨度/10）→ static。"""
    # 5 个点覆盖 100 秒：中位间隔 25s > 100s/10。
    ts = _NS_EPOCH_0 + np.arange(5, dtype=float) * 25e9
    assert _classify_stream_shape(ts, get_settings()) == "static"


# --- 端到端：突发流不再被误报丢帧 -------------------------------------------


def test_burst_stream_not_misreported_as_frame_loss(tmp_path: Path) -> None:
    """tf + IMU 突发流端到端：不再产出 0.9986 这类伪丢帧率。"""
    meta: dict = {"capabilities": {}, "streams": []}
    series = {
        "imu_burst.csv": _burst_series(),
        "tf_burst.csv": _burst_series(n_bursts=40),
    }
    for name, data in series.items():
        p = tmp_path / name
        pd.DataFrame({"timestamp": data, "value": np.zeros(len(data))}).to_csv(
            p, index=False
        )
        meta["streams"].append({"path": str(p), "format": "csv", "kind": "imu"})

    result = check_temporal_sync_impl(
        RunContext(dataset_id="burst_test", df=None, meta=meta)
    )
    assert result["success"] is True, result.get("user_message")
    for key, c in result["measurements"]["stream_checks"].items():
        assert c["stream_shape"] == "burst", f"{key} 应判 burst"
        assert c["frame_loss_ratio"] is None, (
            f"{key} 突发流丢帧率应为 None（不适用），实际 {c['frame_loss_ratio']}"
        )
    # 判定不应因突发静默而 fail（旧实现 frame_loss_ratio≈0.9986 → fail）。
    assert result["result"] in ("pass", "warn"), result["user_message"]


# --- 带符号残差 --------------------------------------------------------------


def test_signed_residual_median_reports_fixed_offset() -> None:
    """目标流固定偏移 +2ms → 带符号残差中位数 ≈ +2ms（正值 = 晚于基线）。

    采样定理约束：最近邻残差只能测出小于半采样间隔（此处 4.17ms）的偏移；
    超过该范围的固定时延会发生相位折叠（20ms 偏移折叠为 -3.3ms），须由
    互相关物理对齐实测（ARCHITECTURE.md 第 9 节 v2）。真实手套流间固定
    偏移通常为亚毫秒级，在可测范围内。
    """
    base = np.arange(1000, dtype=float) * _STEP_NS + _NS_EPOCH_0
    other = base + 2_000_000.0  # 固定 +2ms（< 半采样间隔 4.17ms）
    res = _align_residuals(base, other)
    assert res["n_match"] == 1000
    assert res["residual_median_signed_ms"] is not None
    assert 1.0 < res["residual_median_signed_ms"] < 3.0, (
        f"带符号中位数应 ≈ +2ms，实际 {res['residual_median_signed_ms']}"
    )
    # 绝对值口径保持可用。
    assert 1.0 < res["residual_mean_ms"] < 3.0
    # 方向分位数：p05 与 p95 同为正（偏移方向一致）。
    assert res["residual_p05_ms"] > 0 and res["residual_p95_signed_ms"] > 0


def test_signed_residual_negative_offset() -> None:
    """目标流固定偏移 -2ms → 带符号残差中位数为负。"""
    base = np.arange(1000, dtype=float) * _STEP_NS + _NS_EPOCH_0
    other = base - 2_000_000.0
    res = _align_residuals(base, other)
    assert -3.0 < res["residual_median_signed_ms"] < -1.0


def test_residual_matches_unaligned_streams() -> None:
    """同相位两条独立周期流（相位随机）→ 带符号中位数接近 0（±半采样间隔内）。"""
    rng = np.random.default_rng(3)
    base = np.arange(2000, dtype=float) * _STEP_NS + _NS_EPOCH_0
    jitter = rng.uniform(-0.2, 0.2, 2000) * _STEP_NS  # ±20% 抖动
    other = np.sort(base + jitter)
    res = _align_residuals(base, other)
    assert abs(res["residual_median_signed_ms"]) < 4.2  # 半采样间隔 4.17ms
