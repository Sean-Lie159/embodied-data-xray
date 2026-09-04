"""对齐基线选择回归测试（2026-09-04 基线策略修复）。

守护：旧策略"帧率最低的流当基线"会把静态流推上基线——真实事故中
tf_static.jsonl（105 点静态变换广播，约 2 Hz，全场最低帧率）被选为基线，
导致残差被放大到 2.43e11 ms 量级。

新策略：先按语义角色排除静态/标定/元数据流与稀疏流，再按
"覆盖完整度 × 间隔稳定性 × 样本规模"打分推荐，并输出可转述的推荐理由。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools.check_temporal_sync import (
    _stream_baseline_exclude_reason,
    check_temporal_sync_impl,
)

_STEP_NS = 8_330_000.0  # 120 Hz
_NS_EPOCH_0 = 1_787_294_445_650_951_302


def _periodic_frame(n: int = 6339) -> pd.DataFrame:
    """周期型流：120 Hz 均匀采样约 52.8 s（与真实 wujiGlove 场景同量级）。"""
    ts = np.arange(n, dtype=float) * _STEP_NS + _NS_EPOCH_0
    return pd.DataFrame({"timestamp": ts, "value": np.arange(n, dtype=float)})


def _static_tf_frame(n: int = 105) -> pd.DataFrame:
    """静态变换广播形态：稀疏低频（约 2 Hz）、点数少，模拟 ROS tf_static。"""
    ts = np.arange(n, dtype=float) * 5e8 + _NS_EPOCH_0
    return pd.DataFrame({"timestamp": ts, "value": np.zeros(n)})


def _make_ctx(tmp_path: Path, files: dict[str, pd.DataFrame]) -> RunContext:
    """按 {文件名: DataFrame} 构造数据集上下文。"""
    meta: dict = {"capabilities": {}, "streams": []}
    for name, df in files.items():
        p = tmp_path / name
        df.to_csv(p, index=False)
        meta["streams"].append({
            "path": str(p), "format": "csv",
            "kind": "imu" if "imu" in name else "unknown",
        })
    return RunContext(dataset_id="baseline_test", df=None, meta=meta)


# --- 排除判据单元测试 --------------------------------------------------------


def test_static_name_excluded() -> None:
    """tf_static 命名的流应被排除（静态变换广播不随时间变化）。"""
    reason = _stream_baseline_exclude_reason(
        "C:/data/tf_static.jsonl", {}, 5000, min_samples=100,
    )
    assert reason is not None and "tf_static" in reason


def test_calibration_kind_excluded() -> None:
    """kind=calibration 的流应被排除。"""
    reason = _stream_baseline_exclude_reason(
        "C:/data/whatever.csv", {"kind": "calibration"}, 5000, min_samples=100,
    )
    assert reason is not None and "标定" in reason


def test_few_samples_excluded() -> None:
    """样本数低于阈值（疑似静态/稀疏流）应被排除。"""
    reason = _stream_baseline_exclude_reason(
        "C:/data/sensor.csv", {}, 99, min_samples=100,
    )
    assert reason is not None and "样本数" in reason


def test_normal_periodic_stream_not_excluded() -> None:
    """正常周期型流不应被排除。"""
    assert _stream_baseline_exclude_reason(
        "C:/data/left_glove_imu.csv", {"kind": "imu"}, 42048, min_samples=100,
    ) is None


# --- 端到端：tf_static 不得胜出 ---------------------------------------------


def test_static_tf_stream_not_chosen_as_baseline(tmp_path: Path) -> None:
    """tf_static（帧率全场最低）+ 两条周期流 → 基线必须是周期流。

    旧策略按"帧率最低"选基线，本场景必然选中 tf_static（2 Hz < 120 Hz）；
    新策略必须排除它。
    """
    ctx = _make_ctx(tmp_path, {
        "tf_static.csv": _static_tf_frame(),       # 105 点，约 2 Hz
        "left_imu.csv": _periodic_frame(),         # 120 Hz
        "right_imu.csv": _periodic_frame(),        # 120 Hz
    })
    result = check_temporal_sync_impl(ctx)

    assert result["success"] is True
    baseline = result["baseline_stream"]
    assert baseline is not None
    assert "tf_static" not in Path(baseline).name, (
        f"静态流不得作为基线，实选 {baseline}"
    )
    # 推荐说明应包含排除记录与理由。
    rec = result["baseline_recommendation"]
    assert rec["stream"] == baseline
    assert "覆盖" in rec["reason"] and "稳定" in rec["reason"]
    excluded_names = [e["stream"] for e in rec["excluded"]]
    assert "tf_static.csv" in excluded_names, (
        f"tf_static 应出现在排除清单中，实际 {excluded_names}"
    )
    assert any("静态" in e["reason"] or "tf_static" in e["reason"]
               for e in rec["excluded"])


def test_baseline_prefers_full_coverage(tmp_path: Path) -> None:
    """覆盖率不足（录制中段才开始）的流不得作为基线。"""
    # 一条流从中段才开始（覆盖后半段，覆盖率约 60%）。
    n_total = 6339
    start = 2400
    ts_half = (
        np.arange(n_total - start, dtype=float) * _STEP_NS
        + _NS_EPOCH_0 + start * _STEP_NS
    )
    ctx = _make_ctx(tmp_path, {
        "half_imu.csv": pd.DataFrame(
            {"timestamp": ts_half,
             "value": np.arange(n_total - start, dtype=float)}),
        "full_imu.csv": _periodic_frame(),
    })
    result = check_temporal_sync_impl(ctx)
    assert result["success"] is True
    assert "half_imu" not in Path(result["baseline_stream"]).name
    excluded_names = [e["stream"] for e in result["baseline_recommendation"]["excluded"]]
    assert "half_imu.csv" in excluded_names
