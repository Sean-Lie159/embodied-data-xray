"""三个统计口径修复的单元测试。

1. 丢帧率：抖动数据不虚报（旧口径虚报 2.28%），真实缺口正确检出；
2. 采样率口径统一：inspect_streams 与 check_temporal_sync 对同一文件一致；
3. 四元数范数分层：剔除全零行后有效行范数=1（质量与缺失两信号不混淆）。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools.check_temporal_sync import (
    _single_stream_checks,
    check_temporal_sync_impl,
)
from app.tools.compute_stats import _skeleton_pose_range, compute_stats_impl
from app.tools.inspect_streams import inspect_streams_impl
from app.tools.load_dataset import load_dataset_impl


def _vec(vals: list[float]) -> str:
    return " ".join(f"{v}" for v in vals)


# ---- 修复 1：丢帧率对抖动鲁棒 ----


def test_frame_loss_jitter_not_reported() -> None:
    """抖动数据（间隔 38~44ms 波动、无缺口）→ 丢帧率应为 0（旧口径虚报 2.28%）。"""
    rng = np.random.default_rng(42)
    n = 747
    # 25Hz 基础间隔 40ms，叠加 ±3ms 抖动（保持单调）。
    ts = np.cumsum(0.040 + rng.uniform(-0.003, 0.003, n))
    r = _single_stream_checks(ts, nominal=None)
    assert r["frame_loss_ratio"] == 0.0, (
        f"抖动不应被虚报为丢帧（旧口径会得 2.28%），实际 {r['frame_loss_ratio']}"
    )
    assert r["gap_count"] == 0


def test_frame_loss_real_gap_detected() -> None:
    """真实缺口（整段缺失）→ 丢帧率正确反映缺口比例。"""
    n = 747
    ts = np.cumsum(np.full(n, 0.040))
    # 删掉 100 帧（≈13% 缺口）。
    ts_gap = np.delete(ts, np.arange(300, 400))
    r = _single_stream_checks(ts_gap, nominal=None)
    assert r["frame_loss_ratio"] > 0.10, (
        f"真实缺口应被检出（≈13%），实际 {r['frame_loss_ratio']}"
    )
    assert r["gap_count"] >= 1


def test_frame_loss_no_gap_zero() -> None:
    """完美均匀（无抖动无缺口）→ 丢帧率 0。"""
    ts = np.cumsum(np.full(100, 0.04))
    r = _single_stream_checks(ts, nominal=None)
    assert r["frame_loss_ratio"] == 0.0


def test_actual_rate_uses_mean_not_median() -> None:
    """采样率用平均差分（守恒口径）：均值采样率 × 时长 ≈ 样本数。"""
    rng = np.random.default_rng(7)
    n = 500
    ts = np.cumsum(0.040 + rng.uniform(-0.003, 0.003, n))
    r = _single_stream_checks(ts, nominal=None)
    duration = r["duration_s"]
    assert r["actual_rate_hz"] is not None and duration
    # 守恒：rate × duration ≈ n（偏差 <2%）。
    assert abs(r["actual_rate_hz"] * duration - n) / n < 0.02


# ---- 修复 2：采样率口径统一（inspect_streams 与 check_temporal_sync 一致）----


def test_rate_consistent_across_tools(tmp_path: Path) -> None:
    """同一文件：inspect_streams 与 check_temporal_sync 的采样率应一致（<2% 差）。"""
    root = tmp_path / "ds"
    root.mkdir()
    rng = np.random.default_rng(11)
    n = 300
    ts = np.cumsum(0.040 + rng.uniform(-0.002, 0.002, n))
    pd.DataFrame({
        "timestamp": ts,
        "accel_x": [1.0] * n, "accel_y": [1.0] * n, "accel_z": [9.8] * n,
        "gyro_x": [0.0] * n, "gyro_y": [0.0] * n, "gyro_z": [0.0] * n,
    }).to_csv(root / "imu.csv", index=False)
    # 第二路表（位姿）以满足 ≥2 可对齐流。
    pd.DataFrame({
        "timestamp": ts,
        "pos_x": [1.0] * n, "pos_y": [1.0] * n, "pos_z": [1.0] * n,
        "quat_w": [1.0] * n, "quat_x": [0.0] * n, "quat_y": [0.0] * n, "quat_z": [0.0] * n,
    }).to_csv(root / "pose.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))

    from app.tools.inspect_streams import inspect_streams_impl

    insp = inspect_streams_impl(ctx)
    sync = check_temporal_sync_impl(ctx)
    insp_rates = [
        s["sample_rate"]["sample_rate_hz"]
        for imu in insp["imus"] for s in imu["streams"]
        if s["sample_rate"].get("present")
    ]
    sync_rates = [
        c.get("actual_rate_hz")
        for c in sync["measurements"]["stream_checks"].values()
        if c.get("present") and c.get("actual_rate_hz")
    ]
    assert insp_rates and sync_rates
    for a in insp_rates:
        for b in sync_rates:
            assert abs(a - b) / max(a, b) < 0.02, (
                f"两工具采样率不一致：inspect={a}, sync={b}（应为同一口径）"
            )


# ---- 修复 3：四元数范数分层（剔除全零行）----


def _vec(vals: list[float]) -> str:
    return " ".join(f"{v}" for v in vals)


def test_quaternion_norm_split_zero_and_valid(tmp_path: Path) -> None:
    """含全零行的骨架列：全零行从质量统计剔除，有效行范数=1（不被零行拉低）。"""
    root = tmp_path / "lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 30,
        "features": {
            "observation.head_pose": {
                "dtype": "float32", "shape": [7],
                "names": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
            },
            "observation.left_hand": {
                "dtype": "float32", "shape": [182], "names": ["left_hand_26x7"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }), encoding="utf-8")
    n = 40

    def hand_vec(i: int) -> str:
        """26 块 × 7DoF（每块 3 位置 + 单位四元数），共 182 维，范数=1。"""
        vals: list[float] = []
        for b in range(26):
            vals.extend([0.1 * i + b * 0.01, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        return _vec(vals)

    rows = []
    for i in range(n):
        rows.append({
            "timestamp": i * 0.04,
            "observation.head_pose": _vec([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            # 前 10 行为全零（缺失占位），其余为有效单位四元数。
            "observation.left_hand": (
                _vec([0.0] * 182) if i < 10 else hand_vec(i)
            ),
        })
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = compute_stats_impl(ctx)
    sk = (r.get("metrics") or {}).get("skeleton_pose_range") or {}
    b0 = sk["observation.left_hand"]["blocks"]["block_0"]
    # 全部行平均被零行拉低（0.96 左右），但有效行范数应精确 =1.0。
    assert b0["quaternion_norm_mean"] < 0.99  # 混入零行导致均值 <1
    assert b0["quaternion_norm_mean_valid"] == 1.0, (
        f"剔除全零行后有效行范数应=1.0，实际 {b0['quaternion_norm_mean_valid']}"
    )
    assert b0["quaternion_norm_stable"] is True
    # 全零行说明透出（缺失与质量两信号分离）。
    assert "全零行" in (b0.get("zero_row_note") or "")
