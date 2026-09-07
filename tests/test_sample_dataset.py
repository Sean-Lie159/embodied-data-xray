"""合成示例数据集测试（2026-09-07 Commit 4）。

守护：
- 确定性：固定种子，两次生成逐值一致（"示例永远可复现"的承诺）；
- 缺口存在：IMU 流中段有约 207 帧缺口（演示 locate_gaps 的素材）；
- 端到端：示例数据集经 check_temporal_sync（locate_gaps=True）能定位缺口
  于第 3000 帧附近、估算缺失帧数 ≈207——这是 2026-09 时间对齐改造成果的
  展示通道，断了会直接损害"快速上手看效果"的目标；
- ensure_sample_dataset：文件齐全、二次调用不重新生成。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools.check_temporal_sync import check_temporal_sync_impl
from app.ui.sample_dataset import ensure_sample_dataset, generate_sample_frames


def test_generation_is_deterministic() -> None:
    """固定种子：两次生成逐值一致。"""
    a = generate_sample_frames()
    b = generate_sample_frames()
    assert set(a) == set(b)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_imu_stream_has_designed_gap() -> None:
    """IMU 流中段含约 207 帧缺口（第 3000 帧起，≈1.72s）。"""
    imu = generate_sample_frames()["imu_glove.csv"]
    ts = np.sort(imu["timestamp_ns"].to_numpy(dtype=float))
    diffs = np.diff(ts)
    med = float(np.median(diffs))
    gaps = diffs[diffs > 5 * med]
    assert len(gaps) == 1, f"应恰好 1 个缺口，实得 {len(gaps)}"
    # 缺口时长 ≈ 207 × 8.33ms ≈ 1.72s（±3 帧）。
    assert abs(gaps[0] / _step_ns() - 207) <= 3


def _step_ns() -> float:
    return 8_330_000.0


def test_force_stream_is_gapless_baseline() -> None:
    """力传感流无缺口（对照流，可被基线推荐选中）。"""
    force = generate_sample_frames()["force_sensor.csv"]
    ts = np.sort(force["timestamp_ns"].to_numpy(dtype=float))
    diffs = np.diff(ts)
    med = float(np.median(diffs))
    assert not (diffs > 5 * med).any(), "对照流不应有缺口"


def test_tasks_table_shape() -> None:
    """任务表：20 个 episode，含 success/traj_length 列。"""
    tasks = generate_sample_frames()["tasks.csv"]
    assert len(tasks) == 20
    assert {"episode", "success", "traj_length"}.issubset(tasks.columns)
    assert set(tasks["success"].unique()) <= {0, 1}


def test_ensure_sample_dataset_end_to_end_gap_location(
    tmp_path: Path,
) -> None:
    """端到端：示例加载后 locate_gaps=True 能定位缺口（改造验收的演示路径）。"""
    d = ensure_sample_dataset(tmp_path / "sample")
    assert (d / "imu_glove.csv").exists()
    assert (d / "force_sensor.csv").exists()
    assert (d / "tasks.csv").exists()

    # 二次调用不重新生成（已存在直接返回）。
    mtime_first = (d / "tasks.csv").stat().st_mtime_ns
    ensure_sample_dataset(tmp_path / "sample")
    assert (d / "tasks.csv").stat().st_mtime_ns == mtime_first

    # 端到端：两条流做时间同步检查 + 缺口定位。
    streams = [
        {"path": str(d / name), "format": "csv", "kind": "imu"}
        for name in ("imu_glove.csv", "force_sensor.csv")
    ]
    ctx = RunContext(dataset_id="sample_demo", df=None, meta={
        "capabilities": {}, "streams": streams,
    })
    result = check_temporal_sync_impl(ctx, locate_gaps=True)
    assert result["success"] is True, result.get("user_message")

    checks = result["measurements"]["stream_checks"]
    imu_key = next(k for k in checks if "imu_glove" in Path(k).name)
    force_key = next(k for k in checks if "force_sensor" in Path(k).name)
    # IMU 流检出 1 个缺口、估算缺失帧数 ≈207。
    imu_gaps = checks[imu_key]["gaps"]
    assert imu_gaps["total"] == 1
    first = imu_gaps["gaps"][0]
    assert abs(first["missing_frames_est"] - 207) <= 3
    # 缺口起点 = 缺口前最后一帧（第 2999 帧，epoch0 + 2999×8.33ms，±0.1ms 抖动）。
    expected_start = 1_787_294_445_650_951_302 + 2999 * 8_330_000.0
    assert abs(first["start_ns"] - expected_start) < 5e5  # 容差含抖动
    # 对照流无缺口。
    assert checks[force_key]["gaps"]["total"] == 0
    # 基线：应推荐无缺口的 force_sensor（间隔稳定性更优）。
    assert "force_sensor" in Path(result["baseline_stream"]).name
