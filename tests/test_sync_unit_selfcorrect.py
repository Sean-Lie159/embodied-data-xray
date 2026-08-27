"""Commit C：check_temporal_sync 接入单位自我纠正的单元测试。

覆盖：嗅探判错的单位（如 parquet 时间戳被判 ns 实为 s）经自我纠正后，同步检查
算出物理合理采样率，而非 2.5e10 Hz 之类非物理值。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.check_temporal_sync import check_temporal_sync_impl
from app.tools.load_dataset import load_dataset_impl


def _misjudged_unit_dir(tmp_path: Path) -> Path:
    """构造：主表时间戳实际为秒制，但流登记表 timestamp_unit 误判为 ns。"""
    root = tmp_path / "ds"
    root.mkdir()
    # 秒制时间戳（0, 0.04, 0.08... 即 25Hz），但流登记表标 timestamp_unit=ns。
    ts = [i * 0.04 for i in range(100)]
    pd.DataFrame({
        "timestamp": ts,
        "x": [1.0] * 100, "y": [1.0] * 100, "z": [9.8] * 100,
    }).to_csv(root / "imu.csv", index=False)
    # 第二路表（位姿）作对齐基准，同为秒制。
    pd.DataFrame({
        "timestamp": ts,
        "pos_x": [1.0] * 100, "pos_y": [1.0] * 100, "pos_z": [1.0] * 100,
        "quat_w": [1.0] * 100, "quat_x": [0.0] * 100, "quat_y": [0.0] * 100, "quat_z": [0.0] * 100,
    }).to_csv(root / "pose.csv", index=False)
    return root


def test_sync_self_corrects_misjudged_unit(tmp_path: Path) -> None:
    """嗅探误判 ns（实为 s）→ check_temporal_sync 自我纠正为秒，采样率合理。"""
    root = _misjudged_unit_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    # 强制把流登记表 timestamp_unit 标为 ns（模拟嗅探误判）。
    for s in ctx.meta["streams"]:
        s["timestamp_unit"] = "ns"
    r = check_temporal_sync_impl(ctx)
    assert r["success"] is True
    # 单流检查的实际采样率应为 ~25Hz（秒制），而非 2.5e10 Hz（ns 误判）。
    rates = [
        c.get("actual_rate_hz")
        for c in r["measurements"]["stream_checks"].values()
        if c.get("present") and c.get("actual_rate_hz")
    ]
    assert rates, "应有实测采样率"
    for rate in rates:
        assert 1.0 < rate < 1000.0, f"采样率 {rate} 应为物理合理值（自我纠正未生效）"


def test_sync_no_selfcorrect_when_unit_plausible(tmp_path: Path) -> None:
    """单位本就合理（秒制）→ 采样率正常，不误纠正。"""
    root = _misjudged_unit_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    # 明确标为秒制。
    for s in ctx.meta["streams"]:
        s["timestamp_unit"] = "s"
    r = check_temporal_sync_impl(ctx)
    assert r["success"] is True
    rates = [
        c.get("actual_rate_hz")
        for c in r["measurements"]["stream_checks"].values()
        if c.get("present") and c.get("actual_rate_hz")
    ]
    assert rates
    for rate in rates:
        assert 1.0 < rate < 1000.0
