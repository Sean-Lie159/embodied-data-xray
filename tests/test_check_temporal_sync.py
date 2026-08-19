"""app/tools/check_temporal_sync 工具的单元测试。

验证五种合成时间戳场景的判定：正常对齐（pass）、丢帧超阈（fail）、线性漂移
（fail）、乱序（warn）、重复（warn），以及前置条件不适用与 verification_level。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.check_temporal_sync import check_temporal_sync, check_temporal_sync_impl


def _make_context(tmp_path: Path, stream_ts: list[np.ndarray]) -> RunContext:
    """构造含两条表格流（imu + force）的上下文，写入临时 csv。"""
    meta = {"capabilities": {"has_imu": True}, "streams": []}
    for i, ts in enumerate(stream_ts):
        p = tmp_path / f"stream{i}.csv"
        pd.DataFrame({"timestamp": ts}).to_csv(p, index=False)
        meta["streams"].append({
            "path": str(p), "format": "csv",
            "kind": "imu" if i == 0 else "force",
            "nominal_rate_hz": 100.0,
        })
    return RunContext(dataset_id="sync", df=None, meta=meta)


def test_check_temporal_sync_is_registered() -> None:
    assert isinstance(check_temporal_sync, FunctionTool)
    assert check_temporal_sync.name == "check_temporal_sync"


def test_single_stream_not_applicable(tmp_path: Path) -> None:
    t = np.arange(0, 1.0, 0.01)
    ctx = _make_context(tmp_path, [t])  # 仅 1 路流
    result = check_temporal_sync_impl(ctx)
    assert result["success"] is False
    assert result["error"] == "not_applicable"


def test_normal_alignment_pass(tmp_path: Path) -> None:
    t = np.arange(0, 1.0, 0.01)
    ctx = _make_context(tmp_path, [t, t + 0.0005])  # 0.5ms 固定小偏移
    result = check_temporal_sync_impl(ctx)
    assert result["success"] is True
    assert result["result"] == "pass"
    assert result["verification_level"] == "timestamp_consistency"
    assert result["baseline_stream"] is not None


def test_frame_loss_over_threshold_fail(tmp_path: Path) -> None:
    t = np.arange(0, 1.0, 0.01)
    # 第二流缺失 0.4~0.8 段 → 丢帧约 40%。
    t2 = np.concatenate([np.arange(0, 0.4, 0.01), np.arange(0.8, 1.0, 0.01)])
    ctx = _make_context(tmp_path, [t, t2])
    result = check_temporal_sync_impl(ctx)
    assert result["result"] == "fail"


def test_linear_drift_fail(tmp_path: Path) -> None:
    t = np.arange(0, 1.0, 0.01)
    ctx = _make_context(tmp_path, [t, t + 0.2 * t])  # 20% 线性漂移
    result = check_temporal_sync_impl(ctx)
    assert result["result"] == "fail"
    assert "漂移" in result["user_message"]
    # 有受影响 episode 标注（无 episode 划分 → whole_recording）。
    assert "whole_recording" in result["affected_episodes"]


def test_disordered_timestamps_warn(tmp_path: Path) -> None:
    t = np.arange(0, 1.0, 0.01)
    t4 = t.copy()
    t4[10], t4[11] = t4[11], t4[10]  # 交换一对 → 乱序
    ctx = _make_context(tmp_path, [t, t4])
    result = check_temporal_sync_impl(ctx)
    assert result["result"] == "warn"


def test_duplicate_timestamps_warn(tmp_path: Path) -> None:
    t = np.arange(0, 1.0, 0.01)
    t5 = t.copy()
    t5[20] = t5[19]  # 制造重复时间戳
    ctx = _make_context(tmp_path, [t, t5])
    result = check_temporal_sync_impl(ctx)
    assert result["result"] == "warn"


def _video_only_context() -> RunContext:
    """构造纯多视频流上下文（无表格流）。"""
    meta = {
        "capabilities": {"has_video_streams": True},
        "streams": [
            {"path": "a.mp4", "format": "video", "kind": "video", "channels": []},
            {"path": "b.mp4", "format": "video", "kind": "video", "channels": []},
        ],
    }
    return RunContext(dataset_id="videos", df=None, meta=meta)


def test_pure_video_dataset_not_applicable() -> None:
    """纯视频数据集（无表格流）必须返回"不适用"，禁止判 pass。"""
    ctx = _video_only_context()
    result = check_temporal_sync_impl(ctx)
    assert result["success"] is False
    assert result["error"] == "not_applicable"
    # streams_status 应标明视频流未参与及原因。
    assert "v1" in str(result.get("streams_status", {}))


def test_nominal_missing_shows_skipped(tmp_path: Path) -> None:
    """标称采样率缺失时，实际 vs 标称检查应标 skipped 且可见。"""
    t = np.arange(0, 1.0, 0.01)
    meta = {
        "capabilities": {"has_imu": True},
        "streams": [],
    }
    for i, ts in enumerate([t, t + 0.0005]):
        p = tmp_path / f"s{i}.csv"
        pd.DataFrame({"timestamp": ts}).to_csv(p, index=False)
        meta["streams"].append({
            "path": str(p), "format": "csv", "kind": "imu" if i == 0 else "force",
            # 不设置 nominal_rate_hz，模拟标称缺失
        })
    ctx = RunContext(dataset_id="s", df=None, meta=meta)
    result = check_temporal_sync_impl(ctx)
    # 任一参与流的 nominal_check 应为 skipped。
    nominal_skipped = any(
        c.get("nominal_check", {}).get("status") == "skipped"
        for c in result["measurements"]["stream_checks"].values()
        if c.get("present")
    )
    assert nominal_skipped is True


def test_drift_includes_relativity_note(tmp_path: Path) -> None:
    """检出漂移时，返回应带相对性说明 note。"""
    t = np.arange(0, 1.0, 0.01)
    ctx = _make_context(tmp_path, [t, t + 0.2 * t])  # 强漂移
    result = check_temporal_sync_impl(ctx)
    assert result["result"] == "fail"
    assert result["note"] is not None
    assert "相对量" in result["note"]  # 漂移相对性说明
