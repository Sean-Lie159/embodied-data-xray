"""check_temporal_sync 参数化回归测试（2026-09-04 Commit C）。

守护四个新参数的行为契约：
- baseline_stream：指定生效；无法匹配时报错并列出候选（不静默回退）；
- streams：文件名子串过滤；无匹配时报错并列出可用流名；
- locate_gaps：定位缺口起止时刻与估算缺失帧数；
- time_column：指定列生效（经工具 schema 与端到端覆盖）。

真实需求背景：wujiGlove 双手数据手套，右手低频流缺失约 207 帧，
需回答"缺口落在哪里"——此前无参工具无法定位。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.check_temporal_sync import check_temporal_sync, check_temporal_sync_impl

_STEP_NS = 8_330_000.0  # 120 Hz
_NS_EPOCH_0 = 1_787_294_445_650_951_302


def _frame(n: int = 6339, start_offset: int = 0) -> pd.DataFrame:
    ts = np.arange(n, dtype=float) * _STEP_NS + _NS_EPOCH_0 + start_offset * _STEP_NS
    return pd.DataFrame({"timestamp": ts, "value": np.arange(n, dtype=float)})


def _gap_frame(n: int = 6339, gap_at: int = 3000, gap_size: int = 207) -> pd.DataFrame:
    """中段挖缺口的流：缺失 gap_size 帧（模拟右手手套掉线 1.7 s）。"""
    keep = np.delete(np.arange(n), np.arange(gap_at, gap_at + gap_size))
    ts = keep.astype(float) * _STEP_NS + _NS_EPOCH_0
    return pd.DataFrame({"timestamp": ts, "value": keep.astype(float)})


def _make_ctx(tmp_path: Path, files: dict[str, pd.DataFrame]) -> RunContext:
    meta: dict = {"capabilities": {}, "streams": []}
    for name, df in files.items():
        p = tmp_path / name
        df.to_csv(p, index=False)
        meta["streams"].append({"path": str(p), "format": "csv", "kind": "imu"})
    return RunContext(dataset_id="params_test", df=None, meta=meta)


def test_tool_schema_has_params() -> None:
    """注册给 Agent 的工具 schema 应包含四个参数（模型据此敢于用参数）。"""
    assert isinstance(check_temporal_sync, FunctionTool)
    schema = check_temporal_sync.params_json_schema
    props = schema.get("properties", {})
    for name in ("baseline_stream", "streams", "time_column", "locate_gaps"):
        assert name in props, f"工具 schema 缺少参数 {name}，实际 {list(props)}"


def test_baseline_stream_specified(tmp_path: Path) -> None:
    """指定基线流生效，且推荐说明注明'由用户指定'。"""
    ctx = _make_ctx(tmp_path, {
        "left_glove_emf.csv": _frame(),
        "right_glove_emf.csv": _frame(),
    })
    result = check_temporal_sync_impl(ctx, baseline_stream="right_glove_emf")
    assert result["success"] is True
    assert "right_glove_emf" in Path(result["baseline_stream"]).name
    rec = result["baseline_recommendation"]
    assert "用户指定" in rec["reason"]


def test_baseline_stream_no_match_returns_candidates(tmp_path: Path) -> None:
    """基线名无法匹配 → 结构化错误 + 候选清单（不静默回退到自动推荐）。"""
    ctx = _make_ctx(tmp_path, {
        "left_glove_emf.csv": _frame(),
        "right_glove_emf.csv": _frame(),
    })
    result = check_temporal_sync_impl(ctx, baseline_stream="no_such_stream")
    assert result["success"] is False
    assert result["error"] == "baseline_no_match"
    assert "left_glove_emf.csv" in result["user_message"]
    assert "right_glove_emf.csv" in result["user_message"]


def test_streams_filter_subset(tmp_path: Path) -> None:
    """streams 子串过滤：只检查左右手套（模拟'左右手套低频流成对对齐'）。"""
    ctx = _make_ctx(tmp_path, {
        "left_glove_emf.csv": _frame(),
        "right_glove_emf.csv": _frame(),
        "waist_tf.csv": _frame(),
        "other_sensor.csv": _frame(),
    })
    result = check_temporal_sync_impl(ctx, streams=["glove"])
    assert result["success"] is True
    checked_names = list(result["measurements"]["stream_checks"])
    assert len(checked_names) == 2, f"应只检查 2 条 glove 流，实际 {checked_names}"
    assert all("glove" in Path(k).name for k in checked_names)


def test_streams_filter_no_match_lists_available(tmp_path: Path) -> None:
    """过滤条件无匹配 → 结构化错误并列出可用流名示例。"""
    ctx = _make_ctx(tmp_path, {
        "left_glove_emf.csv": _frame(),
        "right_glove_emf.csv": _frame(),
    })
    result = check_temporal_sync_impl(ctx, streams=["lidar"])
    assert result["success"] is False
    assert result["error"] == "streams_no_match"
    assert "left_glove_emf.csv" in result["user_message"]


def test_locate_gaps_finds_missing_segment(tmp_path: Path) -> None:
    """locate_gaps=True：定位中段缺口的起止时刻与估算缺失帧数。"""
    gap_at, gap_size = 3000, 207
    ctx = _make_ctx(tmp_path, {
        "left_glove_emf.csv": _frame(),
        "right_glove_emf.csv": _gap_frame(gap_at=gap_at, gap_size=gap_size),
    })
    result = check_temporal_sync_impl(ctx, locate_gaps=True)
    assert result["success"] is True

    right_key = next(
        k for k in result["measurements"]["stream_checks"]
        if "right_glove" in Path(k).name
    )
    gaps = result["measurements"]["stream_checks"][right_key].get("gaps")
    assert gaps is not None, "locate_gaps=True 时应输出 gaps"
    assert gaps["total"] >= 1
    first = gaps["gaps"][0]
    # 缺口起点：第 gap_at 帧的时间戳；估算缺失帧数应接近 gap_size。
    expected_start = gap_at * _STEP_NS + _NS_EPOCH_0
    assert abs(first["start_ns"] - expected_start) < _STEP_NS
    assert first["missing_frames_est"] is not None
    assert abs(first["missing_frames_est"] - gap_size) <= 2
    # 缺口时长 ≈ gap_size × 8.33 ms ≈ 1.72 s。
    duration_s = first["duration_ns"] / 1e9
    assert 1.5 < duration_s < 2.0

    # 对照：无缺口的左手套流 gaps.total == 0。
    left_key = next(
        k for k in result["measurements"]["stream_checks"]
        if "left_glove" in Path(k).name
    )
    assert result["measurements"]["stream_checks"][left_key]["gaps"]["total"] == 0


def test_locate_gaps_default_absent(tmp_path: Path) -> None:
    """默认 locate_gaps=False：返回不携带 gaps 字段（控制上下文体积）。"""
    ctx = _make_ctx(tmp_path, {
        "left_glove_emf.csv": _frame(),
        "right_glove_emf.csv": _gap_frame(),
    })
    result = check_temporal_sync_impl(ctx)
    for c in result["measurements"]["stream_checks"].values():
        assert "gaps" not in c


def test_time_column_specified(tmp_path: Path) -> None:
    """time_column 指定：从双时间列（log_time/publish_time）中选指定的列。"""
    n = 500
    ts = np.arange(n, dtype=float) * _STEP_NS + _NS_EPOCH_0
    df = pd.DataFrame({
        "mcap_log_time_ns": ts,
        "mcap_publish_time_ns": ts + 12345.0,  # 略有不同，便于区分
        "value": np.arange(n, dtype=float),
    })
    paths = []
    for name in ("glove_a.csv", "glove_b.csv"):
        p = tmp_path / name
        df.to_csv(p, index=False)
        paths.append(str(p))
    ctx = RunContext(dataset_id="tc", df=None, meta={
        "capabilities": {},
        "streams": [{"path": p, "format": "csv", "kind": "imu"} for p in paths],
    })
    result = check_temporal_sync_impl(ctx, time_column="publish_time")
    assert result["success"] is True, result.get("user_message")
    checks = result["measurements"]["stream_checks"]
    assert len(checks) == 2
    for key, c in checks.items():
        # 指定列生效：实际采用的列名应命中 publish_time。
        assert "publish_time" in (c.get("timestamp_column") or ""), (
            f"{key} 应采用 publish_time 列，实际 {c.get('timestamp_column')}"
        )
