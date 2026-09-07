"""确认 time_column 提升为每流主口径的单元测试。

背景（用户报告）：agent 已确认 data.header.timestamp_us 为传感器采样时间，
但右栏流清单仍显示容器口径的荒谬采样率（~58 万 Hz），且 sync 需每次手动传
time_column。修复：确认列优先于词表（inspect_streams 采样率 + sync 对齐），
confirm=True 落盘后立即应用覆盖到当前会话（含采样率缓存失效重测）。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.tools.check_temporal_sync import check_temporal_sync_impl
from app.tools.inspect_streams import (
    _measure_stream_rate,
    inspect_streams_impl,
)
from app.tools.load_dataset import load_dataset_impl
from app.tools.propose_semantics import propose_stream_semantics_impl

T0_US = 1_787_294_445_600_000


def _batchy_rows(n: int = 200, batch: int = 10) -> list[dict]:
    """传感器时间规整（1kHz）、容器时间批量写入（burst）的信封行。"""
    rows = []
    for i in range(n):
        rows.append({
            "mcap_log_time_ns": T0_US * 1000 + (i // batch) * 50_000_000 + (i % batch) * 1_000,
            "mcap_publish_time_ns": T0_US * 1000 + (i // batch) * 50_000_000 + (i % batch) * 1_000,
            "data": {
                "header": {"timestamp_us": T0_US + i * 1000},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.03},
                "linear_acceleration": {"x": 0.1, "y": 0.2, "z": 9.8},
            },
        })
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_ctx(tmp_path: Path, name: str = "imu.jsonl") -> RunContext:
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / name, _batchy_rows())
    # 第二条流：sync 要求 ≥2 路可对齐流（正确的前置守卫）。
    _write_jsonl(d / "imu_b.jsonl", _batchy_rows())
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    return ctx


def test_confirmed_time_column_prioritized_in_inspect(tmp_path: Path) -> None:
    """确认列（嵌套路径）优先：inspect_streams 采样率为传感器口径（~1000Hz
    periodic），不再是容器口径的荒谬值。"""
    ctx = _make_ctx(tmp_path)
    stream = next(s for s in ctx.meta["streams"])
    stream["time_column"] = "data.header.timestamp_us"  # 模拟已确认落盘
    stream["measured_rate"] = None

    ins = inspect_streams_impl(ctx)
    mr = next(s for s in ins["table_streams"])["sample_rate"]
    assert mr["present"] is True
    assert mr["timestamp_column"] == "data.header.timestamp_us"
    assert mr["stream_shape"] == "periodic"
    assert 900 <= mr["sample_rate_hz"] <= 1100
    assert mr["timestamp_unit"] == "us"


def test_cached_stale_rate_invalidated_by_confirmation(tmp_path: Path) -> None:
    """缓存失效：确认前测的容器口径荒谬值，确认后重测为传感器口径。"""
    ctx = _make_ctx(tmp_path)
    stream = next(s for s in ctx.meta["streams"])
    # 先按旧口径测一次（产生缓存）。
    old = _measure_stream_rate(stream, ctx.meta)
    assert old["timestamp_column"] == "mcap_log_time_ns"
    assert old["stream_shape"] == "burst"

    # 用户确认传感器时间列 → 重测（缓存因列不同而失效）。
    stream["time_column"] = "data.header.timestamp_us"
    new = _measure_stream_rate(stream, ctx.meta)
    assert new["timestamp_column"] == "data.header.timestamp_us"
    assert new["stream_shape"] == "periodic"
    assert 900 <= new["sample_rate_hz"] <= 1100


def test_sync_uses_confirmed_time_column_by_default(tmp_path: Path) -> None:
    """sync 不传 time_column：确认列自动生效（仅作用于被确认的流）。

    逐流确认语义：确认过的流切传感器口径（periodic），未确认的流保持
    词表自动口径（burst）——两条流行为不同是**设计意图**。
    """
    ctx = _make_ctx(tmp_path)
    stream = next(s for s in ctx.meta["streams"])
    stream["time_column"] = "data.header.timestamp_us"

    s = check_temporal_sync_impl(ctx)
    assert s["success"] is True
    checks = s["measurements"]["stream_checks"]
    confirmed = [
        v for v in checks.values()
        if v.get("present")
        and v.get("timestamp_column") == "data.header.timestamp_us"
    ]
    assert len(confirmed) == 1, "仅被确认的流切换口径"
    assert confirmed[0]["stream_shape"] == "periodic"
    assert 900 <= confirmed[0]["actual_rate_hz"] <= 1100
    others = [
        v for v in checks.values()
        if v.get("present")
        and v.get("timestamp_column") != "data.header.timestamp_us"
    ]
    assert len(others) == 1
    assert others[0]["stream_shape"] == "burst"  # 未确认流保持自动口径


def test_global_param_beats_nothing_but_confirmed_wins(tmp_path: Path) -> None:
    """确认列优先于全局参数（登记表 > 调用参数）。"""
    ctx = _make_ctx(tmp_path)
    stream = next(s for s in ctx.meta["streams"])
    stream["time_column"] = "data.header.timestamp_us"

    # 全局参数给主口径列：确认过的流仍走确认列（用户确认 > 调用参数）。
    s = check_temporal_sync_impl(ctx, time_column="mcap_log_time_ns")
    assert s["success"] is True
    checks = s["measurements"]["stream_checks"]
    confirmed = [
        v for v in checks.values()
        if v.get("present")
        and v.get("timestamp_column") == "data.header.timestamp_us"
    ]
    assert len(confirmed) == 1
    assert confirmed[0]["stream_shape"] == "periodic"


def test_propose_confirm_refreshes_session_streams(tmp_path: Path) -> None:
    """confirm=True 后：当前会话流登记表立即应用覆盖（语义 + time_column），
    且采样率缓存被清（重测即用确认列）。"""
    ctx = _make_ctx(tmp_path)
    # 先制造旧口径缓存。
    stream = next(s for s in ctx.meta["streams"] if Path(s["path"]).name == "imu.jsonl")
    _measure_stream_rate(stream, ctx.meta)
    assert stream["measured_rate"]["timestamp_column"] == "mcap_log_time_ns"

    r = propose_stream_semantics_impl(
        ctx,
        [{"file": "imu.jsonl", "kind": "imu", "semantic_label": "手掌 IMU",
          "time_column": "data.header.timestamp_us"}],
        confirm=True,
    )
    assert r["success"] is True and r["confirmed"] == ["imu.jsonl"]

    # 当前会话立即生效（无需重载）。注意：meta["streams"] 被替换为新列表，
    # 需从 meta 重新取流对象（旧引用指向替换前的 dict）。
    refreshed = next(
        s for s in ctx.meta["streams"] if Path(s["path"]).name == "imu.jsonl"
    )
    assert refreshed["semantic_label"] == "手掌 IMU"
    assert refreshed["kind"] == "imu"
    assert refreshed["time_column"] == "data.header.timestamp_us"
    assert refreshed["measured_rate"] is None  # 缓存已清

    # inspect_streams 重测 → 确认流进 imus 分组（kind=imu），采样率为传感器
    # 口径（确认列优先 + 缓存失效重测）。
    ins = inspect_streams_impl(ctx)
    imu_group = ins.get("imus") if isinstance(ins.get("imus"), list) else []
    imu_entry = next(
        s
        for grp in imu_group
        for s in (grp.get("streams") or [])
        if Path(s.get("source", "")).name == "imu.jsonl"
    )
    mr = imu_entry.get("sample_rate") or {}
    assert mr.get("timestamp_column") == "data.header.timestamp_us"
    assert mr.get("stream_shape") == "periodic"


def test_no_confirmation_keeps_automatic_column(tmp_path: Path) -> None:
    """未确认（无登记表 time_column）：行为不变（词表自动选主口径）。"""
    ctx = _make_ctx(tmp_path)
    ins = inspect_streams_impl(ctx)
    mr = next(s for s in ins["table_streams"])["sample_rate"]
    assert mr["timestamp_column"] == "mcap_log_time_ns"
    assert mr["stream_shape"] == "burst"
