"""inspect_streams 双口径交叉验证的单元测试。

背景（wujiGlove_data 实测）：容器批量写入时间（log_ns）使 IMU 的
sample_rate 算出 ~58 万 Hz 荒谬值、左右单位推断不一致；传感器时间口径下
是规整 ~796 Hz 周期。双口径验证透出矛盾，不静默换列。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.tools.inspect_streams import _measure_rate_from_file, inspect_streams_impl
from app.tools.load_dataset import load_dataset_impl

T0_US = 1_787_294_445_600_000


def _batchy_rows(n: int = 200, batch: int = 10) -> list[dict]:
    """传感器时间规整（1kHz）、容器时间批量写入（burst）的信封行。"""
    rows = []
    for i in range(n):
        sensor_us = T0_US + i * 1000
        log_ns = T0_US * 1000 + (i // batch) * 50_000_000 + (i % batch) * 1_000
        rows.append({
            "mcap_log_time_ns": log_ns,
            "mcap_publish_time_ns": log_ns,
            "data": {
                "header": {"timestamp_us": sensor_us},
                "linear_acceleration": {"x": 0.1, "y": 0.2, "z": 9.8},
            },
        })
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_burst_main_with_periodic_alternate_flags_artifact(tmp_path: Path) -> None:
    """主口径 burst + 嵌套候选 periodic → suspected + 双候选 + note。"""
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / "imu.jsonl", _batchy_rows())
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    ins = inspect_streams_impl(ctx)
    stream = next(
        s for s in ins["table_streams"]
        if Path(s["source"]).name == "imu.jsonl"
    )
    mr = stream["sample_rate"]
    assert mr["present"] is True
    assert mr["stream_shape"] == "burst"
    assert mr.get("clock_artifact_suspected") is True
    # 主值已替换为传感器口径真值（UI 渲染主值字段——用户要求显示正确数字）。
    nested_key = next(k for k in mr["clock_candidates"] if k.startswith("data."))
    assert mr["sample_rate_hz"] == mr["clock_candidates"][nested_key]["sample_rate_hz"], (
        "主值应为传感器口径真值"
    )
    assert 900 <= mr["sample_rate_hz"] <= 1100
    # 容器失真值保留在 clock_candidates 供审计。
    assert mr["clock_candidates"]["mcap_log_time_ns"]["shape"] == "burst"
    assert "批量写入" in mr["clock_note"]
    assert "time_column=" in mr["clock_note"]
    assert "传感器时间口径" in mr.get("sample_rate_note", "")


def test_periodic_main_not_flagged(tmp_path: Path) -> None:
    """主口径 periodic（规整 log_ns）→ 不误报。"""
    d = tmp_path / "ds"
    d.mkdir()
    rows = [{
        "mcap_log_time_ns": T0_US * 1000 + i * 1_000_000,
        "data": {"header": {"timestamp_us": T0_US + i * 1000}},
    } for i in range(200)]
    _write_jsonl(d / "a.jsonl", rows)
    _write_jsonl(d / "b.jsonl", rows)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    ins = inspect_streams_impl(ctx)
    for s in ins["table_streams"]:
        mr = s["sample_rate"]
        assert not mr.get("clock_artifact_suspected")
        assert "clock_candidates" not in mr


def test_all_burst_without_alternate_not_flagged(tmp_path: Path) -> None:
    """主口径 burst 但无嵌套候选（如纯 tf 流）→ 不误报、不崩。"""
    d = tmp_path / "ds"
    d.mkdir()
    rows = [{
        "mcap_log_time_ns": T0_US * 1000 + (i // 10) * 50_000_000 + (i % 10) * 1_000,
        "data": {"transforms": [{"parent": "a", "child": "b"}]},
    } for i in range(200)]
    _write_jsonl(d / "tf.jsonl", rows)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    ins = inspect_streams_impl(ctx)
    for s in ins["table_streams"]:
        mr = s["sample_rate"]
        if mr.get("stream_shape") == "burst":
            assert not mr.get("clock_artifact_suspected")


def test_real_world_regression_batched_mcap(tmp_path: Path) -> None:
    """回归：真实信封形态（两个 IMU 流独立文件）双双检出矛盾且速率合理。"""
    d = tmp_path / "ds"
    d.mkdir()
    for name in ("left_glove_imu_data_palm.jsonl", "right_glove_imu_data_palm.jsonl"):
        _write_jsonl(d / name, _batchy_rows())
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    ins = inspect_streams_impl(ctx)
    flagged = 0
    for s in ins["table_streams"]:
        mr = s["sample_rate"]
        if mr.get("clock_artifact_suspected"):
            flagged += 1
            nested = next(
                v for k, v in mr["clock_candidates"].items()
                if k.startswith("data.")
            )
            # 传感器口径速率落在物理合理区间（<2000Hz）。
            assert nested["sample_rate_hz"] < 2000
    assert flagged == 2, f"两条 IMU 流都应检出，实际 {flagged}"
