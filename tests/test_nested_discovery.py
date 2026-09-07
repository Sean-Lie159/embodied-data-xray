"""嵌套字段发现（discover_nested_fields）的单元测试。

信封型 JSONL/JSON：顶层容器时间 + data 内嵌传感器时间与数值信号。
覆盖：双时间候选发现、单位提示（后缀/量级）、信号特征（四元数模长/向量模长/
list 结构）、tf 型（仅 transforms 列表）、非信封格式返回空、发现接入流登记表。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from app.agent.context import RunContext
from app.tools._sniffing import discover_nested_fields
from app.tools.load_dataset import load_dataset_impl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _envelope_rows(n: int = 5, *, t0_us: int = 1_787_294_445_600_000) -> list[dict]:
    """构造信封行：顶层 log_ns + data 内嵌 header.timestamp_us 与 IMU 信号。"""
    rows = []
    for i in range(n):
        ts_us = t0_us + i * 1000  # 1ms 间隔（1000µs）
        rows.append({
            "mcap_log_time_ns": ts_us * 1000 + 10_700_000,  # 容器时间 = 传感器 + ~10.7ms
            "mcap_publish_time_ns": ts_us * 1000 + 10_700_000,
            "data": {
                "header": {"timestamp_us": ts_us, "frame_id": "l_palm_imu_link"},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "linear_acceleration": {"x": 0.5, "y": 0.6, "z": 9.7},
                "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.03},
            },
        })
    return rows


# --- 1. 时间候选发现 --------------------------------------------------------


def test_finds_top_and_nested_time_candidates(tmp_path: Path) -> None:
    """顶层容器时间与嵌套传感器时间都成为候选，单位提示各自正确。"""
    p = tmp_path / "imu.jsonl"
    _write_jsonl(p, _envelope_rows(5))
    r = discover_nested_fields(str(p), "jsonl")

    paths = {c["path"]: c for c in r["time_candidates"]}
    assert "mcap_log_time_ns" in paths
    assert "data.header.timestamp_us" in paths
    assert paths["mcap_log_time_ns"]["unit_hint"] == "ns"
    assert paths["data.header.timestamp_us"]["unit_hint"] == "us"
    # 全行覆盖 + 单调 + 中位间隔（原生单位）
    for c in paths.values():
        assert c["coverage"] == 1.0
        assert c["monotonic"] is True
        assert c["median_interval_native"] is not None
    assert paths["data.header.timestamp_us"]["median_interval_native"] == 1000.0
    assert paths["mcap_log_time_ns"]["median_interval_native"] == 1_000_000.0
    assert paths["mcap_log_time_ns"]["is_epoch"] is True
    assert r["sampled_rows"] == 5


def test_unit_hint_falls_back_to_magnitude(tmp_path: Path) -> None:
    """名字不含时间词但量级达到 epoch 水平 → 仍为候选，量级推断单位。"""
    p = tmp_path / "odd.jsonl"
    rows = [{"tick": 1_787_294_445_600_000_000 + i * 1_000_000} for i in range(5)]
    _write_jsonl(p, rows)
    r = discover_nested_fields(str(p), "jsonl")
    cands = {c["path"]: c for c in r["time_candidates"]}
    assert "tick" in cands
    assert cands["tick"]["unit_hint"] == "ns"


def test_small_scalar_not_time_candidate(tmp_path: Path) -> None:
    """普通小数值字段（如 gain=2.5）不是时间候选（名字与量级都不命中）。"""
    p = tmp_path / "cfg.jsonl"
    _write_jsonl(p, [{"gain": 2.5, "mode": "auto"} for _ in range(3)])
    r = discover_nested_fields(str(p), "jsonl")
    assert r["time_candidates"] == []


# --- 2. 信号字段发现 --------------------------------------------------------


def test_signal_fields_quat_vec3_lists(tmp_path: Path) -> None:
    """四元数模长≈1、加速度模长≈9.8、list_of_dict 结构被正确发现。"""
    p = tmp_path / "imu.jsonl"
    _write_jsonl(p, _envelope_rows(5))
    r = discover_nested_fields(str(p), "jsonl")

    sig = {s["path"]: s for s in r["signal_fields"]}
    quat = sig["data.orientation"]
    assert quat["kind"] == "quat4"
    assert abs(quat["norm_mean"] - 1.0) < 1e-6
    accel = sig["data.linear_acceleration"]
    assert accel["kind"] == "vec3"
    assert abs(accel["mag_mean"] - math.sqrt(0.25 + 0.36 + 94.09)) < 1e-3
    assert "data.angular_velocity" in sig
    assert sig["data.header"]["kind"] == "dict" if "data.header" in sig else True


def test_tf_like_transform_list_discovered(tmp_path: Path) -> None:
    """tf 型：transforms 列表（list_of_dict）与其内嵌时间戳被发现。"""
    p = tmp_path / "tf.jsonl"
    rows = [
        {
            "mcap_log_time_ns": 1_787_294_445_650_000_000 + i * 640_000,
            "mcap_publish_time_ns": 1_787_294_445_650_000_000 + i * 640_000,
            "data": {"transforms": [{
                "timestamp_us": 1_787_294_445_617_000 + i * 640,
                "parent_frame_id": "waist",
                "child_frame_id": "l_wrist",
                "translation": {"x": 0.1, "y": 0.2, "z": 0.3},
            }]},
        }
        for i in range(5)
    ]
    _write_jsonl(p, rows)
    r = discover_nested_fields(str(p), "jsonl")
    time_paths = {c["path"] for c in r["time_candidates"]}
    assert "data.transforms.0.timestamp_us" in time_paths, (
        "transforms 内嵌时间戳应被发现（含 .0 下标路径）"
    )
    sig = {s["path"]: s for s in r["signal_fields"]}
    assert sig["data.transforms"]["kind"] == "list_of_dict"
    assert sig["data.transforms"]["len_mode"] == 1
    assert sig["data.transforms.0.translation"]["kind"] == "vec3"


def test_no_nested_time_returns_empty_not_guess(tmp_path: Path) -> None:
    """data 内无任何时间字段 → 候选为空（如实返回，不硬猜）。"""
    p = tmp_path / "plain.jsonl"
    _write_jsonl(p, [{"mcap_log_time_ns": 1e18 + i, "data": {"value": i}}
                     for i in range(4)])
    # 顶层 log_ns 仍是候选（名字+量级），但 data 内无嵌套时间。
    r = discover_nested_fields(str(p), "jsonl")
    paths = {c["path"] for c in r["time_candidates"]}
    assert "mcap_log_time_ns" in paths
    assert not any(sp.startswith("data.") for sp in paths)


# --- 3. 边界 ----------------------------------------------------------------


def test_non_envelope_format_returns_empty(tmp_path: Path) -> None:
    """csv/parquet 不适用嵌套发现 → 空结果。"""
    assert discover_nested_fields("x.csv", "csv")["time_candidates"] == []
    assert discover_nested_fields("x.parquet", "parquet")["signal_fields"] == []


def test_malformed_file_returns_empty_not_crash(tmp_path: Path) -> None:
    """非法 JSONL：空结果不崩。"""
    p = tmp_path / "bad.jsonl"
    p.write_text("不是 JSON\n{{{\n", encoding="utf-8")
    r = discover_nested_fields(str(p), "jsonl")
    assert r["time_candidates"] == [] and r["signal_fields"] == []


def test_deep_nesting_respects_max_depth(tmp_path: Path) -> None:
    """超深嵌套在 max_depth 处截断，不无限递归。"""
    p = tmp_path / "deep.jsonl"
    deep: dict = {"leaf_ts": 1_787_294_445_600_000_000}
    for _ in range(20):
        deep = {"inner": deep}
    _write_jsonl(p, [deep])
    r = discover_nested_fields(str(p), "jsonl", max_depth=6)
    assert len(json.dumps(r)) < 10_000  # 截断生效（无 20 层路径）


# --- 4. 接入流登记表 --------------------------------------------------------


def test_load_directory_attaches_discovery_to_streams(tmp_path: Path) -> None:
    """目录加载后，jsonl 流带 time_candidates/signal_fields；csv 流不带。"""
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / "imu.jsonl", _envelope_rows(5))
    import pandas as pd

    pd.DataFrame({"t": [1, 2], "v": [0.1, 0.2]}).to_csv(d / "plain.csv", index=False)

    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(d))
    assert r["success"] is True
    by_name = {Path(s["path"]).name: s for s in ctx.meta["streams"]}

    imu = by_name["imu.jsonl"]
    assert isinstance(imu.get("time_candidates"), list)
    cand_paths = {c["path"] for c in imu["time_candidates"]}
    assert "mcap_log_time_ns" in cand_paths
    assert "data.header.timestamp_us" in cand_paths
    assert any(s["path"] == "data.orientation" for s in imu["signal_fields"])

    csv_stream = by_name["plain.csv"]
    assert "time_candidates" not in csv_stream  # 非 jsonl/json 不附加
