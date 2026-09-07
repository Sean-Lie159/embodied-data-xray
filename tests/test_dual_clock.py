"""双口径交叉验证 + 嵌套 time_column + 基线 tie-break 的单元测试。

背景（wujiGlove_data 实测）：容器批量写入时间（log_ns）会让 IMU 判为 burst
（1.7 万假 gap），传感器时间（data.header.timestamp_us）口径下是规整周期。
覆盖：
  - 默认调用检出 clock_artifact_suspected + clock_conflicts + user_message 附注；
  - time_column 指定嵌套路径重算 → periodic、速率正确、单位按 _us 归一化；
  - 嵌套路径不存在 → 结构化"未参与"信息，不崩溃不静默回退；
  - 基线分数并列时按路径字母序确定 + top_candidates 透出。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.tools._data_access import read_nested_time_column
from app.tools.check_temporal_sync import check_temporal_sync_impl
from app.tools.load_dataset import load_dataset_impl

T0_US = 1_787_294_445_600_000
N_ROWS = 120  # 2 秒 @ 1000Hz（传感器口径）


def _batchy_rows(n: int = N_ROWS, batch: int = 10) -> list[dict]:
    """构造"传感器时间规整、容器时间批量写入"的信封行。

    传感器：每行 +1000µs（1kHz 规整周期）。
    容器：每 batch 行一批，批内 +1000ns（挤在同一写入批次），批间跳 50ms
    → log 口径 mean/median 比值 ≫ 3 → burst。
    """
    rows = []
    for i in range(n):
        sensor_us = T0_US + i * 1000
        log_ns = T0_US * 1000 + (i // batch) * 50_000_000 + (i % batch) * 1_000
        rows.append({
            "mcap_log_time_ns": log_ns,
            "mcap_publish_time_ns": log_ns,
            "data": {
                "header": {"timestamp_us": sensor_us, "frame_id": "palm"},
                "linear_acceleration": {"x": 0.1, "y": 0.2, "z": 9.8},
            },
        })
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_dataset(tmp_path: Path, names: tuple[str, ...] = ("imu_a.jsonl", "imu_b.jsonl")) -> Path:
    d = tmp_path / "ds"
    d.mkdir()
    for name in names:
        _write_jsonl(d / name, _batchy_rows())
    return d


# --- 1. 双口径矛盾检出 ------------------------------------------------------


def test_clock_artifact_detected_on_default_call(tmp_path: Path) -> None:
    """默认（log_ns 主口径判 burst）→ 嵌套口径 periodic → 矛盾透出。"""
    d = _make_dataset(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    s = check_temporal_sync_impl(ctx)
    assert s["success"] is True

    conflicts = s.get("clock_conflicts") or []
    assert len(conflicts) == 2, f"两条 IMU 流都应检出矛盾，实际 {len(conflicts)}"
    assert "疑似批量写入" in str(s.get("user_message"))

    checks = s["measurements"]["stream_checks"]
    for key, chk in checks.items():
        assert chk.get("clock_artifact_suspected") is True, key
        cands = chk["clock_candidates"]
        assert cands["mcap_log_time_ns"]["shape"] == "burst"
        nested_key = next(k for k in cands if k.startswith("data."))
        assert cands[nested_key]["shape"] == "periodic"
        assert "批量写入" in chk["clock_note"]
        assert "time_column" in chk["clock_note"]


def test_periodic_main_not_flagged(tmp_path: Path) -> None:
    """主口径本身 periodic（无双口径矛盾）→ 不误报 clock_conflicts。"""
    d = tmp_path / "ds"
    d.mkdir()
    # 直接用规整 log_ns（无批量写入）→ 主口径 periodic → 不进入矛盾检测。
    rows = [{
        "mcap_log_time_ns": T0_US * 1000 + i * 1_000_000,
        "mcap_publish_time_ns": T0_US * 1000 + i * 1_000_000,
        "data": {"header": {"timestamp_us": T0_US + i * 1000}},
    } for i in range(N_ROWS)]
    _write_jsonl(d / "a.jsonl", rows)
    _write_jsonl(d / "b.jsonl", rows)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    s = check_temporal_sync_impl(ctx)
    assert s.get("clock_conflicts") == []
    assert "多时间口径" not in str(s.get("user_message"))


# --- 2. 嵌套 time_column 重算 ----------------------------------------------


def test_nested_time_column_recompute(tmp_path: Path) -> None:
    """time_column=嵌套路径 → periodic、速率 1000Hz、单位按 _us 归一化。"""
    d = _make_dataset(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    s = check_temporal_sync_impl(ctx, time_column="data.header.timestamp_us")
    assert s["success"] is True
    checks = s["measurements"]["stream_checks"]
    for key, chk in checks.items():
        assert chk.get("timestamp_column") == "data.header.timestamp_us"
        assert chk.get("stream_shape") == "periodic", (
            "传感器时间口径下应为规整周期（容器批量写入伪影消失）"
        )
        rate = chk.get("actual_rate_hz")
        assert rate is not None and 990 <= rate <= 1010, f"期望 ~1000Hz，实际 {rate}"
        assert chk.get("timestamp_unit") == "us"
        assert "us" in str(chk.get("timestamp_unit_basis"))
        assert not chk.get("clock_artifact_suspected")


def test_nested_time_column_missing_structured(tmp_path: Path) -> None:
    """嵌套路径不存在 → 结构化"未参与"，不崩溃、不静默回退到主列。"""
    d = _make_dataset(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    s = check_temporal_sync_impl(ctx, time_column="data.no_such_field")
    # 每条流的 status 必须注明指定列无法匹配（或整体 not_applicable）。
    if s.get("success"):
        status = json.dumps(s.get("streams_status", {}), ensure_ascii=False)
        assert "无法匹配" in status or "data.no_such_field" in status
    else:
        assert s.get("error") in ("not_applicable", "baseline_no_match")


def test_read_nested_time_column_direct(tmp_path: Path) -> None:
    """读取器直测：int 原值、缺失行跳过、非信封格式返回 None。"""
    d = _make_dataset(tmp_path)
    p = d / "imu_a.jsonl"
    series = read_nested_time_column(str(p), "jsonl", "data.header.timestamp_us")
    assert series is not None
    assert series.name == "data.header.timestamp_us"
    assert len(series) == N_ROWS
    # int 原值（ns epoch 的 µs 表示超出 float64 精确范围，必须保 int）
    assert all(isinstance(v, int) for v in series.head(5))
    assert read_nested_time_column(str(p), "csv", "x.y") is None
    assert read_nested_time_column(str(p), "jsonl", "data.missing") is None


# --- 3. 基线确定性 ----------------------------------------------------------


def test_baseline_tiebreak_alphabetical_and_top_candidates(tmp_path: Path) -> None:
    """分数并列 → 按路径字母序；recommendation 带 top_candidates。"""
    d = _make_dataset(tmp_path, names=("b_stream.jsonl", "a_stream.jsonl"))
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    # 用传感器时间口径使两流完全一致 → 分数并列。
    s = check_temporal_sync_impl(ctx, time_column="data.header.timestamp_us")
    assert s["success"] is True
    rec = s["baseline_recommendation"]
    assert rec["stream"].endswith("a_stream.jsonl"), (
        f"并列时应按字母序选 a_stream，实际 {rec['stream']}"
    )
    top = rec.get("top_candidates") or []
    assert len(top) == 2
    assert top[0]["stream"] == "a_stream.jsonl"
    assert {t["stream"] for t in top} == {"a_stream.jsonl", "b_stream.jsonl"}
