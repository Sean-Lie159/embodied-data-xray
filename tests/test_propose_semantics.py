"""propose_stream_semantics（批量语义假设：验证 → 确认 → 落盘）的单元测试。

覆盖：imu 强/弱/失败验证、自定义 kind 的结构检查、confirm 闸门（默认不落盘、
确认后落盘且重载生效、failed 不落盘）、空假设/未加载数据集的结构化错误。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl
from app.tools.propose_semantics import propose_stream_semantics_impl

T0_US = 1_787_294_445_600_000


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _imu_rows(n: int = 20) -> list[dict]:
    """合规 IMU 信封行：模长≈1 的四元数 + ≈9.8 的加速度 + 角速度。"""
    rows = []
    for i in range(n):
        rows.append({
            "mcap_log_time_ns": (T0_US + i * 1000) * 1000,
            "data": {
                "header": {"timestamp_us": T0_US + i * 1000},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "linear_acceleration": {"x": 0.1, "y": 0.2, "z": 9.7},
                "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.03},
            },
        })
    return rows


def _bad_imu_rows(n: int = 20) -> list[dict]:
    """不合规：加速度模长 ~100（远超 2g），无四元数。"""
    return [{
        "mcap_log_time_ns": (T0_US + i * 1000) * 1000,
        "data": {"linear_acceleration": {"x": 30.0, "y": 40.0, "z": 80.0}},
    } for i in range(n)]


def _make_dataset(tmp_path: Path) -> RunContext:
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / "good_imu.jsonl", _imu_rows())
    _write_jsonl(d / "bad_imu.jsonl", _bad_imu_rows())
    rows = [
        {"mcap_log_time_ns": (T0_US + i * 8300) * 1000,
         "data": {"fingers": [{"angles": [0.1 * i, 0.2 * i]}]}}
        for i in range(20)
    ]
    _write_jsonl(d / "tactile.jsonl", rows)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    return ctx


def _a(**kw) -> dict:
    return dict(kw)


# --- 1. 验证分级 ------------------------------------------------------------


def test_imu_strong_verification(tmp_path: Path) -> None:
    """合规 IMU（四元数 + 加速度 + 角速度）→ strong。"""
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(
        ctx, [_a(file="good_imu.jsonl", kind="imu",
                           semantic_label="手掌 IMU")]
    )
    assert r["success"] is True
    item = r["results"][0]
    assert item["verified"] == "strong", item
    assert "四元数" in item["evidence"]
    assert r["summary"]["strong"] == 1
    assert r["confirmed"] == []  # 默认不落盘


def test_bad_imu_fails_verification(tmp_path: Path) -> None:
    """加速度模长远超物理范围且无四元数 → failed。"""
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(
        ctx, [_a(file="bad_imu.jsonl", kind="imu")]
    )
    item = r["results"][0]
    assert item["verified"] == "failed", item


def test_custom_kind_structural_weak(tmp_path: Path) -> None:
    """自定义 kind（tactile）：fields 存在 → weak；不存在 → failed。"""
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(ctx, [
        _a(file="tactile.jsonl", kind="tactile",
           semantic_label="触觉阵列", fields=["data.fingers.0.angles.0"]),
        _a(file="tactile.jsonl", kind="tactile",
           semantic_label="误标", fields=["data.no_such"]),
    ])
    assert r["results"][0]["verified"] == "weak"
    assert r["results"][1]["verified"] == "failed"
    assert r["summary"]["weak"] == 1 and r["summary"]["failed"] == 1


def test_stream_not_found_failed(tmp_path: Path) -> None:
    """流登记表不存在的文件名 → failed（不崩溃）。"""
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(
        ctx, [_a(file="ghost.jsonl", kind="imu")]
    )
    assert r["results"][0]["verified"] == "failed"


# --- 2. confirm 闸门 --------------------------------------------------------


def test_confirm_false_does_not_persist(tmp_path: Path) -> None:
    """confirm=False（默认）：验证但不落盘——重载后流清单仍 unknown。"""
    ctx = _make_dataset(tmp_path)
    propose_stream_semantics_impl(
        ctx, [_a(file="good_imu.jsonl", kind="imu",
                           semantic_label="手掌 IMU")]
    )
    ctx2 = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx2, str(tmp_path / "ds"))
    stream = next(s for s in ctx2.meta["streams"]
                  if Path(s["path"]).name == "good_imu.jsonl")
    assert stream.get("semantic_label") != "手掌 IMU"


def test_confirm_persists_and_reload_applies(tmp_path: Path) -> None:
    """confirm=True：落盘 + 重载后语义标签与 time_column 生效（核心闭环）。"""
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(
        ctx,
        [_a(file="good_imu.jsonl", kind="imu",
            semantic_label="手掌 IMU（ROS Imu）",
            time_column="data.header.timestamp_us")],
        confirm=True,
    )
    assert r["confirmed"] == ["good_imu.jsonl"]
    assert "落盘" in r["user_message"]

    # 重新加载：第 4 层覆盖生效。
    ctx2 = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx2, str(tmp_path / "ds"))
    stream = next(s for s in ctx2.meta["streams"]
                  if Path(s["path"]).name == "good_imu.jsonl")
    assert stream.get("semantic_label") == "手掌 IMU（ROS Imu）"
    assert stream.get("kind") == "imu"
    assert stream.get("time_column") == "data.header.timestamp_us"
    assert stream.get("label_source") == "user_confirmed"


def test_confirm_skips_failed_items(tmp_path: Path) -> None:
    """confirm=True 时 failed 条目不落盘（闸门：验证不过不持久化）。"""
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(
        ctx,
        [_a(file="bad_imu.jsonl", kind="imu", semantic_label="坏 IMU")],
        confirm=True,
    )
    assert r["confirmed"] == []
    ctx2 = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx2, str(tmp_path / "ds"))
    stream = next(s for s in ctx2.meta["streams"]
                  if Path(s["path"]).name == "bad_imu.jsonl")
    assert stream.get("semantic_label") != "坏 IMU"


def test_confirm_mixed_batch_persists_only_valid(tmp_path: Path) -> None:
    """混合批次：strong/weak 落盘，failed 跳过。"""
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(
        ctx,
        [
            _a(file="good_imu.jsonl", kind="imu", semantic_label="好 IMU"),
            _a(file="tactile.jsonl", kind="tactile", semantic_label="触觉",
               fields=["data.fingers.0.angles.0"]),
            _a(file="bad_imu.jsonl", kind="imu", semantic_label="坏 IMU"),
        ],
        confirm=True,
    )
    assert sorted(r["confirmed"]) == ["good_imu.jsonl", "tactile.jsonl"]


# --- 3. 结构化错误 ----------------------------------------------------------


def test_no_dataset_structured_error(tmp_path: Path) -> None:
    ctx = RunContext(output_dir=str(tmp_path))
    r = propose_stream_semantics_impl(ctx, [_a(file="x.jsonl", kind="imu")])
    assert r["success"] is False and r["error"] == "no_data_loaded"


def test_empty_assumptions_structured_error(tmp_path: Path) -> None:
    ctx = _make_dataset(tmp_path)
    r = propose_stream_semantics_impl(ctx, [])
    assert r["success"] is False and r["error"] == "empty_assumptions"
