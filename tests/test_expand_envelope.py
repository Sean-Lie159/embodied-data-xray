"""展开视图（expand_envelope / resolve(expand) / sanity+profile 接入）的单元测试。

背景：MCAP 信封流的数值信号嵌套在 data 内，sanity 曾全部 skipped（"无法读取
IMU 数值列"）。展开视图把嵌套数值变成扁平列（只读新 DataFrame），使深层
检查可执行；主表 context.df 语义不受影响。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools._data_access import expand_envelope, resolve_table_name
from app.tools.check_sensor_sanity import check_sensor_sanity_impl
from app.tools.load_dataset import load_dataset_impl
from app.tools.profile_data import profile_data_impl

T0_US = 1_787_294_445_600_000


def _imu_rows(n: int = 2000, seed: int = 7) -> list[dict]:
    """静止 IMU 信封行：模长≈9.8 带微噪声（非恒定，避免恒定故障告警）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "mcap_log_time_ns": (T0_US + i * 1000) * 1000,
            "mcap_publish_time_ns": (T0_US + i * 1000) * 1000,
            "data": {
                "header": {"timestamp_us": T0_US + i * 1000},
                "linear_acceleration": {
                    "x": round(float(rng.normal(0, 0.01)), 5),
                    "y": round(float(rng.normal(0, 0.01)), 5),
                    "z": round(9.8 + float(rng.normal(0, 0.01)), 5),
                },
                "angular_velocity": {"x": 0.001, "y": 0.002, "z": 0.003},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        })
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- 1. expand_envelope 核心 ------------------------------------------------


def test_expand_dict_and_nested(tmp_path: Path) -> None:
    """dict 递归展开为点分列；list_of_dict 按下标展开。"""
    df = pd.DataFrame({
        "t": [1, 2],
        "data": [
            {"a": {"b": 1}, "fingers": [{"angle": 0.1}, {"angle": 0.2}]},
            {"a": {"b": 2}, "fingers": [{"angle": 0.3}, {"angle": 0.4}]},
        ],
    })
    out, note = expand_envelope(df)
    assert note is None
    for col in ("data.a.b", "data.fingers.0.angle", "data.fingers.1.angle"):
        assert col in out.columns, f"缺展开列 {col}"
    assert list(out["data.a.b"]) == [1, 2]
    assert list(out["data.fingers.1.angle"]) == [0.2, 0.4]
    # 原 object 列被展开列取代（不再保留巨型 object 列）。
    assert "data" not in out.columns
    # 原 df 不被修改。
    assert "data" in df.columns


def test_expand_scalar_list(tmp_path: Path) -> None:
    """数值 list 展开为 col.0..col.N。"""
    df = pd.DataFrame({"v": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]})
    out, _ = expand_envelope(df)
    for i in range(3):
        assert f"v.{i}" in out.columns


def test_max_cols_truncation_note() -> None:
    """达到 max_cols → 部分展开说明非空。"""
    df = pd.DataFrame({
        "d": [{f"k{i}": i for i in range(20)} for _ in range(3)],
    })
    out, note = expand_envelope(df, max_cols=5)
    assert note is not None and "部分展开" in note
    assert len([c for c in out.columns if c.startswith("d.")]) == 5


def test_no_envelope_columns_passthrough() -> None:
    """纯数值 df：原样返回（副本），note=None。"""
    df = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    out, note = expand_envelope(df)
    assert note is None
    assert list(out.columns) == ["a", "b"]


# --- 2. resolve_table_name(expand=) -----------------------------------------


def _load_envelope_dataset(tmp_path: Path) -> RunContext:
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / "imu.jsonl", _imu_rows(300))
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    return ctx


def test_resolve_expand_returns_flat_columns(tmp_path: Path) -> None:
    """resolve(expand=True) → 展开列；context.df（主表）不受影响。"""
    ctx = _load_envelope_dataset(tmp_path)
    r = resolve_table_name(ctx, "imu.jsonl", expand=True)
    assert r["success"] is True and r.get("expanded") is True
    df = r["df"]
    assert "data.linear_acceleration.z" in df.columns
    assert "data" not in df.columns
    # 主表不受影响：ctx.df 仍是信封原样（原始 data 列在、展开列不在）。
    assert ctx.df is not None
    assert "data" in ctx.df.columns
    assert "data.linear_acceleration.z" not in ctx.df.columns


def test_resolve_no_expand_keeps_envelope(tmp_path: Path) -> None:
    """默认（expand=False）保持信封原样（向后兼容）。"""
    ctx = _load_envelope_dataset(tmp_path)
    r = resolve_table_name(ctx, "imu.jsonl")
    assert r["success"] is True
    assert "data" in r["df"].columns
    assert "expanded" not in r


# --- 3. sanity 接入：嵌套 IMU 真实检查 --------------------------------------


def test_sanity_expanded_imu_runs_gravity(tmp_path: Path) -> None:
    """sanity(expand=True, table=信封流) → 重力检查真实执行且 pass。"""
    ctx = _load_envelope_dataset(tmp_path)
    s = check_sensor_sanity_impl(ctx, table="imu.jsonl", expand=True)
    assert s["success"] is True, s.get("reason")
    # 不再出现"无法读取 IMU 数值列"的 skipped。
    skipped_text = json.dumps(s.get("skipped_checks", {}), ensure_ascii=False)
    assert "无法读取 IMU 数值列" not in skipped_text
    # 重力检查执行且通过（模长 ≈9.8 m/s²）。
    checks = s.get("checks", {})
    gravity_done = False
    for v in checks.values():
        if not isinstance(v, dict):
            continue  # 部分检查项为 list（如逐通道明细）
        g = v.get("gravity_check") or {}
        if g.get("status") == "done":
            gravity_done = True
            assert g.get("verdict") == "pass", g
    assert gravity_done, f"应有真实执行的重力检查：{json.dumps(checks, ensure_ascii=False)[:400]}"


def test_sanity_expand_relaxes_imu_candidates(tmp_path: Path) -> None:
    """kind=unknown 的信封流：expand=True 时凭 signal_fields 进 IMU 候选。"""
    ctx = _load_envelope_dataset(tmp_path)
    # 不指定 table（显式 table 分支不触发）——expand 候选放宽生效。
    s = check_sensor_sanity_impl(ctx, expand=True)
    assert s["success"] is True, s.get("reason")
    skipped_text = json.dumps(s.get("skipped_checks", {}), ensure_ascii=False)
    assert "无法读取 IMU 数值列" not in skipped_text


def test_sanity_no_expand_keeps_old_behavior(tmp_path: Path) -> None:
    """默认（expand=False）保持旧行为：信封流被 skipped（向后兼容）。"""
    ctx = _load_envelope_dataset(tmp_path)
    s = check_sensor_sanity_impl(ctx, table="imu.jsonl")
    skipped_text = json.dumps(s.get("skipped_checks", {}), ensure_ascii=False)
    assert "无法读取 IMU 数值列" in skipped_text


# --- 4. profile_data 接入 ---------------------------------------------------


def test_profile_expand_surfaces_nested_columns(tmp_path: Path) -> None:
    """profile(expand=True) → 列概况直达嵌套字段。"""
    ctx = _load_envelope_dataset(tmp_path)
    r = profile_data_impl(ctx, table="imu.jsonl", expand=True)
    assert r["success"] is True
    names = {c["name"] for c in r["columns"]}
    assert "data.linear_acceleration.z" in names
    assert "data" not in names
