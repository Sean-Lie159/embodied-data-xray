"""多表支持（目录内多表并存）的单元测试。

覆盖：profile_data 指定表按名惰性读取且不替换主表、缺省行为仍为主表、不存在的表
返回结构化 table_not_found；compute_stats 指定表统计。不依赖真实网络。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.compute_stats import compute_stats_impl
from app.tools.load_dataset import load_dataset_impl
from app.tools.profile_data import profile_data_impl


def _make_dir(tmp_path: Path, name: str = "multi") -> Path:
    """构造目录：主表（动作列）+ accel.csv（IMU）+ gyro.csv。"""
    root = tmp_path / name
    root.mkdir()
    # 主表：episode + qpos 动作列。
    pd.DataFrame({
        "episode": [0, 0, 0, 1, 1, 1],
        "qpos1": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        "success": [0, 0, 1, 0, 0, 1],
    }).to_csv(root / "state.csv", index=False)
    # 次表：IMU（timestamp_ns + x/y/z，1012Hz 量级）。
    ts = [1_780_000_000_000_000_000 + i * 987_000 for i in range(5)]
    pd.DataFrame({
        "timestamp_ns": ts,
        "x": [0.0, 0.1, 0.2, 0.3, 0.4],
        "y": [0.0, 0.1, 0.2, 0.3, 0.4],
        "z": [9.8, 9.8, 9.8, 9.8, 9.8],
    }).to_csv(root / "accel.csv", index=False)
    return root


def _load(ctx: RunContext, root: Path) -> None:
    load_dataset_impl(ctx, str(root))


def test_profile_explicit_table_reads_lazy_and_keeps_main(tmp_path: Path) -> None:
    """指定 table 读取 accel.csv，主表 state 不被动、context.df 仍为主表。"""
    root = _make_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    r = profile_data_impl(ctx, table="accel.csv")
    assert r["success"] is True
    assert r["table_name"] == "accel.csv"
    assert r["dataset"] == "multi"
    assert r["data_source"] == "stream_lazy"
    assert r["n_rows"] == 5
    assert "timestamp_ns" in [c["name"] for c in r["columns"]]

    # 主表状态不被替换：context.df 仍是主表 state.csv。
    assert ctx.df is not None
    assert "episode" in ctx.df.columns
    assert ctx.meta.get("main_table", {}).get("name") == "state.csv"


def test_profile_default_uses_main_table(tmp_path: Path) -> None:
    """缺省 table → 分析主表，行为不变。"""
    root = _make_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    r = profile_data_impl(ctx)
    assert r["success"] is True
    assert r["table_name"] == "state.csv"
    assert r["n_rows"] == 6
    assert "episode" in [c["name"] for c in r["columns"]]


def test_profile_nonexistent_table_structured_error(tmp_path: Path) -> None:
    """不存在的表名 → 结构化 table_not_found，不抛异常。"""
    root = _make_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    r = profile_data_impl(ctx, table="missing.csv")
    assert r["success"] is False
    assert r["error"] == "table_not_found"
    assert "missing.csv" in r["reason"]
    assert "不存在" in r["user_message"]


def test_profile_table_not_found_without_loading(tmp_path: Path) -> None:
    """未加载任何数据集 + 指定表 → 也应结构化失败（主表未加载优先）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    r = profile_data_impl(ctx, table="x.csv")
    assert r["success"] is False
    assert r["error"] == "no_data_loaded"


def test_compute_stats_explicit_table(tmp_path: Path) -> None:
    """compute_stats 指定 accel.csv 统计（列存在但无动作语义时给出结构化结果）。"""
    root = _make_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    r = compute_stats_impl(ctx, table="accel.csv")
    assert r["success"] is True
    assert r["table_name"] == "accel.csv"
    assert r["dataset"] == "multi"
    # 次表无 episode/success → 整段视为一个 episode，仍是成功统计。
    assert r["metrics"]["episode_distribution"]["n_episodes"] == 1
    assert "x" in r["stats"]


def test_compute_stats_nonexistent_table(tmp_path: Path) -> None:
    """compute_stats 指定不存在的表 → table_not_found。"""
    root = _make_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    r = compute_stats_impl(ctx, table="nope.csv")
    assert r["success"] is False
    assert r["error"] == "table_not_found"
