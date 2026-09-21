"""`resolve_table_name` 错误路径与缺省表名解析的测试（多表并列机制阶段 2）。

覆盖（对应 `docs/多表并列机制设计.md` §4.2 与 §6 阶段 2 的验收标准）：

1. **缺省表名必须是可用名**——单文件（h5/mcap）加载路径下
   ``meta["main_table"]["name"]`` 是裸节点路径，直接透出会得到"照抄必失败"的
   假表名（2026-09-21 实测的既有缺陷）；
2. 表不存在时返回**完整可用清单**（不再只有 3 个示例）+ 缺省表，使模型能自我纠正；
3. 清单超上限时截断并如实声明；
4. **仍不做模糊/子串猜表**（纪律不变，防误命中）；
5. 视频不算表，不进候选清单；
6. 目录型加载路径的缺省表名与候选清单同样正确。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools._data_access import (
    _MAX_AVAILABLE_TABLES,
    main_table_candidates,
    resolve_default_table_name,
    resolve_table_name,
)
from app.tools.load_dataset import load_dataset_impl


def _make_dir(tmp_path: Path, name: str = "multi") -> Path:
    """目录型数据集：主表 state.csv + accel.csv + gyro.csv。"""
    root = tmp_path / name
    root.mkdir()
    n = 30
    pd.DataFrame({
        "episode": [i // 10 for i in range(n)],
        "qpos1": [0.1 + i * 0.01 for i in range(n)],
        "success": [1] * n,
    }).to_csv(root / "state.csv", index=False)
    pd.DataFrame({"x": [0.0] * 12, "y": [0.1] * 12}).to_csv(root / "accel.csv", index=False)
    pd.DataFrame({"wx": [0.0] * 8, "wy": [0.0] * 8}).to_csv(root / "gyro.csv", index=False)
    return root


def _make_h5(tmp_path: Path, name: str = "joints.h5") -> Path:
    import h5py

    p = tmp_path / name
    with h5py.File(p, "w") as f:
        grp = f.create_group("state")
        grp.create_dataset("joint", data=[[1.0, 2.0]] * 50)
        f.create_dataset("clock", data=[0.0] * 50)
    return p


# ---------------------------------------------------------------- 缺省表名解析


def test_h5_default_name_is_usable(tmp_path: Path) -> None:
    """**核心回归**：h5 单文件加载的缺省表名必须是可用名（含 stem::node）。

    修复前：``meta["main_table"]["name"] == "state/joint"``（裸节点），
    ``resolve_table_name(None)["table_name"]`` 原样透出它——该名**无法回用**
    （用于 table 参数必失败），并随各工具的结果标注扩散。
    """
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_h5(tmp_path)))

    name = resolve_default_table_name(ctx)
    assert name == "joints::state/joint", f"应为可用名，实际 {name!r}"
    assert "::" in name, "可用名必须含文件 stem 前缀"

    r = resolve_table_name(ctx, None)
    assert r["success"] is True
    assert r["table_name"] == "joints::state/joint"
    # 该名必须真的能回用（闭环）。
    again = resolve_table_name(ctx, r["table_name"])
    assert again["success"] is True
    assert again["table_name"] == r["table_name"]


def test_dir_default_name_usable(tmp_path: Path) -> None:
    """目录型加载：缺省表名为文件名，且可回用。"""
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    name = resolve_default_table_name(ctx)
    assert name == "state.csv"
    assert resolve_table_name(ctx, name)["success"] is True


def test_no_main_table_returns_none() -> None:
    """纯媒体数据集：无缺省表 → None，不得编造。"""
    ctx = RunContext()
    ctx.dataset_id = "media_only"
    ctx.meta["main_table"] = None
    assert resolve_default_table_name(ctx) is None


def test_default_name_missing_returns_none() -> None:
    """main_table 存在但无 name → None（诚实返回，不猜）。"""
    ctx = RunContext()
    ctx.meta["main_table"] = {"reason": "无候选"}
    assert resolve_default_table_name(ctx) is None


# ---------------------------------------------------------------- 候选清单取用


def test_candidates_both_paths(tmp_path: Path) -> None:
    """main_table_candidates 统一两条加载路径的位置差异。"""
    ctx_dir = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx_dir, str(_make_dir(tmp_path)))
    names_dir = {
        (c.get("table_name") or c.get("name")) for c in main_table_candidates(ctx_dir)
    }
    assert {"state.csv", "accel.csv", "gyro.csv"} <= names_dir

    ctx_h5 = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx_h5, str(_make_h5(tmp_path)))
    names_h5 = {
        (c.get("table_name") or c.get("name")) for c in main_table_candidates(ctx_h5)
    }
    assert names_h5, "h5 单文件应有候选"
    assert all("::" in n for n in names_h5), f"候选名应为可用名：{names_h5}"


# ---------------------------------------------------------------- 错误路径增强


def test_not_found_lists_all_available(tmp_path: Path) -> None:
    """表不存在 → 返回**完整**可用清单与缺省表（不只 3 个示例）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    r = resolve_table_name(ctx, "does_not_exist.csv")
    assert r["success"] is False
    assert r["error"] == "table_not_found"
    assert set(r["available_tables"]) >= {"state.csv", "accel.csv", "gyro.csv"}
    assert r["default_table"] == "state.csv"
    assert r["available_tables_truncated"] is False
    # 向后兼容：available_examples 仍存在（老调用方/老测试不破）。
    assert isinstance(r["available_examples"], list)
    # user_message 给出可直接使用的信息。
    assert "does_not_exist.csv" in r["user_message"]
    assert "state.csv" in r["user_message"]
    assert "list_tables" in r["user_message"]


def test_not_found_truncates_large_list(tmp_path: Path) -> None:
    """可用表超上限 → 截断但如实声明总数。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "big"
    ctx.df = pd.DataFrame({"a": [1]})
    ctx.meta["streams"] = [
        {"path": f"/x/t{i}.csv", "format": "csv", "kind": "unknown"}
        for i in range(_MAX_AVAILABLE_TABLES + 7)
    ]

    r = resolve_table_name(ctx, "nope.csv")
    assert len(r["available_tables"]) == _MAX_AVAILABLE_TABLES
    assert r["available_tables_truncated"] is True
    # 如实声明总条数（不得让模型以为总数就是清单长度）。
    assert str(_MAX_AVAILABLE_TABLES + 7) in r["user_message"]


def test_no_fuzzy_matching(tmp_path: Path) -> None:
    """**纪律回归**：不做模糊/子串猜表（防误命中）。

    给清单是为了让模型自我纠正，不是替模型做匹配。子串能命中的写法
    （如只写 "state"）必须仍然失败——否则会静默分析到非预期的表。
    """
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    for fuzzy in ("state", "state.", "accel", "*.csv"):
        r = resolve_table_name(ctx, fuzzy)
        assert r["success"] is False, f"{fuzzy!r} 不应被模糊命中"
        assert r["error"] == "table_not_found"


def test_video_excluded_from_candidates(tmp_path: Path) -> None:
    """视频不是表：不进 available_tables 候选清单。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "vid"
    ctx.df = pd.DataFrame({"a": [1]})
    ctx.meta["streams"] = [
        {"path": "/x/a.csv", "format": "csv", "kind": "unknown"},
        {"path": "/x/cam.mp4", "format": "video", "kind": "video"},
    ]

    r = resolve_table_name(ctx, "nope.csv")
    assert r["available_tables"] == ["a.csv"]
    assert not any("mp4" in n for n in r["available_tables"])


def test_duplicate_streams_deduped(tmp_path: Path) -> None:
    """同一 path 重复登记时清单去重（保持原顺序）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "dup"
    ctx.df = pd.DataFrame({"a": [1]})
    ctx.meta["streams"] = [
        {"path": "/x/a.csv", "format": "csv", "kind": "unknown"},
        {"path": "/x/a.csv", "format": "csv", "kind": "unknown"},
        {"path": "/x/b.csv", "format": "csv", "kind": "unknown"},
    ]

    r = resolve_table_name(ctx, "nope.csv")
    assert r["available_tables"] == ["a.csv", "b.csv"]


def test_no_data_loaded_unchanged() -> None:
    """未加载数据集仍优先返回 no_data_loaded（不被新增字段破坏）。"""
    ctx = RunContext()
    r = resolve_table_name(ctx, "anything.csv")
    assert r["error"] == "no_data_loaded"
    assert "available_tables" not in r


@pytest.mark.parametrize("fmt,path,expected", [
    ("csv", "/x/data.csv", "data.csv"),
    ("json", "/x/data.json", "data.json"),
    ("parquet", "/x/data.parquet", "data.parquet"),
])
def test_independent_file_names(tmp_path: Path, fmt, path, expected) -> None:
    """独立文件流：候选名 = 文件名（含扩展名）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "files"
    ctx.df = pd.DataFrame({"a": [1]})
    ctx.meta["streams"] = [{"path": path, "format": fmt, "kind": "unknown"}]

    r = resolve_table_name(ctx, "nope.csv")
    assert r["available_tables"] == [expected]
