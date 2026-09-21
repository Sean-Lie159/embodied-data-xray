"""`list_tables` 工具的单元测试（多表并列机制阶段 1）。

覆盖（对应 `docs/多表并列机制设计.md` §4.1 与 §6 阶段 1 的验收标准）：

1. 表清单与 ``inspect_streams`` 的 ``table_name`` **完全一致**（命名唯一生成点）；
2. 缺省表被正确标注 ``is_default``，且与 ``meta["main_table"]["name"]`` 对齐；
3. 规模降序排序确定性；
4. 表数超上限时截断并**如实声明**（含省略条数与规模范围）；
5. 未加载数据集返回 no_data_loaded；
6. 视频流不计入表清单（口径与 ``n_table_streams`` 一致）；
7. 纯媒体数据集「无缺省表」是合法状态，不报错。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools.inspect_streams import inspect_streams_impl
from app.tools.list_tables import _MAX_TABLES, list_tables_impl
from app.tools.load_dataset import load_dataset_impl


def _make_multi(tmp_path: Path, name: str = "multi") -> Path:
    """构造目录：主表（动作列，最大）+ accel.csv + gyro.csv + tasks.csv（小）。"""
    root = tmp_path / name
    root.mkdir()
    # 主表：含动作列 → 优先作为主表，且规模最大。
    n = 40
    pd.DataFrame({
        "episode": [i // 10 for i in range(n)],
        "qpos1": [0.1 + i * 0.01 for i in range(n)],
        "qpos2": [0.2 + i * 0.01 for i in range(n)],
        "success": [1] * n,
    }).to_csv(root / "state.csv", index=False)
    # 中表：IMU。
    pd.DataFrame({
        "timestamp_ns": [1_780_000_000_000_000_000 + i * 987_000 for i in range(20)],
        "x": [0.0] * 20, "y": [0.1] * 20, "z": [9.8] * 20,
    }).to_csv(root / "accel.csv", index=False)
    # 中表：gyro（比 accel 小一点，验证排序）。
    pd.DataFrame({
        "timestamp_ns": [1_780_000_000_000_000_000 + i * 987_000 for i in range(15)],
        "wx": [0.0] * 15, "wy": [0.0] * 15,
    }).to_csv(root / "gyro.csv", index=False)
    # 小表：2 行任务清单（验证"小表也必须在清单里"）。
    pd.DataFrame({"task": ["pick", "place"]}).to_csv(root / "tasks.csv", index=False)
    return root


def _load(ctx: RunContext, root: Path) -> None:
    load_dataset_impl(ctx, str(root))


def test_names_match_inspect_streams(tmp_path: Path) -> None:
    """**关键一致性**：本工具给出的 table_name 必须与 inspect_streams 的完全一致。

    若两处口径不一致（本工具另写一套命名规则），模型会看到"清单说 A、
    调用要 B"的自相矛盾，重演 2026-09-20「清单与容错自相矛盾」事故。
    """
    root = _make_multi(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    listed = {e["table_name"] for e in list_tables_impl(ctx)["tables"]}

    # inspect_streams 的表流分散在多个桶：table_streams（动作/未知/空流）、
    # imus[*].streams（IMU）、force_channels（单槽，无 table_name 字段）。
    streams = inspect_streams_impl(ctx)
    all_streams = list(streams.get("table_streams") or [])
    for grp in streams.get("imus") or []:
        all_streams.extend(grp.get("streams") or [])
    inspected = {
        s["table_name"] for s in all_streams
        if s.get("kind") != "video" and s.get("table_name")
    }

    assert listed, "目录含 4 张 csv，表清单不应为空"
    assert listed <= inspected, (
        f"list_tables 给出的表名不在 inspect_streams 口径内：{listed - inspected}"
    )
    # 每张 csv 都应出现（含最小的 2 行 tasks.csv —— 小表/空流不得被静默丢弃）。
    assert {"state.csv", "accel.csv", "gyro.csv", "tasks.csv"} <= listed


def test_default_table_marked(tmp_path: Path) -> None:
    """缺省表被标注 is_default，且与 meta['main_table']['name'] 对齐。"""
    root = _make_multi(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    r = list_tables_impl(ctx)
    assert r["success"] is True
    assert r["dataset"] == "multi"
    # 含动作列且规模最大 → state.csv 是缺省表。
    assert r["default_table"] == "state.csv"
    defaults = [e["table_name"] for e in r["tables"] if e["is_default"]]
    assert defaults == ["state.csv"], "缺省表必须唯一且被标注"
    assert r["default_table"] in r["user_message"]


def test_sorted_by_size_desc(tmp_path: Path) -> None:
    """按规模（行×列）降序，确定性排序。"""
    root = _make_multi(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    entries = list_tables_impl(ctx)["tables"]
    sizes = [
        (e["rows"] or 0) * max(1, e["cols"] or 1) for e in entries
    ]
    assert sizes == sorted(sizes, reverse=True), "应按规模降序"


def test_entry_fields_present(tmp_path: Path) -> None:
    """每条目含模型决策所需字段（表名/规模/语义/缺省标记）。"""
    root = _make_multi(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    _load(ctx, root)

    for e in list_tables_impl(ctx)["tables"]:
        assert e["table_name"]
        assert isinstance(e["rows"], int)
        assert e["is_default"] in (True, False)
        assert "kind" in e and "status" in e


def test_no_data_loaded() -> None:
    """未加载数据集 → 结构化 no_data_loaded，不抛异常。"""
    ctx = RunContext()
    r = list_tables_impl(ctx)
    assert r["success"] is False
    assert r["error"] == "no_data_loaded"
    assert "load_dataset" in r["user_message"]


def test_truncation_declared(tmp_path: Path) -> None:
    """表数超上限 → 截断 + 如实声明（条数与规模范围）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "big"
    # 直接构造流登记表：绕过 IO，验证截断逻辑本身。
    ctx.meta["streams"] = [
        {
            "path": f"/x/t{i}.csv", "format": "csv", "kind": "unknown",
            "n_rows": (i + 1) * 10, "n_cols": 2, "status": "active",
        }
        for i in range(_MAX_TABLES + 5)
    ]
    ctx.meta["main_table"] = {"name": "t0.csv"}

    r = list_tables_impl(ctx)
    assert r["n_tables"] == _MAX_TABLES + 5
    assert r["tables_shown"] == _MAX_TABLES
    assert r["truncated"] is True
    assert r["truncation_note"]
    # 如实声明省略条数（不得静默截断）。
    assert str(_MAX_TABLES + 5 - _MAX_TABLES) in r["truncation_note"]
    # 截断按规模降序 → 前 N 条是规模最大的，最大一条必在前。
    assert r["tables"][0]["table_name"] == f"t{_MAX_TABLES + 4}.csv"


def test_video_not_listed(tmp_path: Path) -> None:
    """视频流不计入表清单（口径与 inspect_streams 的 n_table_streams 一致）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "vid"
    ctx.meta["streams"] = [
        {"path": "/x/a.csv", "format": "csv", "kind": "unknown",
         "n_rows": 10, "n_cols": 2, "status": "active"},
        {"path": "/x/cam.mp4", "format": "video", "kind": "video",
         "n_rows": 100, "n_cols": 0, "status": "active"},
    ]
    ctx.meta["main_table"] = {"name": "a.csv"}

    r = list_tables_impl(ctx)
    names = [e["table_name"] for e in r["tables"]]
    assert names == ["a.csv"], "视频不是表，不得出现在清单"


def test_media_only_dataset_no_default(tmp_path: Path) -> None:
    """纯媒体数据集：无缺省表是合法状态，不得报错或暗示"缺失"。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "media_only"
    ctx.meta["streams"] = [
        {"path": "/x/a.mp4", "format": "video", "kind": "video", "status": "active"},
    ]
    ctx.meta["main_table"] = None

    r = list_tables_impl(ctx)
    assert r["success"] is True, "纯媒体数据集不是错误"
    assert r["n_tables"] == 0
    assert r["default_table"] is None
    assert "没有缺省表" in r["user_message"]


def test_empty_stream_marked_unusable(tmp_path: Path) -> None:
    """空流仍列出（用户需知道它存在）但标注 usable=False。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "has_empty"
    ctx.meta["streams"] = [
        {"path": "/x/a.csv", "format": "csv", "kind": "unknown",
         "n_rows": 10, "n_cols": 2, "status": "active"},
        {"path": "/x/empty.csv", "format": "csv", "kind": "unknown",
         "n_rows": 0, "n_cols": 3, "status": "empty"},
    ]
    ctx.meta["main_table"] = {"name": "a.csv"}

    r = list_tables_impl(ctx)
    by_name = {e["table_name"]: e for e in r["tables"]}
    assert "empty.csv" in by_name, "空流应在清单中可见"
    assert by_name["empty.csv"]["usable"] is False
    assert by_name["a.csv"].get("usable") is not False


def test_frame_layout_exposes_n_frames(tmp_path: Path) -> None:
    """帧布局流透出帧数，让模型理解"行数 = 帧数 × 每帧行数"。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "framed"
    ctx.meta["streams"] = [
        {"path": "/x/joints.h5::state/joint", "format": "hdf5",
         "kind": "joint_state", "n_rows": 8700, "n_cols": 7,
         "status": "active", "frame_layout": True, "n_frames": 2900},
    ]
    # 单文件（h5）加载路径的真实 meta 形态：name 是**裸节点**，可用名在
    # 顶层 candidates[*].table_name（见 load_dataset.py:2613-2626）。
    ctx.meta["main_table"] = {
        "name": "state/joint",
        "candidates": [
            {"table_name": "joints::state/joint", "rows": 8700, "cols": 7},
        ],
    }

    r = list_tables_impl(ctx)
    e = r["tables"][0]
    assert e["n_frames"] == 2900
    assert e["is_default"] is True, "裸节点名须被解析为可用表名后匹配"
    assert r["default_table"] == "joints::state/joint", (
        "default_table 必须是**可用**表名，不得透出裸节点名"
    )


def test_h5_single_file_default_resolved(tmp_path: Path) -> None:
    """真实 HDF5 单文件加载：缺省表须被解析为「stem::node」可用名。

    回归（2026-09-21 实测）：`meta["main_table"]["name"]` 在单文件路径下是
    **裸节点路径**（如 ``state/joint``），照抄去调 profile_data 必失败。
    若 list_tables 直接透出它，就会给出一个"照抄必失败"的假表名——
    正是 2026-09-20「清单与调用口径自相矛盾」事故的同型缺陷。
    """
    import h5py

    p = tmp_path / "joints.h5"
    with h5py.File(p, "w") as f:
        grp = f.create_group("state")
        grp.create_dataset("joint", data=[[1.0, 2.0]] * 50)
        f.create_dataset("clock", data=[0.0] * 50)

    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(p))

    r = list_tables_impl(ctx)
    assert r["success"] is True
    names = {e["table_name"] for e in r["tables"]}
    # 可用名一律含文件 stem 前缀（resolve_table_name 的接受集）。
    assert all("::" in n for n in names), f"表名格式错误：{names}"
    assert r["default_table"] in names, (
        f"default_table {r['default_table']!r} 不在可用表名集合 {names} 内"
    )
    defaults = [e["table_name"] for e in r["tables"] if e["is_default"]]
    assert defaults == [r["default_table"]], "is_default 必须唯一且与 default_table 一致"
    # 规模从 candidates 回填（流登记项虽有 n_rows，两者应一致）。
    for e in r["tables"]:
        assert isinstance(e["rows"], int) and e["rows"] > 0


def test_registered_and_droppable() -> None:
    """工具已注册且登记了可丢字段（防漏配，见 tests/test_output_guard.py）。"""
    from app.agent.agent import _TOOL_DROPPABLE
    from app.services.chat_service import _ALL_TOOLS

    assert "list_tables" in {getattr(t, "name", None) for t in _ALL_TOOLS}
    assert "list_tables" in _TOOL_DROPPABLE
    # default_table 是结论字段，绝不能列为可丢。
    assert "default_table" not in _TOOL_DROPPABLE["list_tables"]
    assert "n_tables" not in _TOOL_DROPPABLE["list_tables"]


@pytest.mark.parametrize("bad_streams", [[], None])
def test_empty_registry_is_not_error(tmp_path: Path, bad_streams) -> None:
    """已加载数据集但流登记表为空 → 返回空清单而非报错。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "loaded_but_no_streams"
    ctx.df = pd.DataFrame({"a": [1]})
    if bad_streams is not None:
        ctx.meta["streams"] = bad_streams

    r = list_tables_impl(ctx)
    assert r["success"] is True
    assert r["n_tables"] == 0
