"""h5 节点登记为流（多表切换分析）的单元测试。

背景（用户需求）：h5 是多表文件（action/observation/pose/meta 各组），
主表只装信息量最大节点，其余节点不可分析。现把全部候选节点登记为流
（复用目录多流机制），表名 "<stem>::<node>"，resolve_table_name 按名
惰性读取——"一个 h5 内多表切换分析"。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from app.agent.context import RunContext
from app.tools._data_access import resolve_table_name
from app.tools.load_dataset import (
    _classify_h5_node,
    _list_hdf5_native_nodes,
    load_dataset_impl,
)

T0 = 1_787_294_445_600_000_000


def _make_envelope_h5(p: Path) -> Path:
    """构造与真实 dataset.h5 同构的层级文件。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 500
    action = np.zeros(
        n, dtype=[("value", "<f4"), ("timestamp", "<f8"), ("raw", "<f4"),
                  ("min", "<f4"), ("max", "<f4")])
    action["timestamp"] = T0 + np.arange(n) * 1e6
    action["value"] = np.linspace(0, 1, n)
    small = np.zeros(50, dtype=[("value", "<f4"), ("timestamp", "<f8")])
    pose = np.zeros(
        400, dtype=[("value", "<f4", (7,)), ("timestamp", "<f8"),
                    ("confidence", "<f4")])
    pose["timestamp"] = T0 + np.arange(400) * 2_500_000
    with h5py.File(p, "w") as f:
        f.create_group("action/left_eef/feedback").create_dataset(
            "motor_command", data=action)
        f.create_group("action/right_eef/feedback").create_dataset(
            "motor_command", data=small)
        f.create_group("pose/ldl_hand_fisheye/feedback").create_dataset(
            "pose_in_chest", data=pose)
        f.create_group("meta/calibration/cam0").create_dataset(
            "intrinsic", data=np.eye(3))
        # file_path 用定长字符串（h5py 不支持 object dtype）。
        cam = np.zeros(
            60, dtype=[("file_path", "S64"), ("timestamp", "<f8")])
        cam["file_path"] = [f"cam0/{j:04d}.jpg".encode() for j in range(60)]
        cam["timestamp"] = T0 + np.arange(60) * 33_000_000
        f.create_group("observation/camera/cam0").create_dataset(
            "frames", data=cam)
    return p


@pytest.fixture()
def envelope_h5(tmp_path: Path) -> Path:
    return _make_envelope_h5(tmp_path / "ds" / "dataset.h5")


def _load(h5_path: Path) -> RunContext:
    ctx = RunContext(output_dir=str(h5_path.parent))
    assert load_dataset_impl(ctx, str(h5_path))["success"] is True
    return ctx


# --- 1. 节点清单与 kind 判定 ------------------------------------------------


def test_list_nodes_sorted_by_information(envelope_h5: Path) -> None:
    """节点清单按 行×列 降序（主表在首）。"""
    nodes = _list_hdf5_native_nodes(str(envelope_h5))
    assert nodes[0]["node"] == "action/left_eef/feedback/motor_command"
    scores = [n["rows"] * n["cols"] for n in nodes]
    assert scores == sorted(scores, reverse=True)


def test_classify_h5_node_kinds() -> None:
    """kind 判定：action 字段→actions、相机路径→frame_index、pose 路径→pose、
    calibration 路径→calibration、其余 unknown。"""
    assert _classify_h5_node(
        ["value", "timestamp", "raw"], "action/left_eef/feedback/motor_command"
    )[0] == "actions"
    assert _classify_h5_node(
        ["file_path", "timestamp"], "observation/camera/cam0"
    )[0] == "frame_index"
    assert _classify_h5_node(
        ["value", "timestamp"], "pose/cam0/feedback/pose_in_chest"
    )[0] == "pose"
    assert _classify_h5_node([], "meta/calibration/cam0/intrinsic")[0] == "calibration"
    assert _classify_h5_node(["x"], "misc/node")[0] == "unknown"


# --- 2. 加载后节点登记为流 --------------------------------------------------


def test_h5_nodes_registered_as_streams(envelope_h5: Path) -> None:
    """加载后全部候选节点登记为流（含 is_main 标记与特征 kind）。"""
    ctx = _load(envelope_h5)
    streams = ctx.meta["streams"]
    kinds = {s["kind"] for s in streams}
    assert {"actions", "pose", "frame_index", "calibration"} <= kinds
    mains = [s for s in streams if s.get("is_main")]
    assert len(mains) == 1
    assert mains[0]["path"].endswith("::action/left_eef/feedback/motor_command")
    assert all("::" in s["path"] for s in streams)


def test_resolve_h5_node_by_node_name(envelope_h5: Path) -> None:
    """按节点路径读非主表流（惰性，不替换主表）。"""
    ctx = _load(envelope_h5)
    r = resolve_table_name(ctx, "action/right_eef/feedback/motor_command")
    assert r["success"] is True and r["source"] == "h5_node"
    assert r["df"].shape == (50, 2)
    assert ctx.df.shape == (500, 5)  # 主表不被替换


def test_resolve_h5_node_by_display_name(envelope_h5: Path) -> None:
    """按显示名（<stem>::<node>）读取。"""
    ctx = _load(envelope_h5)
    r = resolve_table_name(ctx, "dataset::action/right_eef/feedback/motor_command")
    assert r["success"] is True
    assert r["df"].shape == (50, 2)


def test_resolve_h5_node_missing_structured(envelope_h5: Path) -> None:
    """不存在的节点 → 结构化 table_not_found。"""
    ctx = _load(envelope_h5)
    r = resolve_table_name(ctx, "no/such/node")
    assert r["success"] is False and r["error"] == "table_not_found"


def test_resolve_h5_node_with_expand(envelope_h5: Path) -> None:
    """expand 对 h5 节点流同样可用。"""
    ctx = _load(envelope_h5)
    r = resolve_table_name(ctx, "pose/ldl_hand_fisheye/feedback/pose_in_chest",
                           expand=True)
    assert r["success"] is True, (r.get("error"), r.get("reason"))
    assert any("value" in c for c in r["df"].columns)
