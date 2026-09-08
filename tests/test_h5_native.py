"""h5py 原生层级结构的加载测试（非 pandas HDFStore 格式）。

背景（用户实测）：具身智能采集的 dataset.h5 是 h5py 直接写的层级结构
（action/observation/pose/meta 各组下为 compound dtype 结构化数组），
pandas read_hdf 不认——PyTables 能打开、keys 非空但 read_hdf 全败，
此前报"未找到可读取的 DataFrame 表"。现加 h5py 原生层级回退。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools.load_dataset import _load_hdf5, load_dataset_impl


@pytest.fixture()
def envelope_h5(tmp_path: Path) -> Path:
    """构造与真实 dataset.h5 同构的 h5py 层级文件。"""
    p = tmp_path / "dataset.h5"
    n = 500
    compound = np.zeros(
        n,
        dtype=[("value", "<f4"), ("timestamp", "<f8"), ("raw", "<f4"),
               ("min", "<f4"), ("max", "<f4")],
    )
    compound["value"] = np.linspace(0, 1, n)
    compound["timestamp"] = T0 + np.arange(n) * 1e6
    compound["raw"] = np.random.default_rng(0).normal(110, 1, n)
    small = np.zeros(50, dtype=[("value", "<f4"), ("timestamp", "<f8")])
    with h5py.File(p, "w") as f:
        g = f.create_group("action/left_eef/feedback")
        g.create_dataset("motor_command", data=compound)     # 最大 → 主表
        g2 = f.create_group("action/right_eef/feedback")
        g2.create_dataset("motor_command", data=small)
        calib = f.create_group("meta/calibration/cam0")
        calib.create_dataset("intrinsic", data=np.eye(3))
        calib.create_dataset("extrinsic", data=np.eye(4))
        calib.create_dataset("camera_model", data=["pinhole"])  # object 标量
    return p


T0 = 1_787_294_445_600_000_000


def test_native_hierarchy_loads_largest_node(envelope_h5: Path) -> None:
    """h5py 原生层级：主表为信息量最大节点（500 行 compound），列名=字段名。"""
    df = _load_hdf5(str(envelope_h5))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (500, 5)
    assert list(df.columns) == ["value", "timestamp", "raw", "min", "max"]
    assert df.attrs["h5_source_node"] == "action/left_eef/feedback/motor_command"
    # 结构摘要含全部候选节点（action 右手 + 标定矩阵）。
    nodes = {s["node"] for s in df.attrs["h5_structure"]}
    assert "action/right_eef/feedback/motor_command" in nodes
    assert "meta/calibration/cam0/intrinsic" in nodes


def test_native_hierarchy_via_load_dataset(tmp_path: Path) -> None:
    """load_dataset 端到端：h5py 层级文件成功加载，返回带 h5_structure。"""
    p = tmp_path / "ds" / "dataset.h5"
    p.parent.mkdir()
    n = 100
    compound = np.zeros(
        n, dtype=[("value", "<f4"), ("timestamp", "<f8")]
    )
    compound["timestamp"] = T0 + np.arange(n) * 1e6
    with h5py.File(p, "w") as f:
        f.create_group("action/feedback").create_dataset(
            "motor_command", data=compound)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(p))
    assert r["success"] is True, r.get("reason")
    assert r["n_rows"] == n
    assert r.get("h5_source_node") == "action/feedback/motor_command"
    assert "h5_structure" in r
    assert "原生层级" in r["user_message"]
    # context.meta 也带出（供后续工具引用）。
    assert ctx.meta.get("h5_structure")


def test_pandas_hdfstore_still_works(tmp_path: Path) -> None:
    """pandas HDFStore 格式（既有支持）不受影响。"""
    p = tmp_path / "ok.h5"
    pd.DataFrame({"a": [1, 2]}).to_hdf(p, key="data", mode="w")
    df = _load_hdf5(str(p))
    assert list(df.columns) == ["a"]


def test_corrupt_h5_still_parse_failed(tmp_path: Path) -> None:
    """非 HDF5 文件：两条路径都打不开 → parse_failed（"可能损坏"措辞恰当）。"""
    p = tmp_path / "bad.h5"
    p.write_bytes(b"garbage")
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(p))
    assert r["success"] is False
    assert r["error"] == "parse_failed"
