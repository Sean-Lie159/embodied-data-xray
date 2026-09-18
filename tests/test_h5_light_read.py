"""h5 轻量读取（只取列名/单字段）的性能与正确性测试（2026-09-18）。

背景（用户反馈"工具层也有耗时"）：h5 目录数据集（513MB、每帧一组、14135 帧、
27 节点）上实测 ``inspect_streams`` 106 秒、``check_temporal_sync`` 93 秒。

根因：为取**一列**时间戳或**列名**，实现却把 14135 个帧组的**全部字段**读出来
并拼成 DataFrame（单节点约 4.2 秒 × 27 节点）。而该数据集的节点**全都没有时间戳
列**——那 113 秒全花在"逐个确认没有时间戳"上。

修复：
- ``read_hdf5_nodes_metadata``：只读首帧 dtype/shape，一次遍历覆盖全部节点；
- ``read_hdf5_node_field_fast``：只读目标字段在各帧的值，不建表；
- ``_H5Reader.columns/react`` 改走上述接口（此前 ``frame(limit=1)`` 只截取结果，
  内部仍读全量）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from app.tools.load_dataset import (
    _FRAME_LAYOUT_MIN_GROUPS,
    read_hdf5_node,
    read_hdf5_node_field_fast,
    read_hdf5_nodes_metadata,
)

# app.tools 包的 __init__ 把同名 FunctionTool 暴露为包属性，须经 sys.modules 取模块。
import app.tools._readers  # noqa: F401
import app.tools.load_dataset  # noqa: F401

_readers = sys.modules["app.tools._readers"]


@pytest.fixture()
def frame_h5(tmp_path: Path) -> Path:
    """每帧一组的 h5：1D 向量字段 + 2D 多行字段。"""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "aligned.h5"
    n = _FRAME_LAYOUT_MIN_GROUPS + 10
    with h5py.File(path, "w") as f:
        for i in range(n):
            g = f.create_group(str(i))
            # 1D 向量 (3,)：一帧一条观测 → 每帧 1 行、3 列。
            g.create_dataset("action/joint/position",
                             data=np.array([i, i + 1, i + 2], dtype=float))
            # 2D (2,4)：每帧两条观测 → 每帧 2 行、4 列。
            g.create_dataset("state/end/orientation",
                             data=np.array([[i, i + 1, i + 2, i + 3],
                                            [i + 10, i + 11, i + 12, i + 13]],
                                           dtype=float))
    return path


# --- 1. 元信息批量读取 ------------------------------------------------------


def test_metadata_all_nodes_in_one_pass(frame_h5: Path) -> None:
    """一次调用取齐多个节点的列名与形状。"""
    meta = read_hdf5_nodes_metadata(
        str(frame_h5), ["action/joint/position", "state/end/orientation"])
    assert set(meta) == {"action/joint/position", "state/end/orientation"}
    assert meta["action/joint/position"]["columns"] == [
        "position_0", "position_1", "position_2"]
    assert meta["state/end/orientation"]["columns"] == [
        "orientation_0", "orientation_1", "orientation_2", "orientation_3"]


def test_metadata_rows_count_1d_vs_2d(frame_h5: Path) -> None:
    """行数口径：1D 每帧 1 行、2D 每帧 R 行（严防少算一半）。"""
    n = _FRAME_LAYOUT_MIN_GROUPS + 10
    meta = read_hdf5_nodes_metadata(
        str(frame_h5), ["action/joint/position", "state/end/orientation"])
    assert meta["action/joint/position"]["shape"][0] == n
    assert meta["state/end/orientation"]["shape"][0] == 2 * n


def test_metadata_unknown_node_absent(frame_h5: Path) -> None:
    """不存在的节点不出现在结果里（不抛异常）。"""
    meta = read_hdf5_nodes_metadata(str(frame_h5), ["no/such/node"])
    assert meta == {}


def test_metadata_dedups_nodes(frame_h5: Path) -> None:
    """重复节点名去重（避免重复工作）。"""
    meta = read_hdf5_nodes_metadata(
        str(frame_h5), ["action/joint/position"] * 3)
    assert len(meta) == 1


# --- 2. 单字段轻量读取（正确性）-------------------------------------------


def test_fast_field_matches_full_read_1d(frame_h5: Path) -> None:
    """**正确性核心**：1D 字段的轻量读取与全量路径逐值一致。"""
    full = read_hdf5_node(str(frame_h5), "action/joint/position")
    assert full is not None
    fast = read_hdf5_node_field_fast(
        str(frame_h5), "action/joint/position", "position_1")
    assert fast is not None
    assert fast.shape == full["position_1"].shape
    assert np.allclose(fast, np.asarray(full["position_1"], dtype=float))


def test_fast_field_matches_full_read_2d(frame_h5: Path) -> None:
    """**正确性核心（真实缺陷回归）**：2D 字段按"列索引"取全部行。

    踩坑记录：早期实现按**扁平化索引**取值——(2,4) 扁平化后第 0 个元素只对应
    (0,0)，丢掉 (1,0)，该列行数少一半（实测 28270 → 14135）。必须取整列。
    """
    full = read_hdf5_node(str(frame_h5), "state/end/orientation")
    assert full is not None
    n = _FRAME_LAYOUT_MIN_GROUPS + 10
    assert full.shape[0] == 2 * n  # 全量路径：每帧 2 行
    for col in ("orientation_0", "orientation_3"):
        fast = read_hdf5_node_field_fast(
            str(frame_h5), "state/end/orientation", col)
        assert fast is not None, col
        assert fast.shape[0] == 2 * n, f"{col} 行数少算（扁平化索引缺陷）"
        assert np.allclose(fast, np.asarray(full[col], dtype=float)), col


def test_fast_field_out_of_range_returns_none(frame_h5: Path) -> None:
    """列索引越界返回 None（不抛异常、不返回错误数据）。"""
    assert read_hdf5_node_field_fast(
        str(frame_h5), "action/joint/position", "position_99") is None


def test_fast_field_missing_node_returns_none(frame_h5: Path) -> None:
    """节点不存在返回 None。"""
    assert read_hdf5_node_field_fast(str(frame_h5), "no/such", "x_0") is None


# --- 3. Reader 接口走轻量路径 ---------------------------------------------


def test_h5_reader_columns_is_light(frame_h5: Path) -> None:
    """``_H5Reader.columns`` 不构建整表（返回列名即成功）。"""
    reader = _readers.get_reader("h5")
    assert reader is not None
    cols = reader.columns(str(frame_h5), sub="action/joint/position")
    assert cols == ["position_0", "position_1", "position_2"]


def test_h5_reader_nrows_no_data_read(frame_h5: Path) -> None:
    """``nrows`` 走元信息（不读数据内容）。"""
    reader = _readers.get_reader("h5")
    assert reader is not None
    n = _FRAME_LAYOUT_MIN_GROUPS + 10
    assert reader.nrows(str(frame_h5), sub="action/joint/position") == n
    assert reader.nrows(str(frame_h5), sub="state/end/orientation") == 2 * n


def test_h5_reader_all_columns_batch(frame_h5: Path) -> None:
    """读取器提供批量列名接口（inspect_streams 复用它的前提）。"""
    reader = _readers.get_reader("h5")
    assert reader is not None
    assert hasattr(reader, "all_columns")
    got = reader.all_columns(
        str(frame_h5), ["action/joint/position", "state/end/orientation"])
    assert len(got) == 2
    assert got["action/joint/position"] == [
        "position_0", "position_1", "position_2"]


def test_h5_reader_columns_without_sub_returns_none(frame_h5: Path) -> None:
    """无 sub（非容器子流）时不报错，返回 None。"""
    reader = _readers.get_reader("h5")
    assert reader is not None
    assert reader.columns(str(frame_h5), sub=None) is None


# --- 4. 性能红线（守护"不再全量读"）--------------------------------------


def test_metadata_does_not_read_dataset_values(frame_h5: Path, monkeypatch) -> None:
    """**性能红线**：元信息读取不得触碰数据集**值**（只读 dtype/shape）。

    做法：把 h5py.Dataset.__getitem__ 替换为抛异常——若实现仍读数据值，
    测试立即失败。这是防止回退到"读全量再截取"的硬约束。
    """
    h5py = pytest.importorskip("h5py")
    orig = h5py.Dataset.__getitem__

    def _boom(self, key):  # noqa: ANN001
        raise AssertionError(
            "元信息读取触碰了数据集值（应只读 dtype/shape，否则会退化为全量读）"
        )

    monkeypatch.setattr(h5py.Dataset, "__getitem__", _boom)
    try:
        meta = read_hdf5_nodes_metadata(
            str(frame_h5), ["action/joint/position", "state/end/orientation"])
    finally:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", orig)
    assert len(meta) == 2


def test_reader_columns_does_not_read_values(frame_h5: Path, monkeypatch) -> None:
    """``columns`` 同样不得读数据集值。"""
    h5py = pytest.importorskip("h5py")
    orig = h5py.Dataset.__getitem__

    def _boom(self, key):  # noqa: ANN001
        raise AssertionError("columns 触碰了数据集值（应走元信息接口）")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", _boom)
    try:
        reader = _readers.get_reader("h5")
        cols = reader.columns(str(frame_h5), sub="action/joint/position")
    finally:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", orig)
    assert cols == ["position_0", "position_1", "position_2"]
