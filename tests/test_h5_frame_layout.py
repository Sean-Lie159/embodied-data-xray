"""h5"每帧一组"布局的识别、合并与读取测试（2026-09-14 卡死事故）。

背景（用户实测）：加载目录 ``2655849``（含 513MB 的 aligned_joints.h5）时
**界面卡死**。该 h5 的结构是 14135 个数字命名的顶层组（0/、1/、2/…），
每组下是 action/end/orientation 等 6 个叶子数据集——即"每帧一组"布局。

此前的实现把**每个叶子数据集**登记为一条独立流 → **84,810 条流**，并带来：
  1. `context.meta` 常驻 3600 万字符（估算 1200 万 token）；
  2. `inspect_streams` 返回 4718 万字符，护栏为测量体积反复序列化（实测 14.5s）；
  3. `_list_hdf5_native_nodes` 全量遍历 88 万节点耗时约 57 秒；
  4. 用户侧表现为"卡死"（实为纯计算耗时数十秒到分钟级）。

修复后：31 条流、加载 1.36 秒，且合并流可被正常读取与分析。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.agent.context import RunContext

# 取模块对象：`app.tools` 包的 __init__ 把名为 load_dataset 的 FunctionTool
# 暴露为**包属性**，故 `from app.tools import load_dataset` 或
# `import app.tools.load_dataset as X` 拿到的都是工具对象而非模块。
# 必须经 sys.modules 取真实模块（monkeypatch 内部函数也需如此）。
import sys

import app.tools.load_dataset  # noqa: F401  确保模块已加载

_ld = sys.modules["app.tools.load_dataset"]

_FRAME_LAYOUT_MIN_GROUPS = _ld._FRAME_LAYOUT_MIN_GROUPS
_MAX_STREAMS_PER_CONTAINER = _ld._MAX_STREAMS_PER_CONTAINER
_is_frame_number = _ld._is_frame_number
_list_hdf5_native_nodes = _ld._list_hdf5_native_nodes
_merge_frame_layout = _ld._merge_frame_layout
load_dataset_impl = _ld.load_dataset_impl
read_hdf5_node = _ld.read_hdf5_node
register_h5_node_streams = _ld.register_h5_node_streams


@pytest.fixture()
def frame_layout_h5(tmp_path: Path) -> Path:
    """构造一个"每帧一组"布局的 h5（N 帧 × 2 字段）。"""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "aligned_joints.h5"
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    with h5py.File(path, "w") as f:
        for i in range(n_frames):
            g = f.create_group(str(i))
            # position 是 1D 向量 (3,)：一帧一条向量观测（→ 1 行 × 3 列）。
            g.create_dataset("action/end/position", data=np.array([i, i + 1, i + 2],
                                                                  dtype=float))
            # orientation 是 2D (2,4)：一帧两条观测（→ 2 行 × 4 列）。
            g.create_dataset("state/end/orientation",
                             data=np.array([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0]], dtype=float))
    return path


# --- 1. 帧号判定 ------------------------------------------------------------


@pytest.mark.parametrize("name,expect", [
    ("0", True), ("14134", True), ("007", True),
    ("action", False), ("frame_0", False), ("", False), ("1.5", False),
])
def test_is_frame_number(name: str, expect: bool) -> None:
    """帧号判定为"纯数字名"（含前导零），其余不算。"""
    assert _is_frame_number(name) is expect


# --- 2. 布局识别与合并 ------------------------------------------------------


def test_frame_layout_is_merged(frame_layout_h5: Path) -> None:
    """**核心回归**：每帧一组的布局被合并为少量流，而非展开成 N 万条。"""
    nodes = _list_hdf5_native_nodes(str(frame_layout_h5))
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    # 原始叶子节点数是 n_frames * 2（两个字段）；合并后应为 2 条。
    assert len(nodes) == 2, f"未合并：{len(nodes)} 条（原始叶子数 {n_frames * 2}）"
    assert all(n.get("frame_layout") for n in nodes)
    assert all(n.get("n_frames") == n_frames for n in nodes)
    # 路径不含帧号前缀（干净相对路径）。
    assert all(not _is_frame_number(str(n["node"]).split("/")[0]) for n in nodes)


def test_merged_rows_multiply_by_frames(frame_layout_h5: Path) -> None:
    """合并后行数 = 单帧行数 × 帧数（统计口径正确）。"""
    nodes = _list_hdf5_native_nodes(str(frame_layout_h5))
    by_node = {n["node"]: n for n in nodes}
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    # position 每帧 1 行 → n_frames 行；orientation 每帧 2 行 → 2*n_frames 行。
    assert by_node["action/end/position"]["rows"] == n_frames
    assert by_node["state/end/orientation"]["rows"] == 2 * n_frames


def test_normal_layout_is_not_merged(tmp_path: Path) -> None:
    """**零回归**：非帧布局（命名分组）不得被合并。"""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "normal.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("meta/accel", data=np.zeros((10, 3)))
        f.create_dataset("meta/gyro", data=np.zeros((10, 3)))
        f.create_dataset("pose/head", data=np.zeros((10, 7)))
    nodes = _list_hdf5_native_nodes(str(path))
    assert len(nodes) == 3
    assert not any(n.get("frame_layout") for n in nodes)


def test_mixed_layout_is_not_merged(tmp_path: Path) -> None:
    """顶层既有数字也有非数字 → 保守不合并（避免误伤）。"""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "mixed.h5"
    with h5py.File(path, "w") as f:
        for i in range(_FRAME_LAYOUT_MIN_GROUPS + 5):
            f.create_group(str(i)).create_dataset("pos", data=np.zeros((1, 3)))
        f.create_group("meta").create_dataset("cam", data=np.zeros((2, 2)))
    nodes = _list_hdf5_native_nodes(str(path))
    assert not any(n.get("frame_layout") for n in nodes), "混合布局被误合并"


def test_merge_frame_layout_passthrough_below_threshold() -> None:
    """少于阈值的条目原样返回（小文件零回归）。"""
    entries = [{"node": f"{i}/a", "rows": 1, "cols": 1, "fields": []}
               for i in range(3)]
    assert _merge_frame_layout(entries) == entries


def test_merge_inconsistent_frames_recorded_honestly(tmp_path: Path) -> None:
    """各帧字段集合不一致时：合并结果仍可用，但**如实标注**不完整帧。

    背景：帧布局的合并前只核对前若干帧（性能考量），后续帧若缺字段，合并结果
    会以"首帧字段集合"为准。此时不得静默——条目标注 ``inconsistent_frames``
    与样例帧号，供用户判断。
    """
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "odd.h5"
    with h5py.File(path, "w") as f:
        for i in range(_FRAME_LAYOUT_MIN_GROUPS + 5):
            g = f.create_group(str(i))
            g.create_dataset("pos", data=np.zeros((1, 3)))
            if i % 2 == 0:  # 只有偶数帧有第二个字段 → 结构不一致
                g.create_dataset("extra", data=np.zeros((1, 2)))
    nodes = _list_hdf5_native_nodes(str(path))
    merged = [n for n in nodes if n.get("frame_layout")]
    assert merged, "帧布局未被识别（应识别而非误合并）"
    # 至少有一条被标注为帧结构不一致。
    assert any(n.get("inconsistent_frames") for n in merged), (
        f"不一致布局未被如实标注：{[n.get('node') for n in merged]}"
    )


# --- 3. 合并流的读取 --------------------------------------------------------


def test_read_merged_node_concatenates_frames(frame_layout_h5: Path) -> None:
    """**核心回归**：合并节点可按帧纵向拼接读出，含 frame_index 列。"""
    df = read_hdf5_node(str(frame_layout_h5), "action/end/position")
    assert df is not None
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    assert df.shape[0] == n_frames
    assert "frame_index" in df.columns
    assert df["frame_index"].nunique() == n_frames


def test_read_merged_node_2d_field(frame_layout_h5: Path) -> None:
    """2D 字段（每帧多行）垂直堆叠，行数按帧倍增。"""
    df = read_hdf5_node(str(frame_layout_h5), "state/end/orientation")
    assert df is not None
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    assert df.shape[0] == 2 * n_frames


def test_read_merged_node_has_semantic_columns(frame_layout_h5: Path) -> None:
    """**回归**：列名不得是裸序号（0/1/2），应带字段名语义。

    真实踩坑：裸 ndarray 转 DataFrame 后列名是 0/1/2…，列名对用户与模型
    毫无意义；且该列名逻辑曾因缺少 numpy 导入而抛 NameError，被外层
    `except Exception` 静默吞掉 → "流登记成功但读不出数据"。
    """
    df = read_hdf5_node(str(frame_layout_h5), "action/end/position")
    assert df is not None
    for col in df.columns:
        assert not isinstance(col, (int, np.integer)), f"列名是裸序号：{list(df.columns)}"
    assert any(str(c).startswith("position_") for c in df.columns)


def test_read_nonexistent_node_returns_none(frame_layout_h5: Path) -> None:
    """不存在的节点返回 None（不抛异常）。"""
    assert read_hdf5_node(str(frame_layout_h5), "no/such/field") is None


# --- 4. 流登记（含兜底上限）----------------------------------------------


def test_registration_merges_frame_layout(tmp_path: Path, frame_layout_h5: Path) -> None:
    """登记入口：帧布局合并为 2 条流（经 register_h5_node_streams）。"""
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="demo")
    entries = register_h5_node_streams(ctx, frame_layout_h5)
    assert len(entries) == 2
    assert all(e.get("frame_layout") for e in entries)
    assert all(e.get("n_frames") for e in entries)
    # 说明文本让模型知道这是按帧分片的流。
    assert all("每帧一组" in str(e.get("label_evidence")) for e in entries)


def test_registration_caps_stream_count(tmp_path: Path, monkeypatch) -> None:
    """**兜底护栏**：即便出现未被识别的爆炸布局，单容器登记数也被硬上限截断。"""
    fake_nodes = [
        {"node": f"g{i}/col", "rows": 1, "cols": 1, "fields": []}
        for i in range(_MAX_STREAMS_PER_CONTAINER + 50)
    ]
    monkeypatch.setattr(_ld, "_list_hdf5_native_nodes", lambda _p: fake_nodes)
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="demo")
    entries = _ld.register_h5_node_streams(ctx, tmp_path / "x.h5")
    assert len(entries) == _MAX_STREAMS_PER_CONTAINER
    notes = ctx.meta.get("stream_registration_notes") or []
    assert notes and "上限" in notes[0]


# --- 5. 端到端：目录加载不爆炸 ---------------------------------------------


def test_directory_load_does_not_explode(tmp_path: Path) -> None:
    """端到端：含帧布局 h5 的目录加载后，streams 数为个位数（非上万）。"""
    h5py = pytest.importorskip("h5py")
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 20
    with h5py.File(ds_dir / "aligned_joints.h5", "w") as f:
        for i in range(n_frames):
            g = f.create_group(str(i))
            g.create_dataset("action/end/position", data=np.zeros((1, 3)))
    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None)
    result = load_dataset_impl(ctx, str(ds_dir))
    assert result["success"] is True
    streams = ctx.meta.get("streams") or []
    assert len(streams) < 20, f"流数爆炸：{len(streams)}"
    assert any(s.get("frame_layout") for s in streams)
