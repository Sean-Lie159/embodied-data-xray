"""h5 叶子形态接纳：帧内标量/单元素时间戳不得被漏登记（2026-09-20 真实缺陷）。

背景（用户实测）：加载 ``2655849`` 后问"h5 里有没有时间戳"，agent 回答
"h5 里没有时间戳字段，只有 frame_index"。而真实文件
``record/aligned_joints.h5``（513MB）**每一帧**都带：

- ``main_timestamp``：``shape=()``、``dtype=uint64``（**0 维标量**）
- ``timestamp/camera/<相机名>``：``shape=(1,)``、``dtype=uint64``（**单元素数组**）

根因：`_visit` 只按**维数**枚举接纳三种形态（compound / 2D / 1D 且元素 >1），
这两类双双落入 ``else: return`` 被静默丢弃。实测 27 条登记流全是 action/state
数值节点，7 个时间戳节点一条未登记——下游 `align_container_streams` 因此早退为
``no_timestamp``，`inspect_streams` 采样率全部 ``present=False``。

注意：``is_timestamp_like_field("main_timestamp")`` 一直是 True，**字段判据没有错**，
错在节点根本没进登记表。故本文件的断言都针对**登记与读取**，而非判据。

本文件与 ``test_h5_tools_chain.py`` 的区别（关键）：后者的 fixture 把时间戳写成
**compound 字段**（``dtype=[("main_timestamp", "<f8"), ...]``），与真实文件的
"逐帧标量"形态**不一致**——这正是它没能挡住本次缺陷的原因。本文件用真实形态。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import app.tools.load_dataset  # noqa: F401  确保模块已加载

# 取模块对象：`app.tools` 包的 __init__ 把名为 load_dataset 的 FunctionTool
# 暴露为包属性，`from app.tools import load_dataset` 拿到的是工具对象而非模块。
_ld = sys.modules["app.tools.load_dataset"]

_FRAME_LAYOUT_MIN_GROUPS = _ld._FRAME_LAYOUT_MIN_GROUPS
_classify_h5_leaf = _ld._classify_h5_leaf
_is_numeric_dtype = _ld._is_numeric_dtype
_list_hdf5_native_nodes = _ld._list_hdf5_native_nodes
read_hdf5_node = _ld.read_hdf5_node
read_hdf5_nodes_metadata = _ld.read_hdf5_nodes_metadata

T0_NS = 1_756_265_284_805_200_809  # 实测真实文件的首帧 main_timestamp（ns epoch）
FRAME_NS = 33_447_424             # 实测中位间隔（≈29.90 Hz）


def _build_real_shaped_h5(path: Path, n_frames: int) -> Path:
    """构造与真实 aligned_joints.h5 **同构**的帧内标量时间戳布局。

    关键形态（与真实文件逐位对应）：
    - ``main_timestamp``：0 维标量（``np.uint64(...)`` 写入 → shape=()）
    - ``timestamp/camera/<name>``：``shape=(1,)`` 单元素数组
    - ``state/end/errmsg``：``dtype=object``、``shape=(1,)`` 的字符串（**噪声对照**）
    - ``action/joint/position``：``(14,)`` 多元素向量（既有形态，回归对照）
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        for i in range(n_frames):
            g = f.create_group(str(i))
            # 0 维标量时间戳——此前被丢弃。
            g.create_dataset("main_timestamp", data=np.uint64(T0_NS + i * FRAME_NS))
            # 单元素数组时间戳——此前被丢弃。
            g.create_dataset("timestamp/camera/head_color",
                             data=np.array([T0_NS + i * FRAME_NS], dtype=np.uint64))
            g.create_dataset("timestamp/camera/hand_left_color",
                             data=np.array([T0_NS + i * FRAME_NS - 1_000_000],
                                           dtype=np.uint64))
            # 既有形态：多元素向量（回归对照，行数不得变）。
            g.create_dataset("action/joint/position",
                             data=np.arange(14, dtype=float) + i)
            # 噪声对照：object 字符串单元素——**不得**被登记。
            g.create_dataset("state/end/errmsg",
                             data=np.array(["ok"], dtype=object))
    return path


@pytest.fixture()
def real_shaped_h5(tmp_path: Path) -> Path:
    return _build_real_shaped_h5(tmp_path / "aligned_joints.h5",
                                 _FRAME_LAYOUT_MIN_GROUPS + 10)


# --- 1. dtype 闸门单测 -------------------------------------------------------


def test_is_numeric_dtype_gate() -> None:
    """dtype 闸门：数值型接纳，object/字符串/bytes 一律拒绝。"""
    assert _is_numeric_dtype(np.dtype("uint64")) is True
    assert _is_numeric_dtype(np.dtype("float64")) is True
    assert _is_numeric_dtype(np.dtype("int32")) is True
    assert _is_numeric_dtype(np.dtype("bool")) is True
    # 噪声来源：object（字符串/嵌套）必须拒绝。
    assert _is_numeric_dtype(np.dtype("O")) is False
    assert _is_numeric_dtype(np.dtype("<U8")) is False
    assert _is_numeric_dtype(np.dtype("S64")) is False
    assert _is_numeric_dtype(None) is False


# --- 2. 形态判定单测（_classify_h5_leaf）------------------------------------


def test_classify_leaf_scalar_and_single_element() -> None:
    """**核心**：0 维标量与单元素数组被接纳为「1 行 × 1 列」。"""
    # 0 维标量 —— 此前返回 None（被丢弃）。
    assert _classify_h5_leaf((), np.dtype("uint64"), True) == {
        "rows": 1, "cols": 1, "fields": []}
    # 单元素数组 —— 此前返回 None（shape[0] > 1 不成立）。
    assert _classify_h5_leaf((1,), np.dtype("uint64"), True) == {
        "rows": 1, "cols": 1, "fields": []}
    # 帧布局与否，语义一致（每帧一个观测）。
    assert _classify_h5_leaf((), np.dtype("uint64"), False) == {
        "rows": 1, "cols": 1, "fields": []}
    assert _classify_h5_leaf((1,), np.dtype("float64"), False) == {
        "rows": 1, "cols": 1, "fields": []}


def test_compound_dtype_is_numeric() -> None:
    """**关键回归**：compound（结构化）dtype 必须被认作数值。

    为什么单独测：compound 的 ``dtype.kind`` 是 ``"V"``（void），
    ``np.issubdtype(dtype, np.number)`` 返回 **False**——若 dtype 闸门写成
    "排除 kind in (O,U,S,V)" 就会把**全部** compound 节点误杀。
    实测踩坑：本修复初版即如此，会让既有 27 条流里的 compound 流（动作流、
    相机帧索引）从登记表消失——比原缺陷更严重。故 compound 须按
    "至少一个数值字段"递归判定。
    """
    compound = np.dtype([("value", "<f4"), ("timestamp", "<f8")])
    assert compound.kind == "V"  # 记录该事实，防止后人误以为它是数值 kind
    assert np.issubdtype(compound, np.number) is False
    assert _is_numeric_dtype(compound) is True
    # 纯字符串字段的 compound 仍应拒绝（只有 file_path 这类列不成表）。
    str_compound = np.dtype([("path", "S64")])
    assert _is_numeric_dtype(str_compound) is False


def test_classify_leaf_existing_forms_unchanged() -> None:
    """**零回归**：既有形态的行列语义逐条不变。"""
    compound = np.dtype([("value", "<f4"), ("timestamp", "<f8")])
    # compound：字段名即列名。
    assert _classify_h5_leaf((300,), compound, False) == {
        "rows": 300, "cols": 2, "fields": ["value", "timestamp"]}
    # 2D：帧布局 (2,4) → 每帧 2 行 × 4 列。
    assert _classify_h5_leaf((2, 4), np.dtype("float64"), True) == {
        "rows": 2, "cols": 4, "fields": []}
    # 1D 多元素：帧布局 → 1 行 × N 列（一帧一条向量观测）。
    assert _classify_h5_leaf((14,), np.dtype("float64"), True) == {
        "rows": 1, "cols": 14, "fields": []}
    # 1D 多元素：非帧布局 → N 行 × 1 列（一列时间序列，如 imu timestamps）。
    assert _classify_h5_leaf((7466,), np.dtype("float64"), False) == {
        "rows": 7466, "cols": 1, "fields": []}


def test_classify_leaf_rejects_non_tabular() -> None:
    """不接纳：空数组、object 字符串、3 维以上张量。"""
    assert _classify_h5_leaf((0,), np.dtype("float64"), True) is None   # 空数组
    assert _classify_h5_leaf((), np.dtype("O"), True) is None          # object 标量
    assert _classify_h5_leaf((1,), np.dtype("O"), True) is None        # object 单元素
    assert _classify_h5_leaf((1,), np.dtype("S64"), True) is None      # 字节串
    assert _classify_h5_leaf((2, 2, 3), np.dtype("float64"), True) is None  # 3 维


# --- 3. 登记回归：真实形态的时间戳必须进登记表 ------------------------------


def test_scalar_timestamp_is_registered(real_shaped_h5: Path) -> None:
    """**核心回归**：逐帧标量与单元素时间戳进入登记表，行数 = 帧数。

    此前这两类节点完全不出现，agent 因此回答"h5 里没有时间戳"。
    """
    nodes = _list_hdf5_native_nodes(str(real_shaped_h5))
    by_node = {n["node"]: n for n in nodes}
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10

    assert "main_timestamp" in by_node, (
        f"0 维标量时间戳未被登记：{sorted(by_node)}"
    )
    assert by_node["main_timestamp"]["rows"] == n_frames
    assert by_node["main_timestamp"]["cols"] == 1

    cam = "timestamp/camera/head_color"
    assert cam in by_node, f"单元素相机时间戳未被登记：{sorted(by_node)}"
    assert by_node[cam]["rows"] == n_frames
    assert by_node[cam]["cols"] == 1


def test_object_string_leaf_is_not_registered(real_shaped_h5: Path) -> None:
    """**噪声防护**：object 字符串单元素（errmsg）不得被纳入。

    这是新增接纳规则时最容易引入的回归——``state/end/errmsg`` 与
    ``timestamp/camera/*`` 的 shape 完全一样（``(1,)``），只靠 shape 无法区分，
    必须靠 dtype 闸门。
    """
    nodes = _list_hdf5_native_nodes(str(real_shaped_h5))
    assert not any("errmsg" in n["node"] for n in nodes), (
        f"object 字符串字段被误登记：{[n['node'] for n in nodes]}"
    )


def test_existing_multielement_field_rows_unchanged(real_shaped_h5: Path) -> None:
    """**零回归**：既有 (14,) 向量的行/列口径不变。"""
    nodes = _list_hdf5_native_nodes(str(real_shaped_h5))
    by_node = {n["node"]: n for n in nodes}
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    pos = by_node["action/joint/position"]
    assert pos["rows"] == n_frames   # 帧布局：每帧 1 行
    assert pos["cols"] == 14


def test_main_table_still_prefers_wide_stream(real_shaped_h5: Path) -> None:
    """**回归**：主表选择不变——时间戳（1 列）不得抢占动作流（14 列）。

    排序键是 rows × cols 降序。若标量时间戳被误按"1 行 × N 列"处理，
    可能反过来挤掉真正的主表。
    """
    nodes = _list_hdf5_native_nodes(str(real_shaped_h5))
    assert nodes[0]["node"] == "action/joint/position", (
        f"主表被抢占：{[(n['node'], n['rows'], n['cols']) for n in nodes]}"
    )


# --- 4. 元数据与读取口径 ----------------------------------------------------


def test_metadata_reports_scalar_timestamp_shape(real_shaped_h5: Path) -> None:
    """元数据：标量时间戳报为 (帧数, 1)，列名即字段名（可被时间戳判据命中）。"""
    meta = read_hdf5_nodes_metadata(
        str(real_shaped_h5), ["main_timestamp", "timestamp/camera/head_color"])
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10

    mt = meta["main_timestamp"]
    assert mt["shape"] == (n_frames, 1)
    # 列名不得追加 _0 后缀——否则 is_timestamp_like_field 命中失败，
    # _H5Reader.timestamp 挑不到时间轴。
    assert mt["columns"] == ["main_timestamp"]
    assert _ld.is_timestamp_like_field("main_timestamp") is True

    cam = meta["timestamp/camera/head_color"]
    assert cam["shape"] == (n_frames, 1)
    assert cam["columns"] == ["head_color"]
    # 纯相机名不含时间词根，须靠**父级路径** timestamp/ 判定。
    assert _ld.is_timestamp_like_field("head_color") is False
    assert _ld.is_timestamp_like_field("timestamp/camera/head_color") is True


def test_read_scalar_timestamp_node(real_shaped_h5: Path) -> None:
    """**核心回归**：标量时间戳可按帧拼成 DataFrame（含 frame_index 列）。"""
    df = read_hdf5_node(str(real_shaped_h5), "main_timestamp")
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    assert df is not None, "标量时间戳节点读不出（读取口径未同步）"
    assert df.shape[0] == n_frames
    assert "frame_index" in df.columns
    assert "main_timestamp" in df.columns
    # 值与写入一致（首帧）。
    assert int(df["main_timestamp"].iloc[0]) == T0_NS
    assert int(df["main_timestamp"].iloc[-1]) == T0_NS + (n_frames - 1) * FRAME_NS


def test_read_single_element_camera_timestamp_node(real_shaped_h5: Path) -> None:
    """单元素相机时间戳同样可读，列名为相机名本身。"""
    df = read_hdf5_node(str(real_shaped_h5), "timestamp/camera/head_color")
    n_frames = _FRAME_LAYOUT_MIN_GROUPS + 10
    assert df is not None
    assert df.shape[0] == n_frames
    assert "head_color" in df.columns


# --- 5. 端到端：时间戳参与对齐（不再 no_timestamp）--------------------------


def test_inspect_reports_rate_for_scalar_timestamp(
    tmp_path: Path, real_shaped_h5: Path
) -> None:
    """**端到端**：设备清单报出时间戳流的实测采样率（此前 present=False）。

    真实缺陷表现：``inspect_streams`` 对全部流返回
    ``{"present": False, "reason": "子流无时间戳字段（timestamp）"}``。
    """
    from app.agent.context import RunContext
    from app.tools.inspect_streams import inspect_streams_impl

    # 走目录加载（真实路径：h5 在目录内经 register_h5_node_streams 登记）。
    ds_dir = real_shaped_h5.parent
    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None)
    load = _ld.load_dataset_impl(ctx, str(ds_dir))
    assert load["success"] is True

    ins = inspect_streams_impl(ctx)
    ts_streams = [
        s for s in ins["table_streams"]
        if "timestamp" in str(s.get("table_name", "")).lower()
    ]
    assert ts_streams, "时间戳流未出现在设备清单中"
    for s in ts_streams:
        mr = s["sample_rate"]
        assert mr.get("present") is True, f"{s['table_name']} 采样率不可用：{mr}"
        # 期望 ~30 Hz（FRAME_NS=33447424ns → 29.90Hz）。
        assert 28.0 <= float(mr["sample_rate_hz"]) <= 32.0, (
            f"{s['table_name']} 采样率异常：{mr}"
        )


def test_align_container_does_not_early_return_no_timestamp(
    tmp_path: Path, real_shaped_h5: Path
) -> None:
    """**端到端核心**：容器对齐不再早退为 no_timestamp。

    真实缺陷表现：``align_container_streams`` 返回
    ``error="no_timestamp"``、user_message 为"27 个子流均无可用时间戳字段"。
    """
    from app.agent.context import RunContext
    from app.tools.align_container import align_container_streams_impl

    ds_dir = real_shaped_h5.parent
    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None)
    assert _ld.load_dataset_impl(ctx, str(ds_dir))["success"] is True

    al = align_container_streams_impl(ctx, container="aligned_joints.h5")
    assert al.get("error") != "no_timestamp", f"仍早退无时间戳：{al.get('user_message')}"
    assert al["success"] is True, al.get("user_message")
    assert al["n_with_timestamp"] >= 3, (
        f"有时间的子流过少：{al['n_with_timestamp']}/{al['n_substreams']}"
    )
    # 主时钟应是跨度最大的时间戳流。
    assert "main_timestamp" in str(al["master"]["sub"]), al["master"]
    assert 28.0 <= float(al["master"]["rate_hz"]) <= 32.0
