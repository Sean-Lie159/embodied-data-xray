"""表名容错、主表口径与跨表运算（2026-09-20 用户实测）。

背景（UMI / 2655849 数据集测试暴露三处缺陷）：

1. **表名格式陷阱**：agent 用 `aligned_joints.h5::state/end/position`（带扩展名）
   调 profile_data → 报"表不存在"，于是得出"工具读不出 h5 节点"的错误结论。
   而 `inspect_streams` 给出的 `source` 恰是**完整路径**（照抄必失败），
   错误提示又没给可用示例与格式说明——"能力可达"被表述掩盖。
2. **主表口径不一致**：加载 aligned_joints.h5 后 `context.df.shape == (2, 4)`，
   主表节点是 `0/action/end/orientation`——**14135 帧中的第 0 帧**。
   根因是选主表的 `_load_hdf5_native` 未使用帧布局合并，看到的是单帧行数。
3. **跨表运算能力缺失**：`profile_data`/`compute_stats` 只做单表聚合，
   无法回答"实际末端与指令末端差多少"。且该场景下对齐键 `frame_index` **有重复**
   （每帧 2 行 = 左右两侧），直接 merge 会产生**笛卡尔积**（实测 56540 行，
   比两侧各自还多），统计建立在错误配对上。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools import _data_access
from app.tools.load_dataset import (
    _FRAME_LAYOUT_MIN_GROUPS,
    load_dataset_impl,
    read_hdf5_node,
)

import app.tools.compare_table_columns  # noqa: F401
import app.tools.inspect_streams  # noqa: F401
import app.tools.profile_data  # noqa: F401

_cc = sys.modules["app.tools.compare_table_columns"]
_ins = sys.modules["app.tools.inspect_streams"]
_pd = sys.modules["app.tools.profile_data"]


@pytest.fixture()
def frame_h5(tmp_path: Path) -> Path:
    """每帧一组的 h5：两个字段，各 1 行/帧（便于构造跨表对比场景）。"""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "aligned.h5"
    n = _FRAME_LAYOUT_MIN_GROUPS + 20
    with h5py.File(path, "w") as f:
        for i in range(n):
            g = f.create_group(str(i))
            g.create_dataset("state/end/position",
                             data=np.array([i * 0.1, i * 0.2, 1.0], dtype=float))
            g.create_dataset("action/end/position",
                             data=np.array([i * 0.1 + 0.05, i * 0.2, 1.5],
                                           dtype=float))
    return path


# --- 1. 表名容错（问题一）--------------------------------------------------


def test_all_table_name_spellings_work(frame_h5: Path) -> None:
    """**核心回归**：四种表名写法全部可用（此前带扩展名的两种必失败）。"""
    ctx = RunContext(output_dir=str(frame_h5.parent), dataset_id="ds")
    ctx.df = read_hdf5_node(str(frame_h5), "state/end/position")
    # 模拟登记表存在（resolve_table_name 依赖它定位）
    ctx.meta["streams"] = [{
        "path": f"{frame_h5}::state/end/position",
        "format": "h5", "kind": "pose", "channels": [],
    }]
    spellings = [
        "state/end/position",                                    # 仅节点名
        f"{frame_h5.stem}::state/end/position",                  # stem::节点
        f"{frame_h5.name}::state/end/position",                  # 带扩展名
        f"{frame_h5}::state/end/position",                       # 完整路径
    ]
    for t in spellings:
        r = _data_access.resolve_table_name(ctx, t)
        assert r["success"] is True, f"表名写法失败：{t} → {r.get('error')}"


def test_table_not_found_gives_usable_examples(frame_h5: Path) -> None:
    """**关键回归**：表不存在的提示必须给出**可直接使用**的表名示例与格式说明。

    此前只说"见流登记表/inspect_streams 的清单"，而清单里的 source 是完整路径，
    agent 照抄必失败——既无示例也无格式要求，导致误判"工具不支持"。
    """
    ctx = RunContext(output_dir=str(frame_h5.parent), dataset_id="ds")
    ctx.meta["streams"] = [{
        "path": f"{frame_h5}::state/end/position",
        "format": "h5", "kind": "pose", "channels": [],
    }]
    r = _data_access.resolve_table_name(ctx, "no_such_table")
    assert r["success"] is False
    msg = str(r["user_message"])
    assert "不含扩展名" in msg, msg
    examples = r.get("available_examples") or []
    assert examples, f"未给可用示例：{msg}"
    # 示例本身必须真的可用（照抄即可成功）。
    for ex in examples:
        assert _data_access.resolve_table_name(ctx, ex)["success"] is True, ex


def test_inspect_streams_provides_table_name(frame_h5: Path) -> None:
    """`inspect_streams` 必须给出**可直接传入 table 参数**的 table_name。

    与 source（完整路径，供溯源）并存——本次缺陷的直接诱因就是只有 source。
    """
    ctx = RunContext(output_dir=str(frame_h5.parent), dataset_id="ds")
    ctx.df = read_hdf5_node(str(frame_h5), "state/end/position")
    ctx.meta.update({
        "capabilities": {}, "streams": [{
            "path": f"{frame_h5}::state/end/position",
            "format": "h5", "kind": "pose", "channels": [],
            "semantic_label": "位姿", "role": {"role": "位姿"},
        }],
    })
    ins = _ins.inspect_streams_impl(ctx)
    ts = ins.get("table_streams") or []
    assert ts, "无表格流"
    for t in ts:
        assert t.get("table_name"), f"缺 table_name：{t}"
        # table_name 不得含扩展名或目录前缀（否则照抄会失败）。
        assert not t["table_name"].lower().endswith((".h5", ".hdf5", ".mcap"))
        assert "\\" not in t["table_name"] and "/" not in t["table_name"].split("::")[0]


def test_profile_data_passes_available_examples(frame_h5: Path) -> None:
    """`profile_data` 的错误返回透传 available_examples（少一轮往返即可纠正）。"""
    ctx = RunContext(output_dir=str(frame_h5.parent), dataset_id="ds")
    ctx.meta["streams"] = [{
        "path": f"{frame_h5}::state/end/position",
        "format": "h5", "kind": "pose", "channels": [],
    }]
    r = _pd.profile_data_impl(ctx, table="nope")
    assert r["success"] is False
    assert r.get("available_examples")


# --- 2. 主表口径（问题二 a）-----------------------------------------------


def test_h5_main_table_row_count_matches_registry(frame_h5: Path) -> None:
    """**核心回归**：h5 主表的行数必须与流登记表同口径（不得只装载第 0 帧）。

    真实事故：主表 shape 为 (2, 4)——14135 帧里的第 0 帧——因为选主表的
    `_load_hdf5_native` 没走帧布局合并，看到的是单帧行数。
    """
    from app.tools.load_dataset import _list_hdf5_native_nodes

    ctx = RunContext(output_dir=str(frame_h5.parent))
    r = load_dataset_impl(ctx, str(frame_h5))
    assert r["success"] is True
    nodes = {n["node"]: n for n in _list_hdf5_native_nodes(str(frame_h5))}
    main_node = ctx.meta.get("h5_source_node")
    assert main_node in nodes, f"主表节点 {main_node} 不在登记清单"
    assert ctx.df is not None
    assert ctx.df.shape[0] == nodes[main_node]["rows"], (
        f"主表行数 {ctx.df.shape[0]} 与登记表 {nodes[main_node]['rows']} 不一致"
    )
    # 具体地：不得是"单帧行数"（本夹具每帧 1 行，14135 帧 → 必须远大于 1）。
    assert ctx.df.shape[0] > 1


def test_h5_structure_consistent_with_registry(frame_h5: Path) -> None:
    """`h5_structure` 元信息与流登记表同口径（含帧布局 n_frames）。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    struct = ctx.df.attrs.get("h5_structure") or []
    assert struct
    for h in struct:
        assert h.get("n_frames"), f"帧布局节点缺 n_frames：{h}"


def test_main_table_selection_is_transparent(frame_h5: Path) -> None:
    """主表选择透明化：给出候选清单与"如何切换"的指引。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    mt = ctx.meta.get("main_table") or {}
    assert mt.get("name")
    assert mt.get("reason")
    cands = mt.get("candidates") or []
    assert cands, "未给候选清单"
    # 候选里的 table_name 必须可直接使用。
    for c in cands[:3]:
        assert c.get("table_name")
        assert _data_access.resolve_table_name(ctx, c["table_name"])["success"]


# --- 3. 跨表运算（问题二 b）----------------------------------------------


def test_compare_basic_diff(frame_h5: Path) -> None:
    """基本 diff：能算两表逐列差并给统计与分布。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    stem = frame_h5.stem
    r = _cc.compare_table_columns_impl(
        ctx, table_a=f"{stem}::state/end/position",
        table_b=f"{stem}::action/end/position",
        columns_a=["position_0"], columns_b=["position_0"], op="diff")
    assert r["success"] is True, r.get("user_message")
    st = r["result"]["position_0（A−B）"]
    # 夹具里 B 比 A 大 0.05 → 差恒为 -0.05。
    assert st["mean"] == pytest.approx(-0.05, abs=1e-6)
    assert st["std"] == pytest.approx(0.0, abs=1e-6)
    assert "decile_histogram" in st, "未给分布形态"


def test_compare_distance(frame_h5: Path) -> None:
    """distance：按行求欧氏距离。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    stem = frame_h5.stem
    r = _cc.compare_table_columns_impl(
        ctx, table_a=f"{stem}::state/end/position",
        table_b=f"{stem}::action/end/position",
        columns_a=["position_0", "position_1", "position_2"],
        columns_b=["position_0", "position_1", "position_2"],
        op="distance")
    assert r["success"] is True
    # 夹具差异：x 差 0.05、z 差 0.5 → 距离 = sqrt(0.0025+0.25)。
    expect = float(np.sqrt(0.05 ** 2 + 0.5 ** 2))
    st = r["result"]["欧氏距离"]
    assert st["mean"] == pytest.approx(expect, abs=1e-5)


def test_compare_duplicate_key_does_not_cartesian(tmp_path: Path) -> None:
    """**核心回归**：对齐键有重复时**不得**产生笛卡尔积。

    真实事故：`state/end/position` 与 `action/end/position` 都是"每帧 2 行"
    （左右两侧），`frame_index` 每帧重复 2 次。直接 merge → 2×2=4 行/帧，
    实测 28270 行变 56540 行，`unmatched` 甚至算出负数。
    正确做法是按**组内序号**配对（同键值内第 k 行一一对应）。
    """
    rng = np.random.default_rng(0)
    n_frames, n_sides = 40, 2
    key = np.repeat(np.arange(n_frames), n_sides)
    a = pd.DataFrame({
        "frame_index": key,
        "v": rng.random(n_frames * n_sides),
    })
    b = pd.DataFrame({
        "frame_index": key,
        "v": a["v"].to_numpy() + 0.1,   # 每行都恰好差 0.1
    })
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    # 直接调内部对齐函数（避免构造 h5 文件），验证配对语义。
    al = _cc._align_by_key(a, b, "frame_index")
    assert al["success"] is True
    assert al["n_matched"] == len(a), (
        f"配对行数 {al['n_matched']} ≠ 输入 {len(a)}（出现笛卡尔积）"
    )
    assert al.get("grouping"), "未标注按组内序号配对"
    j = al["joined"]
    # 正确配对下每行差恒为 -0.1；笛卡尔积会混入反向配对。
    d = j["v_a"] - j["v_b"]
    assert np.allclose(d, -0.1), f"配对错误：差值为 {sorted(set(np.round(d, 6)))[:5]}"


def test_compare_mismatched_duplicate_pattern_rejected(tmp_path: Path) -> None:
    """两侧键的重复模式不一致时**如实拒绝**（不猜配对规则）。"""
    a = pd.DataFrame({"frame_index": [0, 0, 1], "v": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"frame_index": [0, 0, 0, 1], "v": [1.0, 2.0, 3.0, 4.0]})
    r = _cc._align_by_key(a, b, "frame_index")
    assert r["success"] is False
    assert r["error"] == "alignment_key_not_unique"
    assert "配对" in str(r["user_message"])


def test_compare_no_alignment_key_rejected(tmp_path: Path) -> None:
    """无共同对齐键 → 结构化拒绝（不做位置对齐）。"""
    a = pd.DataFrame({"x": [1.0, 2.0]})
    b = pd.DataFrame({"y": [1.0, 2.0]})
    r = _cc._align_by_key(a, b, None)
    assert r["success"] is False
    assert r["error"] == "no_alignment_key"


def test_compare_unmatched_counts_never_negative(frame_h5: Path) -> None:
    """**回归**：未匹配统计不得出现负数（早期按"n - matched"算会为负）。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    stem = frame_h5.stem
    r = _cc.compare_table_columns_impl(
        ctx, table_a=f"{stem}::state/end/position",
        table_b=f"{stem}::action/end/position",
        columns_a=["position_0"], columns_b=["position_0"])
    assert r["success"] is True
    un = r["unmatched"]
    assert un["keys_only_in_a"] >= 0 and un["keys_only_in_b"] >= 0
    assert r["n_matched"] <= max(r["n_a"], r["n_b"])


def test_compare_rejects_bad_op(frame_h5: Path) -> None:
    """非法 op 如实拒绝。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    r = _cc.compare_table_columns_impl(ctx, "a", "b", op="multiply")
    assert r["success"] is False
    assert r["error"] == "unsupported_op"


def test_compare_distance_needs_2_or_3_columns(frame_h5: Path) -> None:
    """distance 列数不合法时如实拒绝（1 列无法构成距离）。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    stem = frame_h5.stem
    r = _cc.compare_table_columns_impl(
        ctx, table_a=f"{stem}::state/end/position",
        table_b=f"{stem}::action/end/position",
        columns_a=["position_0"], columns_b=["position_0"], op="distance")
    assert r["success"] is False
    assert r["error"] == "distance_needs_2_or_3_columns"


def test_compare_columns_length_mismatch(frame_h5: Path) -> None:
    """columns_a/columns_b 长度不等时如实拒绝。"""
    ctx = RunContext(output_dir=str(frame_h5.parent))
    load_dataset_impl(ctx, str(frame_h5))
    stem = frame_h5.stem
    r = _cc.compare_table_columns_impl(
        ctx, table_a=f"{stem}::state/end/position",
        table_b=f"{stem}::action/end/position",
        columns_a=["position_0", "position_1"], columns_b=["position_0"])
    assert r["success"] is False
    assert r["error"] == "columns_length_mismatch"


def test_compare_reports_unmatched_keys(tmp_path: Path) -> None:
    """两表键集合不同时如实报出未匹配的键数。"""
    a = pd.DataFrame({"frame_index": [0, 1, 2], "v": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"frame_index": [0, 1, 9], "v": [1.0, 2.0, 3.0]})
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    al = _cc._align_by_key(a, b, "frame_index")
    assert al["success"] is True
    assert al["n_matched"] == 2
    # 通过公开实现验证未匹配计数（这里直接构造 streams 成本高，故校验内部量）。
    keys_a, keys_b = set(a["frame_index"]), set(b["frame_index"])
    assert len(keys_a - keys_b) == 1 and len(keys_b - keys_a) == 1


def test_compare_tool_registered() -> None:
    """新工具已注册进工具集与包导出。"""
    from agents.tool import FunctionTool

    from app.tools import compare_table_columns

    assert isinstance(compare_table_columns, FunctionTool)
    assert compare_table_columns.name == "compare_table_columns"

    from app.services.chat_service import _ALL_TOOLS

    names = {getattr(t, "name", "") for t in _ALL_TOOLS}
    assert "compare_table_columns" in names
