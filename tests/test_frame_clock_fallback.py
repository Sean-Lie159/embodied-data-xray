"""帧序时间轴回退：子流无时间戳列时按 frame_index 回退到容器主时钟。

背景（2026-09-21）：真实数据集 ``aligned_joints.h5`` 里 27 条 action/state 流与
``main_timestamp`` **共享同一 14135 帧**（``frame_index`` 0…14134 逐帧对应），
事实上可时间对齐，但它们自身无时间戳列 → ``align_container_streams`` 全部报
``no_timestamp``，用户拿不到"帧 ↔ 时间"映射。

设计要点（见 ``docs/帧序时间轴回退设计.md``）：

1. **严格校验才回退**——帧数相同**可能是巧合同长**，必须校验 ``frame_index``
   连续（0..N-1、无重复）且长度严格一致；不满足即不回退（本文件负例覆盖）。
2. **必须可溯源**——回退行标 ``time_source="container_master"`` +
   ``time_source_node``，且不计入 ``n_with_timestamp``，避免用户误以为这些流
   自身自带时间戳。
3. **主时钟不得被回退流抢占**——回退流的时间轴是借来的，若参选 max(span) 会
   按字典序"赢"过 ``main_timestamp``（实测曾发生）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import app.tools.load_dataset  # noqa: F401  确保模块已加载

_ld = sys.modules["app.tools.load_dataset"]
pick_container_master_clock = _ld.pick_container_master_clock
read_frame_index_aligned_master_clock = _ld.read_frame_index_aligned_master_clock

T0 = 1_756_265_284_805_200_809
STEP = 33_447_424
N_FRAMES = _ld._FRAME_LAYOUT_MIN_GROUPS + 5

from app.agent.context import RunContext  # noqa: E402


def _build_h5(
    path: Path,
    n_frames: int = N_FRAMES,
    *,
    with_main_clock: bool = True,
    frame_index_mode: str = "normal",
    state_rows_per_frame: int = 1,
) -> Path:
    """构造帧布局 h5。

    Args:
        with_main_clock: 是否含 ``main_timestamp`` 主时钟节点。
        frame_index_mode: ``normal``（0..N-1）/ ``dup``（每帧 2 行 → index 重复）
            / ``gap``（缺一个帧号）/ ``offset``（从 1 开始）。
        state_rows_per_frame: state 流每帧的行数（1=每帧一行）。
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as f:
        for i in range(n_frames):
            g = f.create_group(str(i))
            if with_main_clock:
                g.create_dataset("main_timestamp", data=np.uint64(T0 + i * STEP))
            if frame_index_mode == "gap" and i == n_frames - 1:
                continue  # 少一个帧组 → 该流行数少 1（长度不一致）
            g.create_dataset(
                "state/joint/position",
                data=np.arange(14, dtype=float) + i,
            )
            # 每帧多行的流（如双手 (2,4)）→ frame_index 会重复。
            if state_rows_per_frame > 1:
                g.create_dataset(
                    "action/end/orientation",
                    data=np.zeros((state_rows_per_frame, 4), dtype=float) + i,
                )
    return path


def _load(root: Path, tmp_path: Path) -> RunContext:
    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None)
    res = _ld.load_dataset_impl(ctx, str(root))
    assert res["success"] is True, res
    return ctx


# --- 主时钟选取 -------------------------------------------------------------


def test_pick_master_clock_prefers_main_timestamp() -> None:
    """``main_timestamp`` 优先于其它时间戳节点（语义最明确）。"""
    nodes = ["timestamp/camera/head_color", "main_timestamp",
             "timestamp/camera/hand_left_color", "state/joint/position"]
    assert pick_container_master_clock("unused.h5", nodes) == "main_timestamp"


def test_pick_master_clock_avoids_camera_when_no_main() -> None:
    """无 ``main_timestamp`` 时优先非相机时间戳节点，不拿某路相机当全局时钟。"""
    nodes = ["timestamp/camera/head_color", "timestamp/camera/hand_left_color",
             "robot/clock"]
    assert pick_container_master_clock("unused.h5", nodes) == "robot/clock"


def test_pick_master_clock_none_when_no_timestamp_node() -> None:
    """无任何时间戳节点 → None（不回退）。"""
    nodes = ["state/joint/position", "action/joint/position"]
    assert pick_container_master_clock("unused.h5", nodes) is None


def test_pick_master_clock_deterministic() -> None:
    """多候选时选取可复现（字典序取首）。"""
    nodes = ["zzz/clock", "aaa/clock"]
    got = {pick_container_master_clock("unused.h5", nodes) for _ in range(3)}
    assert got == {"aaa/clock"}


# --- frame_index 严格校验（核心）--------------------------------------------


def test_aligned_master_clock_returns_timestamps(tmp_path: Path) -> None:
    """**核心**：frame_index 连续且长度一致 → 回退成功。"""
    h5 = _build_h5(tmp_path / "a.h5")
    ts, basis = read_frame_index_aligned_master_clock(
        str(h5), "state/joint/position", "main_timestamp")
    assert ts is not None, f"应回退成功，实际：{basis}"
    assert len(ts) == N_FRAMES
    assert int(ts[0]) == T0
    assert int(ts[-1]) == T0 + (N_FRAMES - 1) * STEP
    assert "严格一致" in basis


def test_duplicated_frame_index_not_aligned(tmp_path: Path) -> None:
    """**负例（关键）**：frame_index 有重复 → 拒绝回退。

    每帧多行的流（如双手 (2,4) 布局）其 frame_index 会重复——行数看似 2×帧数，
    但按逐帧 1:1 校验不通过。这正是真实数据集里 6 条流保持 no_timestamp 的原因。
    """
    h5 = _build_h5(tmp_path / "a.h5", state_rows_per_frame=2)
    ts, basis = read_frame_index_aligned_master_clock(
        str(h5), "action/end/orientation", "main_timestamp")
    assert ts is None, "frame_index 重复时不得回退（会错位对齐）"
    assert "重复" in basis


def test_length_mismatch_not_aligned(tmp_path: Path) -> None:
    """**负例**：子流行数与主时钟长度不一致 → 拒绝（不做截断/补齐）。"""
    h5 = _build_h5(tmp_path / "a.h5", frame_index_mode="gap")
    # gap 模式少一个帧组 → state 流少 1 行，而主时钟仍为 N_FRAMES。
    ts, basis = read_frame_index_aligned_master_clock(
        str(h5), "state/joint/position", "main_timestamp")
    assert ts is None, f"长度不一致时不得回退，实际 basis={basis}"
    assert "不一致" in basis or "连续" in basis


def test_no_main_clock_not_aligned(tmp_path: Path) -> None:
    """**负例**：容器内无主时钟 → 不回退。"""
    h5 = _build_h5(tmp_path / "a.h5", with_main_clock=False)
    ts, basis = read_frame_index_aligned_master_clock(
        str(h5), "state/joint/position", "main_timestamp")
    assert ts is None


def test_missing_stream_not_aligned(tmp_path: Path) -> None:
    """**负例**：子流不存在 → 不回退（不抛异常）。"""
    h5 = _build_h5(tmp_path / "a.h5")
    ts, basis = read_frame_index_aligned_master_clock(
        str(h5), "state/nonexistent", "main_timestamp")
    assert ts is None


def test_row_count_equal_but_index_not_continuous(tmp_path: Path) -> None:
    """**负例（防巧合）**：行数相等但 frame_index 不连续 → 拒绝回退。

    这是"帧数相同可能是巧合同长"的直接防护：构造一个普通（非帧布局）h5，
    让两条节点行数相同但语义无关。
    """
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "plain.h5"
    n = 10
    with h5py.File(p, "w") as f:
        # 普通层级（非每帧一组）：fake/clock 10 行，state/x 10 行但无 frame_index。
        f.create_dataset("fake/clock",
                         data=np.arange(n, dtype=np.float64) + T0)
        f.create_dataset("state/x", data=np.arange(n, dtype=np.float64))
    ts, basis = read_frame_index_aligned_master_clock(
        str(p), "state/x", "fake/clock")
    # 无 frame_index 列 → 拒绝（行数相等也不认）。
    assert ts is None
    assert "frame_index" in basis


# --- 端到端 -----------------------------------------------------------------


def test_end_to_end_fallback_and_provenance(tmp_path: Path) -> None:
    """**端到端**：回退生效、计数拆分正确、可溯源标注齐备。"""
    from app.tools.align_container import align_container_streams_impl

    root = tmp_path / "ds"
    rec = root / "record"
    rec.mkdir(parents=True)
    h5 = _build_h5(rec / "aligned_joints.h5")

    ctx = _load(root, tmp_path)
    al = align_container_streams_impl(ctx, container="aligned_joints.h5")
    assert al["success"] is True, al.get("user_message")

    # 计数拆开：state 流属回退，不计入 n_with_timestamp。
    assert al["n_with_timestamp"] == 1, al  # 仅 main_timestamp 自身
    assert al["n_by_container_clock"] == 1, al  # 仅 state/joint/position
    assert al["container_master_clock"] == "main_timestamp"

    by_sub = {r["sub"]: r for r in al["streams"]}
    mt = by_sub["main_timestamp"]
    assert mt["time_source"] == "stream"

    st = by_sub["state/joint/position"]
    assert st["status"] == "ok"
    assert st["time_source"] == "container_master"
    assert st["time_source_node"] == "main_timestamp"
    assert "严格一致" in st["time_source_basis"]

    # **主时钟不得被回退流抢占**（回退流是借来的时间轴）。
    assert al["master"]["sub"] == "main_timestamp", al["master"]

    # user_message 必须显式说明时间轴来源，防用户误以为该流自带时间戳。
    msg = al["user_message"]
    assert "非自身携带" in msg or "来自容器主时钟" in msg, msg
    assert "27 条" not in msg  # 本 fixture 只有 1 条回退流


def test_end_to_end_no_fallback_without_main_clock(tmp_path: Path) -> None:
    """无主时钟时保持原有行为（全部 no_timestamp，不误报可对齐）。"""
    from app.tools.align_container import align_container_streams_impl

    root = tmp_path / "ds"
    rec = root / "record"
    rec.mkdir(parents=True)
    _build_h5(rec / "aligned_joints.h5", with_main_clock=False)

    ctx = _load(root, tmp_path)
    al = align_container_streams_impl(ctx, container="aligned_joints.h5")
    assert al.get("error") == "no_timestamp", al


def test_end_to_end_camera_streams_keep_own_clock(tmp_path: Path) -> None:
    """**回归**：自带时间戳的流仍标 ``time_source="stream"``，不被主时钟覆盖。"""
    h5py = pytest.importorskip("h5py")
    from app.tools.align_container import align_container_streams_impl

    root = tmp_path / "ds"
    rec = root / "record"
    rec.mkdir(parents=True)
    with h5py.File(rec / "aligned_joints.h5", "w") as f:
        for i in range(N_FRAMES):
            g = f.create_group(str(i))
            g.create_dataset("main_timestamp", data=np.uint64(T0 + i * STEP))
            g.create_dataset("timestamp/camera/head_color",
                             data=np.array([T0 + i * STEP], dtype=np.uint64))
            g.create_dataset("state/joint/position", data=np.arange(14, dtype=float) + i)

    ctx = _load(root, tmp_path)
    al = align_container_streams_impl(ctx, container="aligned_joints.h5")
    by_sub = {r["sub"]: r for r in al["streams"]}
    cam = by_sub["timestamp/camera/head_color"]
    assert cam["time_source"] == "stream", cam
    assert "time_source_node" not in cam or cam.get("time_source_node") is None
    # 相机流有自身时间戳 → 计入 n_with_timestamp；state 流回退 → 另计。
    assert al["n_with_timestamp"] == 2  # main_timestamp + head_color
    assert al["n_by_container_clock"] == 1  # state/joint/position
