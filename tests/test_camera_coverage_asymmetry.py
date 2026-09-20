"""相机路数不对称标注 + 时间戳流语义标签（用户确认的取舍 2 与 3）。

背景（2026-09-20 用户实测数据集 ``2655849``）：

- **取舍 2（语义标签）**：修复后新增的 7 条 h5 时间戳流若不做语义识别，会全部
  落到 ``kind="unknown"`` / ``semantic_label="未知（无法分类）"``，UI 上显示一堆
  "未分类"，用户反而更困惑。故新增 ``kind="timestamp_index"``。
- **取舍 3（不对称如实标注）**：真实数据集 h5 内 ``timestamp/camera/`` 只有
  **6 路**，而 ``camera/`` 目录下有 **9 路**同名 txt——多出 head_back_fisheye、
  head_left_fisheye、head_right_fisheye。用户问"h5 时间戳与 camera/ 下每个同名
  txt 的对齐"时若工具只报单侧，会让人误以为两侧一一对应。要求**只标注不补齐**
  （不臆造 h5 里不存在的相机时间戳）。

对应设计文档 ``docs/H5节点形态接纳设计.md`` 第 6 节的三点取舍。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import app.tools.load_dataset  # noqa: F401  确保模块已加载

_ld = sys.modules["app.tools.load_dataset"]
_classify_h5_node = _ld._classify_h5_node

from app.agent.context import RunContext
from app.tools.inspect_streams import _camera_coverage_note

# 真实数据集的相机清单（实测，作为测试锚点）。
_H5_CAMS = ["hand_left_color", "hand_right_color", "head_color", "head_depth",
            "head_stereo_left", "head_stereo_right"]
_DIR_ONLY_CAMS = ["head_back_fisheye", "head_left_fisheye", "head_right_fisheye"]
_ALL_DIR_CAMS = sorted(set(_H5_CAMS) | set(_DIR_ONLY_CAMS))


def _stream(path: str, fmt: str) -> dict:
    return {"path": path, "format": fmt}


def _real_shaped_streams() -> list[dict]:
    """构造与真实数据集同形的流登记表（h5 6 路相机 + 目录 9 路 txt）。"""
    out: list[dict] = []
    for c in _H5_CAMS:
        out.append(_stream(rf"C:\ds\record\aligned_joints.h5::timestamp/camera/{c}", "h5"))
    out.append(_stream(r"C:\ds\record\aligned_joints.h5::main_timestamp", "h5"))
    for c in _ALL_DIR_CAMS:
        out.append(_stream(rf"C:\ds\camera\{c}\{c}.txt", "txt"))
    return out


# --- 取舍 2：时间戳流语义标签 -------------------------------------------------


def test_timestamp_index_kind_for_main_and_camera() -> None:
    """主时钟与相机时间戳节流都被识别为 timestamp_index（不再 unknown）。"""
    kind, label = _classify_h5_node([], "main_timestamp")
    assert kind == "timestamp_index"
    assert "主时钟" in label

    kind, label = _classify_h5_node([], "timestamp/camera/head_color")
    assert kind == "timestamp_index"
    assert "相机" in label

    # 纯相机名（无时间词根）靠**父级路径** timestamp/ 判定。
    kind, label = _classify_h5_node([], "timestamp/camera/head_stereo_left")
    assert kind == "timestamp_index"


def test_timestamp_index_kind_not_leaked_to_action_or_state() -> None:
    """**防误报**：action/state 下的同名叶子不得被当成时间戳流。

    这是新增路径判定时最容易引入的回归——``state/*``、``action/*`` 里也可能
    有叫 ``timestamp`` 的字段或类似命名的叶子。
    """
    for path in ("state/timestamp", "action/joint/timestamp",
                 "state/end/errmsg", "action/end/position"):
        kind, _ = _classify_h5_node([], path)
        assert kind != "timestamp_index", f"{path} 被误判为时间戳流"


# --- 取舍 3：不对称标注（inspect_streams） -----------------------------------


def test_coverage_detects_dir_only_cameras() -> None:
    """**核心**：识别出 3 路"仅目录有"的鱼眼相机，且不补齐 h5 侧。"""
    cov = _camera_coverage_note(_real_shaped_streams())
    assert cov["h5_cameras"] == sorted(_H5_CAMS)
    assert cov["dir_cameras"] == _ALL_DIR_CAMS
    assert cov["only_in_dir"] == _DIR_ONLY_CAMS
    assert cov["only_in_h5"] == []
    assert cov["note"] is not None
    # 标注须给出具体路数与具体相机名（不能只说"有差异"）。
    assert "6 路" in cov["note"] and "9 路" in cov["note"]
    for c in _DIR_ONLY_CAMS:
        assert c in cov["note"]
    # 必须声明"不补齐"——防模型据此臆造 h5 里的相机时间戳。
    assert "不补齐" in cov["note"]


def test_coverage_silent_when_symmetric() -> None:
    """两侧一致时 note 为 None（不制造无意义告警）。"""
    streams = [
        _stream(rf"C:\ds\a.h5::timestamp/camera/{c}", "h5") for c in _H5_CAMS
    ] + [
        _stream(rf"C:\ds\camera\{c}\{c}.txt", "txt") for c in _H5_CAMS
    ]
    cov = _camera_coverage_note(streams)
    assert cov["only_in_dir"] == [] and cov["only_in_h5"] == []
    assert cov["note"] is None


def test_coverage_silent_when_only_one_side_present() -> None:
    """单侧存在时不报"不对称"（那不是不对称，是缺一类数据）。"""
    # 只有 h5 侧（无 camera/ 目录）。
    only_h5 = [_stream(rf"C:\ds\a.h5::timestamp/camera/{c}", "h5") for c in _H5_CAMS]
    assert _camera_coverage_note(only_h5)["note"] is None
    # 只有目录侧。
    only_dir = [_stream(rf"C:\ds\camera\{c}\{c}.txt", "txt") for c in _H5_CAMS]
    assert _camera_coverage_note(only_dir)["note"] is None


def test_coverage_ignores_non_camera_txt() -> None:
    """**防误报**：非 camera/ 目录下的 txt 不得被算作相机路数。"""
    streams = _real_shaped_streams() + [
        _stream(r"C:\ds\logs\imu_left.txt", "txt"),
        _stream(r"C:\ds\misc\notes.txt", "txt"),
    ]
    cov = _camera_coverage_note(streams)
    assert cov["dir_cameras"] == _ALL_DIR_CAMS
    assert "imu_left" not in cov["dir_cameras"]
    assert "notes" not in cov["dir_cameras"]


def test_coverage_detects_h5_only_direction() -> None:
    """反向不对称也要报（h5 有、目录无）——不能只做单向检查。"""
    streams = [
        _stream(rf"C:\ds\a.h5::timestamp/camera/{c}", "h5")
        for c in _H5_CAMS + ["extra_cam"]
    ] + [_stream(rf"C:\ds\camera\{c}\{c}.txt", "txt") for c in _H5_CAMS]
    cov = _camera_coverage_note(streams)
    assert cov["only_in_h5"] == ["extra_cam"]
    assert cov["note"] is not None


# --- 端到端：标注出现在工具返回中 -------------------------------------------


def _build_asymmetric_dataset(root: Path) -> Path:
    """构造一个"h5 6 路 + camera/ 9 路"的不对称数据集（真实形态）。"""
    h5py = pytest.importorskip("h5py")
    n_frames = _ld._FRAME_LAYOUT_MIN_GROUPS + 5
    T0 = 1_756_265_284_805_200_809

    rec = root / "record"
    rec.mkdir(parents=True, exist_ok=True)
    with h5py.File(rec / "aligned_joints.h5", "w") as f:
        for i in range(n_frames):
            g = f.create_group(str(i))
            g.create_dataset("main_timestamp", data=np.uint64(T0 + i * 33_447_424))
            for c in _H5_CAMS:
                g.create_dataset(f"timestamp/camera/{c}",
                                 data=np.array([T0 + i * 33_447_424], dtype=np.uint64))
            g.create_dataset("action/joint/position", data=np.arange(14, dtype=float) + i)

    for c in _ALL_DIR_CAMS:
        d = root / "camera" / c
        d.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{T0 + i * 33_447_424} {'I' if i % 2 else 'P'}"
            for i in range(n_frames)
        ]
        (d / f"{c}.txt").write_text("\n".join(lines), encoding="utf-8")
    return root


def test_inspect_streams_reports_camera_coverage(tmp_path: Path) -> None:
    """**端到端**：设备清单返回 camera_coverage，并在 user_message 中提示。"""
    from app.tools.inspect_streams import inspect_streams_impl

    ds = _build_asymmetric_dataset(tmp_path / "ds")
    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None)
    assert _ld.load_dataset_impl(ctx, str(ds))["success"] is True

    ins = inspect_streams_impl(ctx)
    cov = ins["camera_coverage"]
    assert cov["only_in_dir"] == _DIR_ONLY_CAMS, cov
    assert cov["note"] is not None
    # 提示须出现在给模型看的 user_message / unclassified_hint 里，
    # 否则模型看不到（工具返回的其它字段是给 UI 用的）。
    combined = str(ins.get("user_message", "")) + str(ins.get("unclassified_hint", ""))
    assert "不对称" in combined, combined
    for c in _DIR_ONLY_CAMS:
        assert c in combined, combined


def test_align_container_reports_camera_coverage(tmp_path: Path) -> None:
    """**端到端**：容器对齐同样透出 camera_coverage 并写入 user_message。"""
    from app.tools.align_container import align_container_streams_impl

    ds = _build_asymmetric_dataset(tmp_path / "ds")
    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None)
    assert _ld.load_dataset_impl(ctx, str(ds))["success"] is True

    al = align_container_streams_impl(ctx, container="aligned_joints.h5")
    assert al["success"] is True, al.get("user_message")
    cov = al["camera_coverage"]
    assert cov["only_in_dir"] == _DIR_ONLY_CAMS, cov
    assert "不对称" in str(al["user_message"])
    # 对齐主时钟仍是 h5 主时间戳（不被相机流抢占）。
    assert "main_timestamp" in str(al["master"]["sub"])
