"""容器内多子流对齐（align_container_streams）的单元测试。

背景（缺口 2）：h5 / mcap 是"单容器多子流"，各子流采样率与跨度不同（真实
案例：dataset.h5 的 action 100Hz、相机 30Hz、位姿 30Hz 但尾部截断 96.8s）。
既有 check_temporal_sync 能逐流转，但缺"一次看完整容器对齐全貌"的入口。
本工具复用统一读取注册表的 candidates() 枚举（重构后已就绪）。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from app.agent.context import RunContext
from app.tools.align_container import align_container_streams_impl
from app.tools.load_dataset import load_dataset_impl

T0_MS = 1_788_426_767_625


def _comp(n: int, rate_hz: float, start_ms: float = T0_MS):
    """造 compound 节点（value + timestamp）。"""
    arr = np.zeros(n, dtype=[("value", "<f4"), ("timestamp", "<f8")])
    arr["timestamp"] = start_ms + np.arange(n) * (1000.0 / rate_hz)
    arr["value"] = np.linspace(0, 1, n)
    return arr


@pytest.fixture()
def h5_ds(tmp_path: Path) -> Path:
    """同构 dataset.h5：主时钟 30Hz 全覆盖 + 100Hz 动作 + 尾部截断的位姿。"""
    p = tmp_path / "ds" / "dataset.h5"
    p.parent.mkdir()
    master = _comp(300, 30.0)                      # 10s 主时钟
    action = _comp(1000, 100.0)                    # 10s 动作（同跨度）
    # 位姿：只覆盖前 6s（尾部截断 4s）
    pose = _comp(180, 30.0)[:180]
    pose["timestamp"] = T0_MS + np.arange(180) * (1000.0 / 30.0)
    with h5py.File(p, "w") as f:
        f.create_group("camera/master").create_dataset("frames", data=master)
        f.create_group("action/feedback").create_dataset("motor_command", data=action)
        f.create_group("pose/hand").create_dataset("pose_in_chest", data=pose)
    return p


def _load(h5: Path) -> RunContext:
    ctx = RunContext(output_dir=str(h5.parent))
    assert load_dataset_impl(ctx, str(h5))["success"] is True
    return ctx


# --- 1. 基本对齐概览 --------------------------------------------------------


def test_align_reports_all_substreams(h5_ds: Path) -> None:
    """逐子流统计齐全：采样率 / 跨度 / 样本数 / 缺口。"""
    ctx = _load(h5_ds)
    r = align_container_streams_impl(ctx)
    assert r["success"] is True, r.get("user_message")
    assert r["container"] == "dataset.h5"
    assert r["n_with_timestamp"] == 3

    by_sub = {s["sub"]: s for s in r["streams"] if s["status"] == "ok"}
    assert set(by_sub) == {
        "camera/master/frames", "action/feedback/motor_command",
        "pose/hand/pose_in_chest",
    }
    assert by_sub["action/feedback/motor_command"]["rate_hz"] == pytest.approx(100, rel=0.02)
    assert by_sub["camera/master/frames"]["rate_hz"] == pytest.approx(30, rel=0.02)


def test_master_is_longest_span(h5_ds: Path) -> None:
    """主时钟 = 跨度最大者（覆盖最全，作对齐参照）。"""
    ctx = _load(h5_ds)
    r = align_container_streams_impl(ctx)
    # master 与 action 同为 10s：取跨度最大者（并列时任一同跨度，验证 span）。
    assert r["master"]["span_s"] == pytest.approx(10.0, rel=0.02)


def test_tail_truncation_detected(h5_ds: Path) -> None:
    """尾部截断标注：位姿只覆盖 6s（比主时钟早结束 4s）→ 进入 warnings。"""
    ctx = _load(h5_ds)
    r = align_container_streams_impl(ctx)
    pose = next(s for s in r["streams"]
                if s.get("sub") == "pose/hand/pose_in_chest")
    assert pose["tail_offset_s"] == pytest.approx(-4.0, abs=0.3)
    assert any("截断" in n for n in pose["notes"]), pose["notes"]
    assert any("pose/hand/pose_in_chest" in w for w in r["warnings"])


def test_offsets_present_for_aligned_streams(h5_ds: Path) -> None:
    """对齐良好的子流：首尾偏移接近 0，无警告。"""
    ctx = _load(h5_ds)
    r = align_container_streams_impl(ctx)
    action = next(s for s in r["streams"]
                  if s.get("sub") == "action/feedback/motor_command")
    assert abs(action["head_offset_s"]) < 0.5
    assert abs(action["tail_offset_s"]) < 0.5
    assert not action["notes"]


# --- 2. 容器选择与错误语义 --------------------------------------------------


def test_container_param_selects_by_substring(h5_ds: Path) -> None:
    """container 参数按子串选容器。"""
    ctx = _load(h5_ds)
    r = align_container_streams_impl(ctx, container="dataset")
    assert r["success"] is True and r["container"] == "dataset.h5"


def test_container_not_found_structured(h5_ds: Path) -> None:
    """容器不存在 → 结构化错误（不静默回退）。"""
    ctx = _load(h5_ds)
    r = align_container_streams_impl(ctx, container="ghost.h5")
    assert r["success"] is False and r["error"] == "container_not_found"


def test_no_container_structured(tmp_path: Path) -> None:
    """非容器数据集（纯 csv）→ 结构化提示，指向 check_temporal_sync。"""
    d = tmp_path / "ds"
    d.mkdir()
    import pandas as pd

    pd.DataFrame({"a": [1, 2]}).to_csv(d / "t.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    r = align_container_streams_impl(ctx)
    assert r["success"] is False and r["error"] == "no_container"
    assert "check_temporal_sync" in r["user_message"]


def test_max_streams_truncation(h5_ds: Path) -> None:
    """子流数超上限 → 截断并标注。"""
    ctx = _load(h5_ds)
    r = align_container_streams_impl(ctx, max_streams=2)
    assert r["truncated"] is True
    assert r["n_substreams"] == 2
