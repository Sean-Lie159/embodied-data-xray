"""h5 节点流接入全工具链（sync/propose/inspect）的端到端测试。

背景（用户实测三轮对话）：h5 节点登记为流后，只有 profile_data 能读——
check_temporal_sync not_applicable（0 可对齐流）、propose 验证 failed
（"无法读取流内容"）、采样率只能 agent 手工推算（~100Hz 被推成 0.1Hz，
容器 ms epoch 差分 10 落 ns 区间 → 误判 ns → 自我纠正到 s 仍错）。
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools.check_temporal_sync import check_temporal_sync_impl
from app.tools.inspect_streams import inspect_streams_impl
from app.tools.load_dataset import load_dataset_impl
from app.tools.propose_semantics import propose_stream_semantics_impl
from app.tools.timestamp_units import infer_unit

T0_MS = 1_788_426_767_625  # Unix 毫秒 epoch（2026-09）


@pytest.fixture()
def h5_ds(tmp_path: Path) -> Path:
    """与真实 dataset.h5 同构：ms epoch + 100Hz 动作 + 30Hz 相机帧索引。"""
    p = tmp_path / "ds" / "dataset.h5"
    p.parent.mkdir()
    n = 300
    act = np.zeros(n, dtype=[("value", "<f4"), ("timestamp", "<f8"),
                             ("raw", "<f4"), ("min", "<f4"), ("max", "<f4")])
    act["timestamp"] = T0_MS + np.arange(n) * 10.0  # 100 Hz
    cam = np.zeros(90, dtype=[("file_path", "S64"), ("timestamp", "<f8")])
    cam["file_path"] = [f"c/{j:04d}.jpg".encode() for j in range(90)]
    cam["timestamp"] = T0_MS + np.arange(90) * 33.33  # ~30 Hz
    idx = np.zeros(90, dtype=[("main_timestamp", "<f8"),
                              ("aligned_timestamp", "<f8"),
                              ("aligned_index", "<i8"), ("time_diff", "<f8")])
    idx["main_timestamp"] = T0_MS + np.arange(90) * 33.33
    idx["aligned_timestamp"] = idx["main_timestamp"] - 0.5
    idx["aligned_index"] = np.arange(90)
    idx["time_diff"] = -0.5
    with h5py.File(p, "w") as f:
        f.create_group("action/left_eef/feedback").create_dataset(
            "motor_command", data=act)
        f.create_group("observation/camera/cam0").create_dataset(
            "frames", data=cam)
        f.create_group("meta/index_map/observation/camera/cam0").create_dataset(
            "frames", data=idx)
    return p


# --- 1. sync：h5 节点流参与对齐（此前 not_applicable/0 流）-------------------


def test_sync_runs_on_h5_node_streams(h5_ds: Path) -> None:
    """默认调用即对齐 h5 节点流（timestamp 字段，ms 单位正确归一）。

    此前：not_applicable（0 可对齐流）——sync 不认识 h5 节点流。
    """
    ctx = RunContext(output_dir=str(h5_ds.parent))
    assert load_dataset_impl(ctx, str(h5_ds))["success"] is True
    s = check_temporal_sync_impl(ctx)
    assert s["success"] is True, s.get("reason")
    assert s.get("error") != "not_applicable"
    checks = s["measurements"]["stream_checks"]
    present = [v for v in checks.values() if isinstance(v, dict) and v.get("present")]
    assert len(present) >= 2
    for chk in present:
        assert chk["timestamp_column"] == "timestamp"
        assert chk["timestamp_unit"] == "ms"
    # action（300 行 @100Hz）：periodic；相机（90 行 @30Hz）：行数 <100 判 static
    # （min_samples 守卫，正确行为）。
    by_rate = {round(c["actual_rate_hz"]): c for c in present}
    assert by_rate[100]["stream_shape"] == "periodic"


def test_inspect_rates_on_h5_nodes_correct_unit(h5_ds: Path) -> None:
    """inspect 采样率：ms epoch 判 ms（不再 s/0.1Hz 失真）。

    无 timestamp 字段的节点（如 meta/index_map，字段为 main_timestamp 等）
    如实报"无时间戳字段"——不硬凑。
    """
    ctx = RunContext(output_dir=str(h5_ds.parent))
    assert load_dataset_impl(ctx, str(h5_ds))["success"] is True
    ins = inspect_streams_impl(ctx)
    for st in ins["table_streams"]:
        mr = st["sample_rate"]
        node = st["source"].partition("::")[2]
        if "index_map" in node:
            # index_map 节点字段为 main_timestamp 等（无 timestamp）→ 如实报。
            assert mr.get("present") is False
            assert "时间戳字段" in mr.get("reason", "")
        else:
            assert mr.get("present") is True
            assert mr["timestamp_unit"] == "ms"
            expected = 100 if "action" in node else 30
            assert expected * 0.9 <= mr["sample_rate_hz"] <= expected * 1.1


def test_infer_unit_ms_epoch(h5_ds: Path) -> None:
    """epoch 判定补全单测：ms epoch + 10ms 差分 → ms（此前误判 ns）。"""
    ts = (T0_MS + np.arange(300) * 10.0).astype(float)
    r = infer_unit(ts, "timestamp")
    assert r["unit"] == "ms", r
    # 真秒 epoch（1.788e9）判 s。
    r2 = infer_unit(T0_MS / 1000 + np.arange(300) * 0.01, "timestamp")
    assert r2["unit"] == "s"


# --- 2. propose：h5 节点流内容可验证（此前 failed）--------------------------


def test_propose_verifies_h5_node_content(h5_ds: Path) -> None:
    """h5 节点按内容验证：camera 帧索引（含 timestamp 字段）→ 结构存在。"""
    ctx = RunContext(output_dir=str(h5_ds.parent))
    assert load_dataset_impl(ctx, str(h5_ds))["success"] is True
    r = propose_stream_semantics_impl(ctx, [
        {"file": "observation/camera/cam0/frames",
         "kind": "frame_index", "semantic_label": "相机帧索引"}])
    item = r["results"][0]
    # frame_index 为自定义 kind：结构字段存在即可 weak（不强求语义）。
    assert item["verified"] in ("weak", "strong"), item
    assert "不存在该文件名" not in item["evidence"]


def test_propose_locates_h5_node_by_path(h5_ds: Path) -> None:
    """file 用节点路径定位 h5 节点流（此前"流登记表中不存在该文件名"）。"""
    ctx = RunContext(output_dir=str(h5_ds.parent))
    assert load_dataset_impl(ctx, str(h5_ds))["success"] is True
    r = propose_stream_semantics_impl(ctx, [
        {"file": "action/left_eef/feedback/motor_command",
         "kind": "actions", "semantic_label": "左手动作"}])
    item = r["results"][0]
    assert "不存在该文件名" not in item["evidence"]
    assert item["verified"] in ("weak", "strong", "failed")
    # actions 含 timestamp/value 字段 → 至少能读到内容（不是"格式不支持"）。
    assert "无法读取流内容" not in item["evidence"]

def test_propose_parquet_no_longer_unreadable(tmp_path: Path) -> None:
    """回归：parquet 流 propose 不再"无法读取流内容"（此前只吃 jsonl/json，
    6 条 index.parquet 语义确认全部 failed 的根因）。"""
    # 用 test_h5_node_streams 的构造器造一个 parquet + h5 混合目录。
    import h5py
    import numpy as np

    from app.tools.propose_semantics import propose_stream_semantics_impl

    d = tmp_path / "mix"
    d.mkdir()
    df = pd.DataFrame({
        "frame_timestamps_ns": (1_788_426_767_625e6 + np.arange(50) * 33_333_333),
        "frame_index": np.arange(50),
    })
    df.to_parquet(d / "cam0.index.parquet")
    cam = np.zeros(50, dtype=[("file_path", "S64"), ("timestamp", "<f8")])
    with h5py.File(d / "dataset.h5", "w") as f:
        f.create_group("observation/camera/cam0").create_dataset("frames", data=cam)

    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    r = propose_stream_semantics_impl(ctx, [
        {"file": "cam0.index.parquet", "kind": "frame_index",
         "semantic_label": "相机帧索引"}])
    item = r["results"][0]
    assert "无法读取流内容" not in item["evidence"], item
    assert item["verified"] in ("weak", "strong"), item
