"""metainfo 配对推广与时间戳列识别扩展的单元测试。

覆盖：mp4 + 同名 index.parquet（pts/frame_timestamps_ns 列）配对并对齐跑通；
词表未命中但内容指纹命中（单调递增+量级）的时间戳列用例；
一表多时间戳列主列选择（物理 > 帧序号）；版本组去重（变体不参与对齐）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools._sniffing import find_timestamp_columns, pair_streams
from app.tools.check_temporal_sync import check_temporal_sync_impl
from app.tools.load_dataset import load_dataset_impl


def test_pair_streams_parquet_metainfo(tmp_path: Path) -> None:
    """mp4 + 同名 .index.parquet（含 frame_timestamps_ns 物理时间戳）→ media_metainfo 配对。"""
    root = tmp_path / "p"
    root.mkdir()
    (root / "camera-a.mp4").write_bytes(b"f")
    pd.DataFrame({"pts": list(range(3)), "frame_timestamps_ns": [1_700_000_000_000_000_000 + i * 1_000_000 for i in range(3)]}).to_parquet(root / "camera-a.index.parquet")
    pairs = pair_streams([str(root / "camera-a.mp4")], [str(root / "camera-a.index.parquet")], [])
    mm = [p for p in pairs if p["type"] == "media_metainfo"]
    assert len(mm) == 1
    assert mm[0]["media"].endswith("camera-a.mp4")
    assert mm[0]["metainfo"].endswith("camera-a.index.parquet")
    assert mm[0]["metainfo_format"] == "parquet"
    assert mm[0]["timestamp_column"] == "frame_timestamps_ns"
    assert mm[0]["timestamp_source"] == "dictionary"


def test_pair_streams_multiple_candidates_prefers_physical(tmp_path: Path) -> None:
    """同 stem 多候选（json 帧序号 + index.parquet 物理）→ 全登记，优先物理参与对齐。"""
    root = tmp_path / "p2"
    root.mkdir()
    (root / "cam.mp4").write_bytes(b"f")
    # json 只含帧序号（pts）；parquet 含物理时间戳（frame_timestamps_ns）。
    pd.DataFrame({"pts": [0, 1, 2]}).to_json(root / "cam.json", orient="records")
    pd.DataFrame({"pts": list(range(3)), "frame_timestamps_ns": [1_700_000_000_000_000_000 + i * 1_000_000 for i in range(3)]}).to_parquet(root / "cam.index.parquet")
    pairs = pair_streams([str(root / "cam.mp4")], [str(root / "cam.json"), str(root / "cam.index.parquet")], [])
    mm = [p for p in pairs if p["type"] == "media_metainfo"]
    assert len(mm) == 1
    assert mm[0]["metainfo"].endswith("cam.index.parquet")  # 物理时间戳优先
    # 多候选全部登记。
    assert len(mm[0]["all_candidates"]) == 2


def test_find_timestamp_columns_wordlist() -> None:
    """词表命中：frame_timestamps_ns 为物理，pts 为帧序号。"""
    info = find_timestamp_columns(["pts", "frame_timestamps_ns", "data"])
    assert info["main"] == "frame_timestamps_ns"  # 物理 > 帧序号
    assert info["frame_main"] is False
    assert "pts" in info["alternatives"]


def test_find_timestamp_columns_content_fingerprint_fallback() -> None:
    """词表未命中但内容指纹命中（单调递增 + 量级符合时间单位）→ fingerprint 来源。"""
    sample = pd.DataFrame({
        "colA": [5.0, 3.0, 8.0],          # 非单调递增 → 排除
        "colB": [10, 2, 30],              # 非单调 → 排除
        "unknown_ts": [1_700_000_000_000_000_000, 1_700_000_000_001_000_000, 1_700_000_000_002_000_000],
    })
    info = find_timestamp_columns(list(sample.columns), sample)
    assert info["main"] == "unknown_ts"
    assert info["source"] == "fingerprint"
    assert info["frame_main"] is False


def _metainfo_parquet_dir(tmp_path: Path) -> Path:
    """构造：cam.mp4 + 同名 cam.index.parquet（frame_timestamps_ns 物理时间戳）。"""
    root = tmp_path / "ds"
    root.mkdir()
    (root / "cam.mp4").write_bytes(b"fakemovie")
    ts = [1_700_000_000_000_000_000 + i * 33_333_333 for i in range(30)]  # 30Hz
    pd.DataFrame({"pts": list(range(30)), "frame_timestamps_ns": ts}).to_parquet(root / "cam.index.parquet")
    # 另一路含 IMU 的表格流作对齐基准（无同名视频）。
    pd.DataFrame({
        "timestamp_ns": [1_700_000_000_000_000_000 + i * 1_000_000 for i in range(100)],
        "x": [1.0] * 100, "y": [1.0] * 100, "z": [9.8] * 100,
    }).to_csv(root / "imu.csv", index=False)
    return root


def test_check_temporal_sync_parquet_metainfo_runs(tmp_path: Path) -> None:
    """mp4 + index.parquet metainfo 配对后，check_temporal_sync 能跑通对齐。"""
    root = _metainfo_parquet_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    # 配对已建立且 metainfo_format=parquet。
    mm = [p for p in ctx.meta["stream_pairs"] if p["type"] == "media_metainfo"]
    assert mm and mm[0]["metainfo_format"] == "parquet"
    r = check_temporal_sync_impl(ctx)
    assert r["success"] is True  # 对齐跑通（不报"无法读取时间戳列"）
    # 视频流经 parquet metainfo 参与对齐。
    vid_status = {k: v for k, v in r.get("streams_status", {}).items() if k.endswith("cam.mp4")}
    assert any("参与对齐" in v for v in vid_status.values())


def test_version_group_dedup_in_alignment(tmp_path: Path) -> None:
    """版本组去重：变体视频（_480/_pre）不参与对齐，主版本参与。"""
    root = tmp_path / "ds2"
    root.mkdir()
    # 主版本 + 变体，各带同名 metainfo。
    ts = [1_700_000_000_000_000_000 + i * 33_333_333 for i in range(30)]
    for name in ("cam.mp4", "cam_480.mp4"):
        (root / name).write_bytes(b"fakemovie")
        pd.DataFrame({"pts": list(range(30)), "frame_timestamps_ns": ts}).to_parquet(root / f"{name[:-4]}.index.parquet")
    pd.DataFrame({
        "timestamp_ns": [1_700_000_000_000_000_000 + i * 1_000_000 for i in range(100)],
        "x": [1.0] * 100, "y": [1.0] * 100, "z": [9.8] * 100,
    }).to_csv(root / "imu.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = check_temporal_sync_impl(ctx)
    # 变体 cam_480.mp4 不参与对齐（版本组去重）。
    vid_status = {k: v for k, v in r.get("streams_status", {}).items() if "cam" in k and ".mp4" in k}
    assert any("变体" in v for k, v in vid_status.items() if "480" in k)
