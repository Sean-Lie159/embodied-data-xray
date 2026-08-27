"""Commit A：JSON 顶层 dict（行列表键 frames/data）读取修复的单元测试。

覆盖：{frames:[...]} 结构的 LeRobot JSON——行数正确、展开为表格、时间戳列识别
不被标量键（fps）干扰、不误判空流。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.tools import _data_access
from app.tools.inspect_streams import _read_timestamp_only
from app.tools.load_dataset import load_dataset_impl
from app.agent.context import RunContext


def _frames_json(path: Path, n: int = 100) -> Path:
    """写一个 {frames:[...]} 结构的 JSON（含 timestamp 行列表与标量 fps）。"""
    frames = [
        {"timestamp": i * 1_000_000, "frame_index": i, "obs": float(i)}
        for i in range(n)
    ]
    path.write_text(json.dumps({"fps": 60, "episode_index": 0, "frames": frames}), encoding="utf-8")
    return path


def test_read_stream_full_frames_dict() -> None:
    """顶层 dict 含 frames 行列表 → 展开为 DataFrame（不把 fps 当列）。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _frames_json(Path(td) / "ep.json")
        df = _data_access.read_stream_full(str(p), "json")
    assert df is not None
    assert df.shape[0] == 100
    assert "timestamp" in df.columns
    assert "fps" not in df.columns  # 标量键不作为数据列


def test_read_table_nrows_frames_dict() -> None:
    """顶层 dict 含 frames → 行数正确（不再误判 0/空流）。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _frames_json(Path(td) / "ep.json")
        nrows = _data_access.read_table_nrows(str(p), "json")
    assert nrows == 100


def test_read_timestamp_only_frames_dict() -> None:
    """时间戳列识别不受 fps 标量干扰（不把 fps 当时间戳）。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = _frames_json(Path(td) / "ep.json")
        ts = _read_timestamp_only(str(p), "json")
    assert ts is not None
    assert ts.name == "timestamp"  # 主时间戳列是 timestamp，不是 fps
    assert len(ts) == 100


def test_lerobot_json_not_empty_stream(tmp_path: Path) -> None:
    """{frames:[...]} 的 episode JSON 不被判为空流，正确登记为表格流。"""
    root = tmp_path / "lerobot"
    (root / "data" / "chunk-000").mkdir(parents=True)
    # episode json（frames 结构）+ 同名 parquet（同内容）。
    _frames_json(root / "data/chunk-000/episode_000000.json", 100)
    pd.DataFrame({
        "timestamp": [i * 1_000_000 for i in range(100)],
        "frame_index": list(range(100)),
        "obs": [float(i) for i in range(100)],
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    # episode json 不再被判为空流。
    jstream = next((s for s in ctx.meta["streams"] if s["path"].endswith("episode_000000.json")), None)
    assert jstream is not None
    assert jstream.get("status") != "empty"
