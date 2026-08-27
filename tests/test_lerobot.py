"""Commit 2 LeRobot v2 schema 支持的单元测试。

覆盖：目录指纹判定 LeRobot v2；meta/*.json 归类 dataset_metadata（不进流/不对齐/
不算空流）；info.json fps 解释；episode JSON/parquet 镜像检测。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.tools._sniffing import (
    detect_dataset_format,
    detect_episode_mirrors,
    explain_fps_mismatch,
    parse_lerobot_info,
)
from app.tools.load_dataset import load_dataset_impl
from app.agent.context import RunContext


def _lerobot_dir(tmp_path: Path) -> Path:
    """构造 LeRobot v2 布局目录。"""
    root = tmp_path / "lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        '{"fps": 25, "video": {"fps": 60}, "features": {"observation.images.wrist": {"dtype": "video"}}}',
        encoding="utf-8",
    )
    pd.DataFrame({"timestamp_ns": [0, 1_000_000, 2_000_000], "obs": [0, 1, 2]}).to_parquet(
        root / "data/chunk-000/episode_000000.parquet"
    )
    (root / "data/chunk-000/episode_000000.json").write_text(
        '[{"timestamp_ns": 0, "obs": 0}, {"timestamp_ns": 1000000, "obs": 1}, {"timestamp_ns": 2000000, "obs": 2}]',
        encoding="utf-8",
    )
    (root / "videos/chunk-000/observation.images.wrist.mp4").write_bytes(b"f")
    return root


def test_detect_lerobot_format() -> None:
    """目录指纹 meta/info.json + data/chunk-* → lerobot。"""
    import tempfile
    from app.tools._sniffing import probe_directory

    with tempfile.TemporaryDirectory() as td:
        root = _lerobot_dir(Path(td))
        probe = probe_directory(root)
    assert detect_dataset_format(probe) == "lerobot"


def test_parse_info_and_fps_explanation() -> None:
    """info.json 解析 fps/video.fps；帧数不一致时给确定性解释。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = _lerobot_dir(Path(td))
        info = parse_lerobot_info(str(root / "meta/info.json"))
    assert info["fps"] == 25
    assert info["video_fps"] == 60
    # 视频 60fps vs 数据 25Hz → 比例 2.4；表格 3 行 → 期望视频 7 帧。
    expl = explain_fps_mismatch(3, 7, info)
    assert expl is not None
    assert "比例" in expl["note"]
    assert expl["ratio"] == 2.4


def test_episode_mirror_detected() -> None:
    """episode_*.json 与同名 parquet 行数一致 → mirror。"""
    import tempfile
    from app.tools._sniffing import probe_directory

    with tempfile.TemporaryDirectory() as td:
        root = _lerobot_dir(Path(td))
        probe = probe_directory(root)
        mirrors = detect_episode_mirrors(probe)
    assert len(mirrors) == 1
    assert mirrors[0]["rows"] == 3
    assert mirrors[0]["json"].endswith("episode_000000.json")
    assert mirrors[0]["parquet"].endswith("episode_000000.parquet")


def test_lerobot_load_metadata_not_in_streams(tmp_path: Path) -> None:
    """LeRobot 加载：meta/info.json 不进入流清单（dataset_metadata），不崩。"""
    root = _lerobot_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    caps = ctx.meta.get("capabilities", {})
    assert caps.get("dataset_format") == "lerobot"
    assert ctx.meta.get("guessed_type") == "LeRobot"
    # meta/info.json 不进流清单。
    assert not any(s["path"].endswith("info.json") for s in ctx.meta["streams"])
    # 视频与 episode parquet 正常登记。
    names = [Path(s["path"]).name for s in ctx.meta["streams"]]
    assert "episode_000000.parquet" in names
    assert "observation.images.wrist.mp4" in names
