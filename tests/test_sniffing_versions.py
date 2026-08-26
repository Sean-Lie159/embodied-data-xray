"""Commit 2：多分辨率版本组 / 同名配对 / nuScenes 登记级的单元测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools._sniffing import (
    _video_is_main_version,
    pair_streams,
)
from app.tools.inspect_streams import inspect_streams_impl
from app.tools.load_dataset import load_dataset_impl


def test_video_is_main_version() -> None:
    """主版本（无分辨率/预览后缀）判定。"""
    assert _video_is_main_version("/d/camera-rtsp-ll-hand.mp4") is True
    assert _video_is_main_version("/d/camera_480.mp4") is False
    assert _video_is_main_version("/d/camera_pre.mp4") is False
    assert _video_is_main_version("/d/camera_960.mp4") is False


def test_pair_streams_version_group() -> None:
    """xxx.mp4 / xxx_480.mp4 / xxx_960.mp4 / xxx_pre.mp4 归为同一视频源版本组。"""
    videos = [
        "/d/camera-rtsp-ll-hand.mp4",
        "/d/camera-rtsp-ll-hand_480.mp4",
        "/d/camera-rtsp-ll-hand_960.mp4",
        "/d/camera-rtsp-ll-hand_pre.mp4",
    ]
    pairs = pair_streams(videos, [], [])
    vg = [p for p in pairs if p["type"] == "video_version_group"]
    assert len(vg) == 1
    group = vg[0]
    assert group["main"] == "/d/camera-rtsp-ll-hand.mp4"
    assert len(group["variants"]) == 3
    assert all(v["variant_of"] == "/d/camera-rtsp-ll-hand.mp4" for v in group["variants"])


def test_pair_streams_samename_json_mp4() -> None:
    """同名不同扩展：<name>.mp4 ↔ <name>.json 配对（metainfo 规则推广）。"""
    videos = ["/d/cam1.mp4"]
    tables = ["/d/cam1.json"]
    pairs = pair_streams(videos, tables, [])
    sn = [p for p in pairs if p["type"] == "media_samename"]
    assert len(sn) == 1
    assert sn[0]["media"].endswith("cam1.mp4")
    assert sn[0]["json"].endswith("cam1.json")


def _version_dir(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    root.mkdir()
    for name in ("camera-rtsp-ll-hand.mp4", "camera-rtsp-ll-hand_480.mp4",
                 "camera-rtsp-ll-hand_pre.webp"):
        (root / name).write_bytes(b"fakemedia")
    pd.DataFrame({"episode": [0, 1, 2], "qpos1": [0.1, 0.2, 0.3]}).to_csv(root / "state.csv", index=False)
    return root


def test_inspect_streams_surfaces_variant_of(tmp_path: Path) -> None:
    """inspect_streams 对非主版本视频标 variant_of，主版本不标。"""
    root = _version_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    insp = inspect_streams_impl(ctx)
    vs = {Path(v["source"]).name: v for v in insp["video_streams"]}
    main = vs["camera-rtsp-ll-hand.mp4"]
    assert main.get("variant_of") is None  # 主版本无 variant_of
    v480 = vs["camera-rtsp-ll-hand_480.mp4"]
    assert v480.get("variant_of") == "camera-rtsp-ll-hand"
    # _pre.webp 按图片扩展名登记为 image 流（.webp 属 _IMAGE_EXTS），不作为视频变体；
    # 视频变体仅对 mp4 版本组（_480/_960/_pre.mp4）标 variant_of。


def test_nuscenes_json_registers_not_crash(tmp_path: Path) -> None:
    """nuScenes schema 表族（sample.json 等 dict 结构）登记为流，不崩；ego_pose 空文件→空流。"""
    root = tmp_path / "nu"
    root.mkdir()
    # nuScenes 表：dict 结构（含 data 列表）。
    (root / "sample.json").write_text('{"__version__": "1.0", "data": []}', encoding="utf-8")
    (root / "sample_data.json").write_text('{"data": []}', encoding="utf-8")
    (root / "sensor.json").write_text('{"data": []}', encoding="utf-8")
    # 2 字节空文件（如 ego_pose.json）。
    (root / "ego_pose.json").write_bytes(b"{}")
    pd.DataFrame({"episode": [0, 1, 2], "qpos1": [0.1, 0.2, 0.3]}).to_csv(root / "state.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True  # 不崩
    names = [Path(s["path"]).name for s in ctx.meta["streams"]]
    # nuScenes JSON 与 ego_pose 均登记在册（空流标注），不被丢弃、不崩溃。
    assert "sample.json" in names
    assert "sample_data.json" in names
    assert "sensor.json" in names
    assert "ego_pose.json" in names
