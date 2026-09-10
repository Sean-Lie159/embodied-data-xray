"""视频帧抽查（inspect_video_frame）的单元测试。

背景（缺口 1）：项目对视频只有 ffprobe 元数据，"这路相机对着哪里/画面有没有
被遮挡"无法回答——而这正是判断相机语义的关键证据。完整视频理解成本高，
抽一帧看图即可覆盖绝大多数确认需求。

纪律：只读抽取（不改源视频）、不做内容理解（画面含义由用户判读）、
ffmpeg 不可用时结构化降级。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.agent.context import RunContext
from app.tools.inspect_video_frame import (
    _ffmpeg_available,
    inspect_video_frame_impl,
)

T0_MS = 1_788_426_767_625

_FFMPEG = _ffmpeg_available()


def _make_test_video(path: Path, seconds: int = 2, fps: int = 10) -> bool:
    """用 ffmpeg 生成合成测试视频（含视觉可辨的时间戳文字）。"""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", f"testsrc=duration={seconds}:size=160x120:rate={fps}",
             "-pix_fmt", "yuv420p", str(path)],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode == 0 and path.exists()
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _FFMPEG, reason="环境无 ffmpeg，抽帧功能不可用")
def test_extract_frame_creates_image(tmp_path: Path) -> None:
    """端到端：抽帧成功，产出图片文件且尺寸正确。"""
    video = tmp_path / "cam.mp4"
    if not _make_test_video(video):
        pytest.skip("合成测试视频失败")

    ctx = RunContext(output_dir=str(tmp_path))
    r = inspect_video_frame_impl(ctx, str(video), at_seconds=1.0)
    assert r["success"] is True, r.get("user_message")
    assert r["at_seconds"] == 1.0
    assert r["width"] == 160 and r["height"] == 120
    img = tmp_path / r["frame_path"]
    assert img.exists() and img.stat().st_size > 0
    # 只读：源视频未被改动。
    assert video.exists()


@pytest.mark.skipif(not _FFMPEG, reason="环境无 ffmpeg，抽帧功能不可用")
def test_at_seconds_clamped_to_duration(tmp_path: Path) -> None:
    """时刻越界 → 夹到有效范围（不报错）。"""
    video = tmp_path / "cam.mp4"
    if not _make_test_video(video, seconds=2):
        pytest.skip("合成测试视频失败")
    ctx = RunContext(output_dir=str(tmp_path))
    r = inspect_video_frame_impl(ctx, str(video), at_seconds=999.0)
    assert r["success"] is True
    assert r["at_seconds"] < 2.0
    assert r["duration_s"] == pytest.approx(2.0, abs=0.3)


@pytest.mark.skipif(not _FFMPEG, reason="环境无 ffmpeg，抽帧功能不可用")
def test_resolve_by_stream_registry_name(tmp_path: Path) -> None:
    """按流登记表的文件名解析视频（用户/AI 常只知道文件名）。"""
    d = tmp_path / "ds"
    d.mkdir()
    video = d / "camera-rtsp-ll-hand.mp4"
    if not _make_test_video(video):
        pytest.skip("合成测试视频失败")
    (d / "state.csv").write_text("episode,qpos1\n0,0.1\n", encoding="utf-8")

    ctx = RunContext(output_dir=str(tmp_path))
    from app.tools.load_dataset import load_dataset_impl

    assert load_dataset_impl(ctx, str(d))["success"] is True
    # 传文件名（非全路径）——应从流登记表解析到实际路径。
    r = inspect_video_frame_impl(ctx, "camera-rtsp-ll-hand.mp4", at_seconds=0.5)
    assert r["success"] is True, r.get("user_message")
    assert r["video"] == "camera-rtsp-ll-hand.mp4"


def test_video_not_found_structured(tmp_path: Path) -> None:
    """视频不存在 → 结构化错误（指向 inspect_streams 的视频清单）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    r = inspect_video_frame_impl(ctx, "no/such.mp4", at_seconds=0)
    assert r["success"] is False and r["error"] == "video_not_found"


def test_ffmpeg_unavailable_degraded(tmp_path: Path, monkeypatch) -> None:
    """ffmpeg 不可用 → 结构化降级提示（含安装指引），不崩。

    注意：app.tools 包导出的 inspect_video_frame FunctionTool 会遮蔽同名
    模块，必须经 sys.modules 取真实模块再 patch（项目既有约定）。
    """
    import sys

    m = sys.modules["app.tools.inspect_video_frame"]

    monkeypatch.setattr(m, "_ffmpeg_available", lambda: False)
    video = tmp_path / "cam.mp4"
    video.write_bytes(b"fake")
    ctx = RunContext(output_dir=str(tmp_path))
    r = inspect_video_frame_impl(ctx, str(video), at_seconds=0)
    assert r["success"] is False and r["error"] == "ffmpeg_unavailable"
    assert "ffmpeg" in r["user_message"]


@pytest.mark.skipif(not _FFMPEG, reason="环境无 ffmpeg，抽帧功能不可用")
def test_frame_saved_under_output_dir(tmp_path: Path) -> None:
    """图片落在 output_dir 下（不污染数据集源目录）。"""
    d = tmp_path / "ds"
    d.mkdir()
    video = d / "cam.mp4"
    if not _make_test_video(video):
        pytest.skip("合成测试视频失败")
    out = tmp_path / "outputs"
    ctx = RunContext(output_dir=str(out))
    r = inspect_video_frame_impl(ctx, str(video), at_seconds=0.5)
    assert r["success"] is True
    assert (out / r["frame_path"]).exists()
    # 源目录只多了视频本身（无图片落盘）。
    assert [p.name for p in d.iterdir()] == ["cam.mp4"]


def test_custom_output_name(tmp_path: Path) -> None:
    """output_name 生效（自定义文件名）。"""
    if not _FFMPEG:
        pytest.skip("环境无 ffmpeg")
    video = tmp_path / "cam.mp4"
    if not _make_test_video(video):
        pytest.skip("合成测试视频失败")
    ctx = RunContext(output_dir=str(tmp_path))
    r = inspect_video_frame_impl(ctx, str(video), at_seconds=0.5,
                                 output_name="my_frame")
    assert r["success"] is True
    assert r["frame_path"] == "my_frame.jpg"
