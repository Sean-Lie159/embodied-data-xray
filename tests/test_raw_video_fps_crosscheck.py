"""裸流 fps 与同名时间戳清单的交叉校验（2026-09-20 用户实测）。

背景：`head_stereo_left/right` 视频裸流被 ffprobe 报 **60 fps**，用户据此追问
"与 30 Hz 时间戳清单差一倍，是否为了对齐而降采样"。全量解码清点后两路均为
**14142 帧**（与 `.txt` 的 14142 行逐帧对应）——实际就是 30 fps。

结论：**裸码流的 `r_frame_rate` 也会被码流头信息误导**，不能单独采信。
而采集端的逐帧时间戳清单（`<相机>.txt`）是真实帧时刻的直接证据，且与视频同名
同目录——本改动用它做零成本交叉校验（仅读首末两行）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools._sniffing import _fps_from_timestamp_sidecar


def _make_h265_stub(path: Path) -> None:
    """写一个"像裸流"的文件（内容不重要——fps 由同名清单校验）。"""
    path.write_bytes(b"\x00\x00\x00\x01fake-hevc")


def _write_sidecar(video: Path, n: int, step_ns: int, start_ns: int = 1_789_000_000_000_000_000) -> None:
    """写同名时间戳清单：n 行、间隔 step_ns（纳秒）。"""
    rows = [f"{start_ns + i * step_ns} P" for i in range(n)]
    video.with_suffix(".txt").write_text("\n".join(rows), encoding="utf-8")


# --- 1. 清单帧率推算 --------------------------------------------------------


def test_fps_from_sidecar_30hz(tmp_path: Path) -> None:
    """30 Hz 清单（33.33 ms 间隔）→ 推算 30 fps。"""
    v = tmp_path / "cam.h265"
    _make_h265_stub(v)
    _write_sidecar(v, n=100, step_ns=33_333_333)
    fps = _fps_from_timestamp_sidecar(str(v))
    assert fps is not None
    assert fps == pytest.approx(30.0, rel=0.01)


def test_fps_from_sidecar_60hz(tmp_path: Path) -> None:
    """60 Hz 清单 → 推算 60 fps（校验函数本身不偏向 30）。"""
    v = tmp_path / "cam60.h265"
    _make_h265_stub(v)
    _write_sidecar(v, n=100, step_ns=16_666_666)
    fps = _fps_from_timestamp_sidecar(str(v))
    assert fps == pytest.approx(60.0, rel=0.01)


def test_fps_from_sidecar_microsecond_unit(tmp_path: Path) -> None:
    """微秒量级清单也能正确换算（量级判定分支）。"""
    v = tmp_path / "us.h265"
    _make_h265_stub(v)
    _write_sidecar(v, n=50, step_ns=0, start_ns=1_789_000_000_000_000)  # 1e15 → 微秒
    rows = [f"{1_789_000_000_000_000 + i * 33_333} P" for i in range(50)]
    v.with_suffix(".txt").write_text("\n".join(rows), encoding="utf-8")
    fps = _fps_from_timestamp_sidecar(str(v))
    assert fps == pytest.approx(30.0, rel=0.02)


def test_fps_from_sidecar_absent(tmp_path: Path) -> None:
    """无同名清单 → None（不猜）。"""
    v = tmp_path / "bare.h265"
    _make_h265_stub(v)
    assert _fps_from_timestamp_sidecar(str(v)) is None


def test_fps_from_sidecar_too_few_rows(tmp_path: Path) -> None:
    """清单行数不足 → None（无法推算，不猜）。"""
    v = tmp_path / "one.h265"
    _make_h265_stub(v)
    _write_sidecar(v, n=1, step_ns=33_333_333)
    assert _fps_from_timestamp_sidecar(str(v)) is None


def test_fps_from_sidecar_unknown_magnitude(tmp_path: Path) -> None:
    """时间戳量级无法判定（既非 ns/us/ms epoch）→ None（宁可不用也不猜）。"""
    v = tmp_path / "small.h265"
    _make_h265_stub(v)
    _write_sidecar(v, n=100, step_ns=33_333_333, start_ns=1000)
    assert _fps_from_timestamp_sidecar(str(v)) is None


def test_fps_from_sidecar_broken_content(tmp_path: Path) -> None:
    """清单内容不可解析 → None（不抛异常）。"""
    v = tmp_path / "bad.h265"
    _make_h265_stub(v)
    v.with_suffix(".txt").write_text("not-a-number P\nnor-this P\n", encoding="utf-8")
    assert _fps_from_timestamp_sidecar(str(v)) is None


# --- 2. probe_video 集成（需 ffmpeg）---------------------------------------


@pytest.fixture()
def raw_h265_with_mismatch(tmp_path: Path) -> Path:
    """合成一个真实裸流，并写一份"帧率与 ffprobe 不一致"的同名清单。

    做法：用 ffmpeg 生成 30fps 的裸流，但清单按 **60 Hz** 写——这复现了
    "ffprobe 报的与清单不符"的形态（真实事故里方向相反，但校验逻辑相同：
    以清单为准并透出 fp_conflict）。
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("需要 ffmpeg/ffprobe")
    src = tmp_path / "src.mp4"
    raw = tmp_path / "mismatch.h265"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, timeout=120, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx265", "-f", "hevc", str(raw)],
        capture_output=True, timeout=300, check=True,
    )
    # 清单按 60 Hz 写（与裸流的 30fps 冲突）。
    _write_sidecar(raw, n=120, step_ns=16_666_666)
    return raw


def test_probe_video_trusts_sidecar_on_conflict(raw_h265_with_mismatch: Path) -> None:
    """**核心回归**：probe 与清单矛盾时以**清单为准**并透出 `fps_conflict`。"""
    from app.tools._sniffing import probe_video

    meta = probe_video(str(raw_h265_with_mismatch))
    assert meta.get("ffprobe_available") is True
    # 采用清单值（60 Hz），而不是 ffprobe 的 30 fps。
    assert meta["fps"] == pytest.approx(60.0, rel=0.02), meta.get("fps_source")
    assert "timestamp_sidecar" in str(meta.get("fps_source"))
    conf = meta.get("fps_conflict")
    assert conf, "矛盾未透出"
    assert conf["probe_fps"] != conf["sidecar_fps"]
    assert "清单" in conf["note"]
    assert meta.get("fps_from_timestamp_sidecar") is not None


def test_probe_video_no_conflict_when_consistent(tmp_path: Path) -> None:
    """probe 与清单一致时**不产生** conflict（防噪声）。"""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("需要 ffmpeg/ffprobe")
    src = tmp_path / "s.mp4"
    raw = tmp_path / "ok.h265"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, timeout=120, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx265", "-f", "hevc", str(raw)],
        capture_output=True, timeout=300, check=True,
    )
    # 清单按 30 Hz 写（与裸流一致）。
    _write_sidecar(raw, n=60, step_ns=33_333_333)

    from app.tools._sniffing import probe_video

    meta = probe_video(str(raw))
    # 30 fps 与清单一致 → 不报冲突。
    assert "fps_conflict" not in meta, meta.get("fps_conflict")
    assert meta["fps"] == pytest.approx(30.0, abs=1.0)


def test_probe_video_container_unaffected(tmp_path: Path) -> None:
    """容器视频（mp4）不受该校验影响（走 avg_frame_rate，无 sidecar 概念）。"""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("需要 ffmpeg")
    mp4 = tmp_path / "c.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True, timeout=120, check=True,
    )
    from app.tools._sniffing import probe_video

    meta = probe_video(str(mp4))
    assert meta.get("ffprobe_available") is True
    assert meta["fps"] == pytest.approx(25.0, abs=0.5)
    assert meta.get("fps_source") == "ffprobe_avg_frame_rate"
    assert "fps_conflict" not in meta
