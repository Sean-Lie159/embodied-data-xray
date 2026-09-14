"""裸码流视频（.h265/.hevc）接入与抽帧测试（2026-09-14）。

背景（用户实测）：数据集 2655849 有 8 路相机、各一个 563~592 MB 的 ``.h265``。
该类文件是**无容器裸码流**，带来三个与常规 mp4 不同的行为：

  1. ``.h265`` 不在 ``_VIDEO_EXTS`` 内 → 视频流**完全不登记**（能力标签也报
     ``has_video_streams=False``），用户看不到任何相机；
  2. ``ffprobe`` 读不到 ``format.duration``（N/A）与 ``nb_frames``（缺失），
     且报的 ``avg_frame_rate=25/1`` 与真实的 ``r_frame_rate=30/1`` 不一致；
  3. **不支持输入侧 ``-ss`` 跳转**（无索引），且失败时 ``ffmpeg`` 以
     ``returncode=0`` 退出但**不产出任何帧**（0 字节）——只看返回码会把静默
     失败当成成功。

测试用 ffmpeg 现场合成的小裸流（约 12 KB），不依赖外部大文件。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.agent.context import RunContext
from app.tools._sniffing import _RAW_VIDEO_EXTS, _VIDEO_EXTS, probe_video

# 无 ffmpeg 的环境跳过整个模块（该能力本就依赖它）。
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="需要 ffmpeg/ffprobe",
)


@pytest.fixture(scope="module")
def raw_h265(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """合成一个 3 秒 30fps 的裸 HEVC 流（无容器索引）。"""
    d = tmp_path_factory.mktemp("video")
    src = d / "src.mp4"
    raw = d / "cam.h265"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc=size=160x120:rate=30:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)],
        capture_output=True, timeout=120, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx265",
         "-f", "hevc", str(raw)],
        capture_output=True, timeout=300, check=True,
    )
    return raw


@pytest.fixture(scope="module")
def raw_h265_with_txt(raw_h265: Path) -> Path:
    """裸流 + 同名时间戳清单（模拟真实采集形态）。"""
    # 先实测帧数，再按该帧数生成等长的时间戳清单。
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1",
         str(raw_h265)],
        capture_output=True, text=True, timeout=300, check=True,
    )
    n = int(p.stdout.strip())
    stamps = [1756265284737924649 + i * 33_333_333 for i in range(n)]
    raw_h265.with_suffix(".txt").write_text(
        "\n".join(f"{s} P" for s in stamps), encoding="utf-8")
    return raw_h265


# --- 1. 扩展名识别 ----------------------------------------------------------


def test_raw_video_extensions_registered() -> None:
    """`.h265`/`.hevc`/`.h264` 进 videos 分组（此前完全不登记）。"""
    assert ".h265" in _VIDEO_EXTS
    assert ".hevc" in _VIDEO_EXTS
    assert ".h264" in _VIDEO_EXTS
    assert {".h265", ".hevc", ".h264"} <= _RAW_VIDEO_EXTS


def test_probe_video_marks_raw_stream(raw_h265: Path) -> None:
    """裸流被标注 raw_stream 并给出"帧数需实测"的途径说明。"""
    meta = probe_video(str(raw_h265))
    assert meta.get("ffprobe_available") is True
    assert meta.get("raw_stream") is True
    assert "裸码流" in str(meta.get("raw_stream_note"))
    # 必须给出获取精确帧数的方式（不能让下游无从下手）。
    assert "count_frames" in str(meta.get("raw_stream_note"))


def test_probe_video_raw_uses_r_frame_rate(raw_h265: Path) -> None:
    """**关键**：裸流帧率取 r_frame_rate（30）而非 avg_frame_rate（裸流报 25）。

    真实数据实测：avg=25/1 而 r=30/1，且"解码帧数 ÷ 时长 ≈ 30 Hz"与相机
    时间戳清单的 30 Hz 吻合——故报称的 r_frame_rate 才是真实值。
    """
    meta = probe_video(str(raw_h265))
    assert meta.get("fps") == pytest.approx(30.0, abs=0.5)


def test_probe_video_raw_does_not_decode_for_frames(raw_h265: Path) -> None:
    """**性能红线**：探测阶段不得为了拿帧数而全量解码。

    真实事故：对 8 路 563 MB 裸流各做一次 -count_frames，加载耗时 742 秒
    （等同卡死）。故探测阶段帧数应为 None 并显式标注不可用。
    """
    meta = probe_video(str(raw_h265))
    assert meta.get("nb_frames") is None
    assert meta.get("nb_frames_unavailable") is True


def test_probe_video_container_stream_unchanged(raw_h265: Path, tmp_path: Path) -> None:
    """**零回归**：容器视频（mp4）仍走原有路径，不被标记为裸流。"""
    mp4 = tmp_path / "a.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc=size=160x120:rate=30:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True, timeout=120, check=True,
    )
    meta = probe_video(str(mp4))
    assert meta.get("ffprobe_available") is True
    assert not meta.get("raw_stream")
    assert meta.get("duration_s") is not None


# --- 2. 抽帧（裸流的输出侧 seek）------------------------------------------


def _ivf():
    """取 inspect_video_frame 模块（避开 app.tools 包属性遮蔽）。"""
    import sys

    import app.tools.inspect_video_frame  # noqa: F401

    return sys.modules["app.tools.inspect_video_frame"]


def test_extract_frame_from_raw_stream(raw_h265: Path, tmp_path: Path) -> None:
    """**核心回归**：裸流抽帧成功（此前因输入侧 -ss 报 "could not seek" 而失败）。"""
    mod = _ivf()
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    r = mod.inspect_video_frame_impl(ctx, str(raw_h265), at_seconds=1.0)
    assert r["success"] is True, r.get("user_message")
    assert Path(tmp_path / r["frame_path"].split("/")[-1]).exists() or (
        tmp_path / r["frame_path"]
    ).exists()
    assert r.get("raw_stream") is True


def test_extract_frame_first_frame(raw_h265: Path, tmp_path: Path) -> None:
    """at=0 也能出图（不传 -ss）。"""
    mod = _ivf()
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    r = mod.inspect_video_frame_impl(ctx, str(raw_h265), at_seconds=0.0)
    assert r["success"] is True


def test_extract_does_not_leave_empty_file(raw_h265: Path, tmp_path: Path) -> None:
    """**静默失败防护**：失败时不得留下 0 字节残留文件。

    真实行为：裸流 seek 失败时 ffmpeg 以 rc=0 退出但不产出帧，会留下空文件；
    只看 returncode 会把这种情况当成成功。
    """
    mod = _ivf()
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    # 请求一个远超时长的时刻；若夹取生效则会成功（这也是期望行为），
    # 故此处只断言"不留下空文件"这一不变量。
    mod.inspect_video_frame_impl(ctx, str(raw_h265), at_seconds=99_999.0)
    empties = [p for p in tmp_path.rglob("*.jpg") if p.stat().st_size == 0]
    assert empties == [], f"留下空文件：{empties}"


def test_out_of_range_is_clamped_with_txt(raw_h265_with_txt: Path,
                                          tmp_path: Path) -> None:
    """**越界夹取**：裸流 duration 为 N/A 时，用同名时间戳清单推算时长后夹取。

    若不夹取，ffmpeg 会一直解码到文件末尾才发现无帧（实测 19 秒后失败）。
    """
    mod = _ivf()
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    r = mod.inspect_video_frame_impl(
        ctx, str(raw_h265_with_txt), at_seconds=99_999.0)
    assert r["success"] is True, r.get("user_message")
    # 夹取后时刻必须是有限值（3 秒视频应落在 ~3s 内），绝不能是 99999。
    assert r["at_seconds"] < 10


def test_missing_video_reports_structured_error(tmp_path: Path) -> None:
    """不存在的视频返回结构化错误（不抛异常）。"""
    mod = _ivf()
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    r = mod.inspect_video_frame_impl(ctx, str(tmp_path / "nope.h265"))
    assert r["success"] is False
    assert r["error"] == "video_not_found"


# --- 3. 按需实测帧数（count_frames）---------------------------------------


def test_count_frames_off_by_default(raw_h265: Path, tmp_path: Path) -> None:
    """默认不做帧数实测（代价高：全量解码）。"""
    mod = _ivf()
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    r = mod.inspect_video_frame_impl(ctx, str(raw_h265), at_seconds=0.5)
    assert r["success"] is True
    assert "nb_frames_measured" not in r


def test_count_frames_measures_and_matches_timestamps(
    raw_h265_with_txt: Path, tmp_path: Path
) -> None:
    """**端到端**：count_frames=True 实测帧数，并与同名时间戳清单核对。

    这正是用户最初问题的答案形态："视频是否与时间戳逐帧对应"。
    """
    mod = _ivf()
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    r = mod.inspect_video_frame_impl(
        ctx, str(raw_h265_with_txt), at_seconds=0.5, count_frames=True)
    assert r["success"] is True
    n = r.get("nb_frames_measured")
    assert isinstance(n, int) and n > 0
    assert r.get("timestamp_rows") == n
    assert r.get("frames_match_timestamps") is True
    assert "一一对应" in str(r.get("user_message"))


def test_count_frames_reports_mismatch(raw_h265_with_txt: Path,
                                       tmp_path: Path) -> None:
    """帧数与时间戳不一致时**如实报不一致**（不掩盖）。"""
    # 删掉一行时间戳制造不一致。
    p = raw_h265_with_txt.with_suffix(".txt")
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-1]), encoding="utf-8")
    try:
        mod = _ivf()
        ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
        r = mod.inspect_video_frame_impl(
            ctx, str(raw_h265_with_txt), at_seconds=0.5, count_frames=True)
        assert r["success"] is True
        assert r.get("frames_match_timestamps") is False
        assert "不一致" in str(r.get("user_message"))
    finally:
        p.write_text("\n".join(lines), encoding="utf-8")


# --- 4. 目录加载：h265 进入视频流 ----------------------------------------


def test_directory_registers_h265_as_video(raw_h265: Path, tmp_path: Path) -> None:
    """目录加载把 .h265 登记为视频流，并置 has_video_streams=True。"""
    from app.tools.load_dataset import load_dataset_impl

    ds = tmp_path / "dataset"
    (ds / "camera").mkdir(parents=True)
    shutil.copy(raw_h265, ds / "camera" / "cam.h265")
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(ds))
    assert r["success"] is True
    assert (r.get("capabilities") or {}).get("has_video_streams") is True
    vstreams = [s for s in (ctx.meta.get("streams") or [])
                if s.get("kind") == "video"]
    assert any("cam.h265" in str(s.get("path")) for s in vstreams), (
        f"未登记为视频流：{[s.get('path') for s in ctx.meta.get('streams', [])]}"
    )
