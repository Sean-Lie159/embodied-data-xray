"""app/tools/load_dataset 目录加载功能的单元测试。

构造小型合成目录（含 IMU csv / 标定 yaml / 假视频文件），验证文件普查、
能力嗅探、标定检测、视频 ffprobe 降级与 RunContext.meta 写入。
"""

from __future__ import annotations

from pathlib import Path

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl


def _make_dataset(tmp_path: Path, name: str = "robot_dataset") -> Path:
    """构造一个小型合成数据集目录。"""
    root = tmp_path / name
    (root / "imu").mkdir(parents=True)
    (root / "calib").mkdir()
    (root / "videos").mkdir()

    (root / "imu" / "imu.csv").write_text(
        "accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
        "0,0,0,0,0,0\n"
        "1,1,1,1,1,1\n"
        "2,2,2,2,2,2\n",
        encoding="utf-8",
    )
    (root / "calib" / "cam0.yaml").write_text(
        "intrinsic:\n  fx: 500\n  fy: 500\n  cx: 320\n  cy: 240\n",
        encoding="utf-8",
    )
    (root / "videos" / "clip.mp4").write_bytes(b"fakemovie")
    return root


def test_load_directory_probes_and_sets_context(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(root))

    assert result["success"] is True
    assert result["dataset_id"] == "robot_dataset"
    assert result["kind"] == "directory"

    # 文件普查摘要。
    survey = result["file_survey"]
    assert survey["total_files"] == 3
    assert set(survey["subdirs"]) == {"imu", "calib", "videos"}

    # 能力标签。
    caps = result["capabilities"]
    assert caps["has_imu"] is True
    assert caps["imu_axes"] == 6  # accel+gyro → 6 轴
    assert caps["has_calibration"] is True
    assert caps["has_video_streams"] is True

    # 推测类型与置信度。
    assert result["guessed_type"] in {"Ego", "unknown"}
    assert 0.0 <= result["guessed_type_confidence"] <= 1.0

    # 视频只记录路径清单与元数据（不读入内存）。
    assert len(result["video_files"]) == 1
    assert result["video_files"][0].endswith("clip.mp4")
    assert "video_meta" in result  # ffprobe 不可用时元数据为降级标记


def test_load_directory_writes_context_meta(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    ctx = RunContext()

    load_dataset_impl(ctx, str(root))

    assert ctx.dataset_id == "robot_dataset"
    assert ctx.meta["capabilities"]["has_imu"] is True
    assert ctx.meta["capabilities"]["imu_axes"] == 6
    assert ctx.meta["guessed_type"] is not None
    # 视频文件路径已记录在 meta，但不读入内存。
    assert ctx.meta["video_files"][0].endswith("clip.mp4")


def test_load_directory_detects_calibration(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(root))

    assert result["capabilities"]["has_calibration"] is True


def test_load_directory_ffprobe_degraded_when_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    """ffprobe 不可用时，应返回降级提示而非失败。"""
    root = _make_dataset(tmp_path)
    ctx = RunContext()
    monkeypatch.setattr("shutil.which", lambda name: None)  # 模拟无 ffprobe

    result = load_dataset_impl(ctx, str(root))

    assert result["success"] is True  # 目录加载不因视频嗅探失败而中断
    assert "ffprobe" in str(result.get("ffprobe_degraded", "")).lower() or \
        any("ffprobe" in str(v).lower() for v in result.get("video_files", []))


def test_load_directory_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(empty))

    assert result["success"] is True
    assert result["file_survey"]["total_files"] == 0
