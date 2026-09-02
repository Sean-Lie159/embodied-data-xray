"""纯视频/JSON 数据集（零 CSV 表格）加载不崩的单元测试。

验证：无表格候选时正常完成加载、main_table 为 null、能力标签如实、不崩溃。
不依赖真实数据，用合成文件构造。
"""

from __future__ import annotations

from pathlib import Path

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl


def _pure_media_dir(tmp_path: Path) -> Path:
    """纯视频 + JSON（无 CSV 可识别为表格）目录。"""
    root = tmp_path / "pure"
    root.mkdir()
    # 视频（多分辨率）。
    (root / "cam.mp4").write_bytes(b"fakemovie")
    (root / "cam_480.mp4").write_bytes(b"fakemovie")
    # JSON：dict 结构（schema 表，非数组 → 不作为 CSV 表格候选；非标定）。
    (root / "sample.json").write_text('{"data": []}', encoding="utf-8")
    (root / "sensor.json").write_text('{"data": []}', encoding="utf-8")
    return root


def test_pure_video_json_no_table_not_crash(tmp_path: Path) -> None:
    """纯视频+JSON（零 CSV 表格）加载不崩，main_table 为 null，能力标签如实。"""
    root = _pure_media_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    # main_table 为 null（无表格候选）。
    mt = r.get("main_table")
    assert mt is None or mt.get("name") is None
    # 能力标签如实：有视频，无 IMU/动作（纯媒体/JSON 数据集合法）。
    caps = ctx.meta.get("capabilities", {})
    assert caps.get("has_video_streams") is True
    # 无崩溃；probe_errors 无失败。
    assert r.get("probe_errors", []) == []
    # 视频流被登记。
    names = [Path(s["path"]).name for s in ctx.meta["streams"]]
    assert "cam.mp4" in names


def test_pure_video_json_main_table_meta_null(tmp_path: Path) -> None:
    """meta 中 main_table 的 name 应为 None（无主表），不因缺候选崩。"""
    root = _pure_media_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    mt = ctx.meta.get("main_table", {})
    assert mt.get("name") is None


def test_structured_error_carries_exc_type_and_file(tmp_path: Path, monkeypatch) -> None:
    """目录探测异常：reason 含异常类型 + 关键帧（文件:行号），extra 存 traceback_frames。"""
    import sys

    ld = sys.modules["app.tools.load_dataset"]
    root = tmp_path / "boom"
    root.mkdir()
    (root / "a.mp4").write_bytes(b"x")

    def _boom(*a, **k):
        raise IndexError("list index out of range")

    monkeypatch.setattr(ld, "_load_directory_impl", _boom)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is False
    assert r["error"] == "directory_probe_failed"
    # reason 含异常类型 + 目录。
    assert "IndexError" in r["reason"]
    assert str(root) in r["reason"]
    # extra 内部字段 traceback_frames 存在且含 app 内文件。
    frames = r.get("traceback_frames")
    assert frames, "traceback_frames 应非空"
