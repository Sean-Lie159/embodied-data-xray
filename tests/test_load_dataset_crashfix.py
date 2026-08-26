"""Commit 1 崩溃修复的单元测试。

覆盖：空 JSON 数组/对象→空流不崩；desktop.ini 等系统文件被跳过不参与探测；
.index.parquet 双扩展名正确识别为 parquet。不依赖真实网络。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl


def _load_dir(tmp_path: Path, name: str) -> dict:
    root = tmp_path / name
    root.mkdir()
    return root, load_dataset_impl(RunContext(output_dir=str(tmp_path)), str(root))


def test_empty_json_array_marks_empty_stream(tmp_path: Path) -> None:
    """空 JSON 数组 [] → 标记为空流，不抛异常。"""
    root = tmp_path / "d"
    root.mkdir()
    (root / "empty_arr.json").write_text("[]", encoding="utf-8")
    pd.DataFrame({"episode": [0, 1, 2], "qpos1": [0.1, 0.2, 0.3]}).to_csv(root / "state.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    empty = next(s for s in ctx.meta["streams"] if s["path"].endswith("empty_arr.json"))
    assert empty["status"] == "empty"
    assert "空" in empty["semantic_label"] or "未使用" in empty["semantic_label"]


def test_empty_json_object_marks_empty_stream(tmp_path: Path) -> None:
    """空 JSON 对象 {} → 标记为空流，不抛异常。"""
    root = tmp_path / "d"
    root.mkdir()
    (root / "empty_obj.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({"episode": [0, 1, 2], "qpos1": [0.1, 0.2, 0.3]}).to_csv(root / "state.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    empty = next(s for s in ctx.meta["streams"] if s["path"].endswith("empty_obj.json"))
    assert empty["status"] == "empty"


def test_system_files_skipped(tmp_path: Path) -> None:
    """desktop.ini / Thumbs.db / .DS_Store 被跳过，不进文件清单、不参与探测。"""
    root = tmp_path / "d"
    root.mkdir()
    (root / "desktop.ini").write_text("[ViewState]", encoding="utf-8")
    (root / "Thumbs.db").write_bytes(b"\x00\x01")
    (root / ".DS_Store").write_bytes(b"\x00")
    (root / "Thumbs.db").write_bytes(b"\x00")
    pd.DataFrame({"episode": [0, 1, 2], "qpos1": [0.1, 0.2, 0.3]}).to_csv(root / "state.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    # 只有 state.csv 被登记，系统文件未进清单。
    names = [Path(s["path"]).name for s in ctx.meta["streams"]]
    assert "state.csv" in names
    assert "desktop.ini" not in names
    assert "Thumbs.db" not in names
    assert ".DS_Store" not in names
    assert r["file_survey"]["total_files"] == 1


def test_double_extension_parquet_recognized(tmp_path: Path) -> None:
    """.index.parquet 双扩展名：Path.suffix 取最后一个 → 正确识别为 parquet。"""
    root = tmp_path / "d"
    root.mkdir()
    # 写一个最小 parquet（用 pyarrow 或 pandas）。
    df = pd.DataFrame({"episode": [0, 1], "qpos1": [0.1, 0.2]})
    df.to_parquet(root / "idx.index.parquet")
    pd.DataFrame({"episode": [0, 1, 2], "qpos1": [0.1, 0.2, 0.3]}).to_csv(root / "state.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    # .index.parquet 被识别为表格流（parquet）。
    parq = next((s for s in ctx.meta["streams"] if s["path"].endswith(".index.parquet")), None)
    assert parq is not None, ".index.parquet 应被登记为表格流"
    assert parq["format"] == "parquet"
    # 不崩，normal 文件也在。
    assert any(s["path"].endswith("state.csv") for s in ctx.meta["streams"])
