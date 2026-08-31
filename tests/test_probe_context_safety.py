"""文件清单上下文安全（三层防御）的单元测试。

覆盖：
1. 默认排除非数据目录（.git/__pycache__/node_modules 等），且排除目录只记根（不
   枚举其内部数万对象）；include_hidden=True 可纳入。
2. 分组路径截断：超上限时 truncated=True 且 shown/total 准确（不静默抽样）。
3. 体积护栏：极端体积压缩为分组计数摘要，计数仍完整。
4. 完整性：total_files == 各组 total 之和（无静默抽样）。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.tools._sniffing import (
    _MAX_LISTED_PER_GROUP,
    probe_directory,
    probe_full_paths,
)
from app.tools.load_dataset import load_dataset_impl


def _make_dir(tmp_path: Path, n_other: int = 5, with_git: bool = True) -> Path:
    """构造小目录：含 .git 内部对象、数据表、视频，以及若干杂项文件。"""
    root = tmp_path / "ds"
    (root / "data").mkdir(parents=True)
    if with_git:
        # .git/objects/xx/yyy：模拟版本控制内部对象（应被排除）。
        for i in range(30):
            d = root / ".git" / "objects" / f"{i:02x}"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{'a' * 38}").write_bytes(b"x")
        (root / "__pycache__").mkdir(exist_ok=True)
        (root / "__pycache__" / "x.pyc").write_bytes(b"x")
    (root / "data" / "episode_000000.parquet").write_bytes(b"p")
    (root / "vid.mp4").write_bytes(b"v")
    for i in range(n_other):
        (root / f"misc_{i}.bin").write_bytes(b"o")
    return root


def test_default_excludes_non_data_dirs(tmp_path: Path) -> None:
    """默认排除 .git/__pycache__：其内部对象不计入 total_files。"""
    root = _make_dir(tmp_path)
    probe = probe_directory(root)
    # 30 个 .git 对象 + 1 个 pyc 被排除；剩余 = parquet + mp4 + 5 杂项 = 7。
    assert probe["total_files"] == 7
    excluded = probe["excluded_dirs"]
    assert ".git" in excluded
    assert "__pycache__" in excluded
    # 排除目录只记根：不枚举 .git/objects/xx 的子孙。
    assert all(p.count("\\") <= 1 and "objects" not in p for p in excluded)


def test_include_hidden_includes_excluded(tmp_path: Path) -> None:
    """include_hidden=True 时纳入被排除目录（用于需要完整普查的场景）。"""
    root = _make_dir(tmp_path)
    probe = probe_directory(root, include_hidden=True)
    # 30 个 .git 对象 + 1 pyc + 7 = 38。
    assert probe["total_files"] == 38
    assert probe["excluded_dirs"] == []


def test_group_truncation_marks_and_counts(tmp_path: Path) -> None:
    """分组超上限：truncated=True，shown/total 准确（不静默）。"""
    root = tmp_path / "many"
    root.mkdir()
    n = _MAX_LISTED_PER_GROUP + 20
    for i in range(n):
        (root / f"f_{i:03d}.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    probe = probe_directory(root)
    view = probe["tables"]
    assert view["total"] == n
    assert view["shown"] == _MAX_LISTED_PER_GROUP
    assert view["truncated"] is True
    assert len(view["paths"]) == _MAX_LISTED_PER_GROUP
    # 完整清单仍在（供代码探测），未被截断影响。
    assert len(probe_full_paths(probe, "tables")) == n


def test_counts_complete_no_silent_sampling(tmp_path: Path) -> None:
    """total_files == 各组 total 之和（无静默抽样）。"""
    root = _make_dir(tmp_path)
    probe = probe_directory(root)
    s = sum(
        probe[g]["total"]
        for g in ("tables", "videos", "audios", "images", "cals", "others")
    )
    assert s == probe["total_files"]


def test_survey_size_within_budget_on_project_like_dir(tmp_path: Path) -> None:
    """含 .git 的目录：返回体积应在预算内（不得达到百万字符级）。"""
    root = _make_dir(tmp_path, n_other=80)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    survey = r["file_survey"]
    size = len(json.dumps(survey, ensure_ascii=False, default=str))
    # 排除 .git 后体积可控（护栏阈值 50_000 以内）。
    assert size < 50_000, f"file_survey 体积 {size} 超预算"
    assert survey["total_files"] == 82  # parquet + mp4 + 80 杂项
    # 未触发压缩（体积在预算内，路径保留）。
    assert survey.get("note") is None


def test_excluded_dirs_surfaced_in_user_message(tmp_path: Path) -> None:
    """排除的目录在 user_message 中透明告知（不静默）。"""
    root = _make_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert "跳过" in r["user_message"]
    assert "excluded_dirs" in r["user_message"]
