"""输出路径集中管理模块的单测（docs/UI优化总纲与输出目录改造设计.md 第 3.7 节）。

覆盖：目录名净化（危险字符/超长截断/不碰撞/空名兜底/幂等）、
图表与报告落在 by_dataset/<净名>/{charts,reports}/、文件名保留 session_tag 前缀。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.agent.context import RunContext
from app.tools.output_paths import (
    chart_dir,
    dataset_output_dir,
    report_dir,
    sanitize_dataset_dir_name,
)


def test_sanitize_strips_dangerous_chars() -> None:
    """路径分隔符与 Windows 保留字符被替换为下划线。"""
    name = sanitize_dataset_dir_name('a/b\\c:d*e?f"g<h>i|j')
    assert "/" not in name and "\\" not in name
    assert not any(c in name for c in '<>:"|?*')


def test_sanitize_truncates_long_name() -> None:
    """超长 dataset_id 的净名前缀被截断到 60 字符（不含哈希后缀）。"""
    long_id = "x" * 200
    name = sanitize_dataset_dir_name(long_id)
    prefix = name.rsplit("-", 1)[0]
    assert len(prefix) == 60
    assert len(name) == 60 + 1 + 6  # 60 + "-" + 6 位哈希


def test_sanitize_different_long_ids_do_not_collide() -> None:
    """两个前 60 位相同、尾部分不同的长 id 必须映射到不同目录（防串目录）。"""
    common = "y" * 80
    a = sanitize_dataset_dir_name(common + "_run1")
    b = sanitize_dataset_dir_name(common + "_run2")
    assert a != b


def test_sanitize_empty_falls_back_to_misc() -> None:
    """None / 空串 → 固定 _misc（未加载数据集的兜底）。"""
    assert sanitize_dataset_dir_name(None) == "_misc"
    assert sanitize_dataset_dir_name("") == "_misc"
    assert sanitize_dataset_dir_name("   ") == "_misc"


def test_sanitize_is_idempotent_on_cleaned_input() -> None:
    """净化结果再净化一次（作为 dataset_id 传入）语义稳定——同 id 稳定映射。"""
    name1 = sanitize_dataset_dir_name("lerobot")
    name2 = sanitize_dataset_dir_name("lerobot")
    assert name1 == name2
    assert name1.startswith("lerobot-")


def test_dataset_output_dir_structure(tmp_path: Path) -> None:
    """产物根目录为 outputs/by_dataset/<净名>/，且自动创建。"""
    d = dataset_output_dir(str(tmp_path), "lerobot")
    assert d.exists()
    assert d.parent.name == "by_dataset"
    assert d.name == sanitize_dataset_dir_name("lerobot")


def test_chart_and_report_dirs_are_siblings(tmp_path: Path) -> None:
    """图表与报告分别落在 charts/ 与 reports/，同属该数据集目录。"""
    cd = chart_dir(str(tmp_path), "lerobot")
    rd = report_dir(str(tmp_path), "lerobot")
    assert cd.name == "charts" and rd.name == "reports"
    assert cd.parent == rd.parent


def test_chart_path_keeps_session_tag_and_dataset_id(tmp_path: Path) -> None:
    """多会话隔离契约不回退：文件名仍含 session_tag 前缀与 dataset_id。"""
    from app.tools.plot_chart import _output_path

    ctx = RunContext(output_dir=str(tmp_path), dataset_id="lerobot",
                     session_tag="s-1a2b")
    p = _output_path(ctx, "line")
    assert p.name.startswith("s-1a2b_lerobot_line_")
    assert p.suffix == ".png"
    assert p.parent.name == "charts"
    assert p.parent.parent.name == sanitize_dataset_dir_name("lerobot")


def test_report_path_keeps_session_tag_and_dataset_id(tmp_path: Path) -> None:
    """报告文件名同样保留前缀，落在 reports/。"""
    from app.tools.generate_report import _output_report_path

    ctx = RunContext(output_dir=str(tmp_path), dataset_id="lerobot",
                     session_tag="s-1a2b")
    p = _output_report_path(ctx)
    assert p.name.startswith("s-1a2b_lerobot_report_")
    assert p.suffix == ".md"
    assert p.parent.name == "reports"


def test_no_dataset_goes_to_misc(tmp_path: Path) -> None:
    """未加载数据集时产物落 _misc/，不报错。"""
    from app.tools.plot_chart import _output_path

    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None, session_tag="")
    p = _output_path(ctx, "histogram")
    assert "_misc" in p.parts
    assert p.name.startswith("dataset_histogram_")


def test_dir_name_has_no_regex_artifacts() -> None:
    """净名不含连续分隔/残留的特殊字符（净化规则无残留）。"""
    name = sanitize_dataset_dir_name("a::b//c")
    assert "::" not in name and "//" not in name
    assert re.match(r"^[\w.\-\u4e00-\u9fff]+$", name)
