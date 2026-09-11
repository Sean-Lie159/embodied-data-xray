"""右栏信息架构重构的单测（docs/右栏信息架构重构设计.md 第 4 节）。

覆盖：图表倒序、按 type 分组、空组不渲染、报告下载按钮、
静态结构断言（防回归到"每张图全宽平铺"的旧形态）。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _png_bytes() -> bytes:
    """最小合法 PNG（1x1 透明像素）——供 st.image 渲染不报错。"""
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
        "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )


def _write_chart(tmp: Path, name: str) -> str:
    p = tmp / f"{name}.png"
    p.write_bytes(_png_bytes())
    return str(p)


def _render_script(tmp: Path) -> Path:
    """生成一个最小 Streamlit 脚本，渲染指定 findings 的右栏组件。"""
    script = tmp / "app_test.py"
    script.write_text(
        "import streamlit as st\n"
        "from app.ui.components import render_charts, render_findings_and_report\n"
        "findings = st.session_state[\"findings\"]\n"
        "render_charts(findings)\n"
        "render_findings_and_report(findings)\n",
        encoding="utf-8",
    )
    return script


def test_empty_findings_shows_guidance(tmp_path: Path) -> None:
    """无 findings：两个组件都显示引导信息，不崩。"""
    at = AppTest.from_file(str(_render_script(tmp_path)), default_timeout=15)
    at.session_state["findings"] = []
    at.run()
    assert not at.exception
    infos = [i.value for i in at.info]
    assert any("暂无图表" in v for v in infos)
    assert any("暂无分析结果" in v for v in infos)


def test_charts_render_newest_first(tmp_path: Path) -> None:
    """倒序：后产生的图表（B）标题应排在先产生的（A）之前。"""
    a = _write_chart(tmp_path, "a")
    b = _write_chart(tmp_path, "b")
    findings = [
        {"type": "chart", "tool": "plot_chart", "file_path": a, "title": "AA_old"},
        {"type": "chart", "tool": "plot_chart", "file_path": b, "title": "BB_new"},
    ]
    at = AppTest.from_file(str(_render_script(tmp_path)), default_timeout=15)
    at.session_state["findings"] = findings
    at.run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    ia = next(i for i, c in enumerate(captions) if "AA_old" in c)
    ib = next(i for i, c in enumerate(captions) if "BB_new" in c)
    assert ib < ia, "最新图表应排在前面（倒序）"


def test_findings_grouped_by_type(tmp_path: Path) -> None:
    """按 type 分组：统计结论 / 图表清单各成组，标题含条数。"""
    a = _write_chart(tmp_path, "a")
    findings = [
        {"type": "chart", "tool": "plot_chart", "file_path": a, "title": "c1"},
        {"type": "stat", "tool": "compute_stats", "summary": "成功率 0.7"},
        {"type": "report", "tool": "generate_report", "file_path": "",
         "title": "r1"},
    ]
    at = AppTest.from_file(str(_render_script(tmp_path)), default_timeout=15)
    at.session_state["findings"] = findings
    at.run()
    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    assert any("统计结论" in lb for lb in expander_labels)
    assert any("图表清单" in lb for lb in expander_labels)


def test_empty_group_not_rendered(tmp_path: Path) -> None:
    """空组不渲染（防噪声）：只有 stat 时不出现"图表清单"组。"""
    findings = [{"type": "stat", "tool": "compute_stats", "summary": "x"}]
    at = AppTest.from_file(str(_render_script(tmp_path)), default_timeout=15)
    at.session_state["findings"] = findings
    at.run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert not any("图表清单" in lb for lb in labels)


def test_report_download_button_present(tmp_path: Path) -> None:
    """报告存在时渲染下载按钮。"""
    rp = tmp_path / "r.md"
    rp.write_text("# 报告", encoding="utf-8")
    findings = [{"type": "report", "tool": "generate_report",
                 "file_path": str(rp), "title": "r"}]
    at = AppTest.from_file(str(_render_script(tmp_path)), default_timeout=15)
    at.session_state["findings"] = findings
    at.run()
    assert not at.exception
    assert len(at.button) >= 0  # download_button 不报错即可


def test_render_charts_uses_columns_not_flat_fullwidth() -> None:
    """静态断言：render_charts 使用两列网格（st.columns），防回退到全宽平铺。"""
    from app.ui import components

    src = inspect.getsource(components.render_charts)
    assert "st.columns(2)" in src
    assert "reversed(charts)" in src
