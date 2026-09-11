"""视觉主题配置的单测（docs/UI视觉优化设计.md 第 7 节）。

覆盖：config.toml 合法性、图表配色长度与一致性、双主题与侧栏显式配色
（防"自动交换"意外色）、跨层常量无 streamlit 依赖。
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / ".streamlit" / "config.toml"


def _config() -> dict:
    with _CONFIG.open("rb") as f:
        return tomllib.load(f)


def test_config_toml_exists_and_parses() -> None:
    """config.toml 存在且是合法 TOML。"""
    assert _CONFIG.exists(), "缺少 .streamlit/config.toml"
    data = _config()
    assert "theme" in data


def test_chart_colors_exactly_ten() -> None:
    """chartCategoricalColors 必须正好 10 个（Streamlit 硬性要求）。"""
    from app.visual_theme import CHART_COLORS

    assert len(CHART_COLORS) == 10
    theme_colors = _config()["theme"]["chartCategoricalColors"]
    assert len(theme_colors) == 10


def test_chart_colors_match_between_ui_and_config() -> None:
    """config.toml 与 app/visual_theme.py 的配色**逐值一致**（防 UI/图表漂移）。"""
    from app.visual_theme import CHART_COLORS

    theme_colors = _config()["theme"]["chartCategoricalColors"]
    assert list(theme_colors) == list(CHART_COLORS)


def test_primary_color_matches_brand() -> None:
    """config.toml 的 primaryColor 与 visual_theme.BRAND_PRIMARY 一致。"""
    from app.visual_theme import BRAND_PRIMARY

    assert _config()["theme"]["primaryColor"].upper() == BRAND_PRIMARY.upper()


def test_both_themes_defined_with_background_and_text() -> None:
    """light/dark 两套主题均定义，且各自含背景色与正文色。"""
    theme = _config()["theme"]
    for key in ("light", "dark"):
        assert key in theme, f"缺少 [theme.{key}]"
        assert "backgroundColor" in theme[key]
        assert "textColor" in theme[key]


def test_sidebar_colors_explicitly_defined() -> None:
    """两个 [theme.*.sidebar] 都**显式**定义 backgroundColor。

    防"侧栏自动交换规则"：未定义时 Streamlit 会去取 theme.secondaryBackgroundColor
    （反之亦然），导致浅/深主题下侧栏出现意外颜色（docs/UI视觉优化设计.md 2.1）。
    """
    theme = _config()["theme"]
    for key in ("light", "dark"):
        sidebar = theme[key].get("sidebar")
        assert sidebar is not None, f"缺少 [theme.{key}.sidebar]"
        assert "backgroundColor" in sidebar, (
            f"[theme.{key}.sidebar] 必须显式定义 backgroundColor"
        )


def test_visual_theme_has_no_streamlit_dependency() -> None:
    """跨层视觉常量模块不得 import streamlit（分层纪律）。

    用 AST 检查真实 import 语句（而非文本匹配——模块 docstring 里
    提到"不 import streamlit"会误伤文本匹配）。
    """
    import ast

    import app.visual_theme as vt

    tree = ast.parse(Path(vt.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(m == "streamlit" or m.startswith("streamlit.")
                   for m in imported), f"visual_theme 不得依赖 streamlit：{imported}"


def test_ui_constants_reexports_cross_layer() -> None:
    """app/ui/constants.py 再导出跨层配色（UI 层单点引用，同一对象）。"""
    from app.ui import constants
    from app.visual_theme import BRAND_PRIMARY, CHART_COLORS

    assert constants.CHART_COLORS is CHART_COLORS
    assert constants.BRAND_PRIMARY == BRAND_PRIMARY


def test_chart_ink_is_mid_tone() -> None:
    """图表文字色为中明度中性灰（深/浅底均可读）——非纯黑非纯白。"""
    from app.visual_theme import CHART_INK

    r = int(CHART_INK[1:3], 16)
    g = int(CHART_INK[3:5], 16)
    b = int(CHART_INK[5:7], 16)
    # 中明度：避免过暗（深底看不清）或过亮（浅底看不清）。
    assert 80 <= (r + g + b) / 3 <= 180, f"CHART_INK 明度过极端：{CHART_INK}"
    assert not (r == g == b == 0), "不得为纯黑"
    assert not (r == g == b == 255), "不得为纯白"
