"""主题配置类型测试（**真实缺陷回归**，2026-09-22）。

事故：``.streamlit/config.toml`` 里 ``headingFontSizes = [30, 22, ...]``
用了**整数**，而 Streamlit 的该字段是**字符串**（须带 CSS 单位）。
后果是启动时抛：

    Failed to parse the theme.headingFontSizes config option: 14
    TypeError: bad argument type for built-in operation

`headingFontSizes`（以及连带的标题层级）**完全未生效**——
而 ``docs/UI视觉优化设计.md`` 第 2.1 节正是为建立标题层级而写的，
配置形同虚设。

**注意两字段的类型要求相反**（易错点）：
- ``headingFontSizes``：**带单位的字符串**（Streamlit 源码中直接
  ``msg.heading_font_sizes.append(size)``，目标字段为字符串）；
- ``headingFontWeights``：**整数**（源码明确 "either an integer or a
  list of integers"）。
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_CFG = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"


@pytest.fixture(scope="module")
def theme() -> dict:
    return tomllib.loads(_CFG.read_text(encoding="utf-8"))["theme"]


def test_heading_font_sizes_are_unit_strings(theme: dict) -> None:
    """字号必须是**带单位的字符串**（int 会导致配置整项失效）。"""
    sizes = theme.get("headingFontSizes")
    assert sizes, "headingFontSizes 未配置"
    assert isinstance(sizes, list)
    for s in sizes:
        assert isinstance(s, str), (
            f"字号 {s!r} 是 {type(s).__name__}，必须是字符串（如 '30px'）"
            "——整数会导致 Streamlit 解析失败、整项配置失效"
        )
        assert s.strip().endswith(("px", "rem", "em", "%", "vh", "vw")), (
            f"字号 {s!r} 缺少 CSS 单位"
        )


def test_heading_font_weights_are_ints(theme: dict) -> None:
    """字重必须是**整数**（与字号相反，Streamlit 源码明确要求 int）。"""
    weights = theme.get("headingFontWeights")
    assert weights, "headingFontWeights 未配置"
    assert isinstance(weights, list)
    for w in weights:
        assert isinstance(w, int) and not isinstance(w, bool), (
            f"字重 {w!r} 是 {type(w).__name__}，必须是整数"
        )
        assert 100 <= w <= 900, f"字重 {w} 超出 CSS 合法范围 100~900"


def test_heading_levels_have_consistent_length(theme: dict) -> None:
    """字号与字重的元素数量应一致且不超过 6（h1~h6）。"""
    sizes = theme.get("headingFontSizes") or []
    weights = theme.get("headingFontWeights") or []
    assert len(sizes) == len(weights), (
        f"字号 {len(sizes)} 项与字重 {len(weights)} 项数量不一致"
    )
    assert len(sizes) <= 6, "标题层级最多 6 级（h1~h6）"


def test_chart_categorical_colors_exactly_ten(theme: dict) -> None:
    """图表配色必须**正好 10 个**（Streamlit 硬性要求，既有约定）。"""
    colors = theme.get("chartCategoricalColors")
    assert colors, "chartCategoricalColors 未配置"
    assert len(colors) == 10, f"必须正好 10 个颜色，当前 {len(colors)} 个"


def test_chart_colors_match_visual_theme_module() -> None:
    """OHLC 配色须与 `app/visual_theme.py` 的 CHART_COLORS 一致（既有断言）。"""
    from app.visual_theme import CHART_COLORS

    theme = tomllib.loads(_CFG.read_text(encoding="utf-8"))["theme"]
    assert list(theme["chartCategoricalColors"]) == list(CHART_COLORS), (
        "config.toml 与 visual_theme.py 的配色不一致（两处必须同步）"
    )
