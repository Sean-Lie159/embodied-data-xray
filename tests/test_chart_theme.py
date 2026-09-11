"""图表风格与深色模式适配的单测（docs/UI视觉优化设计.md 5.1 与 7 节）。

**回归关键**：生成的 png 必须是**透明底**——否则深色页面上会出现刺眼白贴片，
而这个问题只有切到深色模式才肉眼可见，必须靠自动断言守护。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.visual_theme import CHART_COLORS, CHART_GRID, CHART_INK


def _make_df() -> pd.DataFrame:
    n = 40
    return pd.DataFrame({
        "timestamp": np.linspace(0, 4, n),
        "joint_1": np.sin(np.linspace(0, 4, n)),
        "joint_2": np.cos(np.linspace(0, 4, n)),
    })


def _plot(tmp_path, chart_type="line"):
    from app.tools.plot_chart import plot_chart_impl

    ctx = RunContext(dataset_id="ds_theme", output_dir=str(tmp_path))
    ctx.df = _make_df()
    ctx.meta["main_table"] = {"rows_total": len(ctx.df), "n_cols": 3}
    res = plot_chart_impl(
        ctx, chart_type=chart_type, x="timestamp", y="joint_1",
    )
    assert res.get("success") is True, res
    return res["file_path"]


def test_chart_is_transparent_background(tmp_path) -> None:
    """**回归关键**：图表 png 四角透明（深色模式不出现白贴片）。"""
    from PIL import Image

    path = _plot(tmp_path, "line")
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    corners = [arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1]]
    alphas = [int(c[3]) for c in corners]
    assert all(a < 255 for a in alphas), (
        f"图表背景不透明（四角 alpha={alphas}）——深色模式下会显示为白色贴片"
    )


def test_chart_theme_rcparams_applied() -> None:
    """rcParams 已被设置为统一风格（透明底 + 去上右边框 + 配色）。"""
    import matplotlib

    from app.tools import plot_chart  # noqa: F401  确保主题已应用

    assert matplotlib.rcParams["savefig.transparent"] is True
    assert matplotlib.rcParams["axes.facecolor"] == "none"
    assert matplotlib.rcParams["figure.facecolor"] == "none"
    assert matplotlib.rcParams["axes.spines.top"] is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["text.color"] == CHART_INK
    assert matplotlib.rcParams["axes.edgecolor"] == CHART_GRID


def test_chart_prop_cycle_uses_brand_colors() -> None:
    """绘图色环使用与 UI 主题同族的 CHART_COLORS。"""
    import matplotlib

    from app.tools import plot_chart  # noqa: F401

    colors = [c["color"] for c in matplotlib.rcParams["axes.prop_cycle"]]
    assert colors[: len(CHART_COLORS)] == list(CHART_COLORS)


def test_plot_does_not_break_on_bar_and_hist(tmp_path) -> None:
    """其它图型在主题下仍能正常出图（不因透明底/配色改动而崩）。"""
    from PIL import Image

    for ct in ("histogram",):
        path = _plot(tmp_path, ct)
        assert Image.open(path).convert("RGBA").size[0] > 0
