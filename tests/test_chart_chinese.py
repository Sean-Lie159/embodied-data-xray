"""图表中文化与视觉质量改造的测试（2026-09-14）。

背景（用户反馈"可视化图质量都很差"）：实测发现三类问题——
  1. **文字全英文且无信息量**：标题是 "line chart"、轴标签直接是原始列名
     （qpos_0）；根因是 plot_chart 里 `_safe_title` 把含非 ASCII 的标题**静默
     丢弃**（实测传"关节角度随时间的响应曲线"被回退成 "line"），理由是"怕中文
     方框乱码"。但实测本机装有微软雅黑/黑体/思源黑体，中文渲染零缺字警告——
     前提已不成立。
  2. **多曲线图例无法区分**：轨迹图三条曲线全叫"关节位置"（映射后同名）。
  3. **同秒文件名覆盖**：同一秒内两次同类型绘图互相覆盖。

覆盖：字体探测、列名映射、标题生成、图例可区分、文件名唯一性、端到端渲染。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.chart_labels import (
    describe_chart_title,
    describe_column,
    describe_series_label,
    sanitize_for_display,
)

# app.tools 包的 __init__ 把名为 plot_chart 的 FunctionTool 暴露为包属性，
# `from app.tools import plot_chart` 拿到的是工具对象而非模块，须经 sys.modules。
import sys

import app.tools.plot_chart  # noqa: F401

_pc = sys.modules["app.tools.plot_chart"]


@pytest.fixture()
def ctx(tmp_path: Path) -> RunContext:
    """一个含时间戳/关节/位姿列的合成上下文。"""
    n = 60
    t = np.linspace(0, 6, n)
    df = pd.DataFrame({
        "timestamp": t,
        "qpos_0": np.sin(t),
        "qpos_1": np.cos(t),
        "qpos_2": np.sin(t * 2),
        "ee_x": np.sin(t),
        "ee_y": np.cos(t),
    })
    c = RunContext(output_dir=str(tmp_path), dataset_id="demo")
    c.df = df
    c.meta = {"main_table": {"name": "demo.csv"}, "streams": [], "capabilities": {}}
    return c


# --- 1. 中文字体探测 --------------------------------------------------------


def test_cjk_font_resolution_is_safe() -> None:
    """字体探测不抛异常；返回 None 或字体族名（跨平台都必须安全）。"""
    from app.chart_fonts import has_cjk_font, resolve_cjk_font

    family = resolve_cjk_font()
    assert family is None or isinstance(family, str)
    assert isinstance(has_cjk_font(), bool)
    # 存在性与返回值必须自洽。
    assert has_cjk_font() == (family is not None)


def test_apply_chart_fonts_no_exception() -> None:
    """注册字体不抛异常（无中文字体时返回 None，不得改坏 rcParams）。"""
    import matplotlib

    from app.chart_fonts import apply_chart_fonts

    before = list(matplotlib.rcParams.get("font.sans-serif", []))
    result = apply_chart_fonts()
    if result is None:
        # 找不到字体时不应改动配置（避免引入不存在的族名）。
        assert list(matplotlib.rcParams.get("font.sans-serif", [])) == before
    else:
        assert result in matplotlib.rcParams["font.sans-serif"]
        assert matplotlib.rcParams["axes.unicode_minus"] is False


# --- 2. 列名映射（确定性，不猜测）------------------------------------------


@pytest.mark.parametrize("raw,expect", [
    ("timestamp", "时间戳"),
    ("qpos_0", "关节位置"),
    ("qvel_3", "关节速度"),
    ("ee_x", "末端执行器 · X"),
    ("accel_x", "加速度 · X"),
    ("gyro_z", "陀螺仪 · Z"),
    ("force", "力"),
    ("torque", "力矩"),
    ("episode", "回合"),
    ("success", "成功率"),
])
def test_known_columns_map_to_chinese(raw: str, expect: str) -> None:
    """常见具身智能列名映射为中文语义名。"""
    assert describe_column(raw) == expect


@pytest.mark.parametrize("raw,expect", [
    ("ax", "加速度 · X"),
    ("ay", "加速度 · Y"),
    ("az", "加速度 · Z"),
    ("fx", "力 · X"),
    ("wz", "角速度 · Z"),
    ("tz", "力矩 · Z"),
])
def test_bare_axis_names_get_quantity(raw: str, expect: str) -> None:
    """裸轴名（ax/fx/wz）按首字母还原物理量。

    为什么重要：IMU/力矩数据的真实列名常就是 ax/fx 这种缩写，不含词表关键词；
    只翻译成"X"会丢掉物理量信息（用户不知道是谁的 X）。
    """
    assert describe_column(raw) == expect


@pytest.mark.parametrize("raw", [
    "tf", "state", "xyz", "foo_bar", "col_1", "unknowable_thing",
])
def test_unknown_columns_are_preserved(raw: str) -> None:
    """**未命中词表的列名原样保留**——不猜测语义（真实风险：tf 等缩写在不同
    数据集含义不同，硬猜会误导用户）。"""
    assert describe_column(raw) == raw


def test_empty_and_none_are_safe() -> None:
    """空/None 不抛异常。"""
    assert describe_column("") == ""
    assert describe_column(None) == "None"  # type: ignore[arg-type]


# --- 3. 标题生成（按内容，不空泛）------------------------------------------


def test_line_title_includes_both_axes() -> None:
    """折线图标题含 Y 与 X 的中文名（可读性来自此）。"""
    t = describe_chart_title("line", "关节位置", "时间戳")
    assert "关节位置" in t and "时间戳" in t


def test_histogram_title_is_distribution() -> None:
    """直方图标题表述为"分布"。"""
    assert "分布" in describe_chart_title("histogram", "关节位置", None)


def test_multi_series_count_in_title() -> None:
    """多条曲线时标题体现条数。"""
    t = describe_chart_title("line", "关节位置", "时间戳", 3)
    assert "3" in t


def test_title_never_empty() -> None:
    """任何类型都产出非空标题（不得回退成空字符串）。"""
    for ctype in ("line", "scatter", "histogram", "trajectory", "multi_stream_overlay"):
        assert describe_chart_title(ctype, None, None).strip()


# --- 4. 图例可区分性（真实缺陷回归）---------------------------------------


def test_series_label_maps_column_to_chinese() -> None:
    """多流图例把列名映射为中文，并含采样率。"""
    label = describe_series_label("imu.csv", "ax", sample_rate_hz=100.0)
    assert "imu" in label
    assert "加速度" in label
    assert "100 Hz" in label
    assert ".csv" not in label  # 扩展名是噪音


def test_series_label_without_rate() -> None:
    """无采样率时不留空括号。"""
    label = describe_series_label("force.csv", "fx")
    assert "(" not in label and "（" not in label
    assert "力" in label


def test_joint_curve_labels_are_distinguishable() -> None:
    """**回归**：同前缀多曲线不能全部同名。

    真实缺陷：轨迹图三条曲线都映射为"关节位置"，图例无法区分——必须保留
    原始列名。
    """
    labels = [f"{describe_column(c)}（{c}）" for c in ("qpos_0", "qpos_1", "qpos_2")]
    assert len(set(labels)) == 3, f"图例重名：{labels}"


def test_sanitize_flattens_and_truncates() -> None:
    """展示文本压成单行并截断（防图例被长路径撑爆）。"""
    assert "\n" not in sanitize_for_display("a\nb   c")
    assert len(sanitize_for_display("x" * 200)) <= 60


# --- 5. 文件名唯一性（同秒覆盖回归）---------------------------------------


def test_output_paths_unique_within_same_second(ctx: RunContext) -> None:
    """**回归**：同一秒内连续两次同类型绘图不得得到同一路径。

    真实缺陷：文件名只到秒级，脚本里连续两次 line 绘图后者覆盖前者，
    且两处 findings 指向同一个文件。
    """
    paths = {_pc._output_path(ctx, "line") for _ in range(5)}
    assert len(paths) == 5, f"同秒内路径冲突：{paths}"


def test_output_path_keeps_session_prefix(tmp_path: Path) -> None:
    """多会话隔离契约不变：文件名仍含 session_tag 前缀与 dataset_id。"""
    c = RunContext(output_dir=str(tmp_path), dataset_id="ds", session_tag="s-abcd")
    assert _pc._output_path(c, "line").name.startswith("s-abcd_ds_line_")


# --- 6. 端到端渲染（中文字面落到图上）-------------------------------------


@pytest.mark.skipif(
    not __import__("app.chart_fonts", fromlist=["x"]).has_cjk_font(),
    reason="本机无中文字体（极端环境），中文渲染断言不适用",
)
def test_rendered_title_is_chinese(ctx: RunContext) -> None:
    """端到端：默认标题为中文，且不再是 "line chart" 这类无信息量文案。"""
    r = _pc.plot_chart_impl(ctx, "line", x="timestamp", y="qpos_0")
    assert r["success"] is True
    assert "关节位置" in r["title"]
    assert "时间戳" in r["title"]
    assert r["title"] != "line chart"


@pytest.mark.skipif(
    not __import__("app.chart_fonts", fromlist=["x"]).has_cjk_font(),
    reason="本机无中文字体（极端环境），中文渲染断言不适用",
)
def test_custom_chinese_title_survives(ctx: RunContext) -> None:
    """**回归**：模型/用户传的中文标题必须生效（此前被 `_safe_title` 静默丢弃）。"""
    given = "关节角度随时间的响应曲线"
    r = _pc.plot_chart_impl(ctx, "line", x="timestamp", y="qpos_0", title=given)
    assert r["title"] == given


def test_findings_title_matches_rendered(ctx: RunContext) -> None:
    """findings 里的标题与图上标题一致（避免两处文案不一致）。"""
    r = _pc.plot_chart_impl(ctx, "histogram", y="qpos_0")
    finding = r["findings"][0]
    assert finding["title"] == r["title"]
    assert finding["plot_spec"]["title"] == r["title"]


def test_chart_files_are_actually_written(ctx: RunContext) -> None:
    """图表文件真实落盘（非零字节 png）。"""
    r = _pc.plot_chart_impl(ctx, "line", x="timestamp", y="qpos_0")
    p = Path(r["file_path"])
    assert p.exists() and p.stat().st_size > 1000


def test_multi_stream_has_two_panels(tmp_path: Path) -> None:
    """多流图改双面板（原始量纲 + 归一化），解决量级差异下小流被压平的问题。"""
    out = tmp_path
    t = np.linspace(0, 5, 50)
    pd.DataFrame({"ts": t, "ax": np.sin(t) * 0.2}).to_csv(out / "imu.csv", index=False)
    pd.DataFrame({"ts": t, "fx": np.cos(t) * 10}).to_csv(out / "force.csv", index=False)
    c = RunContext(output_dir=str(out), dataset_id="demo")
    c.meta = {
        "main_table": {"name": "x"},
        "capabilities": {},
        "streams": [
            {"path": str(out / "imu.csv"), "format": "csv", "kind": "imu",
             "channels": ["ts", "ax"]},
            {"path": str(out / "force.csv"), "format": "csv", "kind": "force",
             "channels": ["ts", "fx"]},
        ],
    }
    r = _pc.plot_chart_impl(c, "multi_stream_overlay")
    assert r["success"] is True
    assert r["plot_spec"]["panels"] == ["raw_scale", "z_score_normalized"]
    assert r["plot_spec"]["n_series"] == 2


def test_no_cjk_font_degrades_without_boxes() -> None:
    """无中文字体时降级为纯 ASCII（防方框），而不是照原样画中文。"""
    original = _pc.has_cjk_font
    try:
        _pc.has_cjk_font = lambda: False  # type: ignore[assignment]
        assert _pc._cjk("关节位置 qpos") == "qpos"
        assert _pc._cjk("全部中文") == ""
    finally:
        _pc.has_cjk_font = original  # type: ignore[assignment]
