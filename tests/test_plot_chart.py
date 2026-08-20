"""app/tools/plot_chart 工具的单元测试。

覆盖：合成数据生成图表且文件落盘、无轨迹列返回"不适用"、多流叠加跨流读取、
findings 累积（type=chart）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.plot_chart import plot_chart, plot_chart_impl


def _ctx(df: pd.DataFrame | None, output_dir: str, meta=None) -> RunContext:
    return RunContext(dataset_id="demo", df=df, meta=meta or {"capabilities": {}},
                      output_dir=output_dir)


def test_plot_chart_is_registered() -> None:
    assert isinstance(plot_chart, FunctionTool)
    assert plot_chart.name == "plot_chart"


def test_line_chart_saved_to_outputs(tmp_path: Path) -> None:
    t = np.arange(0, 1.0, 0.01)
    df = pd.DataFrame({"timestamp": t, "value": np.sin(t)})
    ctx = _ctx(df, output_dir=str(tmp_path))

    r = plot_chart_impl(ctx, "line", x="timestamp", y="value")

    assert r["success"] is True
    assert r["chart_type"] == "line"
    assert r["dataset"] == "demo"
    # 文件确实落盘。
    assert Path(r["file_path"]).exists()
    assert r["file_path"].endswith(".png")


def test_histogram_saved(tmp_path: Path) -> None:
    df = pd.DataFrame({"value": np.random.randn(100)})
    ctx = _ctx(df, output_dir=str(tmp_path))

    r = plot_chart_impl(ctx, "histogram", y="value")

    assert r["success"] is True
    assert Path(r["file_path"]).exists()


def test_trajectory_with_joint_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "qpos1": [0.1, 0.2, 0.3], "qpos2": [0.0, 0.1, 0.2], "timestamp": [0, 0.1, 0.2],
    })
    ctx = _ctx(df, output_dir=str(tmp_path))

    r = plot_chart_impl(ctx, "trajectory")

    assert r["success"] is True
    assert "joint" in r["description"].lower() or "Joint" in r["description"]
    assert Path(r["file_path"]).exists()


def test_trajectory_no_columns_not_applicable(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    ctx = _ctx(df, output_dir=str(tmp_path))

    r = plot_chart_impl(ctx, "trajectory")

    assert r["success"] is False
    assert r["error"] == "not_applicable"
    assert "suggested_charts" in r


def test_multi_stream_overlay_cross_stream_read(tmp_path: Path) -> None:
    """多流叠加图跨流读取：主表 + 独立 IMU/力文件。"""
    t = np.arange(0, 1.0, 0.01)
    imu_path = tmp_path / "imu.csv"
    pd.DataFrame({"timestamp": t, "accel_x": np.sin(t)}).to_csv(imu_path, index=False)
    ft_path = tmp_path / "ft.csv"
    pd.DataFrame({"timestamp": t + 0.005, "fx": np.cos(t)}).to_csv(ft_path, index=False)

    meta = {
        "capabilities": {"has_imu": True, "has_force": True},
        "streams": [
            {"path": str(imu_path), "format": "csv", "kind": "imu", "channels": ["accel_x"]},
            {"path": str(ft_path), "format": "csv", "kind": "force", "channels": ["fx"]},
        ],
    }
    ctx = _ctx(pd.DataFrame({"timestamp": t, "main": np.sin(t)}), meta=meta,
               output_dir=str(tmp_path))

    r = plot_chart_impl(ctx, "multi_stream_overlay")

    assert r["success"] is True
    assert "multi_stream_overlay" in r["chart_type"]
    assert Path(r["file_path"]).exists()


def test_findings_accumulated_type_chart(tmp_path: Path) -> None:
    df = pd.DataFrame({"timestamp": [0, 0.1, 0.2], "value": [0, 1, 2]})
    ctx = _ctx(df, output_dir=str(tmp_path))

    plot_chart_impl(ctx, "line", x="timestamp", y="value")

    assert len(ctx.findings) == 1
    assert ctx.findings[0]["type"] == "chart"
    assert "file_path" in ctx.findings[0]


def test_unsupported_chart_type(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    ctx = _ctx(df, output_dir=str(tmp_path))

    r = plot_chart_impl(ctx, "pie")

    assert r["success"] is False
    assert r["error"] == "unsupported_chart_type"
