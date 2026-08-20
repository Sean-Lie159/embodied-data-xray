"""图表绘制工具（可视化层）。

绘制图表并保存到 outputs/。支持通用图（line/scatter/histogram）、轨迹图
（trajectory，按能力标签命中列区分关节空间/末端位姿）、多流时间序列叠加图
（multi_stream_overlay，本工具核心功能，用于直观看清流间对齐/延迟）。

数据来源默认 context.df；多流叠加图按流登记表按需读取各流所需列（时间戳 +
目标数值列），测完释放，不装入 df。

中文乱码防护（设计选择，见 docs）：图表内文字（标题、轴标签、图例）v1 一律
用英文，避免 matplotlib 默认字体缺中文字形导致方框乱码。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 无 GUI 后端，供服务/测试环境绘图

import matplotlib.pyplot as plt  # noqa: E402

from agents import RunContextWrapper  # noqa: E402
from agents.decorators import tool  # noqa: E402

from app.agent.context import RunContext  # noqa: E402
from app.tools import _sniffing  # noqa: E402

# 常见时间戳列名（复用约定）。
_TIMESTAMP_COLS = ("timestamp", "time", "ts", "ts_ns", "t", "stamp", "frame_time")


def _safe_title(title: str | None, fallback: str) -> str:
    """确保图表标题为 ASCII（避免中文方框乱码）；含非 ASCII 时回退英文默认。"""
    if title and title.isascii():
        return title
    return fallback


def _find_timestamp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).lower().strip() in _TIMESTAMP_COLS:
            return str(c)
    return None


def _output_path(context: RunContext, chart_type: str) -> Path:
    """构造输出文件路径：outputs/<dataset>_<chart_type>_<timestamp>.png。"""
    base = Path(context.output_dir) if context.output_dir else Path("outputs")
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{context.dataset_id or 'dataset'}_{chart_type}_{ts}.png"
    return base / name


def _save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_chart_impl(
    context: RunContext,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """绘制图表并保存到 outputs/。

    Args:
        context: 运行时上下文。
        chart_type: 图表类型（line/scatter/histogram/trajectory/multi_stream_overlay）。
        x: 可选，X 轴列名。
        y: 可选，Y 轴列名。
        color: 可选，分组列名。
        title: 可选，标题（建议英文，含中文会回退英文默认以规避乱码）。

    Returns:
        dict，含 success、file_path、chart_type、title、description、dataset、findings。
    """
    chart_type = chart_type.lower().strip()
    dataset_id = context.dataset_id

    # 通用图需要 context.df。
    if chart_type in ("line", "scatter", "histogram"):
        if context.df is None:
            return {
                "success": False, "error": "no_data_loaded",
                "user_message": f"绘制 {chart_type} 需要已加载的数据表。请先调用 load_dataset。",
                "dataset": dataset_id,
            }
        title_safe = _safe_title(title, f"{chart_type} chart")
        path = _output_path(context, chart_type)
        fig, ax = plt.subplots()
        desc, plot_spec = _plot_generic_to_ax(context.df, chart_type, x, y, color, title_safe, fig, ax)
        _save_fig(fig, path)

    elif chart_type == "trajectory":
        if context.df is None:
            return {
                "success": False, "error": "no_data_loaded",
                "user_message": "绘制 trajectory 需要已加载的数据表。请先调用 load_dataset。",
                "dataset": dataset_id,
            }
        title_safe = _safe_title(title, "trajectory")
        path = _output_path(context, chart_type)
        result = _plot_trajectory_to_file(context.df, title_safe, path)
        if result is None:
            return {
                "success": False, "error": "not_applicable",
                "reason": "无关节列（qpos/joint）也无末端位姿列（ee/tcp/pose）",
                "user_message": "绘制 trajectory 需要关节列（qpos/joint）或末端位姿列（ee/tcp/pose）。当前数据集两类列都没有，不适用。建议改用 line/scatter/histogram。",
                "dataset": dataset_id,
                "suggested_charts": ["line", "scatter", "histogram"],
            }
        desc, traj_kind, plot_spec = result

    elif chart_type == "multi_stream_overlay":
        title_safe = _safe_title(title, "multi-stream overlay")
        path = _output_path(context, chart_type)
        result = _plot_multi_stream_to_file(context, title_safe, path)
        if result is None:
            return {
                "success": False, "error": "not_applicable",
                "reason": "无可绘制的数值流",
                "user_message": "multi_stream_overlay 需要至少一个含时间戳与数值列的流。当前数据集无可绘制流，不适用。",
                "dataset": dataset_id,
            }
        desc, _, plot_spec = result

    else:
        return {
            "success": False, "error": "unsupported_chart_type",
            "user_message": f"不支持的图表类型 {chart_type}，支持：line / scatter / histogram / trajectory / multi_stream_overlay。",
            "dataset": dataset_id,
        }

    finding = {
        "tool": "plot_chart",
        "type": "chart",
        "file_path": str(path),
        "chart_type": chart_type,
        "title": _safe_title(title, chart_type),
        "description": desc,
        "plot_spec": plot_spec,
    }
    context.findings.append(finding)

    return {
        "success": True,
        "dataset": dataset_id,
        "file_path": str(path),
        "chart_type": chart_type,
        "title": _safe_title(title, chart_type),
        "description": desc,
        "plot_spec": plot_spec,
        "findings": [finding],
        "user_message": f"已生成 {chart_type} 图表，保存至 {path}。{desc}。",
    }


def _plot_generic_to_ax(
    df, chart_type, x, y, color, title, fig, ax
) -> tuple[str, dict[str, Any]]:
    """把通用图绘制到 ax，返回 (说明, plot_spec)。"""
    if chart_type == "histogram":
        col = y or x
        if col is None:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = nums[0] if nums else None
        if col is None:
            raise ValueError("no numeric column")
        ax.hist(df[col].dropna(), bins=30)
        ax.set_xlabel(str(col)); ax.set_ylabel("count")
        ax.set_title(title)
        spec = {"x_axis": str(col), "y_axis": ["count"], "grouped_by": None, "n_series": 1}
        return f"Histogram of '{col}'", spec
    xcol = x or _find_timestamp_col(df) or df.columns[0]
    ycol = y
    if ycol is None:
        nums = [c for c in df.columns if c != xcol and pd.api.types.is_numeric_dtype(df[c])]
        ycol = nums[0] if nums else xcol
    grouped = color if (color and color in df.columns) else None
    n_series = int(df[grouped].nunique()) if grouped else 1
    if grouped:
        for g, grp in df.groupby(grouped):
            if chart_type == "scatter":
                ax.scatter(grp[xcol], grp[ycol], label=str(g))
            else:
                ax.plot(grp[xcol], grp[ycol], label=str(g))
        ax.legend()
    else:
        if chart_type == "scatter":
            ax.scatter(df[xcol], df[ycol])
        else:
            ax.plot(df[xcol], df[ycol])
    ax.set_xlabel(str(xcol)); ax.set_ylabel(str(ycol))
    ax.set_title(title)
    spec = {"x_axis": str(xcol), "y_axis": [str(ycol)], "grouped_by": grouped, "n_series": n_series}
    return f"{chart_type.capitalize()} of '{ycol}' vs '{xcol}'", spec


def _plot_trajectory_to_file(df, title, path) -> tuple[str, str, dict[str, Any]] | None:
    joint_cols = [c for c in df.columns if str(c).lower().startswith(_sniffing._JOINT_PREFIXES)]
    pose_cols = [c for c in df.columns if _sniffing._is_pose_column(str(c))]

    if joint_cols:
        fig, ax = plt.subplots()
        ts_col = _find_timestamp_col(df)
        x_axis = df[ts_col] if ts_col else np.arange(len(df))
        drawn = [c for c in joint_cols[:8] if pd.api.types.is_numeric_dtype(df[c])]
        for c in drawn:
            ax.plot(x_axis, df[c], label=str(c))
        ax.set_xlabel("time" if ts_col else "step"); ax.set_ylabel("joint position")
        ax.set_title(title); ax.legend()
        _save_fig(fig, path)
        spec = {"x_axis": ts_col or "step", "y_axis": drawn, "grouped_by": None, "n_series": len(drawn)}
        return ("Joint-space trajectory", "joint", spec)

    if pose_cols:
        xyz = [c for c in pose_cols if str(c).lower().split("_")[-1] in ("x", "y", "z")]
        if len(xyz) >= 2 and all(c in df.columns for c in xyz[:2]):
            fig = plt.figure()
            if len(xyz) >= 3 and all(c in df.columns for c in xyz[:3]):
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(df[xyz[0]], df[xyz[1]], df[xyz[2]])
                ax.set_xlabel(xyz[0]); ax.set_ylabel(xyz[1]); ax.set_zlabel(xyz[2])
                desc = "End-effector 3D trajectory"
                spec = {"x_axis": xyz[0], "y_axis": [xyz[1], xyz[2]], "grouped_by": None, "n_series": 1}
            else:
                ax = fig.add_subplot(111)
                ax.plot(df[xyz[0]], df[xyz[1]])
                ax.set_xlabel(xyz[0]); ax.set_ylabel(xyz[1])
                desc = "End-effector XY trajectory"
                spec = {"x_axis": xyz[0], "y_axis": [xyz[1]], "grouped_by": None, "n_series": 1}
            ax.set_title(title)
            _save_fig(fig, path)
            return (desc, "end_effector", spec)
        num_pose = [c for c in pose_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_pose) >= 2:
            fig, ax = plt.subplots()
            ax.plot(df[num_pose[0]], df[num_pose[1]])
            ax.set_xlabel(num_pose[0]); ax.set_ylabel(num_pose[1])
            ax.set_title(title)
            _save_fig(fig, path)
            spec = {"x_axis": num_pose[0], "y_axis": [num_pose[1]], "grouped_by": None, "n_series": 1}
            return ("End-effector trajectory", "end_effector", spec)
    return None


def _plot_multi_stream_to_file(context, title, path) -> tuple[str, str, dict[str, Any]] | None:
    streams = [s for s in context.meta.get("streams", []) if s.get("kind") != "video"]
    fig, ax = plt.subplots()
    plotted = 0
    stream_info: list[dict[str, Any]] = []
    if streams:
        for s in streams:
            if not s.get("path"):
                continue
            from app.tools.check_sensor_sanity import _read_columns

            data = _read_columns(s.get("path", ""), s.get("format", ""), s.get("channels", []))
            if not data:
                continue
            df = pd.DataFrame(data)
            ts_col = _find_timestamp_col(df)
            nums = [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
            if not nums:
                continue
            ycol = nums[0]
            t = df[ts_col].to_numpy(float) if ts_col else np.arange(len(df))
            t_rel = t - t[0] if len(t) else t
            label = Path(s.get("path", "")).name
            ax.plot(t_rel, df[ycol].to_numpy(float), label=f"{label} ({ycol})")
            rate = (s.get("measured_rate") or {}).get("sample_rate_hz") if isinstance(s.get("measured_rate"), dict) else None
            stream_info.append({"name": label, "column": ycol, "sample_rate_hz": rate})
            plotted += 1
    else:
        # 无流登记表：用主表。
        if context.df is not None:
            df = context.df
            ts_col = _find_timestamp_col(df)
            nums = [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
            if nums:
                ycol = nums[0]
                t = df[ts_col].to_numpy(float) if ts_col else np.arange(len(df))
                t_rel = t - t[0] if len(t) else t
                ax.plot(t_rel, df[ycol].to_numpy(float), label=f"main ({ycol})")
                stream_info.append({"name": "main", "column": ycol, "sample_rate_hz": None})
                plotted = 1
    if plotted == 0:
        plt.close(fig)
        return None
    ax.set_xlabel("relative time (s)"); ax.set_ylabel("value")
    ax.set_title(title); ax.legend()
    _save_fig(fig, path)
    spec = {
        "x_axis": "relative_time",
        "y_axis": [si["column"] for si in stream_info],
        "grouped_by": None,
        "n_series": plotted,
        "streams": stream_info,
    }
    return (f"Multi-stream overlay of {plotted} streams", "multi_stream_overlay", spec)


@tool
def plot_chart(
    wrapper: RunContextWrapper[RunContext],
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str | None = None,
) -> dict:
    """绘制图表并保存到 outputs/。

    支持 line / scatter / histogram（通用）、trajectory（轨迹图，按命中列区分
    关节空间/末端位姿）、multi_stream_overlay（多流时间序列叠加，x 轴为相对时间，
    用于看清流间对齐/延迟）。图表文字一律英文（规避中文乱码）。

    Args:
        chart_type: 图表类型。
        x: 可选，X 轴列名。
        y: 可选，Y 轴列名。
        color: 可选，分组列名。
        title: 可选，标题（建议英文）。

    Returns:
        dict，含 success、file_path、chart_type、title、description、dataset、
        findings；无轨迹列/无数据流时返回 not_applicable。
    """
    return plot_chart_impl(wrapper.context, chart_type, x, y, color, title)
