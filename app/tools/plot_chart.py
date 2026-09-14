"""图表绘制工具（可视化层）。

绘制图表并保存到 outputs/。支持通用图（line/scatter/histogram）、轨迹图
（trajectory，按能力标签命中列区分关节空间/末端位姿）、多流时间序列叠加图
（multi_stream_overlay，本工具核心功能，用于直观看清流间对齐/延迟）。

数据来源默认 context.df；多流叠加图按流登记表按需读取各流所需列（时间戳 +
目标数值列），测完释放，不装入 df。

**中文化改造（2026-09-14）**：此前图表内文字（标题/轴标签/图例）一律英文，
以规避"matplotlib 默认字体缺中文字形"的方框问题。实测该前提已不成立（系统
装有微软雅黑/黑体/思源黑体等），而锁英文的代价是用户看到 "line chart" 这种
无信息量标题、轴标签直接是原始列名。现改为：
- 经 ``app.chart_fonts`` 注册系统已有的中文字体（**不下载、不新增依赖**）；
- 经 ``app.chart_labels`` 把列名映射为确定性中文语义名（**未命中的原样保留，
  不猜测语义**）；
- 无可用中文字体时（极端环境）自动降级为不含中文的文案，不出现方框。
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
from app.chart_fonts import apply_chart_fonts, has_cjk_font  # noqa: E402
from app.chart_labels import (  # noqa: E402
    describe_chart_title,
    describe_column,
    describe_series_label,
    sanitize_for_display,
)
from app.tools import _data_access, _sniffing  # noqa: E402
from app.visual_theme import CHART_COLORS, CHART_GRID, CHART_INK  # noqa: E402

# 常见时间戳列名（复用约定）。
_TIMESTAMP_COLS = ("timestamp", "time", "ts", "ts_ns", "t", "stamp", "frame_time")

# 图表尺寸（英寸）与 DPI：默认 6.4×4.8 在宽栏里偏小、文字被压缩；
# 加宽并提高 DPI 让中文标签清晰可读（中文字形比西文密，需要更多像素）。
_CHART_FIGSIZE: tuple[float, float] = (9.0, 5.0)
_CHART_DPI: int = 120

# 曲线线宽与散点尺寸（默认 1.5pt 在多曲线图上会糊成一片）。
_LINE_WIDTH: float = 1.6
_SCATTER_SIZE: float = 10.0


def _apply_chart_theme() -> None:
    """应用统一图表风格（docs/UI视觉优化设计.md 5.1）。

    要点：
    - **透明底**（figure/axes/savefig）——图表是 png，深色页面上若为白底会呈现
      刺眼的"白贴片"；透明底则随页面底色走，浅/深两套主题都适配；
    - **中性灰文字**（CHART_INK）——静态图片无法随主题重绘，纯黑在深底看不清、
      纯白在浅底看不清，故只能取中间明度（一张图同时适配两种主题）；
    - 去上右边框 + 淡网格：减噪，主流图表风格；
    - 配色引用 app/visual_theme.CHART_COLORS（与 Streamlit 主题的
      chartCategoricalColors 同一份值，有单测断言）。

    在模块导入时调用一次（rcParams 是全局状态；本项目全部图表都在本模块绘制，
    影响面一致且更统一）。
    """
    matplotlib.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.transparent": True,
        "savefig.facecolor": "none",
        "axes.edgecolor": CHART_GRID,
        "axes.grid": True,
        "grid.color": CHART_GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        # "semibold" 在部分环境找不到会刷 findfont 警告，用 bold 更稳。
        "axes.titleweight": "bold",
        # 中文标题更长，加大与绘图区的间距，避免贴边。
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 9,
        "text.color": CHART_INK,
        "axes.labelcolor": CHART_INK,
        "xtick.color": CHART_INK,
        "ytick.color": CHART_INK,
        "axes.prop_cycle": matplotlib.cycler(color=list(CHART_COLORS)),
    })
    # 注册中文字体（找不到时不动 rcParams——由文案层降级，不引入无效族名）。
    apply_chart_fonts()


_apply_chart_theme()


def _cjk(text: str) -> str:
    """无中文字体时剔除非 ASCII 内容（防方框乱码）。

    极端环境（如精简 Linux 容器未装任何中文字体）下，中文会渲染成方框——比
    "显示原始列名"更糟。此时降级为纯 ASCII 部分（列名本身通常就是 ASCII），
    保证图至少可读。

    Args:
        text: 目标文案。

    Returns:
        有中文字体时原样返回；否则返回去掉非 ASCII 字符的结果（可能为空串）。
    """
    if has_cjk_font():
        return text
    ascii_only = "".join(ch for ch in str(text) if ch.isascii())
    return " ".join(ascii_only.split())


def _resolve_title(title: str | None, fallback_zh: str) -> str:
    """确定图表标题：优先调用方给的中文标题，否则用按内容生成的中文标题。

    与改造前的关键差异：此前**含非 ASCII 的标题会被静默丢弃**（回退英文默认），
    模型传了"关节角度随时间的响应曲线"也看不到。现在中文标题正常生效。

    Args:
        title: 调用方（模型）传入的标题，可为 None。
        fallback_zh: 按图表内容生成的中文标题。

    Returns:
        最终标题（无中文字体时已降级为 ASCII）。
    """
    chosen = (title or "").strip() or fallback_zh
    return _cjk(sanitize_for_display(chosen))


def _find_timestamp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).lower().strip() in _TIMESTAMP_COLS:
            return str(c)
    return None


def _output_path(context: RunContext, chart_type: str) -> Path:
    """构造输出文件路径：outputs/by_dataset/<数据集>/charts/<...>.png。

    子目录归类见 docs/UI优化总纲与输出目录改造设计.md 第 3 节；文件名含
    session_tag 前缀与 dataset_id，多会话输出隔离契约不变。

    **同秒唯一性（2026-09-14 修复）**：此前名称只到秒级（``..._<ts>.png``），
    同一秒内为同一数据集画两张同类型图会**互相覆盖**（真实事故：脚本里连续
    两次 ``line`` 绘图，后者把前者覆盖，且两处 findings 指向同一个文件）。
    现用 ``time.time_ns()`` 的微秒段补足——不做"存在则加序号"的存在性探测，
    因为那在多会话并发下仍有竞态；时间戳本身唯一最简。
    """
    from app.tools._data_access import output_prefix
    from app.tools.output_paths import chart_dir

    ts = time.strftime("%Y%m%d_%H%M%S")
    # 微秒段（6 位）：同秒内多次绘图不冲突；仍保持时间戳可读、可排序。
    micro = f"{time.time_ns() // 1000 % 1_000_000:06d}"
    name = (f"{output_prefix(context)}{context.dataset_id or 'dataset'}"
            f"_{chart_type}_{ts}_{micro}.png")
    return chart_dir(context.output_dir or "outputs", context.dataset_id) / name


def _save_fig(fig, path: Path) -> None:
    """保存图表为**透明底** png（深色模式下不出现白贴片，见 _apply_chart_theme）。"""
    fig.tight_layout()
    fig.savefig(path, dpi=_CHART_DPI, transparent=True)
    plt.close(fig)


def _new_axes(figsize: tuple[float, float] = _CHART_FIGSIZE):
    """创建统一尺寸的图与坐标轴（2D）。

    统一入口的原因：图表尺寸是"整体观感"的一部分，散落在各处容易遗漏；
    集中后调一处即全局生效。

    Returns:
        (fig, ax)。
    """
    return plt.subplots(figsize=figsize)


def plot_chart_impl(
    context: RunContext,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    title: str | None = None,
    table: str | None = None,
) -> dict[str, Any]:
    """绘制图表并保存到 outputs/。

    Args:
        context: 运行时上下文。
        chart_type: 图表类型（line/scatter/histogram/trajectory/multi_stream_overlay）。
        x: 可选，X 轴列名。
        y: 可选，Y 轴列名。
        color: 可选，分组列名。
        title: 可选，标题（建议英文，含中文会回退英文默认以规避乱码）。
        table: 可选，目标表名（文件名）；缺省用主表/自动定位。

    Returns:
        dict，含 success、file_path、chart_type、title、description、dataset、
        table_name、findings。
    """
    chart_type = chart_type.lower().strip()
    dataset_id = context.dataset_id

    # 通用图：缺省主表，或经统一入口按名惰性读取指定表（不替换主表）。
    if chart_type in ("line", "scatter", "histogram"):
        resolved = _data_access.resolve_table_name(context, table)
        if not resolved["success"]:
            return {
                "success": False, "error": resolved.get("error", "no_data_loaded"),
                "reason": resolved.get("reason"),
                "table": table,
                "user_message": resolved.get("user_message", f"绘制 {chart_type} 需要可用数据表。"),
                "dataset": dataset_id,
            }
        # 标题在 _plot_generic_to_ax 内按实际轴列名生成（此处只透传用户标题）。
        title_safe = title
        path = _output_path(context, chart_type)
        fig, ax = _new_axes()
        # 把数据集声明的语义元数据挂到 df.attrs，供向量列绘图取维度名（不推测）。
        plot_df = resolved["df"]
        try:
            plot_df.attrs["lerobot_info"] = context.meta.get("lerobot_info") or {}
        except Exception:  # noqa: BLE001
            pass
        desc, plot_spec = _plot_generic_to_ax(plot_df, chart_type, x, y, color, title_safe, fig, ax)
        _save_fig(fig, path)
        chart_table_name = resolved["table_name"]

    elif chart_type == "trajectory":
        # 显式指定表 → 统一入口按名读取；否则自动定位状态/动作表。
        if table is not None:
            resolved = _data_access.resolve_table_name(context, table)
            if not resolved["success"]:
                return {
                    "success": False, "error": resolved.get("error", "table_not_found"),
                    "reason": resolved.get("reason"),
                    "table": table,
                    "user_message": resolved.get("user_message", "指定的表不可用。"),
                    "dataset": dataset_id,
                }
            traj_df = resolved["df"]
            chart_table_name = resolved["table_name"]
        else:
            traj_df, _source = _data_access.locate_action_table(context)
            chart_table_name = context.meta.get("main_table", {}).get("name")
        if traj_df is None:
            return {
                "success": False, "error": "no_data_loaded",
                "user_message": "绘制 trajectory 需要已加载的数据表。请先调用 load_dataset。",
                "dataset": dataset_id,
            }
        # 标题在 _plot_trajectory_to_file 内按实际列生成（此处只透传用户标题）。
        title_safe = title
        path = _output_path(context, chart_type)
        # 注入数据集声明的语义元数据，供骨骼块轨迹取维度名（不推测）。
        try:
            traj_df.attrs["lerobot_info"] = context.meta.get("lerobot_info") or {}
        except Exception:  # noqa: BLE001
            pass
        result = _plot_trajectory_to_file(traj_df, title_safe, path)
        if result is None:
            return {
                "success": False, "error": "not_applicable",
                "reason": "无关节列（qpos/joint）、无末端位姿列（ee/tcp/pose）、也无数据集声明的骨骼块列",
                "user_message": (
                    "绘制 trajectory 需要关节列（qpos/joint）、末端位姿列（ee/tcp/pose），"
                    "或数据集声明块分解的骨骼位姿列（如 meta/info.json 中 names=xxx_NxM）。"
                    "当前数据集三类列都没有，不适用。建议改用 line/scatter/histogram。"
                ),
                "dataset": dataset_id,
                "suggested_charts": ["line", "scatter", "histogram"],
            }
        desc, traj_kind, plot_spec = result

    elif chart_type == "multi_stream_overlay":
        # 标题在多流绘图函数内按流数生成（此处只透传用户标题）。
        title_safe = title
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

    # 实际渲染的标题从 plot_spec 取（各绘图函数已按内容生成中文标题并写入）；
    # 找不到时回退用户标题/图表类型，保证 findings 与图上的字一致。
    rendered_title = str(
        plot_spec.get("title")
        or (title or "").strip()
        or chart_type
    )
    finding = {
        "tool": "plot_chart",
        "type": "chart",
        "file_path": str(path),
        "chart_type": chart_type,
        "title": rendered_title,
        "description": desc,
        "plot_spec": plot_spec,
    }
    context.findings.append(finding)

    table_name = chart_table_name if chart_type in ("line", "scatter", "histogram", "trajectory") else None
    return {
        "success": True,
        "dataset": dataset_id,
        "table": table_name,
        "table_name": table_name,
        "file_path": str(path),
        "chart_type": chart_type,
        "title": rendered_title,
        "description": desc,
        "plot_spec": plot_spec,
        "findings": [finding],
        "user_message": f"已生成 {chart_type} 图表（数据集 {dataset_id}" + (f"，表 {table_name}" if table_name else "") + f"），保存至 {path}。{desc}。",
    }


def _plot_generic_to_ax(
    df, chart_type, x, y, color, title, fig, ax
) -> tuple[str, dict[str, Any]]:
    """把通用图绘制到 ax，返回 (说明, plot_spec)。

    说明与 plot_spec 保持**英文口径**（面向模型的技术描述，含原始列名，便于
    溯源）；只有**图上可见的文字**（标题/轴标签/图例）走中文化——这样模型
    引用的列名不会因翻译而对不上数据。
    """
    if chart_type == "histogram":
        col = y or x
        if col is None:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            col = nums[0] if nums else None
        if col is None:
            raise ValueError("no numeric column")
        ax.hist(df[col].dropna(), bins=30, color=CHART_COLORS[0], edgecolor="none")
        ax.set_xlabel(describe_column(str(col)))
        ax.set_ylabel("频数")
        head = _resolve_title(title, describe_chart_title(
            "histogram", describe_column(str(col)), None))
        ax.set_title(head)
        spec = {"x_axis": str(col), "y_axis": ["count"], "grouped_by": None,
                "n_series": 1, "title": head}
        return f"Histogram of '{col}'", spec
    xcol = x or _find_timestamp_col(df) or df.columns[0]
    ycol = y
    if ycol is None:
        nums = [c for c in df.columns if c != xcol and pd.api.types.is_numeric_dtype(df[c])]
        ycol = nums[0] if nums else xcol

    # 向量列（如 LeRobot 的 observation.head_pose，每行为多维数组）：
    # 展开为多条曲线，图例优先用数据集声明的维度名（meta/info.json features.names），
    # 无声明时用 col[i] 占位——不得推测维度含义。
    vec = _vector_matrix(df[ycol]) if ycol in df.columns else None
    if vec is not None:
        dim_names = _dimension_names_for(df, ycol, vec.shape[1])
        x_vals = df[xcol]
        max_dims = 8  # 维度过多时只画前若干，避免图不可读
        drawn = list(range(min(vec.shape[1], max_dims)))
        # 向量列的多条曲线用"维度名"作标签即可区分（维度名来自数据集声明，
        # 本身含语义如 px/py/pz），不额外加列名——否则图例会过长。
        for i in drawn:
            label = dim_names[i] if i < len(dim_names) else f"{ycol}[{i}]"
            label = f"{describe_column(str(label))}（{label}）" if (
                describe_column(str(label)) != str(label)) else str(label)
            if chart_type == "scatter":
                ax.scatter(x_vals, vec[:, i], label=label, s=_SCATTER_SIZE)
            else:
                ax.plot(x_vals, vec[:, i], label=label, linewidth=_LINE_WIDTH)
        ax.legend(fontsize="small", frameon=False)
        ax.set_xlabel(describe_column(str(xcol)))
        ax.set_ylabel(describe_column(str(ycol)))
        head = _resolve_title(title, describe_chart_title(
            chart_type, describe_column(str(ycol)),
            describe_column(str(xcol)), len(drawn)))
        ax.set_title(head)
        spec = {
            "x_axis": str(xcol),
            "y_axis": [dim_names[i] if i < len(dim_names) else f"{ycol}[{i}]" for i in drawn],
            "grouped_by": None,
            "n_series": len(drawn),
            "title": head,
            "vector_column": str(ycol),
            "dims_total": int(vec.shape[1]),
            "dims_drawn": len(drawn),
            "dimension_source": _dimension_source_for(df, ycol),
        }
        note = f"（向量列，共 {vec.shape[1]} 维，已画前 {len(drawn)} 维）"
        return (
            f"{chart_type.capitalize()} of '{ycol}' (expanded by dimension) vs "
            f"'{xcol}'{note}",
            spec,
        )

    grouped = color if (color and color in df.columns) else None
    n_series = int(df[grouped].nunique()) if grouped else 1
    if grouped:
        for g, grp in df.groupby(grouped):
            if chart_type == "scatter":
                ax.scatter(grp[xcol], grp[ycol], label=str(g), s=_SCATTER_SIZE)
            else:
                ax.plot(grp[xcol], grp[ycol], label=str(g), linewidth=_LINE_WIDTH)
        ax.legend(frameon=False)
    else:
        if chart_type == "scatter":
            ax.scatter(df[xcol], df[ycol], s=_SCATTER_SIZE, color=CHART_COLORS[0])
        else:
            ax.plot(df[xcol], df[ycol], linewidth=_LINE_WIDTH, color=CHART_COLORS[0])
    ax.set_xlabel(describe_column(str(xcol)))
    ax.set_ylabel(describe_column(str(ycol)))
    head = _resolve_title(title, describe_chart_title(
        chart_type, describe_column(str(ycol)), describe_column(str(xcol)), n_series))
    ax.set_title(head)
    spec = {"x_axis": str(xcol), "y_axis": [str(ycol)], "grouped_by": grouped,
            "n_series": n_series, "title": head}
    return f"{chart_type.capitalize()} of '{ycol}' vs '{xcol}'", spec


def _vector_matrix(series) -> np.ndarray | None:
    """把向量列解析为 (n_rows, n_dims) 数值矩阵；非向量列返回 None。

    Args:
        series: 列 Series（object，每行可能是向量串/list/ndarray）。

    Returns:
        数值矩阵；多数行不可解析时返回 None。
    """
    from app.tools._data_access import parse_lerobot_vector

    # 标量数值列（int/float）不是向量列：保持既有单序列绘图行为，避免回归。
    if pd.api.types.is_numeric_dtype(series.dtype):
        return None

    vals = series.tolist()
    if not vals:
        return None
    parsed: list[np.ndarray] = []
    for v in vals:
        vec = parse_lerobot_vector(v)
        parsed.append(vec if vec is not None else None)
    ok = [v for v in parsed if v is not None]
    # 多数行可解析且维度一致才认定为向量列。
    if len(ok) < len(vals) * 0.5:
        return None
    dim = ok[0].size
    if any(v.size != dim for v in ok):
        return None
    mat = np.full((len(vals), dim), np.nan)
    for i, v in enumerate(parsed):
        if v is not None:
            mat[i] = v
    return mat


def _dimension_names_for(df, column: str, n_dims: int) -> list[str]:
    """取向量列各维度名：优先数据集声明（meta/info.json），其次 col[i] 占位。"""
    names: list[str] = []
    try:
        info = df.attrs.get("lerobot_info") or {}
        declared = _sniffing.column_dimension_names(info, str(column))
    except Exception:  # noqa: BLE001
        declared = None
    if declared and len(declared) == n_dims:
        names.extend(str(n) for n in declared)
    elif declared and len(declared) == 1:
        # 单一组合名（如 'body_24x7'/'action_fullbody_hands'）：无逐维声明，用占位。
        names.extend(f"{column}[{i}]" for i in range(n_dims))
    else:
        names.extend(f"{column}[{i}]" for i in range(n_dims))
    return names


def _dimension_source_for(df, column: str) -> str:
    """维度名来源说明（供 plot_spec 透出）。"""
    try:
        info = df.attrs.get("lerobot_info") or {}
        declared = _sniffing.column_dimension_names(info, str(column))
    except Exception:  # noqa: BLE001
        declared = None
    if declared:
        return "meta/info.json features（数据集声明）"
    return "数据集未声明维度名，使用 col[i] 占位"


def _skeleton_trajectory_column(df) -> dict[str, Any] | None:
    """找可用于骨骼轨迹绘制的列：向量列 + 数据集声明的块分解（含 3 位置维）。

    仅当 meta/info.json 声明了块分解（names=xxx_NxM 且 N*M==shape[0]）时认定，
    不猜测；向量矩阵解析失败则返回 None（由调用方回退其他轨迹类型）。

    Args:
        df: 数据表（需带 attrs["lerobot_info"]）。

    Returns:
        dict，含 column / mat（n_rows, dims）/ blocks / dof / dim_order /
        order_source / pos_idx；无可用列返回 None。
    """
    info = df.attrs.get("lerobot_info") or {}
    if not info:
        return None
    for col in df.columns:
        decl = _sniffing.parse_block_declaration(info, str(col))
        if decl is None:
            continue
        mat = _vector_matrix(df[col])
        if mat is None or mat.shape[1] != decl["block_count"] * decl["dof_per_block"]:
            continue
        dof = decl["dof_per_block"]
        order = _sniffing.infer_dof_order(info, str(col), dof)
        dim_order = order["order"]
        # 位置维索引：取维度名以 p 开头的（如 px/py/pz）；前三位置维。
        pos_idx = [i for i, n in enumerate(dim_order) if str(n).startswith("p")][:3]
        if len(pos_idx) < 2:
            continue  # 无法判定位置维 → 不画（不猜测）。
        return {
            "column": str(col),
            "mat": mat,
            "blocks": decl["block_count"],
            "dof": dof,
            "declared_name": decl["declared_name"],
            "dim_order": dim_order,
            "order_source": order["source"],
            "order_inferred": order["is_inferred"],
            "pos_idx": pos_idx,
        }
    return None


def _plot_skeleton_trajectory(
    df, skel: dict[str, Any], title: str, path
) -> tuple[str, str, dict[str, Any]] | None:
    """按骨骼块画位置轨迹（3D 若可，否则 XY）。

    Args:
        df: 数据表。
        skel: _skeleton_trajectory_column 的返回。
        title: 图标题。
        path: 输出路径。

    Returns:
        (说明, 轨迹类型, plot_spec)；无法绘制返回 None。
    """
    mat = skel["mat"]
    dof = skel["dof"]
    pos_idx = skel["pos_idx"]
    # 取前若干块绘制（块数过多时图不可读）。
    max_blocks = 6
    drawn_blocks = list(range(min(skel["blocks"], max_blocks)))
    has_z = len(pos_idx) >= 3

    fig = plt.figure()
    if has_z:
        ax = fig.add_subplot(111, projection="3d")
        for b in drawn_blocks:
            seg = mat[:, b * dof : (b + 1) * dof]
            ax.plot(
                seg[:, pos_idx[0]], seg[:, pos_idx[1]], seg[:, pos_idx[2]],
                label=f"block_{b}",
            )
        ax.set_xlabel(str(skel["dim_order"][pos_idx[0]]))
        ax.set_ylabel(str(skel["dim_order"][pos_idx[1]]))
        ax.set_zlabel(str(skel["dim_order"][pos_idx[2]]))
        desc = (
            f"Skeleton 3D trajectory of '{skel['column']}' "
            f"({skel['declared_name']}: {skel['blocks']} blocks x {dof}DoF; "
            f"drew {len(drawn_blocks)} blocks)"
        )
        spec_y = [str(skel["dim_order"][pos_idx[1]]), str(skel["dim_order"][pos_idx[2]])]
    else:
        ax = fig.add_subplot(111)
        for b in drawn_blocks:
            seg = mat[:, b * dof : (b + 1) * dof]
            ax.plot(seg[:, pos_idx[0]], seg[:, pos_idx[1]], label=f"block_{b}")
        ax.set_xlabel(str(skel["dim_order"][pos_idx[0]]))
        ax.set_ylabel(str(skel["dim_order"][pos_idx[1]]))
        desc = (
            f"Skeleton XY trajectory of '{skel['column']}' "
            f"({skel['declared_name']}: {skel['blocks']} blocks x {dof}DoF; "
            f"drew {len(drawn_blocks)} blocks)"
        )
        spec_y = [str(skel["dim_order"][pos_idx[1]])]
    head = _resolve_title(
        title,
        f"骨骼轨迹 · {skel['declared_name']}（{skel['blocks']} 块 × {dof} 自由度）",
    )
    ax.set_title(head)
    ax.legend(fontsize="small", frameon=False)
    _save_fig(fig, path)
    spec = {
        "x_axis": str(skel["dim_order"][pos_idx[0]]),
        "y_axis": spec_y,
        "grouped_by": None,
        "n_series": len(drawn_blocks),
        "title": head,
        "skeleton_column": skel["column"],
        "declared_name": skel["declared_name"],
        "blocks_total": skel["blocks"],
        "blocks_drawn": len(drawn_blocks),
        "dof_per_block": dof,
        "dimension_order": list(skel["dim_order"]),
        "dimension_source": skel["order_source"],
        "dimension_order_inferred": skel["order_inferred"],
    }
    return (desc, "skeleton", spec)


def _plot_trajectory_to_file(df, title, path) -> tuple[str, str, dict[str, Any]] | None:
    joint_cols = [c for c in df.columns if str(c).lower().startswith(_sniffing._JOINT_PREFIXES)]
    pose_cols = [c for c in df.columns if _sniffing._is_pose_column(str(c))]

    # 骨骼位姿列（向量列 + 数据集声明块分解，如 body_24x7 / left_hand_26x7）：
    # 优先按块画位置轨迹。仅处理**声明**的块分解，不做形态猜测。
    skeleton = _skeleton_trajectory_column(df)
    if skeleton is not None:
        result = _plot_skeleton_trajectory(df, skeleton, title, path)
        if result is not None:
            return result

    if joint_cols:
        fig, ax = _new_axes()
        ts_col = _find_timestamp_col(df)
        x_axis = df[ts_col] if ts_col else np.arange(len(df))
        drawn = [c for c in joint_cols[:8] if pd.api.types.is_numeric_dtype(df[c])]
        for c in drawn:
            # 图例保留原始列名：同前缀的多条曲线（qpos_0/qpos_1/qpos_2）若只显示
            # 映射后的"关节位置"会**全部同名而无法区分**（真实缺陷）。故此处
            # 用"中文语义 + 原列名"的形式。
            ax.plot(x_axis, df[c],
                    label=f"{describe_column(str(c))}（{c}）",
                    linewidth=_LINE_WIDTH)
        ax.set_xlabel(describe_column(ts_col) if ts_col else "步数")
        ax.set_ylabel("关节位置")
        head = _resolve_title(title, describe_chart_title(
            "trajectory", "关节位置", describe_column(ts_col) if ts_col else None,
            len(drawn)))
        ax.set_title(head)
        ax.legend(frameon=False)
        _save_fig(fig, path)
        spec = {"x_axis": ts_col or "step", "y_axis": drawn, "grouped_by": None,
                "n_series": len(drawn), "title": head}
        return ("Joint-space trajectory", "joint", spec)

    if pose_cols:
        xyz = [c for c in pose_cols if str(c).lower().split("_")[-1] in ("x", "y", "z")]
        if len(xyz) >= 2 and all(c in df.columns for c in xyz[:2]):
            fig = plt.figure(figsize=_CHART_FIGSIZE)
            if len(xyz) >= 3 and all(c in df.columns for c in xyz[:3]):
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(df[xyz[0]], df[xyz[1]], df[xyz[2]],
                        linewidth=_LINE_WIDTH, color=CHART_COLORS[0])
                ax.set_xlabel(describe_column(xyz[0]))
                ax.set_ylabel(describe_column(xyz[1]))
                ax.set_zlabel(describe_column(xyz[2]))
                desc = "End-effector 3D trajectory"
                spec = {"x_axis": xyz[0], "y_axis": [xyz[1], xyz[2]], "grouped_by": None, "n_series": 1}
            else:
                ax = fig.add_subplot(111)
                ax.plot(df[xyz[0]], df[xyz[1]],
                        linewidth=_LINE_WIDTH, color=CHART_COLORS[0])
                ax.set_xlabel(describe_column(xyz[0]))
                ax.set_ylabel(describe_column(xyz[1]))
                desc = "End-effector XY trajectory"
                spec = {"x_axis": xyz[0], "y_axis": [xyz[1]], "grouped_by": None, "n_series": 1}
            head = _resolve_title(title, "末端执行器运动轨迹")
            ax.set_title(head)
            spec["title"] = head
            _save_fig(fig, path)
            return (desc, "end_effector", spec)
        num_pose = [c for c in pose_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_pose) >= 2:
            fig, ax = _new_axes()
            ax.plot(df[num_pose[0]], df[num_pose[1]],
                    linewidth=_LINE_WIDTH, color=CHART_COLORS[0])
            ax.set_xlabel(describe_column(num_pose[0]))
            ax.set_ylabel(describe_column(num_pose[1]))
            head = _resolve_title(title, "末端执行器运动轨迹")
            ax.set_title(head)
            _save_fig(fig, path)
            spec = {"x_axis": num_pose[0], "y_axis": [num_pose[1]],
                    "grouped_by": None, "n_series": 1, "title": head}
            return ("End-effector trajectory", "end_effector", spec)
    return None


def _plot_multi_stream_to_file(context, title, path) -> tuple[str, str, dict[str, Any]] | None:
    """多流时间序列叠加图（双面板：原始量纲 + 归一化）。

    为什么改成双面板：各流**量纲差异巨大**（实测案例：力/力矩量级 ±10，IMU 加
    速度 ±0.2），直接叠在同一坐标轴时小量级流被压成一条直线——图看不出任何
    信息（真实观感问题）。上panel 保留原始量纲（回答"数值是多少"），下panel
    按 z-score 归一化并做时间偏移补偿前的原始对齐（回答"相位/时序关系如何"），
    两者互补。

    Args:
        context: 运行时上下文（取流登记表）。
        title: 图标题。
        path: 输出路径。

    Returns:
        (说明, 轨迹类型, plot_spec)；无可绘制流返回 None。
    """
    streams = [s for s in context.meta.get("streams", []) if s.get("kind") != "video"]
    plotted_series: list[dict[str, Any]] = []

    def _collect(path_str: str, fmt: str, channels: list[str],
                 name: str, rate: float | None) -> None:
        """读取一条流并收集其首个数值列（失败静默跳过，不阻塞整图）。"""
        from app.tools.check_sensor_sanity import _read_columns

        data = _read_columns(path_str, fmt, channels)
        if not data:
            return
        sdf = pd.DataFrame(data)
        ts_col = _find_timestamp_col(sdf)
        nums = [c for c in sdf.columns
                if c != ts_col and pd.api.types.is_numeric_dtype(sdf[c])]
        if not nums:
            return
        ycol = nums[0]
        t = sdf[ts_col].to_numpy(float) if ts_col else np.arange(len(sdf), dtype=float)
        plotted_series.append({
            "name": name,
            "column": ycol,
            "sample_rate_hz": rate,
            "t": t,
            "y": sdf[ycol].to_numpy(float),
        })

    if streams:
        for s in streams:
            if not s.get("path"):
                continue
            rate = (s.get("measured_rate") or {}).get("sample_rate_hz") \
                if isinstance(s.get("measured_rate"), dict) else None
            _collect(s.get("path", ""), s.get("format", ""),
                     s.get("channels", []), Path(s.get("path", "")).name, rate)
    elif context.df is not None:
        sdf = context.df
        ts_col = _find_timestamp_col(sdf)
        nums = [c for c in sdf.columns
                if c != ts_col and pd.api.types.is_numeric_dtype(sdf[c])]
        if nums:
            t = sdf[ts_col].to_numpy(float) if ts_col else np.arange(len(sdf), dtype=float)
            plotted_series.append({
                "name": "主表", "column": nums[0], "sample_rate_hz": None,
                "t": t, "y": sdf[nums[0]].to_numpy(float),
            })

    plotted = len(plotted_series)
    if plotted == 0:
        plt.close("all")
        return None

    # 双面板（高度比 3:2，归一化面板略矮——它的作用是看相位关系）。
    fig, (ax_raw, ax_norm) = plt.subplots(
        2, 1, figsize=(_CHART_FIGSIZE[0], _CHART_FIGSIZE[1] * 1.35),
        sharex=True, gridspec_kw={"height_ratios": [3, 2]},
    )
    stream_info: list[dict[str, Any]] = []
    for i, ss in enumerate(plotted_series):
        t = ss["t"]
        t_rel = t - t[0] if len(t) else t
        label = describe_series_label(
            ss["name"], ss["column"], sample_rate_hz=ss["sample_rate_hz"])
        color = CHART_COLORS[i % len(CHART_COLORS)]
        ax_raw.plot(t_rel, ss["y"], label=label, linewidth=_LINE_WIDTH, color=color)
        # z-score 归一化：仅当该列非常量（std>0）时画，避免除零。
        y = ss["y"]
        finite = y[np.isfinite(y)]
        if finite.size > 1 and float(np.std(finite)) > 0:
            z = (y - float(np.mean(finite))) / float(np.std(finite))
            ax_norm.plot(t_rel, z, linewidth=_LINE_WIDTH, color=color)
        stream_info.append({
            "name": ss["name"], "column": ss["column"],
            "sample_rate_hz": ss["sample_rate_hz"], "label": label,
        })

    head = _resolve_title(
        title, describe_chart_title("multi_stream_overlay", None, None, plotted))
    ax_raw.set_ylabel("原始数值")
    ax_raw.set_title(head)
    ax_raw.legend(loc="upper right", frameon=False)
    ax_norm.set_ylabel("归一化（标准差）")
    ax_norm.set_xlabel("相对时间（秒）")
    if plotted > 1:
        ax_norm.set_title("归一化对比（消除量纲差异，看时序与相位关系）",
                          fontsize=10)
    _save_fig(fig, path)
    spec = {
        "x_axis": "relative_time",
        "y_axis": [si["column"] for si in stream_info],
        "grouped_by": None,
        "n_series": plotted,
        "title": head,
        "streams": stream_info,
        "panels": ["raw_scale", "z_score_normalized"],
        "note": "上panel 为原始量纲（含量级差异），下panel 为 z-score 归一化对比。",
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
    table: str | None = None,
) -> dict:
    """绘制图表并保存到 outputs/。

    支持 line / scatter / histogram（通用）、trajectory（轨迹图，按命中列区分
    关节空间/末端位姿）、multi_stream_overlay（多流时间序列叠加，双面板：原始
    量纲 + 归一化对比，用于看清量级差异与流间相位关系）。

    **图表文字为中文**：标题、轴标签、图例会自动中文化（列名按词表映射为语义名，
    未命中的列名原样保留，不猜测含义）。给 title 传中文标题会正常生效。

    Args:
        chart_type: 图表类型。
        x: 可选，X 轴列名。
        y: 可选，Y 轴列名。
        color: 可选，分组列名。
        title: 可选，标题（**建议直接给中文**，如"关节角度随时间的响应曲线"）；
            省略时按图表内容自动生成中文标题。
        table: 可选，目标表名（如 "accel.csv"）；缺省用主表/自动定位，指定表
            按名惰性读取、不替换主表。

    Returns:
        dict，含 success、file_path、chart_type、title、description、dataset、
        table_name、findings；无轨迹列/无数据流时返回 not_applicable。
    """
    return plot_chart_impl(wrapper.context, chart_type, x, y, color, title, table)
