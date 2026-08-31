"""任务级统计工具（统计层）。

对状态/动作数据表做任务级统计：通用统计（count/mean/std/min/max/中位数）、
episode 时长与长度分布、按 episode 聚合的成功率、关节活动范围、任务完成时长
离群 episode（IQR 法）。语义不明时标注所用规则与依据列名，不静默假设。

数据来源默认 context.df；若能力标签命中列显示状态/动作在独立表中，按流登记表
按需读取。前置条件（has_actions 或存在状态/动作表）不满足时返回结构化"不适用"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.config import get_settings
from app.tools import _data_access

# episode 列候选（用于识别 episode 划分）。
_EPISODE_COLS = ("episode", "ep", "eps", "episode_id", "traj_id", "trajectory_id")
# success 列候选。
_SUCCESS_COLS = ("success", "successful", "done")
# 关节列前缀（活动范围）。
_JOINT_PREFIXES = ("qpos", "joint", "q_")
# 骨骼位姿块最多展开的块数（避免返回体积爆炸；总数仍完整给出）。
_MAX_SKELETON_BLOCKS_SHOWN = 8


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """在 df 中查找候选列（大小写不敏感）。"""
    for c in df.columns:
        if str(c).lower().strip() in candidates:
            return str(c)
    return None


def _generic_stats(
    df: pd.DataFrame, column: str | None, group_by: str | None
) -> dict[str, Any]:
    """通用统计：count/mean/std/min/max/中位数。

    Args:
        df: 数据表。
        column: 要统计的列；省略时统计所有数值列。
        group_by: 可选，分组列。

    Returns:
        统计结果 dict。
    """
    if column is not None:
        if column not in df.columns:
            return {"error": f"列 {column} 不存在"}
        cols = [column]
    else:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    result: dict[str, Any] = {}
    for c in cols:
        series = df[c]
        if group_by is not None and group_by in df.columns:
            grouped = df.groupby(group_by)[c]
            result[c] = {
                "count": grouped.count().to_dict(),
                "mean": grouped.mean().round(4).to_dict(),
                "std": grouped.std().round(4).to_dict(),
                "min": grouped.min().round(4).to_dict(),
                "max": grouped.max().round(4).to_dict(),
                "median": grouped.median().round(4).to_dict(),
            }
        else:
            result[c] = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 4) if series.notna().any() else None,
                "std": round(float(series.std()), 4) if series.notna().any() else None,
                "min": round(float(series.min()), 4) if series.notna().any() else None,
                "max": round(float(series.max()), 4) if series.notna().any() else None,
                "median": round(float(series.median()), 4) if series.notna().any() else None,
            }
    return result


def _skeleton_pose_range(
    df: pd.DataFrame, lerobot_info: dict[str, Any]
) -> dict[str, Any]:
    """骨骼位姿范围（skeleton_pose_range）：按数据集声明的块分解展开统计。

    与 joint_range_of_motion 严格区分：
    - joint_range_of_motion = **关节角度**范围（机器人学 ROM，标量列，常 rad/deg）；
    - skeleton_pose_range = **骨骼位姿**范围（向量列按 N 块 × M DoF 展开，
      含位置（米）与四元数，非关节角度）。

    仅处理**数据集显式声明**且自洽的块分解（names 为 `xxx_NxM` 且 N*M == shape[0]）；
    块内维度顺序优先用本列逐维声明，否则参照同数据集同 dof 的逐维声明列（如
    head_pose）**推断并标注**；无依据时用 dim_i 占位且不解读语义。

    Args:
        df: 数据表。
        lerobot_info: parse_lerobot_info 的返回（含 column_semantics）。

    Returns:
        dict：列名 → {block_count, dof_per_block, declared_name, blocks(前若干块
        的 min/max/range), dimension_order, semantic_note}；无声明列则返回 {}。
    """
    from app.tools._data_access import parse_lerobot_vector
    from app.tools._sniffing import infer_dof_order, parse_block_declaration

    if not lerobot_info:
        return {}

    result: dict[str, Any] = {}
    for col in df.columns:
        decl = parse_block_declaration(lerobot_info, str(col))
        if decl is None:
            continue  # 未声明块分解 → 不处理（不猜测）。
        blocks, dof = decl["block_count"], decl["dof_per_block"]
        # 解析向量列为 (n_rows, dims) 矩阵。
        mat = None
        vals = df[col].tolist()
        rows: list[np.ndarray] = []
        for v in vals:
            vec = parse_lerobot_vector(v)
            rows.append(vec if vec is not None else None)
        ok = [v for v in rows if v is not None]
        if len(ok) < len(vals) * 0.5 or not ok or any(v.size != dof * blocks for v in ok):
            continue  # 多数行不可解析或维度与声明不符 → 跳过（不猜测）。
        mat = np.full((len(vals), dof * blocks), np.nan)
        for i, v in enumerate(rows):
            if v is not None:
                mat[i] = v

        order_info = infer_dof_order(lerobot_info, str(col), dof)
        dim_order = order_info["order"]
        # 位置/四元数切分：仅在维度名可判定（含 p?/q? 前缀）时给出，否则不解读。
        can_split = all(
            str(n).startswith(("p", "q")) for n in dim_order
        ) and len(dim_order) == dof

        block_stats: dict[str, Any] = {}
        shown = min(blocks, _MAX_SKELETON_BLOCKS_SHOWN)
        for b in range(shown):
            seg = mat[:, b * dof : (b + 1) * dof]
            entry: dict[str, Any] = {
                "position_min": [round(float(x), 4) for x in np.nanmin(seg[:, :3], axis=0)],
                "position_max": [round(float(x), 4) for x in np.nanmax(seg[:, :3], axis=0)],
                "position_range": [
                    round(float(x), 4)
                    for x in (np.nanmax(seg[:, :3], axis=0) - np.nanmin(seg[:, :3], axis=0))
                ],
            }
            if can_split and dof >= 4:
                quats = seg[:, 3:7]
                norms = np.linalg.norm(quats, axis=1)
                entry["quaternion_norm_mean"] = round(float(np.nanmean(norms)), 4)
                entry["quaternion_norm_stable"] = bool(
                    np.nanstd(norms) < 0.01 and abs(float(np.nanmean(norms)) - 1.0) < 0.01
                )
            block_stats[f"block_{b}"] = entry

        result[str(col)] = {
            "block_count": blocks,
            "dof_per_block": dof,
            "declared_name": decl["declared_name"],
            "declaration_source": decl["declaration_source"],
            "blocks_shown": shown,
            "blocks_truncated": blocks > shown,
            "dimension_order": dim_order,
            "dimension_order_source": order_info["source"],
            "dimension_order_inferred": order_info["is_inferred"],
            "blocks": block_stats,
            "semantic_note": (
                f"{blocks} 块 × {dof}DoF（"
                + ("3 位置 + 4 四元数" if can_split else "维度语义未声明")
                + "）；位置单位为米，非关节角度——本指标为骨骼位姿范围，"
                "不等同 joint_range_of_motion（关节角度范围）。"
            ),
        }
    return result


def compute_stats_impl(
    context: RunContext,
    metric: str | None = None,
    column: str | None = None,
    group_by: str | None = None,
    table: str | None = None,
    settings=None,
) -> dict[str, Any]:
    """执行任务级统计。

    Args:
        context: 运行时上下文。
        metric: 可选，指定指标；省略时返回全部可用指标。
        column: 可选，通用统计的列。
        group_by: 可选，分组列。
        table: 可选，目标表名（文件名）；缺省自动定位状态/动作表（主表或独立表）。
        settings: 应用配置；缺省读取 get_settings()。

    Returns:
        dict，含 success、dataset、table_name、stats、metrics、qc_summary、
        semantic_notes、findings、user_message。
    """
    settings = settings or get_settings()
    capabilities = context.meta.get("capabilities", {})
    dataset_id = context.dataset_id

    if table is not None:
        # 显式指定表：经统一多表入口按名惰性读取（不替换主表）。
        resolved = _data_access.resolve_table_name(context, table)
        if not resolved["success"]:
            return {
                "success": False,
                "error": resolved.get("error", "table_unavailable"),
                "reason": resolved.get("reason"),
                "table": table,
                "dataset": dataset_id,
                "user_message": resolved.get("user_message", f"指定的表 {table} 不可用。"),
            }
        df = resolved["df"]
        _source = resolved["source"]
    else:
        resolved = None
        df, _source = _data_access.locate_action_table(context)

    # 前置条件：无状态/动作表 → 不适用。
    if df is None:
        return {
            "success": False,
            "error": "not_applicable",
            "reason": "数据集中无可用的状态/动作数据表",
            "user_message": "compute_stats 需要状态/动作数据表（含 episode、success 或关节列）。当前数据集无可用的任务级数据，不适用。",
            "dataset": dataset_id,
            "suggested_tools": ["profile_data", "inspect_streams"],
        }

    semantic_notes: list[str] = []

    # episode 列识别（语义注明）。
    episode_col = _find_col(df, _EPISODE_COLS)
    if episode_col is None:
        semantic_notes.append("未找到 episode 划分列，将整段数据视为一个 episode。")
        # 构造虚拟 episode 列（整段为一个 episode）。
        ep_series = pd.Series(0, index=df.index)
    else:
        ep_series = df[episode_col]

    # success 列识别（聚合规则注明）。
    success_col = _find_col(df, _SUCCESS_COLS)
    if success_col is not None:
        semantic_notes.append(
            f"success 聚合规则：按 episode 取末帧值（列 {success_col}），该规则为推测，"
            "若数据为逐帧 success 或含奖励，请人工确认。"
        )

    # ---- 内置任务级指标 ----
    metrics: dict[str, Any] = {}

    # episode 时长与长度分布。
    ep_counts = ep_series.value_counts().sort_index()
    metrics["episode_distribution"] = {
        "n_episodes": int(ep_counts.shape[0]),
        "length_per_episode": {str(k): int(v) for k, v in ep_counts.items()},
    }

    # 按 episode 聚合的成功率（success 列存在时）。
    if success_col is not None:
        succ = df.groupby(ep_series)[success_col].last()
        total = int(len(succ))
        n_success = int(succ.sum()) if pd.api.types.is_numeric_dtype(succ) else 0
        metrics["success_rate"] = {
            "per_episode": {str(k): (int(v) if pd.notna(v) else None) for k, v in succ.items()},
            "overall": round(n_success / total, 4) if total else None,
            "n_episodes": total,
            "aggregation_rule": "取每 episode 末帧 success 值",
            "success_column": success_col,
        }

    # 关节活动范围（qpos/joint 命中列）。
    joint_cols = [c for c in df.columns if str(c).lower().startswith(_JOINT_PREFIXES)]
    if joint_cols:
        rom: dict[str, Any] = {}
        for c in joint_cols:
            series = df[c]
            if pd.api.types.is_numeric_dtype(series):
                rom[c] = {
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "range": round(float(series.max() - series.min()), 4),
                }
        metrics["joint_range_of_motion"] = rom

    # 骨骼位姿范围：仅处理数据集声明的块分解列（如 body_24x7 / left_hand_26x7）。
    # 与 joint_range_of_motion（关节角度）严格区分，避免"位姿范围"被当"关节角度"解读。
    lerobot_info = context.meta.get("lerobot_info") or {}
    skeleton = _skeleton_pose_range(df, lerobot_info)
    if skeleton:
        metrics["skeleton_pose_range"] = skeleton
        inferred = [c for c, v in skeleton.items() if v.get("dimension_order_inferred")]
        if inferred:
            semantic_notes.append(
                f"骨骼位姿列 {', '.join(inferred)} 的块内维度顺序为**参照推断**"
                "（本列 names 仅为组合名），非数据集直接声明，引用时请注意。"
            )

    # 任务完成时长离群 episode（IQR 法，基于每 episode 时长）。
    duration_col = None
    for c in df.columns:
        if str(c).lower().strip() in ("duration", "time", "elapsed", "duration_s"):
            duration_col = str(c)
            break
    outlier_episodes: list[Any] = []
    episode_duration: dict[str, Any] | None = None
    if duration_col is not None:
        # 每 episode 的时长（取末帧时长或均值）。
        ep_dur = df.groupby(ep_series)[duration_col].last()
        ep_dur = ep_dur.dropna()
        if len(ep_dur) >= 1:
            episode_duration = {
                "min": round(float(ep_dur.min()), 4),
                "median": round(float(ep_dur.median()), 4),
                "max": round(float(ep_dur.max()), 4),
                "per_episode": {str(k): round(float(v), 4) for k, v in ep_dur.items()},
            }
            metrics["episode_duration"] = episode_duration
        if len(ep_dur) >= 4:
            q1 = float(ep_dur.quantile(0.25))
            q3 = float(ep_dur.quantile(0.75))
            iqr = q3 - q1
            k = settings.stats_outlier_k
            lower, upper = q1 - k * iqr, q3 + k * iqr
            outlier_episodes = [
                {"episode": int(ep), "duration": round(float(d), 4)}
                for ep, d in ep_dur.items() if d < lower or d > upper
            ]
            metrics["outlier_episodes"] = {
                "method": "IQR",
                "k": k,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "outliers": outlier_episodes,
            }

    # ---- 通用统计 ----
    stats = _generic_stats(df, column, group_by)

    # ---- 质检联动 ----
    qc = context.meta.get("qc", {})
    if qc:
        qc_summary = qc
    else:
        qc_summary = {
            "status": "未质检",
            "note": "该数据集尚未运行质检工具，建议先运行 check_temporal_sync 与 check_sensor_sanity。",
        }

    # ---- findings 累积 ----
    finding = {
        "tool": "compute_stats",
        "type": "stat",
        "metric": metric or "task_level",
        "n_episodes": metrics.get("episode_distribution", {}).get("n_episodes"),
        "summary": _make_finding_summary(metrics, metric),
        "semantic_notes": list(semantic_notes),
        # 关键统计数字（供报告渲染明细），内容全部来自 metrics 的真实计算。
        "metrics": {
            "n_episodes": metrics.get("episode_distribution", {}).get("n_episodes"),
            "success_rate": metrics.get("success_rate", {}).get("overall"),
            "joint_range_of_motion": metrics.get("joint_range_of_motion"),
            "skeleton_pose_range": metrics.get("skeleton_pose_range"),
            "outlier_episodes": metrics.get("outlier_episodes"),
            "episode_duration": metrics.get("episode_duration"),
        },
    }
    context.findings.append(finding)

    # ---- 装载完整性继承声明 ----
    # 若主表被截断（超行数阈值），统计基于截断后的数据，必须在返回中明确标注，
    # 避免把样本当全量描述。
    main_table_meta = context.meta.get("main_table", {})
    data_scope: dict[str, Any] | None = None
    if main_table_meta.get("truncated"):
        data_scope = {
            "truncated": True,
            "rows_total": main_table_meta.get("rows_total"),
            "rows_loaded": main_table_meta.get("rows_loaded"),
            "note": (
                f"统计基于截断数据：仅前 {main_table_meta.get('rows_loaded')} 行"
                f"（共 {main_table_meta.get('rows_total')} 行）已装载，"
                "超出部分未载入内存，结论仅代表已装载部分。"
            ),
        }

    # 结果标注表名与数据集归属。
    table_name = (
        resolved["table_name"] if resolved is not None
        else (context.meta.get("main_table", {}).get("name") if _source in ("main", "main_fallback") else None)
    )
    source_note = (
        "（显式指定表，按名惰性读取，未替换主表）"
        if table is not None else
        ("（主表）" if _source == "main" else
         ("（独立状态/动作表）" if _source == "actions_stream" else
          ("（主表兜底）" if _source == "main_fallback" else "")))
    )

    return {
        "success": True,
        "dataset": dataset_id,
        "table": table_name,
        "table_name": table_name,
        "data_source": _source,
        "metric": metric,
        "stats": stats,
        "metrics": metrics,
        "qc_summary": qc_summary,
        "semantic_notes": semantic_notes,
        "findings": [finding],
        "data_scope": data_scope,
        "user_message": (
            f"任务级统计完成（数据集 {dataset_id}，表 {table_name}{source_note}）："
            f"{metrics.get('episode_distribution', {}).get('n_episodes', 0)} 个 episode。"
            + (f" 成功率 {metrics['success_rate']['overall']}。" if 'success_rate' in metrics else "")
            + (f" 离群 episode {len(outlier_episodes)} 个。" if outlier_episodes else "")
            + (
                f" 骨骼位姿范围（{len(skeleton)} 列，按声明块分解，非关节角度）。"
                if skeleton else ""
            )
            + (f" {data_scope['note']}" if data_scope else "")
        ),
    }


def _make_finding_summary(metrics: dict[str, Any], metric: str | None) -> str:
    """生成 findings 的一句话结论摘要。"""
    parts: list[str] = []
    if "episode_distribution" in metrics:
        parts.append(f"{metrics['episode_distribution']['n_episodes']} 个 episode")
    if "success_rate" in metrics:
        parts.append(f"成功率 {metrics['success_rate']['overall']}")
    if "outlier_episodes" in metrics:
        parts.append(f"{len(metrics['outlier_episodes']['outliers'])} 个离群 episode")
    return "；".join(parts) if parts else "任务级统计完成"


@tool
def compute_stats(
    wrapper: RunContextWrapper[RunContext],
    metric: str | None = None,
    column: str | None = None,
    group_by: str | None = None,
    table: str | None = None,
) -> dict:
    """计算任务级统计指标（episode 分布、成功率、关节活动范围、离群 episode 等）。

    数据来源默认自动定位状态/动作表（主表或独立表）；也可显式指定 table（文件名）
    对目录内某张表统计，该表按名惰性读取、不替换主表。语义不明时标注所用规则与
    依据列名（如 success 取末帧、episode 缺失时整段视为一个 episode）。

    Args:
        metric: 可选，指定指标；省略时返回全部可用指标。
        column: 可选，通用统计的列。
        group_by: 可选，分组列。
        table: 可选，目标表名（如 "accel.csv"）；缺省自动定位状态/动作表。

    Returns:
        dict，含 success、dataset、table_name、stats、metrics、qc_summary、
        semantic_notes、findings、user_message；无可用的状态/动作表时返回
        not_applicable；指定表不存在时返回 table_not_found。
    """
    return compute_stats_impl(wrapper.context, metric, column, group_by, table)
