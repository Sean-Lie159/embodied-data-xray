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
