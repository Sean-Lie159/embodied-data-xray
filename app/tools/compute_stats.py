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


def _has_action_columns(df: pd.DataFrame) -> bool:
    """判断 df 是否含状态/动作相关列（episode/success/关节）。"""
    if _find_col(df, _EPISODE_COLS) is not None:
        return True
    if _find_col(df, _SUCCESS_COLS) is not None:
        return True
    return any(str(c).lower().startswith(_JOINT_PREFIXES) for c in df.columns)


def _read_actions_table(context: RunContext) -> pd.DataFrame | None:
    """确定任务级统计的数据表：优先含状态/动作列的主表，否则从流登记表按需读取。

    Args:
        context: 运行时上下文。

    Returns:
        状态/动作数据表；无则返回 None。
    """
    # 优先 context.df，但仅当它含状态/动作列（episode/success/关节）。
    if context.df is not None and _has_action_columns(context.df):
        return context.df

    # 从流登记表读 actions 流（状态/动作在独立表中），读全表（不只 channels）。
    streams = context.meta.get("streams", [])
    for s in streams:
        if s.get("kind") == "actions":
            path = s.get("path", "")
            fmt = s.get("format", "")
            try:
                from app.tools.load_dataset import _detect_encoding

                if fmt == "csv":
                    encoding = _detect_encoding(Path(path).read_bytes())
                    return pd.read_csv(path, encoding=encoding, engine="python")
                if fmt == "parquet":
                    return pd.read_parquet(path)
                if fmt == "json":
                    return pd.read_json(path)
            except Exception:  # noqa: BLE001
                continue
    return None

    # 兜底：主表存在但无状态/动作列时，仍返回主表（后续语义注明会提示 episode 缺失）。
    if context.df is not None:
        return context.df
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
    settings=None,
) -> dict[str, Any]:
    """执行任务级统计。

    Args:
        context: 运行时上下文。
        metric: 可选，指定指标；省略时返回全部可用指标。
        column: 可选，通用统计的列。
        group_by: 可选，分组列。
        settings: 应用配置；缺省读取 get_settings()。

    Returns:
        dict，含 success、dataset、stats、metrics、qc_summary、semantic_notes、
        findings、user_message。
    """
    settings = settings or get_settings()
    capabilities = context.meta.get("capabilities", {})
    dataset_id = context.dataset_id

    df = _read_actions_table(context)

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
    if duration_col is not None:
        # 每 episode 的时长（取末帧时长或均值）。
        ep_dur = df.groupby(ep_series)[duration_col].last()
        ep_dur = ep_dur.dropna()
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
    }
    context.findings.append(finding)

    return {
        "success": True,
        "dataset": dataset_id,
        "metric": metric,
        "stats": stats,
        "metrics": metrics,
        "qc_summary": qc_summary,
        "semantic_notes": semantic_notes,
        "findings": [finding],
        "user_message": (
            f"任务级统计完成（{dataset_id}）：{metrics.get('episode_distribution', {}).get('n_episodes', 0)} 个 episode。"
            + (f" 成功率 {metrics['success_rate']['overall']}。" if 'success_rate' in metrics else "")
            + (f" 离群 episode {len(outlier_episodes)} 个。" if outlier_episodes else "")
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
) -> dict:
    """计算任务级统计指标（episode 分布、成功率、关节活动范围、离群 episode 等）。

    数据来源默认 context.df；若状态/动作在独立表中则按流登记表按需读取。语义不明
    时标注所用规则与依据列名（如 success 取末帧、episode 缺失时整段视为一个 episode）。

    Args:
        metric: 可选，指定指标；省略时返回全部可用指标。
        column: 可选，通用统计的列。
        group_by: 可选，分组列。

    Returns:
        dict，含 success、dataset、stats、metrics、qc_summary、semantic_notes、
        findings、user_message；无可用的状态/动作表时返回 not_applicable。
    """
    return compute_stats_impl(wrapper.context, metric, column, group_by)
