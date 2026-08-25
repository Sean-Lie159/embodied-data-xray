"""共享数据访问辅助（工具层）。

提供统一函数：根据能力标签命中列证据定位状态/动作数据表——主表含目标列则用主表，
否则按流登记表按需读取对应独立表（全表）。供 compute_stats、plot_chart 等需要
访问状态/动作数据的工具复用，避免各自直接读 context.df 造成"独立表误用主表"问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.agent.context import RunContext

# episode 列候选。
_EPISODE_COLS = ("episode", "ep", "eps", "episode_id", "traj_id", "trajectory_id")
# success 列候选。
_SUCCESS_COLS = ("success", "successful", "done")
# 关节列前缀（状态/动作）。
_JOINT_PREFIXES = ("qpos", "qvel", "qacc", "joint")


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """在 df 中查找候选列（大小写不敏感）。

    Args:
        df: 数据表。
        candidates: 候选列名。

    Returns:
        命中的列名；未找到返回 None。
    """
    for c in df.columns:
        if str(c).lower().strip() in candidates:
            return str(c)
    return None


def has_action_columns(df: pd.DataFrame) -> bool:
    """判断 df 是否含状态/动作相关列（episode/success/关节）。

    Args:
        df: 数据表。

    Returns:
        是否含状态/动作列。
    """
    if find_column(df, _EPISODE_COLS) is not None:
        return True
    if find_column(df, _SUCCESS_COLS) is not None:
        return True
    return any(str(c).lower().startswith(_JOINT_PREFIXES) for c in df.columns)


def read_stream_full(path: str, fmt: str) -> pd.DataFrame | None:
    """按需读取流文件的全表。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json）。

    Returns:
        DataFrame；读取失败返回 None。
    """
    from app.tools.load_dataset import _detect_encoding

    try:
        if fmt == "csv":
            encoding = _detect_encoding(Path(path).read_bytes())
            return pd.read_csv(path, encoding=encoding, engine="python")
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "json":
            return pd.read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return None


def resolve_table_name(context: RunContext, table: str | None) -> dict[str, Any]:
    """解析表名对应的 DataFrame 与来源，统一多表入口（惰性读取，不替换主表）。

    缺省（table=None）→ 主表（context.df）；显式给表名 → 按流登记表按文件名查找，
    找到则按需读全表，找不到返回结构化错误（不抛异常）。返回 dict 统一含
    ``df``（DataFrame 或 None）、``table_name``（实际表名，用于结果标注）、
    ``dataset``（归属数据集）、``source``（main / stream_lazy / error）。

    Args:
        context: 运行时上下文。
        table: 可选，目标表名（文件名，如 "accel.csv"）。缺省=主表。

    Returns:
        dict，含 success、df、table_name、dataset、source；表不存在时
        success=False 且 error="table_not_found"。
    """
    dataset_id = context.dataset_id
    # 未加载任何数据集：无论是否指定表都返回 no_data_loaded（优先于表不存在）。
    if dataset_id is None and context.df is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载任何数据集",
            "df": None,
            "table_name": table,
            "dataset": None,
            "source": "error",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset 加载数据，再执行分析。",
        }

    # 缺省 → 主表。
    if table is None:
        return {
            "success": context.df is not None,
            "df": context.df,
            "table_name": context.meta.get("main_table", {}).get("name"),
            "dataset": dataset_id,
            "source": "main",
        }

    # 显式表名 → 按流登记表查找（文件名精确匹配，忽略大小写）。
    name_lower = table.strip().lower()
    for s in context.meta.get("streams", []):
        p = s.get("path", "")
        if Path(p).name.lower() == name_lower:
            df = read_stream_full(p, s.get("format", ""))
            if df is not None:
                return {
                    "success": True,
                    "df": df,
                    "table_name": Path(p).name,
                    "dataset": dataset_id,
                    "source": "stream_lazy",
                }
            return {
                "success": False,
                "error": "table_read_failed",
                "reason": f"流登记表存在 {table} 但读取失败",
                "df": None,
                "table_name": table,
                "dataset": dataset_id,
                "source": "stream_lazy",
                "user_message": f"已找到流 {table}，但按流登记表读取其内容失败，无法分析。",
            }

    return {
        "success": False,
        "error": "table_not_found",
        "reason": f"数据集中不存在表 {table}",
        "df": None,
        "table_name": table,
        "dataset": dataset_id,
        "source": "error",
        "user_message": (
            f"当前数据集 {dataset_id} 中不存在表 {table}。可用表见流登记表/inspect_streams 的表格流清单。"
        ),
    }


def locate_action_table(context: RunContext) -> tuple[pd.DataFrame | None, str | None]:
    """定位状态/动作数据表，返回 (DataFrame, 来源说明)。

    优先主表（仅当主表含状态/动作列）；否则按流登记表读取 kind=actions 的独立表；
    都无则兜底返回主表（来源标为 "main_fallback"）。

    Args:
        context: 运行时上下文。

    Returns:
        (df, source)：df 为数据表（或 None），source 为 "main" / "actions_stream" /
        "main_fallback" / None（无任何表）。
    """
    # 优先主表（含状态/动作列）。
    if context.df is not None and has_action_columns(context.df):
        return context.df, "main"

    # 从流登记表读 actions 独立表（全表）。
    for s in context.meta.get("streams", []):
        if s.get("kind") == "actions":
            df = read_stream_full(s.get("path", ""), s.get("format", ""))
            if df is not None:
                return df, "actions_stream"

    # 兜底：主表存在但无状态/动作列。
    if context.df is not None:
        return context.df, "main_fallback"
    return None, None
