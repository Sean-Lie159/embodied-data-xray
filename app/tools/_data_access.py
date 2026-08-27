"""共享数据访问辅助（工具层）。

提供统一函数：根据能力标签命中列证据定位状态/动作数据表——主表含目标列则用主表，
否则按流登记表按需读取对应独立表（全表）。供 compute_stats、plot_chart 等需要
访问状态/动作数据的工具复用，避免各自直接读 context.df 造成"独立表误用主表"问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


# JSON 行列表键：单一事实来源在 _sniffing（语义角色识别的常量层）。
from app.tools._sniffing import _JSON_ROW_LIST_KEYS  # noqa: E402


def parse_lerobot_vector(value: Any) -> np.ndarray | None:
    """解析 LeRobot 向量值（空格/换行分隔字符串 或 JSON 数组 或 list/tuple）。

    LeRobot 的 object 列（如 observation.left_hand）常存为 "0. 0. 0. \n 0. 0. ..."
    的空格/换行分隔字符串，或 JSON 数组；本函数统一解析为数值数组。

    Args:
        value: 单元格值（str / list / tuple / ndarray）。

    Returns:
        数值数组；无法解析返回 None。
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return np.asarray(value, dtype=float)
        except (ValueError, TypeError):
            return None
    if isinstance(value, (int, float)):
        return np.asarray([float(value)])
    s = str(value).strip()
    if not s:
        return None
    # 去掉可能的中括号/逗号，按空白切分。
    s = s.replace("[", " ").replace("]", " ").replace(",", " ").replace("\n", " ")
    parts = [p for p in s.split() if p]
    if not parts:
        return None
    try:
        return np.asarray([float(p) for p in parts], dtype=float)
    except ValueError:
        return None


def _json_row_list(obj: Any) -> list | None:
    """从已解析的 JSON 对象提取行记录列表。

    顶层为 list → 直接作为行列表；顶层为 dict → 取第一个行列表键（frames/data）的
    值（须为 list）。否则返回 None。

    Args:
        obj: json.loads 的返回。

    Returns:
        行记录列表；无则返回 None。
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in _JSON_ROW_LIST_KEYS:
            v = obj.get(key)
            if isinstance(v, list):
                return v
        return None
    return None


def read_stream_full(path: str, fmt: str) -> pd.DataFrame | None:
    """按需读取流文件的全表。

    JSON 顶层 dict 时按行列表键（frames/data）展开为 DataFrame，避免把标量键
    （如 fps）当数据列。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json）。

    Returns:
        DataFrame；读取失败返回 None。
    """
    import json as _json

    from app.tools.load_dataset import _detect_encoding

    try:
        if fmt == "csv":
            encoding = _detect_encoding(Path(path).read_bytes())
            return pd.read_csv(path, encoding=encoding, engine="python")
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "json":
            encoding = _detect_encoding(Path(path).read_bytes())
            obj = _json.loads(Path(path).read_text(encoding=encoding))
            rows = _json_row_list(obj)
            if rows is None:
                return None
            return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001
        return None
    return None


def read_table_nrows(path: str, fmt: str) -> int | None:
    """只读表格行数（不读全量数据）。

    统一行数读数入口：inspect_streams / check_temporal_sync / load_dataset 主表评分
    都经此函数获取行数，避免各自实现导致同一文件行数读数不一致。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json）。

    Returns:
        行数（不含表头）；读取失败返回 None。
    """
    import json as _json

    from app.tools.load_dataset import _detect_encoding

    try:
        if fmt == "csv":
            encoding = _detect_encoding(Path(path).read_bytes())
            # 用 python 引擎只读首列以降低成本；与全量读同一引擎，行数一致。
            return int(pd.read_csv(path, encoding=encoding, usecols=[0], engine="python").shape[0])
        if fmt == "parquet":
            return int(pd.read_parquet(path, columns=None).shape[0])
        if fmt == "json":
            encoding = _detect_encoding(Path(path).read_bytes())
            obj = _json.loads(Path(path).read_text(encoding=encoding))
            rows = _json_row_list(obj)
            return len(rows) if rows is not None else 0
        return None
    except Exception:  # noqa: BLE001
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
