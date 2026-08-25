"""数据集概况工具。

基于 ``RunContext`` 中已加载的 DataFrame（由 load_dataset 写入）生成精简概况：
行数、列名与类型、缺失统计、数值列基本统计、每列唯一值数量。返回精简 dict，
不返回数据本体。未加载数据时返回"请先调用 load_dataset"的结构化错误。
"""

from __future__ import annotations

from typing import Any

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools import _data_access


def profile_data_impl(context: RunContext, max_unique: int = 20, table: str | None = None) -> dict:
    """分析当前已加载的数据集并返回概况。

    支持多表：缺省 table=None 分析主表；显式传 table 按流登记表按名惰性读取指定
    表，**不替换主表状态**（context.df 保持主表不变）。结果注明表名与数据集归属。

    Args:
        context: 运行时上下文，需已通过 load_dataset 加载数据（context.df 非空）。
        max_unique: 每列最多展示的样例值数量，防止结果过大。
        table: 可选，目标表名（文件名，如 "accel.csv"）；缺省=主表。

    Returns:
        dict，包含 success、dataset（本次结果产自的数据集名）、table_name、
        n_rows、n_cols，以及 columns 列表；每个元素含 name、dtype、n_missing、
        pct_missing、n_unique、sample_values，数值列另有 numeric 子字典
        （mean/std/min/max/median）。未加载数据时返回 success=False 且提示先调用
        load_dataset；指定表不存在时返回 success=False 且 error="table_not_found"。

    Raises:
        不直接抛出异常；错误以结构化 dict 的 error 字段返回，便于 Agent 恢复。
    """
    resolved = _data_access.resolve_table_name(context, table)
    if not resolved["success"]:
        if resolved.get("error") == "table_not_found" or resolved.get("error") == "table_read_failed":
            return {
                "success": False,
                "error": resolved["error"],
                "reason": resolved.get("reason"),
                "table": table,
                "dataset": context.dataset_id,
                "user_message": resolved.get("user_message", "指定的表不可用。"),
            }
        # 主表未加载。
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载任何数据集",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset 加载数据，再执行 profile_data 分析概况。",
        }

    df = resolved["df"]
    dataset = resolved["dataset"]
    table_name = resolved["table_name"]
    columns: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        n_missing = int(series.isna().sum())
        pct_missing = (n_missing / len(df) * 100) if len(df) else 0.0
        try:
            n_unique = int(series.nunique(dropna=True))
        except TypeError:
            n_unique = -1  # 不可哈希类型（如列表），用 -1 占位

        col: dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 2),
            "n_unique": n_unique,
            "sample_values": [
                str(v) for v in series.dropna().head(max_unique).tolist()
            ],
        }

        if series.dtype.kind in "iufc":  # 数值列（int/uint/float/complex）
            desc = series.describe()
            col["numeric"] = {
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min": _safe_float(desc.get("min")),
                "max": _safe_float(desc.get("max")),
                "median": _safe_float(series.median(skipna=True)),
            }

        columns.append(col)

    return {
        "success": True,
        "dataset": dataset,
        "table": table_name,
        "table_name": table_name,
        "data_source": resolved["source"],
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": columns,
        "user_message": (
            f"已分析表 {table_name}（数据集 {dataset}）概况：{int(df.shape[0])} 行 × {int(df.shape[1])} 列。"
            + (" 该表非主表，为按名惰性读取（未替换主表）。" if resolved["source"] == "stream_lazy" else "")
        ),
    }


def _safe_float(value: Any) -> float | None:
    """将值安全转为 float，NaN 或非数值返回 None。"""
    import numpy as np

    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@tool
def profile_data(
    wrapper: RunContextWrapper[RunContext],
    max_unique: int = 20,
    table: str | None = None,
) -> dict:
    """分析当前已加载数据集并返回概况。

    对当前会话中已加载的数据（需先调用 load_dataset）统计行数、列类型、缺失、
    唯一值与数值统计。缺省分析主表；指定 table（文件名）可分析目录内的其他表，
    该表按名惰性读取，不替换主表。

    Args:
        max_unique: 每列最多展示的样例值数量，防止结果过大。
        table: 可选，目标表名（如 "accel.csv"）；缺省=主表。

    Returns:
        dict，包含 success、dataset、table_name、n_rows、n_cols 与 columns 列表；
        未加载数据时返回 success=False 并提示先调用 load_dataset；指定表不存在时
        返回 success=False 且 error="table_not_found"。
    """
    return profile_data_impl(wrapper.context, max_unique, table)
