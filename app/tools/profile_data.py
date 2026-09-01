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
    # 数据集声明的语义元数据（LeRobot meta/*.json，确定性解析结果）：
    # 列维度名（features.names）与每列统计量（stats.json），供直接引用而非推测。
    lerobot_info = context.meta.get("lerobot_info") or {}
    lerobot_stats = context.meta.get("lerobot_stats") or {}
    stats_features = lerobot_stats.get("features")
    stats_features = stats_features if isinstance(stats_features, dict) else {}

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

        # 列维度名：以数据集声明为准（有则引用，无则不得推测）。
        dim_names = _column_dimension_names(lerobot_info, str(name))
        if dim_names:
            col["dimension_names"] = dim_names
            col["dimension_source"] = "meta/info.json features（数据集声明）"
            if isinstance(lerobot_info.get("column_semantics", {}).get(str(name)), dict):
                col["declared_shape"] = lerobot_info["column_semantics"][str(name)].get("shape")

        # 数据集自带统计量（stats.json）：直接采用，避免重算。
        col_stats = stats_features.get(str(name))
        if isinstance(col_stats, dict):
            col["dataset_stats"] = col_stats
            col["dataset_stats_source"] = "meta/stats.json"

        if series.dtype.kind in "iufc":  # 数值列（int/uint/float/complex）
            desc = series.describe()
            col["numeric"] = {
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min": _safe_float(desc.get("min")),
                "max": _safe_float(desc.get("max")),
                "median": _safe_float(series.median(skipna=True)),
            }
        else:
            # object 列：尝试按向量解析（LeRobot 观测/动作列常为空格分隔向量串），
            # 做**全列**零/非零分布统计——样例值只反映列首，不能据此外推整列。
            vstats = _vector_column_stats(series)
            if vstats is not None:
                col["vector_stats"] = vstats
                col["sample_values_note"] = (
                    f"样例值仅取列首前 {max_unique} 行；该列全列分布见 vector_stats"
                    "（起始段可能不代表整体，勿据样例外推全列）"
                )

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


def _column_dimension_names(lerobot_info: dict[str, Any], column: str) -> list[str] | None:
    """取列各维度的权威名称（来自 LeRobot meta/info.json 的 features.names）。

    Args:
        lerobot_info: parse_lerobot_info 的返回。
        column: 列名。

    Returns:
        维度名列表；数据集未声明则返回 None（此时工具不得推测维度含义）。
    """
    if not lerobot_info:
        return None
    from app.tools._sniffing import column_dimension_names

    return column_dimension_names(lerobot_info, column)


def _vector_column_stats(series) -> dict[str, Any] | None:
    """对 object 列做全列向量分布统计（零/非零行占比与维度）。

    逐行尝试用 parse_lerobot_vector 解析（空格/换行分隔向量串 / JSON 数组 / list）。
    仅当**多数行可解析**（占比 ≥ 0.5）时才认定该列为向量列并返回统计，避免把普通
    字符串列误套向量统计。

    Args:
        series: 列 Series（object 类型）。

    Returns:
        dict，含 n_parsed（解析成功行数）、n_zero（全零向量行数）、
        n_nonzero（含非零分量的行数）、zero_row_ratio（全零行占比）、dims（向量维度，
        取众数）；非向量列返回 None。
    """
    import numpy as np

    from app.tools._data_access import parse_lerobot_vector

    total = len(series)
    if total == 0:
        return None
    n_parsed = 0
    n_zero = 0
    n_nonzero = 0
    zero_idx: list[int] = []
    dim_counter: dict[int, int] = {}
    for i, v in enumerate(series):
        vec = parse_lerobot_vector(v)
        if vec is None:
            continue
        n_parsed += 1
        dim_counter[int(vec.size)] = dim_counter.get(int(vec.size), 0) + 1
        if np.any(np.abs(vec) > 1e-9):
            n_nonzero += 1
        else:
            n_zero += 1
            zero_idx.append(i)
    # 多数行可解析才认定是向量列。
    if n_parsed < total * 0.5:
        return None
    dims = max(dim_counter, key=dim_counter.get) if dim_counter else None
    stats: dict[str, Any] = {
        "n_parsed": n_parsed,
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
        "zero_row_ratio": round(n_zero / n_parsed, 4) if n_parsed else None,
        "dims": dims,
    }
    # 全零行位置摘要：缺失"发生在哪"直接影响补全策略（补零/裁剪/插值），
    # 只有数量会让 agent 误把散布的缺失说成"集中在开头"。
    if zero_idx:
        consecutive = zero_idx == list(range(zero_idx[0], zero_idx[0] + len(zero_idx)))
        # 十分位分布（把序列均分 10 段，统计每段全零行数）。
        bins = np.array_split(np.arange(total), 10)
        dist = [int(sum(1 for i in zero_idx if i in seg)) for seg in bins]
        stats["zero_row_positions"] = {
            "first": zero_idx[0],
            "last": zero_idx[-1],
            "contiguous": consecutive,
            "in_head_20": int(sum(1 for i in zero_idx if i < 20)),
            "decile_distribution": dist,
        }
    return stats


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
