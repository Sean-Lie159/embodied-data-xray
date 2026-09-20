"""跨表逐行运算工具（对比层）。

**为什么需要**（2026-09-20 真实需求）：用户/验收方常提出"实际末端位置与指令末端
相差多少""两条流的某个通道是否一致"这类问题——它需要**同时读两张表 + 按对齐键
逐行运算**。而既有的 `profile_data` / `compute_stats` 都只做**单表单列聚合**，
无任何工具支持两表对齐后的行级运算，导致这类问题无法回答。

本工具按**对齐键**（缺省 ``frame_index``）把两张表对齐，对指定列做逐行运算，
产出差异统计与**分布形态**。

设计纪律（与项目既有约定一致）：
- **不对齐就不算**：对齐键缺失或无法对齐时返回结构化错误，不做位置对齐
  （按行号硬对齐会产出看似合理实则无意义的数字）；
- **未匹配不静默丢弃**：两侧行数不等时如实报出差值；
- **单位不猜**：只做"同单位下的数值差"；不做自动换算（换算方向错了会静默
  产出错误结论）；
- **不做合格判定**：只产出数值统计，不判断"差异是否可接受"（阈值属验收标准，
  不在工具职责内）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools import _data_access

# 默认对齐键（"每帧一组"布局与多数时序表都用它）。
_DEFAULT_KEY = "frame_index"

# 返回的样例行数上限（供人工核对，不灌爆上下文）。
_MAX_SAMPLE_ROWS = 5

# 支持的对齐键候选（缺省键不存在时按序尝试；都没有则报错，不猜）。
_KEY_CANDIDATES = ("frame_index", "frame_id", "index", "timestamp", "ts")


def _align_by_key(
    df_a: pd.DataFrame, df_b: pd.DataFrame, on: str | None
) -> dict[str, Any]:
    """按对齐键把两表内连接对齐。

    Args:
        df_a: 表 A。
        df_b: 表 B。
        on: 对齐键列名；None 时按 ``_KEY_CANDIDATES`` 自动探测（两表共有的第一个）。

    Returns:
        dict：success、key、joined（对齐后的 DataFrame）、n_a、n_b、n_matched、
        error/reason（失败时）。
    """
    key = on
    if key is None:
        for cand in _KEY_CANDIDATES:
            if cand in df_a.columns and cand in df_b.columns:
                key = cand
                break
    if not key:
        return {
            "success": False,
            "error": "no_alignment_key",
            "reason": (
                f"两表没有共同的对齐键（已尝试 {list(_KEY_CANDIDATES)}）。"
                f"表 A 列：{list(map(str, df_a.columns))[:10]}；"
                f"表 B 列：{list(map(str, df_b.columns))[:10]}"
            ),
            "user_message": (
                "两张表没有可用于逐行对齐的键（如 frame_index / timestamp）。"
                "请用 on 参数显式指定对齐键列名。"
            ),
        }
    if key not in df_a.columns or key not in df_b.columns:
        return {
            "success": False,
            "error": "alignment_key_missing",
            "reason": f"对齐键 {key!r} 不在两表中同时存在",
            "user_message": (
                f"对齐键 {key!r} 不在两张表中同时存在"
                f"（表 A {'有' if key in df_a.columns else '无'}、"
                f"表 B {'有' if key in df_b.columns else '无'}）。"
            ),
        }

    # **对齐键可能不是唯一键**（2026-09-20 真实缺陷）。
    #
    # 真实案例：``state/end/position`` 与 ``action/end/position`` 都是"每帧 2 行"
    # （左/右两侧），``frame_index`` 有 14135 个唯一值但 28270 行——**每帧重复 2 次**。
    # 直接按 frame_index merge 会产出 **2×2=4 行的笛卡尔积**（实测 56540 行，
    # 比两侧各自还多），统计建立在**错误配对**上（左配右、右配左各一半）。
    #
    # 正确做法：键重复时按**组内序号**（同键值内的第 k 行）配对——这符合
    # "每帧多行 = 帧内多个实体，按出现顺序一一对应"的语义。
    dup_a = int(df_a[key].duplicated().sum()) > 0
    dup_b = int(df_b[key].duplicated().sum()) > 0
    grouping: str | None = None
    if dup_a or dup_b:
        # 两侧的键重复模式必须一致（否则无法确定配对规则，如实拒绝）。
        ca = df_a.groupby(key, sort=False).size()
        cb = df_b.groupby(key, sort=False).size()
        if not ca.equals(cb):
            return {
                "success": False,
                "error": "alignment_key_not_unique",
                "reason": (
                    f"对齐键 {key!r} 在两表中的重复模式不一致"
                    f"（A 侧每组行数分布与 B 侧不同，无法确定配对规则）"
                ),
                "user_message": (
                    f"对齐键 {key!r} 在两表中不是唯一的，且重复模式不一致"
                    "（如同一 frame_index 下 A 侧 2 行、B 侧 3 行），无法确定"
                    "如何配对。请改用真正唯一的键（如 row_id），"
                    "或用 columns 只比较两表中确定一一对应的部分。"
                ),
            }
        ka = "_cmp_k_a"
        kb = "_cmp_k_b"
        ma = df_a.assign(**{ka: df_a.groupby(key, sort=False).cumcount()})
        mb = df_b.assign(**{kb: df_b.groupby(key, sort=False).cumcount()})
        merged = ma.merge(mb, left_on=[key, ka], right_on=[key, kb],
                          how="inner", suffixes=("_a", "_b"))
        merged = merged.drop(columns=[ka, kb])
        grouping = "组内序号（同键值内的第 k 行一一对应）"
    else:
        merged = df_a.merge(df_b, on=key, how="inner", suffixes=("_a", "_b"))
    return {
        "success": True,
        "key": key,
        "joined": merged,
        "n_a": int(len(df_a)),
        "n_b": int(len(df_b)),
        "n_matched": int(len(merged)),
        "grouping": grouping,
    }


def _stats_of(values: np.ndarray) -> dict[str, Any]:
    """一组数值的统计与**分布形态**。

    为什么必须给分布：均值 5 cm 可能是"全部约 5 cm"（系统性偏移），也可能是
    "大部分 0、少数 30 cm"（个别跳变）——两者含义完全不同，只看均值会误判。
    故除分位数外，另外给出 **decile 直方**（把取值范围等分 10 段的计数），
    让"集中"与"离散"一眼可辨。
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(v.size),
        "mean": round(float(v.mean()), 6),
        "std": round(float(v.std()), 6),
        "min": round(float(v.min()), 6),
        "max": round(float(v.max()), 6),
        "median": round(float(np.median(v)), 6),
        "p05": round(float(np.percentile(v, 5)), 6),
        "p95": round(float(np.percentile(v, 95)), 6),
    }
    # decile 直方（等宽 10 段）：只给计数，段边界用 min/max 表达。
    lo, hi = float(v.min()), float(v.max())
    if hi > lo:
        edges = np.linspace(lo, hi, 11)
        counts, _ = np.histogram(v, bins=edges)
        out["decile_histogram"] = {
            "range": [round(lo, 6), round(hi, 6)],
            "counts": [int(c) for c in counts],
            "note": "把取值范围等分 10 段的计数，用于区分「集中」与「多数为 0、少数偏大」",
        }
    return out


def compare_table_columns_impl(
    context: RunContext,
    table_a: str,
    table_b: str,
    columns_a: list[str] | None = None,
    columns_b: list[str] | None = None,
    op: str = "diff",
    on: str | None = None,
) -> dict[str, Any]:
    """按对齐键对齐两张表，对指定列做逐行运算并产出差异统计。

    Args:
        context: 运行时上下文。
        table_a: 表 A 的表名（支持 ``<文件stem>::<节点>``、文件名、节点名）。
        table_b: 表 B 的表名。
        columns_a: 表 A 参与运算的列（按顺序与 columns_b 对应）；省略时取两表同名
            数值列的交集。
        columns_b: 表 B 参与运算的列；省略时同 columns_a。
        op: 运算类型——
            ``diff``：逐元素差（A - B，逐列）；
            ``distance``：欧氏距离（需 2–3 列，按行求模长）；
            ``ratio``：比值（A / B，B 为 0 处置 NaN）。
        on: 对齐键列名；缺省自动探测 ``frame_index`` 等。

    Returns:
        dict，含 success、table_a/table_b、aligned_on、n_a/n_b/n_matched（未匹配数
        如实给出）、columns_used、op、result（各列/各运算的统计与分布）、
        samples（前若干条对齐后的原值与结果，供核对）、user_message。
    """
    op_norm = (op or "diff").strip().lower()
    if op_norm not in ("diff", "distance", "ratio"):
        return {
            "success": False,
            "error": "unsupported_op",
            "reason": f"不支持的运算 {op!r}",
            "user_message": f"不支持的运算 {op!r}，支持：diff / distance / ratio。",
        }

    res_a = _data_access.resolve_table_name(context, table_a)
    if not res_a.get("success"):
        return {
            "success": False,
            "error": res_a.get("error", "table_a_unavailable"),
            "reason": res_a.get("reason"),
            "user_message": f"表 A 不可用：{res_a.get('user_message')}",
            "available_examples": res_a.get("available_examples"),
        }
    res_b = _data_access.resolve_table_name(context, table_b)
    if not res_b.get("success"):
        return {
            "success": False,
            "error": res_b.get("error", "table_b_unavailable"),
            "reason": res_b.get("reason"),
            "user_message": f"表 B 不可用：{res_b.get('user_message')}",
            "available_examples": res_b.get("available_examples"),
        }

    df_a, df_b = res_a["df"], res_b["df"]
    if df_a is None or df_b is None:
        return {
            "success": False,
            "error": "table_read_failed",
            "reason": "两张表之一读取失败",
            "user_message": "指定的表无法读取内容，请检查表名。",
        }

    al = _align_by_key(df_a, df_b, on)
    if not al.get("success"):
        return al
    joined: pd.DataFrame = al["joined"]
    key = al["key"]

    # 确定参与运算的列：未指定时取两表同名的数值列交集（确定性，不猜语义）。
    common = [
        c for c in df_a.columns
        if c in df_b.columns and pd.api.types.is_numeric_dtype(df_a[c])
        and pd.api.types.is_numeric_dtype(df_b[c]) and c != key
    ]
    cols_a = list(columns_a) if columns_a else common
    cols_b = list(columns_b) if columns_b else cols_a
    if not cols_a:
        return {
            "success": False,
            "error": "no_comparable_columns",
            "reason": "两表没有同名的数值列可用于逐行运算",
            "user_message": (
                "两张表没有可比的数值列（未指定 columns_a/columns_b，且无同名数值列）。"
                f"表 A：{list(map(str, df_a.columns))[:10]}；"
                f"表 B：{list(map(str, df_b.columns))[:10]}。"
            ),
        }
    if len(cols_a) != len(cols_b):
        return {
            "success": False,
            "error": "columns_length_mismatch",
            "reason": f"columns_a 与 columns_b 长度不一致（{len(cols_a)} vs {len(cols_b)}）",
            "user_message": (
                "columns_a 与 columns_b 必须等长（按顺序一一对应）。"
            ),
        }
    missing = [c for c in cols_a if c not in df_a.columns] + [
        c for c in cols_b if c not in df_b.columns
    ]
    if missing:
        return {
            "success": False,
            "error": "column_not_found",
            "reason": f"列不存在：{missing}",
            "user_message": f"以下列不存在：{missing}。请检查列名。",
        }
    if op_norm == "distance" and not (2 <= len(cols_a) <= 3):
        return {
            "success": False,
            "error": "distance_needs_2_or_3_columns",
            "reason": f"distance 需要 2–3 列，当前 {len(cols_a)} 列",
            "user_message": (
                f"欧氏距离需要 2–3 个分量列，当前给了 {len(cols_a)} 列。"
            ),
        }

    # 取值（inner join 后两侧列都在；suffixes 只对**同名**列生效，
    # 故同名时列名变为 a_suffix/b_suffix——这里用位置取更稳妥）。
    va_cols = [f"{c}_a" if f"{c}_a" in joined.columns else c for c in cols_a]
    vb_cols = [f"{c}_b" if f"{c}_b" in joined.columns else c for c in cols_b]
    ok = all(c in joined.columns for c in va_cols + vb_cols)
    if not ok:
        return {
            "success": False,
            "error": "join_column_resolution_failed",
            "reason": f"对齐后列名解析失败：{va_cols} / {vb_cols}",
            "user_message": "对齐后无法定位参与运算的列，请检查列名是否在两表中重名。",
        }
    mat_a = joined[va_cols].to_numpy(dtype=float)
    mat_b = joined[vb_cols].to_numpy(dtype=float)

    result: dict[str, Any] = {}
    if op_norm == "diff":
        for i, c in enumerate(cols_a):
            result[f"{c}（A−B）"] = _stats_of(mat_a[:, i] - mat_b[:, i])
    elif op_norm == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            r = mat_a / mat_b
        for i, c in enumerate(cols_a):
            result[f"{c}（A/B）"] = _stats_of(r[:, i])
    else:  # distance
        d = np.sqrt(np.sum((mat_a - mat_b) ** 2, axis=1))
        result["欧氏距离"] = _stats_of(d)

    # 样例：对齐后的原值与结果，供人工核对（不返回全量，防灌爆上下文）。
    samples: list[dict[str, Any]] = []
    idx = np.arange(min(_MAX_SAMPLE_ROWS, len(joined)))
    for i in idx:
        row: dict[str, Any] = {key: joined[key].iloc[i]}
        for j, c in enumerate(cols_a):
            row[f"{c}_A"] = round(float(mat_a[i, j]), 6)
            row[f"{c}_B"] = round(float(mat_b[i, j]), 6)
        if op_norm == "diff":
            row["结果"] = round(float(mat_a[i, 0] - mat_b[i, 0]), 6)
        elif op_norm == "distance":
            row["结果"] = round(
                float(np.sqrt(np.sum((mat_a[i] - mat_b[i]) ** 2))), 6)
        samples.append(row)

    # 未匹配行如实报出（不静默丢弃）。
    #
    # 注意：按"组内序号"配对时，**匹配数可能大于某一侧的键唯一值数**（如两侧各
    # 28270 行、键各 14135 个唯一值，匹配 28270 行）——此时 `a_only` 不能用
    # "n_a - n_matched" 算（会得负数）。改为按"未匹配的键值数"口径统计，
    # 语义更清晰且绝不会为负。
    keys_a = set(df_a[key].unique())
    keys_b = set(df_b[key].unique())
    unmatched = {
        "n_a": al["n_a"], "n_b": al["n_b"], "n_matched": al["n_matched"],
        "keys_a": len(keys_a), "keys_b": len(keys_b),
        "keys_only_in_a": len(keys_a - keys_b),
        "keys_only_in_b": len(keys_b - keys_a),
    }
    if al.get("grouping"):
        unmatched["grouping"] = al["grouping"]

    msg = (
        f"已按 {key} 对齐：表 A {al['n_a']} 行、表 B {al['n_b']} 行，"
        f"对齐后 {al['n_matched']} 行（运算 {op_norm}，参与列 {cols_a}）。"
    )
    if al.get("grouping"):
        msg += f" 对齐键 {key} 有重复值，已按{al['grouping']}配对。"
    if unmatched["keys_only_in_a"] or unmatched["keys_only_in_b"]:
        msg += (
            f" 注意：仅 A 有的键 {unmatched['keys_only_in_a']} 个、"
            f"仅 B 有的键 {unmatched['keys_only_in_b']} 个（未参与统计）。"
        )

    return {
        "success": True,
        "dataset": context.dataset_id,
        "table_a": res_a.get("table_name"),
        "table_b": res_b.get("table_name"),
        "aligned_on": key,
        "n_a": al["n_a"],
        "n_b": al["n_b"],
        "n_matched": al["n_matched"],
        "unmatched": unmatched,
        "columns_used": {"a": cols_a, "b": cols_b},
        "op": op_norm,
        "result": result,
        "samples": samples,
        "note": (
            "统计口径：按对齐键 inner join 后逐行运算；未匹配行已排除并如实计数。"
            "单位为两侧原始单位（工具不做换算，请确认两侧单位一致）。"
        ),
        "user_message": msg,
    }


@tool
def compare_table_columns(
    wrapper: RunContextWrapper[RunContext],
    table_a: str,
    table_b: str,
    columns_a: list[str] | None = None,
    columns_b: list[str] | None = None,
    op: str = "diff",
    on: str | None = None,
) -> dict:
    """按对齐键对齐两张表，对指定列做逐行运算（差异/距离/比值）并给统计与分布。

    适用：回答"实际末端与指令末端相差多少""两条流的某通道是否一致"这类需要
    **跨表逐行运算**的问题（profile_data / compute_stats 只做单表聚合，做不到）。

    Args:
        table_a: 表 A 表名（形如 ``<文件stem>::<节点>``，见 inspect_streams 的
            table_name 字段）。
        table_b: 表 B 表名。
        columns_a: 表 A 参与运算的列（与 columns_b 按顺序对应）；省略时取两表
            同名数值列的交集。``distance`` 运算需 2–3 列。
        columns_b: 表 B 参与运算的列；省略时同 columns_a。
        op: 运算类型：``diff``（逐元素差）/ ``distance``（欧氏距离）/
            ``ratio``（比值）。缺省 diff。
        on: 对齐键列名；缺省自动探测 frame_index / timestamp 等。

    Returns:
        dict，含 success、aligned_on、n_a/n_b/n_matched、unmatched（未匹配行数）、
        result（统计与 decile 分布）、samples（前 5 条供核对）、user_message。
        两表无法对齐 / 无共同列时返回结构化错误（不做位置对齐，不静默丢行）。
    """
    return compare_table_columns_impl(
        wrapper.context, table_a, table_b,
        columns_a=columns_a, columns_b=columns_b, op=op, on=on,
    )
