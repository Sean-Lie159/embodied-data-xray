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


def read_jsonl_rows(
    path: str,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> list[dict]:
    """逐行读取 JSONL 文件的行记录（每行一个 JSON 对象），最多 limit 行。

    **JSONL 与 JSON 严格区分**：本函数逐行解析（等价于 `pd.read_json(lines=True)`
    的语义），绝不把整个文件当作单个 JSON 值解析。空行与非法行跳过而不中断，
    用于嗅探阶段廉价地取得列结构与 dtype（无需读全量）。

    Args:
        path: 文件路径。
        limit: 最多读取的有效行数；None 表示读全部。
        encoding: 文本编码。

    Returns:
        行记录字典列表；无有效行时返回 []。
    """
    import json as _json

    rows: list[dict] = []
    try:
        with Path(path).open("r", encoding=encoding, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except ValueError:
                    continue  # 单行非法：跳过该行，不中断整体读取
                if isinstance(obj, dict):
                    rows.append(obj)
                if limit is not None and len(rows) >= limit:
                    break
    except Exception:  # noqa: BLE001
        return rows
    return rows


def read_nested_time_column(
    path: str, fmt: str, nested_path: str
) -> "pd.Series | None":
    """读取 JSONL/JSON 内嵌时间字段（点分路径，如 data.header.timestamp_us）。

    仅 jsonl / json 支持（csv/parquet 无嵌套概念，返回 None）。逐行提取、
    缺失行跳过；数值保留 int 原值（ns epoch 超 float64 精确范围）。

    Args:
        path: 文件路径。
        fmt: 格式（jsonl / json）。
        nested_path: 点分嵌套路径。

    Returns:
        时间戳 Series（name 为 nested_path）；不足 2 个有效值或格式不支持
        返回 None。
    """
    import json as _json

    if (fmt or "").lower() not in ("jsonl", "json"):
        return None
    parts = nested_path.split(".")

    def _pluck(row: dict) -> "int | float | None":
        node: Any = row
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        if isinstance(node, bool) or not isinstance(node, (int, float)):
            return None
        return node

    try:
        if (fmt or "").lower() == "jsonl":
            rows = read_jsonl_rows(path, limit=None, encoding="utf-8")
        else:
            from app.tools.load_dataset import _detect_encoding

            obj = _json.loads(
                Path(path).read_text(
                    encoding=_detect_encoding(Path(path).read_bytes())
                )
            )
            rows = _json_row_list(obj) or []
        vals = [v for r in rows if (v := _pluck(r)) is not None]
    except Exception:  # noqa: BLE001
        return None
    if len(vals) < 2:
        return None
    return pd.Series(vals, name=nested_path)


def read_stream_full(path: str, fmt: str) -> pd.DataFrame | None:
    """按需读取流文件的全表。

    JSON 顶层 dict 时按行列表键（frames/data）展开为 DataFrame，避免把标量键
    （如 fps）当数据列。JSONL 按行解析（lines=True），与 JSON 严格区分。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json/jsonl）。

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
        if fmt == "jsonl":
            # JSONL：每行一个 JSON 对象 → 必须 lines=True（与 .json 严格区分）。
            encoding = _detect_encoding(Path(path).read_bytes())
            return pd.read_json(path, lines=True, encoding=encoding)
    except Exception:  # noqa: BLE001
        return None
    return None


def read_table_nrows(path: str, fmt: str) -> int | None:
    """只读表格行数（不读全量数据）。

    统一行数读数入口：inspect_streams / check_temporal_sync / load_dataset 主表评分
    都经此函数获取行数，避免各自实现导致同一文件行数读数不一致。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json/jsonl）。

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
        if fmt == "jsonl":
            # JSONL：非空行即一行记录（与 lines=True 读取语义一致）。
            encoding = _detect_encoding(Path(path).read_bytes())
            with Path(path).open("r", encoding=encoding, errors="replace") as f:
                return sum(1 for line in f if line.strip())
        return None
    except Exception:  # noqa: BLE001
        return None


def expand_envelope(
    df: "pd.DataFrame",
    max_depth: int = 6,
    max_cols: int = 64,
) -> tuple["pd.DataFrame", "str | None"]:
    """把信封型 DataFrame 的 object 列（dict/list）展开为扁平列（只读视图）。

    规则：
    - dict 递归展开为 ``data.orientation.x`` 等点分列名（键并集取自前 50 行样本）；
    - 数值 list 展开为 ``col.0..col.N``（N 为模态长度，上限 32；短行补 None）；
    - list_of_dict 按下标展开为 ``fingers.0.angles.4``（与嵌套发现的路径一致）；
    - 达到 max_cols 即停止并在 note 标注"部分展开"；
    - 返回**新 DataFrame**（原 object 列被展开列取代），不修改入参——主表
      ``context.df`` 语义不受影响。

    Args:
        df: 输入 DataFrame（含 object 信封列）。
        max_depth: 最大展开深度。
        max_cols: 展开列数上限。

    Returns:
        (展开后的 DataFrame, 展开说明 note)；无可展开列时返回 (原 df 副本, None)。
    """
    new_cols: dict[str, list] = {}
    expanded_origins: set[str] = set()
    holder = {"truncated": False}

    def _room() -> bool:
        if len(new_cols) < max_cols:
            return True
        holder["truncated"] = True
        return False

    def _emit(sub: "pd.Series", name: str, depth: int) -> None:
        sample = None
        for v in sub:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            sample = v
            break
        if sample is None:
            return
        if depth < max_depth and isinstance(sample, dict):
            _expand_dict(sub, name, depth + 1)
        elif isinstance(sample, list):
            _expand_list(sub, name, depth + 1)
        else:
            if name in new_cols:
                return
            if _room():
                new_cols[name] = list(sub)

    def _expand_dict(series: "pd.Series", prefix: str, depth: int) -> None:
        keys: list[str] = []
        seen: set[str] = set()
        for v in series.head(50):
            if isinstance(v, dict):
                for k in v:
                    kl = str(k)
                    if kl not in seen:
                        seen.add(kl)
                        keys.append(kl)
        for k in keys:
            if not _room():
                return
            vals = []
            for v in series:
                vals.append(v.get(k) if isinstance(v, dict) else None)
            expanded_origins.add(series.name)
            _emit(pd.Series(vals, index=series.index), f"{prefix}.{k}", depth)

    def _expand_list(series: "pd.Series", prefix: str, depth: int) -> None:
        first = next((v for v in series if isinstance(v, list)), None)
        if first is None:
            return
        if first and isinstance(first[0], dict):
            length = min(len(first), 32)
            if len(first) > 32:
                holder["truncated"] = True
            for i in range(length):
                if not _room():
                    return
                vals = [
                    v[i] if isinstance(v, list) and len(v) > i else None
                    for v in series
                ]
                expanded_origins.add(series.name)
                _emit(pd.Series(vals, index=series.index), f"{prefix}.{i}", depth)
        else:
            length = min(len(first), 32)
            if len(first) > 32:
                holder["truncated"] = True
            for i in range(length):
                if not _room():
                    return
                vals = [
                    v[i] if isinstance(v, list) and len(v) > i else None
                    for v in series
                ]
                expanded_origins.add(series.name)
                if f"{prefix}.{i}" not in new_cols:
                    new_cols[f"{prefix}.{i}"] = vals

    for col in list(df.columns):
        series = df[col]
        sample = None
        for v in series:
            if isinstance(v, (dict, list)):
                sample = v
                break
        if sample is None:
            continue
        expanded_origins.add(col)
        if isinstance(sample, dict):
            _expand_dict(series, str(col), 1)
        else:
            _expand_list(series, str(col), 1)

    keep = [c for c in df.columns if c not in expanded_origins]
    out = df[keep].copy()
    for name, vals in new_cols.items():
        out[name] = vals
    note = (
        "部分展开：达到最大列数上限（max_cols={max_cols}），更深的嵌套字段未展开"
        if holder["truncated"] else None
    )
    return out, note


def resolve_table_name(
    context: RunContext, table: str | None, expand: bool = False
) -> dict[str, Any]:
    """解析表名对应的 DataFrame 与来源，统一多表入口（惰性读取，不替换主表）。

    缺省（table=None）→ 主表（context.df）；显式给表名 → 按流登记表按文件名查找，
    找到则按需读全表，找不到返回结构化错误（不抛异常）。返回 dict 统一含
    ``df``（DataFrame 或 None）、``table_name``（实际表名，用于结果标注）、
    ``dataset``（归属数据集）、``source``（main / stream_lazy / error）。

    Args:
        context: 运行时上下文。
        table: 可选，目标表名（文件名，如 "accel.csv"）。缺省=主表。
        expand: 可选，展开信封型 object 列（dict/list → 点分扁平列）。展开
        生成**新 DataFrame**，不修改 context.df 主表；结果含 expanded=True 与
        expand_note（部分展开说明）。

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
        df = context.df
        note = None
        if expand and df is not None:
            df, note = expand_envelope(df)
        result = {
            "success": df is not None,
            "df": df,
            "table_name": context.meta.get("main_table", {}).get("name"),
            "dataset": dataset_id,
            "source": "main",
        }
        if expand:
            result["expanded"] = True
            result["expand_note"] = note
        return result

    # 显式表名 → 按流登记表查找（文件名精确匹配，忽略大小写）。
    name_lower = table.strip().lower()
    for s in context.meta.get("streams", []):
        p = s.get("path", "")
        if Path(p).name.lower() == name_lower:
            df = read_stream_full(p, s.get("format", ""))
            if df is not None:
                note = None
                if expand:
                    df, note = expand_envelope(df)
                result = {
                    "success": True,
                    "df": df,
                    "table_name": Path(p).name,
                    "dataset": dataset_id,
                    "source": "stream_lazy",
                }
                if expand:
                    result["expanded"] = True
                    result["expand_note"] = note
                return result
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
