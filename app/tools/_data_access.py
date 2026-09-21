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

# 「表不存在」错误里附带可用清单的条数上限（体积护栏）。
#
# 为什么需要：真实数据集 ``aligned_joints.h5`` 的「每帧一组」布局在早期实现里
# 登记出 84,810 条流（docs/行为测试.md:623-630），虽已合并到数十条，但 MCAP
# 多 topic、超大规模目录仍可能上百。给清单是为了让模型**自我纠正**，不是让它
# 读完整目录——超限时截断但仍如实声明总数。
_MAX_AVAILABLE_TABLES = 60


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


def _read_frame_impl(path: str, fmt: str) -> pd.DataFrame | None:
    """全表读取的**底层实现**（reader 内部调用；避免经注册表递归）。

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
            encoding = _detect_encoding(Path(path).read_bytes())
            return pd.read_json(path, lines=True, encoding=encoding)
        return None
    except Exception:  # noqa: BLE001
        return None


def read_stream_full(path: str, fmt: str) -> pd.DataFrame | None:
    """按需读取流文件的全表（**薄包装**：内部走统一读取注册表）。

    保留本函数是为了不动既有调用方（工具层多处仍以此签名调用）；实际读取
    经 ``_readers.read_stream`` 单一入口——格式分派、复合路径（``"<file>::<node>"``）
    解析、h5 节点 / mcap topic 分派全部收敛在那里。

    Args:
        path: 文件路径（容器子流形如 "<file>::<node>"）。
        fmt: 格式（csv/parquet/json/jsonl/h5/mcap）。

    Returns:
        DataFrame；读取失败返回 None。
    """
    from app.tools._readers import ReadRequest, read_stream

    req = ReadRequest(path_spec=path, want="frame", fmt=fmt)
    result = read_stream(req)
    return result.frame if result.ok else None


def _read_nrows_impl(path: str, fmt: str) -> int | None:
    """行数读取的**底层实现**（reader 内部调用；避免经注册表递归）。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json/jsonl）。

    Returns:
        行数；读取失败返回 None。
    """
    import json as _json

    from app.tools.load_dataset import _detect_encoding

    try:
        if fmt == "csv":
            encoding = _detect_encoding(Path(path).read_bytes())
            return int(pd.read_csv(path, encoding=encoding, usecols=[0],
                                   engine="python").shape[0])
        if fmt == "parquet":
            return int(pd.read_parquet(path, columns=None).shape[0])
        if fmt == "json":
            encoding = _detect_encoding(Path(path).read_bytes())
            obj = _json.loads(Path(path).read_text(encoding=encoding))
            rows = _json_row_list(obj)
            return len(rows) if rows is not None else 0
        if fmt == "jsonl":
            encoding = _detect_encoding(Path(path).read_bytes())
            with Path(path).open("r", encoding=encoding, errors="replace") as f:
                return sum(1 for line in f if line.strip())
        return None
    except Exception:  # noqa: BLE001
        return None


def read_table_nrows(path: str, fmt: str) -> int | None:
    """只读表格行数（**薄包装**：内部走统一读取注册表）。

    统一行数读数入口：inspect_streams / check_temporal_sync / load_dataset
    主表评分都经此获取行数，避免各自实现导致同一文件行数读数不一致。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json/jsonl）。

    Returns:
        行数（不含表头）；读取失败返回 None。
    """
    from app.tools._readers import ReadRequest, read_stream

    result = read_stream(ReadRequest(path_spec=path, want="nrows", fmt=fmt))
    return result.nrows if result.ok else None


def expand_envelope(
    df: "pd.DataFrame",
    max_depth: int = 6,
    max_cols: int | None = 64,
    focus: list[str] | None = None,
) -> tuple["pd.DataFrame", "str | None"]:
    """把信封型 DataFrame 的 object 列（dict/list）展开为扁平列（只读视图）。

    规则：
    - dict 递归展开为 ``data.orientation.x`` 等点分列名（键并集取自前 50 行样本）；
    - 数值 list 展开为 ``col.0..col.N``（N 为模态长度，上限 32；短行补 None）；
    - list_of_dict 按下标展开为 ``fingers.0.angles.4``（与嵌套发现的路径一致）；
    - 达到 max_cols 即停止并在 note 标注"部分展开"；``max_cols=None`` 表示不限；
    - 返回**新 DataFrame**（原 object 列被展开列取代），不修改入参——主表
      ``context.df`` 语义不受影响。

    Args:
        df: 输入 DataFrame（含 object 信封列）。
        max_depth: 最大展开深度。
        max_cols: 展开列数上限；None 表示不限制（用于结构已知的小文件，如
            标定文件——其字段数天然有限，限死会丢掉关键的内/外参）。
        focus: 可选，**优先展开**的顶层列名列表（各占均分配额）。
            用于"只关心某几个字段"的场景——否则靠前的无关字段会独占 max_cols
            配额，把目标字段挤掉（真实事故：标定文件的 sensors_list 展开 40+
            列后，intrinsic/extrinsic 完全轮不到）。

    Returns:
        (展开后的 DataFrame, 展开说明 note)；无可展开列时返回 (原 df 副本, None)。
    """
    new_cols: dict[str, list] = {}
    expanded_origins: set[str] = set()
    # None 表示不限：用一个大数占位（避免各处判空）。
    limit_default = max_cols if max_cols is not None else 1_000_000
    holder = {"truncated": False, "limit": limit_default}

    def _room() -> bool:
        if len(new_cols) < holder["limit"]:
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

    # 展开配额分配：**focus 指定的字段各占均配额**，其余共享剩余。
    #
    # 为什么需要（2026-09-18 真实事故）：展开受 max_cols 限制，先展开的字段会
    # **独占**配额。标定文件 calibration.json 只有 3 列
    # （sensors_list / intrinsic / extrinsic），sensors_list 一个就展开出 40+
    # 列，把 intrinsic（内参）与 extrinsic（外参）完全挤掉——用户要核对
    # FOV/内外参时工具"什么都读不出来"，而它其实有能力展开。
    #
    # 只改"排序"不够（实测：focus=[intrinsic, extrinsic] 时 intrinsic 仍独占
    # 64 列配额，extrinsic 一个都没展开）。故按字段**分轮设置上限**：每轮给
    # 一个字段分配配额，逐字段推进，保证每个 focus 字段都能拿到份额。

    def _expand_one(col: Any) -> None:
        series = df[col]
        sample = None
        for v in series:
            if isinstance(v, (dict, list)):
                sample = v
                break
        if sample is None:
            return
        expanded_origins.add(col)
        if isinstance(sample, dict):
            _expand_dict(series, str(col), 1)
        else:
            _expand_list(series, str(col), 1)

    wanted = [str(f) for f in (focus or []) if str(f) in [str(c) for c in df.columns]]
    if wanted and max_cols is not None and max_cols > 0:
        # 每个 focus 字段的配额：均分（至少留 4 列给后续字段，避免单个字段吃光）。
        share = max(4, max_cols // (len(wanted) + 1))
        covered: set[str] = set()
        for name in wanted:
            col = next(c for c in df.columns if str(c) == name)
            holder["limit"] = min(max_cols, len(new_cols) + share)
            _expand_one(col)
            covered.add(name)
        # 剩余配额留给其余字段。
        holder["limit"] = max_cols
        for col in df.columns:
            if str(col) not in covered:
                _expand_one(col)
    else:
        # 无 max_cols 限制（None）或未指定 focus：全部字段按序展开。
        for col in df.columns:
            _expand_one(col)

    keep = [c for c in df.columns if c not in expanded_origins]
    # **一次性拼接**而非逐列赋值：展开标定文件这类"字段数达数百"的对象时，
    # `out[name] = vals` 逐列插入会触发 pandas 的
    # "DataFrame is highly fragmented" 性能告警（实测 749 列），并让后续操作
    # 变慢。pd.concat 一次性建表既消除告警，也更快。
    parts = [df[keep].reset_index(drop=True)]
    if new_cols:
        parts.append(pd.DataFrame(new_cols).reset_index(drop=True))
    out = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0].copy()
    note = (
        f"部分展开：达到最大列数上限（max_cols={max_cols}），更深的嵌套字段未展开"
        if holder["truncated"] else None
    )
    return out, note


def output_prefix(context: RunContext) -> str:
    """输出文件名前缀：会话标识（多会话隔离）；无则空串。

    多会话时两个会话可能分析**同一数据集**（dataset_id 相同），仅靠秒级
    时间戳区分会互相覆盖——故文件名前缀带上会话标识。

    Args:
        context: 运行时上下文。

    Returns:
        前缀（形如 "s-1a2b_"）；无会话标识时返回空串（文件名与单会话一致）。
    """
    tag = getattr(context, "session_tag", "") or ""
    return f"{tag}_" if tag else ""


def _split(path_spec: str) -> tuple[str, str | None]:
    """委托 _readers.split_path_spec（避免本模块重复实现解析逻辑）。"""
    from app.tools._readers import split_path_spec

    return split_path_spec(path_spec)


def _read_h5_node_stream(path_spec: str) -> pd.DataFrame | None:
    """读取 h5 节点流（path_spec 形如 "<file>::<node path>"）。"""
    file_part, _, node = path_spec.partition("::")
    if not node:
        return None
    from app.tools.load_dataset import read_hdf5_node

    return read_hdf5_node(file_part, node)


def read_h5_node_field(path_spec: str, field: str) -> "pd.Series | None":
    """读取 h5 节点流指定字段（compound 的 timestamp 等），返回 Series。"""
    file_part, _, node = path_spec.partition("::")
    if not node:
        return None
    from app.tools.load_dataset import read_hdf5_node

    df = read_hdf5_node(file_part, node)
    if df is None or field not in df.columns:
        return None
    series = df[field]
    series.name = field
    return series


def expand_with_focus(
    df: "pd.DataFrame", focus_fields: list[str] | None
) -> tuple["pd.DataFrame", "str | None"]:
    """按 ``focus_fields`` 语义展开信封型列（统一入口，避免各处重复解析）。

    语义（2026-09-20 新增）：
    - ``None`` / 空列表：现有行为（按列序展开至 max_cols=64）；
    - ``["*"]``：展开**全部**字段且**不限列数**——用于字段数有限的配置文件
      （如标定文件），此时"读全"比"读一部分"更有价值；
    - 具体字段名列表：为这些字段保留配额优先展开。

    Args:
        df: 含 object 信封列的 DataFrame。
        focus_fields: 调用方给定的聚焦字段。

    Returns:
        (展开后的 DataFrame, note)。
    """
    if not focus_fields:
        return expand_envelope(df)
    if any(str(f).strip() == "*" for f in focus_fields):
        return expand_envelope(df, max_cols=None)
    names = [str(f) for f in focus_fields if str(f).strip() and str(f).strip() != "*"]
    if not names:
        return expand_envelope(df)
    return expand_envelope(df, focus=names)


def get_main_table_meta(context: RunContext) -> dict[str, Any]:
    """取 ``meta["main_table"]``，并保证返回 dict（可能是 None/缺失）。

    Args:
        context: 运行时上下文。

    Returns:
        main_table 元信息字典（缺失时返回空 dict）。
    """
    meta = context.meta.get("main_table")
    return meta if isinstance(meta, dict) else {}


def main_table_candidates(context: RunContext) -> list[dict[str, Any]]:
    """取主表候选清单，**统一两条加载路径的位置差异**（唯一取用点）。

    load_dataset 的两条加载路径写入的 meta 形态不同（实测）：

    - 目录型（``load_dataset.py:2369-2371``）：候选在
      ``main_table["selection"]["candidates"]``，字段为 ``name / nrows / ncols``；
    - 单文件型（``load_dataset.py:2613-2626``）：候选在顶层
      ``main_table["candidates"]``，字段为 ``table_name / rows / cols``。

    Args:
        context: 运行时上下文。

    Returns:
        候选字典列表（合并两处，原顺序保持）；无候选时为空列表。
    """
    meta = get_main_table_meta(context)
    out: list[dict[str, Any]] = list(meta.get("candidates") or [])
    selection = meta.get("selection")
    if isinstance(selection, dict):
        out.extend(selection.get("candidates") or [])
    return out


def resolve_default_table_name(context: RunContext) -> str | None:
    """解析「当前缺省表」的**可用表名**（唯一权威实现）。

    **为什么需要这个函数**（2026-09-21 实测发现，这是本模块最易踩的坑）：

    ``meta["main_table"]["name"]`` **不是可靠的表名**，两条加载路径语义不同：

    - 目录型：``name`` = ``"state.csv"`` —— 可直接用作 ``table`` 参数；
    - 单文件型：``name`` = ``"state/joint"``（**裸节点路径**）—— 照抄去调用
      ``resolve_table_name`` **必然失败**，可用名是同目录 candidates 里的
      ``table_name`` = ``"joints::state/joint"``。

    此前直接读 ``name``，导致 HDF5/MCAP 数据集在 ``table=None`` 时返回的
    ``table_name`` 是一个**不可回用的假表名**，并随各工具的结果标注扩散出去
    ——这正是 2026-09-20「清单与调用口径自相矛盾」事故的同型缺陷。

    解析顺序：目录型的 ``selection.selected`` 优先（它本身就是可用名）；
    否则用 ``name`` 去 candidates 里按「完全相等」或「以 ``::<name>`` 结尾」
    反查，取完整可用名；反查不到才退回原 ``name``。

    Args:
        context: 运行时上下文。

    Returns:
        可用表名；无主表（纯媒体数据集）时返回 None。
    """
    meta = get_main_table_meta(context)
    selection = meta.get("selection")
    if isinstance(selection, dict) and selection.get("selected"):
        # 目录型：selected 即文件名，本身就是可用名。
        return str(selection["selected"])

    raw_name = meta.get("name")
    if not raw_name:
        return None
    raw_str = str(raw_name)
    for c in main_table_candidates(context):
        cand_name = c.get("table_name") or c.get("name")
        if not cand_name:
            continue
        cand_str = str(cand_name)
        if cand_str == raw_str or cand_str.endswith("::" + raw_str):
            return cand_str
    # 反查不到：如实返回原名（下游按名匹配会失败，但不伪造可用名）。
    return raw_str


def resolve_table_name(
    context: RunContext,
    table: str | None,
    expand: bool = False,
    focus_fields: list[str] | None = None,
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
        success=False 且 error="table_not_found"，并附 ``available_tables``
        （全部可用表名清单，供模型自我纠正）与 ``default_table``（当前缺省表）。
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
            df, note = expand_with_focus(df, focus_fields)
        result = {
            "success": df is not None,
            "df": df,
            # **必须用 resolve_default_table_name 而非直接读 name**（2026-09-21）：
            # 单文件（h5/mcap）加载路径下 meta["main_table"]["name"] 是裸节点路径
            # （如 "state/joint"），直接透出会得到一个**照抄必失败**的表名，
            # 并随各工具的结果标注扩散——同型于 2026-09-20 事故。
            "table_name": resolve_default_table_name(context),
            "dataset": dataset_id,
            "source": "main",
        }
        if expand:
            result["expanded"] = True
            result["expand_note"] = note
        return result

    # 显式表名 → 按流登记表查找（文件名精确匹配，忽略大小写）。
    # h5 节点流的表名 = "<文件stem>::<node>"；mcap topic 流形如
    # "<文件stem>::<topic>"（topic 可含斜杠，如 demo::/imu）。
    #
    # **表名容错**（2026-09-20 真实事故）：用户/模型给出的表名写法不一，此前只认
    # 「stem::节点」与「节点名」两种精确形式，导致这些**同样合法**的写法全部失败：
    #   - `aligned_joints.h5::state/end/position`（带扩展名）
    #   - `C:\...\aligned_joints.h5::state/end/position`（完整路径）
    # 而 inspect_streams 的 `source` 恰好是"完整路径::节点"——agent 照清单抄必失败，
    # 于是得出"工具不具备该能力"的错误结论。现统一折算为候选匹配串集合。
    name_lower = table.strip().lower()
    match_candidates = {name_lower}
    file_part, sub_part = _split(name_lower)
    if sub_part:
        stem = Path(file_part).stem.lower()
        name_only = Path(file_part).name.lower()
        stem_noext = Path(Path(file_part).name).stem.lower()
        for f in (stem, name_only, stem_noext):
            match_candidates.add(f"{f}::{sub_part}")
    for s in context.meta.get("streams", []):
        p = s.get("path", "")
        if s.get("format") == "mcap":
            # 复合路径解析统一经 split_path_spec（唯一解析点）。
            file_part, topic = _split(p)
            if not topic:
                continue
            display = f"{Path(file_part).stem}::{topic}".lower()
            if (display in match_candidates or topic.lower() in match_candidates
                    or name_lower == topic.lower()):
                from app.tools.mcap_reader import read_mcap_topic

                read = read_mcap_topic(file_part, topic)
                if read.get("success") and read.get("df") is not None:
                    df = read["df"]
                    note = None
                    if expand:
                        df, note = expand_with_focus(df, focus_fields)
                    result = {
                        "success": True,
                        "df": df,
                        "table_name": f"{Path(file_part).stem}::{topic}",
                        "dataset": context.dataset_id,
                        "source": "mcap_topic",
                    }
                    if expand:
                        result["expanded"] = True
                        result["expand_note"] = note
                    return result
                return {
                    "success": False,
                    "error": "table_read_failed",
                    "reason": f"mcap topic {topic} 读取失败",
                    "df": None,
                    "table_name": table,
                    "dataset": context.dataset_id,
                    "source": "mcap_topic",
                    "user_message": f"已找到 topic {topic}，但读取其消息失败，无法分析。",
                }
        if s.get("format") == "h5":
            file_part, node_name = _split(p)
            if not node_name:
                continue
            display = f"{Path(file_part).stem}::{node_name}".lower()
            if (display in match_candidates
                    or node_name.lower() in match_candidates
                    or name_lower == node_name.lower()):
                df = _read_h5_node_stream(p)
                if df is not None:
                    result = {
                        "success": True,
                        "df": df,
                        "table_name": f"{Path(file_part).stem}::{node_name}",
                        "dataset": context.dataset_id,
                        "source": "h5_node",
                    }
                    if expand:
                        df2, note2 = expand_with_focus(df, focus_fields)
                        result["df"] = df2
                        result["expand_note"] = note2
                    return result
                return {
                    "success": False,
                    "error": "table_read_failed",
                    "reason": f"h5 节点 {node_name} 读取失败",
                    "df": None,
                    "table_name": table,
                    "dataset": context.dataset_id,
                    "source": "h5_node",
                }
        if Path(p).name.lower() in match_candidates:
            df = read_stream_full(p, s.get("format", ""))
            if df is not None:
                note = None
                if expand:
                    df, note = expand_with_focus(df, focus_fields)
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

    # 错误提示必须**给出可直接使用的示例**（2026-09-20 真实事故）：此前只说
    # "可用表见流登记表/inspect_streams 的表格流清单"，而 agent 照 inspect_streams
    # 的 `source`（完整路径）抄表名**必然失败**——既没示例可对照，也没说明格式
    # 要求，于是"能力可达"被误判为"工具不支持"。
    #
    # 【阶段 2，2026-09-21】此前只给 **3 个示例**，模型无从判断"是真的没有这张表，
    # 还是我写法不对"，只能放弃或凭猜重试。现改为返回**完整可用清单**
    # （available_tables，受 _MAX_AVAILABLE_TABLES 上限保护）与 default_table，
    # 使模型能据清单自我纠正。纪律不变：**仍不做模糊/子串猜表**（防误命中，
    # 见 docs/表访问与跨表改造设计.md:83-84）——给清单，但不替模型做匹配。
    usable: list[str] = []
    for s in context.meta.get("streams", []):
        # 视频不是表，不列入候选（口径与 inspect_streams 的 n_table_streams 一致）。
        if s.get("kind") == "video":
            continue
        p = s.get("path", "")
        if not p:
            continue
        fmt = s.get("format")
        if fmt in ("h5", "mcap"):
            fp, sub_name = _split(p)
            if sub_name:
                usable.append(f"{Path(fp).stem}::{sub_name}")
        else:
            usable.append(Path(p).name)
    # 去重且保持原顺序（同一 path 可能有重复登记）。
    seen: set[str] = set()
    usable_unique: list[str] = []
    for n in usable:
        if n not in seen:
            seen.add(n)
            usable_unique.append(n)

    default_table = resolve_default_table_name(context)
    shown = usable_unique[:_MAX_AVAILABLE_TABLES]
    truncated_tables = len(usable_unique) > _MAX_AVAILABLE_TABLES
    table_hint = (
        f"当前可用表共 {len(usable_unique)} 张"
        + (f"（此处列出前 {_MAX_AVAILABLE_TABLES} 张）：{shown}"
           if truncated_tables else f"：{shown}")
        + "。"
        if usable_unique
        else "当前数据集没有可作为分析对象的表。"
    )
    hint = (
        f"表名形如「<文件stem>::<节点>」（h5/mcap 子流，**不含扩展名**）"
        f"或「<文件名>」（独立文件）。{table_hint}"
        + (
            f"若你想分析的是主表，不传 table 参数即可（当前缺省表是 {default_table}）。"
            if default_table else ""
        )
        + "也可用 list_tables 工具查看完整清单与各表规模。"
    )
    result: dict[str, Any] = {
        "success": False,
        "error": "table_not_found",
        "reason": f"数据集中不存在表 {table}",
        "df": None,
        "table_name": table,
        "dataset": dataset_id,
        "source": "error",
        "available_examples": shown[:3],
        # 【阶段 2】完整可用清单 + 缺省表：让模型能自行纠正，而不是直接放弃。
        "available_tables": shown,
        "available_tables_truncated": truncated_tables,
        "default_table": default_table,
        "user_message": (
            f"当前数据集 {dataset_id} 中不存在表 {table}。{hint}"
        ),
    }
    return result


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
