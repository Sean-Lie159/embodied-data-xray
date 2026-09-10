"""统一读取注册表：所有格式的读取收敛为单一入口。

**为什么需要本模块**（问题陈述，详见 research/research_plan_unified_readers.md）：

本项目每接入一种新数据形态（MCAP 信封 JSONL → h5py 原生 h5 → 目录内嵌 h5
→ compound 子数组 → ms epoch → MCAP 容器），都要在**多处手动插格式分支**，
且总会漏掉几处——实测盘点：15 个按格式分派的读取函数散落在 7 个文件、
两套 fmt 命名体系（带点扩展名 vs 无点格式名）并存、`"<file>::<sub>"` 复合
路径特判 4 处、同一能力（读全表/读时间戳/读列名）各有 2~3 份重复实现。

这是"工具层总是暴露边界"的**机制性根因**：每个工具各自实现"能不能读"，
改一处漏五处（真实事故：sanity wrapper NameError、目录内 h5 完全不可见）。

本模块把"能不能读"收敛为**一处**：

    read_stream(ReadRequest) -> ReadResult      # 单一入口
    register_reader(fmt, reader)                # 新增格式 = 注册一个 reader

新增格式（如 rosbag2）只需实现 Reader 协议并注册，**不改任何工具**。

设计要点：
- **复合路径统一解析**：``"<file>::<sub>"`` 的拆分与分派收在本模块内
  （h5 节点与 mcap topic 同构——都是"单容器多子流"，故共用一套）。
- **fmt 内部统一无点格式名**（"csv"），扩展名转换只在入口做一次。
- **适配器过渡**：当前各 reader 包装既有实现（包装而非重写），保证零回归；
  待稳定后可内联。旧函数保留为薄包装，调用方无需改。
- **错误语义**：``ReadResult.error`` 用结构化码（unsupported_format /
  read_failed / sub_not_found / missing_dependency），``reason`` 供工具转达；
  缺依赖与"文件损坏"严格区分（沿用项目既有纪律）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 请求 / 结果契约
# ---------------------------------------------------------------------------

Want = Literal["frame", "columns", "nrows", "timestamp", "sample", "candidates"]


@dataclass(frozen=True)
class ReadRequest:
    """一次读取请求（工具层构造，注册表消费）。

    Attributes:
        path_spec: ``"<file>"`` 或 ``"<file>::<sub>"``（h5 节点 / mcap topic）。
        want: 要读取的内容——frame（全表）/ columns（列名）/ nrows（行数）/
            timestamp（时间戳列）/ sample（前 N 行）/ candidates（子流清单）。
        fmt: 格式名（无点，如 "csv"）；缺省由扩展名推断。
        column: want="timestamp" 时的列名或嵌套路径（含 "." 视为嵌套）。
        limit: want="sample"/"frame" 的行数上限；None 为全量。
        expand: 是否展开信封型 object 列（dict/list → 点分扁平列）。
    """

    path_spec: str
    want: Want
    fmt: str | None = None
    column: str | None = None
    limit: int | None = None
    expand: bool = False


@dataclass(frozen=True)
class ReadResult:
    """统一读取结果（契约稳定：工具按需取字段，注册表内部可演进）。"""

    ok: bool
    frame: pd.DataFrame | None = None
    columns: list[str] = field(default_factory=list)
    nrows: int | None = None
    timestamp: np.ndarray | None = None
    timestamp_column: str | None = None
    candidates: list[str] = field(default_factory=list)
    fmt: str = ""
    sub: str | None = None
    expand_note: str | None = None
    error: str | None = None
    reason: str | None = None

    @staticmethod
    def fail(error: str, reason: str, *, fmt: str = "", sub: str | None = None) -> "ReadResult":
        """构造失败结果（统一入口，保证 error/reason 语义一致）。"""
        return ReadResult(ok=False, fmt=fmt, sub=sub, error=error, reason=reason)


class Reader(Protocol):
    """格式读取器协议：新增格式实现它并 register_reader 即可接入全部工具。

    可选属性 ``extensions``：该格式对应的扩展名列表（如 [".rosbag"]）。
    注册时自动并入格式推断表——**新格式无需修改本模块的映射表**（这是
    "新格式=注册一次"抽象成立的关键；否则仍要改一处）。
    """

    fmt: str
    extensions: list[str]

    def frame(self, path: str, *, sub: str | None, limit: int | None) -> pd.DataFrame | None:
        """读全表（或前 limit 行）；失败返回 None。"""
        ...

    def columns(self, path: str, *, sub: str | None) -> list[str] | None:
        """只读列名；不支持/失败返回 None。"""
        ...

    def nrows(self, path: str, *, sub: str | None) -> int | None:
        """只读行数；不支持/失败返回 None。"""
        ...

    def sub_streams(self, path: str) -> list[str]:
        """列出子流（仅容器型格式 h5/mcap 有）；非容器返回 []。"""
        ...


# ---------------------------------------------------------------------------
# 格式推断
# ---------------------------------------------------------------------------

# 扩展名 → 内部格式名（无点）。新增格式在此登记扩展名映射。
_EXT_TO_FMT: dict[str, str] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
    ".h5": "h5",
    ".hdf5": "h5",
    ".mcap": "mcap",
}

# 复合路径分隔符（h5 节点 / mcap topic 共用）。
_SUB_SEP = "::"


def split_path_spec(path_spec: str) -> tuple[str, str | None]:
    """拆分 ``"<file>::<sub>"`` 为 (文件路径, 子流名 或 None)。

    这是**唯一**的复合路径解析点——此前分散在 _data_access / inspect_streams /
    check_temporal_sync 共 4 处（h5 与 mcap 各一套，逻辑重复）。

    Args:
        path_spec: 流登记表中的 path 字段。

    Returns:
        (文件路径, 子流名)。无 ``::`` 时子流名为 None。
    """
    file_part, sep, sub = str(path_spec).partition(_SUB_SEP)
    # 空 sub 视为无子流（"a.csv::" 与 "a.csv" 等价）。
    return file_part, (sub if (sep and sub) else None)


def resolve_fmt(path_spec: str, fmt: str | None = None) -> str:
    """确定格式名（内部统一为无点格式名）。

    Args:
        path_spec: 文件路径（可含 ``::sub``）。
        fmt: 显式格式名（可带点）；None 时按扩展名推断。

    Returns:
        无点格式名；无法识别返回 ""。
    """
    if fmt:
        return fmt.lower().lstrip(".").strip()
    file_part, _ = split_path_spec(path_spec)
    return _EXT_TO_FMT.get(Path(file_part).suffix.lower(), "")


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Reader] = {}


def register_reader(reader: Reader) -> None:
    """注册一个格式读取器（覆盖同名注册）。

    若 reader 声明了 ``extensions``，一并并入扩展名推断表——使新格式接入
    只需本函数一次调用（无需改本模块任何映射表）。

    Args:
        reader: 实现 Reader 协议的读取器。
    """
    _REGISTRY[reader.fmt] = reader
    for ext in getattr(reader, "extensions", []) or []:
        _EXT_TO_FMT[ext.lower()] = reader.fmt


def supported_formats() -> list[str]:
    """返回已注册的格式名清单（确定性排序）。"""
    return sorted(_REGISTRY)


def get_reader(fmt: str) -> Reader | None:
    """按格式名取读取器；未注册返回 None。"""
    return _REGISTRY.get(fmt)


# ---------------------------------------------------------------------------
# 单一入口
# ---------------------------------------------------------------------------

def read_stream(req: ReadRequest) -> ReadResult:
    """统一读取入口——所有工具经此读取，不再各自分派格式。

    Args:
        req: 读取请求。

    Returns:
        ReadResult。失败时 ``ok=False`` 且带结构化 ``error`` 与 ``reason``：
        - ``unsupported_format``：格式未注册；
        - ``sub_not_found``：容器子流不存在；
        - ``read_failed``：读取失败（含文件不存在）。

    Note:
        ``want="candidates"`` 时返回子流清单（容器型格式）；非容器格式返回空
        清单（不是错误——单文件格式本就没有子流概念）。
    """
    file_part, sub = split_path_spec(req.path_spec)
    fmt = resolve_fmt(req.path_spec, req.fmt)

    if not fmt:
        return ReadResult.fail(
            "unsupported_format",
            f"无法从路径识别格式：{req.path_spec}",
            fmt="", sub=sub,
        )
    reader = get_reader(fmt)
    if reader is None:
        return ReadResult.fail(
            "unsupported_format",
            f"格式 {fmt} 未注册读取器（已注册：{', '.join(supported_formats())}）",
            fmt=fmt, sub=sub,
        )

    # 容器型：子流清单（无需文件读权限校验之外的处理）。
    if req.want == "candidates":
        return ReadResult(ok=True, fmt=fmt, candidates=reader.sub_streams(file_part))

    if not Path(file_part).exists():
        return ReadResult.fail(
            "read_failed", f"文件不存在：{file_part}", fmt=fmt, sub=sub,
        )

    if req.want == "frame":
        df = reader.frame(file_part, sub=sub, limit=req.limit)
        if df is None:
            return ReadResult.fail(
                "sub_not_found" if sub else "read_failed",
                (f"{fmt} 子流 {sub} 读取失败" if sub else f"{fmt} 文件读取失败：{file_part}"),
                fmt=fmt, sub=sub,
            )
        if sub is None and req.expand:
            df, note = _expand(df)
            return ReadResult(ok=True, frame=df, fmt=fmt, expand_note=note)
        return ReadResult(ok=True, frame=df, fmt=fmt, sub=sub)

    if req.want == "sample":
        limit = req.limit if req.limit is not None else 500
        df = reader.frame(file_part, sub=sub, limit=limit)
        if df is None:
            return ReadResult.fail(
                "read_failed", f"{fmt} 样本读取失败：{req.path_spec}", fmt=fmt, sub=sub,
            )
        if req.expand:
            df, note = _expand(df)
            return ReadResult(ok=True, frame=df, fmt=fmt, expand_note=note)
        return ReadResult(ok=True, frame=df, fmt=fmt, sub=sub)

    if req.want == "columns":
        cols = reader.columns(file_part, sub=sub)
        if cols is None:
            return ReadResult.fail(
                "read_failed", f"{fmt} 列名读取失败：{req.path_spec}", fmt=fmt, sub=sub,
            )
        return ReadResult(ok=True, columns=list(cols), fmt=fmt, sub=sub)

    if req.want == "nrows":
        n = reader.nrows(file_part, sub=sub)
        if n is None:
            return ReadResult.fail(
                "read_failed", f"{fmt} 行数读取失败：{req.path_spec}", fmt=fmt, sub=sub,
            )
        return ReadResult(ok=True, nrows=n, fmt=fmt, sub=sub)

    if req.want == "timestamp":
        series, col_name = reader.timestamp(file_part, sub=sub, column=req.column)  # type: ignore[attr-defined]
        if series is None:
            return ReadResult.fail(
                "read_failed",
                f"{fmt} 时间戳列读取失败：{req.path_spec}"
                + (f"（列 {req.column}）" if req.column else ""),
                fmt=fmt, sub=sub,
            )
        return ReadResult(
            ok=True, timestamp=np.asarray(series, dtype=float),
            timestamp_column=col_name, fmt=fmt, sub=sub,
        )

    return ReadResult.fail("read_failed", f"未知读取意图：{req.want}", fmt=fmt, sub=sub)


def _expand(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """信封展开（委托 _data_access.expand_envelope，保持单一实现）。"""
    from app.tools._data_access import expand_envelope

    out, note = expand_envelope(df)
    return out, note


# ---------------------------------------------------------------------------
# 六个内置 reader（适配器：包装既有实现，保证行为零回归）
# ---------------------------------------------------------------------------

class _CsvReader:
    fmt = "csv"
    extensions = [".csv"]

    def frame(self, path: str, *, sub: str | None, limit: int | None) -> pd.DataFrame | None:
        from app.tools.load_dataset import _detect_encoding, _load_csv

        if limit is None:
            try:
                return _load_csv(path)
            except Exception:  # noqa: BLE001
                return None
        try:
            return pd.read_csv(
                path, encoding=_detect_encoding(Path(path).read_bytes()),
                nrows=limit, engine="python",
            )
        except Exception:  # noqa: BLE001
            return None

    def columns(self, path: str, *, sub: str | None) -> list[str] | None:
        from app.tools.load_dataset import _detect_encoding

        try:
            df = pd.read_csv(
                path, encoding=_detect_encoding(Path(path).read_bytes()),
                nrows=0, engine="python",
            )
            return [str(c) for c in df.columns]
        except Exception:  # noqa: BLE001
            return None

    def nrows(self, path: str, *, sub: str | None) -> int | None:
        from app.tools.load_dataset import _detect_encoding

        try:
            return int(pd.read_csv(
                path, encoding=_detect_encoding(Path(path).read_bytes()),
                usecols=[0], engine="python",
            ).shape[0])
        except Exception:  # noqa: BLE001
            return None

    def sub_streams(self, path: str) -> list[str]:
        return []

    def timestamp(self, path: str, *, sub: str | None, column: str | None):
        from app.tools.inspect_streams import _read_timestamp_only

        series = _read_timestamp_only(path, "csv", column)
        return (series, str(series.name) if series is not None and series.name else None)


class _ParquetReader:
    fmt = "parquet"
    extensions = [".parquet"]

    def frame(self, path: str, *, sub: str | None, limit: int | None) -> pd.DataFrame | None:
        try:
            df = pd.read_parquet(path)
            return df.head(limit) if limit is not None else df
        except Exception:  # noqa: BLE001
            return None

    def columns(self, path: str, *, sub: str | None) -> list[str] | None:
        try:
            return [str(c) for c in pd.read_parquet(path, columns=None).columns[:50]]
        except Exception:  # noqa: BLE001
            return None

    def nrows(self, path: str, *, sub: str | None) -> int | None:
        try:
            return int(pd.read_parquet(path, columns=None).shape[0])
        except Exception:  # noqa: BLE001
            return None

    def sub_streams(self, path: str) -> list[str]:
        return []

    def timestamp(self, path: str, *, sub: str | None, column: str | None):
        from app.tools.inspect_streams import _read_timestamp_only

        series = _read_timestamp_only(path, "parquet", column)
        return (series, str(series.name) if series is not None and series.name else None)


class _JsonReader:
    """JSON（整体一个值）：经统一 reader 展开行列表键（frames/data）。"""

    fmt = "json"
    extensions = [".json"]

    def frame(self, path: str, *, sub: str | None, limit: int | None) -> pd.DataFrame | None:
        from app.tools._data_access import _read_frame_impl

        df = _read_frame_impl(path, "json")
        return df.head(limit) if df is not None and limit is not None else df

    def columns(self, path: str, *, sub: str | None) -> list[str] | None:
        from app.tools.load_dataset import _read_table_columns

        return _read_table_columns(Path(path))

    def nrows(self, path: str, *, sub: str | None) -> int | None:
        from app.tools._data_access import _read_nrows_impl

        return _read_nrows_impl(path, "json")

    def sub_streams(self, path: str) -> list[str]:
        return []

    def timestamp(self, path: str, *, sub: str | None, column: str | None):
        from app.tools.inspect_streams import _read_timestamp_only

        series = _read_timestamp_only(path, "json", column)
        return (series, str(series.name) if series is not None and series.name else None)


class _JsonlReader:
    """JSONL（每行一个 JSON 对象，lines=True）。"""

    fmt = "jsonl"
    extensions = [".jsonl"]

    def frame(self, path: str, *, sub: str | None, limit: int | None) -> pd.DataFrame | None:
        from app.tools._data_access import read_jsonl_rows

        rows = read_jsonl_rows(path, limit=limit)
        return pd.DataFrame(rows) if rows else None

    def columns(self, path: str, *, sub: str | None) -> list[str] | None:
        from app.tools._data_access import read_jsonl_rows

        rows = read_jsonl_rows(path, limit=1)
        return [str(k) for k in rows[0].keys()] if rows else []

    def nrows(self, path: str, *, sub: str | None) -> int | None:
        from app.tools._data_access import _read_nrows_impl

        return _read_nrows_impl(path, "jsonl")

    def sub_streams(self, path: str) -> list[str]:
        return []

    def timestamp(self, path: str, *, sub: str | None, column: str | None):
        from app.tools.inspect_streams import _read_timestamp_only

        series = _read_timestamp_only(path, "jsonl", column)
        return (series, str(series.name) if series is not None and series.name else None)


class _H5Reader:
    """HDF5：单文件多节点（容器型）——path_spec 为 ``"<file>::<node>"``。"""

    fmt = "h5"
    extensions = [".h5", ".hdf5"]

    def frame(self, path: str, *, sub: str | None, limit: int | None) -> pd.DataFrame | None:
        if sub:
            from app.tools.load_dataset import read_hdf5_node

            df = read_hdf5_node(path, sub)
            return df.head(limit) if df is not None and limit is not None else df
        from app.tools.load_dataset import _load_hdf5

        try:
            df = _load_hdf5(path)
        except Exception:  # noqa: BLE001 - 缺依赖等由工具层既有路径转结构化错误
            return None
        return df.head(limit) if limit is not None else df

    def columns(self, path: str, *, sub: str | None) -> list[str] | None:
        df = self.frame(path, sub=sub, limit=1)
        return [str(c) for c in df.columns] if df is not None else None

    def nrows(self, path: str, *, sub: str | None) -> int | None:
        from app.tools.load_dataset import read_hdf5_node

        if not sub:
            return None
        df = read_hdf5_node(path, sub)
        return int(df.shape[0]) if df is not None else None

    def sub_streams(self, path: str) -> list[str]:
        from app.tools.load_dataset import _list_hdf5_native_nodes

        return [n["node"] for n in _list_hdf5_native_nodes(path)]

    def timestamp(self, path: str, *, sub: str | None, column: str | None):
        """读 h5 节点的时间戳列（compound 字段；缺省 timestamp 字段）。"""
        if not sub:
            return (None, None)
        from app.tools._data_access import read_h5_node_field

        field = column or "timestamp"
        series = read_h5_node_field(f"{path}{_SUB_SEP}{sub}", field)
        if series is None and field != "timestamp":
            field = "timestamp"
            series = read_h5_node_field(f"{path}{_SUB_SEP}{sub}", field)
        return (series, field if series is not None else None)


class _McapReader:
    """MCAP 容器：单文件多 topic——path_spec 为 ``"<file>::<topic>"``。"""

    fmt = "mcap"
    extensions = [".mcap"]

    def frame(self, path: str, *, sub: str | None, limit: int | None) -> pd.DataFrame | None:
        from app.tools.mcap_reader import read_mcap_topic

        if not sub:
            # 无 topic：主 topic（消息数最多且可解码）——与单文件加载口径一致。
            from app.tools.mcap_reader import probe_mcap

            probe = probe_mcap(path)
            topics = probe.get("topics") or []
            sub = next(
                (t["topic"] for t in topics if t.get("decodable", True)), None
            )
            if sub is None:
                return None
        result = read_mcap_topic(path, sub, max_messages=limit)
        if not result.get("success"):
            return None
        return result.get("df")

    def columns(self, path: str, *, sub: str | None) -> list[str] | None:
        df = self.frame(path, sub=sub, limit=1)
        return [str(c) for c in df.columns] if df is not None else None

    def nrows(self, path: str, *, sub: str | None) -> int | None:
        from app.tools.mcap_reader import probe_mcap

        if not sub:
            return None
        probe = probe_mcap(path)
        for t in probe.get("topics") or []:
            if t.get("topic") == sub:
                return int(t.get("message_count") or 0)
        return None

    def sub_streams(self, path: str) -> list[str]:
        from app.tools.mcap_reader import probe_mcap

        probe = probe_mcap(path)
        return [t["topic"] for t in (probe.get("topics") or [])]

    def timestamp(self, path: str, *, sub: str | None, column: str | None):
        """MCAP 容器时间为 uint64 纳秒（列名带 _ns 后缀）。"""
        df = self.frame(path, sub=sub, limit=None)
        if df is None:
            return (None, None)
        field = column or "mcap_log_time_ns"
        if field not in df.columns:
            field = "mcap_log_time_ns" if "mcap_log_time_ns" in df.columns else None
        if field is None:
            return (None, None)
        series = df[field]
        series.name = field
        return (series, field)


def _install_default_readers() -> None:
    """注册内置六类 reader（模块导入时执行一次）。"""
    for reader in (
        _CsvReader(), _ParquetReader(), _JsonReader(), _JsonlReader(),
        _H5Reader(), _McapReader(),
    ):
        register_reader(reader)


_install_default_readers()
