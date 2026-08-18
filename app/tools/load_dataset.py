"""数据集加载工具。

接收文件路径，按扩展名分发到对应的 pandas 读取器（第一批支持 .csv / .json /
.parquet / .h5），加载结果写入 ``RunContext.df``，元信息写入
``RunContext.meta``，并返回精简的元信息 dict（不返回数据本体）。不支持的格式
返回结构化错误并列出支持的格式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext

# 本工具支持的扩展名 → 说明。
_SUPPORTED_FORMATS: dict[str, str] = {
    ".csv": "逗号分隔文本",
    ".json": "JSON 数组",
    ".parquet": "Parquet 列式存储",
    ".h5": "HDF5 表",
}

# 尝试解码文本文件时使用的编码回退链。
_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _detect_encoding(raw: bytes) -> str:
    """按回退链探测文本编码，无法识别时兜底使用 latin-1。"""
    for enc in _ENCODINGS:
        try:
            raw[:4096].decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _load_csv(path: str) -> pd.DataFrame:
    """读取 CSV，自动探测编码与分隔符。"""
    import csv

    raw = Path(path).read_bytes()
    encoding = _detect_encoding(raw)

    delimiter = ","
    try:
        sample = raw[:8192].decode(encoding, errors="replace")
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        pass

    return pd.read_csv(
        path,
        encoding=encoding,
        sep=delimiter,
        engine="python",
        on_bad_lines="skip",
    )


def _load_hdf5(path: str) -> pd.DataFrame:
    """读取 HDF5 表；存在多个 key 时尝试逐个定位 DataFrame。"""
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
        for key in keys:
            try:
                df = pd.read_hdf(path, key=key)
                if isinstance(df, pd.DataFrame):
                    return df
            except (KeyError, TypeError, ValueError):
                continue
        raise ValueError(f"HDF5 文件 {path} 中未找到可读取的 DataFrame 表。")
    except OSError as exc:
        raise ValueError(f"无法读取 HDF5 文件：{path}（{exc}）") from exc


def _error(
    error: str,
    reason: str,
    user_message: str,
    *,
    supported_formats: list[str] | None = None,
) -> dict[str, Any]:
    """构造统一的错误返回结构。

    Args:
        error: 机器可读的错误类型标识。
        reason: 具体原因（面向开发者/日志）。
        user_message: 可直接转达给用户的中文说明。
        supported_formats: 支持的格式列表（可选）。

    Returns:
        统一结构的错误 dict：success=False + error/reason/user_message。
        错误返回**不附带文件内容预览**，避免模型把内容片段编造进回答。
    """
    result: dict[str, Any] = {
        "success": False,
        "error": error,
        "reason": reason,
        "user_message": user_message,
    }
    if supported_formats is not None:
        result["supported_formats"] = supported_formats
    return result


def load_dataset_impl(context: RunContext, path: str, fmt: str | None = None) -> dict:
    """加载数据集到上下文，并返回精简元信息。

    Args:
        context: 运行时上下文，加载成功的 DataFrame 会写入 context.df，元信息
            写入 context.meta。
        path: 数据集路径，支持 .csv / .json / .parquet / .h5。
        fmt: 可选，显式指定格式（如 "csv"）；省略时根据扩展名自动推断。

    Returns:
        dict，成功时含 success=True、dataset_id、source、format、n_rows、n_cols、
        columns、dtypes、user_message；若覆盖了旧数据集还含 replaced_previous。
        失败时统一返回 success=False 且含 error、reason、user_message（以及可选
        的 supported_formats）。错误返回不附带文件内容预览。

    Raises:
        不直接抛出异常；错误以结构化 dict 返回，便于 Agent 恢复并如实转达。
    """
    source = Path(path)

    if not source.exists():
        return _error(
            "file_not_found",
            f"文件不存在：{path}",
            f"文件 {path} 不存在，请检查路径是否正确。",
        )

    # 解析格式：优先显式 fmt，否则按扩展名推断。
    if fmt:
        ext = fmt.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
    else:
        ext = source.suffix.lower()

    supported = list(_SUPPORTED_FORMATS.keys())
    if ext not in _SUPPORTED_FORMATS:
        fmt_names = "、".join(supported)
        return _error(
            "unsupported_format",
            f"暂不支持格式 {ext}",
            f"暂不支持 {ext} 格式，目前支持：{fmt_names}。请提供其中一种格式的文件路径。",
            supported_formats=supported,
        )

    try:
        if ext == ".csv":
            df = _load_csv(path)
        elif ext == ".json":
            df = pd.read_json(path, encoding=_detect_encoding(source.read_bytes()))
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".h5":
            df = _load_hdf5(path)
        else:  # pragma: no cover - 防御性分支
            raise ValueError(f"未实现格式：{ext}")
    except Exception as exc:  # noqa: BLE001
        return _error(
            "parse_failed",
            f"解析文件失败：{path}（{exc}）",
            f"文件 {path} 解析失败，可能不是有效的 {ext.lstrip('.')} 数据，或文件已损坏。",
            supported_formats=supported,
        )

    # 单数据集语义：记录被替换的旧数据集，再覆盖 df / dataset_id / meta。
    replaced = context.dataset_id
    dataset_id = source.stem

    context.df = df
    context.dataset_id = dataset_id
    meta: dict[str, Any] = {
        "source": path,
        "format": ext.lstrip("."),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
    }
    context.meta = meta

    result: dict[str, Any] = {
        "success": True,
        "dataset_id": dataset_id,
        **meta,
    }
    # 若覆盖了旧数据集，明确标注，避免模型误以为新旧数据集同时可分析。
    if replaced is not None:
        result["replaced_previous"] = replaced
        result["user_message"] = (
            f"已加载数据集 {dataset_id}，并替换先前加载的 {replaced}。"
            "当前仅可分析本数据集。"
        )
    else:
        result["user_message"] = f"已加载数据集 {dataset_id}，当前可对其进行分析。"
    return result


@tool
def load_dataset(
    wrapper: RunContextWrapper[RunContext],
    path: str,
    fmt: str | None = None,
) -> dict:
    """加载数据集到当前会话，返回元信息。

    读取指定文件并按格式写入会话上下文，供后续统计与可视化工具使用。支持
    .csv / .json / .parquet / .h5。

    Args:
        path: 数据集路径。
        fmt: 可选，显式指定格式（如 "csv"）；省略时按扩展名推断。

    Returns:
        dict，包含 success、source、format、n_rows、n_cols、columns；失败时
        返回 success=False 且含 error 与 suggestion。
    """
    return load_dataset_impl(wrapper.context, path, fmt)
