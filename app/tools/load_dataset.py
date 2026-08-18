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


def load_dataset_impl(context: RunContext, path: str, fmt: str | None = None) -> dict:
    """加载数据集到上下文，并返回精简元信息。

    Args:
        context: 运行时上下文，加载成功的 DataFrame 会写入 context.df，元信息
            写入 context.meta。
        path: 数据集路径，支持 .csv / .json / .parquet / .h5。
        fmt: 可选，显式指定格式（如 "csv"）；省略时根据扩展名自动推断。

    Returns:
        dict，包含 success、source、format、n_rows、n_cols、columns；文件不存在
        或格式不支持时返回 success=False 且含 error 与 suggestion 的结构化信息。

    Raises:
        不直接抛出异常；错误以结构化 dict 的 error 字段返回，便于 Agent 恢复。
    """
    source = Path(path)

    if not source.exists():
        return {
            "success": False,
            "error": f"文件不存在：{path}",
            "suggestion": "请确认路径正确。",
        }

    # 解析格式：优先显式 fmt，否则按扩展名推断。
    if fmt:
        ext = fmt.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
    else:
        ext = source.suffix.lower()

    if ext not in _SUPPORTED_FORMATS:
        supported = "、".join(_SUPPORTED_FORMATS)
        return {
            "success": False,
            "error": f"暂不支持格式 {ext}（或文件 {path} 无扩展名）。",
            "suggestion": f"支持格式：{supported}。",
        }

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
        return {
            "success": False,
            "error": f"解析文件失败：{path}（{exc}）",
            "suggestion": "请确认文件内容与格式匹配且未损坏。",
        }

    # 写入上下文：数据本体与元信息分离，元信息用于后续工具按需查询。
    context.df = df
    meta: dict[str, Any] = {
        "source": path,
        "format": ext.lstrip("."),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
    }
    context.meta = meta

    return {"success": True, **meta}


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
