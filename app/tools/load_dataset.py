"""数据集加载工具。

支持两种输入：
1. **单文件**：按扩展名分发到 pandas 读取器（csv / json / parquet / h5）；
2. **目录**：递归文件普查 + 能力嗅探（表格列名推断、json/yaml 标定检测、视频
   ffprobe 元数据），生成能力标签与推测类型，写入 ``RunContext.meta``。

加载结果写入 ``RunContext.df``，元信息写入 ``RunContext.meta``，返回精简的元信息
dict（不返回数据本体）。目录加载时不把整个数据集读入内存，视频等大文件只记录路径。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools import _sniffing

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


def _read_table_columns(path: Path) -> list[str] | None:
    """只读表格列名（不读全量数据），用于嗅探。

    Args:
        path: 表格文件路径。

    Returns:
        列名列表；读取失败返回 None。
    """
    try:
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(
                path,
                encoding=_detect_encoding(path.read_bytes()),
                nrows=0,
                engine="python",
            )
            return [str(c) for c in df.columns]
        if ext == ".parquet":
            df = pd.read_parquet(path, columns=None)
            return [str(c) for c in df.columns[:50]]
        if ext == ".json":
            with open(path, encoding=_detect_encoding(path.read_bytes())) as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "columns" in obj:
                return [str(c) for c in obj["columns"]]
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return [str(c) for c in obj[0].keys()]
            return []
        return None
    except Exception:  # noqa: BLE001
        return None


def _parse_calibration(path: Path) -> Any:
    """解析 json/yaml 标定候选文件。

    Args:
        path: 文件路径。

    Returns:
        解析后的对象；失败返回 None。
    """
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        if path.suffix.lower() == ".json":
            with open(path, encoding=_detect_encoding(path.read_bytes())) as f:
                return json.load(f)
        return None
    except Exception:  # noqa: BLE001
        return None


def _load_directory_impl(context: RunContext, dir_path: Path) -> dict[str, Any]:
    """目录加载：文件普查 + 能力嗅探，返回精简摘要。

    Args:
        context: 运行时上下文（写入 capabilities / guessed_type / df / meta）。
        dir_path: 数据集目录。

    Returns:
        dict，含 success、dataset_id、file_survey（普查摘要）、capabilities、
        guessed_type、video_files、table_info、user_message。
    """
    probe = _sniffing.probe_directory(dir_path)

    # 表格列名嗅探。
    table_sniffs: list[dict[str, Any]] = []
    main_table: pd.DataFrame | None = None
    main_table_path: str | None = None
    table_info: list[dict[str, Any]] = []
    for p_str in probe["sample_tables"]:
        p = Path(p_str)
        cols = _read_table_columns(p)
        if cols is None:
            continue
        sniff = _sniffing.sniff_table_columns(cols)
        table_sniffs.append(sniff)
        table_info.append({
            "file": str(p),  # 完整路径，供流登记表按需定位
            "name": p.name,
            "columns": cols[:20],
            "sniff": sniff,
        })
        # 选择第一个有内容的表格作为主表（记录但不整表读入，除非是主表）。
        if main_table is None:
            main_table_path = p_str

    # 若存在主表格且数据量不大，装载其头部作为 df 的轻量代表。
    if main_table_path is not None:
        try:
            ext = Path(main_table_path).suffix.lower()
            if ext == ".csv":
                main_table = _load_csv(main_table_path)
            elif ext == ".parquet":
                main_table = pd.read_parquet(main_table_path)
            elif ext == ".json":
                main_table = pd.read_json(main_table_path)
            # 仅保留前 max_rows_in_context 行，避免大表塞入内存。
            cap_rows = getattr(context, "max_rows_in_context", 200)
            if main_table is not None and len(main_table) > cap_rows:
                main_table = main_table.head(cap_rows).copy()
        except Exception:  # noqa: BLE001
            main_table = None

    # 标定检测。
    calib_detected = False
    for p_str in probe["sample_calibs"]:
        obj = _parse_calibration(Path(p_str))
        if _sniffing.is_calibration_file(obj):
            calib_detected = True
            break

    # 视频嗅探（ffprobe，可降级）。
    video_files: list[str] = []
    video_meta: list[dict[str, Any]] = []
    ffprobe_degraded: str | None = None
    for p_str in probe["sample_videos"]:
        video_files.append(p_str)
        meta = _sniffing.probe_video(p_str)
        if not meta.get("ffprobe_available", True):
            ffprobe_degraded = meta.get("user_message")
        video_meta.append({"file": Path(p_str).name, **meta})

    caps_result = _sniffing.build_capabilities(probe, table_sniffs)
    caps_result["capabilities"]["has_calibration"] = calib_detected

    # 记录视频路径清单与元数据（不读入内存）。
    dataset_id = dir_path.name
    meta: dict[str, Any] = {
        "source": str(dir_path),
        "kind": "directory",
        "capabilities": caps_result["capabilities"],
        "guessed_type": caps_result["guessed_type"],
        "guessed_type_confidence": caps_result["guessed_type_confidence"],
        "video_files": video_files,
        "video_meta": video_meta,
        # 流登记表：每条流含 {path, format, kind, channels, role}，供
        # inspect_streams 按需读取时间戳实测采样率。
        "streams": _sniffing.build_streams_registry(probe, table_info, video_meta),
    }
    context.meta = meta
    context.dataset_id = dataset_id
    context.df = main_table

    result: dict[str, Any] = {
        "success": True,
        "dataset_id": dataset_id,
        "kind": "directory",
        "file_survey": {
            "total_files": probe["total_files"],
            "ext_dist": probe["ext_dist"],
            "subdirs": probe["subdirs"][:20],
        },
        "capabilities": caps_result["capabilities"],
        "guessed_type": caps_result["guessed_type"],
        "guessed_type_confidence": caps_result["guessed_type_confidence"],
        "video_files": video_files,
        "video_meta": video_meta,
        "table_info": table_info,
    }
    if main_table is not None:
        result["main_table"] = {
            "file": main_table_path,
            "n_rows": int(main_table.shape[0]),
            "n_cols": int(main_table.shape[1]),
        }
    if ffprobe_degraded:
        result["ffprobe_degraded"] = ffprobe_degraded
    result["user_message"] = (
        f"已探测数据集目录 {dataset_id}：{probe['total_files']} 个文件，"
        f"推测类型 {caps_result['guessed_type']}。"
        + (f" {ffprobe_degraded}" if ffprobe_degraded else "")
    )
    return result


def load_dataset_impl(context: RunContext, path: str, fmt: str | None = None) -> dict:
    """加载数据集到上下文，并返回精简元信息。

    Args:
        context: 运行时上下文，加载成功的 DataFrame 会写入 context.df，元信息
            写入 context.meta。
        path: 数据集路径，支持单文件（.csv/.json/.parquet/.h5）或目录。
        fmt: 可选，显式指定格式（如 "csv"）；省略时根据扩展名自动推断。

    Returns:
        dict，成功时含 success=True、dataset_id、source 及元信息；失败时统一
        返回 success=False 且含 error、reason、user_message（以及可选的
        supported_formats）。错误返回不附带文件内容预览。

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

    # 目录输入：走文件普查与能力嗅探。
    if source.is_dir():
        replaced = context.dataset_id
        result = _load_directory_impl(context, source)
        if result.get("success"):
            if replaced is not None:
                result["replaced_previous"] = replaced
                result["user_message"] += f"（替换先前加载的 {replaced}）"
        return result

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

    支持两种输入：
    - 单文件：按格式读取（.csv / .json / .parquet / .h5）；
    - 目录：递归文件普查 + 能力嗅探，生成能力标签与推测类型（不整表读入内存，
      视频等大文件仅记录路径清单；若 ffprobe 不可用则跳过视频嗅探并提示）。

    Args:
        path: 数据集路径（文件或目录）。
        fmt: 可选，单文件时显式指定格式（如 "csv"）；省略时按扩展名推断。

    Returns:
        dict，单文件含 success、dataset_id、source、format、n_rows、n_cols、
        columns；目录含 success、dataset_id、file_survey、capabilities、
        guessed_type、video_files；失败时返回 success=False 且含 error 与
        user_message。
    """
    return load_dataset_impl(wrapper.context, path, fmt)
