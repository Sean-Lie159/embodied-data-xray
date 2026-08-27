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
from app.tools import profile_store

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
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一的错误返回结构。

    Args:
        error: 机器可读的错误类型标识。
        reason: 具体原因（面向开发者/日志，需可定位：含异常类型+肇事文件/阶段）。
        user_message: 可直接转达给用户的中文说明。
        supported_formats: 支持的格式列表（可选）。
        extra: 额外内部字段（如 traceback 关键帧），供定位调试，不进 user_message。

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
    if extra is not None:
        result.update(extra)
    return result


def _tb_key_frames(exc: Exception) -> list[str]:
    """摘取 traceback 的关键帧（文件名:行号:函数），供错误定位。

    只保留 app/ 内部的帧，过滤外部库噪音；最多返回最近 6 帧。

    Args:
        exc: 已抛出的异常。

    Returns:
        关键帧列表，如 ["app/tools/_sniffing.py:1104:infer_role", ...]。
    """
    import traceback

    frames: list[str] = []
    tb = exc.__traceback__
    while tb is not None:
        fname = tb.tb_frame.f_code.co_filename
        fname = str(fname).replace("\\", "/")
        # 只摘项目内部帧，便于定位到肇事函数。
        if "/app/" in fname:
            frames.append(f"{fname.split('/app/', 1)[-1]}:{tb.tb_lineno}")
        tb = tb.tb_next
    return frames[:6] or traceback.format_exc().splitlines()[:3]


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


def _read_table_nrows(path: Path) -> int | None:
    """只读表格行数（不读全量数据），用于主表选择评分。

    收敛到 _data_access.read_table_nrows 统一读数入口，保证与 inspect_streams /
    check_temporal_sync 行数读数一致。

    Args:
        path: 表格文件路径。

    Returns:
        行数（不含表头）；读取失败返回 None。
    """
    from app.tools import _data_access

    return _data_access.read_table_nrows(str(path), path.suffix.lstrip(".").lower())


def _read_table_sample(path: Path) -> pd.DataFrame | None:
    """读取表格前若干行样本（用于第 2 层内容指纹），不读全量。

    Args:
        path: 表格文件路径。

    Returns:
        前 `_FINGERPRINT_SAMPLE_ROWS` 行样本 DataFrame；读取失败返回 None。
    """
    from app.tools._sniffing import _FINGERPRINT_SAMPLE_ROWS

    try:
        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(
                path,
                encoding=_detect_encoding(path.read_bytes()),
                nrows=_FINGERPRINT_SAMPLE_ROWS,
                engine="python",
            )
        if ext == ".parquet":
            return pd.read_parquet(path, columns=None).head(_FINGERPRINT_SAMPLE_ROWS)
        if ext == ".json":
            with open(path, encoding=_detect_encoding(path.read_bytes())) as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return pd.DataFrame(obj[:_FINGERPRINT_SAMPLE_ROWS])
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                return pd.DataFrame(obj["data"][:_FINGERPRINT_SAMPLE_ROWS])
            return None
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

    # 表格语义识别（四层架构）：覆盖全部表格文件。第 1 层词典线索（raw_sniff）
    # 与第 2 层内容指纹裁判（klass）分别保留——原始线索供主表选择，klass 供流登记。
    # 兜底：单个文件的任何探测操作失败，记录 probe_error 并继续，绝不让单文件打崩
    # 整个加载流程；失败清单汇总到 probe_errors 供调用方查看。
    table_sniffs: list[dict[str, Any]] = []  # 第 2 层 classify 结果
    table_info: list[dict[str, Any]] = []
    probe_errors: list[dict[str, Any]] = []  # 探测失败的文件清单
    # 主表候选：（路径、列名、原始嗅探、分类结果、行数、列数）。
    candidates: list[dict[str, Any]] = []
    # 表格候选 = probe["tables"]（csv/parquet） + cals 中**非标定**的 .json
    # （nuScenes 等数据表 JSON；标定 JSON 仍归 cals，不作为数据表流登记）。
    table_candidates = list(probe["tables"])
    for p_str in probe["cals"]:
        if Path(p_str).suffix.lower() != ".json":
            continue
        obj = _parse_calibration(Path(p_str))
        is_cal = _sniffing.is_calibration_file(obj) or bool(
            _sniffing.fingerprint_calibration(obj).get("present")
        )
        if not is_cal:
            table_candidates.append(p_str)
    for p_str in table_candidates:
        p = Path(p_str)
        try:
            cols = _read_table_columns(p)
            if cols is None:
                # 读列名失败不视为崩溃：记录并继续（该文件不参与主表/流登记）。
                probe_errors.append({
                    "file": str(p),
                    "phase": "enumeration",
                    "probe_error": "读取表列名失败",
                })
                continue
            nrows = _read_table_nrows(p)
            ncols = len(cols)
            raw_sniff = _sniffing.sniff_table_columns(cols)  # 第 1 层词典线索
            sample = _read_table_sample(p)  # 第 2 层内容指纹样本
            klass = _sniffing.classify_table_stream(
                p.name, cols, sample, nrows or 0
            )
            table_sniffs.append(klass)
            candidates.append({
                "file": str(p),
                "name": p.name,
                "sniff": raw_sniff,
                "klass": klass,
                "nrows": nrows,
                "ncols": ncols,
            })
            table_info.append({
                "file": str(p),  # 完整路径，供流登记表按需定位
                "name": p.name,
                "columns": cols[:20],
                "sniff": klass,
                "nrows": nrows,
            })
        except Exception as exc:  # noqa: BLE001 - 单文件探测异常兜底，绝不中断加载
            probe_errors.append({
                "file": str(p),
                "phase": "fingerprint",
                "probe_error": f"{type(exc).__name__}: {exc}",
            })
            continue

    # 主表选择（显式策略）：含状态/动作列 > 行数×列数最大 > 字母序。
    # 空文件（0 行，如空 JSON []/{} 或 2 字节 ego_pose.json）不作为主表候选——
    # 纯媒体/空 JSON 数据集"无有效主表"是合法的，此时 main_table 应为 null。
    # 注意：小但非空（>0 行）的表仍是合法候选（如 2 行 small.csv）。
    selected: dict[str, Any] | None = None
    ranked: list[dict[str, Any]] = []
    data_candidates = [
        c for c in candidates
        if (c.get("nrows") or 0) > 0
    ]
    if data_candidates:
        def _main_table_key(c: dict[str, Any]) -> tuple[int, int, str]:
            has_actions = 1 if c["sniff"]["has_actions"]["present"] else 0
            size = (c["nrows"] or 0) * c["ncols"]
            return (has_actions, size, c["name"])
        ranked = sorted(data_candidates, key=_main_table_key, reverse=True)
        selected = ranked[0]

    main_table: pd.DataFrame | None = None
    main_table_path: str | None = None
    main_table_info: dict[str, Any] = {}
    if selected is not None:
        main_table_path = selected["file"]
        # 主表全量装载（默认全量；超阈值才截断并声明，见下方 rows_total/rows_loaded）。
        try:
            ext = Path(main_table_path).suffix.lower()
            if ext == ".csv":
                main_table = _load_csv(main_table_path)
            elif ext == ".parquet":
                main_table = pd.read_parquet(main_table_path)
            elif ext == ".json":
                main_table = pd.read_json(main_table_path)
        except Exception:  # noqa: BLE001
            main_table = None

    # 装载完整性声明：记录真实总行数 rows_total；超阈值截断到 cap_rows 后 rows_loaded
    # 小于 rows_total，返回必须同时包含两个数字并明确提示截断。
    rows_total: int | None = None
    rows_loaded: int | None = None
    truncated: bool = False
    truncation_note: str | None = None
    if main_table is not None:
        rows_total = int(main_table.shape[0])
        cap_rows = getattr(context, "max_rows_in_context", 500_000)
        if rows_total > cap_rows:
            truncated = True
            main_table = main_table.head(cap_rows).copy()
            rows_loaded = int(main_table.shape[0])
            truncation_note = (
                f"仅装载前 {rows_loaded} 行（共 {rows_total} 行）——"
                "该表超过行数阈值，超出部分未载入内存。"
            )
        else:
            rows_loaded = rows_total

    # 主表选择依据与落选候选（供返回透明化，避免"静默选主表"）。
    main_table_selection: dict[str, Any] = {"selected": None, "reason": None, "candidates": []}
    if selected is not None and ranked:
        has_actions = selected["sniff"]["has_actions"]["present"]
        if has_actions:
            reason = "含状态/动作列，优先作为主表"
        else:
            reason = (
                f"行数×列数最大（约 {selected['nrows'] or '?'} 行 × "
                f"{selected['ncols']} 列，规模 { (selected['nrows'] or 0) * selected['ncols'] }）"
            )
        main_table_selection = {
            "selected": selected["name"],
            "reason": reason,
            "candidates": [
                {
                    "name": c["name"],
                    "has_actions": c["sniff"]["has_actions"]["present"],
                    "nrows": c["nrows"],
                    "ncols": c["ncols"],
                }
                for c in ranked
            ],
        }

    # 标定检测：覆盖全部标定候选文件（json/yaml）。第 1 层词典快检 + 第 2 层
    # 内容指纹（fingerprint_calibration）确认，二者任一命中即判为标定。
    calib_detected = False
    calib_detail: list[dict[str, Any]] = []
    for p_str in probe["cals"]:
        try:
            obj = _parse_calibration(Path(p_str))
            is_cal = _sniffing.is_calibration_file(obj) or bool(
                _sniffing.fingerprint_calibration(obj).get("present")
            )
            if is_cal:
                calib_detected = True
                fp = _sniffing.fingerprint_calibration(obj)
                calib_detail.append({
                    "path": p_str,
                    "name": Path(p_str).name,
                    "keys_found": fp.get("keys_found", []),
                    "evidence": fp.get("evidence", ""),
                })
        except Exception as exc:  # noqa: BLE001 - 单标定文件探测异常兜底，不中断
            probe_errors.append({
                "file": p_str,
                "phase": "fingerprint",
                "probe_error": f"{type(exc).__name__}: {exc}",
            })

    # 视频嗅探（ffprobe，可降级），覆盖全部视频文件。
    video_files: list[str] = []
    video_meta: list[dict[str, Any]] = []
    ffprobe_degraded: str | None = None
    for p_str in probe["videos"]:
        video_files.append(p_str)
        try:
            meta = _sniffing.probe_video(p_str)
        except Exception as exc:  # noqa: BLE001 - 单视频探测异常兜底，不中断加载
            probe_errors.append({
                "file": p_str,
                "phase": "fingerprint",
                "probe_error": f"{type(exc).__name__}: {exc}",
            })
            continue
        if not meta.get("ffprobe_available", True):
            ffprobe_degraded = meta.get("user_message")
        video_meta.append({"file": p_str, **meta})

    # 音频/图片：只登记路径与格式，不读取内容（覆盖全部文件）。
    audio_meta: list[dict[str, Any]] = []
    image_meta: list[dict[str, Any]] = []
    for p_str in probe["audios"]:
        audio_meta.append({"file": p_str, "format": Path(p_str).suffix.lstrip(".").lower()})
    for p_str in probe["images"]:
        image_meta.append({"file": p_str, "format": Path(p_str).suffix.lstrip(".").lower()})

    caps_result = _sniffing.build_capabilities(probe, table_sniffs)
    caps_result["capabilities"]["has_calibration"] = calib_detected

    # 流配对规则（mp4↔metainfo、accel+gyro=六轴IMU）。
    stream_pairs = _sniffing.pair_streams(probe["videos"], probe["tables"], probe["audios"])
    has_imu_6axis_pair = any(p["type"] == "imu_6axis" for p in stream_pairs)
    caps_result["capabilities"]["has_imu_6axis_pair"] = has_imu_6axis_pair
    caps_result["capabilities"]["has_media_metainfo_pair"] = any(
        p["type"] == "media_metainfo" for p in stream_pairs
    )
    # 修复侧栏 IMU 轴数显示 None：has_imu 但聚合轴数为 None 时，若存在 accel+gyro
    # 六轴配对则标 6；否则标 "unknown"（避免 UI 显示 None）。
    if (
        caps_result["capabilities"].get("has_imu")
        and caps_result["capabilities"].get("imu_axes") is None
    ):
        caps_result["capabilities"]["imu_axes"] = 6 if has_imu_6axis_pair else "unknown"

    # 记录路径清单与元数据（不读入内存）。
    dataset_id = dir_path.name
    meta: dict[str, Any] = {
        "source": str(dir_path),
        "kind": "directory",
        "capabilities": caps_result["capabilities"],
        "guessed_type": caps_result["guessed_type"],
        "guessed_type_confidence": caps_result["guessed_type_confidence"],
        "video_files": video_files,
        "video_meta": video_meta,
        "audio_files": [a["file"] for a in audio_meta],
        "image_files": [i["file"] for i in image_meta],
        # 流登记表：覆盖全部表格（读头部判类型）+ 视频/音频/图片（只登记路径）。
        # 每条流含 {path, format, kind, channels, role, semantic_label,
        # label_evidence, label_confidence, status, timestamp_column,
        # quaternion_groups, imu_axes}，供 inspect_streams 按需读取。
        "streams": _sniffing.build_streams_registry(
            probe, table_info, video_meta, audio_meta, image_meta
        ),
        # 流配对规则结果（mp4↔metainfo、accel+gyro=六轴IMU）。
        "stream_pairs": stream_pairs,
        # 探测失败的文件清单（兜底：单文件失败不中断，记录原因供定位）。
        "probe_errors": probe_errors,
        # 标定文件细节（第 2 层指纹确认的键与依据）。
        "calibration_detail": calib_detail,
        # 主表信息：选择依据、装载完整性声明（供后续统计工具继承）。
        "main_table": {
            "file": main_table_path,
            "name": main_table_selection["selected"],
            "selection": main_table_selection,
            "rows_total": rows_total,
            "rows_loaded": rows_loaded,
            "truncated": truncated,
        },
    }

    # 第 4 层：用户确认持久化覆盖。加载时优先读取 outputs/.dataset_profile.json
    # 中该 dataset_id 的已确认映射（来源 user_confirmed），覆盖第 1-3 层自动识别。
    # 文件不存在/损坏时安全降级为无覆盖，不中断加载。
    user_profile = profile_store.load_dataset_profile(context.output_dir, dataset_id)
    if user_profile.get("streams"):
        meta["streams"] = profile_store.apply_profile_overrides(
            meta["streams"], user_profile
        )
    meta["user_profile"] = user_profile

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
            # 完整文件清单，按类型分组（全部路径，不抽样、不省略）。
            "tables": probe["tables"],
            "videos": probe["videos"],
            "audios": probe["audios"],
            "images": probe["images"],
            "cals": probe["cals"],
            "others": probe["others"],
        },
        "capabilities": caps_result["capabilities"],
        "guessed_type": caps_result["guessed_type"],
        "guessed_type_confidence": caps_result["guessed_type_confidence"],
        "video_files": video_files,
        "video_meta": video_meta,
        "audio_files": [a["file"] for a in audio_meta],
        "image_files": [i["file"] for i in image_meta],
        "table_info": table_info,
        "main_table_selection": main_table_selection,
        # 流配对与标定细节透出，供 agent 了解配对关系与标定依据。
        "stream_pairs": stream_pairs,
        "calibration_detail": calib_detail,
        # 探测失败的文件清单（结构化：file + phase + probe_error）。
        "probe_errors": probe_errors,
    }
    if main_table is not None:
        result["main_table"] = {
            "file": main_table_path,
            "name": main_table_selection["selected"],
            "n_rows": int(main_table.shape[0]),
            "n_cols": int(main_table.shape[1]),
            # 装载完整性：真实总行数、实际装载行数、是否截断，三者同时可见。
            "rows_total": rows_total,
            "rows_loaded": rows_loaded,
            "truncated": truncated,
        }
    if ffprobe_degraded:
        result["ffprobe_degraded"] = ffprobe_degraded
    # user_message：含主表选择依据 + 截断声明（若有）。
    msg_parts = [
        f"已探测数据集目录 {dataset_id}：{probe['total_files']} 个文件，"
        f"推测类型 {caps_result['guessed_type']}。"
    ]
    if selected is not None:
        msg_parts.append(
            f"主表选择 {selected['name']}（{main_table_selection['reason']}）。"
        )
    if truncated and truncation_note:
        msg_parts.append(truncation_note)
    if ffprobe_degraded:
        msg_parts.append(ffprobe_degraded)
    if probe_errors:
        msg_parts.append(
            f"{len(probe_errors)} 个文件探测失败（详见 probe_errors），已跳过并继续其余文件。"
        )
    result["user_message"] = " ".join(msg_parts)
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
        try:
            result = _load_directory_impl(context, source)
        except Exception as exc:  # noqa: BLE001 - 目录加载整体兜底，异常转结构化错误
            # 绝不把裸 Python 异常（如 "list index out of range"）原样抛给用户；
            # reason 需可定位：异常类型 + 关键 traceback 帧（文件名:行号）+ 目录。
            frames = _tb_key_frames(exc)
            return _error(
                "directory_probe_failed",
                f"目录探测失败（{type(exc).__name__}: {exc}），目录 {source}；"
                f"关键帧：{frames or '无'}",
                f"加载目录 {source} 时探测失败，已停止本次加载。请检查目录内文件是否含异常数据。",
                extra={"traceback_frames": frames, "probe_error": f"{type(exc).__name__}: {exc}"},
            )
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


def confirm_stream_semantic_impl(
    context: RunContext,
    filename: str,
    *,
    kind: str | None = None,
    role: dict[str, Any] | None = None,
    semantic_label: str | None = None,
    label_evidence: str | None = None,
    imu_axes: int | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """第 4 层用户确认入口（内部函数，非 agent 工具）。

    用户对某条流的语义标签提出确认/纠正后，经此函数把映射写入
    ``outputs/.dataset_profile.json``（来源标 user_confirmed）。下一次加载该数据集
    时，load_dataset 会优先读取并覆盖第 1-3 层的自动识别。

    按既定设计，本函数不注册为 @tool——用户质疑标签时由 agent 用自然语言对话处理，
    确认结果经此函数落盘。

    Args:
        context: 运行时上下文（取 dataset_id 与 output_dir）。
        filename: 被确认流的文件名（不含路径）。
        kind/role/semantic_label/label_evidence/imu_axes/status: 用户确认后的字段。

    Returns:
        dict，success + 覆盖后的流映射 + user_message。
    """
    if not context.dataset_id:
        return {
            "success": False,
            "error": "no_data_loaded",
            "user_message": "尚未加载任何数据集，无法确认语义标签。请先 load_dataset。",
        }
    mapping: dict[str, Any] = {}
    if kind is not None:
        mapping["kind"] = kind
    if role is not None:
        mapping["role"] = role
    if semantic_label is not None:
        mapping["semantic_label"] = semantic_label
    if label_evidence is not None:
        mapping["label_evidence"] = label_evidence
    if imu_axes is not None:
        mapping["imu_axes"] = imu_axes
    if status is not None:
        mapping["status"] = status

    profile = profile_store.save_dataset_profile(
        context.output_dir, context.dataset_id,
        stream_overrides={filename: mapping},
    )
    return {
        "success": True,
        "dataset_id": context.dataset_id,
        "filename": filename,
        "mapping": {**mapping, "source": "user_confirmed"},
        "user_message": (
            f"已记录你对 {filename} 的语义确认为 user_confirmed，"
            f"下次加载 {context.dataset_id} 时将优先采用。"
        ),
    }


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
