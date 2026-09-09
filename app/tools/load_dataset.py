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
from app.tools import _data_access, _sniffing
from app.tools._sniffing import probe_full_paths
from app.tools import profile_store

# 本工具支持的扩展名 → 说明。
# .json（整体一个 JSON 值）与 .jsonl（每行一个 JSON 对象）是两种格式，分别读取，
# 不得混用：.jsonl 一律走 lines=True。
_SUPPORTED_FORMATS: dict[str, str] = {
    ".csv": "逗号分隔文本",
    ".json": "JSON 数组",
    ".jsonl": "JSON Lines（每行一个 JSON 对象）",
    ".parquet": "Parquet 列式存储",
    ".h5": "HDF5 表",
}

# 尝试解码文本文件时使用的编码回退链。
_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

# JSONL 嗅探采样行数：目录嗅探时只取前若干行判列结构与 dtype。
# 取 5 行而非 _FINGERPRINT_SAMPLE_ROWS(500)：JSONL 需逐行解析 JSON，成本高于
# csv/parquet 的列裁剪；而判别列结构与 dtype（含嵌套值 → object）5 行已足够。
# 全量统计仍由 profile_data / check_sensor_sanity 读全表完成，不依赖此采样。
_JSONL_SNIFF_ROWS = 5


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


class MissingDependencyError(RuntimeError):
    """运行环境缺少读某格式所需的可选依赖（如读 HDF5 需 pytables）。

    与"文件损坏/格式不对"严格区分：缺依赖时工具根本没读到文件内容，
    不得给出"可能损坏"之类误导性兜底措辞（诚实降级纪律）。
    """

    def __init__(self, package: str, import_name: str, fmt: str) -> None:
        self.package = package
        self.import_name = import_name
        self.fmt = fmt
        super().__init__(
            f"读取 {fmt} 需要 {package} 包（import {import_name}），当前环境未安装"
        )

    def user_hint(self) -> str:
        """可直接转达给用户的中文修复指引（不含"文件损坏"类误导措辞）。"""
        return (
            f"读取 {self.fmt} 需要环境安装 {self.package} 依赖（pip 包名 "
            f"{self.import_name}），当前环境未安装——文件未被读取，"
            "并非文件损坏。请在运行环境执行 `pip install "
            f"{self.import_name}` 后重新加载。"
        )


def _load_hdf5_native(path: str) -> pd.DataFrame | None:
    """用 h5py 读取原生层级结构的 HDF5（非 pandas HDFStore 格式）。

    真实形态（具身智能采集）：root 下 action/observation/pose/meta 等组，
    数据节点是 compound dtype 的结构化数组（字段如 value/timestamp/file_path，
    可能含子数组字段如 value <f4 (7,)），另有标定矩阵（2D float）与
    object 标量节点。

    选择策略：在全部候选节点（compound 或 2D 数值）中选**信息量最大**者
    （行数×列数）转为主 DataFrame；compound 含子数组字段时保持 object 列
    （每行一个 ndarray，与嵌套向量处理路径一致）。

    Args:
        path: 文件路径。

    Returns:
        主 DataFrame；文件非 HDF5 / 无候选节点 / h5py 不可用返回 None
        （调用方按原路径兜底）。
    """
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            candidates: list[tuple[str, int, int, Any]] = []  # (路径, 行, 列, 数据)
            def _visit(name: str, node: Any) -> None:
                if not isinstance(node, h5py.Dataset):
                    return
                dtype = node.dtype
                if node.dtype == object and node.shape == (1,):
                    return  # object 标量（extra_info / camera_model 等）
                if dtype.names:  # compound：字段名即列名
                    ncols = len(dtype.names)
                elif node.ndim == 2:
                    ncols = int(node.shape[1])
                else:
                    return  # 1D 标量数组（distortion (4,) 等）不构成表
                candidates.append((name, int(node.shape[0]), ncols, node[()]))
            f.visititems(_visit)
            if not candidates:
                return None
            # 信息量最大者为主表；并列时路径字母序（确定性）。
            candidates.sort(key=lambda c: (-(c[1] * c[2]), c[0]))
            best_path, _rows, _cols, data = candidates[0]
            try:
                df = pd.DataFrame(data)
            except ValueError:
                # compound 含子数组字段（如 value <f4 (7,)）→ 逐字段转，
                # 子数组字段保持 object 列（与 _read_hdf5_node 同款防护）。
                names = getattr(data.dtype, "names", None)
                if not names:
                    return None
                cols: dict[str, Any] = {}
                for name in names:
                    col = data[name]
                    cols[name] = (
                        list(col) if col.ndim > 1 else col
                    )  # 子数组字段 → object 列
                df = pd.DataFrame(cols)
            df.attrs["h5_source_node"] = best_path
            df.attrs["h5_structure"] = [
                {"node": c[0], "rows": c[1], "cols": c[2]}
                for c in candidates[:20]
            ]
            return df
    except OSError:
        return None  # 非 HDF5 签名 → 调用方按"可能损坏"兜底（文件确实读过）
    except Exception:  # noqa: BLE001
        return None


def _list_hdf5_native_nodes(path: str) -> list[dict[str, Any]]:
    """列出 h5py 原生层级文件的全部候选数据节点（供流登记）。

    候选与 _load_hdf5_native 同口径：compound dtype（字段名即列名）或
    2D 数值数组；object 标量与 1D 标量数组不算表。

    Args:
        path: 文件路径。

    Returns:
        节点清单 [{node, rows, cols, fields}]，按 行×列 降序（主表在首）；
        非 HDF5/h5py 不可用返回 []。
    """
    try:
        import h5py
    except ImportError:
        return []
    try:
        out: list[dict[str, Any]] = []
        with h5py.File(path, "r") as f:
            def _visit(name: str, node: Any) -> None:
                if not isinstance(node, h5py.Dataset):
                    return
                dtype = node.dtype
                if node.dtype == object and node.shape == (1,):
                    return
                if dtype.names:
                    ncols = len(dtype.names)
                    fields = list(dtype.names)
                elif node.ndim == 2:
                    ncols = int(node.shape[1])
                    fields = []
                else:
                    return
                out.append({
                    "node": name,
                    "rows": int(node.shape[0]),
                    "cols": ncols,
                    "fields": fields,
                })
            f.visititems(_visit)
        out.sort(key=lambda c: (-(c["rows"] * c["cols"]), c["node"]))
        return out
    except Exception:  # noqa: BLE001
        return []


def _classify_h5_node(fields: list[str], node_path: str) -> tuple[str, str]:
    """按节点字段特征判定 kind 与语义标签（确定性，不硬猜语义之外的）。

    Args:
        fields: compound 字段名清单（2D 数值节点为空）。
        node_path: 节点路径（如 action/left_eef/feedback/motor_command）。

    Returns:
        (kind, semantic_label)。
    """
    path_l = node_path.lower()
    fl = [f.lower() for f in fields]
    if any("motor_command" in f or "command" in f for f in fl) or "action" in path_l:
        return "actions", "动作/指令流"
    if any(f in ("timestamp", "file_path", "frame_index") for f in fl) and "camera" in path_l:
        return "frame_index", "相机帧索引"
    if any("orientation" in f or "accel" in f for f in fl) or "imu" in path_l:
        return "imu", "IMU 传感器"
    if "pose" in path_l or any("quat" in f for f in fl):
        return "pose", "位姿流"
    if "calibration" in path_l:
        return "calibration", "标定数据"
    return "unknown", "未知（无法分类）"


def read_hdf5_node(path: str, node: str) -> pd.DataFrame | None:
    """按节点路径读取 h5py 层级文件的单个数据节点为 DataFrame（公开接口，
    供 _data_access / sync / propose 等工具按流登记表读取 h5 节点流）。"""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            node_obj = f.get(node)
            if node_obj is None or not isinstance(node_obj, h5py.Dataset):
                return None
            data = node_obj[()]
            try:
                return pd.DataFrame(data)
            except ValueError:
                # compound 含子数组字段（如 value <f4 (7,)）时整体转 DataFrame
                # 会抛 "must be 1-dimensional"——逐字段转，子数组字段保持
                # object 列（每行一个 ndarray，与嵌套向量处理路径一致）。
                names = getattr(data.dtype, "names", None)
                if not names:
                    return None
                cols: dict[str, Any] = {}
                for name in names:
                    col = data[name]
                    cols[name] = (
                        list(col) if col.ndim > 1 else col
                    )  # 子数组字段 → object 列（每行一个 ndarray）
                return pd.DataFrame(cols)
    except Exception:  # noqa: BLE001
        return None


def register_h5_node_streams(
    context: RunContext, h5_path: Path
) -> list[dict[str, Any]]:
    """把 h5 文件的全部候选数据节点登记为流（目录多流机制复用）。

    目录加载遇到 .h5 时调用：节点流 path 为 "<h5 绝对路径>::<node>"，
    format="h5"，kind 按字段/路径特征判定，主表节点标 is_main（目录语境
    不设 context.df 主表——h5 节点经 resolve_table_name 按需读取）。

    Args:
        context: 运行时上下文（streams 追加到 meta["streams"]）。
        h5_path: h5 文件路径。

    Returns:
        登记的流条目列表。
    """
    nodes = _list_hdf5_native_nodes(str(h5_path))
    entries: list[dict[str, Any]] = []
    for nd in nodes:
        kind, label = _classify_h5_node(nd.get("fields", []), nd["node"])
        entries.append({
            "path": f"{h5_path}::{nd['node']}",
            "format": "h5",
            "kind": kind,
            "semantic_label": label,
            "label_evidence": f"HDF5 节点字段特征（{nd['cols']} 列）",
            "label_confidence": "low" if kind == "unknown" else "medium",
            "label_source": "h5_node_scan",
            "role": {"role": label, "confidence": "medium",
                     "evidence": "HDF5 节点字段特征"},
            "channels": nd.get("fields", []),
            "n_rows": nd["rows"],
            "n_cols": nd["cols"],
            "is_main": nd is nodes[0] if nodes else False,
        })
    if entries:
        context.meta.setdefault("streams", []).extend(entries)
    return entries


def _load_hdf5(path: str) -> pd.DataFrame:
    """读取 HDF5 表；存在多个 key 时尝试逐个定位 DataFrame。

    Raises:
        MissingDependencyError: 环境缺少 pytables（pandas 读 HDF5 的可选依赖，
            pip 包名 tables）——此时文件内容完全未被读取，不得按"文件损坏"
            处理。
        ValueError: 文件确实无法解析（损坏/非 HDF5/无可读 DataFrame 表）。
    """
    keys: list[str] | None
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
    except ImportError as exc:
        # pandas 的可选依赖缺失（Missing optional dependency 'pytables'）。
        # 实测：用户环境装了 h5py（另一个 HDF5 库）但没装 tables，读 .h5 必踩。
        raise MissingDependencyError("pytables", "tables", "HDF5 (.h5)") from exc
    except (OSError, ValueError):
        keys = None  # 非 pandas HDFStore 格式（如 h5py 原生层级）→ 直接走回退。

    # 路径 1：pandas HDFStore 格式 → 逐 key 尝试 read_hdf。
    if keys:
        for key in keys:
            try:
                df = pd.read_hdf(path, key=key)
                if isinstance(df, pd.DataFrame):
                    return df
            except (KeyError, TypeError, ValueError):
                continue

    # 路径 2：h5py 原生层级回退。触发条件包括：非 HDFStore 格式；或 PyTables
    # 能打开、keys 非空但 read_hdf 全败（真实案例：具身智能采集用 h5py 直接
    # 写层级结构——action/observation/pose/meta 各组下是 compound dtype 的
    # 结构化数组，pandas read_hdf 不认，但数据完好且信息量大，实测 46.8MB、
    # 69 个数据节点）。
    native = _load_hdf5_native(path)
    if native is not None:
        return native

    if keys is None:
        # HDFStore 与 h5py 都打不开 → 确实不是有效 HDF5（"可能损坏"措辞恰当）。
        raise ValueError(f"无法读取 HDF5 文件：{path}（非 pandas HDFStore 格式且非 HDF5 层级结构）")
    raise ValueError(f"HDF5 文件 {path} 中未找到可读取的 DataFrame 表。")


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
            from app.tools import _data_access

            rows = _data_access._json_row_list(json.loads(path.read_text(encoding=_detect_encoding(path.read_bytes()))))
            if rows and isinstance(rows[0], dict):
                return [str(c) for c in rows[0].keys()]
            return []
        if ext == ".jsonl":
            # JSONL：逐行解析取首个有效行的键（不读全量）。
            from app.tools._data_access import read_jsonl_rows

            rows = read_jsonl_rows(str(path), limit=1, encoding=_detect_encoding(path.read_bytes()))
            return [str(k) for k in rows[0].keys()] if rows else []
        return None
    except Exception:  # noqa: BLE001
        return None


# file_survey 序列化体积上限（字符数）；超出则压缩为分组计数摘要。
_MAX_SURVEY_CHARS = 50_000


def _compress_file_survey(survey: dict[str, Any]) -> str | None:
    """file_survey 体积护栏：超限时压缩为分组计数摘要，避免撑爆模型上下文。

    压缩策略：保留 total_files / ext_dist（截断）/ 各组**计数**（total/shown/
    truncated），丢弃具体路径列表与 subdirs。返回说明文本（供 note 字段）；
    未超限返回 None。

    Args:
        survey: file_survey（原地压缩）。

    Returns:
        压缩说明文本；未压缩返回 None。
    """
    import json as _json

    def _size(obj: Any) -> int:
        try:
            return len(_json.dumps(obj, ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001
            return _MAX_SURVEY_CHARS + 1

    if _size(survey) <= _MAX_SURVEY_CHARS:
        return None

    # 渐进降级：先砍掉最不关键的 subdirs / ext_dist（保留路径），仍超限才砍路径列表。
    # 这样中等规模目录（几百个文件）能保留具体路径，只有极端情况才退化为纯计数。
    if _size(survey) > _MAX_SURVEY_CHARS:
        survey.pop("subdirs", None)
        survey["note"] = "子目录清单已省略（文件清单体积接近上限）。"
    if _size(survey) > _MAX_SURVEY_CHARS:
        survey.pop("ext_dist", None)
        survey["note"] = "扩展名分布已省略（文件清单体积超过上限）。"
    if _size(survey) <= _MAX_SURVEY_CHARS:
        return survey.get("note")

    compressed: dict[str, Any] = {
        "total_files": survey.get("total_files"),
        "max_listed_per_group": survey.get("max_listed_per_group"),
        "excluded_dirs": survey.get("excluded_dirs", []),
    }
    # 各组只保留计数，不保留路径。
    for key in ("tables", "videos", "audios", "images", "cals", "others"):
        view = survey.get(key)
        if isinstance(view, dict):
            compressed[key] = {
                "total": view.get("total"),
                "shown": view.get("shown"),
                "truncated": view.get("truncated"),
                "paths": [],  # 已压缩，路径列表丢弃
            }
    # ext_dist 只保留数量最多的若干项。
    ext_dist = survey.get("ext_dist") or {}
    if isinstance(ext_dist, dict):
        top = sorted(ext_dist.items(), key=lambda kv: -kv[1])[:20]
        compressed["ext_dist_top"] = dict(top)
    compressed["compressed"] = True
    survey.clear()
    survey.update(compressed)
    return (
        "文件清单因体积超限已压缩为分组计数摘要（各组路径未列出，计数仍完整）；"
        "如需查看具体文件，请缩小目录范围或改用子目录加载。"
    )


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
            from app.tools import _data_access

            rows = _data_access._json_row_list(json.loads(path.read_text(encoding=_detect_encoding(path.read_bytes()))))
            if rows is not None:
                return pd.DataFrame(rows[:_FINGERPRINT_SAMPLE_ROWS])
            return None
        if ext == ".jsonl":
            # JSONL：读前 _JSONL_SNIFF_ROWS 行判列结构与 dtype（不读全量）。
            # 嵌套列表/对象值在此保留为 object dtype，供下游指纹与统计判定。
            from app.tools._data_access import read_jsonl_rows

            rows = read_jsonl_rows(
                str(path), limit=_JSONL_SNIFF_ROWS,
                encoding=_detect_encoding(path.read_bytes()),
            )
            return pd.DataFrame(rows) if rows else None
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


def _attach_nested_discovery(streams: list[dict[str, Any]]) -> None:
    """为信封型流（jsonl/json）附加嵌套时间候选与信号字段（就地修改）。

    单文件失败静默降级为空结果——发现是增强信息，不得阻塞加载主流程。

    Args:
        streams: 流登记表（meta["streams"]，就地修改）。
    """
    for s in streams:
        fmt = str(s.get("format", "")).lower()
        if fmt not in ("jsonl", "json"):
            continue
        path = s.get("path", "")
        try:
            found = _sniffing.discover_nested_fields(path, fmt)
        except Exception:  # noqa: BLE001
            found = {"time_candidates": [], "signal_fields": [],
                     "sampled_rows": 0}
        s["time_candidates"] = found.get("time_candidates", [])
        s["signal_fields"] = found.get("signal_fields", [])


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
    # 表格候选 = probe tables（csv/parquet，用完整路径逐条探测） + cals 中**非标定**
    # 的 .json（数据表 JSON；标定 JSON 仍归 cals，不作为数据表流登记）。
    # 注意：探测必须用完整清单，不能只探测返回里截断的前若干条。
    table_candidates = [
        t for t in _sniffing.probe_full_paths(probe, "tables")
        if not _sniffing._is_dataset_metadata_file(t)
    ]
    for p_str in _sniffing.probe_full_paths(probe, "cals"):
        if Path(p_str).suffix.lower() != ".json":
            continue
        if _sniffing._is_dataset_metadata_file(p_str):
            continue  # meta/*.json 是数据集元数据，不作为数据表流登记
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
                # 经统一 reader（JSON 顶层 dict 按行列表键 frames/data 展开）。
                main_table = _data_access.read_stream_full(main_table_path, "json")
            elif ext == ".jsonl":
                # 经统一 reader（JSONL 逐行解析，lines=True）。
                main_table = _data_access.read_stream_full(main_table_path, "jsonl")
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
    for p_str in _sniffing.probe_full_paths(probe, "cals"):
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
    for p_str in _sniffing.probe_full_paths(probe, "videos"):
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
    for p_str in _sniffing.probe_full_paths(probe, "audios"):
        audio_meta.append({"file": p_str, "format": Path(p_str).suffix.lstrip(".").lower()})
    for p_str in _sniffing.probe_full_paths(probe, "images"):
        image_meta.append({"file": p_str, "format": Path(p_str).suffix.lstrip(".").lower()})

    caps_result = _sniffing.build_capabilities(probe, table_sniffs)
    caps_result["capabilities"]["has_calibration"] = calib_detected

    # 流配对规则（mp4↔metainfo、accel+gyro=六轴IMU）。视频帧数（ffprobe 可用时）用于
    # metainfo 配对的行数匹配，避免把数据表当视频曝光时间戳造成假漂移。
    video_frame_counts: dict[str, int] = {}
    for v in video_meta:
        nf = v.get("nb_frames")
        if isinstance(nf, int) and nf > 0:
            video_frame_counts[v.get("file", "")] = nf
    stream_pairs = _sniffing.pair_streams(
        _sniffing.probe_full_paths(probe, "videos"),
        _sniffing.probe_full_paths(probe, "tables"),
        _sniffing.probe_full_paths(probe, "audios"),
        video_frame_counts=video_frame_counts,
    )
    stream_pairs.extend(_sniffing.detect_episode_mirrors(probe))

    # LeRobot 元数据：确定性解析 meta/info.json（fps / features 列语义 / hand_tracked /
    # robot_type / coordinate_frame / task / total_frames / source）与 meta/stats.json
    # （每列统计量，直接采用不重算）。解析结果供 profile_data 等引用维度名，
    # 避免"疑为/未验证"式推测。
    lerobot_info: dict[str, Any] = {}
    lerobot_stats: dict[str, Any] = {}
    dataset_metadata: list[str] = []
    if _sniffing.detect_dataset_format(probe) == "lerobot":
        meta_files = [
            p for p in (
                _sniffing.probe_full_paths(probe, "cals")
                + _sniffing.probe_full_paths(probe, "tables")
            )
            if _sniffing._is_dataset_metadata_file(p)
        ]
        info_path = next((p for p in meta_files if Path(p).name == "info.json"), None)
        if info_path:
            lerobot_info = _sniffing.parse_lerobot_info(info_path)
        stats_path = next((p for p in meta_files if Path(p).name == "stats.json"), None)
        if stats_path:
            lerobot_stats = _sniffing.parse_lerobot_stats(stats_path)
        dataset_metadata = meta_files
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
        # 流配对规则结果（mp4↔metainfo、accel+gyro=六轴IMU、episode mirror）。
        "stream_pairs": stream_pairs,
        # LeRobot 语义元数据：info.json（fps/列语义/hand_tracked/robot_type/
        # coordinate_frame/task/total_frames/source）与 stats.json（每列统计量）。
        "lerobot_info": lerobot_info,
        "lerobot_stats": lerobot_stats,
        "dataset_metadata": dataset_metadata,
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

    # 嵌套字段发现（信封型 JSONL/JSON）：把 data 内嵌的时间候选与信号字段
    # 附加到流登记表（确定性，读前 5 行；单文件失败静默降级不阻塞加载）。
    _attach_nested_discovery(meta["streams"])

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

    # 目录内含 .h5（此前被普查归入 others 丢弃——真实事故：wujiGlove 同期
    # 采集目录的 dataset.h5 45.8MB 完全不可见）：把其数据节点登记为流，
    # 复用 h5 节点流机制（按节点名切换分析）。
    for other in probe_full_paths(probe, "others"):
        if Path(other).suffix.lower() in (".h5", ".hdf5"):
            try:
                register_h5_node_streams(context, Path(other))
            except Exception:  # noqa: BLE001 - 单个 h5 登记失败不阻塞目录加载
                pass

    file_survey: dict[str, Any] = {
        "total_files": probe["total_files"],
        "ext_dist": probe["ext_dist"],
        "subdirs": probe["subdirs"][:20],
        # 文件清单按类型分组。计数与分类完整（每个文件都归类），但单组路径最多列
        # max_listed_per_group 条并标注 truncated——不静默抽样，也不把数万条路径
        # 灌进上下文（曾导致超百万 token 触发模型上下文超限）。
        "max_listed_per_group": probe.get("max_listed_per_group"),
        "excluded_dirs": probe.get("excluded_dirs", []),
        "tables": probe["tables"],
        "videos": probe["videos"],
        "audios": probe["audios"],
        "images": probe["images"],
        "cals": probe["cals"],
        "others": probe["others"],
    }
    # 体积护栏：极端情况下（如含超长路径的巨型目录）压缩为分组计数摘要，
    # 保证返回不会撑爆上下文。
    survey_note = _compress_file_survey(file_survey)
    if survey_note:
        file_survey["note"] = survey_note

    result: dict[str, Any] = {
        "success": True,
        "dataset_id": dataset_id,
        "kind": "directory",
        "file_survey": file_survey,
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
    # 透明告知：默认排除了版本控制/缓存等非数据目录。
    excluded = probe.get("excluded_dirs") or []
    if excluded:
        msg_parts.append(
            f"已跳过 {len(excluded)} 个非数据目录（如 .git/__pycache__/node_modules 等，"
            "详见 file_survey.excluded_dirs）。"
        )
    # 透明告知：清单被截断/压缩（不静默）。
    truncated_groups = [
        k for k in ("tables", "videos", "audios", "images", "cals", "others")
        if isinstance(probe.get(k), dict) and probe[k].get("truncated")
    ]
    if truncated_groups:
        msg_parts.append(
            f"文件清单中 {len(truncated_groups)} 个分组超过展示上限"
            f"（每组最多 {probe.get('max_listed_per_group')} 条路径），已截断但计数完整，"
            "详见 file_survey 各组的 total/shown/truncated。"
        )
    if file_survey.get("note"):
        msg_parts.append(file_survey["note"])
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
        elif ext == ".jsonl":
            # JSONL：lines=True（每行一个对象）。与 .json 严格区分，不得混用。
            df = pd.read_json(path, lines=True, encoding=_detect_encoding(source.read_bytes()))
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".h5":
            df = _load_hdf5(path)
        else:  # pragma: no cover - 防御性分支
            raise ValueError(f"未实现格式：{ext}")
    except MissingDependencyError as exc:
        # 缺可选依赖 ≠ 文件损坏：文件内容完全未被读取，user_message 不得使用
        # "可能损坏"兜底措辞（真实事故：用户被误导怀疑文件损坏，实际只是环境
        # 缺 pytables）。给出可执行的修复指令。
        return _error(
            "missing_dependency",
            f"读取 {path} 失败：环境缺少依赖（{exc}）",
            exc.user_hint(),
            supported_formats=supported,
        )
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
    # h5 原生层级文件：结构摘要（全部数据节点清单）随主表透出——用户可知
    # 该文件内还有哪些表（主表为信息量最大节点）。
    h5_structure = (
        df.attrs.get("h5_structure") if ext == ".h5" and hasattr(df, "attrs") else None
    )
    meta: dict[str, Any] = {
        "source": path,
        "format": ext.lstrip("."),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
    }
    if h5_structure:
        meta["h5_structure"] = h5_structure
        meta["h5_source_node"] = df.attrs.get("h5_source_node")
        # h5 原生层级：节点登记延后到 context.meta 赋值后（meta 与
        # context.meta 同引用，先赋值再追加才会落到会话 meta 上）。
        meta["h5_node_pending"] = True
    context.meta = meta
    if meta.pop("h5_node_pending", None):
        register_h5_node_streams(context, source)

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
    if h5_structure:
        result["user_message"] += (
            f" 该文件为 HDF5 原生层级结构，主表为节点 {meta['h5_source_node']}，"
            f"共含 {len(h5_structure)} 个数据节点（其余节点清单见 h5_structure 字段）。"
        )
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
    - 单文件：按格式读取（.csv / .json / .jsonl / .parquet / .h5）；
      .jsonl 为 JSON Lines（每行一个 JSON 对象），与 .json 分别处理；
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
