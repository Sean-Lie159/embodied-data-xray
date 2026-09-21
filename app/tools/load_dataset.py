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

import numpy as np
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
    ".mcap": "MCAP 容器（JSON 编码消息；单文件多 topic）",
    # 纯文本数据（2026-09-14 起）：``.txt`` 为"数值首列 + 等宽列"的时间戳清单
    # （真实形态：相机逐帧时间戳），``.INFO``/``.log`` 为 glog 风格日志。
    ".txt": "纯文本数据清单（数值首列 + 等宽列，如逐帧时间戳）",
    ".info": "运行日志（glog 风格：[IWEF]MMDD hh:mm:ss.uuuuuu ...）",
    ".log": "运行日志（glog 风格）",
}

# 尝试解码文本文件时使用的编码回退链。
_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

# 路径规范化：需剥离的**包裹字符**（成对出现，如用户粘贴时带上的引号）。
_PATH_WRAPPERS: tuple[tuple[str, str], ...] = (
    ('"', '"'), ("'", "'"), ("`", "`"),
    ("\u201c", "\u201d"),  # 中文左右双引号 “ ”
    ("\u2018", "\u2019"),  # 中文左右单引号 ‘ ’
    ("\uff02", "\uff02"),  # 全角双引号 ＂
    ("\u300c", "\u300d"),  # 直角引号 「 」
)

# 路径规范化：需从**首尾**剥离的污染字符集合（引号、空白、不可见字符、中文标点）。
# 背景（真实事故 2026-09-10）：侧栏 text_input 粘贴的路径常带引号——用户从对话
# 记录或文档里复制 "C:\...\x.mcap"（含引号）时，引号被当作路径的一部分，
# Path.exists() 返回 False，弹出"文件不存在，请检查路径"的误导性提示；而同样
# 的路径在**对话里**输入却能加载成功，因为模型生成工具参数时会自动剥掉引号。
# 两条路径入参规范不一致 → "偶发"（只在用户粘了引号时发生）。
_PATH_POLLUTION_CHARS: str = (
    '"\'`'                    # ASCII 引号与反引号
    "\u201c\u201d\u2018\u2019"  # 中文左右双/单引号
    "\uff02\uff07"            # 全角双/单引号
    "\u300c\u300d\u300e\u300f"  # 直角引号与双直角引号
    "\uff0c\u3002\uff1b\uff1a"  # 全角逗号、句号、分号、冒号
    " \t\r\n\u00a0\u3000"     # 空白（含不换行空格与全角空格）
    "\u200b\u200c\u200d\ufeff"  # 零宽空格/连接符/BOM
)


def _clean_cell(value: str) -> str:
    """去掉单个值首尾的引号、成对引号与尾随标点（CSV 单元格清洗辅助）。"""
    out = value
    for open_q, close_q in _PATH_WRAPPERS:
        if out.startswith(open_q) and out.endswith(close_q) and len(out) > 1:
            out = out[len(open_q):-len(close_q)]
    return out.strip()


def normalize_path_spec(raw: str) -> str:
    """规范化用户/界面传入的路径字符串（**幂等**，只动首尾、不动路径本体）。

    为什么需要：路径的两个入口（侧栏 text_input 与对话中模型抽取的路径）此前
    规范不一致——模型会剥掉引号，而粘贴的内容原样透传。用户从对话记录里复制
    ``"C:\\...\\x.mcap"``（含引号）粘贴进侧栏，引号成了路径的一部分，
    ``Path.exists()`` 为 False，"偶发"地弹出"文件不存在"（真实事故 2026-09-10）。

    处理顺序（保守，宁可少动）：
      1. 迭代剥离成对包裹的引号（ASCII / 中文 / 全角 / 直角，可能嵌套）；
      2. 去首尾污染字符（引号、空白、零宽字符、尾随中文标点如"。""，"）；
      3. 去掉 ``file://`` / ``file:///`` URL 前缀（浏览器复制路径的常见形态）；
      4. Windows 上把正斜杠统一为反斜杠——仅当盘符形态（如 ``C:/x``）成立时，
         避免误伤 UNC 与已含反斜杠的路径。

    规范化**不保证**路径存在；调用方仍须正常校验并如实报错。返回值只在
    内容确有变化时与原值不同，因此可安全用于"是否需要告知用户已自动纠正"。

    Args:
        raw: 原始路径字符串（可能含引号、空白、URL 前缀等污染）。

    Returns:
        规范化后的路径字符串；输入不是字符串或为空白时原样返回。
    """
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return raw

    # 1) 成对引号（迭代以覆盖 “"path"” 这类嵌套/混合包裹）。
    for _ in range(4):
        before = text
        for open_q, close_q in _PATH_WRAPPERS:
            if (text.startswith(open_q) and text.endswith(close_q)
                    and len(text) > len(open_q) + len(close_q) - 1):
                text = text[len(open_q):-len(close_q)]
        text = text.strip()
        if text == before:
            break

    # 2) 首尾污染字符（引号/空白/零宽/尾随标点）。
    text = text.strip(_PATH_POLLUTION_CHARS)

    # 3) file:// URL 前缀（file:///C:/x → C:/x）。
    low = text.lower()
    for prefix in ("file:///", "file://", "file:/"):
        if low.startswith(prefix):
            text = text[len(prefix):]
            break
    # Windows 常见形态：file:///C:\x（保留盘符前的多余斜杠会被剔除）。
    if len(text) > 2 and text[0] == "/" and text[1].isalpha() and text[2] == ":":
        text = text[1:]

    # 4) 盘符形态的路径统一分隔符（仅 Windows 有意义）。
    if len(text) > 1 and text[1] == ":":
        text = text.replace("/", "\\")

    return text


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
        import h5py  # noqa: F401
    except ImportError:
        return None
    try:
        # **复用统一节点清单**（2026-09-20 真实缺陷修复）。
        #
        # 此前本函数自己遍历 + 自己算行数，**没有**应用 `_merge_frame_layout`
        # 的帧布局合并——于是"每帧一组"布局下它看到的是**单帧行数**（如
        # `0/action/end/orientation` 的 2 行），而流登记表里同名节点已是合并后
        # 的 28270 行。两处口径不一致导致**选错主表**：主表只装载了 14135 帧中
        # 的第 0 帧（`context.df.shape == (2, 4)`），基于它的统计/绘图全部失真。
        #
        # 现改为复用 `_list_hdf5_native_nodes`（含合并），保证"选主表的行数口径"
        # 与"流登记表口径"同源。
        nodes = _list_hdf5_native_nodes(path)
        if not nodes:
            return None
        # 信息量最大者为主表；并列时路径字母序（确定性，由 _list_ 已排序）。
        best = nodes[0]
        best_path = best["node"]
        df = read_hdf5_node(path, best_path)
        if df is None:
            return None
        df.attrs["h5_source_node"] = best_path
        # 结构清单与流登记表同口径（含合并后的行数、帧布局标记）。
        df.attrs["h5_structure"] = [
            {"node": n["node"], "rows": n["rows"], "cols": n["cols"],
             **({"n_frames": n["n_frames"]} if n.get("frame_layout") else {})}
            for n in nodes[:20]
        ]
        return df
    except OSError:
        return None  # 非 HDF5 签名 → 调用方按"可能损坏"兜底（文件确实读过）
    except Exception:  # noqa: BLE001
        return None


# "每帧一组"布局的识别与合并（2026-09-14 真实事故修复）。
#
# 事故：目录内 aligned_joints.h5（513MB）的结构是 14135 个**数字命名的顶层组**
# （0/、1/、2/…），每组下是与帧内各部位对应的同名叶子数据集
# （action/end/orientation、state/end/arm_orientation 等，每个 shape 仅 (2,4) 或
# (14,)）。此前的实现把"每个叶子数据集"都登记为一条独立流 → **84,810 条流**，
# 带来三重后果：
#   1. `context.meta` 常驻 3600 万字符（估算 1200 万 token）；
#   2. `inspect_streams` 返回 4718 万字符，护栏为测量体积要反复序列化（实测 14.5s）；
#   3. 用户在界面上表现为"卡死"（实际是纯计算耗时数十秒到分钟级，非死循环）。
#
# 语义上，这种布局是**同一批字段按帧索引分片**，正确理解是"一组多帧序列"而非
# "N 万条独立流"。故合并：按"去掉顶层帧号后的相对路径"分组，每组登记为一条流，
# 并把帧号作为索引维度记录在 n_frames 上。
#
# 识别条件（必须同时满足，避免误合并正常的多节点文件）：
#   - 顶层条目**全部**是纯数字名（帧号形态）；
#   - 顶层条目数 ≥ _FRAME_LAYOUT_MIN_GROUPS（少于此不值得合并）；
#   - 各帧下相对路径集合一致（同一批字段逐帧重复）。
_FRAME_LAYOUT_MIN_GROUPS = 50
# 合并后单条流记录的帧号样例上限（记录范围而非全部帧号，控制元数据体积）。
_MAX_RECORDED_FRAME_IDS = 20

# 单容器登记的流数硬上限（兜底护栏，2026-09-14）。
# 正常容器：h5 节点数在个位到数十；MCAP topic 数十到数百（实测 26）。
# 取 500 留足余量，同时把"异常布局导致上万条流"这类事故挡在源头。
_MAX_STREAMS_PER_CONTAINER = 500

# 帧布局下扫描的样本帧数（2026-09-14 性能事故）。
# 全量遍历 88 万节点耗时约 57 秒；帧布局下各帧结构按语义必然一致，
# 扫 3 帧即可完整还原字段清单，把成本从 O(全部节点) 降到 O(3 帧)。
_FRAME_SCAN_SAMPLE_FRAMES = 3


def _is_frame_number(name: str) -> bool:
    """判断顶层条目名是否为帧号（纯数字，可带前导零）。"""
    return str(name).strip().isdigit()


def _rel_node_path(node: str) -> str:
    """取节点路径去掉首个（帧号）段后的相对路径。"""
    s = str(node)
    return s.split("/", 1)[1] if "/" in s else ""


def _finalize_frame_layout(
    sampled: list[dict[str, Any]], all_top_keys: list[str]
) -> list[dict[str, Any]]:
    """把"抽样帧扫描结果"整理为全部帧的合并条目（供 :func:`_merge_frame_layout`）。

    背景：为避开 88 万节点的全量遍历（约 57 秒），帧布局只扫描前
    ``_FRAME_SCAN_SAMPLE_FRAMES`` 帧。本函数以**首帧**的字段清单为准，
    并把样本条目的帧号前缀替换为"全部帧数"，避免中间态占用内存。

    **诚实标注**：抽样各帧的字段集合若不一致，字段清单只能以首帧为准——
    此时在每条目上标 ``inconsistent_frames=True`` 并给出不一致的样例帧号，
    供用户判断（不静默掩盖）。

    Args:
        sampled: 抽样帧扫描出的条目（node 形如 ``0/action/end/position``）。
        all_top_keys: 文件的所有顶层键（帧号）。

    Returns:
        整理后的条目列表：每条对应一个字段（相对路径），``n_frames`` 为全部帧数。
    """
    frame_ids = sorted(all_top_keys, key=int)
    if not frame_ids:
        return []
    first = frame_ids[0]
    # 按帧分组抽样结果，用于核对字段集合是否一致。
    by_frame: dict[str, set[str]] = {}
    for e in sampled:
        node = str(e.get("node", ""))
        head, _, rel = node.partition("/")
        if rel:
            by_frame.setdefault(head, set()).add(rel)
    base_fields = by_frame.get(first, set())
    inconsistent = any(fields != base_fields for fields in by_frame.values())
    bad_frames = [f for f, fields in by_frame.items() if fields != base_fields] \
        if inconsistent else []

    out: list[dict[str, Any]] = []
    for e in sampled:
        node = str(e.get("node", ""))
        head, _, rel = node.partition("/")
        if head != first or not rel:
            continue
        entry: dict[str, Any] = {
            "node": f"{first}/{rel}",   # 占位前缀，合并时按 rel 折叠
            "rows": int(e.get("rows") or 0),
            "cols": int(e.get("cols") or 0),
            "fields": e.get("fields", []),
            "n_frames": len(frame_ids),
            "frame_ids_sample": frame_ids[:_MAX_RECORDED_FRAME_IDS],
            "frame_layout": True,
        }
        if inconsistent:
            entry["inconsistent_frames"] = True
            entry["inconsistent_sample"] = bad_frames[:_MAX_RECORDED_FRAME_IDS]
        out.append(entry)
    return out


def _merge_frame_layout(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把"每帧一组"布局的节点清单合并为按字段分组的少量流。

    Args:
        entries: 原始节点清单（每条含 node / rows / cols / fields）。

    Returns:
        合并后的条目清单；不符合帧布局特征时**原样返回**（零回归）。
        合并条目额外带 ``n_frames`` / ``frame_ids_sample`` / ``frame_layout``。
    """
    # 已完成抽样整理的条目（带 frame_layout 标记）：按相对路径折叠，行数 × 帧数。
    # 注意：此分支必须在"条目数下限"检查**之前**——抽样整理后的条目数等于
    # 字段数（本例 6 条），远小于 _FRAME_LAYOUT_MIN_GROUPS，若先做下限检查会
    # 直接返回未合并的条目（真实踩坑：路径仍带 "0/" 前缀、n_rows 未乘帧数）。
    if entries and all(e.get("frame_layout") for e in entries):
        merged_sampled: list[dict[str, Any]] = []
        for e in entries:
            rel = _rel_node_path(e.get("node", ""))
            if not rel:
                continue
            n_frames = int(e.get("n_frames") or 1)
            item: dict[str, Any] = {
                "node": rel,
                "rows": int(e.get("rows") or 0) * n_frames,
                "cols": int(e.get("cols") or 0),
                "fields": e.get("fields", []),
                "n_frames": n_frames,
                "frame_ids_sample": e.get("frame_ids_sample", []),
                "frame_layout": True,
            }
            # 帧结构不一致的标注必须透传（不得在合并时丢失）。
            for key in ("inconsistent_frames", "inconsistent_sample"):
                if e.get(key):
                    item[key] = e[key]
            merged_sampled.append(item)
        if merged_sampled:
            merged_sampled.sort(
                key=lambda c: (-(c["rows"] * max(1, c["cols"])), c["node"])
            )
            return merged_sampled
        return entries

    if len(entries) < _FRAME_LAYOUT_MIN_GROUPS:
        return entries
    frame_groups: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        head = str(e.get("node", "")).split("/", 1)[0]
        if not _is_frame_number(head):
            return entries  # 有非数字顶层条目 → 不是帧布局，保守不合并
        frame_groups.setdefault(head, []).append(e)
    if len(frame_groups) < _FRAME_LAYOUT_MIN_GROUPS:
        return entries

    first_key = next(iter(frame_groups))
    baseline = {_rel_node_path(e.get("node", "")) for e in frame_groups[first_key]}
    baseline.discard("")
    if not baseline:
        return entries
    # 抽样核对若干帧的路径集合是否一致（全量核对代价高）。
    checked = 0
    for fid, items in frame_groups.items():
        if fid == first_key:
            continue
        if {_rel_node_path(e.get("node", "")) for e in items} - {""} != baseline:
            return entries  # 各帧结构不一致 → 不合并
        checked += 1
        if checked >= 5:
            break

    merged: dict[str, dict[str, Any]] = {}
    for fid, items in frame_groups.items():
        for e in items:
            rel = _rel_node_path(e.get("node", ""))
            if not rel:
                continue
            slot = merged.setdefault(rel, {
                "node": rel,
                "rows": 0,
                "cols": int(e.get("cols") or 0),
                "fields": e.get("fields", []),
                "_frame_ids": [],
            })
            slot["rows"] += int(e.get("rows") or 0)
            if len(slot["_frame_ids"]) < _MAX_RECORDED_FRAME_IDS:
                slot["_frame_ids"].append(fid)
    if not merged:
        return entries

    frame_ids = sorted(frame_groups, key=lambda x: int(x))
    out: list[dict[str, Any]] = []
    for rel, slot in merged.items():
        out.append({
            "node": rel,
            "rows": slot["rows"],
            "cols": slot["cols"],
            "fields": slot["fields"],
            "n_frames": len(frame_ids),
            "frame_ids_sample": slot["_frame_ids"],
            "frame_layout": True,
        })
    # 与未合并路径保持同一排序口径：行×列降序，路径字母序。
    out.sort(key=lambda c: (-(c["rows"] * max(1, c["cols"])), c["node"]))
    return out


def _is_numeric_dtype(dtype: Any) -> bool:
    """判断 h5 叶子的 dtype 是否为"可表格化的数值"（**准入的唯一 dtype 闸门**）。

    为什么需要（2026-09-20 真实缺陷修复）：此前 `_visit` 按**维数**枚举接纳形态
    （compound / 2D / 1D 多元素），把 0 维标量与单元素数组一律丢弃——于是真实
    数据集 aligned_joints.h5 每帧的 ``main_timestamp``（``shape=()`` uint64）
    与 ``timestamp/camera/*``（``shape=(1,)`` uint64）**从未进入流登记表**，
    下游 `inspect_streams` / `align_container_streams` 因此对 27 条已登记流
    如实报"无时间戳"，agent 只能回答"h5 里没有时间戳"——而时间戳就在那里。

    改为按 dtype 判定后，接纳规则不再与维数耦合，但必须**同时收紧 dtype**：
    否则 ``state/end/errmsg``（``dtype=object``、``shape=(1,)`` 的字符串字段）
    会被一并纳入，把错误消息当数值流登记（新噪声）。

    Args:
        dtype: h5py 叶子的 dtype。

    Returns:
        True 表示数值型（含整型/浮点/布尔）；object / str / bytes 等返回 False。
    """
    import numpy as _np

    try:
        if dtype is None:
            return False
        names = getattr(dtype, "names", None)
        if names:
            # compound（结构化）dtype 的 kind 是 "V"（void），**不是**数值——
            # 若在此一律拒绝，会把项目既有的 compound 节点（动作流等）全部丢掉
            # （实测踩坑：本函数初版即把 27 条流里的全部 compound 误杀）。
            # compound 的正确判据是"**至少有一个**数值字段"（真实形态如
            # ``[("value","<f4"),("timestamp","<f8")]``；纯字符串字段的
            # compound 仍应拒绝）。
            return any(
                _is_numeric_dtype(dtype.fields[n][0]) for n in names
            )
        if dtype == object or dtype.kind in ("O", "U", "S", "V"):
            return False
        return bool(
            _np.issubdtype(dtype, _np.number) or _np.issubdtype(dtype, _np.bool_)
        )
    except Exception:  # noqa: BLE001
        return False


def _classify_h5_leaf(
    shape: tuple[int, ...], dtype: Any, frame_layout: bool
) -> dict[str, Any] | None:
    """判定一个 h5 叶子能否作为"表"，并给出 (rows, cols, fields)。

    **这是 h5 节点接纳的唯一判据**（2026-09-20 重构）：此前 `_visit` 内联按
    维数枚举，``read_hdf5_nodes_metadata`` 又各写一套列名/行数推导，两处口径
    必须手工保持同步——这次漏掉"0 维标量 / 单元素数组"正是因为**两处都漏**，
    且谁也没有错到显眼。收敛为一处后，新增形态只需改本函数。

    接纳形态与行列语义（``shape`` 为**单帧内**的形状）：

    ====================  ==================  ==========================
    形态                 非帧布局            帧布局（每帧一组）
    ====================  ==================  ==========================
    compound             rows=shape[0]       rows=shape[0]（每帧多行）
                         cols=len(names)     cols=len(names)
    2D 数值              rows=shape[0]       rows=shape[0]（如 (2,4) 双手机）
                         cols=shape[1]       cols=shape[1]
    1D 多元素 (N>1)      rows=shape[0]       rows=1
                         cols=1              cols=shape[0]
    1D 单元素 (N==1)     rows=1, cols=1      rows=1, cols=1
    0 维标量             rows=1, cols=1      rows=1, cols=1
    ====================  ==================  ==========================

    标量/单元素按"**每帧 1 行 × 1 列**"理解，而非 0 行或 1 行 N 列表头：
    ``main_timestamp`` 的语义是"每帧一个时刻观测"，与 ``action/joint/position``
    （每帧 1 行 × 14 列）同级，只是宽度为 1。这样 `_merge_frame_layout` 的
    "行数 × 帧数"折叠规则与 `_read_frame_layout_node` 的按帧拼接都能直接复用，
    无需为标量开特例。

    Args:
        shape: 叶子的 shape（已转为 tuple[int, ...]）。
        dtype: 叶子的 dtype。
        frame_layout: 是否为"每帧一组"布局（影响 1D 的行列语义）。

    Returns:
        {"rows", "cols", "fields"}；不接纳（非数值 / 空维度）返回 None。
    """
    # dtype 闸门：非数值一律不接纳（排除字符串/object/嵌套）。
    if not _is_numeric_dtype(dtype):
        return None

    names = getattr(dtype, "names", None)
    if names:
        # compound：字段名即列名；每帧含 shape[0] 行。
        if len(shape) < 1:
            return None
        return {"rows": int(shape[0]), "cols": len(names), "fields": list(names)}

    ndim = len(shape)
    if ndim == 0:
        # 0 维标量（如 main_timestamp）：每帧一个观测。
        return {"rows": 1, "cols": 1, "fields": []}
    if ndim == 1:
        n = int(shape[0])
        if n == 0:
            return None  # 空数组不构成表
        if n == 1:
            # 单元素数组（如 timestamp/camera/* 的 (1,)）：每帧一个标量观测。
            return {"rows": 1, "cols": 1, "fields": []}
        # 多元素 1D：语义取决于布局（见函数 docstring）。
        if frame_layout:
            return {"rows": 1, "cols": n, "fields": []}
        return {"rows": n, "cols": 1, "fields": []}
    if ndim == 2:
        return {"rows": int(shape[0]), "cols": int(shape[1]), "fields": []}
    # 3 维及以上：不对应二维表，不接纳（如标定矩阵/点云张量）。
    return None


def _list_hdf5_native_nodes(path: str) -> list[dict[str, Any]]:
    """列出 h5py 原生层级文件的全部候选数据节点（供流登记）。

    候选判据收敛在 :func:`_classify_h5_leaf`（唯一准入点）：compound、
    2D 数值、1D 多元素、**1D 单元素**、**0 维数值标量**；非数值 dtype
    （object/字符串）与 3 维以上一律不接纳。

    历史上这里按维数枚举 `if compound / elif ndim==2 / elif ndim==1 and
    shape[0]>1 / else 丢弃`，导致 0 维标量与单元素数组被静默丢弃——真实后果是
    aligned_joints.h5 每帧的 ``main_timestamp`` 与 6 路 ``timestamp/camera/*``
    从未进入流登记表，agent 据此回答"h5 里没有时间戳"。

    **"每帧一组"布局会被合并**（见 :func:`_merge_frame_layout`）：真实数据集
    aligned_joints.h5 的 14135 帧 × 6 字段会被登记为 6 条流而非 84,810 条——
    否则 `context.meta` 常驻上千万 token、下游工具返回上千万字符，表现为界面卡死。

    Args:
        path: 文件路径。

    Returns:
        节点清单 [{node, rows, cols, fields}]，按 行×列 降序（主表在首）；
        帧布局合并后的条目另带 n_frames / frame_ids_sample / frame_layout。
        非 HDF5/h5py 不可用返回 []。
    """
    try:
        import h5py
    except ImportError:
        return []
    try:
        out: list[dict[str, Any]] = []
        with h5py.File(path, "r") as f:
            # 早剪枝（2026-09-14 性能事故）：真实数据集 88 万节点时，
            # 全量 visititems 耗时约 57 秒（为每个节点构造 Python 包装对象），
            # 用户侧表现为卡死。这里先探测**是否为"每帧一组"布局**（顶层全数字
            # 且数量很大），若是则只扫描前若干帧——帧间结构按该布局的语义必然
            # 一致，扫 3 帧即可完整还原字段清单，成本从 O(全部节点) 降到 O(帧数)。
            top_keys = list(f.keys())
            frame_layout = (
                len(top_keys) >= _FRAME_LAYOUT_MIN_GROUPS
                and all(_is_frame_number(k) for k in top_keys)
            )
            scan_roots: list[str] = []
            if frame_layout:
                scan_roots = sorted(top_keys, key=int)[:_FRAME_SCAN_SAMPLE_FRAMES]
            else:
                scan_roots = [""]  # 空串 = 扫描全树

            def _scan(root_node: Any, prefix: str) -> None:
                def _visit(name: str, node: Any) -> None:
                    if not isinstance(node, h5py.Dataset):
                        return
                    # 形状/dtype 判定收敛到唯一入口（新增形态只改 _classify_h5_leaf）。
                    shape = tuple(int(x) for x in node.shape)
                    info = _classify_h5_leaf(shape, node.dtype, frame_layout)
                    if info is None:
                        return
                    full = f"{prefix}/{name}" if prefix else name
                    out.append({
                        "node": full,
                        "rows": info["rows"],
                        "cols": info["cols"],
                        "fields": info["fields"],
                        "ndim": len(shape),
                    })
                root_node.visititems(_visit)

            for root in scan_roots:
                _scan(f[root] if root else f, root)
        # "每帧一组"布局合并（真实事故：14135 帧 × 6 字段 → 84,810 条流）。
        if frame_layout:
            # 抽样帧扫描出的条目：把帧号前缀补成完整帧数，供合并统计。
            out = _finalize_frame_layout(out, top_keys)
        out = _merge_frame_layout(out)
        out.sort(key=lambda c: (-(c["rows"] * max(1, c["cols"])), c["node"]))
        return out
    except Exception:  # noqa: BLE001
        return []


def _classify_h5_node(fields: list[str], node_path: str) -> tuple[str, str]:
    """按节点字段特征判定 kind 与语义标签（确定性，不硬猜语义之外的）。

    Args:
        fields: compound 字段名清单（2D/1D/标量数值节点为空）。
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
    # 时间戳流（2026-09-20 新增）：帧内标量时间戳节点的 fields 为空，
    # 只能靠路径判定。真实形态：
    #   - ``main_timestamp``（帧内 0 维标量，主时钟）
    #   - ``timestamp/camera/<name>``（帧内 (1,) 标量，各相机采集时刻）
    # 若不识别，这 7 条流会全部落到 unknown → 触发 UI 的"未分类"提示，
    # 用户看到一堆"未知（无法分类）"反而更困惑。
    # **必须是"时刻"语义**：排除 time_diff/delta 这类差分量（_is_timestamp_like_field
    # 词表已含 diff/delta 排除规则，此处复用同一判据，避免两套口径）。
    leaf_name = node_path.rsplit("/", 1)[-1]
    if "action" not in path_l and "state" not in path_l and "calibration" not in path_l:
        if is_timestamp_like_field(leaf_name) or is_timestamp_like_field(node_path):
            if "camera" in path_l:
                return "timestamp_index", "相机采集时刻（逐帧时间戳）"
            return "timestamp_index", "主时钟时间戳（逐帧）"
    if any("orientation" in f or "accel" in f for f in fl) or "imu" in path_l:
        return "imu", "IMU 传感器"
    if "pose" in path_l or any("quat" in f for f in fl):
        return "pose", "位姿流"
    if "calibration" in path_l or _is_calibration_extrinsic(node_path):
        return "calibration", _calibration_label(node_path)
    # state/* 遥测流的路径判定（2026-09-21，缺陷 A 修复）。
    #
    # 为什么必须在这里单独判：帧布局下 state/* 是**裸 ndarray**，
    # `_classify_h5_leaf` 为其返回 ``fields=[]``，于是上面所有依赖 ``fl``
    # 的判据（``any("quat" in f ...)`` 等）**恒为 False**；而 state 路径
    # 又不含 ``action``/``pose``/``imu`` 等既有词根 → 24 条真实遥测流
    # （joint/position 14135×14、end/wrench 14135×12、waist/effort 14135×5、
    # head/mode、end/errcode …）全部落到 unknown。
    #
    # 反证：同数据的 ``action/joint/position`` 因路径含 ``action`` 被正确
    # 识别——同类数据一半识别一半不识别，属明确逻辑漏洞。
    #
    # **注意**：这些标签是**基于路径的语义假设**（label_confidence=medium），
    # 仅供用户/模型理解清单，**不代表该流已被验证**，不得用于数值计算
    # （AGENTS.md §3.4）。该约束同时写入 label_evidence 文案。
    if "state" in path_l:
        state_kind = _classify_state_leaf(node_path)
        if state_kind is not None:
            return state_kind
    return "unknown", "未知（无法分类）"


# 相机标定文件的命名惯例（2026-09-21，缺陷 B 修复）。
#
# 背景：真实数据集有 5 条 ``extrinsic_end_T_<camera>_rgbd_aligned.json``，
# 是"末端 → 相机"的变换矩阵（相机外参），语义明确，但路径**不含**
# ``calibration`` 字样 → 既不命中 calibration 分支，也不属空流/日志，
# 最终标为"未知（无法分类）"。工具 evidence 自承"判不出，未做硬猜"。
#
# 判据按命名惯例（整词/前缀，避免子串误命中）：
_CALIB_EXTRINSIC_TOKENS = ("extrinsic",)
_CALIB_INTRINSIC_TOKENS = ("intrinsic",)


def _is_calibration_extrinsic(node_path: str) -> bool:
    """判断路径是否为相机标定文件（内参/外参命名惯例）。

    Args:
        node_path: 节点路径或文件名。

    Returns:
        True 表示按命名惯例属标定文件。
    """
    low = node_path.lower()
    if any(t in low for t in _CALIB_EXTRINSIC_TOKENS):
        return True
    if any(t in low for t in _CALIB_INTRINSIC_TOKENS):
        return True
    # ``<frame>_T_<frame>`` 是变换矩阵的通行命名（如 end_T_hand_left_rgbd）。
    # 用 split 后的整词 ``t`` 匹配，避免误伤含字母 t 的普通名字。
    return "_t_" in low


def _calibration_label(node_path: str) -> str:
    """按命名惯例给出更精确的标定标签（外参/内参/通用）。

    Args:
        node_path: 节点路径或文件名。

    Returns:
        语义标签。
    """
    low = node_path.lower()
    if any(t in low for t in _CALIB_EXTRINSIC_TOKENS) or "_t_" in low:
        return "相机外参（变换矩阵）"
    if any(t in low for t in _CALIB_INTRINSIC_TOKENS):
        return "相机内参"
    return "标定数据"


# state/* 叶子名 → (kind, 语义标签) 的确定性映射（2026-09-21，缺陷 A）。
#
# 顺序敏感：**先匹配更具体的语义词**（wrench/effort/mode 等），再落到通用的
# 位置/速度。否则 ``state/end/arm_position`` 会被通用规则先吃掉。
#
# kind 取值刻意复用下游已识别的 ``pose``/``force``，并为状态类新增
# ``joint_state``/``effort``/``status``——这三个在 inspect_streams 的
# 分流里落入 ``other_streams``（不进 force 单槽），不会覆盖既有语义。
_STATE_LEAF_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    # **力/力矩不归入 ``force``**（2026-09-21 实测修正）：``kind="force"`` 在
    # inspect_streams 里是**单槽汇总**（``force_stream`` 单个 dict，后写覆盖先写），
    # 而 h5 的 ``state/end/wrench`` 与 ``state/left_effector/wrench`` 等可能有多条
    # ——归入 force 会让前几条被静默覆盖，且该槽的 ``channels`` 期望列名清单，
    # 而帧布局裸数组的 channels 恒为空 → 汇总出 ``present=True, n_channels=0``
    # 的自相矛盾结果（实测已复现）。故用独立 kind ``wrench``，在 inspect_streams
    # 里与 joint_state/effort/status 同走 other_streams（列表，不覆盖）。
    (("wrench", "torque"), "wrench", "力/力矩流"),
    # 关节力矩/电流。
    (("effort", "current", "motor_current"), "effort", "关节力矩/电流流"),
    # 控制器状态码/错误码（低信息量但语义重要：mode / errcode / status）。
    (("errcode", "errmsg", "error_code", "status", "mode", "state_code",
      "controller_state"), "status", "控制器状态流"),
    # 位姿（含 orientation / arm_position 这类末端位姿）。
    (("orientation", "pose", "arm_position", "eef_position", "tcp"), "pose", "位姿流"),
    # 关节状态（位置/速度/加速度）。
    (("position", "qpos", "velocity", "qvel", "acceleration", "qacc", "joint"),
     "joint_state", "关节状态流"),
)


def _classify_state_leaf(node_path: str) -> tuple[str, str] | None:
    """按叶子名判定 ``state/*`` 节点的 kind 与语义标签。

    参数是 **路径**（而非 fields）——因为帧布局下 state 节点的 fields 恒为空，
    只能靠命名判定。

    Args:
        node_path: 节点路径（如 state/joint/position）。

    Returns:
        (kind, semantic_label)；无法归类时返回 None（交由调用方标 unknown）。
    """
    leaf = node_path.rsplit("/", 1)[-1].lower()
    parts = [p for p in node_path.lower().split("/") if p]
    for keys, kind, label in _STATE_LEAF_RULES:
        if any(k in leaf for k in keys):
            return kind, label
    # 叶子名未命中时，用中间路径段兜底（如 state/waist/<未知名>）。
    for seg in reversed(parts[:-1]):
        if seg in ("state",):
            continue
        for keys, kind, label in _STATE_LEAF_RULES:
            if any(k in seg for k in keys):
                return kind, label
    return None


def read_hdf5_node_field_fast(
    path: str, node: str, field: str
) -> np.ndarray | None:
    """**只读单个字段**的列值（不构建整表、不做列名语义化）。

    为什么需要（2026-09-18 性能事故）：`inspect_streams` 为每条流实测采样率，
    只需该流的时间戳列；而 :func:`read_hdf5_node` 会把"每帧一组"布局的 14135
    个帧组的**全部字段**读出来并拼成 DataFrame（实测单节点约 4.2 秒），随后
    只取其中一列。本函数仍须遍历各帧（数据按帧分散存放，无法避免），但
    **不构造 DataFrame、不做列名映射、不 concat**，省去建表开销。

    字段解析规则（与 :func:`read_hdf5_node` 的列名口径一致）：
    - 字段名形如 ``<leaf>_<i>``（如 ``position_3``）→ 取该帧 1D/2D 数据的第 i 个
      分量（1D 取 [i]；2D 扁平化后取 [i]）；
    - 字段名恰为叶子名（如 ``timestamp``）→ 尝试 compound 的该字段名；失败则
      视作单列数值数组的第 0 分量。

    Args:
        path: h5 文件路径。
        node: 节点名（合并节点用相对路径，如 ``action/joint/position``）。
        field: 目标字段名。

    Returns:
        值数组（float，NaN 已剔除非数值项）；取不到返回 None。
    """
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            top = list(f.keys())
            frame_layout = (
                len(top) >= _FRAME_LAYOUT_MIN_GROUPS
                and all(_is_frame_number(k) for k in top)
            )
            values: list[float] = []
            if frame_layout:
                roots = sorted((k for k in top if _is_frame_number(k)), key=int)
            else:
                roots = [None]

            # 字段 → 取值方式：判定是"复合字段名"还是"分量索引"。
            compound_field: str | None = None
            comp_index: int | None = None
            if "_" in field:
                suffix = field.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    comp_index = int(suffix)
                else:
                    compound_field = field
            else:
                compound_field = field

            for root in roots:
                container = f[root] if root is not None else f
                if not isinstance(container, h5py.Group):
                    continue
                leaf = container.get(node)
                if leaf is None or not isinstance(leaf, h5py.Dataset):
                    continue
                if compound_field and leaf.dtype.names:
                    if compound_field not in leaf.dtype.names:
                        return None
                    data = leaf[compound_field]
                    num = np.asarray(data, dtype=float).reshape(-1)
                else:
                    data = leaf[()]
                    arr = np.asarray(data)
                    if comp_index is not None:
                        # **按"行内列索引"取值**（与读取口径一致）：1D 帧数据
                        # (N,) 视为「1 行 × N 列」，取 [k]；2D 帧数据 (R, C)
                        # 视为「R 行 × C 列」，取**第 k 列的全部 R 行**。
                        # 不能按扁平化索引取——真实缺陷：orientation 的 (2,4)
                        # 扁平化后第 0 个元素只对应 (0,0)，会丢掉 (1,0)，
                        # 导致该列行数少一半（28270 vs 14135）。
                        if arr.ndim <= 1:
                            flat = arr.reshape(-1)
                            if comp_index >= flat.size:
                                return None
                            # **1D 的分量语义取决于布局**（2026-09-20）：
                            # 帧布局下 (N,) 是"一帧一条 N 维向量"→ 取第 k 个分量
                            # （单值）；非帧布局下 (N,) 是"一列 N 行序列"→
                            # 按 <leaf>_<i> 命名时列数仅 1，故 k 只可能为 0，
                            # 整体返回（不能只取 1 个值，否则行数塌成 1）。
                            if frame_layout:
                                num = np.asarray([flat[comp_index]], dtype=float)
                            else:
                                if arr.ndim == 1:
                                    num = np.asarray(flat, dtype=float)
                                else:
                                    num = np.asarray([flat[comp_index]], dtype=float)
                        else:
                            if comp_index >= arr.shape[-1]:
                                return None
                            num = np.asarray(arr[..., comp_index], dtype=float).reshape(-1)
                    else:
                        num = np.asarray(arr, dtype=float).reshape(-1)
                values.extend(float(v) for v in num)
            if not values:
                return None
            return np.asarray(values, dtype=float)
    except Exception:  # noqa: BLE001
        return None


def is_timestamp_like_field(name: str) -> bool:
    """判断 h5 节点的**列/字段名**是否像时间戳（用于挑时间轴，不猜语义）。

    为什么需要（2026-09-20 真实缺陷）：`_H5Reader.timestamp()` 此前在没找到名为
    ``timestamp`` 的字段时，会**回退把整个数据集扁平化**当时间戳——于是
    ``left/orientation``（姿态四元数）被当成 29864 个时间戳，产出一串虚假异常。

    判据（确定性，只看命名不看数值）：
    - 命中时间戳词表（``_sniffing._TIMESTAMP_COLS``，全项目唯一来源）；
    - 或名字含时间词根（timestamp / stamp / time / clock）；
    - 或形如 ``<前缀>_ns``/``_us``/``_ms``/``_s`` 且前缀含 ``ts``；
    - **排除"时间差/间隔/时长"类**：它们也是时间量，但**不是时刻**，拿它们当
      时间轴会立刻判出荒谬的采样率（真实案例：``time_diff`` 是 -0.5 的常量差值，
      若被当时间戳会算出 0 Hz 或负间隔）。排除词：diff/delta/interval/duration/
      offset/latency/frame_diff。

    Args:
        name: 列名或字段名。

    Returns:
        True 表示可作为时间轴候选。
    """
    from app.tools._sniffing import _TIMESTAMP_COLS

    low = str(name).lower().strip()
    if not low:
        return False
    # 先排除"时间差"类（含 time 词根但语义是差值，绝非时刻）。
    if any(k in low for k in ("diff", "delta", "interval", "duration",
                              "offset", "latency", "elapsed", "gap")):
        return False
    if low in _TIMESTAMP_COLS:
        return True
    # 词表是"精确名"集合，故再放宽到"包含"（如 mcap_log_time_ns、
    # exposure_start_utc_ns 这类带前后缀的列名）。
    for token in ("timestamp", "time_stamp", "stamp", "time", "clock"):
        if token in low:
            return True
    # 纯单位后缀形态（如 t_ns / ts_us）。
    if low.endswith(("_ns", "_us", "_ms", "_s")):
        return any(k in low for k in ("ts", "t_"))
    # 路径上下文形态（2026-09-20 新增）：h5 的帧内时间戳节点命名可能**不含
    # 任何时间词根**——真实形态 ``timestamp/camera/<相机名>``，叶子名是
    # ``head_color`` / ``hand_left_color`` / ``head_stereo_right`` 这类纯相机名。
    # 但**父级路径**里的 ``timestamp`` 已明确宣告其语义，故按路径判定。
    # 约束：父级须含时间词根，且叶子名不得含动作/状态语义（防误伤
    # ``state/*``、``action/*`` 下的同名叶子）。
    if "/" in low:
        parent = low.rsplit("/", 1)[0]
        if any(t in parent for t in ("timestamp", "time_stamp", "stamp", "clock")):
            if not any(k in low for k in ("action", "state", "command", "errmsg",
                                          "errcode", "effort", "velocity")):
                return True
    return False


def read_hdf5_nodes_metadata(
    path: str, nodes: list[str]
) -> dict[str, dict[str, Any]]:
    """**一次遍历**取齐多个节点的列名与形状（不读数据内容）。

    为什么需要（2026-09-18 性能事故）：`inspect_streams` 要对每条流实测采样率，
    需先知道该流的列名（判断有无时间戳列）。而"每帧一组"布局下，逐节点读取
    都要**重新遍历 14135 个帧组**并构造 Python 包装对象——实测单节点约 4.2 秒，
    27 个节点共 **113 秒**；更糟的是该数据集的节点**全都没有时间戳列**，这 113
    秒全花在"逐个确认没有时间戳"上。

    本函数只取**首帧**的叶子数据集读取列名与形状（帧布局下各帧结构一致，首帧
    即代表全貌），把 O(节点数 × 帧数) 的遍历降为 O(帧数) 一次 + O(节点数) 查表。

    非帧布局（普通层级文件）同样支持：节点名直接对应真实路径，取该数据集的
    dtype/shape 即可，无需遍历。

    Args:
        path: h5 文件路径。
        nodes: 需要元信息的节点名列表（合并节点用相对路径，如
            ``action/joint/position``；普通节点用真实路径）。

    Returns:
        {node: {"columns": [...], "shape": (rows, cols)}}；取不到的节点不在结果中。
        columns 对 1D 数值节点形如 ``["position_0", ...]``（与
        :func:`_read_frame_layout_node` 的命名口径一致）。
    """
    try:
        import h5py
    except ImportError:
        return {}
    wanted = list(dict.fromkeys(nodes))  # 去重保序
    out: dict[str, dict[str, Any]] = {}
    try:
        with h5py.File(path, "r") as f:
            top = list(f.keys())
            frame_layout = (
                len(top) >= _FRAME_LAYOUT_MIN_GROUPS
                and all(_is_frame_number(k) for k in top)
            )
            first_frame = sorted(top, key=int)[0] if frame_layout else None
            n_frames = len(top) if frame_layout else None
            for node in wanted:
                if frame_layout:
                    grp = f[first_frame]
                    leaf = grp.get(node) if isinstance(grp, h5py.Group) else None
                    if leaf is None or not isinstance(leaf, h5py.Dataset):
                        continue
                else:
                    leaf = f.get(node)
                    if leaf is None or not isinstance(leaf, h5py.Dataset):
                        continue
                shape = tuple(int(x) for x in leaf.shape)
                field_name = node.rsplit("/", 1)[-1]
                dtype = leaf.dtype
                # 行列语义与 _list_hdf5_native_nodes **同源**（唯一判据
                # _classify_h5_leaf），避免两处口径各写一套导致漏改
                # （2026-09-20：漏掉 0 维标量与单元素数组正是两处都漏）。
                info = _classify_h5_leaf(shape, dtype, bool(frame_layout))
                if info is None:
                    continue
                ndim = len(shape)
                if dtype.names:
                    cols = [str(n) for n in dtype.names]
                elif ndim == 2:
                    cols = [f"{field_name}_{i}" for i in range(shape[1])]
                elif ndim == 1 and shape[0] > 1:
                    # 1D 数组的语义**取决于布局**（2026-09-20 真实缺陷）：
                    # - 帧布局下，(N,) 是"一帧一条 N 维向量观测" → 1 行 × N 列；
                    # - 非帧布局（普通层级 h5）下，(N,) 是**一列 N 行的时间序列**
                    #   （真实案例：imu_data.h5 的 left/timestamps 是 (7466,)
                    #   的一维时间戳数组，此前被报成"1 行 × 7466 列"，行列颠倒，
                    #   使 agent 误判"IMU 时间戳流仅 1 行、数据异常")。
                    if frame_layout:
                        cols = [f"{field_name}_{i}" for i in range(shape[0])]
                    else:
                        cols = [field_name]
                else:
                    # 0 维标量与 1D 单元素（如 main_timestamp / timestamp/camera/*）：
                    # 单列，列名即字段名（不追加 _0——只有一列且语义就是该字段名，
                    # 追加后缀会让 is_timestamp_like_field("main_timestamp") 失效）。
                    cols = [field_name]

                # 行数：帧布局下"每帧贡献 info['rows'] 行"。
                if not frame_layout:
                    rows = int(info["rows"])
                else:
                    rows = int(info["rows"]) * int(n_frames or 0)
                out[node] = {"columns": cols, "shape": (rows, len(cols))}
    except Exception:  # noqa: BLE001
        return {}
    return out


def pick_container_master_clock(
    path: str, nodes: list[str]
) -> str | None:
    """在容器内挑出**主时钟**节点（供无时间戳的子流按帧序回退）。

    为什么需要（2026-09-21）：真实数据集 ``aligned_joints.h5`` 里 27 条
    action/state 流与 ``main_timestamp`` **共享同一 14135 帧**，事实上可时间
    对齐，但它们自身无时间戳列 → ``align_container_streams`` 报
    ``no_timestamp``，用户拿不到"帧 ↔ 时间"映射。

    选取规则（确定性，不猜；结果必须告知用户）：
        1. 名称为 ``main_timestamp`` 的节点**优先**（语义最明确）；
        2. 否则取**非相机**的时间戳节点（路径含 timestamp 但不含 camera）；
        3. 多个候选时按节点名字典序取首（保证可复现）。

    Args:
        path: 容器文件路径。
        nodes: 候选节点名清单（容器内全部子流）。

    Returns:
        主时钟节点名；无合适候选返回 None。
    """
    def _is_ts_node(n: str) -> bool:
        leaf = n.rsplit("/", 1)[-1]
        return bool(
            is_timestamp_like_field(leaf) or is_timestamp_like_field(n)
        )

    ts_nodes = [n for n in nodes if _is_ts_node(n)]
    if not ts_nodes:
        return None
    # 规则 1：main_timestamp 优先。
    for n in ts_nodes:
        if n.rsplit("/", 1)[-1].lower() == "main_timestamp":
            return n
    # 规则 2：排除相机时间戳（它们是各相机自身时刻，不是全局主时钟）。
    non_camera = [n for n in ts_nodes if "camera" not in n.lower()]
    pool = non_camera or ts_nodes
    # 规则 3：字典序取首（可复现）。
    return sorted(pool)[0]


def read_frame_index_aligned_master_clock(
    path: str,
    sub: str,
    master_node: str,
) -> tuple[np.ndarray | None, str]:
    """按 **frame_index 严格校验**后用容器主时钟给子流配时间轴。

    **四个条件必须同时满足**才回退（见 docs/帧序时间轴回退设计.md §3.1），
    任一不满足即返回 (None, 原因) —— 宁可如实报"无时间轴"，也不做可能错误的
    对齐：

        1. 子流带 ``frame_index`` 列；
        2. 主时钟行数 == 子流行数 == ``frame_index`` 唯一值个数；
           （**不做截断/补齐**——帧数一致必须是真的逐帧对应）
        3. ``frame_index`` 是 0…N-1 连续整数（无重复、无缺口、无乱序）；
        4. 主时钟可读出且长度非零。

    为什么要校验得这么死：帧数相同**可能是巧合同长**，不能据此认定逐帧对应；
    真实数据集里 txt 侧 14140~14142 行 vs h5 侧 14135 帧就不一致。只有
    ``frame_index`` 的连续性 + 严格等长才能证明是同一帧序。

    Args:
        path: 容器文件路径。
        sub: 子流节点名（自身无时间戳列者）。
        master_node: 主时钟节点名。

    Returns:
        (时间戳数组, 依据说明)；不满足条件返回 (None, 不满足的原因)。
    """
    try:
        df = read_hdf5_node(path, sub)
        if df is None or df.empty:
            return (None, "该子流读取失败或为空")
        if "frame_index" not in df.columns:
            return (None, "该子流无 frame_index 列，无法建立帧序对应")

        fi = pd.to_numeric(df["frame_index"], errors="coerce")
        if fi.isna().any():
            return (None, "该子流 frame_index 含非法值")
        fi_arr = fi.to_numpy()
        n_rows = len(fi_arr)
        uniq = np.unique(fi_arr)

        if len(uniq) != n_rows:
            return (None, f"该子流 frame_index 有重复（{n_rows} 行 / "
                          f"{len(uniq)} 个唯一值），帧序不唯一")
        if uniq[0] != 0 or uniq[-1] != n_rows - 1:
            return (None, f"该子流 frame_index 非 0..{n_rows - 1} 连续"
                          f"（实际 {uniq[0]}..{uniq[-1]}）")

        master_df = read_hdf5_node(path, master_node)
        if master_df is None or master_df.empty:
            return (None, f"主时钟节点 {master_node} 读取失败")
        # 主时钟单列（剥离 frame_index 辅助列）。
        cand = [c for c in master_df.columns if str(c) != "frame_index"]
        if not cand:
            return (None, f"主时钟节点 {master_node} 无可用的时间列")
        ts = pd.to_numeric(master_df[cand[0]], errors="coerce").to_numpy()
        if len(ts) == 0:
            return (None, f"主时钟节点 {master_node} 长度为 0")
        if len(ts) != n_rows:
            return (None, f"主时钟长度（{len(ts)}）与子流行数（{n_rows}）"
                          f"不一致，不做截断/补齐对齐")

        return (ts, f"frame_index 逐帧对应，{n_rows} 帧严格一致")
    except Exception as e:  # noqa: BLE001
        return (None, f"帧序校验失败：{type(e).__name__}")


def _dataset_to_frame(data: Any) -> pd.DataFrame | None:
    """把 h5py 数据集内容转为 DataFrame（含 compound 子数组字段的降级处理）。"""
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


def _read_frame_layout_node(f: Any, node: str) -> pd.DataFrame | None:
    """读取"每帧一组"布局中的合并节点（如 ``action/end/orientation``）。

    合并节点（见 :func:`_merge_frame_layout`）在文件里**并不存在**——它是
    "帧号/相对路径"的集合。读取时按帧号升序遍历同名叶子数据集、纵向拼接为
    一张表，并附 ``frame_index`` 列标明每行来源帧。

    每帧数据可能是 1D（如 joint position (14,)）或 2D（如 orientation (2,4)）。
    对 1D，每帧贡献「1 行 × N 列」；对 2D，每帧贡献若干行（如 2×4 → 2 行 4 列，
    用于双手这种"同一字段含两条手臂"的情形）。两种都保留数值，不做语义解释。

    Args:
        f: 已打开且模式为只读的 h5py 文件对象。
        node: 合并后的相对路径（不含帧号前缀）。

    Returns:
        纵向拼接后的 DataFrame（含 frame_index 列）；无匹配返回 None。
    """
    import h5py

    frames: list[pd.DataFrame] = []
    for fid in sorted((k for k in f.keys() if _is_frame_number(k)), key=int):
        grp = f[fid]
        if not isinstance(grp, h5py.Group):
            continue
        leaf = grp.get(node)
        if leaf is None or not isinstance(leaf, h5py.Dataset):
            continue
        data = leaf[()]
        # 每帧数据一律规范为「行 × 列」二维形态（每帧一条观测 → 1 行）：
        #
        # - 1D 多元素 (N,)（如 joint position (14,)）：是**一帧一条 N 维向量观测**，
        #   应理解为「1 行 × N 列」。直接 pd.DataFrame(data) 会得到 N 行 1 列
        #   （把向量当成了时间序列），使每帧贡献 N 行、行数与语义都不对。
        # - 1D 单元素 (1,)（如 timestamp/camera/head_color）：reshape(1, 1)。
        #   **此前被漏掉**（判据是 shape[0] > 1）→ 该流即便登记也无法读出，
        #   真实后果是 h5 相机时间戳与主时间戳对流皆空。
        # - 0 维标量（如 main_timestamp，leaf[()] 返回 numpy 标量）：
        #   pd.DataFrame(scalar) 会抛 ValueError，必须显式 reshape(1, 1)。
        #   此前该形态连登记都进不去，本分支是为配套 _classify_h5_leaf 新增。
        arr = np.asarray(data)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        frame_df = _dataset_to_frame(arr)
        if frame_df is None:
            continue
        # 列名语义化：裸 ndarray 转 DataFrame 后列名是 0/1/2… 序号，
        # 对用户与模型毫无意义。用"字段名_序号"命名（如 position 的 (14,)
        # → position_0…position_13），既保留原始字段语义又标明维度。
        #
        # **单列节点例外**（2026-09-20）：0 维标量与 1D 单元素（如
        # main_timestamp / timestamp/camera/head_color）只有一列，追加 ``_0``
        # 反而破坏列名的**时间戳语义**—— ``is_timestamp_like_field("head_color_0")``
        # 为 False、而 ``is_timestamp_like_field("head_color")`` 的调用方
        # （_H5Reader.timestamp 按列名挑时间轴）会因此挑不到列。故单列时
        # 直接用字段名本身。
        field_name = node.rsplit("/", 1)[-1]
        if all(isinstance(c, (int, np.integer)) for c in frame_df.columns):
            if frame_df.shape[1] == 1:
                frame_df.columns = [field_name]
            else:
                frame_df.columns = [
                    f"{field_name}_{i}" for i in range(frame_df.shape[1])
                ]
        frame_df.insert(0, "frame_index", int(fid))
        frames.append(frame_df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def read_hdf5_node(path: str, node: str) -> pd.DataFrame | None:
    """按节点路径读取 h5py 层级文件的单个数据节点为 DataFrame（公开接口，
    供 _data_access / sync / propose 等工具按流登记表读取 h5 节点流）。

    支持两类节点名：
    - **普通节点**：文件里真实存在的路径（如 ``meta/camera_model``）——直接读；
    - **合并节点**：``_merge_frame_layout`` 产出的相对路径（如
      ``action/end/orientation``）——文件里不存在该路径，改按帧号纵向拼接
      （见 :func:`_read_frame_layout_node`）。
    """
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            node_obj = f.get(node)
            if isinstance(node_obj, h5py.Dataset):
                return _dataset_to_frame(node_obj[()])
            # 不是真实节点：可能是合并后的帧布局路径，尝试纵向拼接。
            return _read_frame_layout_node(f, node)
    except Exception as exc:  # noqa: BLE001
        # 不抛异常（工具契约），但**记录**原因——此前这里是静默 return None，
        # 导致"合并节点读取失败"毫无线索（真实踩坑：帧布局读取函数里一个未导入
        # 的 np 被这个兜底吞掉，表现为"流登记成功但读不出数据"）。
        import logging

        logging.getLogger(__name__).warning(
            "读取 h5 节点失败 path=%s node=%s: %s: %s",
            path, node, type(exc).__name__, exc,
        )
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
        登记的流条目列表（**最多 _MAX_STREAMS_PER_CONTAINER 条**，超出时截断并
        在 context.meta 中标注）。
    """
    nodes = _list_hdf5_native_nodes(str(h5_path))
    # 兜底护栏（第二道）：即便某个陌生布局没被 _merge_frame_layout 识别，
    # 单容器也绝不登记超过上限的流数——否则 context.meta 可达上千万 token，
    # 下游工具返回体积爆炸，界面表现为卡死（2026-09-14 真实事故）。
    if len(nodes) > _MAX_STREAMS_PER_CONTAINER:
        truncated_count = len(nodes) - _MAX_STREAMS_PER_CONTAINER
        nodes = nodes[:_MAX_STREAMS_PER_CONTAINER]
        context.meta.setdefault("stream_registration_notes", []).append(
            f"{h5_path.name} 的节点数超出单容器上限 "
            f"{_MAX_STREAMS_PER_CONTAINER}，已截断 {truncated_count} 条"
            "（该文件可能是非常规布局；如需完整节点清单请用 h5py 直接查看）。"
        )
    entries: list[dict[str, Any]] = []
    for nd in nodes:
        kind, label = _classify_h5_node(nd.get("fields", []), nd["node"])
        entry: dict[str, Any] = {
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
        }
        # 帧布局合并条目：透出帧数与帧号样例，让模型知道该流是"按帧分片"的
        # （读取时按帧纵向拼接，含 frame_index 列）。
        if nd.get("frame_layout"):
            entry["frame_layout"] = True
            entry["n_frames"] = nd.get("n_frames")
            entry["frame_ids_sample"] = nd.get("frame_ids_sample", [])
            entry["label_evidence"] = (
                f"HDF5 每帧一组布局（{nd.get('n_frames')} 帧 × 同名叶子节点，"
                f"读取时按帧纵向拼接）"
            )
        # 语义假设免责标注（2026-09-21）：state/* 与 extrinsic 标定的标签是
        # **按路径命名**判定的语义假设（帧布局下这些节点 fields 恒为空，
        # 没有列名可作证据），不是内容验证的结论。按 AGENTS.md §3.4
        # 「LLM 假设不得替代工具验证」，此处显式告知模型：标签仅供理解清单，
        # **不得据此做数值结论**，需要时用工具实测。
        # unknown / 空流不加（避免噪声）。
        if kind in ("joint_state", "effort", "status", "wrench", "calibration"):
            entry["label_evidence"] = (
                f"{entry['label_evidence']}；标签按**节点路径命名**判定"
                "（语义假设，未经内容验证，不得用于数值结论）"
            )
            entry["label_source"] = "h5_node_scan_path"  # 与字段特征判定区分
        entries.append(entry)
    if entries:
        context.meta.setdefault("streams", []).extend(entries)
    return entries


def register_text_data_streams(
    context: RunContext, probe: dict[str, Any]
) -> list[dict[str, Any]]:
    """把目录内的文本数据文件登记为流（``.txt`` 时间戳清单 / ``.INFO`` 日志）。

    为什么需要（2026-09-14 真实能力缺口）：用户数据集的时间戳**就在**
    ``camera/<相机>/<相机>.txt`` 里（每行 ``<纳秒时间戳> <帧状态/序号>``，
    14142 行、30 Hz），但 ``.txt`` 不在支持格式内，目录加载只把路径放进
    ``others`` 且从不打开——agent 因此只能回答"工具根本没读这些文件"。

    识别策略（**先打开确认、再登记**，避免把说明文本/配置误登记为数据流）：
    经统一读取器 ``read_stream`` 尝试解析，解析成功（首列数值、行等宽）才登记。
    这保证"登记即意味着真的能读"，与 h5/mcap 的登记口径一致。

    Args:
        context: 运行时上下文（streams 追加到 meta["streams"]）。
        probe: probe_directory 的返回（从中取 others 分组的**完整路径清单**）。

    Returns:
        登记的流条目列表（可能为空）。
    """
    from app.tools._readers import ReadRequest, read_stream

    candidates: list[str] = []
    for p in probe_full_paths(probe, "others"):
        if Path(p).suffix.lower() in (".txt", ".text", ".info", ".log"):
            candidates.append(p)
    entries: list[dict[str, Any]] = []
    # 安全的解析上限：仅用于"确认可读 + 取列名与采样率"，不读全量（避免大目录
    # 下把几十个日志全量解析）。真正的数据分析由各工具按需读取。
    probe_limit = 20_000
    for p_str in candidates:
        try:
            result = read_stream(ReadRequest(
                path_spec=p_str, want="frame", limit=probe_limit))
        except Exception:  # noqa: BLE001 - 单文件失败不阻塞目录加载
            continue
        if not result.ok or result.frame is None or result.frame.empty:
            continue
        df = result.frame
        # 必须含可识别的时间列才登记为"数据流"（纯日志无时间列时仍登记，
        # 因为它有 message 可检索，但语义标签不同）。
        ts_col = None
        for cand in ("timestamp", "time", "ts", "time_of_day_us"):
            if cand in df.columns:
                ts_col = cand
                break
        is_log = result.fmt == "log"
        if ts_col is None and not is_log:
            continue
        measured_rate = _measure_rate_hz(df, ts_col)
        kind = "log" if is_log else "timestamp_index"
        label = "运行日志（含时间与消息）" if is_log else "时间戳清单（帧采集记录）"
        entries.append({
            "path": p_str,
            "format": result.fmt,
            "kind": kind,
            "semantic_label": label,
            "label_evidence": (
                f"文本解析确认：{df.shape[0]:,} 行 × {df.shape[1]} 列，"
                + (f"时间列 {ts_col}" if ts_col else "含消息列")
            ),
            "label_confidence": "medium",
            "label_source": "text_probe",
            "role": {"role": label, "confidence": "medium",
                     "evidence": "文本解析确认"},
            "channels": [str(c) for c in df.columns],
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "time_column": ts_col,
            "measured_rate": (
                {"sample_rate_hz": round(measured_rate, 3)}
                if measured_rate else None
            ),
            "is_main": False,
        })
    if entries:
        context.meta.setdefault("streams", []).extend(entries)
    return entries


def _measure_rate_hz(df: pd.DataFrame, ts_col: str | None) -> float | None:
    """从时间列估算采样率（Hz）；无法估算返回 None。

    仅用于给流登记表标注"这条流大约多快"（供 UI 与对齐分析参考），
    不做单位强判——量级明显不符合时间语义时直接返回 None（宁缺勿错）。
    """
    if not ts_col or ts_col not in df.columns:
        return None
    try:
        ts = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=float)
    except Exception:  # noqa: BLE001
        return None
    ts = ts[np.isfinite(ts)]
    if len(ts) < 3:
        return None
    from app.tools.timestamp_units import to_ns

    # 用既有单位推断（量级）换算到 ns 再算速率——与 check_temporal_sync 同口径。
    from app.tools.timestamp_units import infer_unit

    unit = infer_unit(ts)["unit"]
    if unit not in ("ns", "us", "ms", "s"):
        return None
    ns = to_ns(ts, unit)
    diffs = np.diff(ns)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    mean_ns = float(np.mean(diffs))
    if mean_ns <= 0:
        return None
    return 1e9 / mean_ns


def register_mcap_topic_streams(
    context: RunContext, mcap_path: Path, max_probe_messages: int = 200,
) -> list[dict[str, Any]]:
    """把 MCAP 文件的全部可解码 topic 登记为流（复用 h5 多流机制）。

    设计见 docs/MCAP支持设计说明.md：MCAP「单文件多 topic」与 HDF5「单文件多
    节点」同构，故沿用节点流范式——流登记 path 记为
    ``"<mcap 绝对路径>::<topic>"``，format="mcap"，kind 由
    ``classify_mcap_topic``（topic 名线索 + 展开样本指纹）判定，消息数最多的
    topic 标 is_main（目录语境不设 context.df 主表——topic 经
    resolve_table_name 按需读取）。

    非 JSON 编码的 topic（如 ROS2 CDR）**不登记为可分析流**，但会在
    ``meta["mcap_summary"]`` 的 topics 中如实标注 decodable=False + 原因——
    诚实降级，不硬解（第二期能力）。

    Args:
        context: 运行时上下文（streams 追加到 meta["streams"]）。
        mcap_path: mcap 文件路径。
        max_probe_messages: 分类时每 topic 最多读取的消息条数（样本，非全量）。

    Returns:
        登记的流条目列表；缺依赖/解析失败返回 []（不抛异常，不阻塞目录加载）。
    """
    from app.tools import mcap_reader  # 局部导入：避免 mcap 未安装时的导入期硬依赖

    try:
        probe = mcap_reader.probe_mcap(str(mcap_path))
    except mcap_reader.McapDependencyError:
        return []
    if not probe.get("success"):
        return []

    entries: list[dict[str, Any]] = []
    decodable_topics = [t for t in probe["topics"] if t["decodable"]]
    for i, t in enumerate(decodable_topics):
        topic = t["topic"]
        # 读样本（受 max_probe_messages 限制）用于分类；展开样本使嵌套信号可达。
        sample_df = None
        nrows = t.get("message_count") or 0
        try:
            read = mcap_reader.read_mcap_topic(
                str(mcap_path), topic, max_messages=max_probe_messages,
            )
            sample_df = read.get("df")
        except Exception:  # noqa: BLE001 - 单 topic 读取失败不阻塞登记
            sample_df = None

        columns = list(sample_df.columns) if sample_df is not None else []
        expanded = None
        if sample_df is not None:
            try:
                expanded, _ = _data_access.expand_envelope(sample_df)
            except Exception:  # noqa: BLE001
                expanded = sample_df

        # 伪文件名：含 topic 名的安全形式，供既有分类器的命名匹配与证据展示。
        pseudo_name = f"{mcap_path.stem}::{topic}"
        try:
            klass = mcap_reader.classify_mcap_topic(
                topic, pseudo_name, columns, expanded, int(nrows),
            )
        except Exception:  # noqa: BLE001
            klass = {
                "kind": "unknown", "semantic_label": "未分类",
                "label_evidence": "分类失败", "label_confidence": "low",
                "label_source": "error", "status": "active", "channels": columns,
            }

        entries.append({
            "path": f"{mcap_path}::{topic}",
            "format": "mcap",
            "topic": topic,
            "kind": klass.get("kind", "unknown"),
            "semantic_label": klass.get("semantic_label"),
            "label_evidence": klass.get("label_evidence"),
            "label_confidence": klass.get("label_confidence"),
            "label_source": klass.get("label_source"),
            "role": {"role": klass.get("semantic_label", "未分类"),
                     "confidence": klass.get("label_confidence", "low"),
                     "evidence": klass.get("label_evidence", "")},
            "channels": klass.get("channels", columns),
            "n_rows": int(nrows),
            "n_cols": len(columns),
            "message_encoding": t.get("message_encoding"),
            "status": klass.get("status", "active"),
            "is_main": i == 0,  # probe 已按消息数降序，首个为最大 topic
        })

    if entries:
        context.meta.setdefault("streams", []).extend(entries)
    # 容器级概览（含非 JSON 编码 topic 的诚实标注）随 meta 透出。
    context.meta["mcap_summary"] = {
        "n_topics": probe.get("n_topics"),
        "message_count": probe.get("message_count"),
        "time_range_ns": probe.get("time_range_ns"),
        "topics": probe.get("topics", []),
    }
    return entries


def _load_mcap_main(path: str) -> pd.DataFrame:
    """读取 MCAP 文件的消息数最多的可解码 topic 作为主表。

    与 h5 的 `_load_hdf5_native` 同款策略：单文件加载时把信息量最大的 topic
    作为主表（其余 topic 经 register_mcap_topic_streams 登记为流，按名切换）。

    Args:
        path: .mcap 文件路径。

    Returns:
        主表 DataFrame（含 mcap_*_ns 容器时间列 + data 列）。

    Raises:
        MissingDependencyError: 环境缺少 mcap 包（文件未被读取，非文件损坏）。
        ValueError: 文件确实无法解析，或无可解码的 JSON topic。
    """
    from app.tools import mcap_reader

    try:
        probe = mcap_reader.probe_mcap(path)
    except mcap_reader.McapDependencyError as exc:
        # 缺依赖 ≠ 文件损坏：转成既有 MissingDependencyError 契约，
        # 复用 load_dataset_impl 的统一处理分支。
        raise MissingDependencyError("mcap", "mcap", "MCAP (.mcap)") from exc

    if not probe.get("success"):
        raise ValueError(
            f"无法读取 MCAP 文件：{path}（{probe.get('user_message', '解析失败')}）"
        )
    decodable = [t for t in probe["topics"] if t["decodable"]]
    if not decodable:
        raise ValueError(
            f"MCAP 文件 {path} 中没有可解码的 JSON 编码 topic"
            "（可能全部为 ROS2 CDR 编码，属第二期能力）。"
        )
    best = decodable[0]  # probe 已按消息数降序
    read = mcap_reader.read_mcap_topic(path, best["topic"])
    if not read.get("success") or read.get("df") is None:
        raise ValueError(f"MCAP topic {best['topic']} 读取失败或无消息。")
    df = read["df"]
    df.attrs["mcap_main_topic"] = best["topic"]
    df.attrs["mcap_summary"] = probe
    return df


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

    **经统一读取注册表**（与 _sniffing._read_table_columns_cheap 的重复实现
    收敛为一处）；容器子流（h5 节点 / mcap topic）同样支持。

    Args:
        path: 表格文件路径。

    Returns:
        列名列表；读取失败返回 None。
    """
    from app.tools._readers import ReadRequest, read_stream

    result = read_stream(ReadRequest(path_spec=str(path), want="columns"))
    return list(result.columns) if result.ok else None


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """读取配置型 JSON 的顶层对象（**只用于非逐行数据的配置/元信息文件**）。

    与 ``_read_frame_impl`` 的分工：后者把 JSON 展成 DataFrame（面向数据表），
    本函数保留原始的键值结构（面向配置），因为配置的语义在"键 → 值"映射里，
    展成表反而丢失信息。

    Args:
        path: JSON 文件路径。

    Returns:
        顶层 dict；文件是数组/标量或解析失败时返回 None。
    """
    import json as _json

    try:
        obj = _json.loads(
            path.read_text(encoding=_detect_encoding(path.read_bytes()))
        )
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


# 配置值透出上限：单个值的字符串长度与列表元素数（防长数组撑爆上下文）。
_CONFIG_VALUE_MAX_CHARS = 200
_CONFIG_LIST_MAX_ITEMS = 20
_CONFIG_MAX_KEYS = 60


def _compact_config(obj: dict[str, Any]) -> dict[str, Any]:
    """压缩配置 JSON 以便安全透出到模型上下文（保留语义、抑制体积）。

    规则：
    - 标量（str/int/float/bool/None）原样保留；
    - 短列表（元素为标量且 ≤ 上限）保留，超限截断并标注总数；
    - 长字符串截断；
    - 嵌套 dict/list 只报类型与规模（不递归展开），避免深层结构膨胀。

    Args:
        obj: 配置顶层对象。

    Returns:
        压缩后的 dict（键数亦受上限约束，超出标注 ``_truncated_keys``）。
    """
    out: dict[str, Any] = {}
    keys = list(obj.keys())
    for k in keys[:_CONFIG_MAX_KEYS]:
        v = obj[k]
        if isinstance(v, (bool, int, float)) or v is None:
            out[str(k)] = v
        elif isinstance(v, str):
            out[str(k)] = (
                v if len(v) <= _CONFIG_VALUE_MAX_CHARS
                else v[:_CONFIG_VALUE_MAX_CHARS] + "…"
            )
        elif isinstance(v, list):
            if all(isinstance(x, (bool, int, float, str)) or x is None for x in v):
                shown = v[:_CONFIG_LIST_MAX_ITEMS]
                out[str(k)] = (
                    shown if len(v) <= _CONFIG_LIST_MAX_ITEMS
                    else [*shown, f"…（共 {len(v)} 项）"]
                )
            else:
                out[str(k)] = f"[嵌套列表，{len(v)} 项]"
        elif isinstance(v, dict):
            out[str(k)] = f"{{嵌套对象，{len(v)} 个键}}"
        else:
            out[str(k)] = f"<{type(v).__name__}>"
    if len(keys) > _CONFIG_MAX_KEYS:
        out["_truncated_keys"] = len(keys) - _CONFIG_MAX_KEYS
    return out


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

    **经统一读取注册表**；样本行数按格式取（JSONL 5 行——需逐行解析 JSON，
    成本高于 csv/parquet 的列裁剪；其余取 _FINGERPRINT_SAMPLE_ROWS）。

    Args:
        path: 表格文件路径。

    Returns:
        前若干行样本 DataFrame；读取失败返回 None。
    """
    from app.tools._readers import ReadRequest, read_stream
    from app.tools._sniffing import _FINGERPRINT_SAMPLE_ROWS

    is_jsonl = path.suffix.lower() == ".jsonl"
    limit = _JSONL_SNIFF_ROWS if is_jsonl else _FINGERPRINT_SAMPLE_ROWS
    result = read_stream(ReadRequest(path_spec=str(path), want="sample", limit=limit))
    return result.frame if result.ok else None


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
                p.name, cols, sample, nrows or 0,
                fmt=p.suffix.lstrip(".").lower(),
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
            entry: dict[str, Any] = {
                "file": str(p),  # 完整路径，供流登记表按需定位
                "name": p.name,
                "columns": cols[:20],
                "sniff": klass,
                "nrows": nrows,
            }
            # 配置型 JSON（顶层单对象、非逐行数据）：把**字段值**解析出来透出。
            #
            # 为什么需要：这类文件的全部价值就在内容里（如 session.json 的
            # nominal_hz=120、hand_mode=both、delay_ns 等录制参数）。若只报
            # "0 行"，下游（含模型）就完全看不到这些参数——实测正是这样导致
            # 用户提供的标称采样率无法与工具实测值对照。
            if klass.get("kind") == "config":
                cfg = _read_json_object(p)
                if cfg is not None:
                    # 只透出标量/短列表，长数组不进上下文（防体积膨胀）。
                    entry["config"] = _compact_config(cfg)
            table_info.append(entry)
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

    # 第 4 层：用户确认持久化覆盖。加载时优先读取该数据集的画像
    # （outputs/by_dataset/<数据集名>/profile.json）中已确认映射（来源
    # user_confirmed），覆盖第 1-3 层自动识别。旧全局格式会自动迁移（见
    # profile_store），文件不存在/损坏时安全降级为无覆盖，不中断加载。
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
        suffix = Path(other).suffix.lower()
        if suffix in (".h5", ".hdf5"):
            try:
                register_h5_node_streams(context, Path(other))
            except Exception:  # noqa: BLE001 - 单个 h5 登记失败不阻塞目录加载
                pass
        elif suffix == ".mcap":
            # 目录内含 .mcap：把其 topic 登记为流（与 h5 同款机制）。
            try:
                register_mcap_topic_streams(context, Path(other))
            except Exception:  # noqa: BLE001 - 单个 mcap 登记失败不阻塞目录加载
                pass

    # 目录内的文本数据文件（.txt 时间戳清单 / .INFO 日志）：此前一律归入 others
    # 且**从不打开**，导致 agent 只能回答"工具根本没看这些文件"——而时间戳实际
    # 就在其中（真实案例 2026-09-14：14 个相机 .txt 各含 14142 行纳秒时间戳，
    # 但因为 .txt 不在支持格式内而被完全忽略）。现在经统一读取器识别并登记为流。
    register_text_data_streams(context, probe)

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
    # 入参规范化（幂等）：剥离粘贴路径常见的引号/空白/零宽字符/URL 前缀。
    # 两个入口（侧栏粘贴 / 对话中模型抽取）此前规范不一致——模型会剥引号，
    # 粘贴原样透传，导致同一路径在对话里可加载、在侧栏报"文件不存在"
    # （真实事故 2026-09-10）。规范化后两者行为一致。
    cleaned = normalize_path_spec(path)
    path_was_cleaned = cleaned != path
    raw_input = path
    path = cleaned
    source = Path(path)

    if not source.exists():
        # 诚实降级：若确实做过剥离，把"纠正后的路径"一并告知——避免用户
        # 对着自己输入的原文反复核对却看不出差异（引号/零宽字符肉眼不可见）。
        if path_was_cleaned:
            return _error(
                "file_not_found",
                f"文件不存在（已在入参中剥离引号/空白等字符后仍不存在）："
                f"原始={raw_input!r}，规范化后={path}",
                f"路径中检测到引号、空白或不可见字符，已自动剥离后重试，但"
                f"路径 {path} 仍不存在。请检查路径是否正确（注意：已自动去掉"
                "首尾的引号/空格）。",
                extra={"normalized_path": path},
            )
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
            # **优先统一 reader，失败再回退 pd.read_json**（2026-09-20）。
            #
            # 为什么不只用统一 reader：它识别顶层 dict 的行列表键
            # （``data``/``frames``）——UMI 主表
            # 20260729...json 是 {name, total, ..., data:[1096 行]}，
            # 顶层标量键与 data 长度不一致，``pd.read_json`` 会抛
            # "All arrays must be of the same length"，此时必须靠统一 reader。
            #
            # 为什么不只用 pd.read_json：部分 JSON 的行列表键是任意名
            # （如 ``{"a": [...], "b": [...]}``、``{"sensors_list": [...]}``），
            # 不在白名单内，统一 reader 返回 None，此时只能靠 pd.read_json。
            # 两者互补，故"先统一、后回退"——保证与目录/按名读取口径一致，
            # 同时不丢原有能力（此前直接换用统一 reader 导致几个测试回归）。
            df = _data_access.read_stream_full(path, "json")
            if df is None:
                df = pd.read_json(
                    path, encoding=_detect_encoding(source.read_bytes()))
        elif ext == ".jsonl":
            # JSONL：lines=True（每行一个对象）。与 .json 严格区分，不得混用。
            # 同样"先统一、后回退"（JSONL 顶层结构差异小，回退主要覆盖
            # 空文件等边界）。
            df = _data_access.read_stream_full(path, "jsonl")
            if df is None:
                df = pd.read_json(
                    path, lines=True,
                    encoding=_detect_encoding(source.read_bytes()))
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".h5":
            df = _load_hdf5(path)
        elif ext == ".mcap":
            df = _load_mcap_main(path)
        elif ext in (".txt", ".text"):
            from app.tools._text_readers import parse_timestamp_lines

            parsed = parse_timestamp_lines(path)
            if parsed is None:
                raise ValueError(
                    "该文本文件不符合数据表形态（需要每行以数值开头、列数一致）；"
                    "它可能是说明文档或配置文件，不是数据。"
                )
            df = parsed
        elif ext in (".info", ".log"):
            from app.tools._text_readers import parse_log_lines

            parsed = parse_log_lines(path)
            if parsed is None:
                raise ValueError(
                    "该文件不符合可识别的日志格式（glog 风格："
                    "[IWEF]MMDD hh:mm:ss.uuuuuu ...），无法解析为结构化数据。"
                )
            df = parsed
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
        # **主表选择透明化**（2026-09-20）：把候选清单与"如何切换"一并透出。
        # 自动选主表按"行×列最大"——判据确定性但**不知道用户想分析什么**，
        # 用户关心的（末端位置、关节角）往往不是最大的那张。此前只给选中的名字，
        # agent 无从得知"还有哪些表、怎么换"，只能反问或放弃。
        meta["main_table"] = {
            "name": df.attrs.get("h5_source_node"),
            "reason": "按「行×列最大」自动选择（信息量最大的数据节点）",
            "candidates": [
                {"table_name": f"{source.stem}::{h['node']}",
                 "rows": h.get("rows"), "cols": h.get("cols")}
                for h in h5_structure[:8]
            ],
            "alternative_hint": (
                "如需分析其他节点，用 table 参数指定其 table_name"
                "（上列 candidates 或 inspect_streams 的 table_name 字段，"
                "格式为「<文件stem>::<节点>」，不含扩展名）。"
            ),
        }
    # MCAP：单文件多 topic。主 topic 已作为主表，其余 topic 登记为流
    # （复用 h5 节点流范式），供按名切换分析。
    mcap_main_topic = (
        df.attrs.get("mcap_main_topic") if ext == ".mcap" and hasattr(df, "attrs")
        else None
    )
    if mcap_main_topic is not None:
        meta["mcap_main_topic"] = mcap_main_topic
        meta["mcap_topics_pending"] = True
    context.meta = meta
    if meta.pop("h5_node_pending", None):
        register_h5_node_streams(context, source)
    if meta.pop("mcap_topics_pending", None):
        register_mcap_topic_streams(context, source)

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
    # 入参被自动纠正时如实告知（不静默）——用户下次粘贴才知道要留意引号/空格。
    if path_was_cleaned:
        result["path_normalized"] = {
            "raw": raw_input,
            "used": path,
            "note": "路径首尾含引号/空白/不可见字符，已自动剥离后加载成功。",
        }
        result["user_message"] += (
            " （提示：原路径首尾含引号或空白字符，已自动剥离后使用。"
            "下次粘贴请确认没有多余的引号。）"
        )
    if h5_structure:
        result["user_message"] += (
            f" 该文件为 HDF5 原生层级结构，主表为节点 {meta['h5_source_node']}，"
            f"共含 {len(h5_structure)} 个数据节点（其余节点清单见 h5_structure 字段）。"
        )
    if mcap_main_topic is not None:
        summary = meta.get("mcap_summary") or {}
        topics = summary.get("topics", [])
        n_decodable = sum(1 for t in topics if t.get("decodable"))
        result["mcap_summary"] = summary
        result["user_message"] += (
            f" 该文件为 MCAP 容器，主表为消息数最多的 topic {mcap_main_topic}，"
            f"共 {len(topics)} 个 topic（{n_decodable} 个可解码）；"
            "其余 topic 已登记为流，可按名切换分析（表名形如 "
            f"{source.stem}::{mcap_main_topic}）。"
        )
        skipped = [t for t in topics if not t.get("decodable")]
        if skipped:
            result["user_message"] += (
                f" 其中 {len(skipped)} 个 topic 编码非 JSON（本期不解包），"
                "详见 mcap_summary.topics 的 decode_note。"
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
