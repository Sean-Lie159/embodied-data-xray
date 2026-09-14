"""纯文本数据读取器：时间戳清单（.txt）与日志（.INFO.log）。

为什么需要（2026-09-14 真实需求）：用户数据集 2655849 的时间戳**就在**
`camera/<相机>/<相机>.txt` 里——每行形如 ``1756265284737924649 P``
（纳秒时间戳 + 帧状态标志），14,142 行、30 Hz、跨度 471 秒、单调递增。
但该扩展名不在支持列表内，工具直接拒读，导致 agent 只能回答"找不到时间戳"
（且如实说明"工具根本没打开这些文件"）——这本身是诚实降级，但能力缺口真实存在。

设计原则：
- **只认确定性形态**：数值首列 + 等宽列（空白分隔）。首列非数值、或行格式
  不一致的文件不解析（返回 None），宁可拒读也不猜出错误的表。
- **不臆造列名**：两列的相机清单文件，第二列按图态判定是"帧状态"还是"帧序号"
  （全为单字符字母 → 状态；全为数字 → 序号），依据数据本身而非文件名。
- **编码回退链**与 CSV 读取器一致（复用 load_dataset._detect_encoding）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 文本数据文件的扩展名（读取器注册用）。
TXT_EXTENSIONS: list[str] = [".txt", ".text"]
LOG_EXTENSIONS: list[str] = [".info", ".log"]

# 采样/全量读取时最多扫描的行数（防超大文件一次性读入内存）。
_MAX_SCAN_ROWS = 5_000_000

# 首列数值的最小占比：低于此比例视为"不是数据表"（不解析）。
_NUMERIC_FIRST_COL_RATIO = 0.9


def _read_text_lines(path: str, limit: int | None) -> list[str] | None:
    """按编码回退链读取文本行；失败返回 None。"""
    from app.tools.load_dataset import _detect_encoding

    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None
    try:
        text = raw.decode(_detect_encoding(raw), errors="replace")
    except Exception:  # noqa: BLE001
        return None
    lines = text.splitlines()
    if limit is not None:
        lines = lines[:limit]
    return lines[:_MAX_SCAN_ROWS]


def _split_token_rows(lines: list[str]) -> list[list[str]] | None:
    """把文本行切成等宽 token 行；非等宽/空行过多返回 None。

    Args:
        lines: 原始文本行。

    Returns:
        token 行列表（每行 token 数一致）；不满足条件返回 None。
    """
    rows: list[list[str]] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        rows.append(s.split())
    if not rows:
        return None
    widths = {len(r) for r in rows}
    # 允许极少数异常行（如文件末尾截断），但主体必须等宽。
    if len(widths) > 3:
        return None
    main_width = max(widths, key=lambda w: sum(1 for r in rows if len(r) == w))
    if main_width < 2:
        return None
    kept = [r for r in rows if len(r) == main_width]
    if len(kept) < len(rows) * 0.95:
        return None
    return kept


def _name_columns(rows: list[list[str]]) -> list[str]:
    """为纯数值列生成确定性列名（不臆造语义，只用可验证的形态判断）。

    第一列默认 ``timestamp``（若其数值量级符合时间戳特征）——这样现有的时间戳
    嗅探（词表 + 量级）能直接命中，无需各工具特判。其余列按内容判定：
    全为单字符字母 → ``status``；全为整数且单调 → ``frame_index``；否则 ``col_N``。

    Args:
        rows: token 行（每行 token 数一致）。

    Returns:
        列名列表（长度 = 每行 token 数）。
    """
    width = len(rows[0])
    names: list[str] = []
    # 首列：判是否像时间戳（纯整数且量级 >= 1e9，即秒级以上 epoch）。
    first = [r[0] for r in rows]
    first_numeric = _as_float_array(first)
    looks_ts = (
        first_numeric is not None
        and np.all(np.isfinite(first_numeric[: min(1000, len(first_numeric))]))
        and float(np.nanmedian(np.abs(first_numeric[: min(1000, len(first_numeric))]))) >= 1e9
    )
    names.append("timestamp" if looks_ts else "col_0")
    for i in range(1, width):
        col = [r[i] for r in rows]
        nums = _as_float_array(col)
        if nums is not None and np.all(np.isfinite(nums)):
            ints = np.all(nums == np.floor(nums))
            if ints and len(nums) > 2 and np.all(np.diff(nums) > 0):
                names.append(f"frame_index_{i}" if i > 1 else "frame_index")
            else:
                names.append(f"value_{i}" if i > 1 else "value")
        elif all(len(c) == 1 and c.isalpha() for c in col):
            names.append("status")
        else:
            names.append(f"col_{i}")
    return names


def _as_float_array(values: list[str]) -> np.ndarray | None:
    """把字符串列表转 float 数组；含无法解析的值时返回 None（不静默丢行）。"""
    try:
        return np.asarray([float(v) for v in values], dtype=float)
    except (ValueError, TypeError):
        return None


def parse_timestamp_lines(path: str, limit: int | None = None) -> pd.DataFrame | None:
    """把"数值首列 + 等宽列"的文本文件解析为 DataFrame。

    典型形态（真实数据集）::

        1756265284737924649 P      # 纳秒时间戳 + 帧状态
        1756265284771714929 P
        ...
        1756265284805200809 1235886  # 纳秒时间戳 + 帧序号（深度相机）

    拒绝解析的情形（宁可拒读也不产出错误的表）：首列非数值占多数、各行 token
    数差异大、或文件为空。

    Args:
        path: 文本文件路径。
        limit: 最多读取的行数（None 为全量）。

    Returns:
        解析后的 DataFrame；不符合数据表形态返回 None。
    """
    lines = _read_text_lines(path, limit)
    if not lines:
        return None
    rows = _split_token_rows(lines)
    if rows is None:
        return None
    # 首列必须是数值（否则不是数据表，可能是配置/说明文本）。
    first = [r[0] for r in rows]
    nums = _as_float_array(first)
    if nums is None:
        return None
    finite_ratio = float(np.mean(np.isfinite(nums))) if len(nums) else 0.0
    if finite_ratio < _NUMERIC_FIRST_COL_RATIO:
        return None

    names = _name_columns(rows)
    width = len(rows[0])
    data: dict[str, Any] = {}
    for i, name in enumerate(names[:width]):
        col = [r[i] for r in rows]
        arr = _as_float_array(col)
        # 数值列（能整个转 float）用数值 dtype；否则保留字符串（如状态标志）。
        data[name] = arr if arr is not None else np.asarray(col, dtype=object)
    df = pd.DataFrame(data)
    if limit is not None:
        df = df.head(limit)
    return df


# 日志行：``I0320 09:37:20.200228 610336 dylog_impl.cpp:641] [DYLOG] msg``
# （glog 风格：级别字母 + MMDD + hh:mm:ss.uuuuuu + 线程 + 文件:行] + 消息）
_LOG_LINE_RE = re.compile(
    r"^([IWEF])(\d{2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\.(\d{1,6})\s+"
    r"(\d+)\s+([^\]]+)\]\s?(.*)$"
)

# 日志年份缺失（格式只有 MMDD）：用文件创建时间或当前年兜底。这里不做年份
# 推断——只把"月日时分秒"作为可排序的时间键输出，并显式标注年份缺失，
# 避免产出看似完整实则错误的绝对时间戳。
_LOG_MAX_LINES = 2_000_000


def parse_log_lines(path: str, limit: int | None = None) -> pd.DataFrame | None:
    """把 glog 风格日志解析为结构化表（时间字段 + 级别 + 来源 + 消息）。

    为什么值得解析：日志里有**可对齐的时间信息**（``I0320 09:37:20.200228``，
    微秒精度）与事件流（文件读写、相机帧率、状态机切换），是排查"某时刻发生了
    什么"的关键证据。此前它只是目录里的一个``others``路径。

    年份缺失的处理（诚实优先）：glog 的 MMDD 不含年份，故**不合成绝对时间戳**，
    而是输出 ``time_of_day_us``（当日微秒数，可排序、可算间隔）与
    ``clock_scope="time_of_day"`` 标注；需要绝对时间时由用户结合文件创建时间判断。

    Args:
        path: 日志文件路径。
        limit: 最多解析的行数。

    Returns:
        结构化 DataFrame（time_of_day_us / level / thread / source / message，
        非日志行归入 message 且 level 为空）；无可解析行返回 None。
    """
    lines = _read_text_lines(path, limit)
    if not lines:
        return None
    lines = lines[:_LOG_MAX_LINES]
    rows: list[dict[str, Any]] = []
    n_parsed = 0
    for ln in lines:
        m = _LOG_LINE_RE.match(ln)
        if m:
            lvl, mo, day, hh, mm, ss, frac, thread, source, msg = m.groups()
            micro = int(frac.ljust(6, "0")[:6])
            tod_us = (
                ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * 1_000_000 + micro
            )
            rows.append({
                "time_of_day_us": tod_us,
                "level": lvl,
                "thread": int(thread),
                "source": source.strip(),
                "message": msg,
            })
            n_parsed += 1
        else:
            s = ln.strip()
            if s:
                rows.append({
                    "time_of_day_us": None, "level": None,
                    "thread": None, "source": None, "message": s,
                })
    # 一行都没解析出来 → 不是可识别的日志格式（拒读，不产出误导性表）。
    if n_parsed == 0:
        return None
    df = pd.DataFrame(rows)
    df.attrs["clock_scope"] = "time_of_day"
    df.attrs["clock_note"] = (
        "日志时间戳为 glog 的 MMDD hh:mm:ss.uuuuuu 格式，**不含年份**；"
        "time_of_day_us 为当日微秒数（可排序、可算间隔），非绝对时间。"
    )
    return df
