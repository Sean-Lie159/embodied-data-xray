"""上传文件落盘的纯函数：文件名净化与保存（不 import streamlit，可单测）。

设计（docs/UI快速上手与模型配置设计.md 3.3）：单文件上传是"粘贴绝对路径"
主路径的辅助通道，仅支持单文件（csv/parquet/json/jsonl），保存到
``outputs/uploads/``（不进 git）；目录型数据集不适用上传。

安全要点：不信任客户端文件名——去路径分隔符、去危险字符、重名加时间戳
后缀防覆盖。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

# 允许的扩展名（小写，含点）。
ALLOWED_SUFFIXES = {".csv", ".parquet", ".json", ".jsonl"}


def sanitize_upload_filename(name: str) -> str:
    """净化上传文件名：去路径成分与危险字符，保留扩展名。

    处理：取 Path(name).name 去掉目录成分；把非字母数字/点/下划线/连字符的
    字符（含中文？中文是合法文件名但为稳妥统一转下划线——不对，中文文件名
    应保留，只清理路径分隔符与控制字符）替换为下划线；空名回退 "upload"。

    Args:
        name: 客户端提供的原始文件名。

    Returns:
        净化后的安全文件名（仅文件名部分，无路径）。
    """
    base = Path(name).name  # 去目录成分（含 "C:\..." 与 "../" 变体）
    # 去控制字符与路径分隔符变体，其余保留（含中文）。
    cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", base, flags=re.UNICODE)
    cleaned = cleaned.strip("._") or "upload"
    return cleaned


def save_upload(data: bytes, original_name: str, target_dir: str | Path) -> Path:
    """把上传的字节内容保存到目标目录（重名加时间戳后缀防覆盖）。

    Args:
        data: 上传文件字节内容。
        original_name: 客户端原始文件名（先净化再使用）。
        target_dir: 保存目录（如 outputs/uploads）。

    Returns:
        实际写入的文件路径。

    Raises:
        ValueError: 扩展名不在 ALLOWED_SUFFIXES 内。
    """
    safe_name = sanitize_upload_filename(original_name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = "、".join(sorted(ALLOWED_SUFFIXES))
        raise ValueError(
            f"不支持的文件类型 {suffix or '（无扩展名）'}，仅支持：{allowed}"
        )
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    dest = target / safe_name
    if dest.exists():
        stem = dest.stem
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest = target / f"{stem}_{ts}{suffix}"
    dest.write_bytes(data)
    return dest
