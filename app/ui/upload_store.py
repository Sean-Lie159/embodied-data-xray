"""zip 压缩包上传与解压（阶段 B 数据通道）：同事整目录数据集进部署机。

安全四防线（docs/实例部署模式设计.md 第 6 节）：
1. 解压前从 central directory 校验未压缩总大小与条目数（zip 炸弹不落盘即拒）；
2. zip slip 防护：成员真实路径必须落在解压目录内（resolve + is_relative_to）；
3. 中文文件名修复：无 UTF-8 标志（flag_bits 0x800）的 zip（Windows 常见）
   按 cp437→gbk 修复；
4. 解压到一次性时间戳目录，不与既有目录混写。

纯函数、不 import streamlit，可独立单测。
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path

# 解压安全上限：未压缩总大小（MB）与条目数（防 zip 炸弹，落盘前拦截）。
MAX_UNCOMPRESSED_MB = 1024
MAX_ENTRIES = 5000


def sanitize_archive_name(name: str) -> str:
    """净化压缩包保存名（同单文件上传口径：去路径成分与危险字符）。"""
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", base, flags=re.UNICODE)
    return cleaned.strip("._") or "dataset.zip"


def decode_zip_member_name(info: zipfile.ZipInfo) -> str:
    """修复 zip 成员文件名编码（中文乱码）。

    有 UTF-8 标志（flag_bits & 0x800）时 zipfile 已正确解码，原样返回；
    否则（Windows 资源管理器/旧工具压缩的常见情况）zipfile 按 cp437 解出
    乱码，按 cp437 还原字节流再以 gbk 解码（失败则保持原样，不硬猜）。
    """
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def extract_zip(
    data: bytes,
    original_name: str,
    target_root: str | Path,
    max_uncompressed_mb: int = MAX_UNCOMPRESSED_MB,
    max_entries: int = MAX_ENTRIES,
) -> Path:
    """把上传的 zip 解压为数据集目录（安全校验后落盘）。

    Args:
        data: 上传的 zip 字节内容。
        original_name: 客户端原始文件名（净化后作为解压目录名的一部分）。
        target_root: 解压根目录（如 outputs/uploads）。
        max_uncompressed_mb: 解压后总大小上限（MB）。
        max_entries: 条目数上限。

    Returns:
        解压出的数据集目录（target_root/<stem>_<时间戳>/）。

    Raises:
        ValueError: 非 zip 内容 / 超过大小或条目上限 / 发现 zip slip 成员。
        zipfile.BadZipFile: 损坏的 zip（调用方捕获转达）。
    """
    if Path(original_name).suffix.lower() != ".zip":
        raise ValueError("仅支持 .zip 压缩包（目录型数据集请打包后上传）。")

    zf = zipfile.ZipFile(io.BytesIO(data))  # 损坏时抛 BadZipFile
    with zf:
        infos = zf.infolist()
        entries = [i for i in infos if not i.is_dir()]
        if len(entries) > max_entries:
            raise ValueError(
                f"压缩包含 {len(entries)} 个文件，超过上限 {max_entries}，已拒绝。"
            )
        total = sum(i.file_size for i in entries)
        if total > max_uncompressed_mb * 1024 * 1024:
            raise ValueError(
                f"解压后总大小约 {total / 1024 / 1024:.0f}MB，"
                f"超过上限 {max_uncompressed_mb}MB，已拒绝。"
            )

        stem = Path(sanitize_archive_name(original_name)).stem
        dest = Path(target_root) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        seq = 2
        while dest.exists():  # 同秒多次上传：追加序号保证目录隔离（测试暴露）
            dest = Path(target_root) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}_{seq}"
            seq += 1
        dest_root_resolved = dest.resolve()
        dest_root_resolved.mkdir(parents=True, exist_ok=True)

        for info in entries:
            member = decode_zip_member_name(info)
            target = (dest / member).resolve()
            # zip slip 防护：成员路径（含 ../ 穿越与绝对路径）必须落在解压目录内。
            if not target.is_relative_to(dest_root_resolved):
                raise ValueError(
                    f"压缩包含非法路径成员（{member!r}），已拒绝解压。"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    return dest
