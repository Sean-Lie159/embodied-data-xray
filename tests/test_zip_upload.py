"""zip 压缩包上传测试（2026-09-07 阶段 B 数据通道，Commit B3）。

守护四道安全防线与中文文件名修复：
- 正常解压：多文件目录结构还原、时间戳目录隔离；
- zip slip：含 ../ 穿越成员的 zip 被拒绝且不落盘；
- 超限：条目数 / 解压总大小超限被拒绝；
- 中文文件名：无 UTF-8 标志（Windows 常见）的 zip 按 cp437→gbk 修复。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.ui.upload_store import (
    decode_zip_member_name,
    extract_zip,
    sanitize_archive_name,
)


def _make_zip(members: dict[str, bytes], utf8_flag: bool = True) -> bytes:
    """构造内存 zip；utf8_flag=False 时模拟 Windows 旧压缩（无 UTF-8 标志）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, content in members.items():
            info = zipfile.ZipInfo(name)
            if not utf8_flag:
                info.flag_bits &= ~0x800  # 清除 UTF-8 标志
            zf.writestr(info, content)
    return buf.getvalue()


# --- 正常路径 ------------------------------------------------------------------


def test_extract_zip_ok(tmp_path: Path) -> None:
    """正常解压：目录结构还原到时间戳子目录。"""
    data = _make_zip({
        "run01/imu.csv": b"t,x\n1,2\n",
        "run01/tasks.csv": b"episode,success\ne1,1\n",
    })
    dest = extract_zip(data, "我的数据集.zip", tmp_path)
    assert (dest / "run01" / "imu.csv").read_bytes() == b"t,x\n1,2\n"
    assert (dest / "run01" / "tasks.csv").exists()


def test_extract_zip_timestamp_isolation(tmp_path: Path) -> None:
    """两次解压同名压缩包：各自独立时间戳目录，不混写。"""
    data = _make_zip({"a.csv": b"1"})
    d1 = extract_zip(data, "ds.zip", tmp_path)
    d2 = extract_zip(data, "ds.zip", tmp_path)
    assert d1 != d2
    assert (d1 / "a.csv").exists() and (d2 / "a.csv").exists()


# --- 安全校防线 ------------------------------------------------------------------


def test_extract_zip_rejects_non_zip_suffix() -> None:
    with pytest.raises(ValueError, match="仅支持 .zip"):
        extract_zip(b"x", "evil.exe", Path("."))


def test_extract_zip_rejects_bad_zip() -> None:
    with pytest.raises(zipfile.BadZipFile):
        extract_zip(b"not a zip", "bad.zip", Path("."))


def test_extract_zip_rejects_zip_slip(tmp_path: Path) -> None:
    """zip slip：../ 穿越成员被拒绝，且不在解压目录外落盘。"""
    data = _make_zip({"../escaped.txt": b"evil"})
    with pytest.raises(ValueError, match="非法路径"):
        extract_zip(data, "slip.zip", tmp_path)
    assert not (tmp_path / "escaped.txt").exists(), "穿越文件不得落盘"


def test_extract_zip_rejects_too_many_entries(tmp_path: Path) -> None:
    """条目数超限拒绝（zip 炸弹防护，落盘前拦截）。"""
    members = {f"f{i}.csv": b"x" for i in range(10)}
    data = _make_zip(members)
    with pytest.raises(ValueError, match="超过上限"):
        extract_zip(data, "many.zip", tmp_path, max_entries=5)


def test_extract_zip_rejects_oversize(tmp_path: Path) -> None:
    """解压总大小超限拒绝（按 central directory 的 file_size 校验）。"""
    data = _make_zip({"big.csv": b"x" * 1000})
    with pytest.raises(ValueError, match="超过上限"):
        extract_zip(data, "big.zip", tmp_path, max_uncompressed_mb=0)
    # 0MB 上限下任何内容都超限；目录应为空（未落盘）。
    assert not any(tmp_path.iterdir())


# --- 中文文件名修复 --------------------------------------------------------------


def test_decode_zip_name_utf8_flag_passthrough() -> None:
    """有 UTF-8 标志：原名返回。"""
    info = zipfile.ZipInfo("数据集/imu.csv")
    info.flag_bits |= 0x800
    assert decode_zip_member_name(info) == "数据集/imu.csv"


def test_decode_zip_name_gbk_repair() -> None:
    """无 UTF-8 标志：cp437 乱码 → gbk 修复。"""
    info = zipfile.ZipInfo("数据集/imu.csv")
    info.flag_bits &= ~0x800
    # 模拟 zipfile 的 cp437 解码结果（gbk 字节被误读为 cp437）。
    garbled = "数据集/imu.csv".encode("gbk").decode("cp437")
    info.filename = garbled
    assert decode_zip_member_name(info) == "数据集/imu.csv"


def test_sanitize_archive_name() -> None:
    """压缩包保存名净化：路径成分剥离、空名回退。"""
    assert sanitize_archive_name("C:\\evil\\ds.zip") == "ds.zip"
    assert sanitize_archive_name("../../ds.zip") == "ds.zip"
    assert sanitize_archive_name("...zip") == "zip"
