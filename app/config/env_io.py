""".env 文件读写模块：为 UI 模型设置表单提供配置落盘能力。

设计要点（见 docs/UI快速上手与模型配置设计.md 3.1）：

- **逐行替换**：只修改目标键所在行，保留文件中其余行原样（包括注释、空行、
  用户手改的其他键与顺序），不做格式重排——用户手写的注释不丢；
- **缺键追加**：目标键不存在时追加到文件末尾；
- **原子写**：先写临时文件再 ``os.replace``，避免写一半损坏配置；
- **纯函数、不 import streamlit**：可独立单测（"工具即壁垒"原则，先能独立
  运行和被测试，再接 UI）。

密钥仍只存在于 ``.env``、仍由 pydantic-settings（``app/config/settings.py``）
读取，本模块只是替代"用户手编 .env 文件"，符合 AGENTS.md 硬性规则 2
（密钥一律从 .env 读取；.env 已在 .gitignore）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def read_env_file(path: str | Path) -> dict[str, str]:
    """读取 .env 文件中的键值对（忽略注释与空行、空值键）。

    解析规则与 pydantic-settings 的 dotenv 兼容子集：``KEY=VALUE``，值取第一个
    ``=`` 之后的全部内容（值中允许含 ``=``，如 ``sk-abc=1`` 不截断）；行首
    ``#`` 为注释；键两侧空白剔除；值为空字符串的键也返回（表示占位未填）。

    Args:
        path: .env 文件路径。

    Returns:
        有序 dict（按文件中出现顺序）；文件不存在返回空 dict。
    """
    result: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return result
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        result[key] = value.strip()
    return result


def update_env_file(path: str | Path, updates: dict[str, str]) -> Path:
    """把若干键值更新写入 .env 文件（逐行替换 + 缺键追加 + 原子写）。

    保留文件中其余行原样（注释、空行、顺序不变）；目标键已存在则替换该行
    （键名保留原行的大小写形态，值替换为 ``KEY=new_value``）；不存在则追加
    到文件末尾。写临时文件后 ``os.replace`` 原子替换。

    Args:
        path: .env 文件路径（不存在则创建，含父目录）。
        updates: 要写入的键值对（值转为字符串，None 写为空串）。

    Returns:
        写入的文件路径。

    Raises:
        不主动抛业务异常；IO 错误（权限/磁盘）由调用方捕获转达。
    """
    env_path = Path(path)
    normalized = {
        k.strip(): "" if v is None else str(v) for k, v in updates.items()
    }
    existing_keys: set[str] = set()
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        for raw in lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.partition("=")[0].strip()
            if key:
                existing_keys.add(key)

    new_lines: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        matched_key: str | None = None
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in normalized:
                matched_key = key
        if matched_key is not None:
            new_lines.append(f"{matched_key}={normalized[matched_key]}")
        else:
            new_lines.append(raw)

    # 缺失键追加（保持 updates 传入顺序）。
    for key, value in normalized.items():
        if key not in existing_keys:
            new_lines.append(f"{key}={value}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(new_lines)
    if content and not content.endswith("\n"):
        content += "\n"
    # 原子写：临时文件与目标同目录（保证同分区 replace），写完即替换。
    fd, tmp_name = tempfile.mkstemp(
        dir=str(env_path.parent), prefix=".env_tmp_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp_name, env_path)
    except Exception:
        # 写入失败时清理临时文件，不留垃圾。
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return env_path


def mask_secret(value: str | None, visible_tail: int = 4) -> str:
    """把密钥转成掩码展示（UI 用）：保留末尾若干位，其余以 * 遮蔽。

    Args:
        value: 原始密钥（可 None/空）。
        visible_tail: 末尾保留的明文位数。

    Returns:
        掩码字符串；空值返回 "（未配置）"，过短值整体遮蔽。
    """
    if not value:
        return "（未配置）"
    v = str(value)
    if len(v) <= visible_tail:
        return "*" * len(v)
    return "*" * (len(v) - visible_tail) + v[-visible_tail:]
