""".env 读写模块单测（2026-09-07 UI 模型配置，设计文档 3.1 节）。

守护行为：
- 读：忽略注释/空行、值含 = 不截断、空值键保留、文件缺失返回空；
- 写：逐行替换保留其余行原样（注释/顺序/用户手改的其他键）、缺键追加、
  文件不存在时创建、原子写不留临时文件；
- 掩码：末尾可见、空值占位、短值整体遮蔽。

这些行为决定 UI 表单能否安全回写 .env（不丢用户手写注释）。
"""

from __future__ import annotations

from pathlib import Path

from app.config.env_io import mask_secret, read_env_file, update_env_file

# 中文注释与既有键，模拟用户手改过的真实 .env（含值中带 #、带 = 的边角）。
_EXISTING_ENV = """\
# OpenAI 兼容接口：填你用的模型服务商的密钥
OPENAI_API_KEY=sk-old-key-123
# 接口地址：DeepSeek 填 https://api.deepseek.com
OPENAI_BASE_URL=https://api.deepseek.com

# 模型名
DEFAULT_MODEL=deepseek-chat
DEFAULT_TEMPERATURE=0.2
WEIRD_VALUE=a=b=c  # 行内注释也算值的一部分
"""


def test_read_env_parses_and_ignores_comments(tmp_path: Path) -> None:
    """读取：解析键值、忽略注释空行、值含 = 不截断。"""
    p = tmp_path / ".env"
    p.write_text(_EXISTING_ENV, encoding="utf-8")
    env = read_env_file(p)
    assert env["OPENAI_API_KEY"] == "sk-old-key-123"
    assert env["OPENAI_BASE_URL"] == "https://api.deepseek.com"
    assert env["DEFAULT_MODEL"] == "deepseek-chat"
    # 值中的 = 不截断（取第一个 = 之后全部内容）。
    assert env["WEIRD_VALUE"] == "a=b=c  # 行内注释也算值的一部分"
    assert "MAX_ROWS_IN_CONTEXT" not in env  # 纯注释键不解析


def test_read_env_missing_file_returns_empty(tmp_path: Path) -> None:
    """文件不存在返回空 dict，不抛异常。"""
    assert read_env_file(tmp_path / "no_such.env") == {}


def test_update_replaces_only_target_lines(tmp_path: Path) -> None:
    """更新：仅替换目标键所在行，注释/空行/顺序/其他键原样保留。"""
    p = tmp_path / ".env"
    p.write_text(_EXISTING_ENV, encoding="utf-8")
    update_env_file(p, {
        "OPENAI_API_KEY": "sk-new-key-999",
        "DEFAULT_MODEL": "gpt-4o-mini",
    })

    lines = p.read_text(encoding="utf-8").splitlines()
    # 目标键已替换。
    assert "OPENAI_API_KEY=sk-new-key-999" in lines
    assert "DEFAULT_MODEL=gpt-4o-mini" in lines
    # 其余内容原样保留（含注释、空行、未触碰的键）。
    assert "# OpenAI 兼容接口：填你用的模型服务商的密钥" in lines
    assert "# 接口地址：DeepSeek 填 https://api.deepseek.com" in lines
    assert "OPENAI_BASE_URL=https://api.deepseek.com" in lines
    assert "DEFAULT_TEMPERATURE=0.2" in lines
    assert "" in lines  # 空行保留
    # 替换发生在原位置：API key 行仍在其注释行的下一行。
    key_idx = next(i for i, l in enumerate(lines) if l.startswith("OPENAI_API_KEY="))
    assert lines[key_idx - 1].startswith("# OpenAI")
    # 未提及的键不被动过。
    env = read_env_file(p)
    assert env["WEIRD_VALUE"] == "a=b=c  # 行内注释也算值的一部分"


def test_update_appends_missing_keys(tmp_path: Path) -> None:
    """目标键不存在时追加到文件末尾（保持传入顺序）。"""
    p = tmp_path / ".env"
    p.write_text(_EXISTING_ENV, encoding="utf-8")
    update_env_file(p, {"NEW_KEY_A": "1", "NEW_KEY_B": "2"})
    env = read_env_file(p)
    assert env["NEW_KEY_A"] == "1" and env["NEW_KEY_B"] == "2"
    # 原有键不受影响。
    assert env["OPENAI_API_KEY"] == "sk-old-key-123"


def test_update_creates_missing_file(tmp_path: Path) -> None:
    """文件不存在时创建（含父目录）。"""
    p = tmp_path / "sub" / "dir" / ".env"
    update_env_file(p, {"OPENAI_API_KEY": "sk-first", "DEFAULT_MODEL": "m1"})
    env = read_env_file(p)
    assert env == {"OPENAI_API_KEY": "sk-first", "DEFAULT_MODEL": "m1"}


def test_update_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    """写入后目录中不留临时文件（原子替换成功路径）。"""
    p = tmp_path / ".env"
    p.write_text(_EXISTING_ENV, encoding="utf-8")
    update_env_file(p, {"DEFAULT_MODEL": "m2"})
    leftovers = [f.name for f in tmp_path.iterdir() if f.name != ".env"]
    assert leftovers == [], f"应无临时文件残留，实得 {leftovers}"


def test_mask_secret() -> None:
    """掩码：末尾可见、空值占位、短值整体遮蔽。"""
    # 15 位密钥：遮蔽前 11 位，保留末尾 4 位。
    assert mask_secret("sk-abcdef123456") == "***********3456"
    assert mask_secret(None) == "（未配置）"
    assert mask_secret("") == "（未配置）"
    assert mask_secret("abc", visible_tail=4) == "***"
    # 过短值（长度 ≤ 保留位数）整体遮蔽，不暴露明文。
    assert mask_secret("abcdef", visible_tail=10) == "******"
