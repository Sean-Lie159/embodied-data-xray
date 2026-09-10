"""路径规范化（引号/空白/不可见字符剥离）的单元测试。

背景（用户实测 2026-09-10）：**偶发**的侧栏数据加载失败——同一 MCAP 路径，
侧栏粘贴报"文件不存在，请检查路径是否正确"，而在对话里让 agent 加载却成功。
根因是路径的两个入口规范不一致：侧栏 `st.text_input` 把用户粘贴的内容原样
透传（用户从对话记录/文档复制时很自然带上引号，如 ``"C:\\...\\x.mcap"``），
而对话路径经模型抽取时引号已被剥离。引号成了路径的一部分 → ``exists()``
返回 False → 误导性提示"文件不存在"（路径其实完好）。

覆盖：各类污染形态的纠正、幂等性、合法路径零误伤、端到端加载与诚实报错。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl, normalize_path_spec

# 一个在多数 Windows 环境下都存在的路径（仅用于"零误伤"断言，不要求可读）。
_WIN_PATH = r"C:\Users\me\data\robot.mcap"


# --- 1. 各类污染形态都能纠正为同一规范形式 ---------------------------------


@pytest.mark.parametrize("name,raw", [
    ("裸路径", _WIN_PATH),
    ("直双引号包裹", f'"{_WIN_PATH}"'),
    ("单引号包裹", f"'{_WIN_PATH}'"),
    ("反引号包裹", f"`{_WIN_PATH}`"),
    ("中文成对引号", "\u201c" + _WIN_PATH + "\u201d"),
    ("中文单引号", "\u2018" + _WIN_PATH + "\u2019"),
    ("直角引号", "\u300c" + _WIN_PATH + "\u300d"),
    ("全角双引号", "\uff02" + _WIN_PATH + "\uff02"),
    ("嵌套混合包裹", "\u201c" + f'"{_WIN_PATH}"' + "\u201d"),
    ("前后空格", f"  {_WIN_PATH}  "),
    ("尾部换行", _WIN_PATH + "\n"),
    ("零宽空格", _WIN_PATH + "\u200b"),
    ("BOM", _WIN_PATH + "\ufeff"),
    ("全角逗号后缀", _WIN_PATH + "\uff0c"),
    ("中文句号后缀", _WIN_PATH + "\u3002"),
    ("正斜杠", _WIN_PATH.replace("\\", "/")),
    ("file URL", "file:///" + _WIN_PATH.replace("\\", "/")),
])
def test_variants_normalize_to_same(name: str, raw: str) -> None:
    """17 种粘贴形态全部规范化为同一裸路径（"偶发"的真正来源）。"""
    assert normalize_path_spec(raw) == _WIN_PATH, f"{name} 未规范化：{raw!r}"


def test_normalize_is_idempotent() -> None:
    """幂等：反复规范化结果不变（可安全重复调用）。"""
    for raw in [f'"{_WIN_PATH}"', "  " + _WIN_PATH + "\u200b",
                "\u201c" + _WIN_PATH + "\u201d\uff0c"]:
        once = normalize_path_spec(raw)
        assert normalize_path_spec(once) == once


def test_normalize_keeps_legit_paths_untouched() -> None:
    """不可误伤：合法路径原样保留（含空格、UNC、正斜杠 Unix 路径、中文）。"""
    legit = [
        _WIN_PATH,
        r"C:\Users\me\My Documents\a b.csv",   # 路径内含空格必须保留
        r"\\server\share\data\x.mcap",         # UNC 路径
        "/home/user/data/x.csv",               # Unix 路径（不以盘符开头）
        "outputs/中文目录/文件.jsonl",
        r"C:\data\a'b.csv",                    # 引号出现在**中间**不动
    ]
    for p in legit:
        assert normalize_path_spec(p) == p, f"合法路径被改动：{p!r}"


def test_normalize_handles_non_string_and_blank() -> None:
    """非字符串/空白输入不抛异常（防御性，原样返回）。"""
    assert normalize_path_spec("") == ""
    assert normalize_path_spec("   ") == "   "
    assert normalize_path_spec(None) is None  # type: ignore[arg-type]


def test_normalize_windows_slash_only_for_drive_paths() -> None:
    """仅盘符形态统一为反斜杠；Unix 路径与 UNC 不被改写。"""
    assert normalize_path_spec("C:/Users/me/a.csv") == r"C:\Users\me\a.csv"
    assert normalize_path_spec("/home/me/a.csv") == "/home/me/a.csv"
    assert normalize_path_spec("//server/share/a.csv") == "//server/share/a.csv"


# --- 2. 端到端：带引号的路径能正常加载 -------------------------------------


def _write_csv(path: Path) -> Path:
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    return path


def test_load_with_quoted_path_succeeds(tmp_path: Path) -> None:
    """**核心回归**：侧栏粘贴带引号的路径 → 加载成功（此前报"文件不存在"）。"""
    f = _write_csv(tmp_path / "t.csv")
    ctx = RunContext()
    result = load_dataset_impl(ctx, f'"{f}"')
    assert result["success"] is True, result.get("user_message")
    assert ctx.dataset_id == "t"


@pytest.mark.parametrize("wrap", [
    '"{p}"', "'{p}'", "  {p}  ", "{p}\n", "\u201c{p}\u201d", "{p}\u200b",
])
def test_load_with_various_pollution_succeeds(tmp_path: Path, wrap: str) -> None:
    """各种污染包裹下都能加载成功。"""
    f = _write_csv(tmp_path / "t.csv")
    result = load_dataset_impl(RunContext(), wrap.format(p=f))
    assert result["success"] is True, f"{wrap!r} 失败：{result.get('user_message')}"


def test_success_reports_auto_correction(tmp_path: Path) -> None:
    """自动纠正不静默：成功返回里如实告知原值与实用值。"""
    f = _write_csv(tmp_path / "t.csv")
    result = load_dataset_impl(RunContext(), f'"{f}"')
    assert result["success"] is True
    norm = result.get("path_normalized")
    assert norm is not None
    assert norm["raw"] == f'"{f}"'
    assert norm["used"] == str(f)
    assert "已自动剥离" in result["user_message"]


def test_clean_path_has_no_normalization_note(tmp_path: Path) -> None:
    """干净路径不产生多余提示（防噪声：只在确有纠正时才说）。"""
    f = _write_csv(tmp_path / "t.csv")
    result = load_dataset_impl(RunContext(), str(f))
    assert result["success"] is True
    assert "path_normalized" not in result
    assert "已自动剥离" not in result["user_message"]


# --- 3. 带引号但确实不存在 → 诚实报错（不静默、不误导）---------------------


def test_quoted_missing_path_reports_honestly(tmp_path: Path) -> None:
    """规范化后仍不存在：报错需说明"已剥离后重试"+ 给出实用路径。"""
    missing = tmp_path / "nope.csv"
    result = load_dataset_impl(RunContext(), f'"{missing}"')
    assert result["success"] is False
    assert result["error"] == "file_not_found"
    # 不得只说"请检查路径"——必须让用户知道我们已剥离过引号。
    assert "已自动剥离" in result["user_message"]
    assert result["normalized_path"] == str(missing)


def test_clean_missing_path_keeps_original_message(tmp_path: Path) -> None:
    """干净路径不存在：保持原有简洁提示（不引入无关的"剥离"说明）。"""
    result = load_dataset_impl(RunContext(), str(tmp_path / "nope.csv"))
    assert result["success"] is False
    assert result["error"] == "file_not_found"
    assert "文件不存在" in result["reason"]
    assert "已自动剥离" not in result["user_message"]


# --- 4. 统一读取入口（容器子流路径）也规范化 ------------------------------


def test_read_stream_normalizes_file_part(tmp_path: Path) -> None:
    """read_stream 对 ``<file>::<sub>`` 的文件部分做规范化，子流名不被动。"""
    from app.tools._readers import ReadRequest, read_stream

    f = _write_csv(tmp_path / "t.csv")
    # 文件部分带引号：应能读到（说明规范化在拆分子流之前生效）。
    r = read_stream(ReadRequest(path_spec=f'"{f}"', want="columns"))
    assert r.ok is True
    assert list(r.columns) == ["a", "b"]


def test_read_stream_preserves_sub_name(tmp_path: Path) -> None:
    """子流名保持原值（不被当作路径清洗）——h5 节点/mcap topic 可能是任意串。"""
    from app.tools._readers import split_path_spec

    # 子流名里出现引号也不该在这一层被剥离（它属于登记表原值）。
    file_part, sub = split_path_spec('"C:\\a\\b.h5"::/sensor/acc"el')
    assert sub == '/sensor/acc"el'


# --- 5. mcap_reader 公开接口同样规范化 ------------------------------------


def test_probe_mcap_normalizes_quoted_path(tmp_path: Path) -> None:
    """probe_mcap 是可被直接调用的公开接口：带引号的路径不再报"不存在"。"""
    from app.tools.mcap_reader import probe_mcap

    # 文件不存在时仍应报 file_not_found，但**不得**因引号而误报——
    # 这里用一个真实存在的空文件，断言错误不是 file_not_found。
    fake = tmp_path / "x.mcap"
    fake.write_bytes(b"\x00" * 16)
    r = probe_mcap(f'"{fake}"')
    assert r.get("error") != "file_not_found", (
        "带引号的已存在文件被误判为不存在（规范化未生效）"
    )
