"""数据加载面板测试（2026-09-07 Commit 3）。

覆盖：
- upload_store：文件名净化（路径成分/危险字符剥离、中文保留）、非法扩展名
  拒绝、重名加时间戳防覆盖；
- data_loader_panel._load_path：加载成功/失败都**进对话流**（决策 3），
  成功消息含数据集名，失败消息如实转达原因（纪律 4）；
- AppTest 冒烟：已配置环境侧栏含"数据加载"面板。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app.config.settings as settings_mod
from app.services.chat_service import ChatService
from app.ui import data_loader_panel as dlp
from app.ui.upload_store import sanitize_upload_filename, save_upload

_APP_ENTRY = Path(__file__).resolve().parent.parent / "streamlit_app.py"


# --- sanitize_upload_filename ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C:\\evil\\path\\data.csv", "data.csv"),  # Windows 路径成分剥离
        ("../../etc/passwd.json", "passwd.json"),  # 相对路径穿越剥离（仅存最后段）
        ("数据集_01.csv", "数据集_01.csv"),  # 中文文件名保留
        ("a b (1).parquet", "a_b_1_.parquet"),  # 空格与括号转下划线
        ("", "upload"),  # 空名回退
        ("...jsonl", "jsonl"),  # 前导点剥离
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_upload_filename(raw) == expected


# --- save_upload --------------------------------------------------------------


def test_save_upload_ok(tmp_path: Path) -> None:
    dest = save_upload(b"col\n1\n", "my_data.csv", tmp_path)
    assert dest.name == "my_data.csv"
    assert dest.read_bytes() == b"col\n1\n"


def test_save_upload_rejects_bad_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不支持的文件类型"):
        save_upload(b"x", "evil.exe", tmp_path)


def test_save_upload_no_overwrite(tmp_path: Path) -> None:
    """重名文件不覆盖：自动加时间戳后缀另存。"""
    save_upload(b"first", "dup.csv", tmp_path)
    second = save_upload(b"second", "dup.csv", tmp_path)
    assert second.name != "dup.csv" and "dup" in second.name
    assert (tmp_path / "dup.csv").read_bytes() == b"first"
    assert second.read_bytes() == b"second"


# --- _load_path：加载结果进对话流（决策 3） -------------------------------------


@pytest.fixture()
def service() -> ChatService:
    """真实 ChatService（测试环境 .env 已配置，先例见 test_streamlit_app）。"""
    settings_mod.get_settings.cache_clear()
    return ChatService()


def test_load_path_success_appends_message(service: ChatService, tmp_path: Path) -> None:
    csv_path = tmp_path / "good_ds.csv"
    pd.DataFrame({
        "episode": [1, 2, 3, 4],
        "success": [1, 0, 1, 1],
        "j1": [0.1, 0.2, 0.3, 0.4],
    }).to_csv(csv_path, index=False)
    messages: list[dict] = []
    ok = dlp._load_path(service, messages, str(csv_path))
    assert ok is True
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "assistant"
    assert "good_ds.csv" in msg["content"]
    assert "数据加载" in msg["content"]  # 注明加载来源（面板），可溯源


def test_load_path_failure_reports_in_messages(service: ChatService) -> None:
    """失败路径：不抛异常，失败原因如实写入对话流（纪律 4）。"""
    messages: list[dict] = []
    ok = dlp._load_path(service, messages, "Z:/no/such/dataset_xyz")
    assert ok is False
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "assistant"
    assert "失败" in msg["content"]


# --- AppTest 冒烟 --------------------------------------------------------------


def test_app_smoke_has_data_loader_panel() -> None:
    """已配置环境：侧栏渲染数据加载面板（caption 含上传说明）。"""
    settings_mod.get_settings.cache_clear()
    at = AppTest.from_file(str(_APP_ENTRY), default_timeout=30)
    at.run()
    assert not at.exception, (
        f"主界面不应有未捕获异常：{at.exception[0].value if at.exception else ''}"
    )
    captions = [c.value for c in at.sidebar.caption]
    assert any("单文件上传" in c for c in captions), (
        f"侧栏应含数据加载面板，实得 captions={captions}"
    )
    # 面板含路径输入框与上传控件。
    assert any(
        t.label == "数据集绝对路径" for t in at.sidebar.text_input
    ), "侧栏应含数据集路径输入框"
