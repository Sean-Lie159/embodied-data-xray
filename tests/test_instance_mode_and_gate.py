"""实例模式与访问口令门测试（2026-09-07 阶段 B，Commit B1/B2）。

守护：
- Settings 解析 INSTANCE_MODE / ACCESS_PASSWORD（环境变量驱动）；
- check_access_password：未启用恒放行、恒时比较、大小写敏感；
- AppTest 实例模式：侧栏无模型设置面板（同事不可见 base_url/model/key）、
  主界面其余功能正常；
- AppTest 口令门：启用后未解锁只见口令页，解锁后见主界面。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.config.settings as settings_mod
from app.ui.access_gate import check_access_password

_APP_ENTRY = Path(__file__).resolve().parent.parent / "streamlit_app.py"


# --- Settings 解析 -------------------------------------------------------------


def _set_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """补齐模型配置环境变量（脱离项目 .env 时满足校验，聚焦测实例标志）。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")


def test_settings_parse_instance_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """INSTANCE_MODE=1 / ACCESS_PASSWORD 从环境变量解析。"""
    monkeypatch.setenv("INSTANCE_MODE", "1")
    monkeypatch.setenv("ACCESS_PASSWORD", "letmein")
    _set_model_env(monkeypatch)
    monkeypatch.chdir(tmp_path)  # 脱离项目 .env，纯环境变量驱动
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    assert s.instance_mode is True
    assert s.access_password == "letmein"
    settings_mod.get_settings.cache_clear()


def test_settings_instance_default_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """未设置环境变量：实例模式与口令门均关闭（阶段 A 现状）。"""
    monkeypatch.delenv("INSTANCE_MODE", raising=False)
    monkeypatch.delenv("ACCESS_PASSWORD", raising=False)
    _set_model_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    assert s.instance_mode is False
    assert s.access_password == ""
    settings_mod.get_settings.cache_clear()


def test_instance_mode_works_even_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """实例标志独立于 key 完整性：key 缺失（ConfigError）时仍生效。

    这是部署实例的关键路径——同事在部署者配好 key 之前也不能看到配置表单。
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("INSTANCE_MODE", "1")
    monkeypatch.chdir(tmp_path)
    settings_mod.get_settings.cache_clear()
    assert settings_mod.get_instance_mode() is True
    settings_mod.get_settings.cache_clear()


# --- check_access_password（纯函数） --------------------------------------------


def test_gate_disabled_when_no_password() -> None:
    """未配置口令：恒放行（不启用门禁）。"""
    assert check_access_password("", "anything") is True
    assert check_access_password(None, None) is True


def test_gate_compare() -> None:
    """启用后：正确放行、错误拒绝、空输入拒绝。"""
    assert check_access_password("s3cret", "s3cret") is True
    assert check_access_password("s3cret", "wrong") is False
    assert check_access_password("s3cret", "") is False
    assert check_access_password("s3cret", None) is False


# --- AppTest：实例模式 -----------------------------------------------------------


@pytest.fixture()
def instance_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """开启实例模式 + 配好 key 的环境（模型配置"已预设"）。"""
    monkeypatch.setenv("INSTANCE_MODE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-instance-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8787/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "hy3")
    monkeypatch.chdir(tmp_path)  # 脱离项目 .env
    settings_mod.get_settings.cache_clear()
    yield
    settings_mod.get_settings.cache_clear()


def test_app_instance_mode_hides_model_panel(instance_env) -> None:
    """实例模式：主界面正常，但侧栏无模型设置面板（含密钥掩码 caption）。"""
    at = AppTest.from_file(str(_APP_ENTRY), default_timeout=30)
    at.run()
    assert not at.exception, (
        f"实例模式主界面不应有异常：{at.exception[0].value if at.exception else ''}"
    )
    captions = [c.value for c in at.sidebar.caption]
    assert not any("接口地址" in c for c in captions), (
        f"实例模式侧栏不得出现模型配置信息，实得 {captions}"
    )
    # 数据加载面板仍在（同事的核心功能）。
    assert any("单文件上传" in c for c in captions), (
        f"实例模式侧栏应保留数据加载面板，实得 {captions}"
    )


def test_app_instance_mode_unconfigured_no_form(
    instance_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实例模式且 key 缺失：引导页只提示联系部署者，不渲染配置表单。"""
    monkeypatch.delenv("OPENAI_API_KEY")
    settings_mod.get_settings.cache_clear()
    at = AppTest.from_file(str(_APP_ENTRY), default_timeout=30)
    at.run()
    assert not at.exception
    # 文本中含"联系部署者"提示。
    all_text = " ".join(
        [s.value for s in at.subheader]
        + [m.value for m in at.markdown]
        + [w.value for w in at.warning]
    )
    assert "部署者" in all_text, f"实例未配置时应提示联系部署者，实得：{all_text[:300]}"
    # 不出现配置表单控件（接口地址输入框）。
    labels = [t.label for t in at.text_input]
    assert not any("OPENAI_BASE_URL" in (l or "") for l in labels), (
        f"实例模式未配置时不得渲染配置表单，实得 labels={labels}"
    )
    settings_mod.get_settings.cache_clear()


# --- AppTest：口令门 --------------------------------------------------------------


def test_app_gate_blocks_until_unlocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """启用口令门：未解锁只见口令页，无主界面；解锁后进入主界面。

    输入交互的判定逻辑已由 check_access_password 纯函数单测覆盖；
    AppTest 对"输口令→点按钮→rerun"两段式交互模拟脆弱（按钮元素在
    rerun 间的存活性问题），故解锁态用预置 session_state 验证，
    完整交互留给真实运行验收（docs/行为测试.md）。
    """
    _set_model_env(monkeypatch)
    monkeypatch.setenv("ACCESS_PASSWORD", "s3cret")
    monkeypatch.chdir(tmp_path)
    settings_mod.get_settings.cache_clear()

    # 未解锁：口令页。
    at = AppTest.from_file(str(_APP_ENTRY), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("访问口令" in t.value for t in at.title), "应渲染口令门"
    assert "access_granted" not in at.session_state

    # 解锁态：进入主界面（对话标题 + 数据面板）。
    at2 = AppTest.from_file(str(_APP_ENTRY), default_timeout=30)
    at2.session_state["access_granted"] = True
    at2.run()
    assert not at2.exception
    assert not any("访问口令" in t.value for t in at2.title), "解锁后不应再渲染口令门"
    captions = [c.value for c in at2.sidebar.caption]
    assert any("单文件上传" in c for c in captions), "解锁后应见主界面侧栏"
    settings_mod.get_settings.cache_clear()
