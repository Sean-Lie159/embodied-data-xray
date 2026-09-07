"""模型连接测试 + 配置面板/引导页冒烟测试（2026-09-07 Commit 2）。

覆盖：
- connection_test.test_model_connection：成功路径与异常分类（中文可操作提示），
  用 fake client 替换，不发真实网络请求；
- is_configured：配置齐全 True / 缺 key False（缓存清除后能感知新值）；
- AppTest 冒烟：已配置环境下主界面正常渲染、侧栏含"模型设置"面板、
  无未捕获异常。

未配置态的引导页渲染（打开 UI 不崩、显示配置表单）依赖全局 .env 状态，
AppTest 进程内模拟成本高，记入 docs/行为测试.md 由真实运行验收。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app.config.settings as settings_mod
import openai
from app.llm import connection_test as ct

# 以模块属性方式引用被测函数，避免 pytest 把 test_ 开头的导入误收集为用例。
_test_connection = ct.test_model_connection
_APP_ENTRY = Path(__file__).resolve().parent.parent / "streamlit_app.py"


# --- test_model_connection：fake client 注入 ---------------------------------


class _FakeCompletions:
    """模拟 chat.completions.create。"""

    def __init__(self, ok: bool, exc: Exception | None = None) -> None:
        self._ok = ok
        self._exc = exc

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        if self._exc is not None:
            raise self._exc

        class _Msg:
            content = "pong"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    """模拟 openai.OpenAI：记录构造参数，按用例成功/抛错。"""

    last_kwargs: dict = {}
    behavior: tuple = (True, None)  # (ok, exc)

    def __init__(self, **kwargs):  # noqa: ANN003
        _FakeClient.last_kwargs = kwargs
        ok, exc = _FakeClient.behavior
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(ok, exc)


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """把 connection_test 的 OpenAI 客户端替换为 fake。"""
    monkeypatch.setattr(ct, "OpenAI", _FakeClient)
    _FakeClient.behavior = (True, None)
    return _FakeClient


def test_connection_success(fake_client) -> None:
    """成功路径：返回 True，说明含模型名，且 base_url/key 传给客户端。"""
    ok, msg = _test_connection(
        "https://api.example.com", "sk-test", "test-model", timeout_s=5
    )
    assert ok is True
    assert "test-model" in msg
    assert fake_client.last_kwargs["base_url"] == "https://api.example.com"
    assert fake_client.last_kwargs["api_key"] == "sk-test"


def test_connection_incomplete_config() -> None:
    """缺任一配置直接报'配置不完整'，不发请求。"""
    ok, msg = _test_connection("", "sk-x", "m")
    assert ok is False
    assert "配置不完整" in msg


def test_connection_error_classification(fake_client) -> None:
    """异常分类：认证/404/429/连接失败 各给对应中文提示。"""
    import httpx

    def _make(cls, status: int | None):
        """构造 openai SDK 异常（其构造器要求 httpx Response/Request）。"""
        if status is None:
            # APIConnectionError(*, request, message=...) 全关键字构造。
            return cls(request=httpx.Request("POST", "http://x"))
        resp = httpx.Response(
            status, request=httpx.Request("POST", "http://x")
        )
        return cls("boom", response=resp, body=None)

    cases = [
        (openai.AuthenticationError, 401, "密钥无效"),
        (openai.NotFoundError, 404, "接口地址或模型名不存在"),
        (openai.RateLimitError, 429, "限流"),
        (openai.APIConnectionError, None, "无法连接"),
    ]
    for exc_type, status, keyword in cases:
        fake_client.behavior = (False, _make(exc_type, status))
        ok, msg = _test_connection("u", "k", "m")
        assert ok is False
        assert keyword in msg, f"{exc_type.__name__} 应提示 {keyword}，实得 {msg}"


def test_classify_error_unknown() -> None:
    """未知异常兜底为'连接失败 + 异常类型名'。"""
    msg = ct._classify_error(RuntimeError("boom"))
    assert "连接失败" in msg and "RuntimeError" in msg


# --- is_configured ------------------------------------------------------------


def test_is_configured_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实测试环境 .env 已配置 → True。"""
    settings_mod.get_settings.cache_clear()
    assert settings_mod.is_configured() is True


def test_is_configured_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings 抛 ConfigError → False（不向外抛异常）。"""
    settings_mod.get_settings.cache_clear()  # 在替换前清缓存（lru_cache 函数还在）

    def _raise() -> None:
        raise settings_mod.ConfigError("配置缺失")

    monkeypatch.setattr(settings_mod, "get_settings", _raise)
    assert settings_mod.is_configured() is False
    # monkeypatch teardown 自动还原 get_settings，无需手动处理。


# --- AppTest 冒烟（已配置态） --------------------------------------------------


def test_app_smoke_with_sidebar_panels() -> None:
    """已配置环境：主界面渲染无异常，侧栏含模型设置面板。"""
    settings_mod.get_settings.cache_clear()
    at = AppTest.from_file(str(_APP_ENTRY), default_timeout=30)
    at.run()
    assert not at.exception, (
        f"主界面不应有未捕获异常：{at.exception[0].value if at.exception else ''}"
    )
    # 侧栏模型设置面板（已配置态）渲染了接口地址 caption。
    captions = [c.value for c in at.sidebar.caption]
    assert any("接口地址" in c for c in captions), (
        f"侧栏应含模型设置面板信息，实得 captions={captions}"
    )


def test_app_onboarding_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """未配置态：引导页出现（快速上手标题），不构造 ChatService。

    关键：AppTest 每次执行脚本用独立 globals，脚本顶层的
    ``from app.config.settings import is_configured`` 会从 sys.modules 的
    settings 模块重新取属性——因此 monkeypatch settings_mod.is_configured
    才能影响脚本执行（monkeypatch streamlit_app 模块属性无效）。
    """
    settings_mod.get_settings.cache_clear()
    monkeypatch.setattr(settings_mod, "is_configured", lambda: False)
    at = AppTest.from_file(str(_APP_ENTRY), default_timeout=30)
    at.run()
    assert not at.exception, (
        f"引导页不应有未捕获异常：{at.exception[0].value if at.exception else ''}"
    )
    subheaders = [s.value for s in at.subheader]
    assert any("快速上手" in s for s in subheaders), (
        f"未配置时应渲染引导页，实得 subheaders={subheaders}"
    )
    # 引导页不应出现主界面对话区（ChatService 未构造）。
    assert "对话" not in subheaders, (
        f"未配置时不应进入主界面（不应渲染'对话'区），实得 {subheaders}"
    )
    assert "service" not in at.session_state, "未配置时不应构造 ChatService"
