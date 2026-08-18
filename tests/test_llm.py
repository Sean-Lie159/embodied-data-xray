"""app/llm 模块的单元测试。

通过 monkeypatch 拦截客户端与 Model 的构造，验证 build_model 的装配逻辑，
不发起真实网络请求。
"""

from __future__ import annotations

import pytest

from app.config.settings import ConfigError, Settings
from app.llm import factory


def _settings(**overrides) -> Settings:
    base = dict(
        openai_api_key="sk-test-key",
        openai_base_url="https://api.moonshot.cn/v1",
        default_model="kimi-k2-0711-preview",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


class _FakeClient:
    """代替 AsyncOpenAI 的假客户端，仅用于记录构造参数。"""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeModel:
    """代替 OpenAIChatCompletionsModel 的假模型，仅用于记录构造参数。"""

    def __init__(self, *, model, openai_client) -> None:
        self.model = model
        self.openai_client = openai_client


def test_build_model_assembles_openai_compatible_model(monkeypatch) -> None:
    monkeypatch.setattr(factory, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(factory, "OpenAIChatCompletionsModel", _FakeModel)

    settings = _settings()
    model = factory.build_model(settings)

    assert isinstance(model, _FakeModel)
    assert model.model == "kimi-k2-0711-preview"
    # 客户端收到了正确的密钥与端点
    assert model.openai_client.kwargs["api_key"] == "sk-test-key"
    assert model.openai_client.kwargs["base_url"] == "https://api.moonshot.cn/v1"


def test_build_model_raises_when_config_incomplete(monkeypatch) -> None:
    monkeypatch.setattr(factory, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(factory, "OpenAIChatCompletionsModel", _FakeModel)

    # 构造一个绕过了 Settings 校验的残缺对象，验证工厂自身的防御性检查。
    incomplete = _settings()
    incomplete.openai_api_key = ""  # 直接改字段，绕过 pydantic 的构造期校验
    with pytest.raises(ConfigError):
        factory.build_model(incomplete)
