"""模型接口层：单一工厂函数。

为上层（agent / services / tools）产出一个 openai-agents 可用的 ``Model`` 对象，
屏蔽具体的 API 端点与密钥来源。上层只依赖本模块的 :func:`build_model`，
不关心底层走的是 DeepSeek、Kimi 还是 OpenAI 端点——切换服务商只需修改 ``.env``。
"""

from __future__ import annotations

from agents import AsyncOpenAI, Model, OpenAIChatCompletionsModel, set_tracing_disabled

from app.config.settings import ConfigError, Settings


def build_model(settings: Settings) -> Model:
    """根据 ``settings`` 构造并返回 OpenAI 兼容的 ``Model`` 对象。

    Args:
        settings: 应用配置，需已通过校验（含 OPENAI_API_KEY / OPENAI_BASE_URL /
            DEFAULT_MODEL）。

    Returns:
        openai-agents 可用的 ``Model`` 实例，可直接传给 ``Agent(model=...)``。

    Raises:
        ConfigError: 配置不完整时抛出中文错误说明。
    """
    if not settings.openai_api_key or not settings.openai_base_url or not settings.default_model:
        raise ConfigError(
            "模型配置不完整，请检查 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL / "
            "DEFAULT_MODEL。"
        )

    # 未配置 OpenAI 官方 key 时，必须禁用 tracing，否则会因鉴权失败返回 401。
    set_tracing_disabled(disabled=True)

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    model = OpenAIChatCompletionsModel(
        model=settings.default_model,
        openai_client=client,
    )
    return model
