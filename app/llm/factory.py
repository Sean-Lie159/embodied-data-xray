"""模型接口层：单一工厂函数。

为上层（agent / services / tools）产出一个 openai-agents 可用的 ``Model`` 对象，
屏蔽具体的 API 端点与密钥来源。上层只依赖本模块的 :func:`build_model`，
不关心底层走的是 DeepSeek、Kimi 还是 OpenAI 端点——切换服务商只需修改 ``.env``。
"""

from __future__ import annotations

from agents import (
    AsyncOpenAI,
    Model,
    ModelSettings,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

from app.config.settings import ConfigError, Settings

# 允许的推理档位（与网关/上游约定一致）。其余值视为"不干预"。
_VALID_REASONING_EFFORTS: tuple[str, ...] = ("high", "low", "off")


def build_reasoning_extra_body(reasoning_effort: str) -> dict[str, object]:
    """把 ``REASONING_EFFORT`` 配置转成请求体的额外字段。

    为什么需要显式传：本项目基准链路（CodeBuddy 网关 → hy3）在**请求未带
    ``reasoning_effort`` 时会自动注入 "high"**，即推理模式由网关替客户端开启。
    实测（2026-09-11）hy3 默认档的思考 token 占总输出 91%~96%——一次 300 字的
    分析问答要烧约 900 个看不见的思考 token，按 13 ms/token 折算，等待时间
    几乎全在思考上。客户端显式传顶层 ``reasoning_effort`` 即可覆盖网关默认值。

    取值语义：
        "high" / "low"：显式指定档位（"low" 实测思考 token 约降 24%）；
        "off"：显式请求关闭思考链；
        ""（留空）：返回空 dict，**完全不干预**，由网关按自身默认处理
            （当前等价于 "high"，但把选择权留给网关）。

    Args:
        reasoning_effort: 配置值（大小写不敏感，自动去空白）。

    Returns:
        可直接展开进 ``ModelSettings.extra_body`` 的 dict；值为空或不识别时
        返回空 dict（不注入任何字段）。
    """
    value = (reasoning_effort or "").strip().lower()
    if value not in _VALID_REASONING_EFFORTS:
        return {}
    return {"reasoning_effort": value}


def build_model(settings: Settings) -> Model:
    """根据 ``settings`` 构造并返回 OpenAI 兼容的 ``Model`` 对象。

    Args:
        settings: 应用配置，需已通过校验（含 OPENAI_API_KEY / OPENAI_BASE_URL /
            DEFAULT_MODEL）。

    Returns:
        openai-agents 可用的 ``Model`` 实例，可直接传给 ``Agent(model=...)``。
        推理档位（``REASONING_EFFORT``）经 ``ModelSettings.extra_body`` 注入。

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


def build_model_settings(settings: Settings) -> ModelSettings | None:
    """构造 Agent 级模型设置（当前仅承载推理档位）。

    推理档位必须经请求体传递（``extra_body``），而 ``OpenAIChatCompletionsModel``
    本身不提供"每次请求默认参数"的入口——SDK 的正规做法是在 ``Agent`` 上设
    ``model_settings``，由 Runner 在每次调用时合并进请求。

    为什么不在 build_model 里挂：Model 对象的属性不是 SDK 的配置面，
    依赖它属于未定义行为；Agent 的 model_settings 是公开契约。

    Args:
        settings: 应用配置。

    Returns:
        ModelSettings（含 extra_body）；未配置推理档位时返回 None（不干预，
        行为与改动前完全一致）。
    """
    extra_body = build_reasoning_extra_body(
        getattr(settings, "reasoning_effort", "")
    )
    if not extra_body:
        return None
    return ModelSettings(extra_body=extra_body)
