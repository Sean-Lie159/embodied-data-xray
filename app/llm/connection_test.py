"""模型连接测试：UI"测试连接"按钮的轻量真实调用（决策 2）。

在保存配置前发一次最小 chat 调用（max_tokens=5），验证 base_url / api_key /
model 三件套可用，避免"填错 key 用到一半才发现"。纯函数、不 import
streamlit（app/llm 禁止），可独立单测。

失败时返回结构化中文原因（认证 / 连接 / 超时 / 模型名等分类），UI 直接转达。
"""

from __future__ import annotations

import openai
from openai import OpenAI

# 轻量调用的超时（秒）：连接测试不应让用户久等。
_DEFAULT_TIMEOUT_S = 15.0


def test_model_connection(
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[bool, str]:
    """用给定配置发一次最小 chat 调用，验证连通性。

    Args:
        base_url: OpenAI 兼容接口地址。
        api_key: API 密钥。
        model: 模型名。
        timeout_s: 请求超时（秒）。

    Returns:
        (是否成功, 中文说明)。成功时说明含模型名；失败时说明为分类后的
        中文原因（不含堆栈），可直接在 UI 展示。
    """
    if not base_url or not api_key or not model:
        return False, "配置不完整：接口地址、密钥、模型名均需填写。"
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        content = (resp.choices[0].message.content or "").strip()
        return True, (
            f"连接成功：模型 {model} 已响应"
            + (f"（返回：{content[:20]}）" if content else "")
        )
    except Exception as exc:  # noqa: BLE001  # 失败分类转中文，UI 直接展示
        return False, _classify_error(exc)


def _classify_error(exc: Exception) -> str:
    """把 openai SDK 异常分类为可操作的中文提示。"""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
        return "密钥无效或无权限（401/403）：请检查 API Key 是否正确、是否已开通该模型。"
    if name == "NotFoundError" or status == 404:
        return (
            "接口地址或模型名不存在（404）：请检查 BASE_URL 是否为主接口地址"
            "（通常含 /v1），以及模型名拼写。"
        )
    if name == "RateLimitError" or status == 429:
        return "触发限流（429）：密钥有效但配额/频率受限，稍后重试或检查套餐。"
    if name == "APITimeoutError":
        return f"请求超时（>{_DEFAULT_TIMEOUT_S:.0f}s）：网络不通或服务响应慢。"
    if name in ("APIConnectionError",):
        return "无法连接接口地址：请检查 BASE_URL 拼写与本机网络/代理设置。"
    if name == "BadRequestError" or status == 400:
        return f"请求被拒绝（400）：模型名可能不正确（{exc}）。"
    return f"连接失败：{type(exc).__name__} {exc}"
