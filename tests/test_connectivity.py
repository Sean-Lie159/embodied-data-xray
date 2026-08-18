"""模型连通性集成测试。

该测试会真实调用模型 API，依赖有效的 .env 配置与网络。默认跳过，避免在
无密钥或离线环境下让测试套件失败。确需验证时设置环境变量
``RUN_LLM_TESTS=1`` 再运行：``pytest -m llm``。
"""

from __future__ import annotations

import asyncio
import os

import pytest
from agents import Agent, Runner

from app.config import get_settings
from app.llm import build_model


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1",
    reason="未设置 RUN_LLM_TESTS=1，跳过真实模型调用测试",
)


def test_llm_connectivity_and_reply() -> None:
    """向 .env 配置的模型发送"你好"，断言返回非空回复。"""
    settings = get_settings()
    model = build_model(settings)
    agent = Agent(
        name="connectivity_check",
        instructions="你是一个乐于助人的助手，请用中文简洁回答。",
        model=model,
    )

    async def _ask() -> str:
        result = await Runner.run(agent, "你好")
        return result.final_output or ""

    reply = asyncio.run(_ask())
    assert reply.strip(), "模型返回了空回复，连通性可能有问题"
    print(f"[connectivity] model={settings.default_model} reply={reply!r}")
