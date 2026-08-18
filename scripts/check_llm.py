"""模型连通性检查脚本。

向 .env 配置的模型发送"你好"并打印回复，用于验证 app/config 与 app/llm
两个模块的配置读取与模型装配是否正确、网络与密钥是否可用。

用法：
    python scripts/check_llm.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 保证无论从哪个目录运行都能导入项目根下的 app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Agent, Runner  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.llm import build_model  # noqa: E402


async def _main() -> None:
    settings = get_settings()
    model = build_model(settings)

    agent = Agent(
        name="connectivity_check",
        instructions="你是一个乐于助人的助手，请用中文简洁回答。",
        model=model,
    )

    print(f"模型: {settings.default_model}")
    print(f"端点: {settings.openai_base_url}")
    print("发送: 你好")

    result = await Runner.run(agent, "你好")
    reply = (result.final_output or "").strip()
    print(f"回复: {reply}")

    if not reply:
        raise SystemExit("模型返回了空回复，连通性可能存在问题。")


if __name__ == "__main__":
    asyncio.run(_main())
