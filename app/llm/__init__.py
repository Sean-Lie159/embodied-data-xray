"""模型接口层：为上层产出 OpenAI 兼容的 Model 对象。"""

from app.llm.factory import build_model

__all__ = ["build_model"]
