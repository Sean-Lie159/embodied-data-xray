"""应用配置模块。

基于 pydantic-settings，从 ``.env`` 与 OS 环境变量读取配置，作为全项目配置的
唯一事实来源。任何需要 API 密钥、端点或默认值的地方都应调用
:func:`get_settings`，而不要直接读取 ``os.environ``。

启动时（首次调用 :func:`get_settings`）会校验 API 密钥已配置；若缺失，抛出带
清晰中文提示的 :class:`ConfigError`，方便使用者定位问题。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """配置缺失或非法时抛出的错误，message 为可读的中文说明。"""


class Settings(BaseSettings):
    """运行时配置。

    值来自环境变量或项目根目录的 ``.env`` 文件（字段名大小写不敏感）。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- 模型接入（OpenAI 兼容端点）------------------------------------------
    openai_api_key: str = ""
    openai_base_url: str = ""
    default_model: str = ""
    default_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- 数据处理与运行 ------------------------------------------------------
    output_dir: str = "outputs"
    max_rows_in_context: int = 200
    max_turns: int = 15

    @model_validator(mode="after")
    def _validate_required(self) -> Settings:
        """校验必须的模型配置是否齐全，缺失时给出中文报错。"""
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY（模型服务商密钥）")
        if not self.openai_base_url:
            missing.append("OPENAI_BASE_URL（模型服务商接口地址）")
        if not self.default_model:
            missing.append("DEFAULT_MODEL（默认模型名）")

        if missing:
            names = "、".join(missing)
            raise ConfigError(
                f"配置缺失：{names} 未在 .env 中设置。"
                "请复制 .env.example 为 .env 并填写对应值。"
            )
        return self

    def output_path(self) -> Path:
        """返回输出目录的绝对路径，目录不存在时会自动创建。"""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的 :class:`Settings` 实例。

    ``lru_cache`` 保证整个进程内配置只解析一次，避免重复读取文件。
    """
    return Settings()
