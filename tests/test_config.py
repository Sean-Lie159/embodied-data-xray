"""app/config 模块的单元测试。

不依赖真实网络或密钥；通过显式传入字段值覆盖 .env 来控制校验行为。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import ConfigError, Settings, get_settings


def _settings_with(**overrides) -> Settings:
    """构造 Settings 并强制忽略 .env 与 OS 环境变量，避免测试被真实配置干扰。"""
    base = dict(
        openai_api_key="sk-test-key",
        openai_base_url="https://api.moonshot.cn/v1",
        default_model="kimi-k2-0711-preview",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_settings_loads_valid_values() -> None:
    s = _settings_with()
    assert s.openai_api_key == "sk-test-key"
    assert s.openai_base_url == "https://api.moonshot.cn/v1"
    assert s.default_model == "kimi-k2-0711-preview"
    assert s.default_temperature == 0.2
    assert s.max_turns == 15
    assert s.max_rows_in_context == 200
    assert s.output_dir == "outputs"


def test_settings_missing_api_key_raises() -> None:
    with pytest.raises(ConfigError) as exc:
        _settings_with(openai_api_key="")
    assert "OPENAI_API_KEY" in str(exc.value)
    assert ".env" in str(exc.value)


def test_settings_missing_base_url_raises() -> None:
    with pytest.raises(ConfigError) as exc:
        _settings_with(openai_base_url="")
    assert "OPENAI_BASE_URL" in str(exc.value)


def test_settings_missing_model_raises() -> None:
    with pytest.raises(ConfigError) as exc:
        _settings_with(default_model="")
    assert "DEFAULT_MODEL" in str(exc.value)


def test_output_path_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "out"
    s = _settings_with(output_dir=str(target))
    created = s.output_path()
    assert created.exists()
    assert created.is_dir()
