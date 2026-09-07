"""模型设置侧栏面板：UI 表单配置模型 API，回写 .env（决策 1）。

交互设计（2026-09-07 拍板）：
- 侧栏折叠面板（expander），随时可改；
- 已配置态：显示掩码（不回显明文 key），"修改配置"展开表单；
- 未配置态：直接展开表单；
- "测试连接"按钮：保存前发一次轻量真实调用验证（决策 2）；
- 保存：env_io 回写 .env → 清 get_settings 缓存 → 重建 ChatService → rerun。

密钥仍只存 .env、仍由 pydantic-settings 读取（AGENTS.md 规则 2）；本面板只是
替代"用户手编 .env 文件"。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config.env_io import mask_secret, update_env_file
from app.config.settings import ConfigError, get_settings, is_configured
from app.llm.connection_test import test_model_connection

# 与 Settings（pydantic-settings env_file=".env"）同一相对基准，保证读写同源。
ENV_PATH = Path(".env")

# 表单可配置的 .env 键（其余键不受影响，env_io 逐行替换时原样保留）。
_MODEL_ENV_KEYS = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "DEFAULT_MODEL",
                   "DEFAULT_TEMPERATURE")


def _apply_model_config(base_url: str, api_key: str, model: str, temperature: str) -> None:
    """把表单值写入 .env 并使配置与 ChatService 生效（缓存清除 + 重建）。"""
    updates: dict[str, str] = {
        "OPENAI_BASE_URL": base_url.strip(),
        "DEFAULT_MODEL": model.strip(),
        "DEFAULT_TEMPERATURE": temperature.strip() or "0.2",
    }
    # key 为空输入 = 不改动现有 key（表单不回显明文，空即"保持"）。
    if api_key.strip():
        updates["OPENAI_API_KEY"] = api_key.strip()
    update_env_file(ENV_PATH, updates)
    get_settings.cache_clear()
    # ChatService 构造时已绑定旧 model 客户端，必须重建；对话历史保留在
    # st.session_state.messages，不受影响。
    st.session_state.pop("service", None)


def _config_error_text() -> str | None:
    """返回当前配置缺失说明（已配置返回 None）。"""
    try:
        get_settings()
        return None
    except ConfigError as exc:
        return str(exc)


def render_model_settings() -> None:
    """渲染侧栏"模型设置"折叠面板（调用方置于 st.sidebar 内）。"""
    configured = is_configured()
    error_text = _config_error_text() if configured else None

    header = "模型设置（已配置）" if configured else "模型设置（未配置）"
    with st.expander(header, expanded=not configured):
        if configured and error_text is None:
            s = get_settings()
            st.caption(
                f"接口地址：`{s.openai_base_url}` · 模型：`{s.default_model}`\n\n"
                f"密钥：`{mask_secret(s.openai_api_key)}`"
            )
            if not st.toggle("修改配置", key="show_model_form"):
                return

        base_url = st.text_input(
            "接口地址（OPENAI_BASE_URL）",
            value=getattr(get_settings(), "openai_base_url", "") if configured else "",
            placeholder="如 https://api.deepseek.com（Kimi 填 https://api.moonshot.cn/v1）",
            key="model_base_url",
        )
        api_key = st.text_input(
            "API 密钥（OPENAI_API_KEY）",
            value="",
            type="password",
            help="已配置时留空 = 保持现有密钥不变；密钥只写入本地 .env，不进 git。",
            key="model_api_key",
        )
        model = st.text_input(
            "模型名（DEFAULT_MODEL）",
            value=getattr(get_settings(), "default_model", "") if configured else "",
            placeholder="如 deepseek-chat / kimi-k2-0711-preview / gpt-4o-mini",
            key="model_name",
        )
        temperature = st.text_input(
            "温度（DEFAULT_TEMPERATURE，0~2）",
            value=(
                str(getattr(get_settings(), "default_temperature", 0.2))
                if configured else "0.2"
            ),
            key="model_temperature",
        )

        col_test, col_save = st.columns(2)
        if col_test.button("测试连接", help="发送一次最小真实调用，验证配置可用"):
            with st.spinner("连接测试中……"):
                # 未填新 key 时用现有配置中的密钥测试（表单不回显明文）。
                if api_key.strip():
                    ok, msg = test_model_connection(base_url, api_key, model)
                else:
                    try:
                        s = get_settings()
                        ok, msg = test_model_connection(
                            base_url, s.openai_api_key, model
                        )
                    except ConfigError:
                        ok, msg = False, "密钥未填写：请输入 API 密钥后再测试。"
            (col_test.success if ok else col_test.error)(msg)

        if col_save.button("保存配置", type="primary"):
            try:
                _apply_model_config(base_url, api_key, model, temperature)
            except OSError as exc:
                st.error(f"写入 .env 失败：{exc}")
                return
            st.success("配置已保存到 .env 并生效。")
            st.rerun()
