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
    # 主表装载行数上限（安全阀）。默认全量装载；仅当单表行数超过此阈值时截断，
    # 且返回必须同时含 rows_total / rows_loaded 并明确提示截断。默认值设得足够大，
    # 使常规采集数据（数万行）不被静默截断。
    max_rows_in_context: int = 500_000
    max_turns: int = 15

    # --- 质检阈值（check_temporal_sync / check_sensor_sanity 可调阈值）--------
    # 丢帧率触发 fail 的阈值（0.02 = 2%）。
    sync_frame_loss_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    # 各流时间戳最大允许偏差（毫秒）。
    sync_max_skew_ms: float = Field(default=10.0, ge=0.0)
    # 漂移检测的窗口数（把每个 episode 按时间切为若干窗口）。
    sync_drift_windows: int = Field(default=10, ge=1)
    # 漂移判定斜率阈值（ms/s）：窗口残差偏移线性拟合斜率超过则判 fail。
    sync_drift_slope_ms_per_s: float = Field(default=0.5, ge=0.0)
    # pass 残差阈值：最高帧率流采样间隔的比例（0.5 = 采样间隔的一半）。
    sync_residual_ratio: float = Field(default=0.5, ge=0.0)

    # --- 传感器合理性（check_sensor_sanity）阈值 -----------------------------
    # 静止段判定：滑动窗口内加速度计模长的方差低于该值视为静止（g² 或 (m/s²)² 量级）。
    sanity_static_var_threshold: float = Field(default=0.02, ge=0.0)
    # 静止段占比低于该值时，依赖静止段的检查降级为 warn（数据可能全程在运动）。
    sanity_static_ratio_warn: float = Field(default=0.05, ge=0.0, le=1.0)
    # 静止段加速度模长相对重力参考值的允许相对偏差（如 0.1 = 10%）。
    sanity_gravity_tolerance: float = Field(default=0.1, ge=0.0)
    # 量程饱和削顶判定：连续重复出现的极值点（等于信号最大/最小值）比例超过该值 → fail。
    sanity_saturation_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    # 恒定通道判定：归一化方差低于该值视为传感器掉线/恒定。
    sanity_constant_var: float = Field(default=1e-6, ge=0.0)
    # NaN/Inf 比例超过该值 → fail。
    sanity_nan_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    # 静止段窗口回退采样率（Hz）：流登记表无 measured_rate 时用此值估算 1 秒窗口。
    sanity_static_window_rate: float = Field(default=100.0, ge=1.0)

    # --- 任务级统计（compute_stats）阈值 -------------------------------------
    # 离群 episode 检测（IQR 法）的 k 值：Q1 - k*IQR / Q3 + k*IQR 之外视为离群。
    stats_outlier_k: float = Field(default=1.5, ge=0.0)

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
