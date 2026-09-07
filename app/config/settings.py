"""应用配置模块。

基于 pydantic-settings，从 ``.env`` 与 OS 环境变量读取配置，作为全项目配置的
唯一事实来源。任何需要 API 密钥、端点或默认值的地方都应调用
:func:`get_settings`，而不要直接读取 ``os.environ``。

启动时（首次调用 :func:`get_settings`）会校验 API 密钥已配置；若缺失，抛出带
清晰中文提示的 :class:`ConfigError`，方便使用者定位问题。
"""

from __future__ import annotations

import os
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

    # --- 上下文预算（防 input length too long / HTTP 400）--------------------
    # 模型上下文窗口（token）。0 = 按内置模型表推定；认不出时按 256K 兜底
    # （以本项目基准模型 HY3 为准）。**换模型时建议显式配置此项**——
    # 内置表可能过时，且兜底值对小上下文模型并不安全。
    context_window_tokens: int = Field(default=0, ge=0)
    # 上下文可用比例：预算 = 上下文窗口 × 此比例，其余留给 system prompt、
    # 本轮用户输入与模型输出。
    context_budget_ratio: float = Field(default=0.6, gt=0.0, le=1.0)
    # 历史压缩触发阈值占总预算的比例（超出即自动压缩，在送模型之前拦截）。
    history_budget_ratio: float = Field(default=0.75, gt=0.0, le=1.0)
    # 单次工具返回硬上限占总预算的比例（第 2 层兜底截断）。
    tool_output_budget_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    # 自动压缩开关。关闭后仅能手动触发，不保证不再撞上下文上限。
    history_compaction_enabled: bool = True
    # 压缩时保留最近若干轮的完整工具返回（保持上下文连贯）。
    history_keep_recent_turns: int = Field(default=3, ge=0)

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
    # 基线推荐：样本数低于该值的流视为静态/稀疏流，不作为对齐基线候选
    # （真实案例：tf_static.jsonl 仅 105 点被推上基线，残差被严重放大）。
    sync_static_min_samples: int = Field(default=100, ge=2)
    # 基线推荐：时间覆盖率（该流跨度 / 最大跨度）低于该值不作为候选。
    sync_baseline_min_coverage: float = Field(default=0.95, ge=0.0, le=1.0)
    # gap 定位（locate_gaps=True）每流最多返回的缺口条数（控制上下文体积）。
    sync_gap_report_limit: int = Field(default=20, ge=1)
    # 突发型流判定：平均间隔 ≥ 中位间隔 × 该倍数即判 burst（突发间静默拉高均值；
    # 真实案例：IMU 突发内 1.79µs、均值 1.26ms，比值约 700）。判定依据见
    # docs/时间对齐能力改造设计.md 4.5 节（实现时修正为 mean/med 比值判据）。
    sync_burst_interval_ratio: float = Field(default=3.0, gt=1.0)

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

    # --- 实例部署模式（局域网共享给同事，docs/实例部署模式设计.md）------------
    # 实例模式：模型配置由部署者预设（部署机本地 .env），UI 隐藏模型设置入口，
    # 访问者不可见不可改（key 仅 owner 持有）。克隆用户默认 False（阶段 A 形态）。
    instance_mode: bool = False
    # 可选访问口令：非空时局域网访问者需先输入口令（简单门禁，非强鉴权）。
    access_password: str = ""

    # --- 可选：Token 成本估算（美元/百万 token）------------------------------
    # 两项都 >0 时才启用成本估算；默认 0 = 未配置，不显示成本。价格是易变信息，
    # 不硬编码，由用户在 .env 按当前服务商定价填写。
    price_input_per_mtok: float = Field(default=0.0, ge=0.0)
    price_output_per_mtok: float = Field(default=0.0, ge=0.0)

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
    注意：通过 ``env_io.update_env_file`` 回写 .env 后，必须调用
    ``get_settings.cache_clear()`` 再取新配置（UI 模型设置面板已处理）。
    """
    return Settings()


def is_configured() -> bool:
    """判断模型配置是否齐全（不抛异常，供 UI 引导页判定使用）。

    缺 key 时 ``get_settings()`` 抛 :class:`ConfigError`，本函数将其转为
    False，使 UI 能"未配置时渲染引导页"而非崩溃。同时缓存清除后的重新
    解析也在这里发生（保存配置后 UI 调 cache_clear，下一次 is_configured
    / get_settings 即读新值）。

    Returns:
        True 表示 get_settings() 可用（配置齐全）。
    """
    try:
        get_settings()
        return True
    except ConfigError:
        return False


def get_instance_mode() -> bool:
    """读取实例模式开关（不抛异常，供 UI 渲染前判定）。

    实例标志必须**独立于模型配置完整性**生效：部署实例在 key 尚未配置时
    （get_settings 抛 ConfigError），同事同样不能看到配置表单——因此
    ConfigError 时回退读原始环境变量（与 pydantic-settings 同源）。

    Returns:
        INSTANCE_MODE 是否开启。
    """
    try:
        return bool(get_settings().instance_mode)
    except ConfigError:
        return os.getenv("INSTANCE_MODE", "").strip().lower() in ("1", "true", "yes", "on")


def get_access_password_lenient() -> str:
    """读取访问口令（不抛异常；key 缺失时回退环境变量，理由同上）。

    Returns:
        ACCESS_PASSWORD 值（空串 = 门禁未启用）。
    """
    try:
        return get_settings().access_password
    except ConfigError:
        return os.getenv("ACCESS_PASSWORD", "")
