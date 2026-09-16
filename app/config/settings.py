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
    # 推理档位（reasoning_effort）：控制模型的思考链深度。
    #
    # 为什么需要显式配置：本项目基准链路是 CodeBuddy 网关 → 混元 hy3，而该网关
    # 在**请求未带该字段时会自动注入 "high"**（见网关 upstream.py 的
    # "Reasoning / chain-of-thought injection"）——即推理模式是网关替我们开的，
    # 客户端不传就吃默认。实测（2026-09-11）：hy3 默认档思考 token 占总输出
    # 91%~96%，一次 300 字的分析问答要烧约 900 个看不见的思考 token，按
    # 13 ms/token 折算，绝大部分等待时间花在思考上而非正文。
    #
    # 取值与实测效果（同一问题，客户端传顶层 reasoning_effort 覆盖网关默认值）：
    #   "high"：网关默认。推理最充分，耗时最长（实测思考 token 最高）。
    #   "low" ：思考 token 约减少 24%（实测 179 → 136），保留推理能力，折中档。
    #   "off" ：网关透传后上游不产生思考链；但**实测未见显著降幅**，且关闭推理
    #           会削弱"识别口径矛盾 / 判断基线合理性"这类需要推断的任务表现，
    #           故默认不采用。
    # 留空（""）：完全不干预，由网关按自己的默认值处理（等价于 high）。
    reasoning_effort: str = ""

    # --- 数据处理与运行 ------------------------------------------------------
    output_dir: str = "outputs"
    # 单文件上传大小上限（MB）：仅 UI 辅助通道，目录型数据集请走路径输入。
    upload_max_mb: int = Field(default=200, ge=1)
    # 流式输出开关：开启时 UI 逐块显示模型正文（首字节即可见）并播报工具调用。
    # 关闭则回退到"等到底再整段显示"的旧行为（流式涉及线程桥接，异常时可一键回退）。
    stream_output_enabled: bool = True
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

    # --- 数据集质检（check_dataset_quality）阈值 -----------------------------
    # 设计依据：docs/标注与质检能力设计.md §5.3。**分层语义是本组配置的核心**：
    # gate_* 属 L1 硬门禁（确定性错误，可判 fail）；diagnostic_* 属 L2/L3
    # 诊断（启发式，仅 warn，**永不自动升级为 fail**）。
    #
    # 为什么必须分层：RDA 作者披露早期版本让 idle_ratio 自动升级为判定，在
    # libero_10 上产生 65% 误报；改成"只有硬门禁能 EXCLUDE"后误报降 97-100%。
    # 根因是阈值与任务相关——70% 空闲比对 push 类任务正常、对 lift 类可疑。
    #
    # L1 硬门禁（确定性）：
    # NaN/Inf 比例超过该值 → fail（与 sanity_nan_ratio 同口径）。
    quality_gate_nan_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    # 帧计数缺口比例（实际行数 vs 时间戳推算应有行数）超过该值 → fail。
    quality_gate_frame_loss_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    # 跨 episode 的 feature/dtype 一致性：不一致列数超过该值时 → fail。
    quality_gate_schema_mismatch_max: int = Field(default=0, ge=0)
    #
    # L2/L3 诊断（启发式，仅 warn）——数值全部可配置且**默认值未经数据集验证**：
    # 动作突波/碰撞判定倍数：θ > 该倍数 × median(|a|)（HF GIGO 公开公式）。
    quality_diagnostic_spike_multiplier: float = Field(default=15.0, gt=1.0)
    # 致动器饱和判定：|a_t - q_{t+1}| 超过该角度（度）视为饱和（HF GIGO 公开公式）。
    quality_diagnostic_saturation_deg: float = Field(default=7.0, gt=0.0)
    # 低速/空闲判定阈值（归一化速度，与 HF GIGO 同口径）。
    quality_diagnostic_idle_speed: float = Field(default=0.1, ge=0.0)
    # 空闲比 warn 阈值（**必须按数据集调整**：高空闲比对 push 类任务可能完全正常）。
    quality_diagnostic_idle_ratio_warn: float = Field(default=0.60, ge=0.0, le=1.0)
    # 视频过暗惩罚阈值（灰度均值 0-255；HF GIGO 刻意只罚过暗、不罚过曝以免误报）。
    quality_diagnostic_dark_mean: float = Field(default=50.0, ge=0.0, le=255.0)
    # 视频模糊度：拉普拉斯方差低于该值判 warn（核 [0,1,0;1,-4,1;0,1,0]）。
    quality_diagnostic_blur_variance: float = Field(default=100.0, ge=0.0)
    # 路径效率 warn 阈值：clip(D/L, 0, 1)，D 为直线距离、L 为实际路径长度；
    # 低于该值提示"犹豫/绕路"（HF GIGO 公开公式）。
    quality_diagnostic_path_efficiency: float = Field(default=0.30, ge=0.0, le=1.0)
    # 动作抖动判定：**高频残差能量占总能量的比例**超过该值判 warn（0.30=30%）。
    # **注意**："Consistency Matters"(arXiv:2412.14309) 明确反对用绝对值判定
    # 抖动，主张按数据分布分组判断；此处用无量纲的方差比例，且只 warn。
    # **不用 P95/median 比值的原因**（实现时实测发现）：平滑的高频正弦其
    # jerk 处处均匀，P95/median ≈ 1.2，该判据完全无法发现平滑高频运动，
    # 而那正是"抖动"的典型形态。改用"低频趋势（5 点滑均）+ 高频残差"分解。
    quality_diagnostic_jerk_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    # 轨迹停顿（robot-induced pause）判定：连续低速步数超过该值判 warn。
    quality_diagnostic_pause_steps: int = Field(default=30, ge=1)
    # 高频振动（arm shaking）判定：加速度二阶差分的高频能量占比超过该值判 warn。
    quality_diagnostic_shake_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    # 诊断明细每项最多列出的 episode 数（控制上下文体积）。
    quality_diagnostic_report_limit: int = Field(default=10, ge=1)

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
