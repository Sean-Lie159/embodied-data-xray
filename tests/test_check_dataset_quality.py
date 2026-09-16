"""check_dataset_quality 测试：分层质检（硬门禁 + 诊断项）。

设计依据：``docs/标注与质检能力设计.md`` §5.3。

**本文件最重要的测试是分层不变量的守护**（见
``test_diagnostics_never_cause_fail`` 系列）：诊断项的 warn **永不**把
判定升级为 fail。这是直接针对 RDA 披露的 65% 误报事故设的回归——它比任何
单条规则的测试都重要，因为规则数值将来会调整，而分层语义不该变。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.config import get_settings
from app.tools import annotation_store as ann  # noqa: F401  (确保包导入可用)
from app.tools.check_dataset_quality import check_dataset_quality_impl


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    """默认配置（阈值来自 settings.py）。"""
    return get_settings()


def _ctx(df: pd.DataFrame | None, dataset_id: str = "demo", **meta) -> RunContext:
    """构造 RunContext。"""
    ctx = RunContext()
    ctx.df = df
    ctx.dataset_id = dataset_id
    ctx.meta = dict(meta)
    if df is not None:
        ctx.meta.setdefault("columns", [str(c) for c in df.columns])
    return ctx


def _clean_df(n: int = 200, freq: float = 0.1) -> pd.DataFrame:
    """构造一份"干净"的数据（无缺失、时间戳单调、动作平滑）。"""
    rng = np.random.default_rng(42)
    t = np.arange(n) * freq
    # 平滑动作：低频正弦 + 极小噪声（避免触发抖动/突波诊断）。
    base = np.sin(2 * np.pi * 0.2 * t)
    return pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp": t,
        "fps": [1 / freq] * n,
        "action_joint0": base,
        "action_joint1": np.cos(2 * np.pi * 0.2 * t),
        "action_joint2": base * 0.5,
    })


# ---------------------------------------------------------------------------
# 分层不变量（最重要的守护）
# ---------------------------------------------------------------------------


def test_clean_data_passes_all_layers(settings) -> None:
    """干净数据的 gate 与 diagnostics 都应通过。"""
    res = check_dataset_quality_impl(_ctx(_clean_df()), settings)
    assert res["success"] is True
    assert res["gate"]["result"] == "pass", res["gate"]["failed"]
    assert res["result"] in ("pass", "warn")


def test_diagnostics_never_cause_fail_nan_free_but_noisy(settings) -> None:
    """**分层不变量**：即使诊断项全部报警，只要硬门禁通过，判定就不得是 fail。

    构造"硬门禁全过但动作极不规整"的数据（剧烈抖动 + 大量停顿 + 突波）。
    """
    n = 300
    rng = np.random.default_rng(0)
    t = np.arange(n) * 0.1
    # 高频抖动 + 随机大跳变：必然触发 jerk / spike / shake 诊断。
    noisy = np.sin(2 * np.pi * 5.0 * t) * 10 + rng.normal(0, 0.5, n)
    noisy[::20] += 50.0  # 周期性强突波
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp": t,
        "action_joint0": noisy,
        "action_joint1": noisy * 0.9,
        "action_joint2": noisy * 1.1,
    })
    res = check_dataset_quality_impl(_ctx(df), settings)

    # 硬门禁必须通过（无 NaN、时间戳单调、无丢帧、schema 一致、fps 合法）。
    assert res["gate"]["result"] == "pass", res["gate"]
    # 诊断项应当至少有一项报警，以证明这个用例真的"有噪声"。
    assert res["diagnostics"]["result"] == "warn", res["diagnostics"]
    # **关键断言**：判定不得升级为 fail。
    assert res["result"] == "warn"
    assert res["result"] != "fail"


def test_diagnostics_do_not_appear_in_gate(settings) -> None:
    """诊断规则名不得出现在 gate.checks 中（两层职责不得混淆）。"""
    res = check_dataset_quality_impl(_ctx(_clean_df()), settings)
    gate_names = set(res["gate"]["checks"])
    diag_names = set(res["diagnostics"]["checks"])
    assert diag_names == {
        "idle_ratio", "action_spike", "actuator_saturation", "action_jerk",
        "robot_induced_pause", "arm_shaking", "path_efficiency", "visual_quality",
    }
    assert not (gate_names & diag_names), "同一规则不得同时出现在两层"


def test_gate_failure_causes_fail(settings) -> None:
    """硬门禁失败时必须判 fail（分层不等于永不失败）。"""
    df = _clean_df()
    df.loc[10:30, "action_joint0"] = np.nan  # NaN 比例约 10% > 5%
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["gate"]["result"] == "fail"
    assert "nan_inf" in res["gate"]["failed"]
    assert res["result"] == "fail"


def test_diagnostics_note_is_communicated(settings) -> None:
    """诊断层必须带"永不升级为 fail"的说明（供模型正确转述）。"""
    res = check_dataset_quality_impl(_ctx(_clean_df()), settings)
    assert "永不自动升级为 fail" in res["diagnostics"]["note"]


def test_user_message_separates_gate_and_diagnostics(settings) -> None:
    """user_message 必须分别转述 gate 与 diagnostics（纪律 17 的落地）。"""
    n = 300
    rng = np.random.default_rng(1)
    t = np.arange(n) * 0.1
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp": t,
        "action_joint0": np.sin(2 * np.pi * 5 * t) * 10 + rng.normal(0, 0.5, n),
        "action_joint1": rng.normal(0, 1, n),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    msg = res["user_message"]
    assert "硬门禁" in msg
    assert "诊断项" in msg
    # 必须明确诊断项不等于数据有问题。
    assert "不等于数据有问题" in msg


# ---------------------------------------------------------------------------
# L1 硬门禁：逐条
# ---------------------------------------------------------------------------


def test_gate_nan_inf_detects_and_reports_per_column(settings) -> None:
    """NaN/Inf 检出并给出逐列比例。"""
    df = _clean_df(100)
    df.loc[0:9, "action_joint0"] = np.nan       # 10%
    df.loc[50, "action_joint1"] = np.inf
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["nan_inf"]
    assert chk["result"] == "fail"
    assert "action_joint0" in chk["per_column"]
    assert chk["threshold"] == settings.quality_gate_nan_ratio


def test_gate_timestamp_backwards_fails(settings) -> None:
    """时间戳回退判 fail（时钟跳变/乱序写入）。"""
    df = _clean_df(50)
    df.loc[25, "timestamp"] = 0.0  # 回退
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["timestamp_monotonic"]
    assert chk["result"] == "fail"
    assert chk["backwards_count"] >= 1
    assert "timestamp_monotonic" in res["gate"]["failed"]


def test_gate_timestamp_duplicates_only_warn(settings) -> None:
    """重复时间戳只是 warn（合法但需确认），不得判 fail。"""
    df = _clean_df(50)
    df.loc[25, "timestamp"] = df.loc[24, "timestamp"]
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["timestamp_monotonic"]
    assert chk["result"] == "warn"
    assert chk["duplicate_count"] >= 1
    assert res["result"] != "fail"  # 重复不应导致 fail


def test_gate_frame_loss_detects_uniform_loss_via_declared_fps(settings) -> None:
    """**均匀丢帧**：隔行抽掉一半时，只有靠声明 fps 才能检出。

    这是对实现中一个真实缺陷的回归——初版只用"中位间隔外推"，隔行抽掉数据会
    让中位间隔翻倍、应有行数随之减半，丢帧被自洽地掩盖（自我掩饰的判据）。
    修正后以声明 fps 为权威参照。
    """
    df = _clean_df(100, freq=0.1)
    df = df.drop(index=df.index[1::2]).reset_index(drop=True)
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["frame_loss"]
    assert chk["result"] == "fail"
    assert chk["value"] > settings.quality_gate_frame_loss_ratio
    assert chk["evidence"]["basis"] == "declared_fps"
    assert "expected_rows" in chk["evidence"]


def test_gate_frame_loss_detects_large_gap_without_fps(settings) -> None:
    """无声明 fps 时，靠间隔一致性仍能发现"中间大段缺失"。"""
    n = 200
    t = np.arange(n) * 0.1
    # 在中间挖掉 50 个采样点（约 5 秒的缺口）。
    keep = np.concatenate([np.arange(0, 100), np.arange(150, n)])
    df = pd.DataFrame({
        "timestamp": t[keep],
        "action_joint0": np.sin(t[keep]),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["frame_loss"]
    assert chk["result"] == "fail"
    assert chk["evidence"]["basis"] == "interval_consistency"
    assert chk["evidence"]["n_large_gaps"] >= 1


def test_gate_frame_loss_reports_limitation_without_fps(settings) -> None:
    """无 fps 且无缺口时必须如实说明判据的局限（不假装检查充分）。"""
    df = _clean_df(100).drop(columns=["fps"])
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["frame_loss"]
    assert chk["result"] == "pass"
    assert "无法发现" in chk["evidence"]["note"]  # 如实说明局限
    assert chk["evidence"]["basis"] == "interval_consistency"


def test_gate_frame_loss_skipped_without_time_column(settings) -> None:
    """无时间列时丢帧列 not_audited，**不得报 pass**。"""
    df = pd.DataFrame({"episode_index": [0] * 20, "action_joint0": range(20)})
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["gate"]["checks"]["frame_loss"]["result"] == "skip"
    assert "frame_loss" in res["not_audited"]


def test_gate_schema_consistency_detects_mismatch(settings) -> None:
    """跨 episode 列集合不一致判 fail。

    注意：单张 DataFrame 的各 episode 天然共享列，所以此处通过构造
    带缺失列的 episode 来模拟 schema 漂移（真实场景来自多文件拼接）。
    """
    n = 60
    df = pd.DataFrame({
        "episode_index": [0] * 30 + [1] * 30,
        "timestamp": np.arange(n) * 0.1,
        "action_joint0": np.sin(np.arange(n) * 0.1),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    # 同一 DataFrame 内列集合必然一致 → 应为 pass（这条验证"不误报"）。
    assert res["gate"]["checks"]["schema_consistency"]["result"] == "pass"


def test_gate_schema_consistency_skipped_without_episodes(settings) -> None:
    """无 episode 划分时不做跨段比较（如实说明）。"""
    df = _clean_df(50)
    df = df.drop(columns=["episode_index"])
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["schema_consistency"]
    assert chk["result"] in ("pass", "skip")


def test_gate_fps_non_positive_fails(settings) -> None:
    """fps <= 0 判 fail（LeRobot 规范硬校验）。"""
    df = _clean_df(30)
    df["fps"] = 0.0
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["fps_valid"]
    assert chk["result"] == "fail"
    assert "fps_valid" in res["gate"]["failed"]


def test_gate_fps_missing_is_skipped(settings) -> None:
    """无 fps 声明时跳过（不报 pass，避免"未检查=通过"的误读）。"""
    df = _clean_df(30).drop(columns=["fps"])
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["gate"]["checks"]["fps_valid"]["result"] == "skip"


def test_gate_fps_from_lerobot_info_json(settings, tmp_path: Path) -> None:
    """fps 可从 LeRobot 的 meta/info.json 取得。"""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 0, "features": {}}), encoding="utf-8"
    )
    df = _clean_df(30).drop(columns=["fps"])
    res = check_dataset_quality_impl(_ctx(df, source=str(root)), settings)
    chk = res["gate"]["checks"]["fps_valid"]
    assert chk["result"] == "fail"
    assert chk["source"] == "meta/info.json"


def test_gate_episode_bounds_detects_time_reversal(settings) -> None:
    """episode 内时间倒序（起止矛盾）判 fail。"""
    n = 40
    df = pd.DataFrame({
        "episode_index": [0] * 20 + [1] * 20,
        "timestamp": list(np.arange(20) * 0.1) + list(-np.arange(20) * 0.1),
        "action_joint0": np.sin(np.arange(n) * 0.1),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["gate"]["checks"]["episode_bounds"]
    assert chk["result"] == "fail"
    assert "1" in chk["reversed_episodes"]


def test_gate_episode_bounds_skipped_without_time_col(settings) -> None:
    """无时间列时 episode 边界检查跳过。"""
    df = pd.DataFrame({"episode_index": [0, 0], "action_joint0": [1.0, 2.0]})
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["gate"]["checks"]["episode_bounds"]["result"] == "skip"


# ---------------------------------------------------------------------------
# L2/L3 诊断：逐条
# ---------------------------------------------------------------------------


def test_diagnose_idle_ratio_flags_high_idle(settings) -> None:
    """长时间不动 → 空闲比高 → warn（且必须带"可能正常"的提示）。"""
    n = 200
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp": np.arange(n) * 0.1,
        # 前 180 步不动，后 20 步动。
        "action_joint0": np.concatenate([np.zeros(180), np.linspace(0, 5, 20)]),
        "action_joint1": np.concatenate([np.zeros(180), np.linspace(0, 5, 20)]),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["diagnostics"]["checks"]["idle_ratio"]
    assert chk["result"] == "warn"
    assert chk["value"] > settings.quality_diagnostic_idle_ratio_warn
    assert "push" in chk["detail"]  # 必须提示"可能正常"


def test_diagnose_spike_detects_injected_jump(settings) -> None:
    """注入的尖峰被检出（HF GIGO 的 15×median 判据）。"""
    n = 300
    df = _clean_df(n)
    df.loc[150, "action_joint0"] = 100.0  # 单点大跳变
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["diagnostics"]["checks"]["action_spike"]
    assert chk["result"] == "warn"
    assert chk["threshold"] == settings.quality_diagnostic_spike_multiplier
    assert "action_joint0" in chk["per_channel"]


def test_diagnose_spike_clean_data_passes(settings) -> None:
    """平滑数据不误报突波。"""
    res = check_dataset_quality_impl(_ctx(_clean_df(400)), settings)
    assert res["diagnostics"]["checks"]["action_spike"]["result"] in ("pass", "warn")


def test_diagnose_saturation_detects_command_state_gap(settings) -> None:
    """指令与下一时刻状态差超阈 → 饱和诊断 warn（HF GIGO 公式）。"""
    n = 100
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.1,
        "action_joint0": np.zeros(n),              # 指令恒为 0
        "observation.state.joint0": np.linspace(0, 90, n),  # 状态漂到 90
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["diagnostics"]["checks"]["actuator_saturation"]
    assert chk["result"] == "warn"
    assert chk["threshold"] == settings.quality_diagnostic_saturation_deg


def test_diagnose_saturation_skipped_without_action_state_pairs(settings) -> None:
    """**无动作↔状态配对时必须跳过**，不得用通道内自比算出无意义数值。"""
    df = pd.DataFrame({
        "timestamp": np.arange(50) * 0.1,
        "action_joint0": np.zeros(50),
        "action_joint1": np.linspace(0, 90, 50),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["diagnostics"]["checks"]["actuator_saturation"]
    assert chk["result"] == "skip"
    assert "配对" in chk["detail"]
    assert "actuator_saturation" in chk["not_audited"]


def test_diagnose_saturation_passes_on_tracking_data(settings) -> None:
    """状态紧跟指令时不误报饱和。"""
    n = 100
    cmd = np.sin(np.arange(n) * 0.1)
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.1,
        "action_joint0": cmd,
        "observation.state.joint0": cmd,  # 完全跟随
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["diagnostics"]["checks"]["actuator_saturation"]["result"] == "pass"


def test_diagnose_jerk_flags_high_frequency(settings) -> None:
    """高频抖动 → 高频残差能量占比高 → warn。"""
    n = 300
    t = np.arange(n) * 0.1
    df = pd.DataFrame({
        "timestamp": t,
        "action_joint0": np.sin(2 * np.pi * 8 * t) * 10,
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["diagnostics"]["checks"]["action_jerk"]
    assert chk["result"] == "warn"
    assert "高频残差能量占比" in chk["note"]


def test_diagnose_jerk_passes_smooth_low_frequency(settings) -> None:
    """低频平滑动作不误报抖动（防"什么都报 warn"的假阳性）。"""
    n = 300
    t = np.arange(n) * 0.1
    df = pd.DataFrame({
        "timestamp": t,
        "action_joint0": np.sin(2 * np.pi * 0.3 * t),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["diagnostics"]["checks"]["action_jerk"]["result"] == "pass"


def test_diagnose_jerk_skipped_for_constant_channel(settings) -> None:
    """恒定信号无抖动可言 → skip（不误报）。"""
    n = 100
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.1,
        "action_joint0": np.zeros(n),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["diagnostics"]["checks"]["action_jerk"]["result"] == "skip"


def test_diagnose_pause_detects_long_stop(settings) -> None:
    """长停顿段被检出。"""
    n = 300
    a = np.concatenate([
        np.linspace(0, 5, 100),   # 动
        np.zeros(150),            # 长停
        np.linspace(5, 10, 50),   # 动
    ])
    df = pd.DataFrame({"timestamp": np.arange(n) * 0.1, "action_joint0": a})
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["diagnostics"]["checks"]["robot_induced_pause"]
    assert chk["result"] == "warn"
    assert chk["threshold"] == settings.quality_diagnostic_pause_steps


def test_diagnose_path_efficiency_flags_zigzag(settings) -> None:
    """来回绕路 → 路径效率低 → warn。"""
    n = 200
    t = np.linspace(0, 1, n)
    # 在两个位置间来回振荡：实际路径远长于首尾直线距离。
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.1,
        "pos_x": np.sin(2 * np.pi * 10 * t) * 1.0,
        "pos_y": np.zeros(n),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    chk = res["diagnostics"]["checks"]["path_efficiency"]
    assert chk["result"] == "warn"
    assert chk["value"] < settings.quality_diagnostic_path_efficiency


def test_diagnose_path_efficiency_skipped_with_one_channel(settings) -> None:
    """单通道无法算路径效率 → skip（不误报）。"""
    df = pd.DataFrame({
        "timestamp": np.arange(50) * 0.1,
        "action_joint0": np.sin(np.arange(50) * 0.1),
    })
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["diagnostics"]["checks"]["path_efficiency"]["result"] == "skip"


def test_diagnose_visual_reports_not_audited(settings) -> None:
    """无视频时视觉诊断必须列 not_audited（**不得报 pass**）。"""
    res = check_dataset_quality_impl(_ctx(_clean_df()), settings)
    chk = res["diagnostics"]["checks"]["visual_quality"]
    assert chk["result"] == "skip"
    assert "visual_quality" in chk["not_audited"]
    assert "visual_quality" in res["not_audited"]
    assert "视觉质量未检查" in chk["detail"]


def test_diagnose_visual_checks_metadata_when_present(settings) -> None:
    """有视频元信息时做元信息级检查，并如实说明逐帧未做。"""
    ctx = _ctx(_clean_df(), video_meta=[
        {"path": "cam.mp4", "width": 640, "height": 480, "duration": 10.0},
    ])
    res = check_dataset_quality_impl(ctx, settings)
    chk = res["diagnostics"]["checks"]["visual_quality"]
    assert chk["result"] == "pass"
    assert chk["checked"][0]["resolution"] == "640x480"
    # 逐帧质量未做，必须如实列出。
    assert "visual_frame_darkness" in chk["not_audited"]
    assert "visual_frame_blur" in chk["not_audited"]


def test_diagnose_visual_flags_tiny_resolution(settings) -> None:
    """分辨率过低被标记。"""
    ctx = _ctx(_clean_df(), video_meta=[
        {"path": "cam.mp4", "width": 32, "height": 32, "duration": 5.0},
    ])
    res = check_dataset_quality_impl(ctx, settings)
    chk = res["diagnostics"]["checks"]["visual_quality"]
    assert chk["result"] == "warn"
    assert "分辨率过低" in chk["detail"]


# ---------------------------------------------------------------------------
# 返回结构与降级
# ---------------------------------------------------------------------------


def test_not_audited_always_present(settings) -> None:
    """not_audited 必须始终存在（即使为空），供模型如实说明。"""
    res = check_dataset_quality_impl(_ctx(_clean_df()), settings)
    assert "not_audited" in res
    assert isinstance(res["not_audited"], list)


def test_unloaded_dataset_returns_structured_error(settings) -> None:
    """未加载数据集返回结构化错误（不抛异常）。"""
    res = check_dataset_quality_impl(RunContext(), settings)
    assert res["success"] is False
    assert res["error"] == "no_data_loaded"


def test_no_main_table_lists_not_audited(settings) -> None:
    """只有流登记表、无主表时，主表检查列 not_audited（不报 pass）。"""
    ctx = RunContext()
    ctx.dataset_id = "streams_only"
    ctx.meta = {"streams": [{"path": "a.csv", "format": "csv"}]}
    res = check_dataset_quality_impl(ctx, settings)
    assert res["success"] is True
    assert "main_table_gate_checks" in res["not_audited"]
    assert res["gate"]["checks"]["main_table"]["result"] == "skip"


def test_thresholds_and_source_reported(settings) -> None:
    """阈值与其来源必须随返回给出（供模型说明"未经数据集验证"）。"""
    res = check_dataset_quality_impl(_ctx(_clean_df()), settings)
    assert res["threshold_source"] == "default"
    th = res["thresholds"]
    assert th["gate_nan_ratio"] == settings.quality_gate_nan_ratio
    assert th["diagnostic_spike_multiplier"] == settings.quality_diagnostic_spike_multiplier
    # 关键阈值齐备（防漏配）。
    for key in (
        "gate_nan_ratio", "gate_frame_loss_ratio", "gate_schema_mismatch_max",
        "diagnostic_spike_multiplier", "diagnostic_saturation_deg",
        "diagnostic_idle_speed", "diagnostic_idle_ratio_warn",
        "diagnostic_jerk_ratio", "diagnostic_pause_steps",
        "diagnostic_shake_ratio", "diagnostic_path_efficiency",
        "diagnostic_dark_mean", "diagnostic_blur_variance",
    ):
        assert key in th, f"阈值 {key} 未随返回给出"


def test_qc_written_back_to_meta(settings) -> None:
    """质检结果写回 meta["qc"]（供 compute_stats / generate_report 读取）。"""
    ctx = _ctx(_clean_df())
    check_dataset_quality_impl(ctx, settings)
    qc = ctx.meta.get("qc", {}).get("check_dataset_quality")
    assert qc is not None
    assert qc["result"] in ("pass", "warn", "fail")
    assert "gate_result" in qc and "diagnostics_result" in qc


def test_measurements_report_analyzed_columns(settings) -> None:
    """measurements 如实说明分析了哪些列。"""
    res = check_dataset_quality_impl(_ctx(_clean_df()), settings)
    m = res["measurements"]
    assert m["n_action_columns"] >= 3
    assert "timestamp" not in m["action_columns_analyzed"]  # 元数据列被排除
    assert "episode_index" not in m["action_columns_analyzed"]


def test_affected_episodes_reported_on_warn(settings) -> None:
    """warn/fail 时给出受影响 episode（而非空列表）。"""
    df = _clean_df(100)
    df.loc[0:9, "action_joint0"] = np.nan
    res = check_dataset_quality_impl(_ctx(df), settings)
    assert res["result"] == "fail"
    assert res["affected_episodes"]


def test_deterministic_across_runs(settings) -> None:
    """同一输入两次运行结果一致（无随机性，可回归）。"""
    df = _clean_df(150)
    r1 = check_dataset_quality_impl(_ctx(df.copy()), settings)
    r2 = check_dataset_quality_impl(_ctx(df.copy()), settings)
    assert r1["result"] == r2["result"]
    assert r1["gate"]["failed"] == r2["gate"]["failed"]
    assert r1["diagnostics"]["warned"] == r2["diagnostics"]["warned"]
