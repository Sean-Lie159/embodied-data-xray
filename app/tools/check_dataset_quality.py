"""数据集质检（L1 硬门禁 + L2/L3 诊断，分层判定）。

设计依据：``docs/标注与质检能力设计.md`` §5.3。

**本工具的核心设计是"分层"，不是"多几条规则"**：

- **L1 硬门禁（gate）**：确定性错误（NaN/Inf、时间戳非单调、丢帧、schema
  不一致、fps 非法、episode 边界超界）。**只有 L1 能判 ``fail``**。
- **L2/L3 诊断（diagnostics）**：启发式指标（空闲比、动作突波、饱和、抖动、
  路径效率、视觉质量…）。**只能 ``warn``，永不自动升级为 fail**。

**为什么必须这样分层**：RDA 作者披露其早期版本（v0.5.x）让 ``idle_ratio``
自动升级为判定，在 libero_10 上产生 **65% 误报**；改成"只有硬门禁能排除"
后误报减少 97-100%。根因很直白——**阈值与任务相关，不存在通用阈值**：
70% 空闲比对 push 类任务正常，对 lift 类就可疑；33 个尖峰对脚本生成数据正常，
对平滑遥操作数据就该报警。

因此本工具的纪律是：
1. 诊断项的 ``warn`` **不等于"数据有问题"**，系统提示词要求模型如实分别转述；
2. 所有阈值集中在 ``app/config/settings.py``（``quality_gate_*`` /
   ``quality_diagnostic_*``），并在返回中带 ``thresholds``；
3. 无法检查的项列在 ``not_audited``，**绝不报为 pass**（避免"笼统说通过了质检"）；
4. 阈值来源为默认值时标注 ``threshold_source: "default"``，提示未经数据集验证。

本模块不 import streamlit（遵循分层纪律）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.config import get_settings
from app.tools._data_access import resolve_default_table_name, resolve_table_name

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_PASS = "pass"
_WARN = "warn"
_FAIL = "fail"

# 阈值来源标记：默认值（未经该数据集验证） vs 数据集覆盖。
_SRC_DEFAULT = "default"
_SRC_OVERRIDE = "dataset_override"

# 各规则能判定的最高严重度（诊断项封顶在 warn——这是分层纪律的代码表达）。
_SEVERITY_CAP = {
    "nan_inf": _FAIL,
    "timestamp_monotonic": _FAIL,
    "frame_loss": _FAIL,
    "schema_consistency": _FAIL,
    "fps_valid": _FAIL,
    "episode_bounds": _FAIL,
}

# 逐轴/逐列名忽略清单（这些列不该参与动作信号分析）。
_NON_ACTION_HINTS = (
    "timestamp", "time", "ts", "frame", "index", "episode", "fps",
    "sequence", "step", "id",
)

# ---------------------------------------------------------------------------
# 可读性元数据：把内部规则码翻译成用户能懂的说明
# ---------------------------------------------------------------------------
#
# **为什么需要（真实可用性缺陷，2026-09-21）**：初版返回的键名是
# `gate` / `diagnostics` / `idle_ratio` / `action_spike` 等内部术语，
# 用户在对话里看到"action_spike: warn"完全不知道在说什么——工具是给人用的，
# 内部命名不该直接暴露。这里为每条规则提供：
#   label  —— 中文名（给人看）
#   what   —— 这条在检查什么（一句话，含"为什么重要"）
#   terms  —— 关键术语的白话解释（首次出现时模型应转述）
#   advice —— 检出后建议怎么做
_RULE_META: dict[str, dict[str, str]] = {
    # ---- 硬门禁（确定性错误）----
    "nan_inf": {
        "label": "缺失值与异常值",
        "what": "统计空值（NaN）与无穷值（Inf）的比例。",
        "terms": "NaN=该格没有数据；Inf=数值溢出到无穷大。二者都会让下游计算失真。",
        "advice": "定位到具体列后，确认是设备未输出该通道还是导出遗漏。",
    },
    "timestamp_monotonic": {
        "label": "时间戳顺序",
        "what": "检查时间戳是否单调递增（时间只能往前走）。",
        "terms": "单调递增=后一帧的时间不早于前一帧。回退通常意味着时钟跳变或行乱序。",
        "advice": "回退需查时钟源；仅重复（时间相等）通常是正常的，确认后可忽略。",
    },
    "frame_loss": {
        "label": "丢帧",
        "what": "按声明的采样率推算本应有多少帧，与实际行数比较。",
        "terms": "丢帧=录制过程中某些时刻的数据没被记录，表现为时间上出现空档。",
        "advice": "若数据本身是低频采样，请先声明正确的 fps，否则会误判。",
    },
    "schema_consistency": {
        "label": "跨段结构一致性",
        "what": "比较各 episode 的列集合是否一致。",
        "terms": "结构不一致=不同片段的字段对不上，拼接后会出现错列。",
        "advice": "确认是否存在部分片段缺字段的情况。",
    },
    "fps_valid": {
        "label": "采样率声明合法性",
        "what": "检查声明的 fps 是否为正数，并与实测采样率对照。",
        "terms": "fps=每秒记录多少帧。声明 120 但实测 100 说明实际未达标。",
        "advice": "以实测值为准；声明与实测不符时请核对设备配置。",
    },
    "episode_bounds": {
        "label": "片段边界自洽性",
        "what": "检查每个 episode 的时间是否有倒流或空片段。",
        "terms": "episode=一段独立的录制片段。",
        "advice": "边界矛盾的片段建议单独排查。",
    },
    # ---- 诊断项（启发式提示）----
    "idle_ratio": {
        "label": "静止占比",
        "what": "统计动作幅度很小（近似不动）的时间占多少。",
        "terms": "占比高说明机器人大段时间没动，可能是等待指令，也可能是任务正常需要停顿。",
        "advice": "结合任务判断：放置、按压类任务本来就多停顿。",
    },
    "action_spike": {
        "label": "动作突跳",
        "what": "找出动作曲线里突然大幅度跳变的时刻。",
        "terms": "突跳=相邻两帧之间动作量剧变，常见于碰撞、人工拖拽或急停。",
        "advice": "对照信号曲线确认是真实操作还是异常；快速动作本身也会产生突跳。",
    },
    "actuator_saturation": {
        "label": "执行器饱和",
        "what": "比较下发的指令与下一时刻的实际状态差多少。",
        "terms": "饱和=指令要求的位置实际达不到，可能是到了关节限位或电机力矩不足。",
        "advice": "需要同时有指令列与状态列才能检查；无配对时该项跳过。",
    },
    "action_jerk": {
        "label": "动作抖动程度",
        "what": "衡量动作里高频抖动占多大比例（低频趋势之外的部分）。",
        "terms": "抖动=本应平滑的动作里出现细碎的高频波动，手抖或信号噪声都会造成。",
        "advice": "手套/遥操作数据普遍偏抖，属正常；若影响训练可考虑滤波。",
    },
    "robot_induced_pause": {
        "label": "中途停顿段",
        "what": "找出持续时间较长的完全静止段。",
        "terms": "停顿=机器人停下来了。可能是等待、演示中断，也可能是人为暂停。",
        "advice": "若为演示中断，可考虑切分或标记为噪声片段。",
    },
    "arm_shaking": {
        "label": "异常振动",
        "what": "用加速度的符号翻转频率衡量高频振动程度。",
        "terms": "振动=机械臂/手部出现来回抖动，可能是机械共振或信号噪声。",
        "advice": "手套类数据本身抖动大，需结合设备特性判断。",
    },
    "path_efficiency": {
        "label": "路径效率",
        "what": "比较起点到终点的直线距离与实际走过的总路程。",
        "terms": "效率=直线距离÷实际路程，接近 1 表示走得很直接，接近 0 表示来回绕。",
        "advice": "手的自然往返（如反复擦拭）会天然偏低，不一定是问题。",
    },
    "visual_quality": {
        "label": "视频质量",
        "what": "检查视频的分辨率、时长等元信息。",
        "terms": "逐帧的明暗与模糊检查需要额外解码依赖，当前未做。",
        "advice": "若需要逐帧画质检查，请告知。",
    },
}


# 判定结果的中文表述（避免把 pass/fail 这类英文直接抛给用户）。
_RESULT_ZH = {
    "pass": "通过",
    "warn": "有提示项（数据可用，但有指标偏高需关注）",
    "fail": "未通过（存在必须处理的确定性错误）",
}


def _decorate_checks(checks: dict[str, Any]) -> dict[str, Any]:
    """给每条检查结果附加中文名与白话说明（提升可读性）。

    Args:
        checks: {规则码: 结果 dict}。

    Returns:
        同结构的新 dict，每项多了 label / what / terms / advice（若该规则已登记）。
        未登记的规则原样保留——**不编造说明**。
    """
    out: dict[str, Any] = {}
    for code, payload in checks.items():
        if not isinstance(payload, dict):
            out[code] = payload
            continue
        meta = _RULE_META.get(code)
        merged = dict(payload)
        if meta:
            # label 放前面更易读；其余键顺序不变。
            ordered: dict[str, Any] = {
                "label": meta["label"],
                "what": meta["what"],
                "terms": meta["terms"],
                "advice": meta["advice"],
            }
            ordered.update(merged)
            out[code] = ordered
        else:
            out[code] = merged
    return out


def _humanize_rule_list(codes: list[str]) -> str:
    """把规则码列表转成**纯中文**名称串（面向用户，不带内部码）。

    设计取舍（2026-09-21 用户反馈）：初版写成"中文名（规则码）"以便追溯，
    但用户明确表示看不懂 `action_spike` 这类标识——**面向用户的文字里
    不应出现内部术语**。内部码仍在结构化字段中保留（`failed`/`warned`/
    `readable_summary.规则码`），需要追溯时程序可读，不必污染给用户的表述。
    """
    if not codes:
        return ""
    return "、".join(
        _RULE_META[c]["label"] if c in _RULE_META else c
        for c in codes
    )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _is_action_col(name: str) -> bool:
    """判断列名是否可能是动作/状态通道（排除时间、索引、episode 等元数据列）。"""
    low = str(name).lower().strip()
    return not any(h in low for h in _NON_ACTION_HINTS)


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """取数值列（动作信号分析只用数值列）。"""
    out: list[str] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and _is_action_col(c):
            out.append(str(c))
    return out


def _nan_inf_ratio(df: pd.DataFrame, cols: list[str]) -> tuple[float, dict[str, float]]:
    """计算 NaN/Inf 比例（总体与逐列）。

    Returns:
        (总体比例, {列名: 比例})。无可用列时返回 (0.0, {})。
    """
    if not cols or df.empty:
        return 0.0, {}
    per: dict[str, float] = {}
    total_bad = 0
    total = 0
    for c in cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        n = arr.size
        if n == 0:
            continue
        bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        per[c] = round(bad / n, 6)
        total_bad += bad
        total += n
    return (round(total_bad / total, 6) if total else 0.0), per


def _find_time_col(df: pd.DataFrame) -> str | None:
    """查找时间列（用于单调性与丢帧检查）。"""
    for cand in ("timestamp", "time", "ts", "log_time"):
        for c in df.columns:
            if str(c).lower().strip() == cand:
                return str(c)
    return None


def _monotonic_violations(series: pd.Series) -> tuple[int, int, list[int]]:
    """检查时间戳单调性。

    Returns:
        (回退次数, 重复次数, 回退位置索引样例——最多 5 个)。
    """
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0, 0, []
    diff = np.diff(arr)
    backwards = np.where(diff < 0)[0]
    duplicates = int(np.sum(diff == 0))
    return int(backwards.size), duplicates, [int(i) for i in backwards[:5]]


def _frame_loss_estimate(
    series: pd.Series,
    timeline_col: str | None,
    declared_fps: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """估算丢帧比例（实际行数 vs 应有行数）。

    **判据依赖声明的时间口径（关键设计取舍）**：丢帧检测必须有一个"应有
    采样率"的参照，否则无法区分"本来就采样慢"和"中间掉了数据"。因此：

    1. 优先用**声明 fps**（``fps`` 列或 LeRobot 的 ``meta/info.json``）——这是
       格式规范给出的权威口径，最可靠；
    2. 无声明 fps 时，退化为"**间隔一致性**"判据：检测是否存在明显大于
       中位间隔的时间缺口（这能发现"中间大段缺失"，但无法发现"均匀丢帧"）。

    **为什么不能只用中位间隔外推**：若隔行抽掉一半数据，中位间隔会翻倍，
    "应有行数"随之减半，丢帧就被自洽地掩盖了（自我掩饰的判据）。这是本
    函数最初版本的错误，已改为上述两条判据。

    Returns:
        (丢帧比例或 None, 证据 dict)。
    """
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return None, {"reason": "时间戳样本不足 3 个，无法估算采样率"}

    diffs = np.diff(arr)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return None, {"reason": "时间戳无正向间隔（可能全为重复值）"}

    median_step = float(np.median(positive))
    span = float(arr[-1] - arr[0])
    actual = float(arr.size)
    if median_step <= 0:
        return None, {"reason": "中位采样间隔非正"}

    # 判据 1：有声明 fps → 用它作为权威参照（最可靠）。
    if declared_fps and declared_fps > 0:
        expected_step = 1.0 / declared_fps
        expected = span / expected_step + 1
        if expected <= 0:
            return None, {"reason": "按声明 fps 推算的应有帧数非正"}
        ratio = max(0.0, (expected - actual) / expected)
        return round(ratio, 6), {
            "basis": "declared_fps",
            "declared_fps": declared_fps,
            "actual_rows": int(actual),
            "expected_rows": round(expected, 2),
            "median_step": median_step,
            "span": span,
            "note": "以声明 fps 为权威参照——它能发现「均匀丢帧」",
        }

    # 判据 2：无声明 fps → 用间隔一致性找"大缺口"（能发现大段缺失）。
    # 阈值取中位间隔的 5 倍：正常抖动不会到 5 倍，而真正的丢帧缺口通常是
    # 整数倍采样周期。
    gap_threshold = median_step * 5.0
    large_gaps = positive[positive > gap_threshold]
    if large_gaps.size == 0:
        return 0.0, {
            "basis": "interval_consistency",
            "actual_rows": int(actual),
            "median_step": median_step,
            "span": span,
            "n_large_gaps": 0,
            "note": (
                "无声明 fps，改用间隔一致性判据：未发现明显时间缺口。"
                "**注意**：该判据无法发现「均匀丢帧」，如需严格丢帧检查请先"
                "确认/声明 fps。"
            ),
        }

    # 缺口造成的缺失帧数估算：每个缺口按 (gap / median_step - 1) 帧计。
    missing = float(np.sum(np.round(large_gaps / median_step) - 1))
    expected = actual + missing
    ratio = max(0.0, missing / expected) if expected > 0 else 0.0
    return round(ratio, 6), {
        "basis": "interval_consistency",
        "actual_rows": int(actual),
        "expected_rows": round(expected, 2),
        "median_step": median_step,
        "span": span,
        "n_large_gaps": int(large_gaps.size),
        "gap_threshold": round(gap_threshold, 6),
        "note": (
            "无声明 fps，改用间隔一致性判据（中位间隔 5 倍以上的缺口视为丢帧）。"
            "该判据无法发现均匀丢帧。"
        ),
    }


def _schema_consistency(df: pd.DataFrame) -> tuple[int, list[dict[str, Any]]]:
    """检查跨 episode 的列集合一致性（LeRobot 规范的硬要求）。

    Returns:
        (不一致的列数, 不一致明细)。无 episode 列或仅一个 episode 时返回 (0, [])。
    """
    ep_col = None
    for cand in ("episode_index", "episode", "ep", "episode_id",
                 "traj_id", "trajectory_id"):
        for c in df.columns:
            if str(c).lower().strip() == cand:
                ep_col = str(c)
                break
        if ep_col:
            break
    if ep_col is None:
        return 0, []

    groups = df.groupby(ep_col, sort=True, dropna=False)
    if groups.ngroups <= 1:
        return 0, []

    ref_key, ref_df = next(iter(groups))
    ref_cols = set(map(str, ref_df.columns))
    mismatches: list[dict[str, Any]] = []
    bad_cols: set[str] = set()
    for key, sub in groups:
        if str(key) == str(ref_key):
            continue
        cols = set(map(str, sub.columns))
        if cols != ref_cols:
            missing = sorted(ref_cols - cols)
            extra = sorted(cols - ref_cols)
            bad_cols.update(missing)
            bad_cols.update(extra)
            if len(mismatches) < 10:
                mismatches.append({
                    "episode": str(key),
                    "missing_columns": missing,
                    "extra_columns": extra,
                })
    return len(bad_cols), mismatches


def _check_fps(context: RunContext, df: pd.DataFrame) -> dict[str, Any]:
    """检查 fps 合法性与实际采样率一致性（LeRobot 有 fps<=0 硬校验）。"""
    fps: float | None = None
    source = ""
    for cand in ("fps", "frame_rate", "rate_hz"):
        for c in df.columns:
            if str(c).lower().strip() == cand:
                try:
                    v = pd.to_numeric(df[c], errors="coerce").dropna()
                    if len(v) >= 1:
                        fps = float(v.iloc[0])
                        source = f"列 {c}"
                except (ValueError, TypeError):
                    pass
                break
        if fps is not None:
            break

    out: dict[str, Any] = {"declared_fps": fps, "source": source or None}

    # LeRobot 的 meta/info.json 也声明 fps。
    if fps is None:
        src = str(context.meta.get("source", "") or "")
        if src:
            import json
            from pathlib import Path

            info = Path(src) / "meta" / "info.json"
            if info.exists():
                try:
                    obj = json.loads(info.read_text(encoding="utf-8"))
                    if isinstance(obj.get("fps"), (int, float)):
                        fps = float(obj["fps"])
                        source = "meta/info.json"
                        out["declared_fps"] = fps
                        out["source"] = source
                except (ValueError, OSError, UnicodeDecodeError):
                    pass

    if fps is None:
        out["result"] = "skip"
        out["detail"] = "未声明 fps（列与 meta/info.json 均无）——无法校验 fps 合法性"
        return out

    if fps <= 0:
        out["result"] = _FAIL
        out["detail"] = f"声明的 fps={fps} 非正（LeRobot 规范要求 fps > 0）"
        return out

    # 与实际时间戳采样率比对。
    tcol = _find_time_col(df)
    if tcol is not None:
        arr = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype="float64")
        arr = arr[~np.isnan(arr)]
        if arr.size >= 3:
            diffs = np.diff(arr)
            diffs = diffs[diffs > 0]
            if diffs.size:
                median_step = float(np.median(diffs))
                if median_step > 0:
                    measured = 1.0 / median_step
                    out["measured_rate_hz"] = round(measured, 4)
                    # 相对偏差 > 10% 提示不一致（诊断性质，只 warn）。
                    if fps > 0 and abs(measured - fps) / fps > 0.10:
                        out["result"] = _WARN
                        out["detail"] = (
                            f"声明 fps={fps} 与实测采样率 {round(measured, 3)}Hz "
                            "相对偏差超 10%"
                        )
                        return out

    out["result"] = _PASS
    out["detail"] = f"fps={fps} 合法"
    return out


def _check_episode_bounds(df: pd.DataFrame) -> dict[str, Any]:
    """检查 episode 的时长/帧数是否自相矛盾（L1 硬门禁）。

    判据：同一 episode 内若存在**重复且递减**的时间戳（即时间倒流），或
    episode 的帧数为 0，属确定性错误。这里只做集合级检查，不做语义推断。
    """
    ep_col = None
    for cand in ("episode_index", "episode", "ep", "episode_id",
                 "traj_id", "trajectory_id"):
        for c in df.columns:
            if str(c).lower().strip() == cand:
                ep_col = str(c)
                break
        if ep_col:
            break
    if ep_col is None:
        return {"result": "skip", "detail": "无 episode 列，跳过边界检查"}

    tcol = _find_time_col(df)
    if tcol is None:
        return {"result": "skip", "detail": "无时间列，跳过 episode 边界检查"}

    empty_eps: list[str] = []
    reversed_eps: list[str] = []
    for key, sub in df.groupby(ep_col, sort=True, dropna=False):
        if len(sub) == 0:
            empty_eps.append(str(key))
            continue
        arr = pd.to_numeric(sub[tcol], errors="coerce").to_numpy(dtype="float64")
        arr = arr[~np.isnan(arr)]
        if arr.size >= 2 and float(arr[-1]) < float(arr[0]):
            reversed_eps.append(str(key))

    if empty_eps or reversed_eps:
        return {
            "result": _FAIL,
            "detail": (
                f"episode 边界自相矛盾：空 episode {len(empty_eps)} 个、"
                f"时间倒序 episode {len(reversed_eps)} 个"
            ),
            "empty_episodes": empty_eps[:10],
            "reversed_episodes": reversed_eps[:10],
        }
    return {"result": _PASS, "detail": "各 episode 边界自洽"}


# ---------------------------------------------------------------------------
# L2/L3 诊断规则（只能 warn）
# ---------------------------------------------------------------------------


def _diagnose_idle(
    df: pd.DataFrame, cols: list[str], idle_speed: float, warn_ratio: float,
    limit: int,
) -> dict[str, Any]:
    """空闲比：归一化速度低于阈值的时间步占比（HF GIGO 口径）。"""
    if not cols or df.empty:
        return {"result": "skip", "detail": "无可用动作列，跳过空闲比诊断"}

    ep_col = None
    for cand in ("episode_index", "episode", "ep"):
        for c in df.columns:
            if str(c).lower().strip() == cand:
                ep_col = str(c)
                break
        if ep_col:
            break

    def _idle_ratio(sub: pd.DataFrame) -> float | None:
        mat = np.column_stack([
            pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype="float64")
            for c in cols if c in sub.columns
        ]) if cols else None
        if mat is None or mat.size == 0:
            return None
        # 归一化速度：逐通道差分绝对值之和，再按该通道自身的 P95 归一，
        # 避免不同通道量纲（角度 vs 米）互相支配。
        speeds = np.abs(np.diff(mat, axis=0))
        if speeds.size == 0:
            return None
        scale = np.nanpercentile(speeds, 95, axis=0)
        scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
        norm = speeds / scale
        speed_mag = np.nanmean(norm, axis=1)
        if speed_mag.size == 0:
            return None
        return float(np.nanmean(speed_mag < idle_speed))

    values: dict[str, float] = {}
    if ep_col is not None and df.groupby(ep_col, dropna=False).ngroups > 1:
        for key, sub in df.groupby(ep_col, sort=True, dropna=False):
            r = _idle_ratio(sub)
            if r is not None:
                values[str(key)] = round(r, 4)
    else:
        r = _idle_ratio(df)
        if r is not None:
            values["__all__"] = round(r, 4)

    if not values:
        return {"result": "skip", "detail": "空闲比不可计算（样本不足）"}

    mean_ratio = float(np.mean(list(values.values())))
    offenders = [k for k, v in values.items() if v > warn_ratio]
    out: dict[str, Any] = {
        "value": round(mean_ratio, 4),
        "threshold": warn_ratio,
        "per_episode_sample": dict(list(values.items())[:limit]),
        "result": _WARN if offenders else _PASS,
    }
    if offenders:
        out["detail"] = (
            f"{len(offenders)} 个 episode 空闲比超过 {warn_ratio}。"
            "**注意**：高空闲比对 push/放置类任务可能是正常的，请结合任务类型判断。"
        )
        out["flagged_episodes"] = offenders[:limit]
    else:
        out["detail"] = f"平均空闲比 {round(mean_ratio, 3)}，未超阈"
    return out


def _diagnose_spike(
    df: pd.DataFrame, cols: list[str], spike_multiplier: float, limit: int,
) -> dict[str, Any]:
    """动作突波（疑似碰撞）：二阶差分超 ``倍数 × median(|a|)`` 的步数（HF GIGO）。"""
    if not cols or df.empty:
        return {"result": "skip", "detail": "无可用动作列，跳过突波诊断"}

    tcol = _find_time_col(df)
    ratios: dict[str, float] = {}
    total_spikes = 0
    for c in cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            continue
        if tcol is not None:
            t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype="float64")
            t = t[np.isfinite(t)]
            if t.size == arr.size:
                dt = np.diff(t)
                dt[dt <= 0] = np.nan
                # 加速度代理：a_t = (q_{t+1} - 2q_t + q_{t-1}) / dt²
                accel = np.full(arr.size, np.nan)
                accel[1:-1] = (
                    arr[:-2] - 2 * arr[1:-1] + arr[2:]
                ) / (dt[1:] ** 2)
            else:
                accel = np.diff(arr, n=2)
        else:
            accel = np.diff(arr, n=2)

        accel = accel[np.isfinite(accel)]
        if accel.size == 0:
            continue
        med = float(np.median(np.abs(accel)))
        if med <= 0:
            continue
        thr = spike_multiplier * med
        n_spikes = int(np.sum(np.abs(accel) > thr))
        if n_spikes:
            ratios[c] = n_spikes
            total_spikes += n_spikes

    if not ratios:
        return {
            "result": _PASS,
            "detail": "未检出超过阈值的动作突波",
            "threshold": spike_multiplier,
        }
    top = dict(sorted(ratios.items(), key=lambda kv: -kv[1])[:limit])
    return {
        "value": total_spikes,
        "threshold": spike_multiplier,
        "per_channel": top,
        "result": _WARN,
        "detail": (
            f"检出 {total_spikes} 处动作突波（超 {spike_multiplier}×median|a|）。"
            "突波常对应碰撞、人工干预或遥操作拖拽；也可能来自任务本身的快速动作，"
            "请结合信号曲线判断（本项为诊断提示，不代表数据错误）。"
        ),
    }


def _pair_action_state_cols(
    cols: list[str],
) -> list[tuple[str, str]]:
    """把列名配对为 (action 列, 对应 state 列)（HF GIGO 公式的输入前提）。

    HF GIGO 的饱和判据是 ``|a_t - q_{t+1}| > 7°``——比较**动作指令**与
    **下一时刻的实际关节状态**。因此需要动作与状态成对，而不是通道内自比。

    配对规则（按名称后缀匹配）：``action_joint0`` ↔ ``observation.state.joint0``
    这类命名在 LeRobot 中很常见。此处用保守策略：
    1. 优先找 ``action_*`` / ``actions*`` 前缀列，与 ``state*`` / ``obs*``
       前缀列按**归一化后缀**配对；
    2. 配对不到时返回空列表——**不猜测**（宁可跳过，不产出错误结论）。
    """
    def _norm(name: str) -> str:
        s = str(name).lower()
        for p in ("action_", "actions.", "actions_", "action.",
                  "observation.state.", "observation_state.", "obs_", "state_",
                  "state.", "qpos_", "joint_"):
            if s.startswith(p):
                s = s[len(p):]
                break
        return s.strip("._")

    action_cols = [
        c for c in cols
        if str(c).lower().startswith(("action_", "actions.", "actions_", "action."))
    ]
    state_cols = [
        c for c in cols
        if str(c).lower().startswith((
            "observation.state.", "observation_state.", "obs_", "state_",
            "state.", "qpos_",
        ))
    ]
    if not action_cols or not state_cols:
        return []

    state_by_norm = {_norm(c): c for c in state_cols}
    pairs: list[tuple[str, str]] = []
    for a in action_cols:
        s = state_by_norm.get(_norm(a))
        if s is not None:
            pairs.append((a, s))
    return pairs


def _diagnose_saturation(
    df: pd.DataFrame, cols: list[str], saturation_deg: float, limit: int,
) -> dict[str, Any]:
    """致动器饱和：``|a_t - q_{t+1}|`` 超过角度阈值的步数（HF GIGO 公开公式）。

    **前提**：需要动作列与状态列**成对**。配对不到时跳过并如实说明——
    通道内自比会得到无意义的数值（且会误报），所以宁可不检。
    """
    if not cols or df.empty:
        return {"result": "skip", "detail": "无可用动作列，跳过饱和诊断"}

    pairs = _pair_action_state_cols(cols)
    if not pairs:
        return {
            "result": "skip",
            "detail": (
                "未找到「动作列 ↔ 状态列」配对，无法按 |a_t - q_{t+1}| 公式计算"
                "饱和（通道内自比无物理意义，故不检）。若数据含 action_* 与 "
                "observation.state.* 类列名，请确认其命名可被识别。"
            ),
            "not_audited": ["actuator_saturation"],
        }

    hits: dict[str, int] = {}
    for a_col, s_col in pairs:
        a = pd.to_numeric(df[a_col], errors="coerce").to_numpy(dtype="float64")
        q = pd.to_numeric(df[s_col], errors="coerce").to_numpy(dtype="float64")
        n = min(a.size, q.size) - 1
        if n < 1:
            continue
        # a_t - q_{t+1}：指令与下一时刻实际状态的差。
        diff = np.abs(a[:n] - q[1:n + 1])
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            continue
        cnt = int(np.sum(diff > saturation_deg))
        if cnt:
            hits[f"{a_col}→{s_col}"] = cnt

    if not hits:
        return {
            "result": _PASS,
            "detail": (
                f"未检出致动器饱和（已配对 {len(pairs)} 组动作↔状态列，"
                f"判据 |a_t - q_{{t+1}}| > {saturation_deg}°）"
            ),
            "threshold": saturation_deg,
            "paired": [f"{a}→{s}" for a, s in pairs][:limit],
        }
    total = sum(hits.values())
    return {
        "value": total,
        "threshold": saturation_deg,
        "per_pair": dict(sorted(hits.items(), key=lambda kv: -kv[1])[:limit]),
        "result": _WARN,
        "detail": (
            f"检出 {total} 处指令-状态差超 {saturation_deg}°（疑似致动器饱和或"
            "指令未被执行）。"
        ),
    }


def _diagnose_jerk(
    df: pd.DataFrame, cols: list[str], jerk_ratio: float, limit: int,
) -> dict[str, Any]:
    """动作抖动：**高频 jitter 能量占总体运动能量的比例**。

    **为什么不用「jerk 的 P95/median 比值」**（本函数初版的错误）：一个平滑
    的 8Hz 正弦，其 jerk 处处均匀，P95/median ≈ 1.2——比值判据**完全无法
    发现平滑的高频运动**，而这类运动恰恰是"抖动"的典型形态。这是实测发现
    的缺陷（测试用例构造正弦高频动作时该规则未报警）。

    **正确的判据**：把信号分解为"低频趋势 + 高频残差"，用高频残差能量占
    总能量的比例衡量抖动程度。这与"Consistency Matters"(arXiv:2412.14309)
    反对绝对阈值的立场一致（比值无量纲），且对高频运动敏感。

    实现：用 5 点滑动平均作为低频趋势，残差 = 信号 − 趋势；
    比值 = Var(残差) / Var(信号 − 总体均值)。
    """
    if not cols or df.empty:
        return {"result": "skip", "detail": "无可用动作列，跳过抖动诊断"}

    ratios: dict[str, float] = {}
    for c in cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size < 8:
            continue

        total_var = float(np.var(arr))
        if total_var <= 0:
            continue  # 恒定信号无抖动可言

        # 5 点滑动平均作为低频趋势（窗口小 → 保留中频，剥离高频）。
        kernel = np.ones(5) / 5.0
        trend = np.convolve(arr, kernel, mode="valid")
        residual = arr[2:-2] - trend
        if residual.size < 3:
            continue

        hf_ratio = float(np.var(residual)) / total_var
        ratios[c] = round(hf_ratio, 4)

    if not ratios:
        return {"result": "skip", "detail": "抖动不可计算（样本不足或全零）"}

    worst = max(ratios.values())
    offenders = {k: v for k, v in ratios.items() if v > jerk_ratio}
    out: dict[str, Any] = {
        "value": worst,
        "threshold": jerk_ratio,
        "per_channel": dict(sorted(ratios.items(), key=lambda kv: -kv[1])[:limit]),
        "result": _WARN if offenders else _PASS,
        "note": (
            "用高频残差能量占比判定（5 点滑动均值分离低频趋势），"
            "对量纲不敏感；阈值含义为「高频能量占总能量比例」"
        ),
    }
    if offenders:
        out["detail"] = (
            f"{len(offenders)} 个通道的高频能量占比超 {jerk_ratio}，"
            "动作抖动明显（诊断提示，非错误）。"
        )
    else:
        out["detail"] = "动作平滑度未超阈"
    return out


def _diagnose_pause(
    df: pd.DataFrame, cols: list[str], idle_speed: float, pause_steps: int,
    limit: int,
) -> dict[str, Any]:
    """轨迹停顿：连续低速步数超阈的段落数（RoboMIND 标准 #2 的量化形式）。"""
    if not cols or df.empty:
        return {"result": "skip", "detail": "无可用动作列，跳过停顿诊断"}

    mat = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        for c in cols if c in df.columns
    ])
    if mat.size == 0 or mat.shape[0] < 2:
        return {"result": "skip", "detail": "样本不足，跳过停顿诊断"}

    speeds = np.abs(np.diff(mat, axis=0))
    scale = np.nanpercentile(speeds, 95, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    speed_mag = np.nanmean(speeds / scale, axis=1)
    slow = speed_mag < idle_speed

    # 数连续低速段的长度，超过 pause_steps 的段计一次。
    pauses: list[dict[str, Any]] = []
    run = 0
    for i, s in enumerate(slow):
        if s:
            run += 1
        else:
            if run >= pause_steps:
                pauses.append({"start_step": i - run, "length": run})
            run = 0
    if run >= pause_steps:
        pauses.append({"start_step": len(slow) - run, "length": run})

    if not pauses:
        return {"result": _PASS, "detail": "未检出长停顿段", "threshold": pause_steps}
    return {
        "value": len(pauses),
        "threshold": pause_steps,
        "samples": pauses[:limit],
        "result": _WARN,
        "detail": (
            f"检出 {len(pauses)} 段长停顿（连续低速步数 ≥ {pause_steps}），"
            "可能对应机器人等待指令、人为暂停或演示中断（诊断提示）。"
        ),
    }


def _diagnose_shake(
    df: pd.DataFrame, cols: list[str], shake_ratio: float, limit: int,
) -> dict[str, Any]:
    """机械臂异常振动：高频能量占总能量比例（RoboMIND 标准 #4 的量化形式）。"""
    if not cols or df.empty:
        return {"result": "skip", "detail": "无可用动作列，跳过振动诊断"}

    ratios: dict[str, float] = {}
    for c in cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size < 8:
            continue
        # 二阶差分（加速度代理）的"过零率"作为高频程度代理。
        accel = np.diff(arr, n=2)
        accel = accel[np.isfinite(accel)]
        if accel.size < 4:
            continue
        sign = np.sign(accel)
        sign = sign[sign != 0]
        if sign.size < 4:
            continue
        # 符号翻转率：高频振动对应高翻转率。
        flips = int(np.sum(np.diff(sign) != 0))
        ratio = flips / sign.size
        ratios[c] = round(float(ratio), 4)

    if not ratios:
        return {"result": "skip", "detail": "振动不可计算（样本不足）"}

    offenders = {k: v for k, v in ratios.items() if v > shake_ratio}
    out: dict[str, Any] = {
        "value": max(ratios.values()),
        "threshold": shake_ratio,
        "per_channel": dict(sorted(ratios.items(), key=lambda kv: -kv[1])[:limit]),
        "result": _WARN if offenders else _PASS,
        "note": "用加速度代理的符号翻转率作为高频程度代理（非 FFT 能量占比）",
    }
    if offenders:
        out["detail"] = (
            f"{len(offenders)} 个通道的高频翻转率超 {shake_ratio}，"
            "疑似机械臂振动（诊断提示）。"
        )
    else:
        out["detail"] = "未检出异常振动"
    return out


def _diagnose_path_efficiency(
    df: pd.DataFrame, cols: list[str], warn_threshold: float,
) -> dict[str, Any]:
    """路径效率：clip(D/L, 0, 1)，D 为起止直线距离、L 为实际路径长度（HF GIGO）。"""
    if len(cols) < 2:
        return {"result": "skip", "detail": "可用位置通道少于 2 个，跳过路径效率诊断"}

    mat = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        for c in cols
    ])
    mat = mat[np.isfinite(mat).all(axis=1)]
    if mat.shape[0] < 3:
        return {"result": "skip", "detail": "有效位置样本不足，跳过路径效率诊断"}

    # 全局归一化（避免量纲差异）。
    span = np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0)
    span[~np.isfinite(span) | (span <= 0)] = 1.0
    norm = mat / span

    steps = np.linalg.norm(np.diff(norm, axis=0), axis=1)
    path_len = float(np.nansum(steps))
    if path_len <= 0:
        return {"result": "skip", "detail": "路径长度为 0，跳过路径效率诊断"}

    straight = float(np.linalg.norm(norm[-1] - norm[0]))
    efficiency = float(np.clip(straight / path_len, 0.0, 1.0))

    return {
        "value": round(efficiency, 4),
        "threshold": warn_threshold,
        "result": _WARN if efficiency < warn_threshold else _PASS,
        "detail": (
            f"路径效率 {round(efficiency, 3)}（直线距离/实际路径长度）"
            + ("，低于阈值，提示轨迹犹豫或绕路（诊断提示）。"
               if efficiency < warn_threshold else "，未见明显犹豫。")
        ),
    }


def _diagnose_visual(
    context: RunContext, blur_variance: float, dark_mean: float,
) -> dict[str, Any]:
    """视频质量诊断（过暗/模糊）。缺依赖或无法读取时列 not_audited，**不报 pass**。"""
    video_meta = context.meta.get("video_meta") or []
    if not video_meta:
        return {
            "result": "skip",
            "detail": "无视频元信息（未识别到视频流或未用 ffprobe 探测），"
                      "视觉质量未检查",
            "not_audited": ["visual_quality"],
        }

    # 视频逐帧读取需要 ffmpeg/opencv，属重依赖；此处只做**元信息级**检查
    # （分辨率、时长、帧率的存在性与合理性），并在返回中如实说明。
    issues: list[str] = []
    checked: list[dict[str, Any]] = []
    for v in video_meta:
        item: dict[str, Any] = {"path": v.get("path") or v.get("file")}
        w = v.get("width")
        h = v.get("height")
        item["resolution"] = f"{w}x{h}" if w and h else None
        if w and h and (int(w) < 64 or int(h) < 64):
            issues.append(f"{item['path']}: 分辨率过低（{item['resolution']}）")
        dur = v.get("duration")
        if isinstance(dur, (int, float)) and dur <= 0:
            issues.append(f"{item['path']}: 时长非正（{dur}）")
        checked.append(item)

    return {
        "result": _WARN if issues else _PASS,
        "detail": (
            "；".join(issues) if issues
            else "视频元信息（分辨率/时长）未见异常"
        ),
        "checked": checked,
        # 逐帧质量（过暗/模糊）需要重依赖，当前版本未做——**必须如实列出**，
        # 避免模型说成"视频质量已通过检查"。
        "not_audited": ["visual_frame_darkness", "visual_frame_blur"],
        "note": (
            f"逐帧过暗判定阈值 {dark_mean}/255、模糊阈值（拉普拉斯方差）"
            f"{blur_variance} 已配置，但逐帧读取需额外视频解码依赖，"
            "当前版本仅做元信息级检查。"
        ),
    }


# ---------------------------------------------------------------------------
# 主实现
# ---------------------------------------------------------------------------


def check_dataset_quality_impl(
    context: RunContext, settings=None, table: str | None = None,
) -> dict[str, Any]:
    """执行数据集质检（L1 硬门禁 + L2/L3 诊断，分层判定）。

    Args:
        context: 运行时上下文（读 df / meta）。
        settings: 可选配置覆盖（测试注入用）。
        table: 可选，要质检的表名（缺省=主表）。多表数据集里"每张表各自的
            质量问题完全不同"——主时钟表该查时间戳单调性，末端位置表该查
            越界与抖动。此前本工具无此参数，**只能质检主表**，是唯一无法
            对非主表做质检的分析工具（见设计文档 §4.4）。

    Returns:
        分层质检返回结构，见模块 docstring 与设计文档 §5.3.2。
        表不存在时返回 table_not_found（含 available_tables 等候选信息）。
    """
    settings = settings or get_settings()

    if context.df is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载任何数据集",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset。",
        }

    dataset = context.dataset_id

    # 取表：显式给 table 时经统一入口解析（惰性读取，**不替换主表**）。
    #
    # 纪律：表不存在时**如实返回 table_not_found 并附候选清单**，不得静默回退
    # 主表——静默回退会让模型以为"我质检了末端表"，实际质检的是主表，
    # 结论被张冠李戴（比报错危险得多）。
    checked_table: str | None = None
    if table is not None:
        resolved = resolve_table_name(context, table)
        if not resolved.get("success"):
            # 原样透传结构化错误（含 available_tables / default_table），
            # 只补上这是"质检"场景与本工具的 check 名，便于模型理解。
            out = dict(resolved)
            out["check"] = "check_dataset_quality"
            out["user_message"] = f"质检未能执行：{resolved.get('user_message', '')}"
            return out
        df = resolved.get("df")
        checked_table = resolved.get("table_name")
    else:
        df = context.df
        checked_table = resolve_default_table_name(context)

    not_audited: list[str] = []

    # ---------- L1 硬门禁 ----------
    gate_checks: dict[str, Any] = {}

    if df is None or df.empty:
        # 无表可查：只有流登记表（纯容器/多流目录未合并），或指定表的 df 为空。
        #
        # 保持既有行为：**标 skip 而非报错**——纯媒体数据集"无有效主表"是合法
        # 状态，不得因无表就判 fail 或抛错。detail 须区分是"整体无表"还是
        # "指定表为空"，否则用户会误以为是数据集的问题。
        if table is not None:
            detail = f"指定的表 {checked_table or table} 无数据行，L1 检查未执行"
            gate_checks["main_table"] = {"result": "skip", "detail": detail}
            not_audited.append("requested_table_gate_checks")
        else:
            gate_checks["main_table"] = {
                "result": "skip",
                "detail": "无主数据表（仅流登记表），L1 主表检查未执行",
            }
            not_audited.append("main_table_gate_checks")
        cols: list[str] = []
    else:
        cols = _numeric_cols(df)

        # 1) NaN/Inf
        #
        # **可读性要点**：把"总体比例"与"逐列比例"分别标注清楚。
        # 此前只给一个 value（总体），但判定用的是逐列——于是会出现
        # "value 2% < threshold 5% 却判 fail"这种看起来自相矛盾的结果
        # （某一列 100% 为空，整体被摊薄）。这会让人怀疑工具算错了。
        nan_ratio, nan_per_col = _nan_inf_ratio(df, cols)
        offenders = {k: v for k, v in nan_per_col.items() if v > settings.quality_gate_nan_ratio}
        sorted_offenders = dict(sorted(offenders.items(), key=lambda kv: -kv[1])[:10])
        if offenders:
            worst_col, worst_ratio = next(iter(sorted_offenders.items()))
            detail = (
                f"共 {len(offenders)} 个字段的缺失/异常值比例超过 "
                f"{settings.quality_gate_nan_ratio:.0%}，最严重的是「{worst_col}」"
                f"（{worst_ratio:.1%} 为空）。表格整体缺失率为 {nan_ratio:.1%}"
                "（整体值被未缺失的字段摊薄，因此要按字段看）。"
            )
        else:
            detail = (
                f"各字段缺失/异常值比例均未超过 "
                f"{settings.quality_gate_nan_ratio:.0%}（整体 {nan_ratio:.1%}）。"
            )
        gate_checks["nan_inf"] = {
            "result": _FAIL if offenders else _PASS,
            # 判定依据：逐字段（而非整体）——避免被摊薄掩盖。
            "judged_by": "per_column",
            "worst_column": (
                next(iter(sorted_offenders)) if sorted_offenders else None),
            "value_overall": nan_ratio,
            "value_worst_column": (
                next(iter(sorted_offenders.values()))
                if sorted_offenders else None),
            "threshold": settings.quality_gate_nan_ratio,
            "per_column": sorted_offenders,
            "detail": detail,
        }

        # 2) 时间戳单调性
        tcol = _find_time_col(df)
        if tcol is None:
            gate_checks["timestamp_monotonic"] = {
                "result": "skip",
                "detail": "未找到时间列，单调性未检查",
            }
            not_audited.append("timestamp_monotonic")
        else:
            back, dup, samples = _monotonic_violations(df[tcol])
            if back > 0:
                result = _FAIL
                detail = f"时间戳出现 {back} 次回退（非单调），疑似时钟跳变或乱序写入"
            elif dup > 0:
                result = _WARN
                detail = f"时间戳有 {dup} 处重复（允许但需确认是否应为唯一）"
            else:
                result = _PASS
                detail = "时间戳严格单调递增"
            gate_checks["timestamp_monotonic"] = {
                "result": result,
                "time_column": tcol,
                "backwards_count": back,
                "duplicate_count": dup,
                "sample_positions": samples,
                "detail": detail,
            }

        # 5) fps 合法性（**必须在丢帧之前算**：丢帧判据需要它的 declared_fps
        # 作权威参照。初版把 fps 放在丢帧之后，导致 declared_fps 永远拿不到，
        # 丢帧退化为自我掩饰的中位间隔外推——这是实测发现的顺序缺陷。）
        fps_check = _check_fps(context, df)
        gate_checks["fps_valid"] = fps_check

        # 4) 丢帧（优先用声明 fps 作权威参照，见 _frame_loss_estimate 的取舍说明）
        if tcol is None:
            gate_checks["frame_loss"] = {
                "result": "skip", "detail": "未找到时间列，丢帧未检查",
            }
            not_audited.append("frame_loss")
        else:
            loss_ratio, evidence = _frame_loss_estimate(
                df[tcol], tcol,
                declared_fps=gate_checks.get("fps_valid", {}).get("declared_fps"),
            )
            if loss_ratio is None:
                gate_checks["frame_loss"] = {
                    "result": "skip",
                    "detail": evidence.get("reason", "丢帧不可估算"),
                }
                not_audited.append("frame_loss")
            else:
                gate_checks["frame_loss"] = {
                    "result": _FAIL if loss_ratio > settings.quality_gate_frame_loss_ratio else _PASS,
                    "value": loss_ratio,
                    "threshold": settings.quality_gate_frame_loss_ratio,
                    "evidence": evidence,
                    "detail": (
                        f"估算丢帧比例 {round(loss_ratio * 100, 2)}%"
                        + ("，超过阈值" if loss_ratio > settings.quality_gate_frame_loss_ratio else "，未超阈")
                        + (
                            "（按声明 fps 推算）" if evidence.get("basis") == "declared_fps"
                            else "（无声明 fps，按间隔一致性估计，无法发现均匀丢帧）"
                        )
                    ),
                }

        # 4) schema 一致性
        n_bad, mismatch_detail = _schema_consistency(df)
        if not mismatch_detail and n_bad == 0:
            gate_checks["schema_consistency"] = {
                "result": _PASS if df.columns.size else "skip",
                "detail": "跨 episode 列集合一致（或无 episode 划分，未做跨段比较）",
            }
        else:
            gate_checks["schema_consistency"] = {
                "result": _FAIL if n_bad > settings.quality_gate_schema_mismatch_max else _PASS,
                "value": n_bad,
                "threshold": settings.quality_gate_schema_mismatch_max,
                "mismatches": mismatch_detail,
                "detail": f"{n_bad} 个列在跨 episode 比较中不一致",
            }

        # 6) episode 边界
        gate_checks["episode_bounds"] = _check_episode_bounds(df)

    # L1 汇总：只有 FAIL 才判 fail；WARN 在此层也照常记录（但 gate 层允许 warn）。
    gate_fails = [k for k, v in gate_checks.items() if v.get("result") == _FAIL]
    gate_warns = [k for k, v in gate_checks.items() if v.get("result") == _WARN]
    gate_result = _FAIL if gate_fails else (_WARN if gate_warns else _PASS)

    # ---------- L2/L3 诊断（永不影响 verdict 的 fail 判定）----------
    diag_checks: dict[str, Any] = {}
    if df is not None and not df.empty and cols:
        diag_checks["idle_ratio"] = _diagnose_idle(
            df, cols, settings.quality_diagnostic_idle_speed,
            settings.quality_diagnostic_idle_ratio_warn,
            settings.quality_diagnostic_report_limit,
        )
        diag_checks["action_spike"] = _diagnose_spike(
            df, cols, settings.quality_diagnostic_spike_multiplier,
            settings.quality_diagnostic_report_limit,
        )
        diag_checks["actuator_saturation"] = _diagnose_saturation(
            df, cols, settings.quality_diagnostic_saturation_deg,
            settings.quality_diagnostic_report_limit,
        )
        diag_checks["action_jerk"] = _diagnose_jerk(
            df, cols, settings.quality_diagnostic_jerk_ratio,
            settings.quality_diagnostic_report_limit,
        )
        diag_checks["robot_induced_pause"] = _diagnose_pause(
            df, cols, settings.quality_diagnostic_idle_speed,
            settings.quality_diagnostic_pause_steps,
            settings.quality_diagnostic_report_limit,
        )
        diag_checks["arm_shaking"] = _diagnose_shake(
            df, cols, settings.quality_diagnostic_shake_ratio,
            settings.quality_diagnostic_report_limit,
        )
        diag_checks["path_efficiency"] = _diagnose_path_efficiency(
            df, cols, settings.quality_diagnostic_path_efficiency,
        )
    else:
        not_audited.append("motion_diagnostics")

    # 视觉诊断（依赖 video_meta，无则列 not_audited）。
    visual = _diagnose_visual(
        context, settings.quality_diagnostic_blur_variance,
        settings.quality_diagnostic_dark_mean,
    )
    diag_checks["visual_quality"] = visual
    not_audited.extend(visual.get("not_audited") or [])

    # 诊断汇总：诊断层的 warn **不参与** fail 判定。
    diag_warns = [k for k, v in diag_checks.items() if v.get("result") == _WARN]
    diag_result = _WARN if diag_warns else _PASS

    # ---------- 最终判定：只有硬门禁能判 fail ----------
    # 这是本模块最重要的不变量：diag 的 warn 永不升级为 fail。
    result = _FAIL if gate_result == _FAIL else (
        _WARN if (gate_result == _WARN or diag_result == _WARN) else _PASS
    )

    affected: list[str] = []
    if result != _PASS:
        if gate_fails or gate_warns:
            affected = ["whole_recording"]
        for k in ("idle_ratio", "action_spike", "actuator_saturation"):
            flagged = (diag_checks.get(k) or {}).get("flagged_episodes")
            if flagged:
                affected.extend(str(x) for x in flagged)
        affected = list(dict.fromkeys(affected))

    thresholds = {
        "gate_nan_ratio": settings.quality_gate_nan_ratio,
        "gate_frame_loss_ratio": settings.quality_gate_frame_loss_ratio,
        "gate_schema_mismatch_max": settings.quality_gate_schema_mismatch_max,
        "diagnostic_spike_multiplier": settings.quality_diagnostic_spike_multiplier,
        "diagnostic_saturation_deg": settings.quality_diagnostic_saturation_deg,
        "diagnostic_idle_speed": settings.quality_diagnostic_idle_speed,
        "diagnostic_idle_ratio_warn": settings.quality_diagnostic_idle_ratio_warn,
        "diagnostic_jerk_ratio": settings.quality_diagnostic_jerk_ratio,
        "diagnostic_pause_steps": settings.quality_diagnostic_pause_steps,
        "diagnostic_shake_ratio": settings.quality_diagnostic_shake_ratio,
        "diagnostic_path_efficiency": settings.quality_diagnostic_path_efficiency,
        "diagnostic_dark_mean": settings.quality_diagnostic_dark_mean,
        "diagnostic_blur_variance": settings.quality_diagnostic_blur_variance,
    }

    # ---- 用户消息：gate 与 diagnostics 必须分别转述（纪律 17）----
    #
    # 【阶段 4】必须**明确说明质检的是哪张表**：多表数据集里，同一份数据的不同表
    # 结论可能完全相反（末端表抖动超阈、主时钟表干净）。若不说表名，用户会把
    # "某张表有问题"误读为"整个数据集有问题"，反之亦然。
    #
    # **可读性要求（2026-09-21 用户反馈）**：此前消息里直接出现 `gate` /
    # `diagnostics` / `idle_ratio` 等内部术语，用户看不懂"action_spike: warn"
    # 是什么意思。现在一律使用中文标签，并在括号内说明白话含义。
    table_clause = f"（本次检查对象：{checked_table}）" if checked_table else ""
    parts = [
        f"质检结论：{_RESULT_ZH.get(result, result)}{table_clause}。"
    ]
    if gate_fails:
        parts.append(
            f"\n\n【必须处理的问题】{len(gate_fails)} 项——"
            + _humanize_rule_list(gate_fails)
            + "。这类是确定性错误（不是「可能有问题」，而是确实不对），"
            "建议修正后再使用数据。"
        )
    elif gate_warns:
        parts.append(
            f"\n\n【需要注意】{len(gate_warns)} 项——"
            + _humanize_rule_list(gate_warns)
            + "。不一定是错误，确认后可忽略。"
        )
    else:
        parts.append("\n\n【必须处理的问题】无——数据的基本完整性没有问题（无缺失值、时间顺序正常、无明显丢帧等）。")

    if diag_warns:
        parts.append(
            f"\n\n【参考提示】{len(diag_warns)} 项指标偏高——"
            + _humanize_rule_list(diag_warns)
            + "。**这些只是提示，不等于数据有问题**：判定阈值与任务类型强相关，"
            "比如「静止占比高」对放置、按压类任务完全正常，对连续抓取类才可疑。"
            "请结合你的任务特点判断。"
        )
    else:
        parts.append("\n\n【参考提示】各诊断指标均在正常范围。")

    if not_audited:
        readable = []
        for code in dict.fromkeys(not_audited):
            readable.append(
                _RULE_META[code]["label"] if code in _RULE_META else code)
        parts.append(
            f"\n\n【本次未检查】{len(readable)} 项——"
            + "、".join(readable)
            + "。**未检查不等于通过**，这些项目前无法判定。"
        )
    parts.append(
        "\n\n说明：以上判定阈值均为默认值，未针对本数据集校准；"
        "不同任务类型的合理范围差异很大，如需按数据集调整请告知。"
    )

    user_message = "".join(parts)

    # 结构化可读摘要：供 UI 面板与模型转述使用（键名全中文，避免内部术语）。
    readable_summary = {
        "结论": _RESULT_ZH.get(result, result),
        "检查对象": checked_table or "主表",
        "必须处理的问题": [
            {"名称": _RULE_META[c]["label"] if c in _RULE_META else c,
             "含义": _RULE_META[c]["what"] if c in _RULE_META else "",
             "建议": _RULE_META[c]["advice"] if c in _RULE_META else "",
             "规则码": c}
            for c in gate_fails
        ],
        "需要注意": [
            {"名称": _RULE_META[c]["label"] if c in _RULE_META else c,
             "规则码": c}
            for c in gate_warns
        ],
        "参考提示": [
            {"名称": _RULE_META[c]["label"] if c in _RULE_META else c,
             "含义": _RULE_META[c]["what"] if c in _RULE_META else "",
             "建议": _RULE_META[c]["advice"] if c in _RULE_META else "",
             "规则码": c}
            for c in diag_warns
        ],
        "本次未检查": [
            _RULE_META[c]["label"] if c in _RULE_META else c
            for c in dict.fromkeys(not_audited)
        ],
    }

    # 写回 meta["qc"]，与既有质检工具同款（供 compute_stats / generate_report 读取）。
    #
    # 【阶段 4】键名按质检对象区分：对非主表质检时写
    # `check_dataset_quality::<表名>`，**避免覆盖主表的质检结论**（此前只用一个
    # 固定键，多表场景下后一次质检会静默覆盖前一次，generate_report 读到的
    # 是哪张表的结论全看调用顺序——典型的静默数据丢失）。
    qc = context.meta.setdefault("qc", {})
    qc_key = (
        "check_dataset_quality"
        if table is None
        else f"check_dataset_quality::{checked_table or table}"
    )
    qc[qc_key] = {
        "result": result,
        "gate_result": gate_result,
        "diagnostics_result": diag_result,
        "dataset": dataset,
        "table": checked_table,
        "detail": {
            "gate_failures": gate_fails,
            "gate_warnings": gate_warns,
            "diagnostic_warnings": diag_warns,
            "not_audited": list(dict.fromkeys(not_audited)),
            "thresholds": thresholds,
        },
    }

    return {
        "success": True,
        "dataset": dataset,
        "check": "check_dataset_quality",
        # 【阶段 4】质检对象：多表数据集里结果必须连同表名一起被引用，
        # 否则"某张表有问题"会被误读为"整个数据集有问题"。
        "table": checked_table,
        "is_default_table": table is None,
        "result": result,
        "result_zh": _RESULT_ZH.get(result, result),
        # 分层语义必须分别给出，模型据此分别转述。
        # 每层附中文名与说明，避免把 gate/diagnostics 这类内部术语直接抛给用户。
        "gate": {
            "label": "必须处理的问题（确定性检查）",
            "explain": (
                "这一层的任何一项不通过，都说明数据确实存在错误——"
                "不是「可能有问题」，而是客观上不对。"
            ),
            "result": gate_result,
            "result_zh": _RESULT_ZH.get(gate_result, gate_result),
            "checks": _decorate_checks(gate_checks),
            "failed": gate_fails,
        },
        "diagnostics": {
            "label": "参考提示（统计性观察）",
            "explain": (
                "这一层是统计指标的观察结果，**超阈不等于数据有问题**："
                "判定阈值与任务类型强相关（如静止占比高对放置类任务完全正常），"
                "仅供你结合任务特点参考。这一层**永远不会**把结论升级为「未通过」。"
            ),
            "result": diag_result,
            "result_zh": _RESULT_ZH.get(diag_result, diag_result),
            "checks": _decorate_checks(diag_checks),
            "warned": diag_warns,
            "note": (
                "诊断项为启发式提示，**永不自动升级为 fail**。"
                "阈值与任务相关，请结合任务类型判断。"
            ),
        },
        # 纯中文的可读摘要（供 UI 面板与模型转述；无内部术语）。
        "readable_summary": readable_summary,
        "glossary": {
            code: {"名称": m["label"], "含义": m["what"], "白话": m["terms"]}
            for code, m in _RULE_META.items()
        },
        "measurements": {
            "n_rows": int(df.shape[0]) if df is not None else 0,
            "n_columns": int(df.columns.size) if df is not None else 0,
            "action_columns_analyzed": cols[:30],
            "n_action_columns": len(cols),
        },
        "thresholds": thresholds,
        "threshold_source": _SRC_DEFAULT,
        "affected_episodes": affected,
        "not_audited": list(dict.fromkeys(not_audited)),
        "user_message": user_message,
    }


@tool
def check_dataset_quality(
    wrapper: RunContextWrapper[RunContext], table: str | None = None
) -> dict:
    """对已加载数据集做质检（分层：硬门禁 + 诊断项）。

    一次调用即返回全部检查结果，不需逐项调用。返回结构分两层，**必须分别
    转述**：

    - ``gate``（L1 硬门禁，确定性错误，**可判 fail**）：NaN/Inf、时间戳
      单调性、丢帧比例、跨 episode schema 一致性、fps 合法性、episode 边界。
    - ``diagnostics``（L2/L3 诊断，启发式，**只 warn**）：空闲比、动作突波
      （疑似碰撞/人工干预）、致动器饱和、动作抖动、长停顿、异常振动、
      路径效率、视频元信息质量。

    **转述纪律（重要）**：``diagnostics`` 的 warn **不等于"数据有问题"**
    ——阈值与任务相关，70% 空闲比对 push/放置类任务可能完全正常。请把
    gate 与 diagnostics 分别转述，不要笼统说"质检未通过"。``not_audited``
    中的项是**未检查**，不得说成"通过了质检"。所有阈值均为默认值、
    未经该数据集验证，如需按数据集调整请提示用户改配置。

    涉及多张表时**每张表须分别质检**：不同表的质量问题性质不同（主时钟表查
    时间戳单调性，末端位置表查越界与抖动），结论不可互相代替。转述时必须带上
    ``table`` 字段说明质检对象，不得把某张表的结论说成整个数据集的结论。

    Args:
        table: 可选，要质检的表名（缺省=主表）。表名先经 list_tables 确认；
            不确定表名时不要猜测。传了不存在的表会返回 table_not_found 及
            可用表清单，**不会**静默回退到主表。

    Returns:
        分层质检返回：table（实际质检的表名）、is_default_table、result
        （pass/warn/fail，只有 gate 能判 fail）、gate（{result, checks,
        failed}）、diagnostics（{result, checks, warned}）、measurements、
        thresholds、threshold_source（default/dataset_override）、
        affected_episodes、not_audited、user_message。
    """
    return check_dataset_quality_impl(wrapper.context, table=table)
