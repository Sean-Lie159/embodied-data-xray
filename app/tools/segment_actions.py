"""任务切片边界检测（确定性信号计算，**只给边界不给名字**）。

设计依据：``docs/标注与质检能力设计.md`` §5.2.2。

**为什么只给边界不给名字**：切分点可以从机器人本体信号里**确定性**算出来
（速度突变、夹爪开合、停顿），但"这一段叫什么"（``atomic_action`` /
``action_description``）需要看懂画面——当前模型 ``hy3`` 不支持视觉输入
（三重官方证据：腾讯云 TokenHub 文档明示"混元语言模型不支持图片、视频等
多模态输入"、模型列表能力列无视觉、CodeBuddy 文档标 ``supportsImages:false``）。
若用纯文本模型"猜"动作名，就是无证据的编造，直接违反项目 SYSTEM_PROMPT
纪律 1（不得编造）与纪律 9（假设不得伪装成结论）。

因此本工具的输出是**候选边界草稿**，来源标记为 ``signal_derived``，
姿态形如 ``{episode_key, start_s, end_s, start_frame, end_frame, boundary_evidence}``
——可以直接喂给 :func:`app.tools.annotation_store.normalize_record` 落盘，
但动作名留空由用户在对话中填写。

**四类信号**（按可得性自动降级，全部确定性）：

1. **速度变化点**：状态通道一阶差分的幅度，突增处为动作切换的高概率位置；
2. **夹爪开合状态变化**：抓取/释放的物理标志，是**最可靠的语义边界**
   （RDA 实测发现人工接管边界处动作不连续尖峰富集 3.1 倍，印证本体信号
   能捕捉语义上重要的事件）；
3. **加速度突波**：``θ > 倍数 × median(|a|)``（HF GIGO 公开公式），
   对应碰撞/干预；
4. **低速段（停顿）边界**：速度低于阈值的连续段，其两端是天然切片点。

**诚实降级（关键纪律）**：以上信号**全部不可得**时返回 ``no_motion_signal``
并明确说明缺少哪些列，**不做等分兜底**——公开基准显示"每 5.77 秒等分"
的 F1 仅 0.070（等价于无信息），且会诱导用户误信它是有意义的切分。

粒度：``min_segment_s`` / ``max_segment_s`` **按数据集可配置**——业界对
"原子动作合理时长"无公认标准（DROID 5-20s、AgiBot 30-60s、OXE 多 <5s 均指
**整段演示**而非原子片段），故默认值仅为占位建议值并在返回中标注。

本模块不 import streamlit。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.config import get_settings
from app.tools import _data_access
from app.tools.timestamp_units import infer_unit, self_correct_unit, unit_to_ns_factor

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 边界证据来源标记。
_EV_VELOCITY = "velocity_change_point"
_EV_GRIPPER = "gripper_state_change"
_EV_SPIKE = "acceleration_spike"
_EV_PAUSE = "pause_boundary"

# 元数据列（不参与运动信号分析）。
_NON_MOTION_HINTS = (
    "timestamp", "time", "ts", "frame", "index", "episode", "fps",
    "sequence", "step", "id", "success", "reward", "done", "task",
)

# 夹爪列候选（抓取/释放的物理标志）。
_GRIPPER_COLS = (
    "gripper", "gripper_position", "gripper_state", "gripper_action",
    "gripper_open", "gripper_width", "grip", "hand_state", "finger",
    "gripper_effort", "gripper_velocity",
)
# 时间列候选（与其它工具同源）。
_TIME_COLS = (
    "timestamp", "time", "ts", "log_time", "frame_timestamp",
    "exposure_time", "time_stamp",
)
# episode 列候选。
_EPISODE_COLS = (
    "episode_index", "episode", "ep", "eps", "episode_id",
    "traj_id", "trajectory_id", "traj_index",
)
# 帧号列候选。
_FRAME_COLS = ("frame_index", "frame", "frame_id", "index")


# ---------------------------------------------------------------------------
# 列识别与信号提取
# ---------------------------------------------------------------------------


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """按候选名查找列（大小写不敏感、去空白）。

    两级匹配：**先精确**（避免 ``time`` 误命中 ``time_stamp`` 之类的歧义），
    **再前缀**（真实数据的列名常带单位后缀，如 ``timestamp_ms`` /
    ``timestamp_ns``——``timestamp_units.infer_unit`` 正是靠这些后缀判单位，
    因此这里必须能识别它们，否则时间轴会整体失效）。
    """
    lowered = {str(c).lower().strip(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    # 前缀匹配：候选名 + 分隔符（如 timestamp_ms / time_us）。
    for cand in candidates:
        for low, orig in lowered.items():
            if low.startswith(cand + "_") or low.startswith(cand + "."):
                return orig
    return None


def _find_cols_containing(
    df: pd.DataFrame, candidates: tuple[str, ...]
) -> list[str]:
    """查找所有候选名作为**子串**出现的列（夹爪列命名差异大，用包含匹配）。"""
    out: list[str] = []
    for c in df.columns:
        low = str(c).lower()
        if any(cand in low for cand in candidates):
            out.append(str(c))
    return out


def _motion_cols(df: pd.DataFrame) -> list[str]:
    """取参与运动分析的数值列（排除时间/索引/episode 等元数据列）。"""
    out: list[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        low = str(c).lower().strip()
        if any(h in low for h in _NON_MOTION_HINTS):
            continue
        out.append(str(c))
    return out


def _seconds_axis(
    df: pd.DataFrame, time_col: str | None, fps: float | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """构造以秒为单位的时间轴（供片段时间边界输出）。

    单位推断复用 ``timestamp_units``（列名后缀优先，其次差分量级，并自我
    纠正）；无时间列时按 fps 由帧号推算；两者都无则返回 None（诚实降级）。

    Returns:
        (秒数组或 None, 时间口径说明 dict)。
    """
    if time_col is not None and time_col in df.columns:
        raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype="float64")
        info = infer_unit(raw, col_name=time_col)
        unit = info.get("unit", "unknown")
        # 采样率合理性自我纠正（防单位误判）。
        corrected = self_correct_unit(raw, unit) if unit != "unknown" else None
        if corrected and corrected.get("corrected"):
            unit = corrected.get("unit", unit)
        factor = unit_to_ns_factor(unit)
        if factor is not None:
            ns = raw * factor
            # 归零到该表的起点，得到"相对秒"（片段时间边界用它更稳定，
            # 不受绝对时钟起点影响）。
            base = float(np.nanmin(ns))
            return (ns - base) / 1e9, {
                "basis": "time_column",
                "time_column": time_col,
                "unit": unit,
                "unit_basis": info.get("unit_basis", ""),
                "corrected": bool(corrected and corrected.get("corrected")),
                "note": "秒为相对该表起点的偏移（已按推断单位换算并归零）",
            }
        return None, {
            "basis": "unusable",
            "time_column": time_col,
            "unit": unit,
            "reason": (
                f"时间列 {time_col} 的单位推断为 {unit}，无法换算为秒"
                "（疑似帧序号或全重复值）——已放弃时间轴，不做猜测",
            ),
        }

    if fps and fps > 0:
        n = len(df)
        return np.arange(n, dtype="float64") / float(fps), {
            "basis": "frame_index_with_fps",
            "fps": float(fps),
            "note": f"无时间列，按 fps={fps} 由行序推算秒（假设行序即帧序）",
        }

    return None, {
        "basis": "no_time_axis",
        "reason": "既无可用时间列、也无 fps——无法给出以秒为单位的时间边界",
    }


def _gripper_fps(df: pd.DataFrame) -> float | None:
    """取声明的 fps（列或数据集画像），供帧↔秒换算。"""
    col = _find_col(df, ("fps", "frame_rate", "rate_hz"))
    if col is not None:
        try:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) >= 1 and float(v.iloc[0]) > 0:
                return float(v.iloc[0])
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# 边界信号计算
# ---------------------------------------------------------------------------


def _speed_profile(
    df: pd.DataFrame, motion_cols: list[str]
) -> tuple[np.ndarray | None, list[str]]:
    """计算归一化速度曲线（逐时刻的"运动强度"）。

    **为什么归一化**：不同通道量纲不同（关节角弧度 vs 位置米），未归一化时
    量程大的通道会支配结果。此处按各通道自身 P95 差分归一，再取均值。

    Returns:
        (速度曲线 数组或 None, 实际使用的列)。
    """
    used: list[str] = []
    series: list[np.ndarray] = []
    for c in motion_cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        if arr.size < 3:
            continue
        # 用前向填充处理偶发 NaN，避免差分被 NaN 污染整段。
        s = pd.Series(arr).ffill().bfill().to_numpy(dtype="float64")
        if not np.isfinite(s).all():
            continue
        used.append(c)
        series.append(s)

    if not series:
        return None, []

    n = min(s.size for s in series)
    mat = np.column_stack([s[:n] for s in series])
    diffs = np.abs(np.diff(mat, axis=0))
    if diffs.size == 0:
        return None, []

    # **恒定通道排查**：所有通道都无变化时，不存在任何运动信号——必须返回
    # None（走 no_motion_signal 分支），而不是给出一条全 0 的"速度曲线"，
    # 否则下游会把"无信息"误当成"有效信号"。
    if not np.any(diffs > 0):
        return None, []

    scale = np.nanpercentile(diffs, 95, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    norm = diffs / scale
    speed = np.nanmean(norm, axis=1)
    # 速度曲线长度 = n-1（差分少一个点），前补一个点对齐到原始行数。
    speed = np.concatenate([[speed[0] if speed.size else 0.0], speed])
    return speed, used


def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """滑动均值平滑（边界用 edge padding，保持长度不变）。"""
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window, dtype="float64") / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return out[:arr.size]


def _velocity_change_points(
    speed: np.ndarray, ratio: float, smooth_window: int = 0,
) -> list[int]:
    """速度变化点：**速度水平发生持续跃迁**的位置（动作切换的高概率点）。

    **关键设计（两轮实测修正，务必勿改回瞬时梯度判据）**：

    *初版*：直接找 ``|gradient(speed)| > ratio × median`` 的位置 → 在正弦型
    轨迹上产出 **120** 个候选点（正弦过零点梯度本就大），切片失去意义。

    *修正版*：先平滑再找"连续满足阈值的带"、每带取一点 → 仍产出 **49** 点。
    原因（这是本质）：**高频动作段的速度瞬时波动本身就大**，用"与全序列
    中位数的比值"作阈值的判据，会把高频段内部的周期性波动全部误判为切换点。

    *本版*：改用**水平跃迁判据**——比较窗口前后的速度**水平**（窗口均值），
    只有当"水平"发生显著变化时才认为发生了动作切换。

    **已知局限（必须诚实告知用户，勿宣称它可靠）**：实测表明该判据**无法
    区分"动作切换"与"运动强度变化"**。例如相邻两段都是正弦运动、但频率
    不同时，高频段的窗口均速天然更高，会被误判为切换点（实测在 4 段不同
    频率的合成轨迹上产出 12 个候选点，其中真实边界只有 3 个）。这是
    **通用信号判据的固有上限**——仅凭关节速度无法判定"语义上是否换了动作"。

    因此本函数的输出定位是**"值得人工查看的候选位置"**，而非"检测到的
    动作边界"；返回结果在 `boundary_evidence` 中如实标注用了速度信号及其
    局限，且对外层 `signals_used` 的语义说明为"候选"。

    **更可靠的是 `_gripper_events`**（夹爪开合是物理事件，实测在合成轨迹上
    精确命中全部真实边界）——它应作为首选信号。

    Returns:
        候选切分点的行索引（升序）。
    """
    if speed.size < 8:
        return []

    sp = _smooth(speed, smooth_window or max(3, speed.size // 100))
    n = sp.size

    # 水平窗口：约为序列长度的 2%（至少 3 点，最多 25 点）——太短则退化为
    # 瞬时判据（重蹈覆辙），太长则丢失切换位置的分辨率。
    win = int(min(25, max(3, n // 50)))

    # 各位置 i 的"水平跃迁量"：以 i 为中心，比较前 win 点与后 win 点的
    # 窗口均值之差。位置 i 的有效区间为 [win, n-win)。
    # 用累计和向量化计算两个滑动均值（O(n)）。
    csum = np.concatenate([[0.0], np.cumsum(sp)])
    idx = np.arange(win, n - win)  # 有效中心点
    if idx.size < 2:
        return []
    prev_mean = (csum[idx] - csum[idx - win]) / win       # [i-win, i)
    nxt_mean = (csum[idx + win] - csum[idx]) / win        # [i, i+win)
    level_shift = np.abs(nxt_mean - prev_mean)

    med = float(np.nanmedian(level_shift))
    if med <= 0:
        nz = level_shift[level_shift > 0]
        if nz.size == 0:
            return []
        med = float(np.nanmedian(nz))

    thr = ratio * med
    hot_local = level_shift > thr

    # 连续满足的区域归为一"跃迁带"，每带取跃迁量最大处的中心点为代表。
    candidates: list[int] = []
    i = 0
    m = hot_local.size
    while i < m:
        if not hot_local[i]:
            i += 1
            continue
        j = i
        while j < m and hot_local[j]:
            j += 1
        seg = level_shift[i:j]
        if seg.size:
            candidates.append(int(idx[i + int(np.nanargmax(seg))]))
        i = j
    return candidates


def _gripper_events(
    df: pd.DataFrame, gripper_cols: list[str], threshold: float = 0.5,
) -> tuple[list[int], list[str]]:
    """夹爪开合状态变化点（抓取/释放的物理标志，最可靠的语义边界）。

    做法：把夹爪通道二值化（相对其中位值的偏离超过 ``threshold`` 视为
    "开"），检测二值翻转位置。

    Returns:
        (事件行索引升序, 使用的列名)。
    """
    events: set[int] = set()
    used: list[str] = []
    for c in gripper_cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        if arr.size < 3 or not np.isfinite(arr).any():
            continue
        s = pd.Series(arr).ffill().bfill().to_numpy(dtype="float64")
        span = float(np.nanmax(s) - np.nanmin(s))
        if span <= 0:
            continue  # 恒定通道：无开合事件
        norm = (s - np.nanmin(s)) / span  # 归一到 0..1
        binary = (norm > threshold).astype(int)
        flips = np.where(np.diff(binary) != 0)[0]
        if flips.size:
            events.update(int(i) + 1 for i in flips)
            used.append(c)
    return sorted(events), used


def _spike_points(
    df: pd.DataFrame, motion_cols: list[str], multiplier: float,
) -> tuple[list[int], list[str]]:
    """加速度突波位置（HF GIGO 的 ``θ > 倍数 × median(|a|)`` 判据）。"""
    events: set[int] = set()
    used: list[str] = []
    for c in motion_cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        if arr.size < 4 or not np.isfinite(arr).any():
            continue
        s = pd.Series(arr).ffill().bfill().to_numpy(dtype="float64")
        accel = np.diff(s, n=2)
        accel = accel[np.isfinite(accel)]
        if accel.size == 0:
            continue
        med = float(np.median(np.abs(accel)))
        if med <= 0:
            continue
        idx = np.where(np.abs(accel) > multiplier * med)[0]
        if idx.size:
            events.update(int(i) + 2 for i in idx)
            used.append(c)
    return sorted(events), used


def _pause_boundaries(
    speed: np.ndarray, idle_speed: float, min_pause_steps: int,
) -> list[int]:
    """停顿段的两端（低速连续段的起止），是天然的切片点。

    Returns:
        边界行索引升序（段的起点与终点）。
    """
    if speed.size < 2:
        return []
    slow = speed < idle_speed
    boundaries: list[int] = []
    run_start: int | None = None
    for i, is_slow in enumerate(slow):
        if is_slow and run_start is None:
            run_start = i
        elif not is_slow and run_start is not None:
            if i - run_start >= min_pause_steps:
                boundaries.append(run_start)
                boundaries.append(i)
            run_start = None
    if run_start is not None and len(slow) - run_start >= min_pause_steps:
        boundaries.append(run_start)
        boundaries.append(len(slow) - 1)
    return boundaries


# ---------------------------------------------------------------------------
# 片段生成
# ---------------------------------------------------------------------------


def _merge_boundaries(
    boundaries: list[int], n_rows: int, min_gap: int,
) -> list[int]:
    """合并过近的边界，并强制包含 0 与 n_rows（首尾）。

    Args:
        boundaries: 候选边界行索引。
        n_rows: 总行数。
        min_gap: 相邻边界的最小间隔（行），小于此值的并入前一个。

    Returns:
        排序去重后的边界列表，首元素必为 0、末元素必为 n_rows。
    """
    pts = sorted({int(b) for b in boundaries if 0 < int(b) < n_rows})
    merged: list[int] = []
    for p in pts:
        if not merged or p - merged[-1] >= min_gap:
            merged.append(p)
    return [0, *merged, n_rows]


def _boundaries_to_segments(
    bounds: list[int],
    seconds: np.ndarray | None,
    fps: float | None,
) -> list[dict[str, Any]]:
    """把边界索引转成片段（含秒与帧号双表示）。

    秒的获取优先用时间轴数组；否则按 fps 由帧号推算（fps 不可得时为 None，
    **不猜**）。
    """
    segs: list[dict[str, Any]] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        start_frame = int(a)
        end_frame = int(b - 1)

        start_s: float | None = None
        end_s: float | None = None
        if seconds is not None:
            hi = min(b, seconds.size - 1)
            lo = min(a, seconds.size - 1)
            if hi >= 0 and lo >= 0:
                start_s = round(float(seconds[lo]), 6)
                # 片段末端取下一段起点的时间（半开区间 [a, b)），避免相邻
                # 片段之间出现人为空隙——与标注 schema 的 [start, end] 语义一致。
                end_s = round(
                    float(seconds[hi]) if hi < seconds.size else float(seconds[-1]),
                    6,
                )
                if end_s < start_s:
                    end_s = start_s
        elif fps and fps > 0:
            start_s = round(start_frame / fps, 6)
            end_s = round((end_frame + 1) / fps, 6)

        segs.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_s": start_s,
            "end_s": end_s,
            "n_frames": int(b - a),
        })
    return segs


def _apply_max_duration(
    segs: list[dict[str, Any]],
    speed: np.ndarray | None,
    max_segment_s: float,
    fps: float | None,
) -> list[dict[str, Any]]:
    """对过长片段在"最低信号谷值"处再切（不按固定长度硬切）。

    **为什么在谷值处切**：等分硬切会切断动作，而速度谷值（动作间的短暂
    停顿/稳定点）是更自然的次级边界。
    """
    if not segs or not max_segment_s or max_segment_s <= 0:
        return segs

    out: list[dict[str, Any]] = []
    for seg in segs:
        dur = None
        if seg.get("start_s") is not None and seg.get("end_s") is not None:
            dur = seg["end_s"] - seg["start_s"]
        else:
            dur = seg["n_frames"] / fps if fps else None

        if dur is None or dur <= max_segment_s:
            out.append(seg)
            continue

        # 需要切的段数（上取整），并算出每段的目标行数。
        n_parts = int(np.ceil(dur / max_segment_s))
        a, b = seg["start_frame"], seg["end_frame"] + 1
        total = b - a
        if total < n_parts or speed is None:
            out.append(seg)
            continue

        # 依次找谷值点：在每个目标位置的邻域窗口内取速度最小处。
        cuts = [a]
        for k in range(1, n_parts):
            target = a + int(round(total * k / n_parts))
            w = max(1, total // (n_parts * 4))
            lo = max(cuts[-1] + 1, target - w)
            hi = min(b - 1, target + w)
            if hi <= lo:
                cut = target
            else:
                window = speed[lo:hi]
                if window.size == 0:
                    cut = target
                else:
                    cut = lo + int(np.nanargmin(window))
            cuts.append(cut)
        cuts.append(b)

        sub = _boundaries_to_segments(cuts, None, fps)
        # 秒值需从父片段线性换算（保留原始时间口径）。
        if seg.get("start_s") is not None and seg.get("end_s") is not None:
            s0, s1 = seg["start_s"], seg["end_s"]
            span = s1 - s0
            for item in sub:
                item["start_s"] = round(
                    s0 + span * (item["start_frame"] - a) / total, 6)
                item["end_s"] = round(
                    s0 + span * ((item["end_frame"] + 1) - a) / total, 6)
        out.extend(sub)
    return out


def _merge_short_segments(
    segs: list[dict[str, Any]], min_segment_s: float, fps: float | None,
) -> list[dict[str, Any]]:
    """合并过短片段到相邻片段（避免产出大量无意义的碎片）。"""
    if not segs or not min_segment_s or min_segment_s <= 0:
        return segs

    def _dur(s: dict[str, Any]) -> float | None:
        if s.get("start_s") is not None and s.get("end_s") is not None:
            return s["end_s"] - s["start_s"]
        if fps:
            return s["n_frames"] / fps
        return None

    out: list[dict[str, Any]] = [dict(segs[0])]
    for seg in segs[1:]:
        d = _dur(seg)
        if d is not None and d < min_segment_s:
            # 并入前一段（延长其末端）。
            out[-1]["end_frame"] = seg["end_frame"]
            out[-1]["n_frames"] = (
                out[-1]["end_frame"] - out[-1]["start_frame"] + 1
            )
            if out[-1].get("end_s") is not None and seg.get("end_s") is not None:
                out[-1]["end_s"] = seg["end_s"]
        else:
            out.append(dict(seg))
    # 若首段过短，并入后一段。
    if len(out) > 1:
        d0 = _dur(out[0])
        if d0 is not None and d0 < min_segment_s:
            out[1]["start_frame"] = out[0]["start_frame"]
            out[1]["n_frames"] = (
                out[1]["end_frame"] - out[1]["start_frame"] + 1
            )
            if out[0].get("start_s") is not None:
                out[1]["start_s"] = out[0]["start_s"]
            out.pop(0)
    return out


# ---------------------------------------------------------------------------
# 主实现
# ---------------------------------------------------------------------------


def _segment_one_episode(
    sub: pd.DataFrame,
    *,
    method: str,
    settings,
    time_col: str | None,
    fps: float | None,
) -> dict[str, Any]:
    """对单个 episode（行子集）做切分，返回片段列表与所用/缺失信号。"""
    motion_cols = _motion_cols(sub)
    speed, used_cols = _speed_profile(sub, motion_cols)
    gripper_cols = _find_cols_containing(sub, _GRIPPER_COLS)

    signals_used: list[str] = []
    signals_missing: list[str] = []
    boundaries: list[int] = []

    want = set()
    if method == "auto":
        want = {"velocity", "gripper", "spike", "pause"}
    else:
        want = {method}

    # --- 夹爪（最高优先级的语义边界）---
    if "gripper" in want:
        if gripper_cols:
            ev, gused = _gripper_events(sub, gripper_cols)
            if gused:
                boundaries.extend(ev)
                signals_used.append(_EV_GRIPPER)
            else:
                signals_missing.append(
                    f"夹爪列 {gripper_cols} 存在但为恒定值（无开合事件）"
                )
        else:
            signals_missing.append("未找到夹爪列（gripper*）")

    # --- 速度变化点 ---
    if "velocity" in want:
        if speed is not None and used_cols:
            boundaries.extend(_velocity_change_points(
                speed, settings.annotation_change_point_ratio))
            signals_used.append(_EV_VELOCITY)
        else:
            signals_missing.append("无法计算速度曲线（无可用运动通道）")

    # --- 加速度突波 ---
    if "spike" in want:
        if motion_cols:
            ev, sused = _spike_points(
                sub, motion_cols, settings.quality_diagnostic_spike_multiplier)
            if sused:
                boundaries.extend(ev)
                if ev:
                    signals_used.append(_EV_SPIKE)
            else:
                signals_missing.append("突波不可计算（加速度中位数为 0 或样本不足）")
        else:
            signals_missing.append("无运动通道，突波不可计算")

    # --- 停顿边界 ---
    if "pause" in want:
        if speed is not None:
            pb = _pause_boundaries(
                speed, settings.annotation_idle_speed,
                max(2, settings.annotation_min_pause_steps),
            )
            if pb:
                boundaries.extend(pb)
                signals_used.append(_EV_PAUSE)
        else:
            signals_missing.append("无速度曲线，停顿边界不可计算")

    if not boundaries:
        return {
            "segments": [],
            "signals_used": [],
            "signals_missing": signals_missing or ["未检出任何边界信号"],
            "n_motion_channels": len(motion_cols),
            "n_motion_channels_used": len(used_cols),
        }

    # 时间轴
    seconds, time_info = _seconds_axis(sub, time_col, fps)

    # 边界合并 → 片段 → 粒度约束
    min_gap = 1
    eff_fps = fps
    if seconds is not None and seconds.size > 1:
        # 由秒轴推算真实采样率，用于把"最小片段秒数"折算成行数。
        total_s = float(seconds[-1] - seconds[0])
        if total_s > 0:
            eff_fps = (seconds.size - 1) / total_s
    if eff_fps and eff_fps > 0:
        min_gap = max(1, int(round(
            settings.annotation_min_segment_s / 4 * eff_fps)))

    bounds = _merge_boundaries(boundaries, len(sub), min_gap)
    segs = _boundaries_to_segments(bounds, seconds, eff_fps)

    # 先切过长、再并过短（顺序有讲究：先保证不超上限，再消除碎片）。
    segs = _apply_max_duration(
        segs, speed, settings.annotation_max_segment_s, eff_fps)
    segs = _merge_short_segments(
        segs, settings.annotation_min_segment_s, eff_fps)

    # 为每个片段附边界证据（该片段起点用了哪些信号）。
    bset = {
        _EV_GRIPPER: set(_gripper_events(sub, gripper_cols)[0]) if gripper_cols else set(),
        _EV_PAUSE: set(
            _pause_boundaries(
                speed, settings.annotation_idle_speed,
                max(2, settings.annotation_min_pause_steps))
        ) if speed is not None else set(),
    }
    for s in segs:
        ev: list[str] = []
        sf = s["start_frame"]
        if sf in bset[_EV_GRIPPER]:
            ev.append(_EV_GRIPPER)
        if sf in bset[_EV_PAUSE]:
            ev.append(_EV_PAUSE)
        if sf == 0:
            ev.append("sequence_start")
        s["boundary_evidence"] = {
            "signals": ev or ["velocity_change_point"],
            "note": (
                "边界由本体信号确定性计算得出；动作名称需人工填写或经视觉模型生成。"
                "**边界本身是候选**：夹爪开合事件（物理事件）可靠度高；"
                "速度变化点无法区分「动作切换」与「运动强度变化」，可能存在误报，"
                "请人工复核。"
            ),
        }

    return {
        "segments": segs,
        "signals_used": sorted(set(signals_used)),
        "signals_missing": signals_missing,
        "n_motion_channels": len(motion_cols),
        "n_motion_channels_used": len(used_cols),
        "time_info": time_info,
        "speed_available": speed is not None,
    }


def segment_actions_impl(
    context: RunContext,
    episode_key: str | None = None,
    table: str | None = None,
    method: str = "auto",
    min_segment_s: float | None = None,
    max_segment_s: float | None = None,
    settings=None,
) -> dict[str, Any]:
    """用机器人本体信号做确定性动作切片，输出**候选边界**（不含动作名称）。

    Args:
        context: 运行时上下文。
        episode_key: 指定 episode（缺省对全部锚点执行）。取值来自
            ``resolve_anchors`` 返回的 ``episode_key``。
        table: 可选，指定状态/动作表名（缺省自动定位）。
        method: auto / velocity / gripper / spike / pause（缺省 auto）。
        min_segment_s: 最小片段时长（秒），短于此的相邻片段合并。
        max_segment_s: 最大片段时长（秒），超过则在速度谷值处再切。
        settings: 可选配置覆盖（测试注入）。

    Returns:
        dict，含 success、episodes（逐 episode 的候选片段列表）、method_used、
        signals_used、signals_missing、thresholds、n_segments、user_message。
        **信号全缺失时** success=False 且 error="no_motion_signal"。
    """
    settings = settings or get_settings()

    if min_segment_s is not None:
        settings = settings.model_copy(
            update={"annotation_min_segment_s": float(min_segment_s)})
    if max_segment_s is not None:
        settings = settings.model_copy(
            update={"annotation_max_segment_s": float(max_segment_s)})

    if method not in ("auto", "velocity", "gripper", "spike", "pause"):
        return {
            "success": False,
            "error": "invalid_method",
            "reason": f"method 必须是 auto/velocity/gripper/spike/pause，收到 {method!r}",
            "user_message": "切片方法参数非法。请使用 auto（自动按可得信号）或指定 velocity/gripper/spike/pause。",
        }

    dataset_id = context.dataset_id
    if dataset_id is None and context.df is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载任何数据集",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset。",
        }

    # 定位状态/动作表。
    if table is not None:
        resolved = _data_access.resolve_table_name(context, table)
        if not resolved["success"]:
            return {
                "success": False,
                "error": resolved.get("error", "table_unavailable"),
                "reason": resolved.get("reason"),
                "table": table,
                "dataset": dataset_id,
                "user_message": resolved.get(
                    "user_message", f"指定的表 {table} 不可用。"),
            }
        df = resolved["df"]
        table_name = resolved.get("table_name") or table
    else:
        df, _src = _data_access.locate_action_table(context)
        table_name = "main" if _src == "main" else (_src or "unknown")

    if df is None or df.empty:
        return {
            "success": False,
            "error": "not_applicable",
            "reason": "数据集中无可用的状态/动作数据表",
            "dataset": dataset_id,
            "user_message": (
                "动作切片需要状态/动作数据表（含关节/末端执行器通道）。"
                "当前数据集无可用的此类数据。"
            ),
            "suggested_tools": ["profile_data", "inspect_streams"],
        }

    time_col = _find_col(df, _TIME_COLS)
    fps = _gripper_fps(df)
    ep_col = _find_col(df, _EPISODE_COLS)
    frame_col = _find_col(df, _FRAME_COLS)

    # 按 episode 分组（无 episode 列时整段视为一个 episode）。
    if ep_col is not None:
        groups = list(df.groupby(ep_col, sort=True, dropna=False))
    else:
        groups = [("__all__", df)]

    episodes: list[dict[str, Any]] = []
    all_used: set[str] = set()
    all_missing: set[str] = set()
    any_segments = False

    for key, sub in groups:
        skey = str(key)
        if episode_key is not None and skey != str(episode_key):
            continue
        sub = sub.reset_index(drop=True)
        res = _segment_one_episode(
            sub, method=method, settings=settings, time_col=time_col, fps=fps)

        # 帧号：以该 episode 内的原始帧号为准（若有帧列），否则用行序。
        if frame_col is not None and frame_col in sub.columns:
            try:
                fr = pd.to_numeric(sub[frame_col], errors="coerce").dropna()
                base_frame = int(fr.iloc[0]) if len(fr) else 0
            except (ValueError, TypeError):
                base_frame = 0
        else:
            base_frame = 0

        segs = []
        for s in res["segments"]:
            item = dict(s)
            item["start_frame"] = base_frame + int(s["start_frame"])
            item["end_frame"] = base_frame + int(s["end_frame"])
            segs.append(item)

        all_used.update(res["signals_used"])
        all_missing.update(res["signals_missing"])
        if segs:
            any_segments = True

        episodes.append({
            "episode_key": skey,
            "n_frames": int(len(sub)),
            "n_segments": len(segs),
            "segments": segs,
            "signals_used": res["signals_used"],
            "signals_missing": res["signals_missing"],
            "n_motion_channels": res["n_motion_channels"],
            "n_motion_channels_used": res.get("n_motion_channels_used", 0),
            "time_info": res.get("time_info"),
        })

    if episode_key is not None and not episodes:
        return {
            "success": False,
            "error": "episode_not_found",
            "reason": f"未找到 episode_key={episode_key!r}",
            "dataset": dataset_id,
            "available_hint": "请先调用 resolve_anchors 获取有效 episode_key 清单",
            "user_message": (
                f"未找到 episode {episode_key}。请先取 episode 锚点清单，"
                "确认键名后再指定。"
            ),
        }

    # ---- 诚实降级：完全无信号时明确失败，**不做等分兜底** ----
    if not any_segments:
        return {
            "success": False,
            "error": "no_motion_signal",
            "dataset": dataset_id,
            "table": table_name,
            "reason": (
                "未能从数据中检出任何可用于切分的运动信号"
                "（无可用的速度/夹爪/突波/停顿信号）"
            ),
            "signals_missing": sorted(all_missing),
            "episodes_probed": len(episodes),
            "user_message": (
                "未能在该数据中找到可用于切分的运动信号，因此**没有产出切片**。\n"
                f"缺少的信号：{'；'.join(sorted(all_missing)) or '（未识别到运动通道）'}\n\n"
                "为什么不做等分兜底：把轨迹按固定时长等分产生的边界没有物理依据"
                "（公开基准显示等分基线的标注 F1 仅 0.070，等价于无信息），"
                "会给下游标注和训练引入噪声。\n\n"
                "建议：① 用 inspect_streams / profile_data 确认数据是否含关节或"
                "末端执行器通道；② 若动作数据在独立文件里，用 table 参数指定；"
                "③ 若确实无本体信号（例如只有视频），请告知，可考虑接入视觉模型"
                "或直接人工标注。"
            ),
            "suggested_tools": ["inspect_streams", "profile_data", "resolve_anchors"],
        }

    # 汇总阈值（供模型如实说明"默认值未经数据集验证"）。
    thresholds = {
        "change_point_ratio": settings.annotation_change_point_ratio,
        "idle_speed": settings.annotation_idle_speed,
        "min_pause_steps": settings.annotation_min_pause_steps,
        "min_segment_s": settings.annotation_min_segment_s,
        "max_segment_s": settings.annotation_max_segment_s,
        "spike_multiplier": settings.quality_diagnostic_spike_multiplier,
    }

    total_segs = sum(e["n_segments"] for e in episodes)
    missing_note = ""
    if all_missing:
        missing_note = (
            f"以下信号不可得（已跳过）：{'；'.join(sorted(all_missing))}。"
        )

    reliability = ""
    if _EV_GRIPPER in all_used:
        reliability = (
            "其中夹爪开合事件是**物理事件**，可靠度最高；"
        )
    if _EV_VELOCITY in all_used:
        reliability += (
            "速度变化点**只是候选**——它无法区分「动作切换」与「运动强度变化」"
            "（例如相邻两段都是正弦运动但频率不同时会被误判），请人工复核每一条边界。"
        )

    user_message = (
        f"已产出 {total_segs} 个候选片段（{len(episodes)} 个 episode），"
        f"使用的信号：{'、'.join(sorted(all_used))}。\n"
        f"{missing_note}"
        f"{reliability}\n"
        "**这是只有边界、没有动作名称的草稿**——动作名称需要看懂画面，"
        "当前模型不支持视觉输入，因此不能自动生成。请人工命名、修正边界，"
        "或先接入视觉模型再让它提出待确认的名称。\n"
        "另外：切分阈值均为默认值，未经该数据集验证，"
        "原子动作的合理时长业界无公认标准，请按数据实际情况调整。"
    )

    return {
        "success": True,
        "dataset": dataset_id,
        "table": table_name,
        "method_used": method,
        "episodes": episodes,
        "n_episodes": len(episodes),
        "n_segments": total_segs,
        "signals_used": sorted(all_used),
        "signals_missing": sorted(all_missing),
        "thresholds": thresholds,
        "threshold_source": "default",
        "boundary_only": True,
        "action_name_generated": False,
        "note": (
            "仅产出时间边界（来源 signal_derived）；atomic_action 与 "
            "action_description 需人工填写或经视觉模型生成，本工具不生成。"
        ),
        "user_message": user_message,
    }


@tool
def segment_actions(
    wrapper: RunContextWrapper[RunContext],
    episode_key: str | None = None,
    table: str | None = None,
    method: str = "auto",
    min_segment_s: float | None = None,
    max_segment_s: float | None = None,
) -> dict:
    """用机器人本体信号做确定性动作切片，输出**候选边界**（不含动作名称）。

    **只给边界，不给名字**——命名需要看懂画面，当前模型不支持视觉输入，
    请让用户在对话中命名，或先接入视觉模型再生成待确认的名称。

    四类信号（按可得性自动降级，全部确定性）：夹爪开合状态变化（最可靠的
    语义边界）、速度变化点、加速度突波、停顿段边界。

    Args:
        episode_key: 指定 episode（缺省对全部 episode 执行）。取值来自
            resolve_anchors 返回的 episode_key。
        table: 可选，指定状态/动作表名（缺省自动定位）。
        method: auto / velocity / gripper / spike / pause（缺省 auto 按可得信号选）。
        min_segment_s: 最小片段时长（秒），短于此的相邻片段合并；缺省取配置值。
        max_segment_s: 最大片段时长（秒），超过则在速度谷值处再切；缺省取配置值。

    Returns:
        dict，含 episodes（逐 episode 的候选片段，每段含 start_s/end_s/
        start_frame/end_frame/boundary_evidence）、method_used、signals_used、
        signals_missing、thresholds、n_segments。**信号全缺失时** success=False
        且 error="no_motion_signal"——此时**不要**自行编造边界或按固定时长
        等分，请如实告知用户缺少哪些信号并建议下一步。
    """
    return segment_actions_impl(
        wrapper.context,
        episode_key=episode_key,
        table=table,
        method=method,
        min_segment_s=min_segment_s,
        max_segment_s=max_segment_s,
    )
