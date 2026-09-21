"""指标单一来源（metrics 物化）：同一事实只由本模块写入 `meta["metrics"]`。

设计依据：``docs/指标单一来源与数据集画像设计.md`` 第一部分（A）。

## 要解决的问题（真实事故，2026-09-21）

同一份手套数据（Forsense-G7，``20260921_151733``）、同一个事实（采样率），
**四个位置给出四种说法**：

| 位置 | 显示 | 原因 |
|---|---|---|
| 报告流明细 | 未知 | ``measured_rate`` 为 None（单位推断失败） |
| 时间同步检查 | 120.007 Hz | 它自己现读时间戳列重算（算法不同） |
| 质检采样率校验 | skip | 只读 ``fps`` 列与 LeRobot 元数据，**不读缓存** |
| 标称率核对 | skipped | ``nominal_rate_hz`` 字段**只有读、没有写** |

排查确认采样率有 **7 处独立实现**，其中大部分复用不了缓存。
**根因**：``RunContext.meta`` 被当作"可写可不写的备忘录"，而非
"唯一事实来源"——同一 key 有多个写入者、算法不同、谁后跑谁覆盖。

## 本模块的设计（借鉴 Deequ 的 Analyzer/Check 分离）

1. **单一写入口**：只有本模块写 ``meta["metrics"]``。有 AST 扫描的守护测试
   （``test_only_metrics_store_writes_metrics``）强制这一点——不靠自律，
   靠结构。
2. **实测值与标称值并存、不竞争**：
   - ``sample_rate_hz``（provenance=``measured``）——**唯一权威**，供计算；
   - ``nominal_rate_hz``（provenance=``declared``）——**只作比较基准**，
     永不覆盖实测。
   这是对"权威链"方案的**否决性修正**：若让实测覆盖标称，就会永久丢失
   "声明 120 而实际只有 100"这种真正该报警的能力。二者是"设计目标"与
   "实际表现"的关系，不该塞进同一个 key。
3. **一致性用容差判定**：``deviation_ratio <= tolerance_ratio`` 即算一致。
   默认容差 5%（依据：Soda 的 ``percent_threshold`` 默认 0.05、
   Great Expectations 同为百分比阈值）。
   实测 120.007 vs 声明 120 的偏差是 **0.0058%**，判为一致——符合直觉。
4. **统一算法链**：时间列名（含 ``_ns``/``_us`` 后缀线索）→ ``infer_unit``
   （**必须传列名**）→ ``self_correct_unit`` 自纠 → 归一化到 ns → 差分取
   中位数 → ``1e9 / median``。**先归一化再算**，杜绝无量纲差分。

本模块不 import streamlit。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.agent.context import RunContext

# 来源标记（provenance）。
PROV_MEASURED = "measured"      # 由数据实测得出
PROV_DECLARED = "declared"      # 由配置/元数据声明
PROV_INFERRED = "inferred"      # 由启发式推断（置信度最低）

# 默认容差：|measured - nominal| / nominal。
DEFAULT_TOLERANCE_RATIO = 0.05

# 时间列候选（与其它工具同源；此处独立列出以免语义漂移）。
_TIME_COL_CANDIDATES = (
    "timestamp", "time", "ts", "log_time", "frame_timestamp",
    "exposure_time", "time_stamp",
)
# 声明采样率的列候选。
_FPS_COL_CANDIDATES = ("fps", "frame_rate", "rate_hz", "sample_rate", "hz")
# 配置文件中可能承载"标称采样率"的键（**必须明确表达采样率语义**，
# 不做模糊匹配——避免把无关数字当采样率）。
_CONFIG_RATE_KEYS = (
    "nominal_hz", "sample_rate_hz", "sampling_rate_hz", "rate_hz",
    "nominal_hz_actual", "fps", "sample_rate", "sampling_rate", "hz",
)


def _now_iso() -> str:
    """当前 UTC 时间（ISO 8601，秒精度）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """查找列：精确匹配优先，其次前缀匹配（识别 ``timestamp_ns`` 这类带单位后缀名）。"""
    lowered = {str(c).lower().strip(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:
        for low, orig in lowered.items():
            if low.startswith(cand + "_") or low.startswith(cand + "."):
                return orig
    return None


# ---------------------------------------------------------------------------
# 实测值：统一算法链
# ---------------------------------------------------------------------------


def measure_rate_hz(
    values: pd.Series | np.ndarray,
    *,
    col_name: str | None = None,
) -> dict[str, Any] | None:
    """从时间戳序列估算采样率（**统一算法链，全项目唯一实现**）。

    **为什么必须传 ``col_name``**（缺陷 A2-1）：``timestamp_units.infer_unit``
    把**列名后缀**（``_ns``/``_us``/``_ms``）当作最高优先级的单位线索。
    不传列名就退化为纯量级猜测，对 ``timestamp_ns`` 可能失准或返回 unknown
    ——这正是"报告流明细显示未知、而时间同步检查却算出 120.007Hz"的根因
    （前者漏传列名，后者传了）。

    **为什么必须先归一化再算**（缺陷 A2-2）：初版
    ``check_dataset_quality._check_fps`` 直接做 ``1.0 / median_step``——
    对纳秒间隔 8333100 会得出 ``1.2e-7`` 这种无意义值。本实现统一先换算到
    纳秒再求速率。

    统计量用**中位数**（而非均值）：对偶发的丢帧/抖动更稳健。

    Args:
        values: 时间戳序列（数值）。
        col_name: 该时间列的名称——**强烈建议传**，携带单位后缀线索。

    Returns:
        dict（``value`` / ``provenance`` / ``source`` / ``unit`` / ``evidence``
        / ``n_samples``），无法估算时 None（诚实降级，不猜）。
    """
    from app.tools.timestamp_units import infer_unit, self_correct_unit, to_ns

    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return None

    info = infer_unit(arr, col_name=col_name) if col_name else infer_unit(arr)
    unit = info.get("unit", "unknown")

    # 采样率合理性自纠：能纠正"把秒级间隔误判为 ns"这类错误。
    corrected = False
    if unit != "unknown":
        c = self_correct_unit(arr, unit)
        if c and c.get("corrected"):
            unit = c.get("unit", unit)
            corrected = True

    if unit not in ("ns", "us", "ms", "s"):
        return None

    try:
        ns = to_ns(arr, unit)
    except (ValueError, TypeError):
        return None

    diffs = np.diff(ns)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    median_ns = float(np.median(diffs))
    if median_ns <= 0:
        return None

    rate = 1e9 / median_ns
    return {
        "value": round(float(rate), 6),
        "provenance": PROV_MEASURED,
        "source": "timestamp_diffs",
        "unit": unit,
        "unit_corrected": corrected,
        "unit_basis": info.get("unit_basis", ""),
        "median_interval_ns": median_ns,
        "n_samples": int(arr.size),
        "evidence": (
            f"对时间列差分取中位数（{median_ns:.1f} ns）后求倒数；"
            f"单位推断为 {unit}"
            + ("（经采样率合理性自纠）" if corrected else "")
            + (f"，列名证据：{info.get('unit_basis')}" if info.get("unit_basis") else "")
        ),
        "measured_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 标称值：声明来源
# ---------------------------------------------------------------------------


def nominal_rate_from_config(run_context: RunContext) -> dict[str, Any] | None:
    """从**配置型 JSON**（``status == "config"`` 的流）读标称采样率。

    这是缺陷 A2-3 的修复：``session.json`` 的 ``nominal_hz: 120`` 此前只被
    登记为 ``config_keys``（键名清单），**从未流入任何采样率计算**，导致
    ``nominal_check`` 永久 skipped。

    取用规则**保守**：仅当键名明确表达"采样率"语义（见 ``_CONFIG_RATE_KEYS``，
    如 ``nominal_hz`` / ``sample_rate_hz``）时才取；不做模糊匹配。

    Args:
        run_context: 运行时上下文（读 meta.streams 中 config 流的 config 字段）。

    Returns:
        dict（``value`` / ``provenance`` / ``source`` / ``evidence``）或 None。
    """
    for s in run_context.meta.get("streams", []) or []:
        if s.get("status") != "config":
            continue
        cfg = s.get("config")
        if not isinstance(cfg, dict):
            continue
        for key in _CONFIG_RATE_KEYS:
            if key not in cfg:
                continue
            raw = cfg[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            if raw <= 0:
                continue
            name = s.get("path", "config")
            from pathlib import Path as _P

            return {
                "value": float(raw),
                "provenance": PROV_DECLARED,
                "source": f"{_P(str(name)).name}:{key}",
                "evidence": f"配置文件 {_P(str(name)).name} 声明的 {key}",
                "declared_at": _now_iso(),
            }
    return None


def nominal_rate_from_columns(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """从数据列的 ``fps`` / ``frame_rate`` / ``rate_hz`` 读标称采样率。"""
    if df is None or df.empty:
        return None
    col = _find_col(df, _FPS_COL_CANDIDATES)
    if col is None:
        return None
    try:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
    except (ValueError, TypeError):
        return None
    if vals.empty:
        return None
    v = float(vals.iloc[0])
    if v <= 0:
        return None
    return {
        "value": v,
        "provenance": PROV_DECLARED,
        "source": f"column:{col}",
        "evidence": f"数据列 {col} 声明的采样率",
        "declared_at": _now_iso(),
    }


def nominal_rate_from_lerobot(run_context: RunContext) -> dict[str, Any] | None:
    """从 LeRobot ``meta/info.json`` 的 ``fps`` 读标称采样率。"""
    import json
    from pathlib import Path as _P

    src = str(run_context.meta.get("source", "") or "")
    if not src:
        return None
    info = _P(src) / "meta" / "info.json"
    if not info.exists():
        return None
    try:
        obj = json.loads(info.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    fps = obj.get("fps") if isinstance(obj, dict) else None
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        return None
    return {
        "value": float(fps),
        "provenance": PROV_DECLARED,
        "source": "meta/info.json:fps",
        "evidence": "LeRobot meta/info.json 声明的 fps",
        "declared_at": _now_iso(),
    }


def _collect_nominal(run_context: RunContext, df: pd.DataFrame | None) -> dict[str, Any] | None:
    """按优先级收集**数据集级**标称采样率（只写 nominal，不碰实测）。

    优先级：平台配置文件 > 数据列 > LeRobot 元数据。
    全部缺失时返回 None（**如实说无法比较，不猜**）。
    """
    for getter in (
        lambda: nominal_rate_from_config(run_context),
        lambda: nominal_rate_from_columns(df),
        lambda: nominal_rate_from_lerobot(run_context),
    ):
        try:
            got = getter()
        except (ValueError, TypeError, OSError):
            got = None
        if got:
            return got
    return None


# ---------------------------------------------------------------------------
# 物化与读取（唯一写入口）
# ---------------------------------------------------------------------------


def materialize(
    run_context: RunContext,
    *,
    df: pd.DataFrame | None = None,
    force: bool = False,
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> dict[str, Any]:
    """物化采样率指标到 ``meta["metrics"]``（**唯一写入口，幂等**）。

    幂等性很重要：重复调用不应重算、更不应产生第二个值。已物化且未
    ``force`` 时直接返回既有结果——这是"单一来源"的保障。

    Args:
        run_context: 运行时上下文。
        df: 用于测量/读声明的数据表；缺省用 ``run_context.df``。
        force: 强制重算（如换了主表）。
        tolerance_ratio: 一致性判定容差（默认 5%）。

    Returns:
        ``meta["metrics"]`` 的内容（含 ``sample_rate_hz`` 实测与
        ``nominal_rate_hz`` 标称两个**独立**字段，以及 ``consistency``）。
    """
    metrics = run_context.meta.get("metrics")
    if isinstance(metrics, dict) and not force and metrics.get("_materialized"):
        return metrics

    table = df if df is not None else run_context.df

    measured: dict[str, Any] | None = None
    if table is not None and not table.empty:
        col = _find_col(table, _TIME_COL_CANDIDATES)
        if col is not None:
            # **必须传列名**（缺陷 A2-1 的修复点）。
            measured = measure_rate_hz(table[col], col_name=col)
            if measured is not None:
                measured["time_column"] = col

    nominal = _collect_nominal(run_context, table)

    out: dict[str, Any] = {
        "_materialized": True,
        "_materialized_at": _now_iso(),
        # 实测与标称是**两个独立字段**，语义不同、互不覆盖。
        "sample_rate_hz": measured,
        "nominal_rate_hz": nominal,
        "tolerance_ratio": float(tolerance_ratio),
        "consistency": compare_rate(
            measured, nominal, tolerance_ratio=tolerance_ratio),
    }
    run_context.meta["metrics"] = out

    # **兼容镜像**（设计 A.6.2）：旧字段继续存在，值由本模块统一写入，
    # 避免一次性改动全部读取方；下一大版本再评估移除。
    _sync_legacy_mirror(run_context, out, table)

    return out


def _sync_legacy_mirror(
    run_context: RunContext, metrics: dict[str, Any], table: pd.DataFrame | None,
) -> None:
    """把物化结果镜像到旧字段 ``meta.streams[*].measured_rate``（只读兼容）。"""
    measured = metrics.get("sample_rate_hz")
    if not measured:
        return
    for s in run_context.meta.get("streams", []) or []:
        if not isinstance(s, dict):
            continue
        # 只为"承载该时间列的流"写镜像；无法确定的流跳过（不硬塞）。
        s.setdefault("measured_rate", {
            "sample_rate_hz": measured["value"],
            "provenance": measured["provenance"],
            "source": measured["source"],
            "note": "由 metrics_store 统一物化（镜像字段，权威值见 meta.metrics）",
        })


def get_measured_rate_hz(run_context: RunContext) -> dict[str, Any] | None:
    """读**实测**采样率（所有工具的唯一读取入口）。

    Returns:
        ``{"value": float, "provenance": "measured", ...}`` 或 None。
    """
    m = run_context.meta.get("metrics")
    if not isinstance(m, dict):
        return None
    got = m.get("sample_rate_hz")
    return got if isinstance(got, dict) else None


def get_nominal_rate_hz(run_context: RunContext) -> dict[str, Any] | None:
    """读**标称**采样率（仅供一致性比较，**不用于计算**）。

    Returns:
        ``{"value": float, "provenance": "declared", ...}`` 或 None。
    """
    m = run_context.meta.get("metrics")
    if not isinstance(m, dict):
        return None
    got = m.get("nominal_rate_hz")
    return got if isinstance(got, dict) else None


def compare_rate(
    measured: dict[str, Any] | None,
    nominal: dict[str, Any] | None,
    *,
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
) -> dict[str, Any]:
    """用**容差**比较实测与标称是否一致（不要求相等）。

    设计要点（回答用户疑问："120.007 vs 120 实际可以认为二者一致"）：
    二者是「实际表现」与「设计目标」的关系，偏差 0.0058% 属正常。
    判定式 ``|measured - nominal| / nominal <= tolerance_ratio``。

    Args:
        measured: 实测值 dict（含 ``value``）。
        nominal: 标称值 dict（含 ``value``）。
        tolerance_ratio: 容差比例（默认 0.05）。

    Returns:
        dict，含 ``consistent``（True/False/**None**）、``deviation_ratio``、
        ``tolerance_ratio``、``explain``。**任一侧缺失时 consistent=None**
        ——如实说"无法比较"，不猜。
    """
    if not measured or not nominal:
        missing = []
        if not measured:
            missing.append("实测值")
        if not nominal:
            missing.append("标称值")
        return {
            "consistent": None,
            "deviation_ratio": None,
            "tolerance_ratio": tolerance_ratio,
            "explain": (
                f"缺少{'与'.join(missing)}，**无法比较**"
                "（不做推测；可确认数据是否声明了采样率）"
            ),
        }

    m = float(measured["value"])
    n = float(nominal["value"])
    if n <= 0:
        return {
            "consistent": None,
            "deviation_ratio": None,
            "tolerance_ratio": tolerance_ratio,
            "explain": "标称值非正，无法计算偏差",
        }

    dev = abs(m - n) / n
    ok = dev <= tolerance_ratio
    return {
        "consistent": ok,
        "deviation_ratio": round(dev, 8),
        "deviation_percent": round(dev * 100, 4),
        "tolerance_ratio": tolerance_ratio,
        "measured_value": m,
        "nominal_value": n,
        "explain": (
            f"实测 {m} Hz，声明 {n} Hz，偏差 {dev * 100:.4f}%"
            f"（容差 {tolerance_ratio * 100:.2f}%）→ "
            + ("一致" if ok else "**不一致**，建议核对设备配置")
        ),
    }


def describe(run_context: RunContext) -> dict[str, Any]:
    """给出采样率的可读描述（供报告与模型转述，**同时给两个数字**）。

    呈现纪律（设计 A.5.5）：即使判为一致也要给出实测与标称两个数值，
    否则用户无法判断偏差的方向与量级。

    Returns:
        dict，含 ``measured_hz`` / ``nominal_hz`` / ``consistent`` /
        ``text``（一句可直接引用的话）。
    """
    m = get_measured_rate_hz(run_context)
    n = get_nominal_rate_hz(run_context)
    cons = run_context.meta.get("metrics", {}).get("consistency") if isinstance(
        run_context.meta.get("metrics"), dict) else None
    if not isinstance(cons, dict):
        cons = compare_rate(m, n)

    m_v = m["value"] if m else None
    n_v = n["value"] if n else None

    if m_v is not None and n_v is not None:
        text = (
            f"实测 {m_v} Hz（来源：{m.get('source')}），"
            f"声明 {n_v} Hz（来源：{n.get('source')}），"
            f"偏差 {cons.get('deviation_percent')}% → "
            + ("一致" if cons.get("consistent") else "不一致")
        )
    elif m_v is not None:
        text = f"实测 {m_v} Hz；无声明值可供对照（{cons.get('explain')}）"
    elif n_v is not None:
        text = f"声明 {n_v} Hz；未测出实测值（{cons.get('explain')}）"
    else:
        text = f"采样率未知：{cons.get('explain')}"

    return {
        "measured_hz": m_v,
        "nominal_hz": n_v,
        "consistent": cons.get("consistent"),
        "deviation_percent": cons.get("deviation_percent"),
        "tolerance_ratio": cons.get("tolerance_ratio"),
        "text": text,
    }
