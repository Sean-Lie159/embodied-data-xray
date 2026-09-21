"""metrics_store 测试：指标单一来源、实测/标称分离、容差判定。

设计依据：``docs/指标单一来源与数据集画像设计.md`` 第一部分（A）。

**本文件最重要的两条守护**：

1. ``test_only_metrics_store_writes_metrics``——AST 扫描强制"单一写入口"，
   不靠自律靠结构（这是本次架构缺陷的根治手段）；
2. ``test_rate_comparison_uses_tolerance``——120 vs 120.007 判为一致
   （用户明确提出的顾虑：二者实际可认为相同）。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools import metrics_store as ms


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def _ctx(df: pd.DataFrame | None = None, **meta) -> RunContext:
    ctx = RunContext()
    ctx.df = df
    ctx.dataset_id = "demo"
    ctx.meta = dict(meta)
    return ctx


def _ns_df(n: int = 100, hz: float = 120.0) -> pd.DataFrame:
    """纳秒时间戳表（模拟手套数据：timestamp_ns，120Hz）。"""
    step = 1e9 / hz
    return pd.DataFrame({
        "timestamp_ns": (np.arange(n) * step).astype("int64"),
        "a": np.sin(np.arange(n) * 0.1),
    })


# ---------------------------------------------------------------------------
# 单一写入口（结构性守护，最重要）
# ---------------------------------------------------------------------------


def test_only_metrics_store_writes_metrics() -> None:
    """**结构性守护**：除 metrics_store 外，任何模块不得写 meta["metrics"]。

    这是把"多来源写同一 key"从**靠自律**改为**靠结构**的关键：
    此前 measured_rate 有两个写入者、算法不同、谁后跑谁覆盖，导致同一数据集
    出现两个采样率。用 AST 扫描静态禁止其它写入点。
    """
    tools_dir = Path(__file__).resolve().parent.parent / "app" / "tools"
    offenders: list[str] = []

    for py in sorted(tools_dir.glob("*.py")):
        if py.name == "metrics_store.py":
            continue  # 唯一允许写的地方
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # 匹配 `meta["metrics"] = ...` / `context.meta["metrics"] = ...`
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                sl = target.slice
                # 只关心常量字符串下标：meta["metrics"]。
                # 注意 Python 3.9+ 下标直接是表达式节点（3.14 下 Slice 已
                # 不再是取值节点），故不做 Slice 分支——非 Constant 一律忽略。
                if isinstance(sl, ast.Constant) and sl.value == "metrics":
                    offenders.append(f"{py.name}:{node.lineno}")

    assert not offenders, (
        "以下位置直接写了 meta['metrics']，违反单一写入口约定——"
        "请改为调用 metrics_store.materialize()：\n" + "\n".join(offenders)
    )


def test_materialize_is_idempotent() -> None:
    """**幂等**：重复物化不重算、不产生第二个值。"""
    ctx = _ctx(_ns_df())
    first = ms.materialize(ctx)
    v1 = first["sample_rate_hz"]["value"]

    # 篡改数据后再次物化（未 force）——应返回既有结果，不重算。
    ctx.df = _ns_df(hz=50.0)
    second = ms.materialize(ctx)
    assert second["sample_rate_hz"]["value"] == v1

    # force=True 才重算。
    third = ms.materialize(ctx, force=True)
    assert third["sample_rate_hz"]["value"] == pytest.approx(50.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 实测值：统一算法链（修缺陷 A2-1 / A2-2）
# ---------------------------------------------------------------------------


def test_measure_rate_hz_uses_column_name_hint() -> None:
    """**必须利用列名后缀线索**（缺陷 A2-1 的核心）。

    `timestamp_ns` 的后缀 `_ns` 是最强单位证据；不传列名会退化为量级猜测。
    """
    df = _ns_df(n=100, hz=120.0)
    got = ms.measure_rate_hz(df["timestamp_ns"], col_name="timestamp_ns")
    assert got is not None
    assert got["value"] == pytest.approx(120.0, rel=1e-3)
    assert got["unit"] == "ns"
    assert got["provenance"] == ms.PROV_MEASURED


def test_measure_rate_hz_normalizes_unit_before_computing() -> None:
    """**先归一化再算**（缺陷 A2-2）：不得对原始数值取倒数。

    初版 `1.0 / median_step` 对纳秒间隔 8333100 会得出 1.2e-7。
    修正后无论时间戳是 ns/us/ms/s，都应得到同一个采样率。
    """
    n = 50
    hz = 100.0
    step_s = 1.0 / hz
    variants = {
        "ns": (np.arange(n) * step_s * 1e9).astype("int64"),
        "us": (np.arange(n) * step_s * 1e6).astype("int64"),
        "ms": (np.arange(n) * step_s * 1e3).astype("int64"),
        "s": np.arange(n) * step_s,
    }
    for unit, arr in variants.items():
        got = ms.measure_rate_hz(arr, col_name=f"timestamp_{unit}")
        assert got is not None, f"{unit} 未能测出采样率"
        assert got["value"] == pytest.approx(hz, rel=1e-2), (
            f"{unit} 单位下测得 {got['value']}，期望约 {hz}——单位归一化失效"
        )


def test_measure_rate_hz_uses_median_not_mean() -> None:
    """用中位数而非均值：对偶发丢帧更稳健（均值会被单个大间隔拉低）。"""
    n = 100
    step = 1e9 / 100.0
    ts = np.arange(n) * step
    ts[50] = ts[50] + step * 20  # 注入一个大缺口
    got = ms.measure_rate_hz(ts, col_name="timestamp_ns")
    assert got is not None
    # 中位数仍是 100Hz（均值会被拉低到 ~96）。
    assert got["value"] == pytest.approx(100.0, rel=1e-6)


def test_measure_rate_hz_returns_none_when_undeterminable() -> None:
    """无法判定时返回 None（诚实降级，不猜）。"""
    assert ms.measure_rate_hz([1.0, 1.0, 1.0], col_name="x") is None  # 全重复
    assert ms.measure_rate_hz([1.0, 2.0], col_name="x") is None      # 样本不足


def test_measure_rate_hz_evidence_is_explainable() -> None:
    """证据文字须可转述（说明依据与单位）。"""
    got = ms.measure_rate_hz(_ns_df()["timestamp_ns"], col_name="timestamp_ns")
    assert "中位数" in got["evidence"]
    assert "ns" in got["evidence"]


# ---------------------------------------------------------------------------
# 标称值来源链（修缺陷 A2-3）
# ---------------------------------------------------------------------------


def test_nominal_rate_from_config_file() -> None:
    """**配置文件里的 nominal_hz 必须被接入**（缺陷 A2-3 的核心）。

    事故：session.json 的 nominal_hz=120 此前只被登记为 config_keys
    （键名清单），从未流入采样率计算，导致 nominal_check 永久 skipped。
    """
    ctx = _ctx(
        _ns_df(),
        streams=[{
            "path": "/data/session.json",
            "status": "config",
            "config": {"nominal_hz": 120, "hand_mode": "both"},
        }],
    )
    got = ms.nominal_rate_from_config(ctx)
    assert got is not None
    assert got["value"] == 120.0
    assert got["provenance"] == ms.PROV_DECLARED
    assert "nominal_hz" in got["source"]


def test_nominal_rate_from_config_ignores_unrelated_keys() -> None:
    """配置里无采样率语义的键不得被误当采样率（保守取用）。"""
    ctx = _ctx(
        _ns_df(),
        streams=[{"path": "s.json", "status": "config",
                  "config": {"delay_ns": 25000000, "format_version": "3"}}],
    )
    assert ms.nominal_rate_from_config(ctx) is None


def test_nominal_rate_from_fps_column() -> None:
    """数据列声明的 fps 作为标称值。"""
    df = _ns_df()
    df["fps"] = 120.0
    got = ms.nominal_rate_from_columns(df)
    assert got is not None
    assert got["value"] == 120.0
    assert got["provenance"] == ms.PROV_DECLARED


def test_nominal_rate_from_lerobot_info(tmp_path: Path) -> None:
    """LeRobot meta/info.json 的 fps 作为标称值。"""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 30, "features": {}}), encoding="utf-8")
    ctx = _ctx(_ns_df(), source=str(root))
    got = ms.nominal_rate_from_lerobot(ctx)
    assert got is not None
    assert got["value"] == 30.0


def test_nominal_rate_none_when_nothing_declared() -> None:
    """无任何声明时返回 None（**不得用实测值顶替**）。"""
    ctx = _ctx(_ns_df())
    ms.materialize(ctx)
    assert ms.get_nominal_rate_hz(ctx) is None


# ---------------------------------------------------------------------------
# 实测与标称分离（否决"权威链"方案的核心保证）
# ---------------------------------------------------------------------------


def test_measured_and_nominal_are_separate_fields() -> None:
    """**实测与标称是两个独立字段，互不覆盖**。

    这是对"权威链"方案的否决性修正：若让实测覆盖标称，就永久丢失
    "声明 120 而实际只有 100"这种真正该报警的能力。
    """
    df = _ns_df(n=100, hz=100.0)
    df["fps"] = 120.0  # 声明 120，实际 100
    ctx = _ctx(df)
    m = ms.materialize(ctx)

    assert m["sample_rate_hz"]["value"] == pytest.approx(100.0, rel=1e-3)
    assert m["nominal_rate_hz"]["value"] == 120.0
    # 两者都在，且 provenance 不同。
    assert m["sample_rate_hz"]["provenance"] == ms.PROV_MEASURED
    assert m["nominal_rate_hz"]["provenance"] == ms.PROV_DECLARED


def test_real_declaration_mismatch_is_flagged() -> None:
    """声明 120 而实际 100 → 判为**不一致**（真正该报警的情况）。"""
    df = _ns_df(n=200, hz=100.0)
    df["fps"] = 120.0
    ctx = _ctx(df)
    m = ms.materialize(ctx)
    cons = m["consistency"]
    assert cons["consistent"] is False
    assert cons["deviation_percent"] == pytest.approx(16.67, abs=0.5)
    assert "不一致" in cons["explain"]


# ---------------------------------------------------------------------------
# 容差判定（用户明确提出的顾虑）
# ---------------------------------------------------------------------------


def test_rate_comparison_uses_tolerance() -> None:
    """**120 vs 120.007 必须判为一致**（容差判定，不要求相等）。

    用户原话："本次测试实测值是 120.007Hz，而配置项里是 120Hz，实际上可以
    认为二者一致"。二者是「实际表现」与「设计目标」的关系，偏差 0.006%
    属物理常态，判为一致。
    """
    cons = ms.compare_rate(
        {"value": 120.007}, {"value": 120.0}, tolerance_ratio=0.05)
    assert cons["consistent"] is True
    assert cons["deviation_percent"] == pytest.approx(0.0058, abs=0.001)
    assert "一致" in cons["explain"]
    # 必须同时给出两个数值（不能只报"一致"）。
    assert cons["measured_value"] == 120.007
    assert cons["nominal_value"] == 120.0


def test_rate_comparison_at_tolerance_boundary() -> None:
    """边界：偏差恰等于容差算一致，超出算不一致。"""
    exactly = ms.compare_rate({"value": 105.0}, {"value": 100.0},
                              tolerance_ratio=0.05)
    assert exactly["consistent"] is True
    beyond = ms.compare_rate({"value": 105.1}, {"value": 100.0},
                             tolerance_ratio=0.05)
    assert beyond["consistent"] is False


def test_rate_comparison_says_unknown_when_missing() -> None:
    """**任一侧缺失时如实返回"无法比较"**，不猜。"""
    only_measured = ms.compare_rate({"value": 120.0}, None)
    assert only_measured["consistent"] is None
    assert "无法比较" in only_measured["explain"]

    only_nominal = ms.compare_rate(None, {"value": 120.0})
    assert only_nominal["consistent"] is None

    neither = ms.compare_rate(None, None)
    assert neither["consistent"] is None


def test_compare_rate_handles_zero_nominal() -> None:
    """标称值为 0 时不除零，返回无法比较。"""
    cons = ms.compare_rate({"value": 100.0}, {"value": 0.0})
    assert cons["consistent"] is None


# ---------------------------------------------------------------------------
# 兼容镜像
# ---------------------------------------------------------------------------


def test_legacy_mirror_written_for_streams() -> None:
    """旧字段 measured_rate 仍被写入（兼容只读镜像）。"""
    ctx = _ctx(
        _ns_df(),
        streams=[{"path": "a.csv", "status": "ok"}],
    )
    ms.materialize(ctx)
    s = ctx.meta["streams"][0]
    assert "measured_rate" in s
    assert s["measured_rate"]["sample_rate_hz"] == pytest.approx(120.0, rel=1e-3)


# ---------------------------------------------------------------------------
# describe：可读描述
# ---------------------------------------------------------------------------


def test_describe_gives_both_numbers_and_verdict() -> None:
    """可读描述须同时给出实测与声明两个数值（呈现纪律）。"""
    df = _ns_df(n=100, hz=120.007)
    df["fps"] = 120.0
    ctx = _ctx(df)
    ms.materialize(ctx)
    d = ms.describe(ctx)
    assert d["measured_hz"] == pytest.approx(120.007, rel=1e-2)
    assert d["nominal_hz"] == 120.0
    assert d["consistent"] is True
    assert "实测" in d["text"] and "声明" in d["text"]


def test_describe_handles_missing_values() -> None:
    """缺失时如实说明（不显示 None 当数字）。"""
    ctx = _ctx(None)
    ms.materialize(ctx)
    d = ms.describe(ctx)
    assert d["measured_hz"] is None
    assert "未知" in d["text"] or "无声明值" in d["text"]


# ---------------------------------------------------------------------------
# 端到端：跨工具一致性（本次事故的核心验收）
# ---------------------------------------------------------------------------


def test_cross_tool_rate_consistency(tmp_path: Path) -> None:
    """**核心验收**：同一数据集在质检与时间同步两处得到的采样率必须一致。

    事故现场：报告流明细"未知"、时间同步检查 120.007Hz、质检 skip——
    同一份数据三种说法。修复后三者都读 metrics_store，必须数值一致。
    """
    from app.config import get_settings
    from app.tools.check_dataset_quality import check_dataset_quality_impl

    n = 300
    hz = 120.0
    step = 1e9 / hz
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp_ns": (np.arange(n) * step).astype("int64"),
        "fps": [hz] * n,  # 声明与实测一致
        "action_joint0": np.sin(np.arange(n) * 0.1),
        "action_joint1": np.cos(np.arange(n) * 0.1),
        "action_joint2": np.sin(np.arange(n) * 0.05),
    })
    ctx = _ctx(df)
    ctx.output_dir = str(tmp_path)

    # 物化（load_dataset 会在真实流程里做这步）。
    m = ms.materialize(ctx, tolerance_ratio=0.05)
    measured = m["sample_rate_hz"]["value"]
    assert measured == pytest.approx(hz, rel=1e-3)

    # 质检读到的实测值必须与之完全一致。
    res = check_dataset_quality_impl(ctx, get_settings())
    fps_chk = res["gate"]["checks"]["fps_valid"]
    assert fps_chk.get("measured_rate_hz") == pytest.approx(measured, abs=1e-4)
    # 声明与实测一致 → pass（不得因单位问题误判）。
    assert fps_chk["result"] == "pass", fps_chk
    assert fps_chk["deviation_ratio"] < 0.01
