"""数据集画像确认测试：类型确认通道 + 预期缺失。

设计依据：``docs/指标单一来源与数据集画像设计.md`` 第二部分（B-3.2）。

**背景（真实事故，2026-09-21）**：用户确认了"这是双手手套位姿追踪数据"
和"左右前臂本就不采集"，但
1. 报告仍显示 ``推测类型: unknown``（类型无确认通道，算完即固定）；
2. 质检仍把 14 个 forearm 列判为不可通过（预期缺失无处记录），
   用户每轮都要口头解释一遍。

**本文件最重要的两条守护**：
- ``test_confirmed_type_survives_reload``——确认后重载仍是该类型（跨会话）；
- ``test_expected_missing_excludes_from_nan_inf``——预期缺失不再误判 fail，
  且**如实标注**为预期缺失（不隐藏）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.config import get_settings
from app.tools import profile_store
from app.tools.confirm_dataset import confirm_dataset_profile_impl


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> str:
    d = tmp_path / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _em_payload(
    patterns: dict[str, list[str]] | dict[str, str],
    *,
    note: str = "",
    caps: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造确认 payload（复杂映射以 JSON 字符串传递——SDK 严格 schema 限制）。"""
    import json as _json

    p: dict[str, Any] = {"expected_missing_json": _json.dumps(patterns)}
    if note:
        p["expected_missing_note"] = note
    if caps is not None:
        p["capabilities_override_json"] = _json.dumps(caps)
    return p


def _glove_ctx(output_dir: str, *, missing_forearm: bool = True) -> RunContext:
    """构造模拟手套数据：含 forearm 列（默认全空）。"""
    n = 200
    data: dict[str, object] = {
        "episode_index": [0] * n,
        "timestamp_ns": (np.arange(n) * (1e9 / 120)).astype("int64"),
        "L_tip_thumb_pos_x": np.sin(np.arange(n) * 0.1),
        "L_tip_thumb_pos_y": np.cos(np.arange(n) * 0.1),
        "L_tip_index_pos_x": np.sin(np.arange(n) * 0.12),
    }
    for axis in ("x", "y", "z"):
        data[f"L_tip_forearm_pos_{axis}"] = (
            np.full(n, np.nan) if missing_forearm
            else np.sin(np.arange(n) * 0.08)
        )
        data[f"R_tip_forearm_pos_{axis}"] = (
            np.full(n, np.nan) if missing_forearm
            else np.sin(np.arange(n) * 0.08)
        )
    ctx = RunContext()
    ctx.df = pd.DataFrame(data)
    ctx.dataset_id = "20260921_151733"
    ctx.output_dir = output_dir
    ctx.meta = {"guessed_type": "unknown", "columns": list(data)}
    return ctx


# ---------------------------------------------------------------------------
# 类型确认通道（缺陷 B1-2）
# ---------------------------------------------------------------------------


def test_confirm_dataset_type_writes_profile(output_dir: str) -> None:
    """确认类型后写入画像（来源标 user_confirmed）。"""
    ctx = _glove_ctx(output_dir)
    res = confirm_dataset_profile_impl(ctx, {
        "dataset_type": "双手手套位姿追踪数据",
        "type_note": "Forsense-G7，左右前臂不采集",
    })
    assert res["success"] is True
    assert res["saved"] is True

    prof = profile_store.get_confirmed_dataset_type(output_dir, ctx.dataset_id)
    assert prof is not None
    assert prof["value"] == "双手手套位姿追踪数据"
    assert prof["source"] == profile_store.SOURCE_USER
    assert "前臂" in prof["note"]


def test_confirmed_type_survives_reload(output_dir: str) -> None:
    """**核心守护**：确认的类型跨会话/跨加载持续生效（不再回落 unknown）。"""
    ctx = _glove_ctx(output_dir)
    confirm_dataset_profile_impl(ctx, {"dataset_type": "手套位姿追踪数据"})

    # 模拟"新会话重新加载"：全新 context，只有相同的 output_dir/dataset_id。
    fresh = RunContext()
    fresh.dataset_id = ctx.dataset_id
    fresh.output_dir = output_dir
    got = profile_store.get_confirmed_dataset_type(output_dir, fresh.dataset_id)
    assert got is not None
    assert got["value"] == "手套位姿追踪数据"


def test_confirm_keeps_auto_detected_for_comparison(output_dir: str) -> None:
    """保留自动识别结果供对照（看得出"确认改了什么"）。"""
    ctx = _glove_ctx(output_dir)
    confirm_dataset_profile_impl(ctx, {"dataset_type": "手套位姿追踪数据"})
    prof = profile_store.get_confirmed_dataset_type(output_dir, ctx.dataset_id)
    assert prof["auto_detected"] == "unknown"


def test_confirm_applies_to_current_context_immediately(output_dir: str) -> None:
    """确认后**当前会话即时生效**（无需重新加载）。"""
    ctx = _glove_ctx(output_dir)
    confirm_dataset_profile_impl(ctx, {"dataset_type": "手套位姿追踪数据"})
    assert ctx.meta["guessed_type"] == "手套位姿追踪数据"
    assert ctx.meta["guessed_type_source"] == profile_store.SOURCE_USER


def test_report_shows_confirmed_type_not_unknown(output_dir: str) -> None:
    """报告须显示已确认类型（而非 unknown）——本次事故的直接验收点。"""
    from app.tools.generate_report import _build_dataset_overview

    ctx = _glove_ctx(output_dir)
    ctx.meta.update({"streams": [], "format": "csv", "n_rows": 200})
    confirm_dataset_profile_impl(ctx, {
        "dataset_type": "双手手套位姿追踪数据",
        "type_note": "Forsense-G7",
    })
    text = _build_dataset_overview(ctx)
    assert "双手手套位姿追踪数据" in text
    assert "用户确认" in text
    assert "**推测类型**: unknown" not in text
    assert "Forsense-G7" in text


def test_report_hints_confirm_when_unknown(output_dir: str) -> None:
    """unknown 时报告应**主动提示**可确认（而非静默呈现 unknown）。"""
    from app.tools.generate_report import _build_dataset_overview

    ctx = _glove_ctx(output_dir)
    ctx.meta.update({"streams": [], "format": "csv", "n_rows": 200})
    text = _build_dataset_overview(ctx)
    assert "推测类型" in text
    assert "如需固定类型" in text


def test_confirm_requires_dataset(output_dir: str) -> None:
    """未加载数据集时拒绝（画像按数据集隔离）。"""
    ctx = RunContext()
    ctx.output_dir = output_dir
    res = confirm_dataset_profile_impl(ctx, {"dataset_type": "x"})
    assert res["success"] is False
    assert res["error"] == "no_dataset"


def test_confirm_rejects_empty_payload(output_dir: str) -> None:
    """空 payload 拒绝（不写空画像）。"""
    ctx = _glove_ctx(output_dir)
    res = confirm_dataset_profile_impl(ctx, {})
    assert res["success"] is False
    assert res["error"] == "nothing_to_confirm"


# ---------------------------------------------------------------------------
# 预期缺失（用户"前臂本来就没有"的固化）
# ---------------------------------------------------------------------------


def test_confirm_expected_missing_writes_profile(output_dir: str) -> None:
    """写入预期缺失（含通配符模式）。"""
    ctx = _glove_ctx(output_dir)
    res = confirm_dataset_profile_impl(ctx, _em_payload(
        {"tips_trajectory.csv": ["*forearm_pos_*"]},
        note="手套无前臂传感器",
    ))
    assert res["success"] is True
    em = profile_store.get_expected_missing(output_dir, ctx.dataset_id)
    assert em is not None
    assert em["patterns"]["tips_trajectory.csv"] == ["*forearm_pos_*"]
    assert "前臂" in em["note"]


def test_confirm_rejects_invalid_json(output_dir: str) -> None:
    """非法 JSON 参数拒绝并说明原因（不静默忽略）。"""
    ctx = _glove_ctx(output_dir)
    res = confirm_dataset_profile_impl(ctx, {
        "expected_missing_json": "{ 不是合法 JSON",
    })
    assert res["success"] is False
    assert res["error"] == "invalid_json"
    assert "JSON" in res["user_message"]


def test_confirm_rejects_non_object_json(output_dir: str) -> None:
    """JSON 为数组时拒绝（要求对象）。"""
    ctx = _glove_ctx(output_dir)
    res = confirm_dataset_profile_impl(ctx, {
        "expected_missing_json": '["a"]',
    })
    assert res["success"] is False
    assert res["error"] == "invalid_json"


def test_expected_missing_normalizes_string_to_list(output_dir: str) -> None:
    """误传字符串时规范化为列表（防误用）。"""
    ctx = _glove_ctx(output_dir)
    confirm_dataset_profile_impl(ctx, _em_payload({"a.csv": "*forearm*"}))
    em = profile_store.get_expected_missing(output_dir, ctx.dataset_id)
    assert em["patterns"]["a.csv"] == ["*forearm*"]


def test_match_expected_missing_returns_hit_columns(output_dir: str) -> None:
    """匹配返回**实际命中的列名**（让用户看到排除了哪些，防误伤无感知）。"""
    cols = [
        "L_tip_forearm_pos_x", "L_tip_forearm_pos_y", "R_tip_forearm_pos_z",
        "L_tip_thumb_pos_x",
    ]
    hit = profile_store.match_expected_missing(
        {"tips_trajectory.csv": ["*forearm_pos_*"]},
        "tips_trajectory.csv", cols,
    )
    assert "L_tip_forearm_pos_x" in hit
    assert "R_tip_forearm_pos_z" in hit
    assert "L_tip_thumb_pos_x" not in hit, "非预期缺失的列不得被误排除"


def test_match_expected_missing_no_match_returns_empty() -> None:
    """无命中返回空列表。"""
    assert profile_store.match_expected_missing(
        {"a.csv": ["*forearm*"]}, "b.csv", ["x"] * 3) == []


def test_expected_missing_excludes_from_nan_inf(output_dir: str) -> None:
    """**核心守护**：已确认的预期缺失不再导致缺失值检查 fail。

    事故：tips_trajectory.csv 的 14 个 forearm 列 100% 空，每轮都报 fail，
    用户每轮都要解释"前臂本来就没有"。
    """
    from app.tools.check_dataset_quality import check_dataset_quality_impl

    ctx = _glove_ctx(output_dir)
    ctx.meta["source"] = str(Path(output_dir) / "tips_trajectory.csv")

    # 未确认前：forearm 全空 → fail。
    before = check_dataset_quality_impl(ctx, get_settings())
    assert before["gate"]["checks"]["nan_inf"]["result"] == "fail"

    # 确认预期缺失后：排除判定 → 不再 fail。
    confirm_dataset_profile_impl(ctx, _em_payload(
        {"tips_trajectory.csv": ["*forearm_pos_*"]},
        note="手套无前臂传感器",
    ))
    after = check_dataset_quality_impl(ctx, get_settings())
    nan = after["gate"]["checks"]["nan_inf"]
    assert nan["result"] == "pass", nan
    # **必须如实标注**为预期缺失（不隐藏、不静默）。
    assert nan["expected_missing_columns"], "预期缺失的列必须被列出"
    assert "L_tip_forearm_pos_x" in nan["expected_missing_columns"]
    assert "预期缺失" in nan["detail"]
    assert "前臂" in nan["detail"]


def test_expected_missing_does_not_hide_unexpected_gaps(output_dir: str) -> None:
    """**预期缺失不得掩盖意外缺失**（重要边界）。"""
    from app.tools.check_dataset_quality import check_dataset_quality_impl

    ctx = _glove_ctx(output_dir)
    # 额外让一个**非**预期列全空（意外缺失）。
    ctx.df["L_tip_thumb_quat_w"] = np.nan
    ctx.meta["source"] = str(Path(output_dir) / "tips_trajectory.csv")

    confirm_dataset_profile_impl(
        ctx, _em_payload({"tips_trajectory.csv": ["*forearm_pos_*"]}))
    res = check_dataset_quality_impl(ctx, get_settings())
    nan = res["gate"]["checks"]["nan_inf"]
    assert nan["result"] == "fail", "意外缺失必须仍然报 fail"
    assert "L_tip_thumb_quat_w" in nan["worst_column"] or (
        "L_tip_thumb_quat_w" in nan["per_column"])


def test_capabilities_override_applied(output_dir: str) -> None:
    """能力标签覆盖生效（人工纠正嗅探结果）。"""
    ctx = _glove_ctx(output_dir)
    ctx.meta["capabilities"] = {"has_pose": True, "has_force": False}
    confirm_dataset_profile_impl(ctx, _em_payload({}, caps={"has_force": True}))
    assert ctx.meta["capabilities"]["has_force"] is True
    got = profile_store.get_capabilities_override(output_dir, ctx.dataset_id)
    assert got == {"has_force": True}


def test_confirm_merges_without_clobbering_streams(output_dir: str) -> None:
    """确认画像不得覆盖已确认的**流**映射（合并而非整体替换）。"""
    profile_store.save_dataset_profile(
        output_dir, "ds1",
        stream_overrides={"a.csv": {"kind": "imu"}},
    )
    ctx = RunContext()
    ctx.dataset_id = "ds1"
    ctx.output_dir = output_dir
    confirm_dataset_profile_impl(ctx, {"dataset_type": "测试类型"})

    entry = profile_store.load_dataset_profile(output_dir, "ds1")
    assert entry["streams"]["a.csv"]["kind"] == "imu", "流映射被误清空"
    assert entry["dataset_type"]["value"] == "测试类型"
