"""嵌套标定文件读取 + 无 findings 时报告生成（2026-09-20 用户实测）。

背景（UMI 数据集验收测试）：用户要求核对"内参/FOV/外参"，agent 两次尝试
（单文件加载、目录加载）后回答"工具读不出来，请把 JSON 原文贴给我"——用户
只能手工贴 28KB 文本，agent 再人工核算。这暴露三个问题：

1. **嵌套读取被列数上限挤掉**：`calibration.json` 只有 3 个顶层字段
   （sensors_list / intrinsic / extrinsic），但 `expand_envelope` 的
   `max_cols=64` 被 `sensors_list` 一个字段独占（它展开出 40+ 列），
   内参与外参**完全没有展开**——工具"有能力但读不到"。
2. **报告生成被 no_findings 拦住**：`generate_report` 只拼装
   `context.findings`，而 load_dataset / profile_data / inspect_streams /
   check_temporal_sync **都不写 findings**，于是用户做完分析要报告时得到
   "当前会话尚无分析结果"。
3. **单文件报告显示"文件数 0 / 总大小 0.0 B"**：概况只从 `meta["streams"]` 取，
   单文件加载没有流登记表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools._data_access import expand_envelope, expand_with_focus

import app.tools.generate_report  # noqa: F401
import app.tools.load_dataset  # noqa: F401
import app.tools.profile_data  # noqa: F401

_gen = sys.modules["app.tools.generate_report"]
_ld = sys.modules["app.tools.load_dataset"]
_pd = sys.modules["app.tools.profile_data"]


def _calibration_like() -> pd.DataFrame:
    """构造与真实标定文件同构的 3 行表：big 字段会独占配额。

    ``sensors_list`` 故意做成"展开后很长"——真实文件里它展开出 40+ 列，
    足以吃满默认 max_cols=64，使排在其后的 ``intrinsic`` / ``extrinsic``
    完全轮不到。故这里用 20 个相机 × 4 字段 = 80+ 列来复现同一现象。
    """
    big = {
        "project": "RD001",
        "topic list": {f"/cam_{i}": {"name": f"cam {i}", "link": f"l{i}",
                                     "sn": str(i), "robot-id": "X1"}
                       for i in range(20)},
    }
    intrinsic = {
        "project": "RD001",
        "topic list": {
            f"/cam_{i}": {
                "camera_matrix": {"rows": 3, "cols": 3,
                                  "data": [600.0 + i, 0, 800.0,
                                           0, 600.0 + i, 650.0, 0, 0, 1]},
                "dist_coeffs": {"rows": 4, "cols": 1,
                                "data": [0.05, 0.01, -0.003, 0.001]},
            }
            for i in range(3)
        },
    }
    extrinsic = {
        "project": "RD001",
        "topic list": {
            f"/cam_{i}": {"T": {"rows": 4, "cols": 4,
                                "data": [1, 0, 0, float(i), 0, 1, 0, 0,
                                         0, 0, 1, 0, 0, 0, 0, 1]}}
            for i in range(3)
        },
    }
    return pd.DataFrame({"sensors_list": [big],
                         "intrinsic": [intrinsic],
                         "extrinsic": [extrinsic]})


# --- 1. focus 配额分配（真实缺陷回归）-------------------------------------


def test_without_focus_intrinsic_is_squeezed_out() -> None:
    """**缺陷复现**：不指定 focus 时，靠前的字段独占配额，内参读不到。

    这不是要修的 bug，而是"为什么需要 focus"的事实记录——锁定该行为，
    使将来若有人改了默认配额策略能立刻发现。
    """
    exp, _ = expand_envelope(_calibration_like())
    cols = list(exp.columns)
    assert not any("camera_matrix" in c for c in cols), (
        "默认配额下内参居然可见了——默认策略已变，请同步更新文档与测试"
    )


def test_focus_intrinsic_makes_it_visible() -> None:
    """**核心回归**：focus 指定后，内参能展开出来（含 fx）。"""
    exp, _ = expand_envelope(_calibration_like(), focus=["intrinsic"])
    cols = list(exp.columns)
    cm = [c for c in cols if "camera_matrix" in c]
    assert cm, f"focus=intrinsic 后仍读不到内参：{cols[:10]}"
    fx_cols = [c for c in cm if c.endswith("data.0")]
    assert fx_cols, "未展开到 camera_matrix.data.0（fx 所在列）"


def test_focus_both_intrinsic_and_extrinsic() -> None:
    """**关键回归**：focus 传两个字段时，**两个都要拿到**（不能只保第一个）。

    早期实现只改"展开顺序"，导致 intrinsic 仍独占全部配额、extrinsic 一个
    列都没有——必须按字段**分配配额**而非仅排序。
    """
    exp, _ = expand_envelope(
        _calibration_like(), focus=["intrinsic", "extrinsic"])
    cols = list(exp.columns)
    assert any("camera_matrix" in c for c in cols), "内参未展开"
    assert any(c.startswith("extrinsic") and "T.data" in c for c in cols), (
        "外参未展开（配额被内参独占）"
    )


def test_max_cols_none_expands_everything() -> None:
    """``max_cols=None`` 表示不限制：字段数有限的配置类文件能全量展开。"""
    exp, note = expand_envelope(_calibration_like(), max_cols=None)
    cols = list(exp.columns)
    assert any("camera_matrix" in c for c in cols)
    assert any(c.startswith("extrinsic") for c in cols)
    assert note is None, f"不限列数时不应报截断：{note}"


def test_expand_with_focus_star_means_all() -> None:
    """``expand_with_focus`` 的 ``["*"]`` 语义 = 全量展开。"""
    exp, note = expand_with_focus(_calibration_like(), ["*"])
    cols = list(exp.columns)
    assert any("camera_matrix" in c for c in cols)
    assert any(c.startswith("extrinsic") for c in cols)
    assert note is None


def test_expand_with_focus_none_keeps_legacy_behavior() -> None:
    """**零回归**：``focus_fields=None`` 时行为与改动前一致（默认配额）。"""
    legacy, note_legacy = expand_envelope(_calibration_like())
    vianew, note_new = expand_with_focus(_calibration_like(), None)
    assert list(legacy.columns) == list(vianew.columns)
    assert note_legacy == note_new


def test_expand_with_focus_empty_list_legacy_behavior() -> None:
    """空列表同样回落到默认行为。"""
    a, _ = expand_with_focus(_calibration_like(), [])
    b, _ = expand_envelope(_calibration_like())
    assert list(a.columns) == list(b.columns)


def test_truncation_note_is_interpolated() -> None:
    """**真实缺陷回归**：截断说明里的 max_cols 必须被插值。

    此前 note 用普通字符串书写（缺 f 前缀），用户看到字面量
    ``max_cols={max_cols}``，无法得知实际上限。
    """
    df = _calibration_like()
    _, note = expand_envelope(df, max_cols=8)
    if note:
        assert "{max_cols}" not in note, f"note 未插值：{note}"
        assert "8" in note


def test_no_fragmentation_warning() -> None:
    """**性能回归**：展开大量列不得触发 pandas 碎片化告警。

    此前逐列 ``out[name] = vals`` 会在数百列时触发
    ``PerformanceWarning: DataFrame is highly fragmented``（实测 749 列）。
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exp, _ = expand_envelope(_calibration_like(), max_cols=None)
    frag = [w for w in caught if "fragmented" in str(w.message).lower()]
    assert not frag, f"出现碎片化告警：{frag[:2]}"
    assert exp.shape[1] > 20


# --- 2. 报告生成：无 findings 时仍可用 -------------------------------------


def test_report_generated_without_findings(tmp_path: Path) -> None:
    """**核心回归**：有数据集但无 findings 时，报告仍能生成。

    真实事故：用户跑完 load_dataset + profile_data + inspect_streams +
    check_temporal_sync（都不写 findings），要求生成 md 报告却得到
    "当前会话尚无分析结果"，只能自己手工整理。
    """
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="demo")
    ctx.meta = {"source": str(tmp_path / "x.json"), "format": "json",
                "n_rows": 3, "n_cols": 3, "capabilities": {}}
    ctx.df = pd.DataFrame({"a": [1]})
    r = _gen.generate_report_impl(ctx)
    assert r["success"] is True, r.get("user_message")
    p = Path(r["file_path"])
    assert p.exists() and p.stat().st_size > 100


def test_report_still_rejected_without_any_data(tmp_path: Path) -> None:
    """**边界**：既无数据集也无 findings → 仍拒绝（不能凭空造报告）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    r = _gen.generate_report_impl(ctx)
    assert r["success"] is False
    assert r["error"] == "no_data_loaded"


def test_report_single_file_shows_real_size(tmp_path: Path) -> None:
    """**真实缺陷回归**：单文件加载时报告须显示真实来源/大小，不得是 0 B。"""
    f = tmp_path / "cal.json"
    f.write_text(json.dumps({"intrinsic": [1], "extrinsic": [2]}),
                 encoding="utf-8")
    ctx = RunContext(output_dir=str(tmp_path))
    _ld.load_dataset_impl(ctx, str(f))
    r = _gen.generate_report_impl(ctx)
    assert r["success"] is True
    md = Path(r["file_path"]).read_text(encoding="utf-8")
    assert "0.0 B" not in md, "单文件报告仍显示 0.0 B"
    assert str(f.name) in md or str(f) in md


# --- 3. 工具层贯通：profile_data 的 focus_fields --------------------------


def test_profile_data_passes_focus_fields(tmp_path: Path) -> None:
    """``profile_data`` 的 ``focus_fields=["*"]`` 能读到内参列（端到端）。"""
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps({
        "sensors_list": [{"topic list": {f"/c{i}": {"sn": i} for i in range(12)}}],
        "intrinsic": [{"topic list": {"/c0": {"camera_matrix": {
            "rows": 3, "cols": 3, "data": [600.0, 0, 800.0, 0, 600.0, 650.0, 0, 0, 1]}}}}],
        "extrinsic": [{"topic list": {"/c0": {"T": {"rows": 4, "cols": 4,
            "data": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}}}}],
    }), encoding="utf-8")

    ctx = RunContext(output_dir=str(tmp_path))
    _ld.load_dataset_impl(ctx, str(f))
    r = _pd.profile_data_impl(ctx, expand=True, focus_fields=["*"])
    assert r["success"] is True
    names = [c.get("name") for c in (r.get("columns") or [])]
    assert any("camera_matrix" in str(n) for n in names), (
        f"未读到内参列：{names[:12]}"
    )


def test_profile_data_default_focus_is_legacy(tmp_path: Path) -> None:
    """**零回归**：不传 focus_fields 时与改动前一致（内参仍被挤掉）。"""
    f = tmp_path / "calibration.json"
    # sensors_list 需足够"胖"才能吃满默认配额（否则内参照样能挤进来，
    # 测试就测不到"零回归"这件事本身）。
    f.write_text(json.dumps({
        "sensors_list": [{"topic list": {
            f"/c{i}": {"name": f"c{i}", "link": f"l{i}", "sn": str(i),
                       "robot-id": "X1"} for i in range(20)}}],
        "intrinsic": [{"topic list": {"/c0": {"camera_matrix": {
            "rows": 3, "cols": 3, "data": [600.0, 0, 800.0, 0, 600.0, 650.0, 0, 0, 1]}}}}],
    }), encoding="utf-8")
    ctx = RunContext(output_dir=str(tmp_path))
    _ld.load_dataset_impl(ctx, str(f))
    r = _pd.profile_data_impl(ctx, expand=True)
    names = [c.get("name") for c in (r.get("columns") or [])]
    assert not any("camera_matrix" in str(n) for n in names), (
        "默认配额下内参可见——默认策略已变，请同步更新本测试与文档"
    )


def test_focus_field_not_present_is_safe(tmp_path: Path) -> None:
    """focus 指定了不存在的字段 → 不报错，回落到常规展开。"""
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"a": [{"x": 1}], "b": [{"y": 2}]}),
                 encoding="utf-8")
    ctx = RunContext(output_dir=str(tmp_path))
    _ld.load_dataset_impl(ctx, str(f))
    r = _pd.profile_data_impl(ctx, expand=True, focus_fields=["nope"])
    assert r["success"] is True
