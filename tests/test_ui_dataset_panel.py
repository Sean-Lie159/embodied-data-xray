"""数据集状态面板的单测（docs/数据集状态面板设计.md 第 4 节）。

覆盖：inspect_streams 回写 qc_state、dataset_summary 透出新字段、
UI 自算口径与工具口径一致（防"UI 与工具打架"）、
"未检查"与"无问题"措辞区分、截断提示。
"""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.agent.context import RunContext
from app.services.chat_service import ChatService


def _panel_script(tmp_path) -> str:
    script = tmp_path / "panel_test.py"
    script.write_text(
        "import streamlit as st\n"
        "from app.ui.components import render_dataset_overview\n"
        "render_dataset_overview(st.session_state['summary'])\n",
        encoding="utf-8",
    )
    return str(script)


def _run_panel(tmp_path, summary: dict) -> AppTest:
    at = AppTest.from_file(_panel_script(tmp_path), default_timeout=15)
    at.session_state["summary"] = summary
    at.run()
    return at


def _base_summary(**over) -> dict:
    s = {"dataset_id": "ds", "capabilities": {"has_imu": True, "imu_axes": 6,
                                              "has_video_streams": True},
         "streams": [], "guessed_type": "Ego", "video_fps_by_file": {},
         "main_table": {}, "qc_state": None, "qc": {}}
    s.update(over)
    return s


def test_no_dataset_shows_guidance(tmp_path) -> None:
    """未加载数据集：引导信息，不崩。"""
    at = _run_panel(tmp_path, {"dataset_id": None})
    assert not at.exception
    assert any("尚未加载数据集" in i.value for i in at.info)


def test_panel_renders_all_segments(tmp_path) -> None:
    """五段齐全时渲染不崩，且流清单含新增的语义标签/来源列。"""
    streams = [{"path": "a/imu.csv", "kind": "imu", "semantic_label": "IMU",
                "label_source": "user_confirmed", "measured_rate": None}]
    summary = _base_summary(
        streams=streams,
        qc_state={"n_streams": 1, "n_classified": 1, "n_unclassified": 0,
                  "unclassified_hint": None},
        main_table={"rows_total": 100, "rows_loaded": 100, "n_cols": 5},
    )
    at = _run_panel(tmp_path, summary)
    assert not at.exception
    markdowns = " ".join(m.value for m in at.markdown)
    assert "能力标签" in markdowns
    assert "语义确认进度" in markdowns
    assert "数据概况" in markdowns


def test_quality_not_checked_vs_no_warning(tmp_path) -> None:
    """措辞区分"未检查"与"无问题"（防误导为"没问题"）。"""
    # 未检查：qc 为空。
    at = _run_panel(tmp_path, _base_summary(qc={}))
    assert not at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "尚未执行" in caps

    # 已检查无告警。
    qc = {"check_temporal_sync": {"result": "pass", "detail": {
        "unit_warnings": [], "clock_conflicts": []}}}
    at2 = _run_panel(tmp_path, _base_summary(qc=qc))
    assert not at2.exception
    caps2 = " ".join(c.value for c in at2.caption)
    assert "无单位告警" in caps2


def test_unit_warnings_shown(tmp_path) -> None:
    """有单位告警时出现告警文案与展开清单。"""
    qc = {"check_temporal_sync": {"result": "warn", "detail": {
        "unit_warnings": ["流 a.csv：时间戳单位未知", "流 b.csv：时间戳单位未知"],
        "clock_conflicts": []}}}
    at = _run_panel(tmp_path, _base_summary(qc=qc))
    assert not at.exception
    caps = " ".join(c.value for c in at.caption)
    assert "2 条流时间戳单位未知" in caps


def test_main_table_truncation_warning(tmp_path) -> None:
    """主表截断时给出明确提示。"""
    summary = _base_summary(main_table={"rows_total": 1000, "rows_loaded": 100,
                                        "n_cols": 5})
    at = _run_panel(tmp_path, summary)
    assert not at.exception
    warns = " ".join(w.value for w in at.warning)
    assert "截断" in warns


def test_dataset_summary_exposes_new_fields() -> None:
    """dataset_summary 透出 main_table / qc_state / qc（供面板使用）。"""
    svc = ChatService.__new__(ChatService)  # 不构造 agent，仅测摘要方法
    ctx = RunContext(dataset_id="ds")
    ctx.meta["main_table"] = {"rows_total": 1, "rows_loaded": 1, "n_cols": 2}
    ctx.meta["qc_state"] = {"n_streams": 3, "n_classified": 1}
    ctx.meta["qc"] = {"check_temporal_sync": {"result": "pass"}}
    svc.context = ctx
    s = svc.dataset_summary()
    assert s["main_table"]["n_cols"] == 2
    assert s["qc_state"]["n_streams"] == 3
    assert "check_temporal_sync" in s["qc"]


def test_ui_count_matches_inspect_streams(tmp_path) -> None:
    """UI 自算的"已分类数"与 inspect_streams 的口径一致（防打架）。

    造一组混合流（user_confirmed / 已识别 / unknown），分别用
    inspect_streams 的 classified 统计与 UI 的 _is_classified 计数，
    断言两者相等。
    """
    from app.ui.components import _is_classified

    streams = [
        {"path": "a.csv", "kind": "imu", "semantic_label": "IMU",
         "label_source": "user_confirmed"},
        {"path": "b.csv", "kind": "force", "semantic_label": "力/力矩"},
        {"path": "c.csv", "kind": "unknown", "semantic_label": "未知"},
        {"path": "d.csv", "kind": None, "semantic_label": None},
    ]
    # inspect_streams 的判据（原文复制其 classified 计算式）。
    by_tool = sum(
        1 for s in streams
        if s.get("label_source") == "user_confirmed"
        or (s.get("semantic_label") or "").find("未知") < 0
        and s.get("kind") not in (None, "unknown")
    )
    by_ui = sum(1 for s in streams if _is_classified(s))
    assert by_ui == by_tool == 2


def test_qc_state_written_by_inspect_streams(tmp_path) -> None:
    """inspect_streams 调用后 meta 中留下 qc_state（供面板稳定读取）。"""
    from app.tools.inspect_streams import inspect_streams_impl

    ctx = RunContext(dataset_id="ds")
    ctx.meta["streams"] = [
        {"path": "a.csv", "kind": "imu", "semantic_label": "IMU"},
        {"path": "b.csv", "kind": "unknown", "semantic_label": "未知"},
    ]
    ctx.meta["capabilities"] = {"has_imu": True}
    inspect_streams_impl(ctx)
    qs = ctx.meta.get("qc_state")
    assert qs is not None
    assert qs["n_streams"] == 2
    assert qs["n_unclassified"] == 1
