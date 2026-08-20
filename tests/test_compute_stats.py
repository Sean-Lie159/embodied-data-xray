"""app/tools/compute_stats 工具的单元测试。

覆盖：任务级统计、episode 缺失、success 缺失、离群 episode 检出、无动作表
不适用、findings 累积、质检联动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.compute_stats import compute_stats, compute_stats_impl


def _ctx(df: pd.DataFrame | None, has_actions: bool = True, qc=None) -> RunContext:
    meta: dict = {"capabilities": {"has_actions": has_actions}}
    if qc is not None:
        meta["qc"] = qc
    return RunContext(dataset_id="demo", df=df, meta=meta)


def _task_df() -> pd.DataFrame:
    """5 个 episode，其中 episode 4 时长明显离群。"""
    rows = []
    durations = {0: 1.0, 1: 1.2, 2: 1.1, 3: 1.3, 4: 8.0}  # ep4 离群
    for ep, dur in durations.items():
        for j in range(3):
            rows.append({
                "episode": ep,
                "success": 1 if j == 2 else 0,  # 末帧 success
                "qpos1": 0.1 * ep + 0.01 * j,
                "qpos2": 0.2 * ep + 0.01 * j,
                "duration": dur + 0.01 * j,
            })
    return pd.DataFrame(rows)


def test_compute_stats_is_registered() -> None:
    assert isinstance(compute_stats, FunctionTool)
    assert compute_stats.name == "compute_stats"


def test_task_level_stats(tmp_path) -> None:
    ctx = _ctx(_task_df())
    r = compute_stats_impl(ctx)

    assert r["success"] is True
    assert r["dataset"] == "demo"
    # episode 分布。
    assert r["metrics"]["episode_distribution"]["n_episodes"] == 5
    # 成功率（每 episode 末帧 success，5 个 episode 全成功末帧=1）。
    assert r["metrics"]["success_rate"]["overall"] == 1.0
    assert "取每 episode 末帧" in r["metrics"]["success_rate"]["aggregation_rule"]
    # 关节活动范围。
    assert "qpos1" in r["metrics"]["joint_range_of_motion"]


def test_outlier_episode_detected(tmp_path) -> None:
    ctx = _ctx(_task_df())
    r = compute_stats_impl(ctx)

    outliers = r["metrics"]["outlier_episodes"]["outliers"]
    # episode 4 时长 8.0 明显离群，应被 IQR 检出。
    assert any(o["episode"] == 4 for o in outliers)
    assert r["metrics"]["outlier_episodes"]["method"] == "IQR"


def test_episode_missing_notes_single_episode(tmp_path) -> None:
    df = pd.DataFrame({"success": [0, 0, 1], "qpos1": [0.1, 0.2, 0.3]})
    ctx = _ctx(df)
    r = compute_stats_impl(ctx)

    assert r["success"] is True
    # 注明整段视为一个 episode。
    assert any("整段" in n or "一个 episode" in n for n in r["semantic_notes"])
    assert r["metrics"]["episode_distribution"]["n_episodes"] == 1


def test_success_missing_no_success_rate(tmp_path) -> None:
    df = pd.DataFrame({"episode": [0, 0, 1, 1], "qpos1": [0.1, 0.2, 0.3, 0.4]})
    ctx = _ctx(df)
    r = compute_stats_impl(ctx)

    assert r["success"] is True
    assert "success_rate" not in r["metrics"]


def test_no_actions_table_not_applicable(tmp_path) -> None:
    ctx = _ctx(None, has_actions=False)
    r = compute_stats_impl(ctx)

    assert r["success"] is False
    assert r["error"] == "not_applicable"
    assert "suggested_tools" in r


def test_findings_accumulated(tmp_path) -> None:
    ctx = _ctx(_task_df())
    compute_stats_impl(ctx)

    assert len(ctx.findings) == 1
    assert ctx.findings[0]["tool"] == "compute_stats"


def test_qc_summary_from_meta(tmp_path) -> None:
    qc = {
        "check_sensor_sanity": {"result": "fail"},
        "check_temporal_sync": {"result": "pass"},
    }
    ctx = _ctx(_task_df(), qc=qc)
    r = compute_stats_impl(ctx)

    assert r["qc_summary"]["check_sensor_sanity"]["result"] == "fail"
    assert r["qc_summary"]["check_temporal_sync"]["result"] == "pass"


def test_qc_not_done_notes(tmp_path) -> None:
    ctx = _ctx(_task_df())
    r = compute_stats_impl(ctx)

    assert r["qc_summary"]["status"] == "未质检"
    assert "建议先运行" in r["qc_summary"]["note"]


def test_generic_stats_column(tmp_path) -> None:
    ctx = _ctx(_task_df())
    r = compute_stats_impl(ctx, column="qpos1")

    assert "qpos1" in r["stats"]
    assert "count" in r["stats"]["qpos1"]
    assert "mean" in r["stats"]["qpos1"]
