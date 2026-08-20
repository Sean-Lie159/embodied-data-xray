"""compute_stats 独立表数据来源 bug 的回归用例。

回归：状态/动作在独立表（主表是 IMU、actions 在流登记表独立文件）时，
compute_stats 应读 actions 流全表，而不是误用主表（IMU，无 episode 列），
否则 n_episodes 会错误地算成 1。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools.compute_stats import compute_stats_impl


def _context_with_separate_actions(tmp_path: Path) -> RunContext:
    """构造主表是 IMU、状态/动作在独立 tasks.csv 的上下文。"""
    # 主表：IMU（无 episode/success/关节列）。
    t = np.arange(0, 1.0, 0.01)
    imu = pd.DataFrame({
        "timestamp": t,
        "accel_x": np.sin(t), "accel_y": np.cos(t), "accel_z": 9.8,
    })
    # 独立 actions 表：tasks.csv（含 episode/success/qpos/duration，5 个 episode）。
    rows = []
    durations = {0: 1.0, 1: 1.2, 2: 1.1, 3: 1.3, 4: 8.0}
    for ep, dur in durations.items():
        for j in range(3):
            rows.append({
                "episode": ep, "success": 1 if j == 2 else 0,
                "qpos1": 0.1 * ep + 0.01 * j, "duration": dur + 0.01 * j,
            })
    tasks = pd.DataFrame(rows)
    tasks_path = tmp_path / "tasks.csv"
    tasks.to_csv(tasks_path, index=False)

    meta = {
        "capabilities": {"has_actions": True, "action_channels": ["qpos1"]},
        "streams": [
            {"path": str(tasks_path), "format": "csv", "kind": "actions",
             "channels": ["qpos1"]},
        ],
    }
    return RunContext(dataset_id="demo", df=imu, meta=meta)


def test_actions_in_separate_table_not_mistaken_for_main(tmp_path: Path) -> None:
    """主表是 IMU、actions 在独立表时，应读 actions 流，n_episodes 正确。"""
    ctx = _context_with_separate_actions(tmp_path)
    r = compute_stats_impl(ctx)

    assert r["success"] is True
    # 应读到独立 actions 表的 5 个 episode（而非主表 IMU 的"整段视为一个 episode"）。
    assert r["metrics"]["episode_distribution"]["n_episodes"] == 5
    # 成功率应按独立表的 success 列计算。
    assert r["metrics"]["success_rate"]["overall"] == 1.0
    # 离群 episode（duration 8.0 的 episode 4）应被检出。
    outliers = r["metrics"]["outlier_episodes"]["outliers"]
    assert any(o["episode"] == 4 for o in outliers)
