"""主表选择与装载完整性（本修复第二部分）的单元测试。

覆盖：主表选择显式策略（行数×列数最大 > 字母序；含状态/动作列优先）、
装载完整性声明（全量装载 rows_total==rows_loaded；超阈值截断时两个数字同时
存在且 user_message 明确提示）、以及"大表胜过先抽到的小表"场景。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl


def _big_csv(path: Path, n_rows: int, n_cols: int, header_prefix: str = "c") -> None:
    cols = [f"{header_prefix}{i}" for i in range(n_cols)]
    data = {c: range(n_rows) for c in cols}
    pd.DataFrame(data).to_csv(path, index=False)


def test_main_table_picks_larger_table_over_earlier_small(tmp_path: Path) -> None:
    """主表策略"行数×列数最大"应胜过"先抽到的小表"（两者都不含动作列）。"""
    root = tmp_path / "ds"
    root.mkdir()
    # 小表在前（字母序 small < big），不含动作列。
    (root / "small.csv").write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    # 大表在后：行数×列数明显更大，不含动作列。
    _big_csv(root / "big.csv", n_rows=100, n_cols=20)

    ctx = RunContext()
    result = load_dataset_impl(ctx, str(root))

    assert result["success"] is True
    assert result["main_table"]["name"] == "big.csv"
    # 依据应说明"行数×列数最大"。
    assert "行数×列数最大" in result["main_table_selection"]["reason"]
    # 落选候选应含 small.csv。
    names = [c["name"] for c in result["main_table_selection"]["candidates"]]
    assert "small.csv" in names


def test_main_table_prefers_actions_table(tmp_path: Path) -> None:
    """含状态/动作列的表优先于仅按规模更大的表。"""
    root = tmp_path / "ds"
    root.mkdir()
    # 大表无动作列。
    _big_csv(root / "big.csv", n_rows=200, n_cols=30)
    # 小表含动作列（qpos）。
    (root / "ctrl.csv").write_text(
        "qpos1,qpos2,success\n1,2,1\n3,4,0\n5,6,1\n", encoding="utf-8"
    )

    ctx = RunContext()
    result = load_dataset_impl(ctx, str(root))

    assert result["main_table"]["name"] == "ctrl.csv"
    assert "含状态/动作列" in result["main_table_selection"]["reason"]


def test_small_file_loaded_in_full_no_truncation(tmp_path: Path) -> None:
    """小文件默认全量装载：rows_total == rows_loaded，无截断声明。"""
    root = tmp_path / "ds"
    root.mkdir()
    _big_csv(root / "data.csv", n_rows=50, n_cols=8)

    ctx = RunContext()
    result = load_dataset_impl(ctx, str(root))

    mt = result["main_table"]
    assert mt["rows_total"] == 50
    assert mt["rows_loaded"] == 50
    assert mt["truncated"] is False
    # user_message 不应含截断提示。
    assert "仅装载前" not in result["user_message"]


def test_large_file_truncation_declares_both_numbers(tmp_path: Path) -> None:
    """超阈值文件截断：rows_total 与 rows_loaded 同时存在，user_message 明确提示。"""
    root = tmp_path / "ds"
    root.mkdir()
    _big_csv(root / "huge.csv", n_rows=15, n_cols=5)

    ctx = RunContext()
    # 设低阈值，触发截断。
    ctx.max_rows_in_context = 10
    result = load_dataset_impl(ctx, str(root))

    mt = result["main_table"]
    assert mt["rows_total"] == 15
    assert mt["rows_loaded"] == 10
    assert mt["truncated"] is True
    # 两个数字同时在声明中可见。
    assert "仅装载前 10 行（共 15 行）" in result["user_message"]
    # context.meta 也记录，供后续统计继承。
    assert ctx.meta["main_table"]["rows_total"] == 15
    assert ctx.meta["main_table"]["rows_loaded"] == 10
    assert ctx.meta["main_table"]["truncated"] is True


def test_compute_stats_inherits_truncation_scope(tmp_path: Path) -> None:
    """基于截断数据的统计，返回继承 data_scope 声明。"""
    from app.tools.compute_stats import compute_stats_impl

    root = tmp_path / "ds"
    root.mkdir()
    # 含动作列（success），触发 compute_stats 适用，且超阈值截断。
    (root / "ep.csv").write_text(
        "episode,success\n" + "\n".join(f"{i},{i % 2}" for i in range(15)) + "\n",
        encoding="utf-8",
    )

    ctx = RunContext()
    ctx.max_rows_in_context = 10
    load_dataset_impl(ctx, str(root))

    stats_result = compute_stats_impl(ctx, "task_level")
    assert stats_result["success"] is True
    assert stats_result["data_scope"] is not None
    assert stats_result["data_scope"]["truncated"] is True
    assert stats_result["data_scope"]["rows_total"] == 15
    assert stats_result["data_scope"]["rows_loaded"] == 10
    assert "仅前 10 行" in stats_result["data_scope"]["note"]
