"""profile_data 对象列全列向量分布统计的单元测试。

复现真实 PICO LeRobot 误判：向量串列起始段全零但整体有值，旧版 profile_data 的
sample_values 只取列首 → agent 误判"全 0 疑似未采集"。修复后 profile_data 提供
全列 zero/nonzero 分布（vector_stats）与样例值说明（sample_values_note）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl
from app.tools.profile_data import _vector_column_stats, profile_data_impl


def _vec_str(vals: list[float]) -> str:
    """构造 LeRobot 风格空格分隔向量串。"""
    return " ".join(f"{v}" for v in vals)


def test_vector_stats_partial_zero_column() -> None:
    """起始段全零但整体有值 → n_nonzero>0、zero_row_ratio<1（不误判全零）。"""
    series = pd.Series([_vec_str([0.0, 0.0, 0.0])] * 20 + [_vec_str([1.0, 2.0, 3.0])] * 30)
    stats = _vector_column_stats(series)
    assert stats is not None
    assert stats["n_parsed"] == 50
    assert stats["n_zero"] == 20
    assert stats["n_nonzero"] == 30
    assert stats["zero_row_ratio"] == 0.4
    assert stats["dims"] == 3


def test_vector_stats_all_zero_column() -> None:
    """全列全零 → zero_row_ratio=1（此时判全零是正确的）。"""
    series = pd.Series([_vec_str([0.0, 0.0])] * 20)
    stats = _vector_column_stats(series)
    assert stats is not None
    assert stats["n_nonzero"] == 0
    assert stats["zero_row_ratio"] == 1.0


def test_vector_stats_non_vector_column_returns_none() -> None:
    """普通字符串列（多数不可解析为向量）→ 不套向量统计（返回 None）。"""
    series = pd.Series(["hello world", "foo bar", "baz qux"] * 10)
    assert _vector_column_stats(series) is None


def test_profile_reports_vector_stats_not_all_zero(tmp_path: Path) -> None:
    """端到端：parquet 向量串列起始段全零但整体有值 → profile 带 vector_stats 且
    nonzero_rows>0，样例值说明明确"勿据样例外推全列"。"""
    root = tmp_path / "ds"
    root.mkdir()
    n = 50
    df = pd.DataFrame({
        "timestamp": [i * 0.04 for i in range(n)],
        "observation.left_hand": [_vec_str([0.0, 0.0, 0.0])] * 15 + [_vec_str([1.0, 2.0, 3.0])] * (n - 15),
    })
    df.to_parquet(root / "episode_000000.parquet")
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = profile_data_impl(ctx)
    assert r["success"] is True
    col = next(c for c in r["columns"] if c["name"] == "observation.left_hand")
    vstats = col.get("vector_stats")
    assert vstats is not None
    assert vstats["n_nonzero"] == n - 15
    assert vstats["n_zero"] == 15
    # 样例值说明：明确样例仅取列首、勿据此外推全列。
    assert "外推" in col.get("sample_values_note", "")
