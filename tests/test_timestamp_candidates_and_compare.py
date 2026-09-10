"""缺口 3 的测试：无词表命中列的时间候选发现 + 数据集对比。

3a 时间候选：原实现只在"词表完全未命中"时做内容指纹回退——若表中已有 1 个
   命中词表的时间列，其它自研命名的真时间列（如 aligned_ts_epoch）会被漏掉。
3b 数据集对比：同任务多次录制/两批次的并排对比，只读、不切换当前数据集。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools._sniffing import find_timestamp_columns
from app.tools.compare_datasets import compare_datasets_impl

T0_MS = 1_788_426_767_625


# --- 3a. 时间列候选发现 ------------------------------------------------------


def test_fingerprint_finds_unnamed_time_column_alongside_named() -> None:
    """表里既有词表命中的 timestamp，又有自研命名的时间列 → 两者都成候选。"""
    n = 50
    df = pd.DataFrame({
        "timestamp": T0_MS + np.arange(n) * 10.0,          # 词表命中
        "aligned_ts_epoch": T0_MS + np.arange(n) * 10.0 - 0.5,  # 自研命名（未命中词表）
        "value": np.random.default_rng(0).normal(0, 1, n),  # 非时间列
    })
    info = find_timestamp_columns(list(df.columns), df)
    names = {c["name"] for c in info["all_candidates"]}
    assert "timestamp" in names
    assert "aligned_ts_epoch" in names, "自研命名的时间列应被指纹发现"
    # 词表命中的列仍为主列（指纹候选不抢占）。
    assert info["main"] == "timestamp"
    sources = {c["name"]: c["source"] for c in info["all_candidates"]}
    assert sources["timestamp"] == "dictionary"
    assert sources["aligned_ts_epoch"] == "fingerprint"


def test_non_time_columns_not_misidentified() -> None:
    """非单调/非时间量级的数值列不被误认为时间列。"""
    n = 50
    df = pd.DataFrame({
        "timestamp": T0_MS + np.arange(n) * 10.0,
        "noise": np.random.default_rng(1).normal(0, 1, n),  # 非单调
        "counter": np.arange(n)[::-1],                       # 单调递减
    })
    info = find_timestamp_columns(list(df.columns), df)
    names = {c["name"] for c in info["all_candidates"]}
    assert "noise" not in names
    assert "counter" not in names


def test_no_sample_only_dictionary() -> None:
    """无样本时只做词表匹配（行为不变）。"""
    info = find_timestamp_columns(["timestamp", "aligned_ts_epoch"])
    names = {c["name"] for c in info["all_candidates"]}
    assert names == {"timestamp"}


# --- 3b. 数据集对比 ----------------------------------------------------------


def _make_ds(root: Path, n_rows: int, gap: bool = False) -> Path:
    """造一个含 jsonl 流的数据集目录（可选制造缺口）。"""
    d = root
    d.mkdir(parents=True, exist_ok=True)
    ts = T0_MS + np.arange(n_rows) * 10.0
    if gap:
        ts[n_rows // 2:] += 5000.0  # 中段缺口 5s
    rows = [{"timestamp": float(t), "v": float(i)} for i, t in enumerate(ts)]
    with (d / "stream.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def test_compare_two_datasets(tmp_path: Path) -> None:
    """对比两个数据集：同名流行数/时长/采样率并排。"""
    a = _make_ds(tmp_path / "A", 100)
    b = _make_ds(tmp_path / "B", 100)
    ctx = RunContext(output_dir=str(tmp_path))
    r = compare_datasets_impl(ctx, str(a), str(b))
    assert r["success"] is True, r.get("user_message")
    assert r["summary"]["n_matched"] >= 1
    row = next(m for m in r["matched"] if m["stream"] == "stream.jsonl")
    assert row["same_n"] is True and row["a_n"] == row["b_n"] == 100


def test_compare_detects_row_count_difference(tmp_path: Path) -> None:
    """行数不同 → same_n=False 且计入 diff_rows。"""
    a = _make_ds(tmp_path / "A", 100)
    b = _make_ds(tmp_path / "B", 80)
    ctx = RunContext(output_dir=str(tmp_path))
    r = compare_datasets_impl(ctx, str(a), str(b))
    row = next(m for m in r["matched"] if m["stream"] == "stream.jsonl")
    assert row["same_n"] is False
    assert r["summary"]["n_diff_rows"] >= 1
    assert "行数不同" in r["user_message"]


def test_compare_detects_span_difference(tmp_path: Path) -> None:
    """B 有 5s 缺口 → 时长差被检出。"""
    a = _make_ds(tmp_path / "A", 100)
    b = _make_ds(tmp_path / "B", 100)
    # B 的中段缺口（重写）
    ts = T0_MS + np.arange(100) * 10.0
    ts[50:] += 5000.0
    with (b / "stream.jsonl").open("w", encoding="utf-8") as f:
        for i, t in enumerate(ts):
            f.write(json.dumps({"timestamp": float(t), "v": float(i)}) + "\n")
    ctx = RunContext(output_dir=str(tmp_path))
    r = compare_datasets_impl(ctx, str(a), str(b))
    row = next(m for m in r["matched"] if m["stream"] == "stream.jsonl")
    assert row["same_span"] is False
    assert r["summary"]["n_diff_span"] >= 1


def test_compare_only_in_one_side(tmp_path: Path) -> None:
    """单侧独有的流被列出。"""
    a = _make_ds(tmp_path / "A", 50)
    b = _make_ds(tmp_path / "B", 50)
    (b / "extra.jsonl").write_text(
        json.dumps({"timestamp": float(T0_MS), "v": 1.0}) + "\n", encoding="utf-8")
    ctx = RunContext(output_dir=str(tmp_path))
    r = compare_datasets_impl(ctx, str(a), str(b))
    assert "extra.jsonl" in r["only_in_b"]
    assert r["summary"]["n_only_b"] >= 1


def test_compare_path_not_found_structured(tmp_path: Path) -> None:
    """路径不存在 → 结构化错误。"""
    a = _make_ds(tmp_path / "A", 10)
    ctx = RunContext(output_dir=str(tmp_path))
    r = compare_datasets_impl(ctx, str(a), str(tmp_path / "ghost"))
    assert r["success"] is False and r["error"] == "path_b_not_found"


def test_compare_does_not_switch_current_dataset(tmp_path: Path) -> None:
    """**只读对比**：不切换 context 的当前数据集。"""
    a = _make_ds(tmp_path / "A", 10)
    b = _make_ds(tmp_path / "B", 10)
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "existing_dataset"
    ctx.meta = {"streams": []}
    compare_datasets_impl(ctx, str(a), str(b))
    assert ctx.dataset_id == "existing_dataset"
    assert ctx.meta == {"streams": []}
