"""inspect_streams 未分类引导 + burst 有效速率的单元测试。

覆盖（用户报告"概况里语义不明确"的修复）：
  - unknown 占比高 → unclassified_hint + user_message 追加 propose 引导；
  - 已确认为主 → 不出引导（防噪声）；
  - burst 流（无候选可换，如 tf）→ sample_rate_hz=None + effective_rate_hz
    （常规口径失真值不再作为主值暴露）。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.tools.inspect_streams import _measure_rate_from_file, inspect_streams_impl
from app.tools.load_dataset import load_dataset_impl

T0_US = 1_787_294_445_600_000


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _batchy_rows(n: int = 100, batch: int = 10) -> list[dict]:
    return [{
        "mcap_log_time_ns": T0_US * 1000 + (i // batch) * 50_000_000 + (i % batch) * 1_000,
        "data": {"transforms": [{"parent": "a", "child": "b"}]},
    } for i in range(n)]


def test_unclassified_hint_when_mostly_unknown(tmp_path: Path) -> None:
    """unknown 占比高 → hint 出现且 user_message 含 propose 引导。"""
    d = tmp_path / "ds"
    d.mkdir()
    for i in range(6):
        _write_jsonl(d / f"t{i}.jsonl", [
            {"mcap_log_time_ns": T0_US * 1000 + j * 1_000_000,
             "data": {"v": float(j)}}
            for j in range(50)
        ])
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True

    ins = inspect_streams_impl(ctx)
    hint = ins.get("unclassified_hint")
    assert hint is not None
    assert "propose_stream_semantics" in hint
    assert "未分类" in hint
    assert "propose_stream_semantics" in ins["user_message"]


def test_no_hint_when_all_confirmed(tmp_path: Path) -> None:
    """全部已确认（label_source=user_confirmed）→ 不出引导。"""
    d = tmp_path / "ds"
    d.mkdir()
    for i in range(6):
        _write_jsonl(d / f"t{i}.jsonl", [
            {"mcap_log_time_ns": T0_US * 1000 + j * 1_000_000,
             "data": {"v": float(j)}}
            for j in range(50)
        ])
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    # 模拟全部确认。
    for s in ctx.meta["streams"]:
        s["label_source"] = "user_confirmed"
        s["semantic_label"] = f"流 {Path(s['path']).name}"

    ins = inspect_streams_impl(ctx)
    assert ins.get("unclassified_hint") is None
    assert "propose" not in ins["user_message"]


def test_burst_without_alternate_nulls_main_rate(tmp_path: Path) -> None:
    """burst 流无嵌套候选（tf 型）→ 主值置空 + effective_rate_hz（失真值不再暴露）。"""
    p = tmp_path / "tf.jsonl"
    _write_jsonl(p, _batchy_rows(300))
    mr = _measure_rate_from_file(str(p), "jsonl", [])
    assert mr["present"] is True
    assert mr["stream_shape"] == "burst"
    assert mr["sample_rate_hz"] is None, (
        f"失真主值应置空，实际 {mr['sample_rate_hz']}"
    )
    assert mr.get("effective_rate_hz", 0) > 0
    assert "不适用" in mr.get("sample_rate_note", "")
