"""tf 验证器展开列模式 + hint 阈值边界 + 自然语言翻译纪律的测试。

背景（用户实测）：tf.jsonl 确认请求验证 failed（"未发现 transforms 类列表
结构"）——根因是展开视图把 list[dict] 消化为扁平列，验证器找原始 list 永远
落空 → tf 流永远无法通过验证（hard bug）。hint 阈值恰半数确认时（13/26
< 0.5 为 False）漏报。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.agent.agent import SYSTEM_PROMPT
from app.tools.inspect_streams import inspect_streams_impl
from app.tools.load_dataset import load_dataset_impl
from app.tools.propose_semantics import propose_stream_semantics_impl

T0_US = 1_787_294_445_600_000


def _tf_rows(n: int = 30) -> list[dict]:
    """与真实 tf.jsonl 同构：data.transforms 为 list[dict]，含 frame_id 键。"""
    return [{
        "mcap_log_time_ns": (T0_US + i * 640_000) * 1000,
        "data": {"transforms": [{
            "timestamp_us": T0_US + i,
            "parent_frame_id": "waist",
            "child_frame_id": "r_wrist",
            "translation": [0.1, 0.15, 0.05],
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }]},
    } for i in range(n)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_ctx(tmp_path: Path, n_tf: int = 1, n_other: int = 0) -> RunContext:
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / "tf.jsonl", _tf_rows())
    for i in range(n_other):
        _write_jsonl(d / f"other_{i}.jsonl", [
            {"mcap_log_time_ns": T0_US * 1000 + j * 1_000_000,
             "data": {"v": float(j)}}
            for j in range(50)
        ])
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    return ctx


def test_tf_verification_strong_via_expanded_columns(tmp_path: Path) -> None:
    """tf 假设经展开列名特征验证 strong（回归：此前永远 failed）。"""
    ctx = _make_ctx(tmp_path)
    r = propose_stream_semantics_impl(
        ctx,
        [{"file": "tf.jsonl", "kind": "tf", "semantic_label": "坐标变换",
          "time_column": "mcap_log_time_ns"}],
    )
    item = r["results"][0]
    assert item["verified"] == "strong", item
    assert "frame_id" in item["evidence"] or "坐标变换特征键" in item["evidence"]


def test_tf_confirm_persists(tmp_path: Path) -> None:
    """tf 确认落盘后重载生效（完整闭环，此前因验证 bug 走不通）。"""
    ctx = _make_ctx(tmp_path)
    r = propose_stream_semantics_impl(
        ctx,
        [{"file": "tf.jsonl", "kind": "tf", "semantic_label": "坐标变换",
          "time_column": "mcap_log_time_ns"}],
        confirm=True,
    )
    assert r["confirmed"] == ["tf.jsonl"]
    ctx2 = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx2, str(tmp_path / "ds"))
    stream = next(s for s in ctx2.meta["streams"]
                  if Path(s["path"]).name == "tf.jsonl")
    assert stream.get("semantic_label") == "坐标变换"
    assert stream.get("kind") == "tf"


def test_hint_fires_at_exactly_half_confirmed(tmp_path: Path) -> None:
    """恰半数确认（13/26 场景的抽象）→ hint 仍触发（边界回归）。"""
    d = tmp_path / "ds"
    d.mkdir()
    # 6 流，确认 3 → 恰好 50%（hint 要求流数 ≥5 防小数据集噪声）。
    for i in range(6):
        _write_jsonl(d / f"t{i}.jsonl", [
            {"mcap_log_time_ns": T0_US * 1000 + j * 1_000_000,
             "data": {"v": float(j)}}
            for j in range(50)
        ])
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    for s in ctx.meta["streams"][:3]:
        s["label_source"] = "user_confirmed"
        s["semantic_label"] = "已确认流"

    ins = inspect_streams_impl(ctx)
    assert ins.get("unclassified_hint") is not None, (
        "恰半数未分类时 hint 应触发（旧实现 <0.5 会漏报）"
    )


def test_prompt_auto_translates_natural_language() -> None:
    """prompt 含自然语言确认请求的自动翻译纪律。"""
    assert "自动翻译" in SYSTEM_PROMPT
    assert "不要求用户提供字段路径" in SYSTEM_PROMPT
    assert "不要求也不展示工具名与参数" in SYSTEM_PROMPT
