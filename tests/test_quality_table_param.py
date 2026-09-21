"""`check_dataset_quality` 的 `table` 参数测试（多表并列机制阶段 4）。

覆盖（对应 `docs/多表并列机制设计.md` §4.4 与 §6 阶段 4 的验收标准）：

1. 能对**非缺省表**出质检结论，且返回注明检查对象（`table` / `is_default_table`）；
2. `user_message` 明确说明质检对象表名（避免"某张表有问题"被误读为"整个数据集有问题"）；
3. 表不存在 → 返回 `table_not_found` 并附可用清单，**不静默回退主表**；
4. `meta["qc"]` 按表名区分键，**多表质检不互相覆盖**；
5. 无主表 / 指定表为空时仍标 `skip` 而非报错（保持既有行为）；
6. 缺省行为（不传 table）与改造前一致，`table` 字段为缺省表名。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.check_dataset_quality import check_dataset_quality_impl
from app.tools.load_dataset import load_dataset_impl


def _make_dir(tmp_path: Path, name: str = "multi") -> Path:
    """主表 state.csv（干净）+ accel.csv（含 NaN，应判 fail）。"""
    root = tmp_path / name
    root.mkdir()
    n = 40
    pd.DataFrame({
        "episode": [i // 10 for i in range(n)],
        "qpos1": [0.1 + i * 0.01 for i in range(n)],
        "success": [1] * n,
    }).to_csv(root / "state.csv", index=False)
    # 非主表：故意注入 NaN，质检应判 fail。
    m = 30
    pd.DataFrame({
        "t": list(range(m)),
        "x": [0.0] * (m - 6) + [float("nan")] * 6,
    }).to_csv(root / "accel.csv", index=False)
    return root


def test_check_non_default_table(tmp_path: Path) -> None:
    """**核心**：能对非缺省表出质检结论并注明对象。"""
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    r = check_dataset_quality_impl(ctx, table="accel.csv")
    assert r["success"] is True
    assert r["table"] == "accel.csv", "必须注明质检对象"
    assert r["is_default_table"] is False
    # accel.csv 有 NaN → 硬门禁应判 fail（证明真的检查了这张表）。
    assert r["result"] == "fail"
    assert "nan_inf" in r["gate"]["failed"]
    # user_message 必须点明表名。
    assert "accel.csv" in r["user_message"]


def test_default_table_unchanged(tmp_path: Path) -> None:
    """不传 table → 行为与改造前一致，table 字段为缺省表名。"""
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    r = check_dataset_quality_impl(ctx)
    assert r["success"] is True
    assert r["table"] == "state.csv", "缺省应质检主表"
    assert r["is_default_table"] is True
    # 主表干净 → 不应因 NaN 判 fail。
    assert "nan_inf" not in (r["gate"]["failed"] or [])


def test_different_tables_different_verdicts(tmp_path: Path) -> None:
    """**多表价值证明**：同一数据集，不同表的结论可以相反。

    这是本参数存在的根本理由——主表干净而次表有硬缺陷，若不支持选表，
    用户得到的"pass"会让他误以为整份数据都没问题。
    """
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    main_r = check_dataset_quality_impl(ctx)
    other_r = check_dataset_quality_impl(ctx, table="accel.csv")
    assert main_r["result"] != other_r["result"], (
        "主表与含 NaN 的次表应给出不同结论，否则质检未真正针对指定表"
    )


def test_table_not_found_no_silent_fallback(tmp_path: Path) -> None:
    """表不存在 → table_not_found + 候选清单，**不得静默回退主表**。

    静默回退是最坏的失败模式：模型会以为"已质检末端表"，实际质检的是主表，
    结论被张冠李戴。
    """
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    r = check_dataset_quality_impl(ctx, table="no_such.csv")
    assert r["success"] is False
    assert r["error"] == "table_not_found"
    # 错误里须带候选清单，便于模型自我纠正。
    assert "accel.csv" in r["available_tables"]
    assert r["default_table"] == "state.csv"
    # 明确这是质检场景。
    assert r["check"] == "check_dataset_quality"
    assert "质检未能执行" in r["user_message"]


def test_qc_meta_key_per_table(tmp_path: Path) -> None:
    """meta["qc"] 按表名区分键：多表质检**不互相覆盖**。

    回归：此前只用一个固定键 `check_dataset_quality`，多表场景下后一次质检会
    静默覆盖前一次，generate_report 读到哪张表的结论全看调用顺序。
    """
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(_make_dir(tmp_path)))

    check_dataset_quality_impl(ctx)                       # 缺省表
    check_dataset_quality_impl(ctx, table="accel.csv")    # 次表

    qc = ctx.meta["qc"]
    assert "check_dataset_quality" in qc, "缺省表结论应保留"
    assert "check_dataset_quality::accel.csv" in qc, "次表结论应独立存放"
    # 两份结论互不覆盖，且各自标明了表。
    assert qc["check_dataset_quality"]["table"] == "state.csv"
    assert qc["check_dataset_quality::accel.csv"]["table"] == "accel.csv"


def test_no_data_loaded_unchanged() -> None:
    """未加载数据集仍返回 no_data_loaded。"""
    ctx = RunContext()
    r = check_dataset_quality_impl(ctx, table="x.csv")
    assert r["error"] == "no_data_loaded"


def test_media_only_no_table_skips(tmp_path: Path) -> None:
    """无主表（纯媒体）→ 标 skip 而非报错（保持既有行为）。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "media_only"
    ctx.meta["streams"] = [
        {"path": "/x/a.mp4", "format": "video", "kind": "video"},
    ]
    ctx.meta["main_table"] = None

    r = check_dataset_quality_impl(ctx)
    assert r["success"] is True, "无主表是合法状态，不得报错"
    assert r["gate"]["checks"]["main_table"]["result"] == "skip"
    assert r["table"] is None, "无表时 table 为 None，不得编造"


def test_empty_requested_table_skips_with_table_name(tmp_path: Path) -> None:
    """指定表为空 → 标 skip，且 detail 应点明是哪张表为空。"""
    ctx = RunContext(output_dir=str(tmp_path))
    ctx.dataset_id = "has_empty"
    ctx.df = pd.DataFrame({"a": [1.0]})
    ctx.meta["streams"] = [
        {"path": "/x/empty.csv", "format": "csv", "kind": "unknown"},
    ]
    ctx.meta["main_table"] = {"name": "a.csv"}
    # 直接注入空 df 的解析结果不可行，改为构造真实空 csv 目录。
    root = tmp_path / "emptydir"
    root.mkdir()
    pd.DataFrame({"a": [1.0, 2.0]}).to_csv(root / "main.csv", index=False)
    pd.DataFrame({"b": []}).to_csv(root / "empty.csv", index=False)
    ctx2 = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx2, str(root))

    r = check_dataset_quality_impl(ctx2, table="empty.csv")
    # 空表要么被判 skip（无数据行），要么被 load_dataset 归为 empty 流而
    # table_not_found——两者都是**如实反映**，不得静默回退主表。
    if r["success"]:
        assert r["table"] == "empty.csv"
        assert r["gate"]["checks"]["main_table"]["result"] == "skip"
        assert "empty.csv" in r["gate"]["checks"]["main_table"]["detail"]
    else:
        assert r["error"] == "table_not_found"


def test_tool_schema_exposes_table() -> None:
    """@tool 生成的 schema 必须包含 table 参数（否则模型无法指定表）。

    注意：``@tool`` 装饰后拿到的是 ``FunctionTool`` 对象而非函数，
    不能用 ``inspect.signature``——须读它生成的 ``params_json_schema``。
    """
    from app.services.chat_service import _ALL_TOOLS

    tool_obj = next(
        t for t in _ALL_TOOLS if getattr(t, "name", None) == "check_dataset_quality"
    )
    schema = tool_obj.params_json_schema
    props = schema.get("properties") or {}
    assert "table" in props, f"schema 必须含 table 参数：{list(props)}"
    # 可空字符串（缺省=主表）。
    assert {"type": "string"} in props["table"]["anyOf"]
    assert {"type": "null"} in props["table"]["anyOf"]
    # 参数描述须给模型足够指引。
    assert "list_tables" in props["table"]["description"]
    # 工具描述里必须写明"每张表须分别质检"的纪律。
    assert "分别质检" in tool_obj.description
