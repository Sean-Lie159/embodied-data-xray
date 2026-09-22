"""入参归一测试（**真实事故回归**，2026-09-22）。

事故：模型把 JSON 的 ``null`` 传成**字符串** ``"null"``，
``check_temporal_sync(baseline_stream="null")`` 拿它去匹配文件名（必然失败）
→ 返回 ``baseline_no_match`` → 模型误以为参数无效而**重试整轮**，
白耗一次工具循环 + 模型往返（当时正值长思考链，加剧了 UI 卡死）。

同类风险遍布 14 个工具的 31 个可选参数，故在工具调用的**唯一经过点**
（``guard_tools``）统一归一，而非逐个工具修改。
"""

from __future__ import annotations

import json

import pytest

from app.agent.agent import _normalize_tool_input
from app.tools._arg_normalize import (
    NULLISH_LITERALS,
    normalize_optional_int,
    normalize_optional_list,
    normalize_optional_str,
)


# ---------------------------------------------------------------------------
# 基础归一函数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("literal", [
    "null", "NULL", " null ", "none", "None", "undefined", "nil",
    "", "  ", "-", "n/a", "N/A", "na", "auto", "default",
    "自动", "无", "空",
])
def test_nullish_string_normalizes_to_none(literal: str) -> None:
    """各种"空值字面量"（含大小写与空白变体）都归一为 None。"""
    assert normalize_optional_str(literal) is None


@pytest.mark.parametrize("value,expected", [
    ("joints_position.csv", "joints_position.csv"),
    ("  a.csv  ", "a.csv"),
    ("all", "all"),          # 注意："all" 语义是"全部"，非"未指定"
    ("全部", "全部"),
])
def test_real_values_are_preserved(value: str, expected: str) -> None:
    """真实值必须原样保留（尤其 'all' 表达明确要求全量，不能当空值）。"""
    assert normalize_optional_str(value) == expected


def test_none_and_non_string_handled() -> None:
    """None 返回 None；非字符串转为字符串。"""
    assert normalize_optional_str(None) is None
    assert normalize_optional_str(123) == "123"


def test_list_normalization_drops_nullish_items() -> None:
    """列表中的空值元素被剔除；全空则返回 None。"""
    assert normalize_optional_list(["a.csv", "null", "b.csv"]) == ["a.csv", "b.csv"]
    assert normalize_optional_list(["null", "none"]) is None
    assert normalize_optional_list(None) is None
    assert normalize_optional_list([]) is None


def test_list_from_single_string_is_wrapped() -> None:
    """**误传单个字符串时包成单元素列表**（而非逐字符拆开）。

    这是真实踩坑风险：``"a.csv"`` 是可迭代的，若直接迭代会得到
    ``"a"``、``"."``、``"c"`` … 产生大量无意义匹配。
    """
    assert normalize_optional_list("a.csv") == ["a.csv"]
    assert normalize_optional_list("null") is None


def test_int_normalization() -> None:
    """整数参数归一（含数字字符串）。"""
    assert normalize_optional_int(5) == 5
    assert normalize_optional_int("5") == 5
    assert normalize_optional_int("5.0") == 5
    assert normalize_optional_int("null") is None
    assert normalize_optional_int(None) is None
    assert normalize_optional_int("abc") is None


def test_nullish_set_excludes_all() -> None:
    """**设计边界**：'all'/'全部' 不在空值集合内（语义不同）。"""
    assert "all" not in NULLISH_LITERALS
    assert "全部" not in NULLISH_LITERALS


# ---------------------------------------------------------------------------
# Agent 层统一拦截（覆盖全部工具）
# ---------------------------------------------------------------------------


def test_agent_layer_strips_nullish_scalar() -> None:
    """工具入参中的空值标量被删除（等价"未指定"）。"""
    out = _normalize_tool_input(
        json.dumps({"baseline_stream": "null", "locate_gaps": True}), "t")
    assert json.loads(out) == {"locate_gaps": True}


def test_agent_layer_strips_nullish_in_list() -> None:
    """列表内的空值元素被剔除。"""
    out = _normalize_tool_input(
        json.dumps({"streams": ["a.csv", "null"], "x": 1}), "t")
    assert json.loads(out) == {"streams": ["a.csv"], "x": 1}


def test_agent_layer_removes_emptied_list() -> None:
    """列表被剔空后删除该键。"""
    out = _normalize_tool_input(json.dumps({"streams": ["null", "none"]}), "t")
    assert json.loads(out) == {}


def test_agent_layer_preserves_normal_input() -> None:
    """正常入参**逐字节不变**（零回归保证）。"""
    original = json.dumps({"path": "data/x", "fmt": "csv", "n": 3})
    assert _normalize_tool_input(original, "load_dataset") == original


def test_agent_layer_preserves_all_keyword() -> None:
    """'all' 不得被当作空值删除（语义是"全部流"）。"""
    out = _normalize_tool_input(json.dumps({"streams": ["all"]}), "t")
    assert json.loads(out) == {"streams": ["all"]}


def test_agent_layer_tolerates_invalid_json() -> None:
    """非法 JSON 原样返回（不干扰正常流程）。"""
    bad = "{not json"
    assert _normalize_tool_input(bad, "t") == bad
    assert _normalize_tool_input("", "t") == ""


def test_agent_layer_tolerates_non_object_json() -> None:
    """顶层非对象时原样返回。"""
    arr = json.dumps([1, 2, 3])
    assert _normalize_tool_input(arr, "t") == arr


# ---------------------------------------------------------------------------
# 端到端：事故场景不再重试
# ---------------------------------------------------------------------------


def test_baseline_no_match_no_longer_triggered_by_null_string() -> None:
    """**事故场景验收**：``baseline_stream="null"`` 不再走到 ``baseline_no_match``。

    此前：工具拿字符串 ``"null"`` 去匹配文件名子串 → 必然失败 →
    返回 ``baseline_no_match`` → 模型误以为参数无效而重试整轮。

    现在：Agent 层先归一（删除该键）→ 工具收到的是"未指定" →
    走缺省分支（自动推荐基线），**不再产生匹配失败**。

    本测试验证「归一 → 缺省分支」这条路径成立：用同一份依赖，对比
    "传 'null'" 与 "不传" 的**行为等价**。
    """
    import numpy as np
    import pandas as pd

    from app.agent.context import RunContext
    from app.config import get_settings
    from app.tools.check_temporal_sync import check_temporal_sync_impl

    n = 300
    step = 1e9 / 120
    df = pd.DataFrame({
        "timestamp_ns": (np.arange(n) * step).astype("int64"),
        "a": np.sin(np.arange(n) * 0.1),
        "b": np.cos(np.arange(n) * 0.1),
    })
    ctx = RunContext()
    ctx.df = df
    ctx.dataset_id = "demo"
    ctx.meta = {"streams": [], "capabilities": {}}

    # 走一遍 Agent 层归一：'null' 被删除。
    normalized = json.loads(_normalize_tool_input(
        json.dumps({"baseline_stream": "null"}), "check_temporal_sync"))
    assert "baseline_stream" not in normalized, "归一未删除 'null' 键"

    # 归一后的等价入参（未指定基线）与"完全不传"必须得到同一结果。
    a = check_temporal_sync_impl(
        ctx, get_settings(), baseline_stream=normalized.get("baseline_stream"))
    b = check_temporal_sync_impl(ctx, get_settings())
    assert a.get("error") == b.get("error")
    # 关键：不得是 'baseline_no_match'（那正是当初触发重试的错误）。
    assert a.get("error") != "baseline_no_match"
    # 若走到匹配分支则错误，说明归一失效；这里因无流而 not_applicable 是可接受的。
    assert a.get("error") in (None, "not_applicable")

