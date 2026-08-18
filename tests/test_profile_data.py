"""app/tools/profile_data 工具的单元测试。

基于 RunContext 中已加载的 DataFrame 生成概况，不依赖网络。用合成 DataFrame
直接注入 RunContext 模拟"已加载"状态。
"""

from __future__ import annotations

import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.profile_data import profile_data, profile_data_impl


def test_profile_data_is_registered_as_function_tool() -> None:
    """profile_data 应被 @tool 注册为 FunctionTool 实例。"""
    assert isinstance(profile_data, FunctionTool)
    assert profile_data.name == "profile_data"


def test_profile_data_returns_row_and_col_counts() -> None:
    ctx = RunContext(df=pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))

    result = profile_data_impl(ctx)

    assert result["success"] is True
    assert result["n_rows"] == 3
    assert result["n_cols"] == 2
    assert [c["name"] for c in result["columns"]] == ["a", "b"]


def test_profile_data_missing_value_stats() -> None:
    # b 列第三行为空，应统计出 1 个缺失、约 33.33%。
    ctx = RunContext(df=pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]}))

    result = profile_data_impl(ctx)
    col_b = next(c for c in result["columns"] if c["name"] == "b")

    assert col_b["n_missing"] == 1
    assert col_b["pct_missing"] == 33.33
    assert col_b["n_unique"] == 2


def test_profile_data_numeric_stats() -> None:
    # 数值列 c：10.5, 20, 30.25 → mean 20.25, min 10.5, max 30.25, median 20。
    ctx = RunContext(
        df=pd.DataFrame({"a": [1, 2, 3], "c": [10.5, 20, 30.25]})
    )

    result = profile_data_impl(ctx)
    col_c = next(c for c in result["columns"] if c["name"] == "c")

    numeric = col_c["numeric"]
    assert numeric["mean"] == 20.25
    assert numeric["min"] == 10.5
    assert numeric["max"] == 30.25
    assert numeric["median"] == 20.0


def test_profile_data_unique_counts_and_samples() -> None:
    ctx = RunContext(df=pd.DataFrame({"a": [1, 1, 2, 2, 2]}))

    # 显式设置 max_unique=2，验证样例值被截断。
    result = profile_data_impl(ctx, max_unique=2)
    col_a = next(c for c in result["columns"] if c["name"] == "a")

    assert col_a["n_unique"] == 2
    assert col_a["sample_values"] == ["1", "1"]


def test_profile_data_unloaded_returns_guidance() -> None:
    ctx = RunContext()  # 未加载任何数据

    result = profile_data_impl(ctx)

    assert result["success"] is False
    assert "load_dataset" in result["suggestion"]
    assert "尚未加载" in result["error"]
