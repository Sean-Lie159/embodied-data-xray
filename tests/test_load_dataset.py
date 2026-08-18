"""app/tools/load_dataset 工具的单元测试。

使用小型合成数据（CSV 与 Parquet）验证加载、格式分发、错误处理与 RunContext
写入逻辑，不依赖网络。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset, load_dataset_impl


def test_load_dataset_is_registered_as_function_tool() -> None:
    """load_dataset 应被 @tool 注册为 FunctionTool 实例。"""
    assert isinstance(load_dataset, FunctionTool)
    assert load_dataset.name == "load_dataset"


def test_load_csv_writes_context(tmp_path: Path) -> None:
    csv_path = tmp_path / "robot.csv"
    csv_path.write_text("episode,success\n0,1\n1,0\n2,1\n", encoding="utf-8")
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(csv_path))

    assert result["success"] is True
    assert result["format"] == "csv"
    assert result["n_rows"] == 3
    assert result["n_cols"] == 2
    assert result["columns"] == ["episode", "success"]
    # 数据写入 RunContext.df，元信息写入 RunContext.meta。
    assert ctx.df is not None
    assert ctx.df.shape == (3, 2)
    assert ctx.meta["n_rows"] == 3
    assert ctx.meta["source"] == str(csv_path)


def test_load_parquet_writes_context(tmp_path: Path) -> None:
    parquet_path = tmp_path / "robot.parquet"
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]}).to_parquet(
        parquet_path
    )
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(parquet_path))

    assert result["success"] is True
    assert result["format"] == "parquet"
    assert result["n_rows"] == 3
    assert ctx.df is not None
    assert list(ctx.df.columns) == ["x", "label"]


def test_load_unsupported_format_returns_error(tmp_path: Path) -> None:
    bad = tmp_path / "data.xyz"
    bad.write_text("x")
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(bad))

    assert result["success"] is False
    assert "error" in result
    assert "suggestion" in result
    assert ".csv" in result["suggestion"]
    assert ".parquet" in result["suggestion"]


def test_load_missing_file_returns_error(tmp_path: Path) -> None:
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(tmp_path / "not_exist.csv"))

    assert result["success"] is False
    assert "文件不存在" in result["error"]


def test_load_corrupted_parquet_returns_error(tmp_path: Path) -> None:
    bad = tmp_path / "broken.parquet"
    bad.write_bytes(b"not a real parquet file")
    ctx = RunContext()

    result = load_dataset_impl(ctx, str(bad))

    assert result["success"] is False
    assert "error" in result
