"""Commit 1 确定性缺陷修复的单元测试。

覆盖：表行数读数统一（read_table_nrows 与全量读一致，csv/parquet/json）；
恒定通道检测排除索引/元数据列、object/数组列全零检测。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.tools import _data_access
from app.tools.check_sensor_sanity import _constant_columns, _is_index_metadata_column


def test_read_table_nrows_matches_full_read(tmp_path: Path) -> None:
    """统一行数读数与全量读一致（csv/parquet/json）。"""
    # csv。
    csv_path = tmp_path / "a.csv"
    pd.DataFrame({"x": list(range(20)), "y": list(range(20))}).to_csv(csv_path, index=False)
    assert _data_access.read_table_nrows(str(csv_path), "csv") == 20
    assert _data_access.read_stream_full(str(csv_path), "csv").shape[0] == 20
    # parquet。
    pq_path = tmp_path / "b.parquet"
    pd.DataFrame({"x": list(range(15))}).to_parquet(pq_path)
    assert _data_access.read_table_nrows(str(pq_path), "parquet") == 15
    assert _data_access.read_stream_full(str(pq_path), "parquet").shape[0] == 15
    # json（数组）。
    json_path = tmp_path / "c.json"
    pd.DataFrame({"x": list(range(10))}).to_json(json_path, orient="records")
    assert _data_access.read_table_nrows(str(json_path), "json") == 10


def test_is_index_metadata_column() -> None:
    """索引/元数据列识别。"""
    assert _is_index_metadata_column("frame_index") is True
    assert _is_index_metadata_column("timestamp_ns") is True
    assert _is_index_metadata_column("packet_index") is True
    assert _is_index_metadata_column("row_id") is True
    assert _is_index_metadata_column("accel_x") is False
    assert _is_index_metadata_column("left_joint0_pos_x") is False


def test_constant_columns_excludes_index_columns() -> None:
    """恒定检测排除索引/元数据列（frame_index 恒定时不误报掉线）。"""
    data = {
        "frame_index": np.zeros(50),          # 恒定索引，应排除
        "accel_x": np.zeros(50),              # 数值恒定 → 报掉线
        "left_joint0_x": np.array([0.0] * 25 + [1.0] * 25),  # 有变化 → 不报
    }
    constant = _constant_columns(data, 1e-6)
    assert "frame_index" not in constant  # 索引列被排除
    assert "accel_x" in constant          # 数值恒定报掉线


def test_constant_columns_object_all_zero() -> None:
    """object/数组列全零 → 纳入掉线检测。"""
    data = {
        "left_hand_joints": np.array([{"a": 0}] * 20, dtype=object),  # 全零对象
        "right_hand_joints": np.array([{"a": 1}, {"a": 2}] * 10, dtype=object),  # 非零
    }
    constant = _constant_columns(data, 1e-6)
    assert "left_hand_joints" in constant
    assert "right_hand_joints" not in constant
