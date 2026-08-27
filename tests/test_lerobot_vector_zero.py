"""Commit B：LeRobot 向量串解析 + 全零判定修复的单元测试。

覆盖：空格/换行分隔向量串解析；全列全零才判掉线（起始段全零但整体有值不误判）。
"""

from __future__ import annotations

import numpy as np

from app.tools._data_access import parse_lerobot_vector
from app.tools.check_sensor_sanity import _constant_columns


def test_parse_lerobot_vector_space_separated() -> None:
    """空格/换行分隔的向量串解析为数值数组。"""
    v = "0. 0. 0. 0. \n 1. 2. 3."
    vec = parse_lerobot_vector(v)
    assert vec is not None
    assert vec.shape == (7,)
    assert np.allclose(vec[:4], 0.0)
    assert np.allclose(vec[4:], [1.0, 2.0, 3.0])


def test_parse_lerobot_vector_json_array() -> None:
    """JSON 数组字符串解析。"""
    vec = parse_lerobot_vector("[1.0, 2.0, 3.0]")
    assert vec is not None
    assert np.allclose(vec, [1.0, 2.0, 3.0])


def test_parse_lerobot_vector_list() -> None:
    """list/tuple 直接转换。"""
    assert np.allclose(parse_lerobot_vector([0.0, 0.5, 1.0]), [0.0, 0.5, 1.0])


def test_vector_column_partially_zero_not_constant() -> None:
    """向量串列"起始段全零但整体有值" → 不判为全零掉线（基于全列分布）。"""
    data = {
        "observation.left_hand": np.array(
            ["0. 0. 0."] * 20 + ["1. 2. 3."] * 20, dtype=object
        ),
    }
    constant = _constant_columns(data, 1e-6)
    assert "observation.left_hand" not in constant


def test_vector_column_all_zero_constant() -> None:
    """向量串列全列全零 → 判为掉线。"""
    data = {
        "observation.left_hand": np.array(["0. 0. 0."] * 20, dtype=object),
    }
    constant = _constant_columns(data, 1e-6)
    assert "observation.left_hand" in constant
