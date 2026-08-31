"""改动 B：plot_chart 支持向量列的单元测试。

覆盖：向量列（LeRobot observation.head_pose 等）折线/散点图按维度展开绘制，
图例用数据集声明的维度名；高维列只画前若干维并注明；非向量列行为不变。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl
from app.tools.plot_chart import _vector_matrix, plot_chart_impl


def _lerobot_dir(tmp_path: Path) -> Path:
    """构造 LeRobot 目录：向量列 head_pose（7 维，names 已声明）+ 标量 timestamp。"""
    root = tmp_path / "lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 30,
        "robot_type": "TestRobot",
        "features": {
            "observation.head_pose": {
                "dtype": "float32", "shape": [7],
                "names": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }), encoding="utf-8")
    n = 30
    pd.DataFrame({
        "timestamp": [i * 0.033 for i in range(n)],
        "observation.head_pose": [
            f"[{i * 0.1} {i * 0.2} {i * 0.3} 0.0 0.0 0.0 1.0]" for i in range(n)
        ],
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    return root


def test_vector_matrix_parses() -> None:
    """向量列解析为 (n_rows, n_dims) 矩阵；非向量列返回 None。"""
    s = pd.Series(["[1.0 2.0 3.0]", "[4.0 5.0 6.0]", "[7.0 8.0 9.0]"])
    mat = _vector_matrix(s)
    assert mat is not None
    assert mat.shape == (3, 3)
    assert _vector_matrix(pd.Series(["hello", "world", "foo"])) is None


def test_vector_column_line_chart_expands_by_dimension(tmp_path: Path) -> None:
    """向量列折线图：按维度展开，图例/plot_spec 用声明的维度名（不推测）。"""
    root = _lerobot_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = plot_chart_impl(ctx, "line", x="timestamp", y="observation.head_pose")
    assert r["success"] is True, r.get("error")
    spec = r["plot_spec"]
    assert spec["vector_column"] == "observation.head_pose"
    assert spec["dims_total"] == 7
    assert spec["dims_drawn"] == 7
    # 图例/轴用数据集声明的维度名。
    assert spec["y_axis"] == ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    assert "info.json" in spec["dimension_source"]
    assert Path(r["file_path"]).exists()


def test_vector_column_scatter_chart(tmp_path: Path) -> None:
    """向量列散点图同样按维度展开。"""
    root = _lerobot_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = plot_chart_impl(ctx, "scatter", x="timestamp", y="observation.head_pose")
    assert r["success"] is True
    assert r["plot_spec"]["n_series"] == 7


def test_high_dim_vector_drawn_partially(tmp_path: Path) -> None:
    """高维向量列：只画前若干维并在说明中注明总维度（避免图不可读）。"""
    root = tmp_path / "hd"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 30,
        "features": {
            "observation.left_hand": {
                "dtype": "float32", "shape": [182], "names": ["left_hand_26x7"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }), encoding="utf-8")
    n = 20
    pd.DataFrame({
        "timestamp": [i * 0.04 for i in range(n)],
        "observation.left_hand": [" ".join(["0.1"] * 182)] * n,
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = plot_chart_impl(ctx, "line", x="timestamp", y="observation.left_hand")
    assert r["success"] is True
    spec = r["plot_spec"]
    assert spec["dims_total"] == 182
    assert spec["dims_drawn"] < 182
    assert "182" in r["description"]


def test_scalar_column_unchanged(tmp_path: Path) -> None:
    """非向量（标量）列行为不变：单序列，无 vector_column 字段。"""
    root = tmp_path / "scalar"
    root.mkdir()
    n = 20
    pd.DataFrame({"timestamp": [i * 0.04 for i in range(n)],
                  "value": [float(i) for i in range(n)]}).to_csv(
        root / "data.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = plot_chart_impl(ctx, "line", x="timestamp", y="value")
    assert r["success"] is True
    assert r["plot_spec"].get("vector_column") is None
    assert r["plot_spec"]["y_axis"] == ["value"]
