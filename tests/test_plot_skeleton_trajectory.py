"""trajectory 支持骨骼块（声明驱动）的单元测试。

覆盖：声明块分解的向量列可画骨骼 3D 轨迹（轴标签用声明维度名）；块数超限只画
前若干块并计数完整；维度顺序推断被标注；无声明/无位置维/无 meta 时回退既有行为
（joint/end_effector），不得误报成功。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl
from app.tools.plot_chart import plot_chart_impl


def _lerobot_dir(tmp_path: Path, blocks: int = 24, dof: int = 7) -> Path:
    root = tmp_path / "lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    dim = blocks * dof
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 30,
        "features": {
            "observation.head_pose": {
                "dtype": "float32", "shape": [dof],
                "names": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
            },
            "observation.full_body": {
                "dtype": "float32", "shape": [dim],
                "names": [f"body_{blocks}x{dof}"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }), encoding="utf-8")
    n = 30

    def vec(base: float) -> str:
        # 每块 7 维：3 位置（随时间变化）+ 4 四元数（0,0,0,1）。
        vals: list[float] = []
        for b in range(blocks):
            vals.extend([base + b * 0.01, base + b * 0.02, base + b * 0.03, 0.0, 0.0, 0.0, 1.0])
        return " ".join(f"{v}" for v in vals[:dim])

    pd.DataFrame({
        "timestamp": [i * 0.033 for i in range(n)],
        "observation.head_pose": [vec(0.0) for _ in range(n)],
        "observation.full_body": [vec(i * 0.01) for i in range(n)],
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    return root


def test_skeleton_3d_trajectory_drawn(tmp_path: Path) -> None:
    """trajectory：声明块分解列 → 骨骼 3D 轨迹，轴标签用声明维度名。"""
    root = _lerobot_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = plot_chart_impl(ctx, "trajectory")
    assert r["success"] is True, r.get("error")
    spec = r["plot_spec"]
    assert spec["skeleton_column"] == "observation.full_body"
    assert spec["declared_name"] == "body_24x7"
    assert spec["blocks_total"] == 24
    assert spec["blocks_drawn"] < 24  # 只画前若干块
    assert spec["x_axis"] == "px"
    assert spec["y_axis"] == ["py", "pz"]
    assert spec["dimension_order_inferred"] is True  # 参照 head_pose 推断并标注
    assert Path(r["file_path"]).exists()


def test_no_block_declaration_falls_back(tmp_path: Path) -> None:
    """无块分解声明（纯标量列）→ 不走骨骼轨迹；无 joint/pose 列则 not_applicable。"""
    root = tmp_path / "plain"
    root.mkdir()
    n = 20
    pd.DataFrame({
        "timestamp": [i * 0.04 for i in range(n)],
        "value": [float(i) for i in range(n)],
    }).to_csv(root / "data.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = plot_chart_impl(ctx, "trajectory")
    # 无 joint/pose/骨骼列 → 不适用（保持既有行为，不误报成功）。
    assert r["success"] is False
    assert r["error"] == "not_applicable"


def test_joint_columns_still_work(tmp_path: Path) -> None:
    """有标量关节列时仍走 joint 轨迹（骨骼分支不抢占既有路径）。"""
    root = tmp_path / "joints"
    root.mkdir()
    n = 20
    pd.DataFrame({
        "timestamp": [i * 0.04 for i in range(n)],
        "qpos1": [float(i) for i in range(n)],
        "qpos2": [float(i * 2) for i in range(n)],
    }).to_csv(root / "data.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = plot_chart_impl(ctx, "trajectory")
    assert r["success"] is True
    # joint 轨迹无 skeleton_column 字段（未被骨骼分支抢占）。
    assert r["plot_spec"].get("skeleton_column") is None
