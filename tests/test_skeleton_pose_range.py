"""skeleton_pose_range（骨骼位姿范围）的单元测试。

覆盖：块分解声明解析（NxM 自洽才采信，不自洽不猜测）；指标与 joint_range_of_motion
严格区分；块内维度顺序优先本列声明、否则参照同数据集同 dof 列推断并标注；
无声明列不处理；块数超限时截断但计数完整。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools._sniffing import infer_dof_order, parse_block_declaration
from app.tools.compute_stats import compute_stats_impl
from app.tools.load_dataset import load_dataset_impl


def _vec(vals: list[float]) -> str:
    return " ".join(f"{v}" for v in vals)


def _info(features: dict) -> dict:
    """构造 parse_lerobot_info 风格的返回。"""
    return {
        "column_semantics": {
            k: {"dtype": "float32", "shape": v.get("shape"), "names": v.get("names")}
            for k, v in features.items()
        }
    }


def test_parse_block_declaration_valid_and_inconsistent() -> None:
    """NxM 声明自洽才采信；不自洽/逐维声明/无名均返回 None（不猜测）。"""
    info = _info({
        "observation.full_body": {"shape": [168], "names": ["body_24x7"]},
        "bad": {"shape": [10], "names": ["bad_2x3"]},  # 2x3 != 10 → 不自洽
        "observation.head_pose": {
            "shape": [7],
            "names": ["px", "py", "pz", "qx", "qy", "qz", "qw"],  # 逐维 → 非块声明
        },
        "noname": {"shape": [5], "names": None},
    })
    decl = parse_block_declaration(info, "observation.full_body")
    assert decl is not None
    assert decl["block_count"] == 24 and decl["dof_per_block"] == 7
    assert parse_block_declaration(info, "bad") is None
    assert parse_block_declaration(info, "observation.head_pose") is None
    assert parse_block_declaration(info, "noname") is None


def test_infer_dof_order_prefers_own_then_reference() -> None:
    """维度顺序：本列有逐维声明直接用；否则参照同数据集同 dof 列并标 inferred。"""
    info = _info({
        "observation.head_pose": {
            "shape": [7], "names": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
        },
        "observation.full_body": {"shape": [168], "names": ["body_24x7"]},
    })
    own = infer_dof_order(info, "observation.head_pose", 7)
    assert own["is_inferred"] is False
    assert own["order"] == ["px", "py", "pz", "qx", "qy", "qz", "qw"]

    ref = infer_dof_order(info, "observation.full_body", 7)
    assert ref["is_inferred"] is True
    assert ref["order"] == ["px", "py", "pz", "qx", "qy", "qz", "qw"]

    # 无同 dof 参照 → 占位且不解读语义。
    info2 = _info({"x": {"shape": [9], "names": ["x_3x3"]}})
    none = infer_dof_order(info2, "x", 3)
    assert none["order"] == ["dim_0", "dim_1", "dim_2"]
    assert "未声明" in none["source"]


def _lerobot_skeleton_dir(tmp_path: Path) -> Path:
    """LeRobot 目录：full_body(24x7=168) + left_hand(26x7=182) + head_pose(7) + timestamp。"""
    root = tmp_path / "lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "fps": 30,
        "robot_type": "TestRobot",
        "hand_tracked": True,
        "features": {
            "observation.head_pose": {
                "dtype": "float32", "shape": [7],
                "names": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
            },
            "observation.full_body": {
                "dtype": "float32", "shape": [168], "names": ["body_24x7"],
            },
            "observation.left_hand": {
                "dtype": "float32", "shape": [182], "names": ["left_hand_26x7"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }), encoding="utf-8")
    n = 20

    def block_vec(dim: int, base: float) -> str:
        return _vec([base + (i % 7) * 0.1 for i in range(dim)])

    pd.DataFrame({
        "timestamp": [i * 0.033 for i in range(n)],
        "observation.head_pose": [block_vec(7, 0.0) for _ in range(n)],
        "observation.full_body": [block_vec(168, 1.0) for _ in range(n)],
        "observation.left_hand": [block_vec(182, 2.0) for _ in range(n)],
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    return root


def test_skeleton_pose_range_computed_and_distinct_from_rom(tmp_path: Path) -> None:
    """端到端：skeleton_pose_range 覆盖声明列；joint_range_of_motion 不混淆（本例为 None）。"""
    root = _lerobot_skeleton_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = compute_stats_impl(ctx)
    assert r["success"] is True
    sk = (r.get("metrics") or {}).get("skeleton_pose_range") or {}
    # full_body 与 left_hand 被分解；head_pose 是逐维声明（非块）→ 不在该指标内。
    assert set(sk.keys()) == {"observation.full_body", "observation.left_hand"}
    fb = sk["observation.full_body"]
    assert fb["block_count"] == 24 and fb["dof_per_block"] == 7
    assert fb["blocks_truncated"] is True and fb["blocks_shown"] < 24
    assert fb["dimension_order_inferred"] is True
    assert "head_pose" in fb["dimension_order_source"]
    # 语义说明明确"非关节角度"。
    assert "非关节角度" in fb["semantic_note"]
    # joint_range_of_motion 不因向量列而产出（严格区分）。
    assert (r.get("metrics") or {}).get("joint_range_of_motion") is None


def test_skeleton_blocks_have_position_and_quaternion_stats(tmp_path: Path) -> None:
    """块内统计：位置 min/max/range + 四元数模长稳定性。"""
    root = _lerobot_skeleton_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = compute_stats_impl(ctx)
    sk = (r.get("metrics") or {}).get("skeleton_pose_range") or {}
    b0 = sk["observation.full_body"]["blocks"]["block_0"]
    assert len(b0["position_min"]) == 3
    assert len(b0["position_max"]) == 3
    assert len(b0["position_range"]) == 3
    assert "quaternion_norm_mean" in b0
    assert isinstance(b0["quaternion_norm_stable"], bool)


def test_no_declaration_no_skeleton_range(tmp_path: Path) -> None:
    """无块分解声明 → 不产出 skeleton_pose_range（不猜测）。"""
    root = tmp_path / "plain"
    root.mkdir()
    n = 20
    pd.DataFrame({
        "timestamp": [i * 0.04 for i in range(n)],
        "value": [float(i) for i in range(n)],
    }).to_csv(root / "data.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    r = compute_stats_impl(ctx)
    assert r["success"] is True
    assert (r.get("metrics") or {}).get("skeleton_pose_range") is None
