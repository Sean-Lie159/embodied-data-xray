"""改动 A：LeRobot meta 语义元数据解析与引用的单元测试。

覆盖：info.json 的 features 列语义（dtype/shape/names）与数据集级语义
（hand_tracked/robot_type/coordinate_frame/task/total_frames/source）；stats.json
每列统计量直接采用；profile_data 引用维度名而非推测。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools._sniffing import (
    column_dimension_names,
    parse_lerobot_info,
    parse_lerobot_stats,
)
from app.tools.load_dataset import load_dataset_impl
from app.tools.profile_data import profile_data_impl


def _lerobot_dir(tmp_path: Path) -> Path:
    """构造 LeRobot v2 目录（info.json 含 features 列语义 + stats.json）。"""
    root = tmp_path / "lerobot"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "repo_id": "test/v1",
        "total_episodes": 1,
        "total_frames": 20,
        "fps": 59,
        "robot_type": "TestRobot",
        "coordinate_frame": "Y-up floor",
        "hand_tracked": True,
        "task": "测试任务",
        "source": {"device": "TestDevice"},
        "features": {
            "observation.head_pose": {
                "dtype": "float32", "shape": [7],
                "names": ["px", "py", "pz", "qx", "qy", "qz", "qw"],
            },
            "observation.left_hand": {
                "dtype": "float32", "shape": [182], "names": ["left_hand_26x7"],
            },
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
    }), encoding="utf-8")
    (root / "meta" / "stats.json").write_text(json.dumps({
        "computed_on": "2026-01-01",
        "features": {
            "observation.head_pose": {"mean": [0.0] * 7, "std": [1.0] * 7,
                                      "min": [-1.0] * 7, "max": [1.0] * 7},
        },
    }), encoding="utf-8")
    n = 20
    pd.DataFrame({
        "timestamp": [i * 0.04 for i in range(n)],
        "observation.head_pose": [f"[{i}.0 0.0 0.0 0.0 0.0 0.0 1.0]" for i in range(n)],
        "observation.left_hand": [f"[0.0 {i}.0 0.0]" for i in range(n)],
    }).to_parquet(root / "data/chunk-000/episode_000000.parquet")
    (root / "videos/chunk-000/episode_000000.mp4").write_bytes(b"f")
    return root


def test_parse_info_extracts_column_semantics(tmp_path: Path) -> None:
    """info.json 的 features → 列语义表（dtype/shape/names）。"""
    root = _lerobot_dir(tmp_path)
    info = parse_lerobot_info(str(root / "meta/info.json"))
    sem = info.get("column_semantics") or {}
    assert set(sem.keys()) == {
        "observation.head_pose", "observation.left_hand", "timestamp",
    }
    assert sem["observation.head_pose"]["shape"] == [7]
    assert sem["observation.head_pose"]["names"] == [
        "px", "py", "pz", "qx", "qy", "qz", "qw",
    ]
    assert sem["timestamp"]["names"] is None


def test_parse_info_extracts_dataset_level_semantics(tmp_path: Path) -> None:
    """info.json 的数据集级语义字段（hand_tracked/robot_type/task/total_frames）。"""
    root = _lerobot_dir(tmp_path)
    info = parse_lerobot_info(str(root / "meta/info.json"))
    assert info["hand_tracked"] is True
    assert info["robot_type"] == "TestRobot"
    assert info["task"] == "测试任务"
    assert info["total_frames"] == 20
    assert info["fps"] == 59
    assert info["source"]["device"] == "TestDevice"


def test_column_dimension_names_helper(tmp_path: Path) -> None:
    """column_dimension_names：有声明返回名列表，无声明返回 None（不得推测）。"""
    root = _lerobot_dir(tmp_path)
    info = parse_lerobot_info(str(root / "meta/info.json"))
    assert column_dimension_names(info, "observation.head_pose") == [
        "px", "py", "pz", "qx", "qy", "qz", "qw",
    ]
    assert column_dimension_names(info, "timestamp") is None
    assert column_dimension_names({}, "observation.head_pose") is None


def test_parse_stats_extracts_per_column_stats(tmp_path: Path) -> None:
    """stats.json 的每列统计量直接采用（不重算）。"""
    root = _lerobot_dir(tmp_path)
    stats = parse_lerobot_stats(str(root / "meta/stats.json"))
    assert stats.get("computed_on") == "2026-01-01"
    feats = stats.get("features") or {}
    assert "observation.head_pose" in feats
    assert "mean" in feats["observation.head_pose"]


def test_profile_references_declared_dimension_names(tmp_path: Path) -> None:
    """端到端：加载后 profile_data 引用 meta 声明的维度名（不得自行推测）。"""
    root = _lerobot_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    assert (ctx.meta.get("lerobot_info") or {}).get("hand_tracked") is True
    pr = profile_data_impl(ctx)
    assert pr["success"] is True
    col = next(c for c in pr["columns"] if c["name"] == "observation.head_pose")
    assert col["dimension_names"] == ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    assert "info.json" in col["dimension_source"]
    assert col["declared_shape"] == [7]
    # stats.json 统计量被引用。
    assert col.get("dataset_stats") is not None
    assert "stats.json" in col["dataset_stats_source"]
