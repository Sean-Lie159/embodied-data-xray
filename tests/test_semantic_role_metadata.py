"""语义角色优先原则：元数据文件内容判定的单元测试。

判定绑定语义角色（小尺寸 + 配置型键 + 无标定键 + 无实质行列表），不绑定目录布局
（meta/ 仅为线索）。覆盖正反例：info.json 是元数据；含实质行列表的 episode JSON
是数据表；标定 JSON 是标定角色；大尺寸 JSON 不是元数据。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.tools._sniffing import _is_dataset_metadata_file


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_info_json_with_config_keys_is_metadata(tmp_path: Path) -> None:
    """含 fps/features 配置键的小 dict → 元数据（无论是否在 meta/ 目录下）。"""
    a = _write_json(tmp_path / "meta/info.json", {"fps": 25, "video": {"fps": 60}})
    b = _write_json(tmp_path / "other.json", {"robot_type": "arm", "fps": 30})
    assert _is_dataset_metadata_file(str(a)) is True
    assert _is_dataset_metadata_file(str(b)) is True


def test_episode_json_with_rowlist_is_data_table(tmp_path: Path) -> None:
    """含实质行列表（frames/data 非空）→ 数据表角色，即使含 fps 配置键。"""
    p = _write_json(
        tmp_path / "data/chunk-000/episode_000000.json",
        {"fps": 60, "episode_index": 0,
         "frames": [{"timestamp": i, "obs": i} for i in range(50)]},
    )
    assert _is_dataset_metadata_file(str(p)) is False


def test_calibration_json_is_not_metadata(tmp_path: Path) -> None:
    """含标定键 → 标定角色，不是元数据。"""
    p = _write_json(tmp_path / "imu_calibration.json", {"bias": [0.0], "scale": [1.0]})
    assert _is_dataset_metadata_file(str(p)) is False


def test_large_json_is_not_metadata(tmp_path: Path) -> None:
    """大尺寸 JSON 超过元数据上限 → 不视为元数据。"""
    big = {"fps": 25, "payload": "x" * 200_000}
    p = _write_json(tmp_path / "big.json", big)
    assert p.stat().st_size > 100_000
    assert _is_dataset_metadata_file(str(p)) is False


def test_meta_dir_minimal_json_is_metadata_fallback(tmp_path: Path) -> None:
    """meta/ 下极简配置 JSON → 兜底按元数据处理。"""
    p = _write_json(tmp_path / "meta/tasks.json", {"tasks": []})
    assert _is_dataset_metadata_file(str(p)) is True
