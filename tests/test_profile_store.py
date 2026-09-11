"""第 4 层用户持久化（Commit B）的单元测试。

覆盖：profile_store 读写（按 dataset_id 索引）、load_dataset 加载时优先覆盖、
confirm_stream_semantic_impl 写 user_confirmed、损坏文件安全降级、inspect_streams
透出 label_source / user_confirmed_overrides。不依赖真实网络。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools import profile_store
from app.tools.load_dataset import (
    confirm_stream_semantic_impl,
    load_dataset_impl,
)
from app.tools.inspect_streams import inspect_streams_impl


def test_save_and_load_profile(tmp_path: Path) -> None:
    """写入后读取，流映射与来源应一致。"""
    profile_store.save_dataset_profile(
        str(tmp_path), "ds_a",
        stream_overrides={"accel.csv": {"kind": "imu", "semantic_label": "IMU"}},
    )
    prof = profile_store.load_dataset_profile(str(tmp_path), "ds_a")
    assert prof["streams"]["accel.csv"]["kind"] == "imu"
    assert prof["streams"]["accel.csv"]["source"] == "user_confirmed"


def test_missing_profile_returns_empty(tmp_path: Path) -> None:
    """无画像文件时返回空 streams，不抛异常。"""
    prof = profile_store.load_dataset_profile(str(tmp_path), "nope")
    assert prof.get("streams", {}) == {}
    assert prof.get("pairs", []) == []


def test_corrupt_profile_degrades(tmp_path: Path) -> None:
    """损坏的画像文件安全降级为空，不抛异常。"""
    from app.tools.profile_store import _profile_path

    p = _profile_path(str(tmp_path), "x")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    prof = profile_store.load_dataset_profile(str(tmp_path), "x")
    assert prof.get("streams", {}) == {}
    assert prof.get("pairs", []) == []


def test_profile_lands_in_dataset_subdir(tmp_path: Path) -> None:
    """画像落在 outputs/by_dataset/<净名>/profile.json（子目录化）。"""
    from app.tools.output_paths import sanitize_dataset_dir_name

    profile_store.save_dataset_profile(
        str(tmp_path), "lerobot",
        stream_overrides={"a.csv": {"kind": "imu"}},
    )
    expected = (tmp_path / "by_dataset" / sanitize_dataset_dir_name("lerobot")
                / "profile.json")
    assert expected.exists()


def test_legacy_profile_migrated_on_read(tmp_path: Path) -> None:
    """旧全局画像被惰性迁移读取（跨会话确认不丢）。"""
    import json

    legacy = {
        "schema_version": 1,
        "datasets": {
            "old_ds": {"streams": {"accel.csv": {"kind": "imu",
                                                 "source": "user_confirmed"}},
                       "pairs": []},
            "other_ds": {"streams": {"x.csv": {"kind": "y"}}, "pairs": []},
        },
    }
    (tmp_path / ".dataset_profile.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    # 读取旧数据集 → 应迁移出该分片。
    prof = profile_store.load_dataset_profile(str(tmp_path), "old_ds")
    assert prof["streams"]["accel.csv"]["kind"] == "imu"

    # 且已写入新路径。
    from app.tools.output_paths import sanitize_dataset_dir_name

    new_path = (tmp_path / "by_dataset" / sanitize_dataset_dir_name("old_ds")
                / "profile.json")
    assert new_path.exists()

    # 旧文件保留不删（保守）；其它数据集分片不受影响。
    assert (tmp_path / ".dataset_profile.json").exists()
    other = profile_store.load_dataset_profile(str(tmp_path), "other_ds")
    assert other["streams"]["x.csv"]["kind"] == "y"


def test_new_path_takes_priority_over_legacy(tmp_path: Path) -> None:
    """新路径已有画像时，不再读旧文件（避免旧数据覆盖新确认）。"""
    import json

    profile_store.save_dataset_profile(
        str(tmp_path), "ds", stream_overrides={"a.csv": {"kind": "new"}},
    )
    legacy = {"schema_version": 1,
              "datasets": {"ds": {"streams": {"a.csv": {"kind": "old"}},
                                  "pairs": []}}}
    (tmp_path / ".dataset_profile.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    prof = profile_store.load_dataset_profile(str(tmp_path), "ds")
    assert prof["streams"]["a.csv"]["kind"] == "new"


def test_apply_profile_overrides_marks_source(tmp_path: Path) -> None:
    """apply_profile_overrides 按文件名覆盖并标注 label_source=user_confirmed。"""
    streams = [
        {"path": "/d/accel.csv", "kind": "unknown", "semantic_label": "未知"},
        {"path": "/d/gyro.csv", "kind": "imu", "semantic_label": "IMU"},
    ]
    profile = {"streams": {"accel.csv": {"kind": "imu", "semantic_label": "IMU(确认)"}}}
    out = profile_store.apply_profile_overrides(streams, profile)
    accel = next(s for s in out if s["path"].endswith("accel.csv"))
    assert accel["kind"] == "imu"
    assert accel["semantic_label"] == "IMU(确认)"
    assert accel["label_source"] == "user_confirmed"
    # 未覆盖流保持原样。
    gyro = next(s for s in out if s["path"].endswith("gyro.csv"))
    assert gyro["label_source"] is None


def _make_small_dataset(tmp_path: Path) -> Path:
    """构造含两表格流的小合成目录（accel.csv + gyro.csv，均 ≥3 行）。"""
    root = tmp_path / "ds_small"
    root.mkdir()
    ts = list(range(0, 3000, 1000))
    pd.DataFrame({
        "timestamp_ns": ts, "x": [1, 2, 3], "y": [1, 2, 3], "z": [1, 2, 3],
    }).to_csv(root / "accel.csv", index=False)
    pd.DataFrame({
        "timestamp_ns": ts, "x": [1, 2, 3], "y": [1, 2, 3], "z": [1, 2, 3],
    }).to_csv(root / "gyro.csv", index=False)
    return root


def test_load_applies_profile_override(tmp_path: Path) -> None:
    """加载时若画像存在 accel.csv 的 user_confirmed，应覆盖自动识别。"""
    root = _make_small_dataset(tmp_path)
    # 先写入画像（把 accel.csv 标为未知，验证覆盖优先）。
    profile_store.save_dataset_profile(
        str(tmp_path), "ds_small",
        stream_overrides={"accel.csv": {"kind": "unknown", "semantic_label": "用户标未知"}},
    )
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))
    accel = next(s for s in ctx.meta["streams"] if s["path"].endswith("accel.csv"))
    assert accel["kind"] == "unknown"  # 被 user_confirmed 覆盖
    assert accel["label_source"] == "user_confirmed"


def test_confirm_then_inspect_shows_override(tmp_path: Path) -> None:
    """confirm_stream_semantic_impl 写入后，inspect_streams 透出 user_confirmed。"""
    root = _make_small_dataset(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(root))

    res = confirm_stream_semantic_impl(
        ctx, "accel.csv", kind="imu", semantic_label="IMU(用户确认)",
        label_evidence="用户确认",
    )
    assert res["success"] is True
    assert res["mapping"]["source"] == "user_confirmed"

    # 重新加载，应用覆盖。
    ctx2 = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx2, str(root))
    insp = inspect_streams_impl(ctx2)
    assert insp["summary"]["n_user_confirmed"] == 1
    # accel.csv 被识别为 IMU，故落在 imus[].streams 中。
    all_table = [
        *insp["table_streams"],
        *[s for imu in insp["imus"] for s in imu.get("streams", [])],
    ]
    accel = next(s for s in all_table
                 if s is not None and (s.get("path") or s.get("source", "")).endswith("accel.csv"))
    assert accel["label_source"] == "user_confirmed"
    assert accel["semantic_label"] == "IMU(用户确认)"
    assert any(
        o["filename"] == "accel.csv" for o in insp["user_confirmed_overrides"]
    )


def test_confirm_requires_loaded_dataset(tmp_path: Path) -> None:
    """未加载数据集时 confirm 应返回 not_applicable。"""
    ctx = RunContext(output_dir=str(tmp_path))
    res = confirm_stream_semantic_impl(ctx, "accel.csv", kind="imu")
    assert res["success"] is False
    assert res["error"] == "no_data_loaded"
