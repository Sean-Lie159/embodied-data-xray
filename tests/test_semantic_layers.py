"""四层语义识别架构（Commit A：语义识别内核）的单元测试。

覆盖第 2 层内容指纹（时间戳/四元数/标定/力）、流配对规则（mp4↔metainfo、
accel+gyro=六轴 IMU）、空流检测、已知误判修正（camera_params 标定、
hand_tracking 不标 IMU、无力不报力），以及第 1 层词典降级。命名尽量贴合真实
数据 `20260821_144044` 的文件特征。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.tools._sniffing import (
    classify_table_stream,
    detect_quaternion_groups,
    fingerprint_calibration,
    fingerprint_timestamp,
    infer_role,
    pair_streams,
)


def _ts_sample(col: str, n: int = 10) -> pd.DataFrame:
    """生成单调递增时间戳样本（间隔均匀）用于指纹测试。"""
    vals = np.arange(0, n, dtype="int64") * 1000 + 1787294445000000000
    return pd.DataFrame({col: vals})


def test_timestamp_fingerprint_pts_us() -> None:
    """pts_us 单调递增、差分均匀 → 时间戳指纹命中。"""
    df = _ts_sample("pts_us")
    res = fingerprint_timestamp(df, ["pts_us"])
    assert res["present"] is True
    assert res["column"] == "pts_us"
    assert "单调递增" in res["evidence"]


def test_timestamp_fingerprint_exposure_start_utc_ns() -> None:
    """exposure_start_utc_ns 命名命中且数值单调递增 → 时间戳指纹。"""
    df = _ts_sample("exposure_start_utc_ns")
    res = fingerprint_timestamp(df, ["exposure_start_utc_ns"])
    assert res["present"] is True
    assert res["column"] == "exposure_start_utc_ns"


def test_timestamp_fingerprint_rejects_non_monotonic() -> None:
    """非单调递增列不命中时间戳指纹。"""
    df = pd.DataFrame({"x": [3, 1, 2, 0, 4]})
    res = fingerprint_timestamp(df, ["x"])
    assert res["present"] is False


def test_quaternion_fingerprint_modulus_one() -> None:
    """四列成组且每行模长≈1 → 四元数指纹命中。"""
    n = 8
    # 构造归一化四元数（每行为单位四元数）。
    q = np.zeros((n, 4))
    q[:, 0] = 1.0  # (1,0,0,0) 单位四元数
    df = pd.DataFrame({
        "left_joint0_orientation_x": q[:, 0],
        "left_joint0_orientation_y": q[:, 1],
        "left_joint0_orientation_z": q[:, 2],
        "left_joint0_orientation_w": q[:, 3],
    })
    groups = detect_quaternion_groups(df, list(df.columns))
    assert len(groups) == 1
    assert groups[0]["prefix"] == "left_joint0_orientation"
    assert "模长≈1" in groups[0]["evidence"]


def test_calibration_fingerprint_camera_params_rgb() -> None:
    """camera_params_rgb.json 含 intrinsics/extrinsics → 标定指纹命中。"""
    obj = {
        "group": "rgb",
        "cameras": [{"eye": "left", "intrinsics": {"focalX": 560.0,
                   "centerX": 965.0}, "extrinsics": {"rotation": [0, 0, 0, 1]}}],
    }
    res = fingerprint_calibration(obj)
    assert res["present"] is True
    assert "intrinsics" in res["keys_found"] or "extrinsics" in res["keys_found"]


def test_calibration_fingerprint_imu_calibration() -> None:
    """imu_calibration.json 含 bias/noise → 标定指纹命中。"""
    obj = {"device_uid": "x", "imu": {"bias": {"accelerometer_mps2": [0, 0, 0]},
            "noise": {"accel_noise_std_mps2": [0.1, 0.1, 0.1]}}}
    res = fingerprint_calibration(obj)
    assert res["present"] is True
    assert "bias" in res["keys_found"]


def test_calibration_fingerprint_rejects_plain_json() -> None:
    """纯业务 JSON（无标定键）不命中标定指纹。"""
    res = fingerprint_calibration({"note": "hello", "value": 1})
    assert res["present"] is False


def test_pair_streams_video_metainfo() -> None:
    """<name>.mp4 ↔ <name>_metainfo.csv 配对。"""
    videos = ["/d/rgb.mp4", "/d/ctrl.mp4"]
    tables = ["/d/rgb_metainfo.csv", "/d/ctrl_metainfo.csv", "/d/accel.csv"]
    pairs = pair_streams(videos, tables, [])
    media_pairs = [p for p in pairs if p["type"] == "media_metainfo"]
    assert len(media_pairs) == 2
    assert media_pairs[0]["media_kind"] == "video"
    assert media_pairs[0]["metainfo"].endswith("rgb_metainfo.csv")


def test_pair_streams_imu_6axis() -> None:
    """accel + gyro 同目录 → 六轴 IMU 配对。"""
    tables = ["/d/accel.csv", "/d/gyro.csv"]
    pairs = pair_streams([], tables, [])
    imu_pairs = [p for p in pairs if p["type"] == "imu_6axis"]
    assert len(imu_pairs) == 1
    assert imu_pairs[0]["imu_axes"] == 6
    assert {Path(p).name for p in imu_pairs[0]["streams"]} == {"accel.csv", "gyro.csv"}


def test_empty_stream_detection() -> None:
    """行数 ≤ 2 的表标记为空流，不计入活跃流。"""
    res = classify_table_stream("controller_poses.csv", ["pose_x", "pose_y"], None, 1)
    assert res["status"] == "empty"
    assert res["semantic_label"] == "未使用/空流"
    assert res["kind"] == "unknown"


def test_hand_tracking_not_imu() -> None:
    """hand_tracking 不得标为 IMU，应识别为手部跟踪。"""
    cols = [
        "frame_number", "left_active", "left_joint0_id", "left_joint0_name",
        "left_joint0_orientation_x", "left_joint0_orientation_y",
        "left_joint0_orientation_z", "left_joint0_orientation_w",
    ]
    # 四元数样本（单位四元数）。
    df = pd.DataFrame({
        "frame_number": [0, 1, 2],
        "left_active": [1, 1, 1],
        "left_joint0_id": [0, 0, 0],
        "left_joint0_name": ["PALM", "PALM", "PALM"],
        "left_joint0_orientation_x": [1.0, 1.0, 1.0],
        "left_joint0_orientation_y": [0.0, 0.0, 0.0],
        "left_joint0_orientation_z": [0.0, 0.0, 0.0],
        "left_joint0_orientation_w": [0.0, 0.0, 0.0],
    })
    res = classify_table_stream("hand_tracking.csv", cols, df, 3)
    assert res["kind"] == "hand_tracking"
    assert res["kind"] != "imu"
    assert "手部跟踪" in res["semantic_label"]


def test_no_force_without_evidence() -> None:
    """列名似力/力矩但数值退化（无方差）→ 不报力。"""
    # 力列恒值 → 无数值信号，应降级不报力。
    df = pd.DataFrame({
        "fx": [0.0, 0.0, 0.0], "fy": [0.0, 0.0, 0.0],
        "timestamp": [1000, 2000, 3000],
    })
    res = classify_table_stream("head_pose.csv", ["fx", "fy", "timestamp"], df, 3)
    # 该表无 accel/gyro、无四元数、无 hand_tracking，force 未获佐证 → 不报力。
    assert res["kind"] != "force"


def test_camera_params_role_is_calibration() -> None:
    """camera_params_*.json 的 infer_role 应为"标定文件"，而非 RGB 相机。"""
    r = infer_role("camera_params_rgb.json")
    assert r["role"] == "标定文件"


def test_classify_accel_gyro_as_imu() -> None:
    """accel.csv / gyro.csv 含 accel/gyro 列 + 时间戳指纹 → 识别为 IMU。"""
    df = pd.DataFrame({
        "timestamp_ns": _ts_sample("timestamp_ns")["timestamp_ns"],
        "x": np.ones(10), "y": np.ones(10), "z": np.ones(10),
    })
    res = classify_table_stream("accel.csv", ["timestamp_ns", "x", "y", "z"], df, 10)
    assert res["kind"] == "imu"
    assert res["timestamp_column"] == "timestamp_ns"


def test_chinese_filename_image_not_misclassified() -> None:
    """中文文件名图片（信息.jpg）不影响分类逻辑。"""
    r = infer_role("信息.jpg")
    # 图片扩展名无语义角色，但不应误判为标定/相机传感器。
    assert r["role"] != "标定文件"
