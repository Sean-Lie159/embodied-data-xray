"""app/tools/inspect_streams 工具的单元测试。

验证设备清单生成：未加载数据兜底、基于流登记表按需实测多文件采样率、缓存生效、
力通道、标定、时钟来源与角色组合推断。不依赖真实网络。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools._sniffing import infer_role
from app.tools.inspect_streams import inspect_streams, inspect_streams_impl


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_inspect_streams_is_registered_as_function_tool() -> None:
    assert isinstance(inspect_streams, FunctionTool)
    assert inspect_streams.name == "inspect_streams"


def test_unloaded_returns_not_applicable() -> None:
    ctx = RunContext()  # 未加载
    result = inspect_streams_impl(ctx)
    assert result["success"] is False
    assert result["error"] == "no_data_loaded"


def test_multi_file_streams_all_measured(tmp_path: Path) -> None:
    """主表 + 独立 IMU + 独立力矩文件，验证所有流采样率都能实测。"""
    ts = np.arange(0, 1.0, 0.01)  # 100Hz
    imu_path = _write_csv(
        tmp_path / "imu.csv",
        pd.DataFrame({
            "timestamp": ts,
            "accel_x": np.sin(ts), "accel_y": 0, "accel_z": 9.8,
            "gyro_x": 0, "gyro_y": 0, "gyro_z": 0,
        }),
    )
    ft_ts = np.arange(0, 1.0, 0.005)  # 200Hz
    ft_path = _write_csv(
        tmp_path / "force_torque.csv",
        pd.DataFrame({
            "timestamp": ft_ts, "fx": np.sin(ft_ts), "fy": 0, "fz": 9.8,
            "tx": 0, "ty": 0, "tz": 0,
        }),
    )

    meta = {
        "capabilities": {
            "has_imu": True, "imu_axes": 6,
            "imu_channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
            "has_force": True,
            "force_channels": ["fx", "fy", "fz", "tx", "ty", "tz"],
        },
        "streams": [
            {"path": str(imu_path), "format": "csv", "kind": "imu",
             "channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
             "role": infer_role(str(imu_path))},
            {"path": str(ft_path), "format": "csv", "kind": "force",
             "channels": ["fx", "fy", "fz", "tx", "ty", "tz"],
             "role": infer_role(str(ft_path))},
        ],
    }
    ctx = RunContext(dataset_id="sensors", df=pd.DataFrame({"a": [1]}), meta=meta)

    result = inspect_streams_impl(ctx)

    # IMU 100Hz 实测。
    assert result["imus"][0]["streams"][0]["sample_rate"]["present"] is True
    assert result["imus"][0]["streams"][0]["sample_rate"]["sample_rate_hz"] == 100.0
    # 力/力矩 200Hz 实测（修复的直接证据：不再是 unknown）。
    fc = result["force_channels"]
    assert fc["present"] is True
    assert fc["sample_rate"]["present"] is True
    assert fc["sample_rate"]["sample_rate_hz"] == 200.0


def test_single_stream_failure_does_not_affect_others(tmp_path: Path) -> None:
    """单条流失败（无时间戳列，非单调/量级不符）标 unknown，不影响其他流测量。"""
    # 列非单调递增、量级不符时间单位 → 词表与内容指纹都无法识别为时间戳。
    bad_path = _write_csv(
        tmp_path / "no_ts.csv",
        pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]}),
    )
    ts = np.arange(0, 1.0, 0.01)
    good_path = _write_csv(
        tmp_path / "imu.csv",
        pd.DataFrame({"timestamp": ts, "accel_x": ts, "gyro_x": ts}),
    )
    meta = {
        "capabilities": {"has_imu": True, "has_force": True},
        "streams": [
            {"path": str(good_path), "format": "csv", "kind": "imu",
             "channels": ["accel_x", "gyro_x"], "role": infer_role(str(good_path))},
            {"path": str(bad_path), "format": "csv", "kind": "force",
             "channels": ["a", "b"], "role": infer_role(str(bad_path))},
        ],
    }
    ctx = RunContext(dataset_id="mixed", df=pd.DataFrame({"a": [1]}), meta=meta)

    result = inspect_streams_impl(ctx)

    # 好流正常实测。
    assert result["imus"][0]["streams"][0]["sample_rate"]["present"] is True
    # 坏流标 unknown 且不影响整体。
    assert result["force_channels"]["sample_rate"]["present"] is False
    assert result["force_channels"]["sample_rate"]["reason"]


def test_measured_rate_cached_no_repeat_read(tmp_path: Path) -> None:
    """第二次调用应命中缓存，不再读盘（更快且结果一致）。"""
    ts = np.arange(0, 1.0, 0.01)
    imu_path = _write_csv(
        tmp_path / "imu.csv",
        pd.DataFrame({"timestamp": ts, "accel_x": ts, "gyro_x": ts}),
    )
    meta = {
        "capabilities": {"has_imu": True},
        "streams": [
            {"path": str(imu_path), "format": "csv", "kind": "imu",
             "channels": ["accel_x", "gyro_x"], "role": infer_role(str(imu_path))},
        ],
    }
    ctx = RunContext(dataset_id="imu", df=pd.DataFrame({"a": [1]}), meta=meta)

    r1 = inspect_streams_impl(ctx)
    # 第一次后，meta["streams"][0] 应已缓存 measured_rate。
    assert "measured_rate" in ctx.meta["streams"][0]
    assert ctx.meta["streams"][0]["measured_rate"]["present"] is True

    # 第二次：缓存命中，结果一致。
    r2 = inspect_streams_impl(ctx)
    assert r2["imus"][0]["streams"][0]["sample_rate"] == \
        r1["imus"][0]["streams"][0]["sample_rate"]


def test_force_channels_reported() -> None:
    ctx = RunContext(
        dataset_id="force_demo",
        df=pd.DataFrame({"fx": [0.1], "fy": [0.2], "fz": [9.8]}),
        meta={"capabilities": {"has_force": True, "force_channels": ["fx", "fy", "fz"]}},
    )
    result = inspect_streams_impl(ctx)
    fc = result["force_channels"]
    assert fc["present"] is False  # 无登记流时力通道缺省
    assert fc["n_channels"] == 0


def test_calibration_present_and_clock_source_unknown() -> None:
    ctx = RunContext(
        dataset_id="demo",
        df=pd.DataFrame({"a": [1]}),
        meta={"capabilities": {"has_calibration": True}},
    )
    result = inspect_streams_impl(ctx)
    assert result["calibration"]["present"] is True
    assert result["clock_source"] == "unknown"  # 无声明 → unknown


def test_clock_source_declared_in_meta() -> None:
    ctx = RunContext(
        dataset_id="demo",
        df=pd.DataFrame({"a": [1]}),
        meta={"clock_source": "unified", "capabilities": {}},
    )
    result = inspect_streams_impl(ctx)
    assert result["clock_source"] == "unified"


def test_video_ffprobe_degraded_reports_unknown() -> None:
    ctx = RunContext(
        dataset_id="demo",
        df=pd.DataFrame({"a": [1]}),
        meta={
            "capabilities": {"has_video_streams": True},
            "video_meta": [
                {"file": "cam.mp4", "ffprobe_available": False,
                 "user_message": "未检测到 ffprobe，已跳过视频元数据嗅探。"}
            ],
        },
    )
    result = inspect_streams_impl(ctx)
    vs = result["video_streams"][0]
    assert vs["status"] == "unknown"
    assert "ffprobe" in vs["reason"]


def test_video_nb_frames_one_estimated_and_labeled() -> None:
    """nb_frames=1 明显错误 → 改用 duration×fps 估算，并标注来源与'估算'字样。"""
    ctx = RunContext(
        dataset_id="demo",
        df=pd.DataFrame({"a": [1]}),
        meta={
            "capabilities": {"has_video_streams": True},
            "video_meta": [
                {
                    "file": "ctrl.mp4",
                    "ffprobe_available": True,
                    "fps": 30.0,
                    "nb_frames": 1,            # 明显错误
                    "duration": "10.0",        # 10s × 30fps ≈ 300 帧
                    "width": 640, "height": 480,
                    "codec": "h264",
                    "nb_frames_source": "estimated",
                    "nb_frames_basis": "ffprobe 返回 nb_frames=1（明显错误）；改用 duration(10.0s)×fps(30.0) 估算帧数≈300（估算值，非实测）",
                    "nb_frames_trusted": False,
                }
            ],
        },
    )
    result = inspect_streams_impl(ctx)
    vs = result["video_streams"][0]
    assert vs["nb_frames_source"] == "estimated"
    # 估算值展示必须带"估算"字样，不得伪装成实测。
    assert "估算" in str(vs["nb_frames"])
    assert vs["nb_frames_basis"]  # 透出估算依据
    assert "30.0" in vs["nb_frames_basis"]


def test_video_nb_frames_trusted_uses_probe() -> None:
    """nb_frames 可信（与 duration×fps 一致）→ 来源标 probe，无'估算'字样。"""
    ctx = RunContext(
        dataset_id="demo",
        df=pd.DataFrame({"a": [1]}),
        meta={
            "capabilities": {"has_video_streams": True},
            "video_meta": [
                {
                    "file": "rgb.mp4",
                    "ffprobe_available": True,
                    "fps": 30.0,
                    "nb_frames": 300,
                    "duration": "10.0",
                    "width": 640, "height": 480,
                    "codec": "h264",
                    "nb_frames_source": "probe",
                    "nb_frames_basis": "ffprobe 实测帧数与 duration×fps 推算值一致（可信）",
                    "nb_frames_trusted": True,
                }
            ],
        },
    )
    result = inspect_streams_impl(ctx)
    vs = result["video_streams"][0]
    assert vs["nb_frames_source"] == "probe"
    assert vs["nb_frames"] == 300
    assert "估算" not in str(vs["nb_frames"])


# --- 角色组合推断（修复2） ---------------------------------------------

def test_role_combined_left_wrist() -> None:
    r = infer_role("left_wrist_cam.mp4")
    assert r["role"] == "腕部相机（左）"
    assert r["confidence"] == "high"


def test_role_combined_head_rgb_left() -> None:
    r = infer_role("head_rgb_left.mp4")
    # 主角色取置信度最高的（头部相机 0.7 > RGB 相机 0.6），方位组合。
    assert r["role"] == "头部相机（左）"


def test_role_single_wrist_cam() -> None:
    r = infer_role("wrist_cam.mp4")
    assert r["role"] == "腕部相机"


def test_role_imu_sensor() -> None:
    r = infer_role("imu_data.csv")
    assert r["role"] == "IMU 传感器"


def test_role_unknown() -> None:
    r = infer_role("randomfile.txt")
    assert r["role"] == "unknown"
    assert r["confidence"] == "low"
