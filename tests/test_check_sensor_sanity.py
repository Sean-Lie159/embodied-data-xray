"""app/tools/check_sensor_sanity 工具的单元测试。

覆盖：正常 IMU（m/s² 与 g 系）、重力异常、单位无法确定、NaN 注入、恒定通道、
力流饱和削顶、无 IMU/力不适用。使用合成 CSV 数据，不依赖网络。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.check_sensor_sanity import check_sensor_sanity, check_sensor_sanity_impl


def _imu_ctx(tmp_path: Path, accel_norm: float = 9.8, inject_nan: bool = False,
             constant_chan: bool = False, gyro_constant: bool = False) -> RunContext:
    """构造含 IMU 流的上下文（前 1s 静止、后 1s 运动）。"""
    t = np.arange(0, 2.0, 0.01)
    acc = np.zeros((len(t), 3))
    for i, tt in enumerate(t):
        if tt < 1.0:
            acc[i] = [0.0, 0.0, accel_norm]
        else:
            amp = 0.5 * accel_norm
            acc[i] = [amp * np.sin(10 * tt), amp * np.cos(7 * tt), accel_norm + amp * np.sin(13 * tt)]
    gyro = {
        "gyro_x": 0.05 * np.sin(t) if not gyro_constant else 0.01,
        "gyro_y": 0.05 * np.cos(t) if not gyro_constant else 0.01,
        "gyro_z": 0.05 * np.sin(2 * t) if not gyro_constant else 0.01,
    }
    df = pd.DataFrame({"accel_x": acc[:, 0], "accel_y": acc[:, 1], "accel_z": acc[:, 2], **gyro})
    if inject_nan:
        df.loc[10:30, "accel_x"] = np.nan
    if constant_chan:
        df["accel_x"] = 0.0
    p = tmp_path / "imu.csv"
    df.to_csv(p, index=False)
    meta = {
        "capabilities": {"has_imu": True},
        "streams": [{
            "path": str(p), "format": "csv", "kind": "imu",
            "channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        }],
    }
    return RunContext(dataset_id="imu", df=None, meta=meta)


def _force_ctx(tmp_path: Path, saturate: bool = False) -> RunContext:
    t = np.arange(0, 2.0, 0.01)
    fx = np.sin(2 * np.pi * t)
    if saturate:
        fx = np.clip(fx, -0.9, 0.9)
    df = pd.DataFrame({"fx": fx, "fy": 0.1 * np.sin(t), "fz": np.full(len(t), 9.8)})
    p = tmp_path / "ft.csv"
    df.to_csv(p, index=False)
    meta = {
        "capabilities": {"has_force": True},
        "streams": [{
            "path": str(p), "format": "csv", "kind": "force",
            "channels": ["fx", "fy", "fz"],
        }],
    }
    return RunContext(dataset_id="ft", df=None, meta=meta)


def _first_check(r):
    return next(iter(r["checks"]))


def test_check_sensor_sanity_is_registered() -> None:
    assert isinstance(check_sensor_sanity, FunctionTool)
    assert check_sensor_sanity.name == "check_sensor_sanity"


def test_normal_imu_ms2_pass(tmp_path: Path) -> None:
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 9.8, gyro_constant=False))
    assert r["success"] is True
    assert r["result"] == "pass"
    k = _first_check(r)
    assert r["checks"][k]["accel_unit"] == "m/s2"
    assert r["checks"][k]["gravity_check"]["verdict"] == "pass"
    assert r["checks"][k]["static_ratio"] >= 0.3


def test_normal_imu_g_pass(tmp_path: Path) -> None:
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 1.0, gyro_constant=False))
    assert r["result"] == "pass"
    k = _first_check(r)
    assert r["checks"][k]["accel_unit"] == "g"
    assert r["checks"][k]["gravity_check"]["verdict"] == "pass"


def test_gravity_anomaly_fail(tmp_path: Path) -> None:
    """静止模长 8.8（m/s² 系）偏离 9.8 超容差 → 判 fail。"""
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 8.8, gyro_constant=False))
    assert r["result"] == "fail"
    k = _first_check(r)
    assert r["checks"][k]["accel_unit"] == "m/s2"
    assert r["checks"][k]["gravity_check"]["verdict"] == "fail"


def test_unit_unresolved_skips_gravity(tmp_path: Path) -> None:
    """静止模长 5.0 无法确定单位 → 重力检查 skipped，不硬套阈值。"""
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 5.0, gyro_constant=False))
    k = _first_check(r)
    assert r["checks"][k]["accel_unit"] == "无法确定"
    assert r["checks"][k]["gravity_check"]["status"] == "skipped"


def test_nan_injection_fail(tmp_path: Path) -> None:
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 9.8, inject_nan=True))
    assert r["result"] == "fail"
    k = _first_check(r)
    assert r["checks"][k]["nan_ratio"] > 0.0


def test_constant_channel_detected(tmp_path: Path) -> None:
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 9.8, constant_chan=True))
    assert r["result"] in ("warn", "fail")
    assert "accel_x" in r["constant_channels"]


def test_force_saturation_fail(tmp_path: Path) -> None:
    r = check_sensor_sanity_impl(_force_ctx(tmp_path, saturate=True))
    assert r["result"] == "fail"
    k = _first_check(r)
    assert r["checks"][k]["saturation_ratio"] > 0.0


def test_no_sensor_streams_not_applicable(tmp_path: Path) -> None:
    ctx = RunContext(dataset_id="none", df=None,
                     meta={"capabilities": {}, "streams": []})
    r = check_sensor_sanity_impl(ctx)
    assert r["success"] is False
    assert r["error"] == "not_applicable"


def test_dataset_field_present(tmp_path: Path) -> None:
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 9.8))
    assert r["dataset"] == "imu"


def _constant_fault_ctx(tmp_path: Path) -> RunContext:
    """z 恒 9.8、x/y 恒 0（故障恒定输出），不得判健康静止。"""
    t = np.arange(0, 2.0, 0.01)
    n = len(t)
    df = pd.DataFrame({
        "accel_x": np.full(n, 0.0), "accel_y": np.full(n, 0.0), "accel_z": np.full(n, 9.8),
        "gyro_x": np.full(n, 0.0), "gyro_y": np.full(n, 0.0), "gyro_z": np.full(n, 0.0),
    })
    p = tmp_path / "fault.csv"
    df.to_csv(p, index=False)
    meta = {
        "capabilities": {"has_imu": True},
        "streams": [{
            "path": str(p), "format": "csv", "kind": "imu",
            "channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        }],
    }
    return RunContext(dataset_id="fault", df=None, meta=meta)


def test_constant_fault_not_healthy_static(tmp_path: Path) -> None:
    """恒定故障（z=9.8、x/y=0）不得判为健康静止，重力检查降级。"""
    r = check_sensor_sanity_impl(_constant_fault_ctx(tmp_path))
    k = _first_check(r)
    # 必须触发恒定通道告警（warn/fail），不得 pass。
    assert r["result"] in ("warn", "fail")
    assert r["constant_channels"]  # 恒定通道非空
    # 重力检查必须 skipped（静止段判定不可信），不得 pass。
    assert r["checks"][k]["gravity_check"]["status"] == "skipped"
    assert "恒定" in r["checks"][k]["gravity_check"]["reason"]
    # 静止占比应为 0（排除恒定段后）。
    assert r["checks"][k]["static_ratio"] == 0.0


def test_static_window_adaptive_200hz(tmp_path: Path) -> None:
    """200Hz 数据：静止段窗口应自适应为 200 样本。"""
    t = np.arange(0, 2.0, 0.005)  # 200Hz
    n = len(t)
    acc = np.zeros((n, 3))
    for i, tt in enumerate(t):
        acc[i] = [0.0, 0.0, 9.8] if tt < 1.0 else [4.9 * np.sin(10 * tt), 0, 9.8]
    df = pd.DataFrame({"accel_x": acc[:, 0], "accel_y": acc[:, 1], "accel_z": acc[:, 2],
                       "gyro_x": 0.05 * np.sin(t), "gyro_y": 0.05, "gyro_z": 0.05})
    p = tmp_path / "imu200.csv"
    df.to_csv(p, index=False)
    meta = {
        "capabilities": {"has_imu": True},
        "streams": [{
            "path": str(p), "format": "csv", "kind": "imu",
            "channels": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
            "measured_rate": {"sample_rate_hz": 200.0},
        }],
    }
    r = check_sensor_sanity_impl(RunContext(dataset_id="d", df=None, meta=meta))
    k = _first_check(r)
    assert "200 样本" in r["checks"][k]["static_window_note"]


def test_per_axis_output_present(tmp_path: Path) -> None:
    """静止段统计应输出逐轴中位数。"""
    r = check_sensor_sanity_impl(_imu_ctx(tmp_path, 9.8))
    k = _first_check(r)
    assert "per_axis_accel" in r["checks"][k]
    assert "accel_z" in r["checks"][k]["per_axis_accel"]
