"""check_sensor_sanity 非主表 IMU（补齐盲区）的单元测试。

背景：多表支持后 accel/gyro 作为非主表注册在流注册表，且真实数据列为通用 x/y/z
（非 accel_x），旧逻辑因列名不含 "accel" 而误跳过 IMU 数值列检查。本测试验证：
1) 主表非 IMU、IMU 为非主表（x/y/z 通用列）时，重力/零漂/饱和检查**实际执行**而非跳过；
2) accel+gyro 六轴配对（stream_pairs）路径；
3) 显式 table 参数检查指定表。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools.check_sensor_sanity import check_sensor_sanity_impl


def _write_csv(root: Path, name: str, df: pd.DataFrame) -> Path:
    p = root / name
    df.to_csv(p, index=False)
    return p


def _nonmain_imu_ctx(
    tmp_path: Path,
    accel_norm: float = 9.8,
    include_pair: bool = True,
) -> RunContext:
    """构造：主表 state.csv（动作，非 IMU）+ accel.csv/gyro.csv（x/y/z 通用列 IMU）。"""
    root = tmp_path / "ds"
    root.mkdir()
    # 主表（动作列）——不是 IMU。
    _write_csv(root, "state.csv", pd.DataFrame({
        "episode": [0, 0, 1, 1], "qpos1": [0.1, 0.2, 0.3, 0.4],
    }))
    # accel：x/y/z 通用列，前 1s 静止（z≈重力）、后 1s 运动。
    t = np.arange(0, 2.0, 0.01)
    n = len(t)
    acc = np.zeros((n, 3))
    for i, tt in enumerate(t):
        if tt < 1.0:
            acc[i] = [0.0, 0.0, accel_norm]
        else:
            amp = 0.5 * accel_norm
            acc[i] = [amp * np.sin(10 * tt), amp * np.cos(7 * tt), accel_norm + amp * np.sin(13 * tt)]
    accel_path = _write_csv(root, "accel.csv", pd.DataFrame({
        "timestamp_ns": np.arange(n) * 987_000,
        "x": acc[:, 0], "y": acc[:, 1], "z": acc[:, 2],
    }))
    gyro_path = _write_csv(root, "gyro.csv", pd.DataFrame({
        "timestamp_ns": np.arange(n) * 987_000,
        "x": 0.05 * np.sin(t), "y": 0.05 * np.cos(t), "z": 0.05 * np.sin(2 * t),
    }))

    accel_s = {"path": str(accel_path), "format": "csv", "kind": "imu",
               "channels": ["timestamp_ns", "x", "y", "z"]}
    gyro_s = {"path": str(gyro_path), "format": "csv", "kind": "imu",
              "channels": ["timestamp_ns", "x", "y", "z"]}
    meta: dict = {
        "capabilities": {"has_imu": True, "has_actions": True},
        "main_table": {"name": "state.csv"},
        "streams": [accel_s, gyro_s],
        "stream_pairs": (
            [{"type": "imu_6axis", "streams": [str(accel_path), str(gyro_path)], "imu_axes": 6}]
            if include_pair else []
        ),
    }
    # context.df 是主表 state.csv（非 IMU）。
    ctx = RunContext(
        dataset_id="ds",
        df=pd.DataFrame({"episode": [0, 0, 1, 1], "qpos1": [0.1, 0.2, 0.3, 0.4]}),
        meta=meta,
    )
    return ctx


def test_nonmain_imu_pair_executes_not_skipped(tmp_path: Path) -> None:
    """主表非 IMU + accel/gyro 六轴配对：重力/零漂/饱和实际执行（非跳过）。"""
    ctx = _nonmain_imu_ctx(tmp_path, 9.8)
    r = check_sensor_sanity_impl(ctx)
    assert r["success"] is True
    # 配对单元 key 存在且为 done（非 skipped）。
    pair_key = next(k for k in r["checks"] if "accel" in k and "gyro" in k)
    chk = r["checks"][pair_key]
    assert chk["type"] == "imu"
    assert chk["accel_unit"] == "m/s2"
    # 重力检查实际执行并 pass。
    assert chk["gravity_check"]["status"] == "done"
    assert chk["gravity_check"]["verdict"] == "pass"
    # 陀螺仪零漂检查实际执行。
    assert chk["gyro_check"]["status"] == "done"
    assert chk["gyro_check"]["verdict"] == "pass"
    # 饱和检查实际执行。
    assert chk["saturation_check"]["verdict"] == "pass"
    # 检查项注明作用表名（accel + gyro）。
    assert "accel.csv" in chk["table_name"]
    assert "gyro.csv" in chk["table_name"]
    # 无跳过原因（IMU 数值列检查未被跳过）。
    assert not any("无法读取 IMU 数值列" in str(s.get("reason", "")) for s in r["skipped_checks"].values())


def test_nonmain_imu_gravity_anomaly_fail(tmp_path: Path) -> None:
    """静止模长 8.8（m/s²）偏离 9.8 → 配对单元重力检查判 fail。"""
    ctx = _nonmain_imu_ctx(tmp_path, 8.8)
    r = check_sensor_sanity_impl(ctx)
    pair_key = next(k for k in r["checks"] if "accel" in k and "gyro" in k)
    gc = r["checks"][pair_key]["gravity_check"]
    assert gc["status"] == "done"
    assert gc["verdict"] == "fail"
    assert r["result"] == "fail"


def test_nonmain_imu_without_pair_still_checks(tmp_path: Path) -> None:
    """无 imu_6axis 配对时，accel.csv（文件名含 accel + x/y/z）仍应被检查。"""
    ctx = _nonmain_imu_ctx(tmp_path, 9.8, include_pair=False)
    r = check_sensor_sanity_impl(ctx)
    assert r["success"] is True
    # accel.csv 应作为单个 IMU 流被检查（文件名归属 x/y/z → accel），重力 pass。
    accel_key = next(k for k in r["checks"] if k.endswith("accel.csv"))
    chk = r["checks"][accel_key]
    assert chk["accel_unit"] == "m/s2"
    assert chk["gravity_check"]["verdict"] == "pass"


def test_explicit_table_accel(tmp_path: Path) -> None:
    """显式 table="accel.csv" → 仅检查 accel 表（重力/零漂执行）。"""
    ctx = _nonmain_imu_ctx(tmp_path, 9.8)
    r = check_sensor_sanity_impl(ctx, table="accel.csv")
    assert r["success"] is True
    accel_key = next(k for k in r["checks"] if k.endswith("accel.csv"))
    chk = r["checks"][accel_key]
    assert chk["table_name"] == "accel.csv"
    assert chk["gravity_check"]["status"] == "done"
    assert chk["gravity_check"]["verdict"] == "pass"


def test_explicit_table_not_found(tmp_path: Path) -> None:
    """显式 table 不存在 → 无 IMU 检查单元，返回成功但无检查项（结构化安全）。"""
    ctx = _nonmain_imu_ctx(tmp_path, 9.8)
    r = check_sensor_sanity_impl(ctx, table="missing.csv")
    # 找不到该表：不抛异常；由于无 imu/force 单元被构建，视为无检查结果。
    assert r["success"] is True  # 不适用判定被跳过，返回成功空检查
    assert r["checks"] == {} or "missing" not in str(r)
