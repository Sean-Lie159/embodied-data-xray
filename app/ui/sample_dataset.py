"""合成示例数据集：小型 IMU 流（含数据缺口）+ 任务表（确定性生成）。

用途（决策 5）：侧栏数据面板的**可选**一键演示，30 秒看到
"概况 → 质检（时间同步 + 缺口定位）→ 统计 → 绘图"全链路效果。

内容（docs/UI快速上手与模型配置设计.md 3.4）：
- ``imu_glove.csv``：120 Hz IMU 周期流，中段挖一段约 207 帧缺口——
  演示 ``check_temporal_sync``（locate_gaps=True）回答"缺口发生在哪"，
  即 2026-09 wujiGlove 时间对齐改造成果的展示；
- ``force_sensor.csv``：同起点同时长的完整周期流（对照流，间隔无缺口，
  会被基线推荐算法优先选为对齐基准）；
- ``tasks.csv``：20 个 episode 的任务表（success / 关节角列），演示统计与绘图。

生成是**确定性**的（固定随机种子），两次生成逐字节一致——这是回归测试的
断言点，也是"示例永远可复现"的承诺。数据不进 git（outputs/ 已排除），
本模块与 scripts/make_sample_dataset.py 进 git。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 120 Hz 周期流；纳秒 epoch 起点（2026 年 UTC 量级，保证单位推断为 ns）。
_STEP_NS = 8_330_000.0
_NS_EPOCH_0 = 1_787_294_445_650_951_302
_N_FRAMES = 6000
_GAP_AT = 3000
_GAP_SIZE = 207  # ≈1.7 s 的缺口（120 Hz），与真实 wujiGlove 右手套场景同量级
_N_EPISODES = 20


def generate_sample_frames() -> dict[str, pd.DataFrame]:
    """生成示例数据集的各表（确定性，固定种子）。

    Returns:
        {"imu_glove.csv": ..., "force_sensor.csv": ..., "tasks.csv": ...}。
    """
    ts = np.arange(_N_FRAMES, dtype=float) * _STEP_NS + _NS_EPOCH_0
    rng = np.random.default_rng(42)

    # IMU 流：周期采样 + 轻微抖动，中段挖 207 帧缺口。
    keep = np.delete(np.arange(_N_FRAMES), np.arange(_GAP_AT, _GAP_AT + _GAP_SIZE))
    imu_ts = ts[keep] + rng.uniform(-1.0e5, 1.0e5, len(keep))  # ±0.1ms 抖动
    imu = pd.DataFrame({
        "timestamp_ns": imu_ts.round().astype("int64"),
        "ax_mss2": rng.normal(0.0, 0.05, len(keep)),
        "ay_mss2": rng.normal(0.0, 0.05, len(keep)),
        "az_mss2": rng.normal(9.8, 0.08, len(keep)),  # 含重力
        "gx_radps": rng.normal(0.0, 0.01, len(keep)),
        "gy_radps": rng.normal(0.0, 0.01, len(keep)),
        "gz_radps": rng.normal(0.0, 0.01, len(keep)),
    })

    # 力传感流：同起点同终点、无缺口的完整周期流（对照/基线候选）。
    force = pd.DataFrame({
        "timestamp_ns": ts.round().astype("int64"),
        "force_n": np.abs(rng.normal(2.0, 0.4, _N_FRAMES)),
    })

    # 任务表：20 个 episode，成功率约 70%，含离群 episode（长轨迹）。
    success = (rng.uniform(0.0, 1.0, _N_EPISODES) < 0.7).astype(int)
    base_len = rng.integers(180, 320, _N_EPISODES).astype(float)
    base_len[7] = 900.0  # 人为离群：episode_08 轨迹异常长
    joint_mean = rng.normal(0.5, 0.15, _N_EPISODES).round(4)
    tasks = pd.DataFrame({
        "episode": [f"episode_{i + 1:02d}" for i in range(_N_EPISODES)],
        "success": success,
        "traj_length": base_len.astype(int),
        "joint_angle_mean_rad": joint_mean,
    })
    return {
        "imu_glove.csv": imu,
        "force_sensor.csv": force,
        "tasks.csv": tasks,
    }


def ensure_sample_dataset(target_dir: str | Path | None = None) -> Path:
    """确保示例数据集存在（不存在则生成），返回目录路径。

    已存在时直接返回（不重新生成——用户手动改动过的示例不覆盖）；
    需要重建可先删除目录。

    Args:
        target_dir: 目标目录，缺省 ``outputs/sample_dataset``。

    Returns:
        示例数据集目录路径。
    """
    from app.config.settings import get_settings

    d = Path(target_dir) if target_dir else Path(get_settings().output_dir) / "sample_dataset"
    if (d / "tasks.csv").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    for name, df in generate_sample_frames().items():
        df.to_csv(d / name, index=False)
    return d
