"""app/tools/_sniffing 列名嗅探的单元测试。

覆盖位姿整词匹配的正反例、关节状态排除、能力标签的命中列名清单。
"""

from __future__ import annotations

from app.tools._sniffing import sniff_table_columns


def _pose_present(columns: list[str]) -> bool:
    return sniff_table_columns(columns)["has_pose"]["present"]


def _pose_columns(columns: list[str]) -> list[str]:
    return sniff_table_columns(columns)["has_pose"]["columns"]


def test_joint_state_not_marked_as_pose() -> None:
    """qpos* 应判为状态/动作，不得命中位姿。"""
    assert _pose_present(["qpos1", "qpos2", "qpos3"]) is False
    assert _pose_present(["qvel2"]) is False
    assert _pose_present(["qacc0"]) is False
    assert _pose_present(["joint1", "joint2"]) is False


def test_pose_columns_detected() -> None:
    """ee_*、*_pose、position 等应判为位姿。"""
    assert _pose_present(["ee_pos_x", "ee_pos_y"]) is True
    assert _pose_present(["tcp_pose"]) is True
    assert _pose_present(["position"]) is True
    assert _pose_present(["pose"]) is True


def test_unrelated_column_not_detected() -> None:
    """与位姿无关的列名不得误判。"""
    assert _pose_present(["week", "days"]) is False
    assert _pose_present(["col0", "col1"]) is False
    assert _pose_present(["temperature", "humidity"]) is False


def test_pose_columns_attached_as_evidence() -> None:
    """位姿能力标签应附带命中列名清单作为推测依据。"""
    result = sniff_table_columns(["ee_pos_x", "ee_pos_y", "temperature"])
    assert result["has_pose"]["columns"] == ["ee_pos_x", "ee_pos_y"]
    # 未命中的列不应出现在依据里。
    assert "temperature" not in result["has_pose"]["columns"]


def test_action_columns_attached_as_evidence() -> None:
    """状态/动作能力标签应附带命中列名清单。"""
    result = sniff_table_columns(["qpos1", "qpos2", "obs"])
    assert result["has_actions"]["present"] is True
    assert set(result["has_actions"]["columns"]) == {"qpos1", "qpos2", "obs"}


def test_imu_axes_and_evidence() -> None:
    """IMU 推断：accel+gyro → 6 轴，并附带命中列名。"""
    result = sniff_table_columns(["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
    assert result["has_imu"]["present"] is True
    assert result["has_imu"]["confidence"] == "high"
    assert result["imu_axes"] == 6
    assert result["has_imu"]["columns"] == [
        "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    ]


def test_imu_with_mag_is_9_axes() -> None:
    """accel+gyro+mag → 9 轴。"""
    result = sniff_table_columns(
        ["accel_x", "gyro_x", "mag_x", "mag_y", "mag_z"]
    )
    assert result["has_imu"]["present"] is True
    assert result["imu_axes"] == 9
