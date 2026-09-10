"""合成 MCAP 生成器（测试辅助代码，非 agent 工具）。

用 `mcap` 包写入**确定性的** JSON 编码 MCAP 文件，供原型验证与回归测试使用。
只依赖 `mcap`（写入器），不依赖 ROS 环境。

生成的形态刻意贴近真实场景（参考 wujiGlove 导出形态）：
- topic ``/imu``：JSON 编码，消息体内含 ``header.timestamp_us``（嵌套传感器时间，
  与容器 ``log_time`` 构成「双时钟」）+ ``orientation``（四元数）；
- topic ``/joint_states``：JSON 编码，含 ``position`` 数值数组（actions 类）；
- topic ``/tf_static``：JSON 编码，仅少量消息（静态流）。

用法（测试内调用）::

    from tests.fixtures.make_mcap import build_mcap
    p = build_mcap(tmp_path / "demo.mcap")
"""

from __future__ import annotations

import json
from pathlib import Path

# 固定基准时刻（纳秒 epoch），保证生成确定性。
_BASE_NS = 1_787_294_456_000_000_000


def build_mcap(
    dest: Path,
    n_imu: int = 100,
    n_joint: int = 50,
    n_tf_static: int = 3,
    topic_prefix: str = "",
) -> Path:
    """生成一个 JSON 编码的合成 MCAP 文件。

    Args:
        dest: 目标文件路径（父目录会自动创建）。
        n_imu: /imu topic 的消息条数。
        n_joint: /joint_states topic 的消息条数。
        n_tf_static: /tf_static topic 的消息条数（静态流，少量）。
        topic_prefix: 可选，topic 名统一前缀（用于多文件场景消歧）。

    Returns:
        生成的文件路径。
    """
    from mcap.writer import Writer  # 延迟导入：仅在生成夹具时需要

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        w = Writer(f)
        w.start()
        # JSON 消息：schema encoding 用 jsonschema（自描述）；data 可留空。
        sid = w.register_schema(name="json_schema", encoding="jsonschema", data=b"")

        imu_ch = w.register_channel(
            topic=f"{topic_prefix}/imu", message_encoding="json", schema_id=sid,
        )
        joint_ch = w.register_channel(
            topic=f"{topic_prefix}/joint_states", message_encoding="json",
            schema_id=sid,
        )
        tf_ch = w.register_channel(
            topic=f"{topic_prefix}/tf_static", message_encoding="json", schema_id=sid,
        )

        for i in range(n_imu):
            log_ns = _BASE_NS + i * 1_000_000  # 1 kHz（容器口径）
            payload = {
                "header": {"timestamp_us": (log_ns // 1000) + 10},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
            }
            w.add_message(
                channel_id=imu_ch, log_time=log_ns, publish_time=log_ns - 1000,
                data=json.dumps(payload).encode(),
            )

        for i in range(n_joint):
            log_ns = _BASE_NS + i * 2_000_000  # 500 Hz
            payload = {
                "header": {"timestamp_us": (log_ns // 1000) + 5},
                "name": ["j0", "j1", "j2"],
                "position": [0.1 * i, 0.2 * i, 0.3 * i],
                "velocity": [0.0, 0.0, 0.0],
            }
            w.add_message(
                channel_id=joint_ch, log_time=log_ns, publish_time=log_ns,
                data=json.dumps(payload).encode(),
            )

        for i in range(n_tf_static):
            log_ns = _BASE_NS + i * 1_000_000
            payload = {
                "transforms": [{
                    "parent_frame_id": "base",
                    "child_frame_id": f"cam{i}",
                }],
            }
            w.add_message(
                channel_id=tf_ch, log_time=log_ns, publish_time=log_ns,
                data=json.dumps(payload).encode(),
            )

        w.finish()
    return dest


def build_mcap_with_cdr_topic(dest: Path, n_json: int = 10) -> Path:
    """生成含一个「非 JSON 编码」topic 的 MCAP（验证不硬解、如实标注）。

    用 protobuf 编码标记一个 topic（不实际写合规 protobuf 载荷）——
    目的是让 probe 判定 decodable=False 并给出 decode_note，验证诚实降级。
    """
    from mcap.writer import Writer

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="json_schema", encoding="jsonschema", data=b"")
        json_ch = w.register_channel(
            topic="/imu", message_encoding="json", schema_id=sid,
        )
        cdr_ch = w.register_channel(
            topic="/scan", message_encoding="cdr", schema_id=0,
        )
        for i in range(n_json):
            log_ns = _BASE_NS + i * 1_000_000
            w.add_message(
                channel_id=json_ch, log_time=log_ns, publish_time=log_ns,
                data=json.dumps({"v": i}).encode(),
            )
        # cdr topic 写入一条占位消息（内容不解码）。
        w.add_message(
            channel_id=cdr_ch, log_time=_BASE_NS, publish_time=_BASE_NS,
            data=b"\x00\x01\x00\x00",
        )
        w.finish()
    return dest
