"""能力标签注册表：给 ``capabilities`` 的每个键配中文名与白话说明。

设计依据：``docs/指标单一来源与数据集画像设计.md`` 第二部分（B-3.1）。

**为什么需要（真实缺陷）**：报告里的"模态矩阵"是**硬编码的六行**
（``generate_report.py`` 初版）：

```python
rows = [
    ("视频流", caps.get("has_video_streams")),
    ("IMU", caps.get("has_imu")),
    ("力/力矩", caps.get("has_force")),
    ("标定", caps.get("has_calibration")),
    ("状态/动作", caps.get("has_actions")),
    ("语言标注", caps.get("has_language")),
]
```

而 ``capabilities`` 实际有 **10 个键**——漏掉了 ``has_pose``、``has_audio``、
``has_hand_tracking``、``imu_axes``、``dataset_format``。后果：在一份
**手套位姿追踪**数据上（核心能力 ``has_pose=True``），矩阵显示出
"IMU ✗ / 力 ✗ / 标定 ✗"却**唯独没有"位姿"这一行**——用户看到的是
"一堆没有"，最关键的信息反而消失了。

**修法**：矩阵改为**从 capabilities 动态生成**，中文名从本注册表查。
新增能力标签时若忘配中文名，守护测试
（``test_every_capability_has_chinese_label``）会失败——与质检的
``_RULE_META`` 守护同构。

本模块不含逻辑，只有数据；不 import streamlit。
"""

from __future__ import annotations

from typing import Any

# 能力标签注册表：键名 → 展示元信息。
#
# label     —— 矩阵行名（中文，给人看）
# what      —— 这项能力指什么（一句话）
# is_detail —— True 表示"附注型"信息，不单独占矩阵一行（如轴数、格式名）
# absent_note —— 值为 False/None 时的补充说明（可选，避免用户误读"没有=缺陷"）
# true_note —— 值为 True 时的补充说明（可选）
_CAPABILITY_META: dict[str, dict[str, Any]] = {
    "has_video_streams": {
        "label": "视频流",
        "what": "含摄像头视频（RGB/深度等）",
        "absent_note": "纯信号数据集通常无视频，不是缺陷",
    },
    "has_audio": {
        "label": "音频",
        "what": "含麦克风或音频轨",
    },
    "has_imu": {
        "label": "IMU",
        "what": "含惯性测量单元（加速度/角速度）",
    },
    "imu_axes": {
        "label": "IMU 轴向",
        "what": "IMU 的通道轴向数",
        "is_detail": True,
    },
    "has_force": {
        "label": "力/力矩",
        "what": "含力或力矩传感通道",
    },
    "has_calibration": {
        "label": "标定",
        "what": "含标定文件或内外参",
        "absent_note": "无标定文件不代表数据不可用",
    },
    "has_actions": {
        "label": "状态/动作",
        "what": "含可作为动作学习信号的状态/动作通道",
    },
    "has_hand_tracking": {
        "label": "手部追踪",
        "what": "含手部或手指关节追踪数据",
    },
    "has_pose": {
        "label": "位姿",
        "what": "含空间位姿（位置或朝向四元数）",
    },
    "dataset_format": {
        "label": "数据集格式",
        "what": "匹配到的公开数据格式",
        "is_detail": True,
    },
}


def capability_label(key: str) -> str:
    """取能力标签的中文名；未登记时**原样返回键名**（不猜中文名）。

    Args:
        key: ``capabilities`` 的键名。

    Returns:
        中文名，或未登记时的原键名。
    """
    meta = _CAPABILITY_META.get(key)
    return str(meta["label"]) if meta else str(key)


def capability_what(key: str) -> str:
    """取该能力的白话说明；未登记时返回空串。"""
    meta = _CAPABILITY_META.get(key)
    return str(meta.get("what", "")) if meta else ""


def is_detail(key: str) -> bool:
    """该能力是否属"附注型"（不单独占矩阵一行）。"""
    meta = _CAPABILITY_META.get(key)
    return bool(meta.get("is_detail")) if meta else False


def absent_note(key: str) -> str:
    """值为 False/None 时的补充说明（避免用户把"没有"误读成"缺陷"）。"""
    meta = _CAPABILITY_META.get(key)
    return str(meta.get("absent_note", "")) if meta else ""


def build_capability_matrix(capabilities: dict[str, Any]) -> dict[str, Any]:
    """把 ``capabilities`` 转成可展示的模态矩阵（**动态生成，不硬编码**）。

    规则：
    - 遍历 capabilities 的**全部**键（包括未来新增的）；
    - 布尔键 → 一行（✅/✗ + 中文名 + 白话说明）；
    - ``is_detail`` 的键 → 进 ``details``（附注），不占矩阵行；
    - 未登记的键 → 也出一行，但中文名用键名并标 ``未登记``——
      **不隐藏、不编造中文名**（让遗漏立刻可见，而非静默丢弃）。

    Args:
        capabilities: ``context.meta["capabilities"]``。

    Returns:
        dict，含：
        - ``rows``: [{key, label, present, what, absent_note}]——布尔型能力
        - ``details``: [{key, label, value}]——附注型（轴数/格式名等）
        - ``unregistered``: [key]——未登记中文名的键（便于发现遗漏）
    """
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    unregistered: list[str] = []

    if not isinstance(capabilities, dict):
        return {"rows": [], "details": [], "unregistered": []}

    for key, value in capabilities.items():
        if key not in _CAPABILITY_META:
            unregistered.append(str(key))

        if is_detail(key):
            details.append({
                "key": str(key),
                "label": capability_label(key),
                "value": value,
            })
            continue

        # 布尔型能力：None 视为"未探测"（三态，不与 False 混同）。
        if isinstance(value, bool) or value is None:
            rows.append({
                "key": str(key),
                "label": capability_label(key),
                "present": value,
                "what": capability_what(key),
                "absent_note": absent_note(key) if not value else "",
            })
        else:
            # 非布尔的非 detail 键：归入附注，避免矩阵出现非布尔行。
            details.append({
                "key": str(key),
                "label": capability_label(key),
                "value": value,
            })

    return {"rows": rows, "details": details, "unregistered": unregistered}
