"""图表文案中文化（列名/轴标签/标题/图例的语义映射，不 import streamlit）。

为什么需要（2026-09-14）：图表此前标题是 "line chart"、轴标签直接是 "qpos_0"
这类**原始列名**——对不熟悉数据结构的用户毫无信息量。本模块把常见的具身智能
数据列名映射为中文语义名，并为图表生成描述性标题。

**纪律：只做确定性映射，不猜测语义。**
- 词表命中的列名 → 中文名（如 ``qpos_0`` → ``位置 qpos_0``）；
- 未命中的列名 → **原样保留**，绝不凭字形编造含义（真实案例：tf / imu 这类
  缩写在不同数据集含义不同，硬猜会误导用户）；
- 数据集声明的维度名（meta/info.json）优先于本词表——声明是事实，词表是约定。
"""

from __future__ import annotations

import re

# 列名前缀/关键词 → 中文名。**按最长匹配优先**（见 _match_rule）。
# 只收录具身智能数据里语义明确的常见列；含义有歧义的（如 tf、state）不收录。
_COLUMN_RULES: tuple[tuple[str, str], ...] = (
    # 时间
    ("timestamp", "时间戳"),
    ("time_stamp", "时间戳"),
    ("relative_time", "相对时间"),
    ("frame_time", "帧时间"),
    ("log_time", "日志时间"),
    ("publish_time", "发布时间"),
    ("stamp", "时间戳"),
    # 关节/动作/状态
    ("qpos", "关节位置"),
    ("qvel", "关节速度"),
    ("qacc", "关节加速度"),
    ("joint_position", "关节位置"),
    ("joint_velocity", "关节速度"),
    ("joint", "关节"),
    ("eef", "末端执行器"),
    ("ee", "末端执行器"),
    ("tcp", "工具中心点"),
    ("pose", "位姿"),
    ("action", "动作"),
    ("motor_command", "电机指令"),
    # 传感器
    ("accelerometer", "加速度计"),
    ("accel", "加速度"),
    ("angular_velocity", "角速度"),
    ("gyro", "陀螺仪"),
    ("magnetometer", "磁力计"),
    ("orientation", "姿态四元数"),
    ("quaternion", "四元数"),
    ("force", "力"),
    ("torque", "力矩"),
    ("tactile", "触觉"),
    ("pressure", "压力"),
    ("temperature", "温度"),
    # 图像/视觉
    ("fps", "帧率"),
    ("frame_index", "帧序号"),
    ("exposure", "曝光"),
    ("resolution", "分辨率"),
    # 统计/质检
    ("success", "成功率"),
    ("episode", "回合"),
    ("reward", "奖励"),
    ("count", "计数"),
    ("value", "数值"),
    ("rate", "速率"),
    ("duration", "时长"),
    ("interval", "间隔"),
    ("gap", "缺口"),
    ("loss", "丢失"),
)

# 坐标轴后缀（最后一个下划线后的部分，或**整个列名**）→ 中文。
# 后者覆盖"列名本身就是轴名"的情形：IMU 常把三轴导成 ax/ay/az、
# 力矩导成 fx/fy/fz（真实数据形态）。
_AXIS_SUFFIX: dict[str, str] = {
    "x": "X", "y": "Y", "z": "Z",
    "w": "W", "roll": "横滚", "pitch": "俯仰", "yaw": "偏航",
    "ax": "X", "ay": "Y", "az": "Z",
    "wx": "X", "wy": "Y", "wz": "Z",
    "fx": "X", "fy": "Y", "fz": "Z",
    "tx": "X", "ty": "Y", "tz": "Z",
}

# 裸轴名 → 所属物理量（列名本身就是 ax/fx 这类时，光有"X"没有信息量）。
_BARE_AXIS_QUANTITY: dict[str, str] = {
    "a": "加速度", "w": "角速度", "f": "力", "t": "力矩",
}


def _match_rule(name: str) -> str | None:
    """在词表里找该列名的中文映射（最长关键词优先，避免 "ee" 抢 "eef"）。"""
    lower = name.lower()
    best: tuple[str, str] | None = None
    for keyword, zh in _COLUMN_RULES:
        if keyword in lower and (best is None or len(keyword) > len(best[0])):
            best = (keyword, zh)
    return best[1] if best else None


def _bare_axis_label(name: str) -> str | None:
    """处理"列名本身就是轴缩写"的情形（ax / ay / az / fx / wx ...）。

    为什么需要单独一条：这类列名（IMU 三轴、力矩三轴的真实命名）不含任何
    词表关键词，`_match_rule` 命中不了；而只翻译成"X"又丢掉了物理量信息
    （用户不知道 X 是什么的 X）。故按首字母还原物理量：a→加速度、
    w→角速度、f→力、t→力矩。

    Args:
        name: 原始列名。

    Returns:
        如 "加速度 · X"；不匹配返回 None。
    """
    lower = name.lower().strip()
    if len(lower) != 2 or lower[1] not in ("x", "y", "z"):
        return None
    quantity = _BARE_AXIS_QUANTITY.get(lower[0])
    if quantity is None:
        return None
    return f"{quantity} · {_AXIS_SUFFIX[lower[1]]}"


def describe_column(name: str) -> str:
    """把单个列名翻译为适合做轴标签的中文文案。

    规则（确定性，不猜测）：
    1. 词表命中 → "中文名 (原列名)"，同时保留原始名（用户可能要对照）；
    2. 词表未命中但形如 ``前缀_x``/``前缀_roll`` 等坐标后缀 → "中文前缀 · 坐标"；
    3. 完全未命中 → 原样返回列名（**不编造**）。

    Args:
        name: 原始列名。

    Returns:
        中文轴标签文案（可能仍含原始列名）。
    """
    raw = str(name)
    if not raw:
        return raw
    # 先处理"裸轴名"（ax/fy/wz）——它们不含词表关键词，需按首字母还原物理量。
    bare = _bare_axis_label(raw)
    if bare:
        return bare
    zh = _match_rule(raw)
    # 坐标后缀（ee_x → 末端执行器 · X）。
    suffix_key = raw.lower().rsplit("_", 1)[-1]
    suffix = _AXIS_SUFFIX.get(suffix_key)
    if zh:
        return f"{zh} · {suffix}" if suffix and raw.lower() != suffix_key else zh
    if suffix and "_" in raw:
        prefix = raw.rsplit("_", 1)[0]
        prefix_zh = _match_rule(prefix)
        if prefix_zh:
            return f"{prefix_zh} · {suffix}"
    return raw


def describe_chart_title(
    chart_type: str,
    y_label: str | None,
    x_label: str | None,
    n_series: int | None = None,
    *,
    dataset_id: str | None = None,
) -> str:
    """为图表生成描述性中文标题（无中文标题时使用）。

    为什么不让模型每张图都起标题：模型调用 plot_chart 时未必给 title，而
    "line chart" 这种默认标题对用户毫无信息量。本函数按图表的**实际内容**
    （类型 + 轴列名 + 系列数）拼出可直接理解的标题。

    Args:
        chart_type: 图表类型（line/scatter/histogram/trajectory/multi_stream_overlay）。
        y_label: Y 轴中文标签（已 describe_column）。
        x_label: X 轴中文标签。
        n_series: 系列数（多条曲线时体现在标题里）。
        dataset_id: 可选，数据集名（仅在无其他信息时兜底用）。

    Returns:
        中文标题（不含图表类型英文名）。
    """
    ctype = (chart_type or "").lower()
    multi = f"{n_series} 条曲线" if n_series and n_series > 1 else None
    if ctype == "histogram":
        base = f"{y_label or '数值'}的分布" if y_label else "数值分布"
        return base
    if ctype == "scatter":
        if y_label and x_label and y_label != x_label:
            return f"{y_label} 与 {x_label} 的关系"
        return y_label or x_label or "散点图"
    if ctype == "line":
        if y_label and x_label and y_label != x_label:
            head = f"{y_label} 随 {x_label} 变化"
        else:
            head = y_label or x_label or "时序曲线"
        return f"{head}（{multi}）" if multi else head
    if ctype == "trajectory":
        return "运动轨迹" + (f"（{multi}）" if multi else "")
    if ctype == "multi_stream_overlay":
        n = f"（{n_series} 路流）" if n_series else ""
        return f"多流时间序列对齐{n}"
    return dataset_id or chart_type or "图表"


def describe_series_label(
    stream_name: str,
    column: str,
    *,
    sample_rate_hz: float | None = None,
) -> str:
    """多流叠加图的图例文案：流名 + 中文列名（可选采样率）。

    为什么带上采样率：多流对齐图上"哪条流多快"是判断对齐问题的关键信息，
    放在图例里比让用户去别处查更方便。

    Args:
        stream_name: 流文件名（如 imu.csv）。
        column: 该流绘制的数值列。
        sample_rate_hz: 可选，实测采样率。

    Returns:
        图例文案，如 "imu.csv · 加速度（100 Hz）"。
    """
    # 文件名去掉扩展名（图例里 ".csv" 是噪音）。
    name = stream_name
    if "." in name:
        name = name.rsplit(".", 1)[0]
    # 列名映射：命中词表时用中文名（更易读），未命中时原样显示。
    # 不再附加原始列名——多流图例已含流名，再加列名会三重叠（实测
    # "imu · 加速度 · X（ax）（100 Hz）" 过长且挤占绘图区）。
    col_txt = describe_column(column)
    parts = [name, col_txt]
    label = " · ".join(p for p in parts if p)
    if sample_rate_hz:
        rate_txt = (f"{sample_rate_hz:.0f} Hz" if sample_rate_hz >= 10
                    else f"{sample_rate_hz:.2f} Hz")
        label = f"{label}（{rate_txt}）"
    return label or stream_name or "流"


def sanitize_for_display(text: str) -> str:
    """清理供图表显示的文本：去换行、压空白、截断超长。

    为什么需要：列名/流名可能来自用户数据，含换行或超长路径时会**破坏图例布局**
    （单行图例被撑爆或换行错位）。

    Args:
        text: 原始文本。

    Returns:
        单行、长度受限（≤60 字符）的展示文本。
    """
    one_line = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(one_line) > 60:
        one_line = one_line[:57] + "…"
    return one_line
