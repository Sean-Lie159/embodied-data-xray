"""数据集目录嗅探（纯逻辑，可独立测试）。

面向"原始采集数据目录"做文件普查与能力嗅探：
- 递归普查文件（扩展名分布、目录结构）；
- 表格类（csv/json/parquet）读列名，推断是 IMU / 位姿 / 状态动作；
- json/yaml 检查是否为标定文件；
- 视频类用 ffprobe 读元数据（ffprobe 缺失时降级，返回结构化提示）。

所有推测结果均带置信度（high / low），无法确定时标 unknown。
"""

from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# 视频扩展名（用于视频嗅探）。
_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

# 表格扩展名（用于列名嗅探）。
_TABLE_EXTS = {".csv", ".json", ".parquet"}

# 标定相关扩展名。
_CALIB_EXTS = {".yaml", ".yml", ".json"}

# 标定文件关键键。
_CALIB_KEYS = {"intrinsic", "extrinsic", "k", "d", "intrinsics", "extrinsics", "matrix"}

# 状态/动作的关键列名片段（子串匹配，用于状态/动作推断）。
_ACTION_STATE_COLS = ("qpos", "qvel", "qacc", "action", "obs", "state", "cmd")


def _random_sample(paths: list[Path], k: int = 3) -> list[Path]:
    """从列表中随机抽 k 个（不足则全取）。"""
    if len(paths) <= k:
        return list(paths)
    return random.sample(paths, k)


def probe_directory(root: Path) -> dict[str, Any]:
    """递归普查目录，返回扩展名分布与目录结构。

    Args:
        root: 数据集根目录。

    Returns:
        dict，包含 total_files、ext_dist（扩展名→数量）、subdirs（子目录清单）、
        以及按类别抽取的样例文件路径清单。
    """
    ext_counter: Counter[str] = Counter()
    subdirs: list[str] = []
    all_files: list[Path] = []
    for item in root.rglob("*"):
        if item.is_dir():
            subdirs.append(str(item.relative_to(root)))
        elif item.is_file():
            all_files.append(item)
            ext_counter[item.suffix.lower()] += 1

    # 按类别抽样例文件。
    tables = [p for p in all_files if p.suffix.lower() in _TABLE_EXTS]
    videos = [p for p in all_files if p.suffix.lower() in _VIDEO_EXTS]
    cals = [p for p in all_files if p.suffix.lower() in _CALIB_EXTS]

    return {
        "total_files": len(all_files),
        "ext_dist": dict(ext_counter),
        "subdirs": subdirs,
        "sample_tables": [str(p) for p in _random_sample(tables)],
        "sample_videos": [str(p) for p in _random_sample(videos)],
        "sample_calibs": [str(p) for p in _random_sample(cals)],
    }


# 位姿整词 token（整词匹配，避免子串误命中）。
_POSE_TOKENS = {"pos", "pose", "position", "tcp", "tool"}

# 关节状态列的前缀：一律判为关节状态，不得命中位姿。
_JOINT_PREFIXES = ("qpos", "qvel", "qacc", "joint")

# 位姿列模式：ee_*、*_pose、tcp_*。
def _is_pose_column(name: str) -> bool:
    """判断单个列名是否为位姿列（分词整词匹配 + 模式匹配 + 关节排除）。

    Args:
        name: 列名（已小写）。

    Returns:
        是否判定为位姿列。
    """
    if name.startswith(_JOINT_PREFIXES):
        return False  # 关节状态/速度列，显式排除。
    tokens = {t for t in _split_tokens(name)}
    if tokens & _POSE_TOKENS:
        return True  # 整词命中 pos/pose/position/tcp/tool。
    if name.startswith("ee_") or name.startswith("tcp_") or name.endswith("_pose"):
        return True  # 模式匹配。
    return False


def _split_tokens(name: str) -> list[str]:
    """将列名按非字母数字字符切分为 token 并小写化。

    Args:
        name: 原始列名。

    Returns:
        小写 token 列表；空串被过滤。
    """
    import re

    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def sniff_table_columns(columns: list[str]) -> dict[str, Any]:
    """从表格列名推断数据类型（IMU / 位姿 / 状态动作）与置信度。

    位姿判定采用分词后整词匹配（避免 qpos 被 "pos" 子串误命中），并显式排除
    关节状态列（qpos* / qvel* / qacc* / joint*）。每个能力标签附带命中列名清单
    作为推测依据。

    Args:
        columns: 表格的列名列表。

    Returns:
        dict，含 has_imu（present + confidence + columns）、imu_axes、
        has_pose、has_actions 等能力推断；每项附命中列名清单，未知或无命中时
        如实标注。
    """
    cols = [str(c).lower() for c in columns]
    joined = " ".join(cols)

    # IMU 推断：accel+gyro → 6轴；+mag → 9轴；仅 quat/orientation → 置信低。
    accel_cols = [c for c in cols if "accel" in c]
    gyro_cols = [c for c in cols if ("gyro" in c or "gyr" in c)]
    mag_cols = [c for c in cols if "mag" in c]
    quat_cols = [c for c in cols if ("quat" in c or "orientation" in c)]

    imu: dict[str, Any] = {"present": False, "confidence": "low", "columns": []}
    imu_axes: Any = None
    if accel_cols and gyro_cols:
        imu["present"] = True
        imu["confidence"] = "high"
        imu_axes = 9 if mag_cols else 6
        imu["columns"] = accel_cols + gyro_cols + mag_cols
    elif quat_cols:
        imu["present"] = True
        imu["confidence"] = "low"
        imu_axes = "unknown"
        imu["columns"] = quat_cols
    elif accel_cols or gyro_cols:
        imu["present"] = True
        imu["confidence"] = "low"
        imu_axes = "unknown"
        imu["columns"] = accel_cols + gyro_cols

    # 位姿推断（分词整词匹配，排除关节列）。
    pose_cols = [c for c in cols if _is_pose_column(c)]
    pose: dict[str, Any] = {
        "present": bool(pose_cols),
        "confidence": "high" if pose_cols else "low",
        "columns": pose_cols,
    }

    # 状态/动作推断。
    action_cols = [
        c for c in cols if any(k in c for k in _ACTION_STATE_COLS)
    ]
    actions: dict[str, Any] = {
        "present": bool(action_cols),
        "confidence": "high"
        if any(("action" in c or c.startswith(_JOINT_PREFIXES)) for c in cols)
        else "low",
        "columns": action_cols,
    }

    # 力/力矩通道推断（force / torque / ft / force_torque）。
    force_cols = [
        c for c in cols
        if any(k in c for k in ("force", "torque", "ft_", "force_torque", "_ft", "fx", "fy", "fz", "tx", "ty", "tz"))
    ]
    force: dict[str, Any] = {
        "present": bool(force_cols),
        "confidence": "high" if any(("force" in c or "torque" in c) for c in force_cols) else "low",
        "columns": force_cols,
    }

    return {
        "has_imu": imu,
        "imu_axes": imu_axes,
        "has_pose": pose,
        "has_actions": actions,
        "has_force": force,
        "matched_keywords": joined[:200],
    }


def is_calibration_file(obj: Any) -> bool:
    """判断一个 json/yaml 解析对象是否为标定文件。

    Args:
        obj: 解析后的 JSON/YAML 对象。

    Returns:
        是否含标定关键键。
    """
    if not isinstance(obj, dict):
        return False
    keys = {str(k).lower() for k in obj.keys()}
    return bool(keys & _CALIB_KEYS)


def probe_video(path: str) -> dict[str, Any]:
    """用 ffprobe 读取视频元数据。

    若 ffprobe 不可用，返回结构化降级提示（ffprobe_available=False）。

    Args:
        path: 视频文件路径。

    Returns:
        dict，ffprobe_available 为 True 时含 fps、width、height、nb_frames、
        duration、codec；为 False 时含降级说明 user_message。
    """
    import shutil

    if shutil.which("ffprobe") is None:
        return {
            "ffprobe_available": False,
            "user_message": "未检测到 ffprobe（ffmpeg），已跳过视频元数据嗅探。"
            "可安装 ffmpeg 后重试以获得视频信息。",
        }

    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            return {
                "ffprobe_available": True,
                "ffprobe_error": proc.stderr.strip()[:200],
                "user_message": "ffprobe 读取视频失败，无法获取元数据。",
            }
        data = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        return {
            "ffprobe_available": False,
            "ffprobe_error": str(exc)[:200],
            "user_message": "ffprobe 调用失败，已跳过视频元数据嗅探。",
        }

    # 提取视频流信息。
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
    )
    fmt = data.get("format", {})
    fps: Any = None
    try:
        r = video_stream.get("avg_frame_rate", "0/1")
        num, den = r.split("/")
        fps = round(float(num) / float(den), 3) if float(den) else None
    except (ValueError, ZeroDivisionError):
        fps = None

    return {
        "ffprobe_available": True,
        "fps": fps,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "nb_frames": video_stream.get("nb_frames"),
        "duration": fmt.get("duration"),
        "codec": video_stream.get("codec_name"),
    }


def build_capabilities(probe: dict[str, Any], table_sniffs: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总能力标签与推测类型。

    Args:
        probe: probe_directory 的结果。
        table_sniffs: 各抽样表格的 sniff_table_columns 结果列表。

    Returns:
        dict，含 capabilities（能力标签）与 guessed_type / guessed_type_confidence。
    """
    has_video = len(probe["sample_videos"]) > 0 or any(
        v in _VIDEO_EXTS for v in probe["ext_dist"]
    )

    # 汇总表格嗅探：取多数 / 任一。
    imu_present = any(s["has_imu"]["present"] for s in table_sniffs)
    imu_conf = "high" if any(s["has_imu"]["confidence"] == "high" for s in table_sniffs) else ("low" if imu_present else "unknown")
    axes: set[Any] = {s["imu_axes"] for s in table_sniffs if isinstance(s["imu_axes"], int)}
    imu_axes = max(axes) if axes else None

    pose_present = any(s["has_pose"]["present"] for s in table_sniffs)
    actions_present = any(s["has_actions"]["present"] for s in table_sniffs)
    force_present = any(s["has_force"]["present"] for s in table_sniffs)

    # 汇总命中列名清单（供 inspect_streams 复用，不重复嗅探）。
    imu_channels = sorted({c for s in table_sniffs for c in s["has_imu"]["columns"]})
    action_channels = sorted({c for s in table_sniffs for c in s["has_actions"]["columns"]})
    force_channels = sorted({c for s in table_sniffs for c in s["has_force"]["columns"]})

    capabilities: dict[str, Any] = {
        "has_video_streams": has_video,
        "has_imu": imu_present,
        "has_force": force_present,
        "has_calibration": len(probe["sample_calibs"]) > 0,
        "has_actions": actions_present,
    }
    if imu_axes is not None:
        capabilities["imu_axes"] = imu_axes
    if imu_channels:
        capabilities["imu_channels"] = imu_channels
    if action_channels:
        capabilities["action_channels"] = action_channels
    if force_channels:
        capabilities["force_channels"] = force_channels

    # 推测类型：综合判断。
    guessed_type = "unknown"
    conf = 0.0
    if has_video and imu_present and pose_present:
        guessed_type, conf = "Ego", 0.7
    elif has_video and (imu_present or pose_present):
        guessed_type, conf = "Ego", 0.5
    elif imu_present and pose_present:
        guessed_type, conf = "遥操/动捕", 0.5
    elif actions_present:
        guessed_type, conf = "状态/动作数据", 0.6
    elif imu_present:
        guessed_type, conf = "IMU 数据", 0.6

    return {
        "capabilities": capabilities,
        "guessed_type": guessed_type,
        "guessed_type_confidence": conf,
        "imu_confidence": imu_conf,
    }


# 角色推断：触发词 → 角色名 + 基准置信度。
# 注意：触发词应为足够长的词元，避免短子串误命中（如 "ft" 会命中 "left"）。
# 方位词（left/right/front 等）由 _POSITION_WORDS 单独处理，不在此作为独立角色。
_ROLE_PATTERNS: list[tuple[tuple[str, ...], str, float]] = [
    (("wrist", "wristcam"), "腕部相机", 0.8),
    (("head", "headcam", "eye"), "头部相机", 0.7),
    (("rgb", "camera"), "RGB 相机", 0.6),
    (("imu",), "IMU 传感器", 0.8),
    (("force", "force_torque", "ft_sensor", "wrench", "torque"), "力/力矩传感器", 0.8),
    (("action", "qpos", "qvel", "state", "obs"), "状态/动作流", 0.7),
]

# 方位修饰词：识别后作为角色后缀（如"腕部相机（左）"）。
_POSITION_WORDS = {
    "left": "左",
    "right": "右",
    "front": "前",
    "back": "后",
    "top": "上",
    "bottom": "下",
}


def infer_role(source: str) -> dict[str, Any]:
    """从文件名/路径推测设备角色，收集全部命中语义线索并组合输出。

    与嗅探的置信度标注保持一致：角色为推测，附带依据（命中的关键词）与
    confidence（high / low）。无命中时保持 unknown。

    Args:
        source: 文件路径或名称。

    Returns:
        dict，含 role（推测角色，可组合如"腕部相机（左）"）、confidence
        （high/low）、evidence（命中的语义线索）。
    """
    name = Path(source).name.lower()

    # 方位词单独收集（left/right/front/back/top/bottom）。
    positions = [zh for en, zh in _POSITION_WORDS.items() if en in name]
    positions = list(dict.fromkeys(positions))  # 去重保序

    # 角色触发词收集（词元子串，但触发词本身足够长，避免短子串误命中）。
    hits: list[tuple[str, float, str]] = []  # (角色, 置信度, 命中词)
    for keys, role, conf in _ROLE_PATTERNS:
        for k in keys:
            if k in name and role not in [h[0] for h in hits]:
                hits.append((role, conf, k))

    if not hits and not positions:
        return {
            "role": "unknown",
            "confidence": "low",
            "evidence": "文件名无可识别模式",
        }

    # 主角色取置信度最高的命中；方位词作为后缀修饰。
    hits.sort(key=lambda h: -h[1])
    main_role, main_conf, main_word = hits[0]
    pos_suffix = f"（{'、'.join(positions)}）" if positions else ""
    role = f"{main_role}{pos_suffix}"

    evidence_parts = [f"命中 {h[2]}" for h in hits]
    if positions:
        evidence_parts.append("方位词 " + "、".join(positions))
    return {
        "role": role,
        "confidence": "high" if main_conf >= 0.75 else "low",
        "evidence": "；".join(evidence_parts),
    }


_KIND_ROLE = {
    "imu": "IMU 传感器",
    "force": "力/力矩传感器",
    "actions": "状态/动作流",
    "pose": "位姿流",
}


def _role_for_kind(kind: str, path: str) -> dict[str, Any]:
    """根据流的 kind 给出明确角色，并附加文件名方位词。

    表格流的 kind 由列名嗅探确定（比文件名更可信），因此优先使用 kind 对应的
    角色，避免 "wrist_imu" 这类命名被误判为"腕部相机"。

    Args:
        kind: 流类型（imu/force/actions/pose/unknown/video）。
        path: 文件路径（用于提取方位修饰词）。

    Returns:
        角色 dict（role / confidence / evidence）。
    """
    if kind in _KIND_ROLE:
        # 提取文件名中的方位词作为修饰（如"IMU 传感器（左）"）。
        name = Path(path).name.lower()
        positions = [zh for en, zh in _POSITION_WORDS.items() if en in name]
        pos_suffix = f"（{'、'.join(dict.fromkeys(positions))}）" if positions else ""
        return {
            "role": f"{_KIND_ROLE[kind]}{pos_suffix}",
            "confidence": "high",
            "evidence": f"列名嗅探判定为 {kind}",
        }
    # 未知或视频流：退回文件名角色推断。
    return infer_role(path)


def build_streams_registry(
    probe: dict[str, Any],
    table_info: list[dict[str, Any]],
    video_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构建流登记表（供 inspect_streams 按需读取）。

    每条流含 {path, format, kind, channels, role}。表格流按列名嗅探的命中结果
    归入对应 kind（imu/force/actions/pose/unknown）；视频流标记 kind="video"。

    Args:
        probe: probe_directory 的结果。
        table_info: 表格嗅探信息列表（含 file 与 sniff）。
        video_meta: 视频元数据列表。

    Returns:
        流登记表列表。
    """
    streams: list[dict[str, Any]] = []

    # 表格流：根据嗅探结果判断 kind 与通道。
    for t in table_info:
        path = t["file"]
        sniff = t["sniff"]
        kind = "unknown"
        channels: list[str] = []
        if sniff["has_imu"]["present"]:
            kind = "imu"
            channels = sniff["has_imu"]["columns"]
        elif sniff["has_force"]["present"]:
            kind = "force"
            channels = sniff["has_force"]["columns"]
        elif sniff["has_actions"]["present"]:
            kind = "actions"
            channels = sniff["has_actions"]["columns"]
        elif sniff["has_pose"]["present"]:
            kind = "pose"
            channels = sniff["has_pose"]["columns"]
        streams.append({
            "path": path,
            "format": Path(path).suffix.lstrip(".").lower(),
            "kind": kind,
            "channels": channels,
            "role": _role_for_kind(kind, path),
        })

    # 视频流。
    for v in video_meta:
        path = v.get("file", "unknown")
        streams.append({
            "path": path,
            "format": "video",
            "kind": "video",
            "channels": [],
            "role": infer_role(path),
        })

    return streams
