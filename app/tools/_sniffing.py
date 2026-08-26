"""数据集目录嗅探（纯逻辑，可独立测试）。

面向"原始采集数据目录"做文件普查与能力嗅探。语义识别采用**四层架构**：

- 第 1 层 词典先验：现有词典匹配降级为"线索"，命中只产生候选，未命中不得判死刑；
- 第 2 层 内容指纹（裁判）：读取少量样本数据做确定性数值判定（时间戳/四元数/标定/
  力），每项判定必须附"依据说明"；
- 第 3 层 LLM 语义假设 + 工具验证：验证函数落地为内部函数（见 classify_table_stream），
  结果写入流注册表的 semantic_label / label_evidence / label_confidence，由
  inspect_streams 透出供 LLM 读取证据（不注册新 agent 工具）；
- 第 4 层 用户确认 + 持久化：由 load_dataset 加载时读取 outputs/.dataset_profile.json
  覆盖上述结果（见 load_dataset）。

所有推测结果均带置信度（high / low），且第 2 层指纹判定必须附"依据说明"。
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# 视频扩展名（用于视频嗅探）。
_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

# 音频扩展名（仅登记路径与格式，不做深度嗅探）。
_AUDIO_EXTS = {".m4a", ".wav", ".mp3", ".flac", ".aac", ".ogg"}

# 图片扩展名（仅登记路径与格式）。
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# 表格扩展名（用于列名嗅探）。
_TABLE_EXTS = {".csv", ".json", ".parquet"}

# 标定相关扩展名（json/yaml 可能含标定键，需进一步解析判定）。
_CALIB_EXTS = {".yaml", ".yml", ".json"}

# 桌面/系统文件：不参与任何探测，直接跳过（不崩、不进清单）。
_SYSTEM_FILES = {"desktop.ini", "thumbs.db", ".ds_store", ".ds_store_"}
# 系统文件扩展名兜底（.DS_Store 等以点开头的隐藏文件）。
_SYSTEM_FILE_PREFIXES = (".ds_store",)

# 空流阈值：行数 ≤ 此值视为未使用/空流，不计入可对齐流。
EMPTY_STREAM_MAX_ROWS = 2

# 内容指纹采样行数：每条表格流最多读取如此多行样本做数值指纹。
_FINGERPRINT_SAMPLE_ROWS = 500

# 标定时钟/标定键（小尺寸数值矩阵特征）。
_CALIB_KEYS = {
    "intrinsic", "intrinsics", "extrinsic", "extrinsics", "k", "d",
    "matrix", "focal", "focalx", "focaly", "centerx", "centery",
    "radialdistortion", "rotation", "position", "bias", "scale", "noise",
    "distortion", "projection",
}

# 状态/动作的关键列名片段（子串匹配，用于第 1 层状态/动作线索）。
_ACTION_STATE_COLS = ("qpos", "qvel", "qacc", "action", "obs", "state", "cmd")

# 力/力矩的关键列名片段（子串匹配，第 1 层线索，第 2 层需数值佐证才生效）。
_FORCE_COLS = ("force", "torque", "ft_", "force_torque", "_ft", "fx", "fy",
               "fz", "tx", "ty", "tz", "wrench")

# 常见时间戳列名（用于第 2 层时间戳指纹）。
_TIMESTAMP_COLS = ("timestamp", "timestamp_ns", "time", "ts", "ts_ns", "t", "stamp",
                   "frame_time", "pts_us", "capture_utc_ns",
                   "exposure_start_utc_ns", "mid_exposure_utc_ns",
                   "exposure_duration_ns", "packet_index", "frame_index",
                   "frame_id")

# 四元数列名模式：列名以四元数分量后缀成组（x/y/z/w），或含 orientation/quat。
_QUAT_SUFFIXES = ("_x", "_y", "_z", "_w")
_QUAT_TOKENS = ("quat", "orientation", "rot")


def _is_system_file(path: Path) -> bool:
    """判断是否为桌面/系统文件（desktop.ini、Thumbs.db、.DS_Store 等）。

    Args:
        path: 文件路径。

    Returns:
        是系统文件返回 True，应跳过不参与探测。
    """
    name = path.name
    lower = name.lower()
    if lower in _SYSTEM_FILES:
        return True
    # 隐藏的系统文件：以 .ds_store 开头（含 .DS_Store / .DS_Store_）。
    if lower.startswith(_SYSTEM_FILE_PREFIXES):
        return True
    return False


def probe_directory(root: Path) -> dict[str, Any]:
    """递归普查目录，返回完整文件清单（按类型分组）与扩展名分布。

    返回完整路径清单、不抽样、不省略；所有列表按相对路径排序，保证同一目录
    两次加载结果完全一致（确定性）。

    Args:
        root: 数据集根目录。

    Returns:
        dict，含 total_files、ext_dist（扩展名→数量）、subdirs（子目录清单）、
        以及按类型分组的完整文件路径清单：tables / videos / audios / images /
        cals（标定候选）/ others（其余）。所有路径均为完整路径字符串，按相对
        root 的顺序排序，保证确定性。
    """
    ext_counter: Counter[str] = Counter()
    subdirs: list[str] = []
    grouped: dict[str, list[Path]] = {
        "tables": [], "videos": [], "audios": [], "images": [],
        "cals": [], "others": [],
    }
    total_files = 0
    for item in root.rglob("*"):
        if item.is_dir():
            subdirs.append(str(item.relative_to(root)))
        elif item.is_file():
            # 桌面/系统文件直接跳过，不参与任何探测（不崩、不进清单）。
            if _is_system_file(item):
                continue
            ext = item.suffix.lower()
            total_files += 1
            ext_counter[ext] += 1
            # 标定候选（json/yaml）优先归入 cals，避免被当作数据表。.json 既可能是
            # 标定也可能是数据表（nuScenes 等）——由 load_dataset 在标定判定后把
            # 非标定 JSON 补入 tables 探测（见 _load_directory_impl）。
            if ext in _CALIB_EXTS:
                grouped["cals"].append(item)
            elif ext in _TABLE_EXTS:
                grouped["tables"].append(item)
            elif ext in _VIDEO_EXTS:
                grouped["videos"].append(item)
            elif ext in _AUDIO_EXTS:
                grouped["audios"].append(item)
            elif ext in _IMAGE_EXTS:
                grouped["images"].append(item)
            else:
                grouped["others"].append(item)

    # 固定排序（按相对 root 的路径），再转完整路径字符串：保证确定性。
    subdirs.sort()
    for key in grouped:
        grouped[key].sort(key=lambda p: str(p.relative_to(root)))
    for key in grouped:
        grouped[key] = [str(p) for p in grouped[key]]

    return {
        "total_files": total_files,
        "ext_dist": dict(sorted(ext_counter.items())),
        "subdirs": subdirs,
        "tables": grouped["tables"],
        "videos": grouped["videos"],
        "audios": grouped["audios"],
        "images": grouped["images"],
        "cals": grouped["cals"],
        "others": grouped["others"],
    }


# --- 第 1 层：词典先验（降级为线索） ----------------------------------------


def _split_tokens(name: str) -> list[str]:
    """将列名按非字母数字字符切分为 token 并小写化。

    Args:
        name: 原始列名。

    Returns:
        小写 token 列表；空串被过滤。
    """
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


# 位姿整词 token（整词匹配，避免子串误命中）。
_POSE_TOKENS = {"pos", "pose", "position", "tcp", "tool"}

# 关节状态列的前缀：一律判为关节状态，不得命中位姿。
_JOINT_PREFIXES = ("qpos", "qvel", "qacc", "joint")


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


def sniff_table_columns(columns: list[str]) -> dict[str, Any]:
    """第 1 层：从表格列名推断数据类型线索（IMU / 位姿 / 状态动作 / 力）。

    词典匹配仅作为"线索候选"，未命中不得判死刑（见 classify_table_stream 的
    第 2 层指纹裁判）。每一项附命中列名清单作为推测依据；无命中时如实标注。

    Args:
        columns: 表格的列名列表。

    Returns:
        dict，含 has_imu / has_pose / has_actions / has_force 等线索，每项附
        present、confidence、columns（命中列名）；并附 layer="dictionary" 标注。
    """
    cols = [str(c).lower() for c in columns]
    joined = " ".join(cols)

    # IMU 线索：accel+gyro → 6轴；+mag → 9轴；仅 quat/orientation → 低置信；
    # 仅 accel/gyro 之一 → 低置信。
    accel_cols = [c for c in cols if "accel" in c]
    gyro_cols = [c for c in cols if ("gyro" in c or "gyr" in c)]
    mag_cols = [c for c in cols if "mag" in c]
    quat_cols = [c for c in cols if ("quat" in c or "orientation" in c)]

    imu: dict[str, Any] = {"present": False, "confidence": "low", "columns": [],
                            "layer": "dictionary"}
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

    # 位姿线索（分词整词匹配，排除关节列）。
    pose_cols = [c for c in cols if _is_pose_column(c)]
    pose: dict[str, Any] = {
        "present": bool(pose_cols),
        "confidence": "high" if pose_cols else "low",
        "columns": pose_cols,
        "layer": "dictionary",
    }

    # 状态/动作线索。
    action_cols = [c for c in cols if any(k in c for k in _ACTION_STATE_COLS)]
    actions: dict[str, Any] = {
        "present": bool(action_cols),
        "confidence": "high"
        if any(("action" in c or c.startswith(_JOINT_PREFIXES)) for c in cols)
        else "low",
        "columns": action_cols,
        "layer": "dictionary",
    }

    # 力/力矩线索（词典命中，第 2 层需数值佐证才生效）。
    force_cols = [c for c in cols if any(k in c for k in _FORCE_COLS)]
    force: dict[str, Any] = {
        "present": bool(force_cols),
        "confidence": "high" if any(("force" in c or "torque" in c) for c in force_cols) else "low",
        "columns": force_cols,
        "layer": "dictionary",
    }

    # 手部跟踪线索：列含 joint* / orientation* / *_active / *_name 等。
    hand_tracking_cols = [c for c in cols if any(
        k in c for k in ("joint", "orientation", "_active", "_name", "_id")
    )]
    hand_tracking: dict[str, Any] = {
        "present": len(hand_tracking_cols) >= 4,  # 多关节 + 元数据列才判为手部跟踪
        "confidence": "high" if len(hand_tracking_cols) >= 4 else "low",
        "columns": hand_tracking_cols,
        "layer": "dictionary",
    }

    return {
        "has_imu": imu,
        "imu_axes": imu_axes,
        "has_pose": pose,
        "has_actions": actions,
        "has_force": force,
        "has_hand_tracking": hand_tracking,
        "matched_keywords": joined[:200],
    }


def is_calibration_file(obj: Any) -> bool:
    """第 1 层：判断一个 json/yaml 解析对象是否为标定文件（词典级线索）。

    Args:
        obj: 解析后的 JSON/YAML 对象。

    Returns:
        是否含标定关键键。
    """
    if not isinstance(obj, dict):
        return False
    keys = {str(k).lower() for k in obj.keys()}
    return bool(keys & _CALIB_KEYS)


# --- 第 2 层：内容指纹（确定性裁判，必须附依据说明） ------------------------

def fingerprint_timestamp(sample: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """第 2 层：时间戳指纹。

    规则：列名属时间戳候选（_TIMESTAMP_COLS）且数值列单调递增、差分大致均匀
    （中位数间隔下，差分 < 1.5×中位数的占比 ≥ 0.9）。返回命中的时间戳列与依据。

    Args:
        sample: 表格样本 DataFrame（前若干行）。
        columns: 列名列表。

    Returns:
        dict，present、column（时间戳列名或 None）、evidence（依据说明）、
        layer="content_fingerprint"。
    """
    candidate_cols = [
        c for c in columns
        if str(c).lower().strip() in _TIMESTAMP_COLS and c in sample.columns
    ]
    from app.tools.timestamp_units import infer_unit, unit_to_ns_factor

    for col in candidate_cols:
        s = pd.to_numeric(sample[col], errors="coerce").dropna()
        if len(s) < 3:
            continue
        if not bool((s.diff().dropna() >= 0).all()):
            continue  # 非单调递增
        diffs = s.diff().dropna()
        med = float(diffs.median())
        if med <= 0:
            continue
        # 差分大致均匀：偏离中位数 1.5 倍以内的占比。
        uniform_ratio = float((diffs <= med * 1.5).mean())
        if uniform_ratio >= 0.9:
            # 单位推断：优先列名后缀，其次按中位差分量级；无法推断时标 unknown。
            unit_info = infer_unit(s.to_numpy(), col)
            unit = unit_info["unit"]
            unit_basis = unit_info["unit_basis"]
            # 依据里附上到纳秒的换算因子（时间单位才有）。
            unit_extra = (
                f"，到纳秒换算 ×{unit_to_ns_factor(unit)}"
                if unit in ("s", "ms", "us", "ns") else ""
            )
            return {
                "present": True,
                "column": col,
                "evidence": (
                    f"列 {col} 单调递增，间隔中位数 {med:.3g}，"
                    f"差分均匀占比 {uniform_ratio:.0%}"
                ),
                # 单位推断结果透出（供采样率 / 对齐归一化使用）。
                "timestamp_unit": unit,
                "timestamp_unit_basis": f"{unit_basis}{unit_extra}",
                "layer": "content_fingerprint",
            }
    return {
        "present": False, "column": None,
        "evidence": "无可识别的时间戳列（需列名命中且数值单调递增、差分均匀）",
        "timestamp_unit": "unknown",
        "timestamp_unit_basis": "无时间戳列，无法推断单位",
        "layer": "content_fingerprint",
    }


def detect_quaternion_groups(sample: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """第 2 层：四元数指纹。

    规则：列名以 _x/_y/_z/_w 后缀成组，或含 quat/orientation 且四列成组，且每行
    模长 ≈ 1（最大偏差 < 0.01，归一化四元数）。返回成组信息。

    Args:
        sample: 表格样本 DataFrame。
        columns: 列名列表。

    Returns:
        list[dict]，每组含 prefix、columns、evidence、layer。
    """
    groups: list[dict[str, Any]] = []
    # 以 _x/_y/_z/_w 后缀成组。
    bases: dict[str, dict[str, str]] = {}
    for c in columns:
        cl = str(c).lower()
        for suf in _QUAT_SUFFIXES:
            if cl.endswith(suf):
                base = cl[: -len(suf)]
                bases.setdefault(base, {})[suf] = c
                break
    for base, suf_map in bases.items():
        if set(_QUAT_SUFFIXES).issubset(suf_map.keys()):
            cols = [suf_map[s] for s in _QUAT_SUFFIXES]
            try:
                mat = sample[cols].apply(pd.to_numeric, errors="coerce")
                norms = (mat ** 2).sum(axis=1).pow(0.5)
                if len(norms) >= 2 and norms.dropna().std() < 0.01 and norms.dropna().mean() > 0.99:
                    groups.append({
                        "prefix": base,
                        "columns": cols,
                        "evidence": (
                            f"列 {base}_x/y/z/w 成组，每行模长≈1"
                            f"（均值 {norms.dropna().mean():.4f}，标准差 {norms.dropna().std():.2e}）"
                        ),
                        "layer": "content_fingerprint",
                    })
            except Exception:  # noqa: BLE001
                continue
    return groups


def fingerprint_calibration(obj: Any) -> dict[str, Any]:
    """第 2 层：标定 JSON 指纹（确定性，必附依据）。

    规则：JSON 含小尺寸数值矩阵特征——内参（focal*/center*/radialDistortion）、
    外参四元数（extrinsics.rotation 四元数）、bias/scale/noise（IMU 标定）等。

    Args:
        obj: 解析后的 JSON 对象。

    Returns:
        dict，present、keys_found、evidence、layer。
    """
    if not isinstance(obj, dict):
        return {"present": False, "keys_found": [],
                "evidence": "非字典结构，非标定", "layer": "content_fingerprint"}
    found = sorted({str(k).lower() for k in obj.keys()} & _CALIB_KEYS)
    # 递归查找嵌套键（intrinsics/extrinsics 子对象）。
    nested_found: set[str] = set()
    try:
        blob = json.dumps(obj, ensure_ascii=False).lower()
        for k in _CALIB_KEYS:
            if re.search(r'["\']' + re.escape(k) + r'["\']', blob):
                nested_found.add(k)
    except Exception:  # noqa: BLE001
        pass
    all_found = sorted(set(found) | nested_found)
    if all_found:
        return {
            "present": True,
            "keys_found": all_found,
            "evidence": f"含标定键 {all_found}",
            "layer": "content_fingerprint",
        }
    return {
        "present": False, "keys_found": [],
        "evidence": "未含已知标定键（intrinsics/extrinsics/rotation/bias/scale 等）",
        "layer": "content_fingerprint",
    }


def fingerprint_force(sample: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """第 2 层：力/力矩指纹（需数值佐证，不能仅凭列名）。

    规则：列名含力/力矩片段，且数值在力/力矩合理量纲（非纯坐标）。由于真实数据集
    无力传感器，这里仅在列名命中且数值非退化（存在非零方差）时给 low 置信。

    Args:
        sample: 表格样本 DataFrame。
        columns: 列名列表。

    Returns:
        dict，present、columns、evidence、layer、confidence。
    """
    force_cols = [c for c in columns if any(k in str(c).lower() for k in _FORCE_COLS)]
    if not force_cols:
        return {"present": False, "columns": [],
                "evidence": "列名无 force/torque/ft/fx..tz 片段",
                "layer": "content_fingerprint", "confidence": "low"}
    # 数值佐证：至少一个力列有非零方差（排除恒值/坐标退化）。
    has_signal = False
    for c in force_cols:
        if c in sample.columns:
            s = pd.to_numeric(sample[c], errors="coerce").dropna()
            if len(s) >= 2 and s.var() > 1e-9:
                has_signal = True
                break
    if has_signal:
        return {
            "present": True, "columns": force_cols, "confidence": "low",
            "evidence": f"列名含力/力矩片段且存在非零方差信号：{force_cols}",
            "layer": "content_fingerprint",
        }
    return {
        "present": False, "columns": force_cols, "confidence": "low",
        "evidence": "力/力矩列名命中但数值退化（恒值/无方差），不确认力信号",
        "layer": "content_fingerprint",
    }


# --- 流分类裁判：综合第 1 / 2 层 -------------------------------------------

def classify_table_stream(
    name: str,
    columns: list[str],
    sample: pd.DataFrame | None,
    nrows: int,
) -> dict[str, Any]:
    """综合第 1 层词典线索与第 2 层内容指纹，给出流的语义标签（裁判）。

    优先级（高覆盖低）：
    1. 空流检测：nrows ≤ EMPTY_STREAM_MAX_ROWS → 标记空流，不计入可对齐流；
    2. 词典/命名修正：hand_tracking* → 手部跟踪（不得标 IMU）；
    3. accel/gyro 配对（在 pair_streams 中标记 imu_axes=6）；
    4. 内容指纹确认 IMU（accel/gyro 命中 + 时间戳指纹）；
    5. 四元数指纹 → 位姿（6DoF）或并入 IMU；
    6. 力指纹（需数值佐证）；
    7. 词典线索降级兜底。

    Args:
        name: 文件名（含扩展名）。
        columns: 列名列表。
        sample: 表格样本 DataFrame（前若干行）；可为 None（无样本时仅用词典）。
        nrows: 全表行数（不含表头）。

    Returns:
        dict，含 kind、semantic_label、label_evidence、label_confidence、
        status、channels、timestamp_column、quaternion_groups、imu_axes。

    约定：当第 1 层词典与第 2 层内容指纹均无法判定时，返回 kind="unknown"、
        semantic_label="未知（无法分类）"，并明确列出已排查的层级（label_evidence）。
        **不得硬猜**——决策余地留给第 3 层 LLM 语义假设（未接模型，见
        docs/技术债.md）或第 4 层用户确认。
    """
    # 1. 空流检测。
    if nrows <= EMPTY_STREAM_MAX_ROWS:
        return {
            "kind": "unknown",
            "semantic_label": "未使用/空流",
            "label_evidence": f"行数 {nrows} ≤ {EMPTY_STREAM_MAX_ROWS}，判定为空流",
            "label_confidence": "high",
            "status": "empty",
            "channels": [],
            "timestamp_column": None,
            "quaternion_groups": [],
            "imu_axes": None,
        }

    lower = name.lower()
    dict_sniff = sniff_table_columns(columns)

    # 第 2 层指纹（需样本）。
    ts_fp: dict[str, Any] = {"present": False, "column": None,
                             "evidence": "无样本，跳过时间戳指纹", "layer": "content_fingerprint"}
    quat_groups: list[dict[str, Any]] = []
    force_fp: dict[str, Any] = {"present": False, "columns": [],
                                "evidence": "无样本，跳过力指纹", "layer": "content_fingerprint"}
    if sample is not None:
        ts_fp = fingerprint_timestamp(sample, columns)
        quat_groups = detect_quaternion_groups(sample, columns)
        force_fp = fingerprint_force(sample, columns)

    # 时间戳单位（第 2 层推断）：透出到流登记表，供采样率/对齐归一化使用。
    ts_unit = ts_fp.get("timestamp_unit", "unknown")
    ts_unit_basis = ts_fp.get("timestamp_unit_basis", "未推断")

    # 2. 手部跟踪修正：命名含 hand_tracking 或词典线索强（多 joint/orientation）。
    if "hand_tracking" in lower or dict_sniff["has_hand_tracking"]["present"]:
        label = "手部跟踪（关节+四元数）"
        evidence = "命名 hand_tracking 或列含多关节/orientation/active 元数据"
        if quat_groups:
            evidence += f"；四元数指纹确认：{quat_groups[0]['evidence']}"
        return {
            "kind": "hand_tracking",
            "semantic_label": label,
            "label_evidence": evidence,
            "label_confidence": "high",
            "status": "active",
            "channels": columns,
            "timestamp_column": ts_fp.get("column"),
            "timestamp_unit": ts_unit,
            "timestamp_unit_basis": ts_unit_basis,
            "quaternion_groups": quat_groups,
            "imu_axes": None,
        }

    # 3/4. IMU：列名含 accel/gyro、或文件名含 accel/gyro（真实 accel.csv 列名为
    # x/y/z，靠文件名 + 时间戳指纹识别）；需第 1 层线索或第 2 层时间戳指纹佐证。
    has_accel = any("accel" in c for c in columns)
    has_gyro = any(("gyro" in c or "gyr" in c) for c in columns)
    name_accel = "accel" in lower
    name_gyro = "gyro" in lower or "gyr" in lower
    dict_imu = dict_sniff["has_imu"]["present"]
    name_imu = name_accel or name_gyro
    imu_candidate = (has_accel or has_gyro or name_imu) and (
        dict_imu or ts_fp["present"] or name_imu
    )
    if imu_candidate:
        # 6 轴：accel 列 + gyro 列同时出现，或文件名 accel 与 gyro 配对（见 pair_streams）。
        axes = 6 if ((has_accel and has_gyro) or (name_accel and name_gyro)) else None
        evidence = "命名含 accel/gyro" if name_imu else "词典命中 accel/gyro"
        if ts_fp["present"]:
            evidence += f"；{ts_fp['evidence']}"
        return {
            "kind": "imu",
            "semantic_label": "IMU（加速度/角速度）",
            "label_evidence": evidence,
            "label_confidence": "high" if (dict_imu and ts_fp["present"]) else "low",
            "status": "active",
            "channels": columns,
            "timestamp_column": ts_fp.get("column"),
            "timestamp_unit": ts_unit,
            "timestamp_unit_basis": ts_unit_basis,
            "quaternion_groups": quat_groups,
            "imu_axes": axes,
        }

    # 5. 四元数指针对位姿。
    if quat_groups or dict_sniff["has_pose"]["present"]:
        evidence = "位姿词典线索"
        if quat_groups:
            evidence = f"四元数指纹：{quat_groups[0]['evidence']}"
        return {
            "kind": "pose",
            "semantic_label": "位姿（6DoF：位置+四元数）",
            "label_evidence": evidence,
            "label_confidence": "high" if quat_groups else "low",
            "status": "active",
            "channels": columns,
            "timestamp_column": ts_fp.get("column"),
            "timestamp_unit": ts_unit,
            "timestamp_unit_basis": ts_unit_basis,
            "quaternion_groups": quat_groups,
            "imu_axes": None,
        }

    # 6. 力指纹（需数值佐证才生效）。
    if dict_sniff["has_force"]["present"] and force_fp["present"]:
        return {
            "kind": "force",
            "semantic_label": "力/力矩",
            "label_evidence": force_fp["evidence"],
            "label_confidence": "low",
            "status": "active",
            "channels": force_fp["columns"],
            "timestamp_column": ts_fp.get("column"),
            "timestamp_unit": ts_unit,
            "timestamp_unit_basis": ts_unit_basis,
            "quaternion_groups": quat_groups,
            "imu_axes": None,
        }
    if dict_sniff["has_force"]["present"] and not force_fp["present"]:
        # 词典命中但无数值佐证：降级，不报力（修正"无力误报力"）。
        return {
            "kind": "unknown",
            "semantic_label": "未知（列名似力/力矩但无数值佐证）",
            "label_evidence": force_fp.get("evidence", "力/力矩未获数值佐证"),
            "label_confidence": "low",
            "status": "active",
            "channels": [],
            "timestamp_column": ts_fp.get("column"),
            "timestamp_unit": ts_unit,
            "timestamp_unit_basis": ts_unit_basis,
            "quaternion_groups": quat_groups,
            "imu_axes": None,
        }

    # 7. 词典降级兜底。
    if dict_sniff["has_actions"]["present"]:
        return {
            "kind": "actions",
            "semantic_label": "状态/动作",
            "label_evidence": f"状态/动作词典线索：{dict_sniff['has_actions']['columns']}",
            "label_confidence": dict_sniff["has_actions"]["confidence"],
            "status": "active",
            "channels": dict_sniff["has_actions"]["columns"],
            "timestamp_column": ts_fp.get("column"),
            "timestamp_unit": ts_unit,
            "timestamp_unit_basis": ts_unit_basis,
            "quaternion_groups": quat_groups,
            "imu_axes": None,
        }
    # 8. 判不出：所有层级均未命中，明确返回 unknown，绝不硬猜。
    # 证据需说明"第 1 层词典线索 + 第 2 层内容指纹均无法判定"，把决策空间留给
    # 第 3 层 LLM 语义假设（目前未接模型，见 docs/技术债.md）或第 4 层用户确认。
    checked = []
    checked.append("词典线索（动作/IMU/位姿/力/手部跟踪均未命中）")
    if sample is not None:
        checked.append(
            f"内容指纹（时间戳：{ts_fp.get('evidence','')}；"
            f"四元数：{'命中' if quat_groups else '未命中'}；"
            f"力：{force_fp.get('evidence','')}）"
        )
    else:
        checked.append("内容指纹（无样本，跳过）")
    return {
        "kind": "unknown",
        "semantic_label": "未知（无法分类）",
        "label_evidence": "判不出，未做硬猜；已排查：" + "；".join(checked),
        "label_confidence": "low（无法判定）",
        "status": "active",
        "channels": [],
        "timestamp_column": ts_fp.get("column"),
        "timestamp_unit": ts_unit,
        "timestamp_unit_basis": ts_unit_basis,
        "quaternion_groups": quat_groups,
        "imu_axes": None,
    }


# --- 流配对规则 ------------------------------------------------------------

def pair_streams(
    video_files: list[str],
    table_files: list[str],
    audio_files: list[str],
) -> list[dict[str, Any]]:
    """流配对规则。

    规则 1：<name>.mp4 ↔ <name>_metainfo.csv（同名 + 前缀 _metainfo）→ 视频时间戳
        来源配对（含音频 <name>.m4a ↔ <name>_metainfo.csv）。
    规则 2：目录内 accel*.csv 与 gyro*.csv 且均识别为 IMU → 配对为一组六轴 IMU
        （imu_axes=6）。

    Args:
        video_files: 视频文件路径列表。
        table_files: 表格文件路径列表。
        audio_files: 音频文件路径列表。

    Returns:
        list[dict]，每项含 type、video/audio/metainfo（规则1）或 streams/imu_axes
        （规则2）、source。
    """
    pairs: list[dict[str, Any]] = []

    # 规则 1：媒体 ↔ metainfo。
    media_files = [f for f in (video_files + audio_files)]
    for media in media_files:
        mp = Path(media)
        stem = mp.stem  # 去扩展名
        suffix = mp.suffix.lower()
        if suffix in (".mp4", ".m4a", ".avi", ".mov", ".mkv", ".webm"):
            # 匹配 <stem>_metainfo.csv
            meta_name = f"{stem}_metainfo.csv"
            meta_path = next(
                (t for t in table_files if Path(t).name == meta_name), None
            )
            if meta_path is not None:
                media_kind = "video" if suffix == ".mp4" else "audio"
                pairs.append({
                    "type": "media_metainfo",
                    "media": media,
                    "media_kind": media_kind,
                    "metainfo": meta_path,
                    "source": "content_fingerprint",
                    "evidence": f"{media_kind} {stem} 与 {meta_name} 同名配对，"
                                "metainfo 表登记为该媒体流的时间戳来源",
                })

    # 规则 2：accel + gyro → 六轴 IMU。
    accel_files = [t for t in table_files if "accel" in Path(t).name.lower()]
    gyro_files = [t for t in table_files if "gyro" in Path(t).name.lower()
                  or "gyr" in Path(t).name.lower()]
    if accel_files and gyro_files:
        pairs.append({
            "type": "imu_6axis",
            "streams": [*accel_files, *gyro_files],
            "imu_axes": 6,
            "source": "content_fingerprint",
            "evidence": "accel 与 gyro 同目录配对为一组六轴 IMU",
        })

    return pairs


# --- 视频 / 角色推断（保留，修正误判） -------------------------------------

def _ffprobe_runs() -> bool:
    """真正尝试调用一次 ffprobe，确认其可执行（比 shutil.which 更可靠）。

    Returns:
        ffprobe -version 能在超时内成功执行返回 True，否则 False。
    """
    try:
        proc = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True, text=True, timeout=10, encoding="utf-8",
            errors="replace",
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


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

    # 仅用 shutil.which 并不可靠：用户可能已安装 ffmpeg，但 PATH 中的实际目录
    # 与约定路径（如 C:\ffmpeg\bin）不符，which 会返回 None 而误判"未安装"。
    # 改为真正尝试调用一次 ffprobe -version：能跑通即认为可用，彻底消除误报。
    if shutil.which("ffprobe") is None and not _ffprobe_runs():
        return {
            "ffprobe_available": False,
            "user_message": (
                "未检测到可用的 ffprobe（ffmpeg）。视频元数据嗅探已跳过，"
                "其余功能不受影响。请确认已安装 ffmpeg，并将其 bin 目录加入当前"
                "会话的 PATH（用 `where ffprobe` 或 `ffprobe -version` 验证路径），"
                "再重新加载含视频的数据集。"
            ),
        }

    try:
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, encoding="utf-8",
            errors="replace",
        )
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

    nb_frames_raw = video_stream.get("nb_frames")
    duration_raw = fmt.get("duration")
    try:
        duration_s = float(duration_raw) if duration_raw is not None else None
    except (ValueError, TypeError):
        duration_s = None

    # 帧数可信度甄别：ffprobe 对部分 mp4 会返回 nb_frames=1（明显错误），或缺失、
    # 或与 duration×fps 推算值偏差超过一个数量级。不可信时改用 duration×fps 估算，
    # 并显式标注来源（probe / estimated），禁止把估算值伪装成实测。
    estimated = _estimate_frame_count(nb_frames_raw, fps, duration_s)
    nb_frames = estimated["nb_frames"]
    nb_frames_source = estimated["source"]
    nb_frames_basis = estimated["basis"]
    nb_frames_trusted = estimated["trusted"]

    return {
        "ffprobe_available": True,
        "fps": fps,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "nb_frames": nb_frames,
        "nb_frames_source": nb_frames_source,
        "nb_frames_basis": nb_frames_basis,
        "nb_frames_trusted": nb_frames_trusted,
        "duration": duration_raw,
        "codec": video_stream.get("codec_name"),
    }


def _estimate_frame_count(
    nb_frames_raw: Any,
    fps: Any,
    duration_s: float | None,
) -> dict[str, Any]:
    """甄别视频帧数探测结果可信度，必要时用 duration×fps 估算。

    ffprobe 对部分 mp4 会返回 nb_frames=1（明显错误），或该字段缺失，或与其
    duration×fps 推算值偏差超过一个数量级。这些情况标记为不可信并改用估算值，
    同时把来源（probe / estimated）与依据一并返回，供下游透出"估算"字样。

    Args:
        nb_frames_raw: ffprobe 原始 nb_frames（可能是 None / int / 字符串）。
        fps: 平均帧率（float 或 None）。
        duration_s: 时长（秒，float 或 None）。

    Returns:
        dict，含 nb_frames（展示用值）、source（"probe" / "estimated"）、
        basis（依据说明）、trusted（是否可信）。
    """
    try:
        nb = int(nb_frames_raw) if nb_frames_raw not in (None, "") else None
    except (ValueError, TypeError):
        nb = None

    # 推算值：duration × fps（两者皆有时才可用）。
    estimated_n = None
    if duration_s and fps:
        estimated_n = int(round(duration_s * fps))

    # 可信：nb 存在、>1，且与推算值偏差不超过一个数量级（或无法推算时 nb 合理）。
    trusted = False
    reason = ""
    if nb is None:
        reason = "ffprobe 未返回 nb_frames"
    elif nb <= 1:
        reason = f"ffprobe 返回 nb_frames={nb}（明显错误，单帧视频极罕见）"
    elif estimated_n is not None and (
        nb == 0 or estimated_n == 0 or max(nb, estimated_n) / max(min(nb, estimated_n), 1) > 10
    ):
        reason = (
            f"ffprobe 返回 nb_frames={nb}，与 duration×fps 推算值 "
            f"≈{estimated_n} 偏差超过一个数量级"
        )
    else:
        trusted = True

    if trusted:
        return {
            "nb_frames": nb,
            "source": "probe",
            "basis": "ffprobe 实测帧数与 duration×fps 推算值一致（可信）",
            "trusted": True,
        }
    # 不可信 → 估算。
    if estimated_n is not None:
        return {
            "nb_frames": estimated_n,
            "source": "estimated",
            "basis": (
                f"{reason}；改用 duration({duration_s}s)×fps({fps}) 估算帧数≈"
                f"{estimated_n}（估算值，非实测）"
            ),
            "trusted": False,
        }
    # 既不可信又无法估算（缺 duration 或 fps）：保留原始值但明确不可信、无法估算。
    return {
        "nb_frames": nb,
        "source": "probe",
        "basis": f"{reason}；且缺少 duration/fps 无法估算，帧数仅供参考",
        "trusted": False,
    }


# 角色推断：触发词 → 角色名 + 基准置信度。
# 修正：camera_params 一律归标定（不在 _ROLE_PATTERNS 命中"RGB 相机"）；
# force/imu 等仍保留，但 hand_tracking 命名也单独识别。
_ROLE_PATTERNS: list[tuple[tuple[str, ...], str, float]] = [
    (("wrist", "wristcam"), "腕部相机", 0.8),
    (("head", "headcam", "eye"), "头部相机", 0.7),
    (("imu",), "IMU 传感器", 0.8),
    (("force", "force_torque", "ft_sensor", "wrench", "torque"), "力/力矩传感器", 0.8),
    (("action", "qpos", "qvel", "state", "obs"), "状态/动作流", 0.7),
    (("hand", "tracking"), "手部跟踪", 0.7),
]

# 方位修饰词：识别后作为角色后缀（如"腕部相机（左）"）。
_POSITION_WORDS = {
    "left": "左", "right": "右", "front": "前", "back": "后",
    "top": "上", "bottom": "下",
}


def infer_role(source: str) -> dict[str, Any]:
    """从文件名/路径推测设备角色（第 1 层词典线索，仅作参考）。

    Args:
        source: 文件路径或名称。

    Returns:
        dict，含 role、confidence（high/low）、evidence。
    """
    name = Path(source).name.lower()

    # 标定文件（camera_params*/imu_calibration 等）：明确角色为"标定"。
    if "camera_params" in name or "calibration" in name:
        return {
            "role": "标定文件",
            "confidence": "high",
            "evidence": "文件名含 camera_params/calibration，判为标定",
        }

    positions = [zh for en, zh in _POSITION_WORDS.items() if en in name]
    positions = list(dict.fromkeys(positions))

    hits: list[tuple[str, float, str]] = []
    for keys, role, conf in _ROLE_PATTERNS:
        for k in keys:
            if k in name and role not in [h[0] for h in hits]:
                hits.append((role, conf, k))

    if not hits and not positions:
        return {"role": "unknown", "confidence": "low", "evidence": "文件名无可识别模式"}

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
    "hand_tracking": "手部跟踪",
}


def _role_for_kind(kind: str, path: str, semantic_label: str | None = None) -> dict[str, Any]:
    """根据流的 kind / 语义标签给出明确角色（优先 kind，其次 semantic_label）。

    Args:
        kind: 流类型（imu/force/actions/pose/hand_tracking/unknown/video）。
        path: 文件路径（用于提取方位修饰词）。
        semantic_label: 第 2 层语义标签（可选，覆盖默认角色名）。

    Returns:
        角色 dict（role / confidence / evidence）。
    """
    if kind in _KIND_ROLE:
        name = Path(path).name.lower()
        positions = [zh for en, zh in _POSITION_WORDS.items() if en in name]
        pos_suffix = f"（{'、'.join(dict.fromkeys(positions))}）" if positions else ""
        role = semantic_label or _KIND_ROLE[kind]
        return {
            "role": f"{role}{pos_suffix}",
            "confidence": "high",
            "evidence": f"内容指纹判定为 {kind}",
        }
    return infer_role(path)


# --- 能力聚合与流登记表 ----------------------------------------------------

def build_capabilities(probe: dict[str, Any], table_sniffs: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总能力标签与推测类型。

    Args:
        probe: probe_directory 的结果。
        table_sniffs: 各表格的 classify_table_stream 结果列表（含 kind）。

    Returns:
        dict，含 capabilities、guessed_type、guessed_type_confidence、imu_confidence。
    """
    has_video = len(probe["videos"]) > 0 or any(
        v in _VIDEO_EXTS for v in probe["ext_dist"]
    )
    has_audio = len(probe["audios"]) > 0

    # 汇总表格流 kind。
    kinds = [s.get("kind") for s in table_sniffs]
    imu_present = any(k == "imu" for k in kinds)
    pose_present = any(k == "pose" for k in kinds)
    hand_present = any(k == "hand_tracking" for k in kinds)
    actions_present = any(k == "actions" for k in kinds)
    force_present = any(k == "force" for k in kinds)

    # IMU 轴数聚合：单表 klass.imu_axes（配对时为 6）取最大值。
    imu_axes_vals = [s.get("imu_axes") for s in table_sniffs
                     if isinstance(s.get("imu_axes"), int)]
    imu_axes = max(imu_axes_vals) if imu_axes_vals else None

    capabilities: dict[str, Any] = {
        "has_video_streams": has_video,
        "has_audio": has_audio,
        "has_imu": imu_present,
        "imu_axes": imu_axes,
        "has_force": force_present,
        "has_calibration": len(probe["cals"]) > 0,
        "has_actions": actions_present,
        "has_hand_tracking": hand_present,
        "has_pose": pose_present,
    }

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
        "imu_confidence": "high" if imu_present else "unknown",
    }


def build_streams_registry(
    probe: dict[str, Any],
    table_info: list[dict[str, Any]],
    video_meta: list[dict[str, Any]],
    audio_meta: list[dict[str, Any]] | None = None,
    image_meta: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """构建流登记表（供 inspect_streams 按需读取）。

    每条表格流含 {path, format, kind, channels, role, semantic_label,
    label_evidence, label_confidence, status, timestamp_column,
    quaternion_groups, imu_axes}。标定文件单独登记 kind="calibration"。
    视频/音频/图片流只登记路径与格式。

    Args:
        probe: probe_directory 的结果。
        table_info: 表格分类信息列表（含 file、name、columns、sniff、nrows）。
        video_meta: 视频元数据列表。
        audio_meta: 音频元数据列表。
        image_meta: 图片元数据列表。

    Returns:
        流登记表列表。
    """
    streams: list[dict[str, Any]] = []

    # 表格流：按 classify_table_stream 结果登记。
    for t in table_info:
        path = t["file"]
        klass = t["sniff"]  # classify_table_stream 的结果
        streams.append({
            "path": path,
            "format": Path(path).suffix.lstrip(".").lower(),
            "kind": klass.get("kind", "unknown"),
            "channels": klass.get("channels", []),
            "role": _role_for_kind(
                klass.get("kind", "unknown"), path,
                klass.get("semantic_label"),
            ),
            "semantic_label": klass.get("semantic_label"),
            "label_evidence": klass.get("label_evidence"),
            "label_confidence": klass.get("label_confidence"),
            "status": klass.get("status", "active"),
            "timestamp_column": klass.get("timestamp_column"),
            "timestamp_unit": klass.get("timestamp_unit", "unknown"),
            "timestamp_unit_basis": klass.get("timestamp_unit_basis", "未推断"),
            "quaternion_groups": klass.get("quaternion_groups", []),
            "imu_axes": klass.get("imu_axes"),
        })

    # 视频流（覆盖全部视频文件）。
    for v in video_meta:
        path = v.get("file", "unknown")
        streams.append({
            "path": path,
            "format": "video",
            "kind": "video",
            "channels": [],
            "role": infer_role(path),
            "semantic_label": "视频流",
            "label_evidence": "视频文件扩展名",
            "label_confidence": "low",
            "status": "active",
            "timestamp_column": None,
            "quaternion_groups": [],
            "imu_axes": None,
        })

    # 音频流：只登记路径与格式。
    for a in audio_meta or []:
        path = a.get("file", "unknown")
        streams.append({
            "path": path,
            "format": Path(path).suffix.lstrip(".").lower(),
            "kind": "audio",
            "channels": [],
            "role": infer_role(path),
            "semantic_label": "音频流",
            "label_evidence": "音频文件扩展名",
            "label_confidence": "low",
            "status": "active",
            "timestamp_column": None,
            "quaternion_groups": [],
            "imu_axes": None,
        })

    # 图片流：只登记路径与格式。
    for im in image_meta or []:
        path = im.get("file", "unknown")
        streams.append({
            "path": path,
            "format": Path(path).suffix.lstrip(".").lower(),
            "kind": "image",
            "channels": [],
            "role": infer_role(path),
            "semantic_label": "图片流",
            "label_evidence": "图片文件扩展名",
            "label_confidence": "low",
            "status": "active",
            "timestamp_column": None,
            "quaternion_groups": [],
            "imu_axes": None,
        })

    return streams
