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
# 注意：.jsonl 与 .json 是两种不同格式（前者每行一个 JSON 对象，后者整体一个
# JSON 值），必须并列登记、分别处理，不得混用。
_TABLE_EXTS = {".csv", ".json", ".jsonl", ".parquet"}

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

# 常见时间戳列名（用于第 2 层时间戳指纹）。扩展词表覆盖帧索引重组布局的
# index.parquet（pts / frame_timestamps_ns）与曝光时间戳（exposure_*）。
_TIMESTAMP_COLS = ("timestamp", "timestamp_ns", "time", "ts", "ts_ns", "t", "stamp",
                   "frame_time", "pts", "pts_us", "frame_timestamps_ns",
                   "frame_timestamp_ns", "pts_ns", "capture_utc_ns",
                   "exposure_start_utc_ns", "mid_exposure_utc_ns",
                   "exposure_duration_ns", "packet_index", "frame_index",
                   "frame_id")
# 帧序号类时间戳列（无物理时间，不参与跨流对齐残差，仅单流检查）。
_FRAME_TIMESTAMP_COLS = ("pts", "pts_us", "frame_index", "frame_id", "packet_index", "frame_number")


def classify_timestamp_column(name: str) -> str:
    """把列名分类为时间戳类型：physical / frame / ""（非时间戳）。

    Args:
        name: 列名。

    Returns:
        "physical"（物理时间，参与对齐）、"frame"（帧序号，不参与跨流对齐）、
        ""（列名不是时间戳）。
    """
    lower = str(name).lower().strip()
    if lower in _FRAME_TIMESTAMP_COLS or lower in ("frame", "packet"):
        return "frame"
    if lower in _TIMESTAMP_COLS:
        return "physical"
    return ""


def find_timestamp_columns(
    columns: list[str], sample: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """从表格列中识别主时间戳列与备选，返回结构化结果。

    识别顺序：
    1. 词表命中（_TIMESTAMP_COLS）：物理时间列优先（带 ns/us/utc 后缀或非帧序号），
       帧序号列（pts/frame_index 等）作为备选；
    2. 词表未命中但提供 sample：内容指纹回退——数值列单调递增且量级符合已知时间单位
       的视为候选，来源标 fingerprint；
    3. 都不行则无时间戳列。

    Args:
        columns: 表格列名列表。
        sample: 可选样本 DataFrame（用于内容指纹回退；不提供则只做词表匹配）。

    Returns:
        dict，含 main（主时间戳列或 None）、alternatives（备选列名列表）、
        source（"dictionary" / "fingerprint" / "none"）、frame_main（主时间戳是否为
        帧序号）、all_candidates（全部候选，含类型）。
    """
    candidates: list[dict[str, Any]] = []
    for c in columns:
        typ = classify_timestamp_column(c)
        if typ:
            candidates.append({"name": str(c), "type": typ, "source": "dictionary"})

    # 词表未命中 → 内容指纹回退（需样本）。
    if not candidates and sample is not None:
        from app.tools.timestamp_units import infer_unit

        for c in columns:
            if not pd.api.types.is_numeric_dtype(sample[c]):
                continue
            s = sample[c].dropna()
            if len(s) < 3:
                continue
            if not bool((s.diff().dropna() >= 0).all()):
                continue  # 非单调递增
            unit_info = infer_unit(s.to_numpy(), str(c))
            if unit_info["unit"] in ("s", "ms", "us", "ns"):
                candidates.append({
                    "name": str(c), "type": "physical",
                    "source": "fingerprint",
                    "unit": unit_info["unit"],
                })

    if not candidates:
        return {"main": None, "alternatives": [], "source": "none",
                "frame_main": None, "all_candidates": []}

    # 主列选择：物理时间优先（否则帧序号）；同类时按列序。
    physical = [c for c in candidates if c["type"] == "physical"]
    frame = [c for c in candidates if c["type"] == "frame"]
    ordered = physical + frame
    main = ordered[0]
    return {
        "main": main["name"],
        "alternatives": [c["name"] for c in ordered[1:]],
        "source": main["source"],
        "frame_main": main["type"] == "frame",
        "all_candidates": ordered,
    }

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


# 默认排除的目录名（版本控制 / 缓存 / 依赖等非数据目录）。这些目录不是"数据角色"，
# 普查它们只会产生噪音（如 .git/objects 可达数万文件，撑爆上下文）。
_EXCLUDED_DIR_NAMES = {
    ".git", ".svn", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", ".idea", ".vscode",
    ".DS_Store",
}
# 文件清单每组最多列出的路径条数（超出则截断并声明，避免上下文爆炸）。
_MAX_LISTED_PER_GROUP = 50


def probe_directory(root: Path, include_hidden: bool = False) -> dict[str, Any]:
    """递归普查目录，返回文件清单（按类型分组）与扩展名分布。

    普查完整性：每个文件都被归类到某个类型组；分组计数准确；**不静默抽样**——
    单组路径超过 _MAX_LISTED_PER_GROUP 时截断并显式标注 truncated/shown/total。

    默认跳过版本控制与缓存目录（.git/__pycache__/node_modules 等）：它们不是数据
    角色，且可达数万文件（曾导致返回百万 token 撑爆上下文）。如需完整普查（含
    这些目录），传 include_hidden=True。

    Args:
        root: 数据集根目录。
        include_hidden: 是否纳入默认排除的目录（默认 False）。

    Returns:
        dict，含 total_files、ext_dist、subdirs、excluded_dirs（被排除的目录名
        清单）、以及按类型分组的路径清单（每组可能为截断视图，见各组 truncated）。
    """
    ext_counter: Counter[str] = Counter()
    subdirs: list[str] = []
    excluded_dirs: list[str] = []
    grouped: dict[str, list[Path]] = {
        "tables": [], "videos": [], "audios": [], "images": [],
        "cals": [], "others": [],
    }
    total_files = 0

    def _walk(directory: Path) -> None:
        """递归遍历；遇到排除目录即剪枝（不深入、不枚举其子孙）。"""
        nonlocal total_files
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for item in entries:
            if item.is_dir():
                rel = str(item.relative_to(root))
                # 默认排除非数据目录：记录排除根（不深入），避免枚举 .git 内部数万对象。
                if not include_hidden and item.name in _EXCLUDED_DIR_NAMES:
                    excluded_dirs.append(rel)
                    continue
                subdirs.append(rel)
                _walk(item)
            elif item.is_file():
                if _is_system_file(item):
                    continue
                _classify_file(item)

    def _classify_file(item: Path) -> None:
        nonlocal total_files
        ext = item.suffix.lower()
        total_files += 1
        ext_counter[ext] += 1
        # 标定候选（json/yaml）优先归入 cals，避免被当作数据表。.json 既可能是
        # 标定也可能是数据表——由 load_dataset 在标定判定后把
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

    _walk(root)

    # 固定排序（按相对 root 的路径），再转完整路径字符串：保证确定性。
    subdirs.sort()
    for key in grouped:
        grouped[key].sort(key=lambda p: str(p.relative_to(root)))

    # 每组路径转为"截断视图"：总数准确，超出上限时标注 truncated（不静默抽样）。
    group_views: dict[str, dict[str, Any]] = {}
    for key, paths in grouped.items():
        total = len(paths)
        shown = [str(p) for p in paths[:_MAX_LISTED_PER_GROUP]]
        group_views[key] = {
            "total": total,
            "shown": len(shown),
            "truncated": total > _MAX_LISTED_PER_GROUP,
            "paths": shown,
        }

    # 完整清单（供代码内部探测使用，不进返回给模型的字段，避免上下文爆炸）。
    full_paths = {
        key: [str(p) for p in paths] for key, paths in grouped.items()
    }

    return {
        "total_files": total_files,
        "ext_dist": dict(sorted(ext_counter.items())),
        "subdirs": subdirs,
        "excluded_dirs": sorted(set(excluded_dirs)),
        "max_listed_per_group": _MAX_LISTED_PER_GROUP,
        # 公开视图（截断，供模型/返回）；完整路径见 _full。
        "tables": group_views["tables"],
        "videos": group_views["videos"],
        "audios": group_views["audios"],
        "images": group_views["images"],
        "cals": group_views["cals"],
        "others": group_views["others"],
        "_full": full_paths,
    }


def probe_full_paths(probe: dict[str, Any], group: str) -> list[str]:
    """取某类型组的**完整**路径列表（供代码探测使用，不受返回截断影响）。

    Args:
        probe: probe_directory 的返回。
        group: 组名（tables/videos/audios/images/cals/others）。

    Returns:
        完整路径列表；缺失返回 []。
    """
    full = probe.get("_full")
    if isinstance(full, dict):
        paths = full.get(group)
        if isinstance(paths, list):
            return list(paths)
    # 兜底：无 _full 时用视图 paths（旧结构或已截断）。
    view = probe.get(group)
    if isinstance(view, dict):
        paths = view.get("paths")
        return list(paths) if isinstance(paths, list) else []
    if isinstance(view, list):
        return list(view)
    return []


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

def _read_table_columns_cheap(path: str, fmt: str) -> list[str] | None:
    """只读表格列名（不读全量数据），用于配对时判断是否含时间戳列。

    与 load_dataset._read_table_columns 等价，但作为 _sniffing 内部轻量实现，
    避免配对阶段加载全量（某些采集格式的 .json 元信息可达数百 KB，只读列名是
    廉价的）。

    Args:
        path: 文件路径。
        fmt: 格式（csv/parquet/json/jsonl）。

    Returns:
        列名列表；读取失败返回 None。
    """
    import json as _json

    try:
        p = Path(path)
        if fmt == "csv":
            import pandas as pd
            return [str(c) for c in pd.read_csv(p, nrows=0, engine="python").columns]
        if fmt == "parquet":
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(p).schema.names)
        if fmt == "jsonl":
            # JSONL：逐行解析，取首个有效行的键（等价于 lines=True 的列结构）。
            from app.tools._data_access import read_jsonl_rows

            rows = read_jsonl_rows(path, limit=1, encoding="utf-8")
            return [str(k) for k in rows[0].keys()] if rows else []
        if fmt == "json":
            encoding = "utf-8"
            obj = _json.loads(p.read_text(encoding=encoding))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return list(obj[0].keys())
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list) and obj["data"]:
                return list(obj["data"][0].keys()) if isinstance(obj["data"][0], dict) else []
            return []
        return None
    except Exception:  # noqa: BLE001
        return None


def _is_frame_level_table(cols: list[str]) -> bool:
    """判断表格是否为"帧级/曝光级"时间戳表（可作为视频 metainfo）。

    判据：主时间戳列必须是帧级/曝光级——即 find_timestamp_columns 的主列为
    frame 类型（frame_index/pts 等）或以 exposure_ 开头。仅以通用 timestamp 列为主
    时间戳的普通数据表不视为 metainfo（避免把数据表当视频曝光时间戳造成假漂移）。

    Args:
        cols: 表格列名列表。

    Returns:
        是帧级/曝光级时间戳表返回 True。
    """
    ts_info = find_timestamp_columns(cols)
    main = ts_info.get("main")
    if main is None:
        return False
    lower_main = str(main).lower().strip()
    # 主时间戳列本身是帧序号（frame/pts/index）或曝光时间戳。
    if ts_info.get("frame_main"):
        return True
    if lower_main.startswith("exposure_"):
        return True
    if lower_main in ("frame_index", "frame_id", "pts", "pts_us", "frame"):
        return True
    if lower_main.startswith("frame_"):
        return True
    return False


def _rows_match_frames(path: str, fmt: str, video_frames: int) -> bool:
    """判断表格行数是否与视频帧数相符（同量级，容忍 10% 偏差）。

    Args:
        path: 表格文件路径。
        fmt: 格式。
        video_frames: 视频帧数。

    Returns:
        行数可判定且与帧数相符返回 True；行数不可判定返回 True（不因未知而误拒）。
    """
    from app.tools import _data_access

    rows = _data_access.read_table_nrows(path, fmt)
    if rows is None:
        return True  # 行数未知，不因无法判定而误拒
    if video_frames <= 0:
        return True
    return abs(rows - video_frames) / video_frames < 0.1


def pair_streams(
    video_files: list[str],
    table_files: list[str],
    audio_files: list[str],
    video_frame_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """流配对规则。

    规则 1：媒体 ↔ metainfo（语义角色匹配）。视频/音频文件与其**同 stem** 的表格文件
        （任意已支持格式 csv/parquet/json，含 *_metainfo.csv 命名与同名前缀）配对——
        若该表格列名含可识别时间戳列则建立 metainfo 配对；_metainfo.csv 命名降级为
        线索之一而非唯一路径。同 stem 多候选（.json 与 .index.parquet 并存）全部登记，
        其中含物理时间戳（非帧序号）的优先参与对齐，选择依据透出。
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

    # 规则 1：媒体 ↔ metainfo（语义角色匹配）。媒体与同 stem 表格配对，若表格列名含
    # 可识别时间戳列则建 metainfo 配对；_metainfo.csv 命名降级为线索之一。
    media_files = [f for f in (video_files + audio_files)]
    for media in media_files:
        mp = Path(media)
        stem = mp.stem  # 去扩展名（含主版本名）
        suffix = mp.suffix.lower()
        if suffix not in (".mp4", ".m4a", ".avi", ".mov", ".mkv", ".webm"):
            continue
        media_kind = "video" if suffix == ".mp4" else "audio"
        # 同 stem 表格候选：<stem> 前缀的表格文件（含 _metainfo.csv 与同名不同扩展）。
        same_stem = [
            t for t in table_files
            if Path(t).stem.startswith(stem) or stem.startswith(Path(t).stem)
        ]
        # 判定每个候选是否含可识别时间戳列。_metainfo.csv 是命名线索（快速路径，无需
        # 读文件即视为 metainfo）；其余同 stem 表格需读列名确认含时间戳列。
        # 附加判据：候选须为"帧级/曝光级"时间戳表（命名线索或含 frame/exposure/pts 列），
        # 普通数据表（如 episode_*.parquet 的 timestamp 列）不作为视频 metainfo；
        # 且行数需与视频帧数相符（能拿到帧数时），避免把数据表当曝光时间戳造成假漂移。
        metainfo_candidates: list[dict[str, Any]] = []
        video_frames = video_frame_counts.get(media) if video_frame_counts else None
        for t in same_stem:
            tname = Path(t).name
            fmt = Path(t).suffix.lstrip(".").lower()
            if tname.endswith("_metainfo.csv"):
                # _metainfo.csv 命名即线索：默认为 metainfo（帧序号/曝光时间戳类）。
                ts_info = {"main": None, "frame_main": True, "source": "dictionary"}
                if video_frames is not None and not _rows_match_frames(t, fmt, video_frames):
                    continue  # 行数与视频帧数不符，不配（避免数据表当 metainfo）
                metainfo_candidates.append({
                    "path": t, "format": fmt, "ts_info": ts_info,
                    "by_naming": True,
                })
                continue
            cols = _read_table_columns_cheap(t, fmt)
            if cols is None:
                continue
            ts_info = find_timestamp_columns(cols)
            if ts_info["main"] is None:
                continue  # 列名无时间戳（未做内容指纹，配对阶段保持廉价）
            if not _is_frame_level_table(cols):
                continue  # 普通数据表（仅 timestamp 列、无 frame/exposure/index 特征）不配
            if video_frames is not None and not _rows_match_frames(t, fmt, video_frames):
                continue  # 行数与视频帧数不符
            metainfo_candidates.append({
                "path": t,
                "format": fmt,
                "ts_info": ts_info,
            })
        if not metainfo_candidates:
            continue
        # 多候选全部登记；含物理时间戳（非帧序号）的优先参与对齐。
        metainfo_candidates.sort(key=lambda c: (c["ts_info"]["frame_main"], c["path"]))
        primary = next((c for c in metainfo_candidates if not c["ts_info"]["frame_main"]), metainfo_candidates[0])
        pairs.append({
            "type": "media_metainfo",
            "media": media,
            "media_kind": media_kind,
            "metainfo": primary["path"],
            "metainfo_format": primary["format"],
            "timestamp_column": primary["ts_info"]["main"],
            "timestamp_source": primary["ts_info"]["source"],
            "all_candidates": [
                {"path": c["path"], "timestamp_column": c["ts_info"]["main"],
                 "frame_only": c["ts_info"]["frame_main"]}
                for c in metainfo_candidates
            ],
            "source": "content_fingerprint",
            "evidence": (
                f"{media_kind} {stem} 与同 stem 表格 "
                f"{', '.join(Path(c['path']).name for c in metainfo_candidates)} 配对；"
                f"主时间戳列 {primary['ts_info']['main']}"
                f"（来源 {primary['ts_info']['source']}"
                + ("，帧序号，不参与跨流对齐" if primary["ts_info"]["frame_main"] else "，物理时间，参与对齐")
                + "）。_metainfo.csv 命名仅为线索之一，现按语义角色匹配。"
            ),
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

    # 规则 3：多分辨率/预览版本组（登记级）。xxx.mp4 / xxx_480.mp4 / xxx_960.mp4 /
    # xxx_pre.mp4 / xxx_pre.webp 归为同一视频源的不同版本。主版本 = 无分辨率/预览
    # 后缀的文件；变体标 variant_of 指向主版本。只做登记，不做版本内容合并。
    _video_base_groups = _group_video_versions(video_files)
    for base, members in _video_base_groups.items():
        if len(members) < 2:
            continue
        main = next((m for m in members if _video_is_main_version(m)), None)
        variants = [
            {"path": m, "variant_of": main} for m in members
            if m != main and main is not None
        ]
        pairs.append({
            "type": "video_version_group",
            "base": base,
            "main": main,
            "members": members,
            "variants": variants,
            "source": "content_fingerprint",
            "evidence": f"视频 {base} 存在多分辨率/预览版本：{', '.join(Path(m).name for m in members)}，"
                        "归为同一视频源版本组（登记级，不做版本合并）",
        })

    # 规则 4：同名不同扩展配对（metainfo 规则推广）。<name>.mp4 ↔ <name>.json
    # （如相机同名 json 840KB 与 mp4 配对）；仅登记，供后续关联。
    for media in video_files + audio_files:
        mp = Path(media)
        media_stem = mp.stem
        same_name_json = next(
            (t for t in table_files if Path(t).stem == media_stem and Path(t).suffix.lower() == ".json"),
            None
        )
        if same_name_json is not None:
            pairs.append({
                "type": "media_samename",
                "media": media,
                "json": same_name_json,
                "source": "dictionary",
                "evidence": f"{mp.name} 与 {Path(same_name_json).name} 同名配对（推广 metainfo 规则），"
                            "JSON 与该媒体流同源登记",
            })

    return pairs


# 视频多分辨率/预览版本后缀（登记级识别，用于版本组配对）。
_VIDEO_VARIANT_SUFFIXES = ("_480", "_720", "_960", "_1080", "_2k", "_4k", "_pre", "_preview", "_thumb", "_hd")


def _video_is_main_version(path: str) -> bool:
    """判断视频文件是否为主版本（文件名不含分辨率/预览后缀）。

    Args:
        path: 视频文件路径。

    Returns:
        主版本返回 True。
    """
    name = Path(path).stem.lower()
    return not any(name.endswith(suf) for suf in _VIDEO_VARIANT_SUFFIXES)


def _group_video_versions(video_files: list[str]) -> dict[str, list[str]]:
    """把视频按"基础名"（去掉版本后缀）分组，识别同一视频源的不同版本。

    Args:
        video_files: 视频文件路径列表。

    Returns:
        {基础名: [成员路径...]}；基础名 = 文件名去掉版本后缀后的部分（含视频扩展名
        区分，如 xxx 与 xxx_480 归一组）。
    """
    groups: dict[str, list[str]] = {}
    for vf in video_files:
        p = Path(vf)
        base = _video_base_name(p)
        groups.setdefault(base, []).append(vf)
    return groups


def _video_base_name(path: Path) -> str:
    """返回视频的基础名（去掉 _480/_960/_pre 等版本后缀，保留视频扩展名）。"""
    stem = path.stem.lower()
    for suf in _VIDEO_VARIANT_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return f"{stem}{path.suffix.lower()}"


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

    if not hits:
        # 无角色命中时不得对空 hits 做下标访问（hits[0] 会 IndexError）。
        # 即使有方位词（如 left/right），也先判 unknown，方位词仅作说明附加。
        pos_suffix = f"（{'、'.join(positions)}）" if positions else ""
        return {
            "role": f"unknown{pos_suffix}" if positions else "unknown",
            "confidence": "low",
            "evidence": "文件名无可识别模式" + (f"；方位词 {'、'.join(positions)}" if positions else ""),
        }

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

def _is_lerobot_layout(probe: dict[str, Any]) -> bool:
    """判定目录是否为 LeRobot v2 布局。

    指纹：meta/info.json + data/chunk-* 或 videos/chunk-*（LeRobot v2 的元数据与
    chunk 数据/视频目录）。

    Args:
        probe: probe_directory 的结果。

    Returns:
        是 LeRobot v2 布局返回 True。
    """
    paths = (
        probe_full_paths(probe, "tables")
        + probe_full_paths(probe, "videos")
        + probe_full_paths(probe, "cals")
    )
    rel = [Path(p).as_posix() for p in paths]
    has_info = any("/meta/info.json" in p or p.endswith("meta/info.json") for p in rel)
    has_data_chunk = any("/data/chunk-" in p for p in rel)
    has_video_chunk = any("/videos/chunk-" in p for p in rel)
    return has_info and (has_data_chunk or has_video_chunk)


# 元数据配置型键（小写）：命中即视为数据集元数据（与标定键区分）。
_METADATA_CONFIG_KEYS = {
    "fps", "robot_type", "features", "code_keys", "video", "tasks",
    "total_episodes", "total_frames", "total_videos", "chunks_size",
    "data_files_size_in_mb", "videos_size_in_mb",
}
# 元数据文件的尺寸上限：超过此值视为数据表而非配置。
_METADATA_MAX_BYTES = 100_000
# 标定键（小写副本）：含任一即归标定角色，不参与元数据判定。
_CALIB_KEYS_LOWER = {k.lower() for k in _CALIB_KEYS}
# JSON 行列表键：顶层 dict 中这些键的值为"行记录列表"（部分格式用 frames，
# 部分用 data），应展开为表格；其余标量键（如 fps）不是数据行。
_JSON_ROW_LIST_KEYS = ("frames", "data")


def _is_dataset_metadata_content(path: str) -> bool:
    """按内容特征判定 JSON 是否为数据集元数据（语义角色，不依赖目录布局）。

    判据（全部满足）：可解析为 dict、尺寸 < 100KB、含至少一个配置型键
    （fps/robot_type/features 等）、不含标定型键（intrinsics/extrinsics/matrix/
    bias 等——含标定键的归标定角色，如 imu_calibration.json）。

    Args:
        path: 文件路径。

    Returns:
        是数据集元数据返回 True。
    """
    import json as _json

    p = Path(path)
    if p.suffix.lower() != ".json":
        return False
    try:
        if p.stat().st_size > _METADATA_MAX_BYTES:
            return False
        obj = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(obj, dict):
        return False
    keys = {str(k).lower() for k in obj.keys()}
    # 含标定键 → 标定角色，不是元数据。
    if keys & _CALIB_KEYS_LOWER:
        return False
    # 含实质性行列表（frames/data 非空）→ 数据表角色（如 LeRobot episode json，
    # 与同名 parquet 内容一致），不是元数据。
    for key in _JSON_ROW_LIST_KEYS:
        v = obj.get(key)
        if isinstance(v, list) and v:
            return False
    # 命中配置型键 → 元数据角色。
    if keys & _METADATA_CONFIG_KEYS:
        return True
    return False


def _is_dataset_metadata_file(path: str) -> bool:
    """判定文件是否为数据集元数据角色（不进流清单/不参与对齐）。

    语义角色判据：内容特征（小尺寸 + 配置型键 + 无标定键）。``meta/`` 目录布局
    仅是线索之一：LeRobot 的 info.json 通常在 meta/ 下，但目录名不作排他判据，
    以内容为准。

    Args:
        path: 文件路径。

    Returns:
        是数据集元数据返回 True。
    """
    p = Path(path).as_posix()
    if not p.endswith(".json"):
        return False
    if _is_dataset_metadata_content(path):
        return True
    # 目录线索兜底：meta/ 下的小 JSON 且为"配置型结构"（含配置键、无实质行列表、
    # 无标定键）时按元数据角色处理；纯数据表（含 frames 等非空行列表）不在此列。
    if "/meta/" in p:
        import json as _json

        try:
            pp = Path(path)
            if pp.stat().st_size > _METADATA_MAX_BYTES:
                return False
            obj = _json.loads(pp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(obj, dict):
            return False
        keys = {str(k).lower() for k in obj.keys()}
        if keys & _CALIB_KEYS_LOWER:
            return False
        for key in _JSON_ROW_LIST_KEYS:
            v = obj.get(key)
            if isinstance(v, list) and v:
                return False
        return bool(keys & _METADATA_CONFIG_KEYS) or len(keys) <= 3
    return False


def parse_lerobot_info(info_path: str) -> dict[str, Any]:
    """解析 LeRobot meta/info.json 的语义元数据（确定性解析，仅供工具引用）。

    提取内容（全部为确定性字段，不含推测）：
    - fps / video.fps：帧率（fps 为视频帧率；数据表降采样后频率可能不同）；
    - features：展开为"列名 → dtype/shape/names"的列语义表（names 是该列各
      维度的权威定义，如 head_pose 的 ['px','py','pz','qx','qy','qz','qw']）；
    - hand_tracked / robot_type / coordinate_frame / task / total_frames /
      total_episodes / source：数据集级语义（hand_tracked 可直接终结"手部数据
      是否采集"的推测）。

    Args:
        info_path: meta/info.json 路径。

    Returns:
        dict，含 fps、video_fps、features（原始）、column_semantics（列语义表）、
        以及数据集级语义字段；解析失败返回 {}。
    """
    import json as _json

    try:
        obj = _json.loads(Path(info_path).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {}
        result: dict[str, Any] = {}
        if "fps" in obj:
            result["fps"] = obj.get("fps")
        video = obj.get("video") if isinstance(obj.get("video"), dict) else None
        if video and "fps" in video:
            result["video_fps"] = video.get("fps")

        # features → 列语义表（列名 → dtype/shape/names）。
        features = obj.get("features")
        if isinstance(features, dict):
            result["features"] = features
            semantics: dict[str, dict[str, Any]] = {}
            for col, spec in features.items():
                if not isinstance(spec, dict):
                    continue
                names = spec.get("names")
                semantics[str(col)] = {
                    "dtype": spec.get("dtype"),
                    "shape": spec.get("shape"),
                    "names": list(names) if isinstance(names, (list, tuple)) else None,
                }
            result["column_semantics"] = semantics

        # 数据集级语义字段（确定性，直接取）。
        for key in ("hand_tracked", "robot_type", "coordinate_frame", "task",
                    "total_frames", "total_episodes", "source", "repo_id"):
            if key in obj:
                result[key] = obj.get(key)
        return result
    except Exception:  # noqa: BLE001
        return {}


def parse_block_declaration(info: dict[str, Any], column: str) -> dict[str, Any] | None:
    """解析向量列的"块分解"声明（如 body_24x7 = 24 块 × 7DoF）。

    仅当数据集**显式声明**且自洽时才认定：names 为单一组合名且匹配 `NxM` 模式，
    且 N*M == shape[0]。不做任何形态猜测——不满足即返回 None（调用方不得推测）。

    Args:
        info: parse_lerobot_info 的返回。
        column: 列名。

    Returns:
        dict，含 block_count / dof_per_block / declared_name；无声明返回 None。
    """
    import re as _re

    semantics = info.get("column_semantics")
    if not isinstance(semantics, dict):
        return None
    spec = semantics.get(str(column))
    if not isinstance(spec, dict):
        return None
    names = spec.get("names")
    shape = spec.get("shape")
    if not isinstance(names, list) or len(names) != 1:
        return None  # 逐维声明（如 head_pose）或无名 → 不是块分解声明。
    if not isinstance(shape, list) or not shape or not isinstance(shape[0], int):
        return None
    m = _re.fullmatch(r"([A-Za-z0-9_]+)_(\d+)x(\d+)", str(names[0]))
    if not m:
        return None
    blocks, dof = int(m.group(2)), int(m.group(3))
    if blocks * dof != shape[0]:
        return None  # 声明不自洽 → 不采信（不猜测）。
    return {
        "block_count": blocks,
        "dof_per_block": dof,
        "declared_name": str(names[0]),
        "declaration_source": "meta/info.json features.names（数据集声明）",
    }


def infer_dof_order(info: dict[str, Any], column: str, dof: int) -> dict[str, Any]:
    """推断块内 dof 维的排列顺序（参照同数据集的逐维声明列，如 head_pose）。

    full_body/left_hand 的 names 只有组合名（body_24x7），**没有逐维名**；而同一
    数据集的 head_pose 有逐维声明 ['px','py','pz','qx','qy','qz','qw']。因此块内
    "前 3 位置 + 后 4 四元数" 属**参照推断**（有依据但非本列直接声明），必须标注。

    Args:
        info: parse_lerobot_info 的返回。
        column: 列名（用于说明）。
        dof: 每块自由度数。

    Returns:
        dict，含 order（维度角色列表）、source（来源说明）、is_inferred（是否推断）。
    """
    # 优先：本列自身有逐维声明 → 直接采用（非推断）。
    own = column_dimension_names(info, column)
    if own and len(own) == dof:
        return {
            "order": list(own),
            "source": f"本列 meta 声明（{column}.names）",
            "is_inferred": False,
        }
    # 次选：参照同数据集中 dof 相同且带逐维名的列（如 head_pose 7 维）。
    semantics = info.get("column_semantics")
    if isinstance(semantics, dict):
        for other, spec in semantics.items():
            if str(other) == str(column) or not isinstance(spec, dict):
                continue
            other_names = spec.get("names")
            other_shape = spec.get("shape")
            if (
                isinstance(other_names, list)
                and len(other_names) == dof
                and isinstance(other_shape, list)
                and other_shape
                and other_shape[0] == dof
            ):
                return {
                    "order": list(other_names),
                    "source": (
                        f"参照同数据集 {other} 的逐维声明推断"
                        "（本列 names 仅为组合名，非直接声明）"
                    ),
                    "is_inferred": True,
                }
    # 无依据：返回占位，标注未声明（调用方不得当作位置/四元数解读）。
    return {
        "order": [f"dim_{i}" for i in range(dof)],
        "source": "数据集未声明维度顺序（占位，不得解读为位置/四元数）",
        "is_inferred": True,
    }


def parse_lerobot_stats(stats_path: str) -> dict[str, Any]:
    """解析 LeRobot meta/stats.json 的每列统计量（确定性，直接采用不重算）。

    stats.json 结构为 {"computed_on": ..., "features": {列名: {min/max/mean/std...}}}。
    直接采用可避免重复计算，且比本地重算更权威（由数据集生产者计算）。

    Args:
        stats_path: meta/stats.json 路径。

    Returns:
        dict，含 computed_on 与 features（列名 → 统计量）；解析失败返回 {}。
    """
    import json as _json

    try:
        obj = _json.loads(Path(stats_path).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {}
        result: dict[str, Any] = {}
        if "computed_on" in obj:
            result["computed_on"] = obj.get("computed_on")
        features = obj.get("features")
        if isinstance(features, dict):
            result["features"] = features
        return result
    except Exception:  # noqa: BLE001
        return {}


def column_dimension_names(info: dict[str, Any], column: str) -> list[str] | None:
    """取某列各维度的权威名称（来自 info.json 的 features.names）。

    Args:
        info: parse_lerobot_info 的返回。
        column: 列名。

    Returns:
        维度名列表；无声明返回 None（此时不得推测维度含义）。
    """
    semantics = info.get("column_semantics")
    if not isinstance(semantics, dict):
        return None
    spec = semantics.get(str(column))
    if not isinstance(spec, dict):
        return None
    names = spec.get("names")
    return list(names) if isinstance(names, list) else None


def explain_fps_mismatch(table_rows: int, video_frames: int, info: dict[str, Any]) -> dict[str, Any] | None:
    """用 info.json 的 fps 解释视频帧数与表行数不一致（确定性说明）。

    当 video.fps 与数据 fps 存在比例关系且能解释帧数差异时，给出确定性结论。

    Args:
        table_rows: 表格行数。
        video_frames: 视频帧数（约）。
        info: parse_lerobot_info 的结果。

    Returns:
        解释 dict（含 ratio、note）或 None（无法解释）。
    """
    data_fps = info.get("fps")
    video_fps = info.get("video_fps")
    if not (data_fps and video_fps and data_fps > 0 and video_fps > 0):
        return None
    ratio = round(video_fps / data_fps, 3)
    expected_frames = round(table_rows * ratio)
    if abs(expected_frames - video_frames) / max(video_frames, 1) < 0.1:
        return {
            "data_fps": data_fps,
            "video_fps": video_fps,
            "ratio": ratio,
            "expected_video_frames": expected_frames,
            "note": f"视频 {video_fps}fps vs 数据 {data_fps}Hz，比例 {ratio} 吻合，正常",
        }
    return None


def detect_episode_mirrors(probe: dict[str, Any]) -> list[dict[str, Any]]:
    """检测 episode JSON 与其同名 parquet 的镜像（同一数据的 JSON 镜像）。

    LeRobot 中 episode_XXXXXX.json 与同名 parquet 行数一致 → 标 mirror，不参与对齐。

    Args:
        probe: probe_directory 的结果。

    Returns:
        list[dict]，每项含 json、parquet、rows（行数）。
    """
    import re as _re

    from app.tools import _data_access

    # episode JSON 可能登记在 cals（json 路由），parquet 在 tables；都纳入（用完整
    # 路径，不受返回截断影响）。
    all_files = probe_full_paths(probe, "tables") + probe_full_paths(probe, "cals")
    ep_json = {}
    ep_parquet = {}
    for t in all_files:
        p = Path(t)
        if _re.match(r"episode_\d+\.json", p.name):
            ep_json[p.name] = t
        elif _re.match(r"episode_\d+\.parquet", p.name):
            ep_parquet[p.name] = t
    mirrors: list[dict[str, Any]] = []
    for name, jpath in ep_json.items():
        pname = name.replace(".json", ".parquet")
        ppath = ep_parquet.get(pname)
        if ppath is None:
            continue
        jrows = _data_access.read_table_nrows(jpath, "json")
        prows = _data_access.read_table_nrows(ppath, "parquet")
        if jrows is not None and prows is not None and jrows == prows:
            mirrors.append({
                "type": "episode_mirror",
                "json": jpath, "parquet": ppath, "rows": jrows,
                "source": "content_fingerprint",
                "evidence": f"episode JSON {Path(jpath).name} 与 parquet {Path(ppath).name} 行数一致（{jrows}），为同一数据镜像",
            })
    return mirrors


def detect_dataset_format(probe: dict[str, Any]) -> str:
    """识别数据集格式（当前仅 LeRobot v2）。

    Args:
        probe: probe_directory 的结果。

    Returns:
        格式名（"lerobot" / "unknown"）。
    """
    if _is_lerobot_layout(probe):
        return "lerobot"
    return "unknown"


def build_capabilities(probe: dict[str, Any], table_sniffs: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总能力标签与推测类型。

    Args:
        probe: probe_directory 的结果。
        table_sniffs: 各表格的 classify_table_stream 结果列表（含 kind）。

    Returns:
        dict，含 capabilities、guessed_type、guessed_type_confidence、imu_confidence。
    """
    has_video = len(probe_full_paths(probe, "videos")) > 0 or any(
        v in _VIDEO_EXTS for v in probe["ext_dist"]
    )
    has_audio = len(probe_full_paths(probe, "audios")) > 0

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

    dataset_format = detect_dataset_format(probe)
    capabilities: dict[str, Any] = {
        "has_video_streams": has_video,
        "has_audio": has_audio,
        "has_imu": imu_present,
        "imu_axes": imu_axes,
        "has_force": force_present,
        "has_calibration": len(probe_full_paths(probe, "cals")) > 0,
        "has_actions": actions_present,
        "has_hand_tracking": hand_present,
        "has_pose": pose_present,
        "dataset_format": dataset_format,
    }

    guessed_type = "unknown"
    conf = 0.0
    if dataset_format == "lerobot":
        guessed_type, conf = "LeRobot", 0.8
    elif has_video and imu_present and pose_present:
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
