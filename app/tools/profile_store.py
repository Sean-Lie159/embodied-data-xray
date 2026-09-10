"""数据集语义画像持久化（四层架构第 4 层）。

把用户对语义识别结果的确认持久化到项目 ``outputs/.dataset_profile.json``（按
dataset_id 索引，集中管理，不进数据集目录）。再次加载该数据集时，load_dataset
优先读取此文件覆盖第 1-3 层的自动识别结果。

每个映射记录来源：user_confirmed（用户确认）/ content_fingerprint（内容指纹）/
dictionary（词典）。文件不可用时（不存在/损坏）安全降级为"无覆盖"，不抛异常。

本模块为纯 Python（不 import streamlit），供 load_dataset 与用户确认入口复用。
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# 持久化文件名（落在项目 outputs/ 下，不进数据集目录）。
_PROFILE_FILENAME = ".dataset_profile.json"

# 合法来源标记。
SOURCE_USER = "user_confirmed"
SOURCE_FINGERPRINT = "content_fingerprint"
SOURCE_DICTIONARY = "dictionary"


# 跨会话写入锁的等待上限（秒）与轮询间隔。
_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.05


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """原子写 JSON：先写临时文件再 os.replace（Windows 上原子）。

    为什么需要：多会话并行确认时，直接 write_text 中途失败/并发读会留下
    **半截文件**（JSON 损坏 → 后续全部确认丢失）。

    Args:
        path: 目标路径。
        payload: 待写入的 dict。
    """
    import os
    import uuid

    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _profile_path(output_dir: str) -> Path:
    """返回持久化文件路径。

    Args:
        output_dir: 项目输出目录（settings.output_dir）。

    Returns:
        outputs/.dataset_profile.json 的完整路径。
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / _PROFILE_FILENAME


@contextmanager
def _file_lock(path: Path, timeout_s: float = _LOCK_TIMEOUT_S):
    """基于"锁文件 + O_EXCL 创建"的简单跨会话互斥。

    适用场景：本项目为**单实例本地运行**，并发来自 Streamlit 的多会话
    （同进程多线程重跑）。用锁文件（而非 threading.Lock）是因为它同时能
    防住"同机多进程"的边缘情况，且无额外依赖。

    Args:
        path: 被保护的资源路径（锁文件取其同级 .lock）。
        timeout_s: 获取锁的等待上限；超时后**放弃加锁继续执行**（不阻塞
            用户操作——数据一致性由"重读合并"兜底）。

    Yields:
        None。
    """
    import time
    import uuid

    lock = path.with_name(f"{path.name}.lock")
    deadline = time.monotonic() + timeout_s
    fd = None
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                break  # 超时：放弃加锁（重读合并仍能保住大部分一致性）
            time.sleep(_LOCK_POLL_S)
        except OSError:
            break
    try:
        if fd is not None:
            # 写入持有者标识（便于排查残留锁）。
            try:
                os.write(fd, uuid.uuid4().hex.encode())
            except OSError:
                pass
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                lock.unlink()
            except OSError:
                pass


def load_profile(output_dir: str) -> dict[str, Any]:
    """读取持久化画像（全量）。文件不存在/损坏时返回空 dict，不抛异常。

    Args:
        output_dir: 项目输出目录。

    Returns:
        dict，含 schema_version、datasets（按 dataset_id 索引的画像）。
    """
    path = _profile_path(output_dir)
    if not path.exists():
        return {"schema_version": 1, "datasets": {}}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return {"schema_version": 1, "datasets": {}}
        obj.setdefault("schema_version", 1)
        obj.setdefault("datasets", {})
        return obj
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # 损坏文件安全降级，不中断加载流程。
        return {"schema_version": 1, "datasets": {}}


def load_dataset_profile(output_dir: str, dataset_id: str) -> dict[str, Any]:
    """读取指定数据集的已确认画像（流映射 + 配对覆盖）。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。

    Returns:
        dict，含 streams（文件名→覆盖映射）、pairs（覆盖配对）等；无记录返回空 dict。
    """
    profile = load_profile(output_dir)
    return profile.get("datasets", {}).get(dataset_id, {})


def save_dataset_profile(
    output_dir: str,
    dataset_id: str,
    *,
    stream_overrides: dict[str, dict[str, Any]] | None = None,
    pair_overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """写入/合并指定数据集的已确认画像，返回更新后的全量画像。

    用户确认映射的来源固定标为 user_confirmed；其余字段（内容指纹/词典）由
    load_dataset 自动识别时附带，不在本函数写入。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。
        stream_overrides: 流覆盖映射（文件名→{kind/role/semantic_label/...}）。
        pair_overrides: 配对覆盖列表。

    Returns:
        更新后的全量画像 dict（已落盘）。
    """
    path = _profile_path(output_dir)

    # 多会话并发保护：加锁 → **重读**（拿盘上最新状态）→ 合并 → 原子写。
    # 为什么必须重读：两个会话同时对同一数据集确认不同流时，若各自基于
    # 进入函数时的旧快照写入，后写的会丢掉先写的确认（读-改-写竞态）。
    with _file_lock(path):
        profile = load_profile(output_dir)
        datasets = profile.setdefault("datasets", {})
        entry = datasets.setdefault(dataset_id, {
            "streams": {},
            "pairs": [],
        })
        streams = entry.setdefault("streams", {})
        pairs = entry.setdefault("pairs", [])

        if stream_overrides:
            # 按文件名**合并**（不动其它条目——保住其它会话已确认的流）。
            for fname, mapping in stream_overrides.items():
                rec = dict(mapping)
                rec["source"] = SOURCE_USER
                streams[fname] = rec
        if pair_overrides is not None:
            # 用户确认的配对整体覆盖（来源标 user_confirmed）。
            for p in pair_overrides:
                p = dict(p)
                p["source"] = SOURCE_USER
            entry["pairs"] = pair_overrides

        _atomic_write(path, profile)
        return profile


def apply_profile_overrides(
    streams: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """将持久化画像中的用户确认映射应用到流注册表。

    按文件名匹配，覆盖 kind / role / semantic_label / label_evidence /
    label_confidence / imu_axes 等字段，并标注 label_source="user_confirmed"，
    使 inspect_streams 能看到"该标签来自用户确认"而非自动识别。

    Args:
        streams: 流注册表（load_dataset 生成的 meta.streams）。
        profile: 指定数据集的画像（含 streams 覆盖映射）。

    Returns:
        应用覆盖后的流注册表（新列表，不修改入参）。
    """
    overrides = profile.get("streams", {})
    if not overrides:
        return streams
    result: list[dict[str, Any]] = []
    for s in streams:
        name = Path(s.get("path", "")).name
        ov = overrides.get(name)
        new_s = dict(s)
        if ov:
            for key in ("kind", "role", "semantic_label", "label_evidence",
                        "label_confidence", "imu_axes", "status", "time_column"):
                if key in ov:
                    new_s[key] = ov[key]
            new_s["label_source"] = SOURCE_USER
        else:
            new_s.setdefault("label_source", ov.get("source") if ov else None)
        result.append(new_s)
    return result
