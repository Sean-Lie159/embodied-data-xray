"""数据集语义画像持久化（四层架构第 4 层）。

把用户对语义识别结果的确认持久化到 **每数据集一个文件**：

    outputs/by_dataset/<数据集名净化>/profile.json

（2026-09-11 改造，见 docs/UI优化总纲与输出目录改造设计.md 3.2/3.5。）
改造前所有数据集共用一个 ``outputs/.dataset_profile.json``，多会话并发写要靠
全局锁 + 重读合并保护；拆分后**跨数据集不再竞争**，锁粒度降到"每数据集"。
再次加载该数据集时，load_dataset 优先读取画像覆盖第 1-3 层的自动识别结果。

**兼容迁移**：旧格式（单个 ``.dataset_profile.json``，按 dataset_id 索引）在
首次读取新路径未命中时**惰性迁移**——取出该 dataset_id 的分片写入新路径；
旧文件保留不删（保守，避免误删用户确认）。

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

# 画像文件名（落在 outputs/by_dataset/<净名>/profile.json）。
_PROFILE_FILENAME = "profile.json"

# 改造前的全局画像文件名（outputs/.dataset_profile.json），仅用于迁移读取。
_LEGACY_PROFILE_FILENAME = ".dataset_profile.json"

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


def _profile_path(output_dir: str, dataset_id: str) -> Path:
    """返回某数据集画像文件路径（outputs/by_dataset/<净名>/profile.json）。

    Args:
        output_dir: 项目输出目录（settings.output_dir）。
        dataset_id: 数据集标识名。

    Returns:
        该数据集画像文件的完整路径（父目录会创建）。
    """
    from app.tools.output_paths import dataset_output_dir

    return dataset_output_dir(output_dir, dataset_id) / _PROFILE_FILENAME


def _legacy_profile_path(output_dir: str) -> Path:
    """返回改造前的全局画像文件路径（outputs/.dataset_profile.json）。"""
    return Path(output_dir) / _LEGACY_PROFILE_FILENAME


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    """读取 JSON 文件为 dict；不存在/损坏/非 dict 时返回 None（安全降级）。"""
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


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


def _empty_profile() -> dict[str, Any]:
    """空画像结构（与旧全量格式同构，便于迁移与调用方兼容）。"""
    return {"schema_version": 1, "datasets": {}}


def load_profile(output_dir: str, dataset_id: str) -> dict[str, Any]:
    """读取指定数据集画像（新路径），未命中时回退旧全局文件并迁移。

    迁移语义（docs/UI优化总纲与输出目录改造设计.md 3.5）：
    1. 先读新路径 ``by_dataset/<净名>/profile.json``，命中即返回；
    2. 未命中 → 读旧全局 ``outputs/.dataset_profile.json``，取该 dataset_id 的分片；
    3. 命中旧数据 → **写入新路径**（惰性迁移）后返回；旧文件保留不删；
    4. 都没有 → 返回空画像。

    返回结构与旧格式同构（``{"schema_version":1,"datasets":{<id>:...}}``），
    使既有调用方（save/load_dataset_profile）无需大改。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。

    Returns:
        dict，含 schema_version 与 datasets（**只含本数据集**的分片）。
    """
    path = _profile_path(output_dir, dataset_id)
    obj = _read_json_dict(path)
    if obj is not None:
        obj.setdefault("schema_version", 1)
        datasets = obj.setdefault("datasets", {})
        if not isinstance(datasets, dict):
            obj["datasets"] = {}
        return obj

    # 新路径未命中 → 尝试从旧全局文件迁移本数据集的分片。
    legacy = _read_json_dict(_legacy_profile_path(output_dir))
    if legacy is not None:
        shard = (legacy.get("datasets") or {}).get(dataset_id)
        if isinstance(shard, dict):
            migrated = {"schema_version": legacy.get("schema_version", 1),
                        "datasets": {dataset_id: shard}}
            _atomic_write(path, migrated)  # 迁移落盘；失败则下次再试，不抛异常
            return migrated

    return _empty_profile()


def load_dataset_profile(output_dir: str, dataset_id: str) -> dict[str, Any]:
    """读取指定数据集的已确认画像（流映射 + 配对覆盖）。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。

    Returns:
        dict，含 streams（文件名→覆盖映射）、pairs（覆盖配对）等；无记录返回空 dict。
    """
    profile = load_profile(output_dir, dataset_id)
    return profile.get("datasets", {}).get(dataset_id, {})


def save_dataset_profile(
    output_dir: str,
    dataset_id: str,
    *,
    stream_overrides: dict[str, dict[str, Any]] | None = None,
    pair_overrides: list[dict[str, Any]] | None = None,
    dataset_type: str | None = None,
    auto_detected_type: str | None = None,
    dataset_type_note: str | None = None,
    expected_missing: dict[str, list[str]] | None = None,
    expected_missing_note: str | None = None,
    capabilities_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """写入/合并指定数据集的已确认画像，返回更新后的全量画像。

    用户确认映射的来源固定标为 user_confirmed；其余字段（内容指纹/词典）由
    load_dataset 自动识别时附带，不在本函数写入。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。
        stream_overrides: 流覆盖映射（文件名→{kind/role/semantic_label/...}）。
        pair_overrides: 配对覆盖列表。
        dataset_type: **用户确认的数据集类型**（覆盖自动识别的 unknown）。
            这是设计 B-3.2 的核心：此前 ``guessed_type`` 算完即固定，
            用户即使确认了数据性质也无处记录，报告永远显示 unknown。
        auto_detected_type: 自动识别结果（保留供对照，便于看出"确认改了什么"）。
        dataset_type_note: 用户对类型的说明（如"左右前臂不采集"）。
        expected_missing: **已确认的预期缺失**，形如
            ``{"tips_trajectory.csv": ["*forearm_pos_*"]}``（通配符或精确列名）。
            作用：质检的缺失值检查会把这些列**排除**在判定外，
            并如实标注"属已确认的预期缺失"——避免每轮都把已知的预期缺失
            报成 fail，用户反复口头解释。
        expected_missing_note: 预期缺失的原因说明。
        capabilities_override: 能力标签覆盖（人工纠正嗅探结果）。

    Returns:
        更新后的全量画像 dict（已落盘）。
    """
    path = _profile_path(output_dir, dataset_id)

    # 多会话并发保护：加锁 → **重读**（拿盘上最新状态）→ 合并 → 原子写。
    # 为什么必须重读：两个会话同时对同一数据集确认不同流时，若各自基于
    # 进入函数时的旧快照写入，后写的会丢掉先写的确认（读-改-写竞态）。
    # 注：拆分后锁粒度是"每数据集"，跨数据集不再竞争（较旧全局单文件更优）。
    with _file_lock(path):
        profile = load_profile(output_dir, dataset_id)
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

        # ---- 数据集类型确认（设计 B-3.2）----
        if dataset_type is not None:
            entry["dataset_type"] = {
                "value": str(dataset_type),
                "source": SOURCE_USER,
                "auto_detected": auto_detected_type,
                "note": dataset_type_note or "",
                "confirmed_at": _utc_now_iso(),
            }

        # ---- 已确认的预期缺失（设计 B-3.2）----
        if expected_missing is not None:
            entry["expected_missing"] = {
                "patterns": {
                    str(f): [str(p) for p in pats]
                    for f, pats in expected_missing.items()
                },
                "source": SOURCE_USER,
                "note": expected_missing_note or "",
                "confirmed_at": _utc_now_iso(),
            }

        # ---- 能力标签覆盖 ----
        if capabilities_override:
            entry["capabilities_override"] = {
                "values": dict(capabilities_override),
                "source": SOURCE_USER,
                "confirmed_at": _utc_now_iso(),
            }

        _atomic_write(path, profile)
        return profile


def _utc_now_iso() -> str:
    """当前 UTC 时间（ISO 8601，秒精度）——用于记录确认时间。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_confirmed_dataset_type(
    output_dir: str, dataset_id: str,
) -> dict[str, Any] | None:
    """读**用户确认的**数据集类型；未确认返回 None。

    Returns:
        ``{"value", "source", "auto_detected", "note", "confirmed_at"}`` 或 None。
    """
    entry = load_dataset_profile(output_dir, dataset_id)
    got = entry.get("dataset_type")
    return got if isinstance(got, dict) and got.get("value") else None


def get_expected_missing(
    output_dir: str, dataset_id: str,
) -> dict[str, Any] | None:
    """读**已确认的预期缺失**；未确认返回 None。

    Returns:
        ``{"patterns": {文件名: [模式...]}, "source", "note", "confirmed_at"}``
        或 None。
    """
    entry = load_dataset_profile(output_dir, dataset_id)
    got = entry.get("expected_missing")
    return got if isinstance(got, dict) and got.get("patterns") else None


def get_capabilities_override(
    output_dir: str, dataset_id: str,
) -> dict[str, Any] | None:
    """读**用户确认的能力标签覆盖**；未确认返回 None。"""
    entry = load_dataset_profile(output_dir, dataset_id)
    got = entry.get("capabilities_override")
    if isinstance(got, dict) and got.get("values"):
        return got["values"]
    return None


def match_expected_missing(
    patterns: dict[str, list[str]],
    filename: str,
    columns: list[str] | None = None,
) -> list[str]:
    """在指定文件的列中匹配"已确认预期缺失"的模式。

    匹配用 ``fnmatch``（支持 ``*forearm_pos_*`` 这类通配符）——
    设计决策见设计文档"待裁决决策点 3"：选通配符（简洁）**且回显命中清单**
    （让用户看到实际排除了哪些列，避免误伤无感知）。

    Args:
        patterns: ``{文件名: [模式...]}``。
        filename: 当前文件名（按 basename 匹配，兼容带路径的键）。
        columns: 该文件的列名清单；为 None 时只返回模式本身（不做列匹配）。

    Returns:
        命中该文件的模式列表（列匹配时为**实际命中的列名**）。
    """
    import fnmatch
    from pathlib import Path as _P

    base = _P(str(filename)).name
    # 先找匹配该文件的模式集合（键可以是文件名或通配）。
    pats: list[str] = []
    for key, val in (patterns or {}).items():
        if _P(str(key)).name == base or fnmatch.fnmatch(base, str(key)):
            pats.extend(str(p) for p in val)

    if not pats:
        return []
    if columns is None:
        return pats

    hit: list[str] = []
    for col in columns:
        for p in pats:
            if fnmatch.fnmatch(str(col), p):
                hit.append(str(col))
                break
    return hit


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
