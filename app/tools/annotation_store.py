"""标注存储与规范化层（格式无关的锚点抽象 + 读写 + 版本管理）。

设计依据：``docs/标注与质检能力设计.md`` 第 4 节。

**为什么需要这一层**：目标数据集有三种格式（LeRobot v2 / 多流采集目录 /
HDF5），三者的 episode 语义互不相同——LeRobot 靠 ``episode_index`` 列、
HDF5 靠容器节点路径、多流目录只能猜列名（``episode`` / ``traj_id`` …）或
整段视为一个 episode。若不做抽象，切片边界检测、标注落盘、质检三处都要
写格式分支（共 9 个点），且行为必然漂移。

本模块把格式差异收敛到 :func:`resolve_anchors` 一处，其余全链路格式无关
（贯彻 ``docs/ARCHITECTURE.md`` §3.3.1「语义角色优先，不绑定物理形态」）。

**落盘结构**（``outputs/by_dataset/<净名>/annotations/``）::

    annotations/
    ├── tasks.jsonl              # 当前态（任务级），每 episode 一行
    ├── segments.jsonl           # 当前态（切片级），每片段一行
    ├── revisions/
    │   ├── revisions.jsonl      # 追加式修订日志（只增不改）
    │   └── snapshots/           # 交付前冻结的快照
    └── qc/<check>_<ts>.json     # 质检结果（由质检工具写入）

**关键纪律**（与项目既有约定一致）：

- **来源分级**：每条标注的 ``source`` 取值受限词表（``user_confirmed`` /
  ``signal_derived`` / ``llm_proposed`` / ``imported``），且**用途合规性**
  由此决定——``llm_proposed`` 不得进入训练真值路径（见 :func:`check_source_for_use`）。
- **只读源数据**：本模块只写 ``outputs/``，绝不改动源数据集。
- **绝不静默截断**：修订日志超软上限时**如实报告**体积并提示归档。
- **原子写 + 跨会话锁**：与 ``profile_store`` 同款（原子替换 + 锁文件 + 重读合并）。

本模块为纯 Python（不 import streamlit），可单测。
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.agent.context import RunContext

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 标注目录名（落在 outputs/by_dataset/<净名>/ 下）。
_ANNOTATION_SUBDIR = "annotations"
_TASKS_FILENAME = "tasks.jsonl"
_SEGMENTS_FILENAME = "segments.jsonl"
_REVISIONS_SUBDIR = "revisions"
_REVISIONS_FILENAME = "revisions.jsonl"
_SNAPSHOTS_SUBDIR = "snapshots"

# 标注作用域（scope）。
SCOPE_TASK = "task"
SCOPE_SEGMENT = "segment"
_VALID_SCOPES = (SCOPE_TASK, SCOPE_SEGMENT)

# 来源分级（受限词表）。用途合规性由 check_source_for_use 判定。
SOURCE_USER = "user_confirmed"
SOURCE_SIGNAL = "signal_derived"
SOURCE_LLM = "llm_proposed"
SOURCE_IMPORTED = "imported"
_VALID_SOURCES = (SOURCE_USER, SOURCE_SIGNAL, SOURCE_LLM, SOURCE_IMPORTED)

# 可信度分级。
_VALID_CONFIDENCES = ("high", "medium", "low")

# 用途（决定来源是否合规）。
USE_TRAINING = "training"          # 下游训练真值
USE_PUBLICATION = "publication"    # 数据集发布元数据
USE_AUDIT = "audit"                # 内部数据清查
_VALID_USES = (USE_TRAINING, USE_PUBLICATION, USE_AUDIT)

# episode_key 兜底值：无任何划分线索时整段视为一个 episode。
WHOLE_DATASET_KEY = "__all__"

# 锚点来源标记。
ANCHOR_LEROBOT = "lerobot_index"
ANCHOR_EPISODE_COLUMN = "episode_column"
ANCHOR_H5_NODE = "h5_node"
ANCHOR_WHOLE = "whole_dataset"

# 版本管理参数。
# 修订日志软上限（行）：超出时如实报告并提示归档（**不静默截断**）。
_REVISION_SOFT_LIMIT = 50_000

# 跨会话写入锁的等待上限（秒）与轮询间隔（与 profile_store 同款）。
_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.05

# episode 列候选（与 _data_access._EPISODE_COLS 同源；此处独立列出以免
# 因 _data_access 的取值变更而静默影响标注锚点语义）。
_EPISODE_COL_CANDIDATES = (
    "episode", "ep", "eps", "episode_id", "traj_id", "trajectory_id",
    # LeRobot v2 的 episode 列就叫 episode_index——必须在候选内，否则
    # LeRobot 数据集会被误判为"无 episode 划分"而降级为整段单元。
    "episode_index", "episode_idx", "traj_index", "trajectory_index",
)
# 时间列候选（用于把秒映射回帧；词表命中优先）。
_TIME_COL_CANDIDATES = (
    "timestamp", "time", "ts", "frame_timestamp", "exposure_time",
    "timestamp_us", "timestamp_ns", "timestamp_ms", "time_us", "time_ns",
)
# fps 列候选（取每 episode 的帧率，用于帧↔秒换算）。
_FPS_COL_CANDIDATES = ("fps", "frame_rate", "rate_hz")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeAnchor:
    """一个 episode 的统一锚点（格式无关）。

    为什么用 frozen dataclass：锚点会被多处读取并作为落盘索引键，不可变可
    避免下游无意修改导致"标注对不上数据"。

    Attributes:
        episode_key: 稳定标识。LeRobot 用 ``ep000123``；h5 用净化后的节点
            路径；多流目录用 episode 列值；无可划分时用 ``__all__``。
        label: 面向用户的展示名（如 ``episode 123`` / ``容器节点 obs/...``）。
        start_s: 该 episode 起始秒（相对该 episode/流起点）；不可得为 None。
        end_s: 结束秒；不可得为 None。
        start_frame: 起始帧号；不可得为 None。
        end_frame: 结束帧号；不可得为 None。
        n_frames: 帧数（-1 表示不可得）。
        time_column: 该 episode 的时间列名（用于秒↔帧映射）；无则 None。
        anchor_source: 锚点判定来源（见 ``ANCHOR_*`` 常量）。
        evidence: 判定依据（必须可转述，供模型如实说明"锚点是怎么来的"）。
    """

    episode_key: str
    label: str
    start_s: float | None = None
    end_s: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    n_frames: int = -1
    time_column: str | None = None
    anchor_source: str = ANCHOR_WHOLE
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（供工具返回与测试断言）。"""
        return {
            "episode_key": self.episode_key,
            "label": self.label,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "n_frames": self.n_frames,
            "time_column": self.time_column,
            "anchor_source": self.anchor_source,
            "evidence": self.evidence,
        }

    @property
    def duration_s(self) -> float | None:
        """该 episode 时长（秒）；起止不可得时返回 None（不猜）。"""
        if self.start_s is None or self.end_s is None:
            return None
        return max(0.0, self.end_s - self.start_s)


# ---------------------------------------------------------------------------
# 路径与磁盘辅助
# ---------------------------------------------------------------------------


def annotation_dir(output_dir: str, dataset_id: str | None) -> Path:
    """返回某数据集的标注目录（``.../by_dataset/<净名>/annotations/``），不存在则创建。

    Args:
        output_dir: 项目输出目录（settings.output_dir / RunContext.output_dir）。
        dataset_id: 数据集标识名；None/空 → 走 ``_misc`` 兜底（与其它产物一致）。

    Returns:
        标注目录（Path）。
    """
    from app.tools.output_paths import dataset_output_dir

    d = dataset_output_dir(output_dir, dataset_id) / _ANNOTATION_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _revisions_dir(output_dir: str, dataset_id: str | None) -> Path:
    """修订日志目录（``.../annotations/revisions/``），不存在则创建。"""
    d = annotation_dir(output_dir, dataset_id) / _REVISIONS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshots_dir(output_dir: str, dataset_id: str | None) -> Path:
    """快照目录（``.../annotations/revisions/snapshots/``），不存在则创建。"""
    d = _revisions_dir(output_dir, dataset_id) / _SNAPSHOTS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scope_path(output_dir: str, dataset_id: str | None, scope: str) -> Path:
    """某作用域的当前态文件路径。"""
    name = _TASKS_FILENAME if scope == SCOPE_TASK else _SEGMENTS_FILENAME
    return annotation_dir(output_dir, dataset_id) / name


def _now_iso() -> str:
    """当前 UTC 时间的 ISO 8601 字符串（秒精度，带 Z 后缀）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    """原子写文本：先写临时文件再 ``os.replace``（Windows 上原子）。

    为什么需要：多会话并行写标注时，直接 ``write_text`` 中途失败/并发读会
    留下**半截文件**（JSONL 尾部残缺 → 后续全部解析失败）。
    """
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_append_lines(path: Path, lines: list[str]) -> None:
    """原子追加多行：读出全部现有内容 → 拼接 → 原子写回。

    为什么不用 ``open(path, "a")``：追加不是原子操作，并发或中途失败会留下
    残缺行。标注（尤其修订日志）必须要么完整写入、要么完全不变。
    """
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            existing = ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    payload = existing + "\n".join(lines)
    if payload and not payload.endswith("\n"):
        payload += "\n"
    _atomic_write_text(path, payload)


@contextmanager
def _file_lock(path: Path, timeout_s: float = _LOCK_TIMEOUT_S):
    """基于"锁文件 + O_EXCL 创建"的简单跨会话互斥（与 profile_store 同款）。

    适用场景：本项目为**单实例本地运行**，并发来自 Streamlit 的多会话
    （同进程多线程重跑）。用锁文件（而非 threading.Lock）同时能防住"同机
    多进程"的边缘情况，且无额外依赖。

    Args:
        path: 被保护的资源路径（锁文件取其同级 ``.lock``）。
        timeout_s: 获取锁的等待上限；超时后**放弃加锁继续执行**（不阻塞用户
            操作——数据一致性由"重读合并"兜底）。

    Yields:
        None。
    """
    import time

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


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """读取 JSONL 文件，返回 (有效的 dict 行, 损坏行数)。

    **损坏行跳过而非中断**：单行 JSON 损坏不应导致整个标注文件不可读；跳过
    的行数如实返回，由调用方决定是否提示用户（绝不静默假装没有损坏）。
    """
    rows: list[dict[str, Any]] = []
    bad = 0
    if not path.exists():
        return rows, 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    bad += 1
    except OSError:
        return rows, bad
    return rows, bad


# ---------------------------------------------------------------------------
# 锚点解析（格式 → 锚点的唯一适配点）
# ---------------------------------------------------------------------------


def _sanitize_key(raw: str) -> str:
    """把任意标识净化为安全的 ``episode_key``（保留中文、字母数字、``-_.``）。

    为什么需要：h5 节点路径含 ``/``、多流目录的列值可能含空格或冒号——这些
    会污染落盘索引键与未来可能的路由（如按 key 切分文件）。
    """
    import re

    cleaned = re.sub(r"[^\w.\-]+", "_", str(raw), flags=re.UNICODE).strip("._")
    return cleaned or "unknown"


def _find_col(columns: Iterable[Any], candidates: tuple[str, ...]) -> str | None:
    """在列名中查找候选列（大小写不敏感，去空白）。"""
    lowered = {str(c).lower().strip(): str(c) for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _is_lerobot_dataset(context: RunContext) -> bool:
    """判断是否 LeRobot 布局（meta/info.json + data/chunk-* 的语义特征）。

    判据取**结构特征**而非目录名字面：源路径下存在 ``meta/info.json``
    且其内容含 ``features`` 或 ``fps`` 键，或流登记表出现 ``meta/info.json``。
    """
    src = str(context.meta.get("source", "") or "")
    if not src:
        return False
    p = Path(src)
    if not p.is_dir():
        return False
    info = p / "meta" / "info.json"
    if not info.exists():
        return False
    try:
        obj = json.loads(info.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        return False
    return isinstance(obj, dict) and (
        "features" in obj or "fps" in obj or "codebase_version" in obj
    )


def _lerobot_fps(context: RunContext) -> float | None:
    """读取 LeRobot ``meta/info.json`` 的 fps（用于帧↔秒换算）；失败返回 None。"""
    src = str(context.meta.get("source", "") or "")
    if not src:
        return None
    try:
        obj = json.loads(
            (Path(src) / "meta" / "info.json").read_text(encoding="utf-8")
        )
        fps = obj.get("fps")
        if isinstance(fps, (int, float)) and fps > 0:
            return float(fps)
    except (ValueError, OSError, UnicodeDecodeError, TypeError):
        pass
    return None


def _anchors_from_episode_column(
    df: Any, *, time_column: str | None, fps: float | None,
    anchor_source: str, evidence_prefix: str,
) -> list[EpisodeAnchor]:
    """按 episode 列的唯一值划分锚点（LeRobot 与多流目录共用）。

    Args:
        df: 数据表（含 episode 列）。
        time_column: 每 episode 的时间列名（可为 None）。
        fps: 每帧秒数换算用的帧率（可为 None）。
        anchor_source: 锚点来源标记。
        evidence_prefix: 证据前缀（区分格式来源）。

    Returns:
        锚点列表（按 episode 键排序）。
    """
    ep_col = _find_col(df.columns, _EPISODE_COL_CANDIDATES)
    if ep_col is None:
        return []

    anchors: list[EpisodeAnchor] = []
    # 逐 episode 取该段的行范围（保留原始行序，不排序——时间顺序由索引顺序表达）。
    groups = df.groupby(ep_col, sort=True, dropna=False)
    for ep_value, sub in groups:
        n = int(len(sub))
        start_frame: int | None = None
        end_frame: int | None = None
        start_s: float | None = None
        end_s: float | None = None

        # 优先从时间列取真实起止秒。
        if time_column is not None and time_column in sub.columns:
            try:
                ts = sub[time_column].dropna()
                if len(ts) >= 1:
                    t0 = float(ts.iloc[0])
                    t1 = float(ts.iloc[-1])
                    # 时间单位由 timestamp_units 模块判定更可靠；此处只做
                    # "已是秒"的保守判断——量级过大（>=1e6）时不猜单位。
                    if abs(t0) < 1e6 and abs(t1) < 1e6:
                        start_s, end_s = t0, t1
            except (ValueError, TypeError):
                pass

        # 帧号：有 frame_index / frame 列则用，否则按 fps 推算。
        fr_col = _find_col(
            sub.columns, ("frame_index", "frame", "frame_id", "index")
        )
        if fr_col is not None:
            try:
                fr = sub[fr_col].dropna()
                if len(fr) >= 1:
                    start_frame = int(fr.iloc[0])
                    end_frame = int(fr.iloc[-1])
            except (ValueError, TypeError):
                pass
        if start_frame is None and fps:
            start_frame, end_frame = 0, max(0, n - 1)
        if start_s is None and fps and start_frame is not None:
            start_s = start_frame / fps
            end_s = (end_frame + 1) / fps if end_frame is not None else None

        key = _sanitize_key(str(ep_value))
        anchors.append(EpisodeAnchor(
            episode_key=key,
            label=f"{evidence_prefix} {ep_value}",
            start_s=start_s,
            end_s=end_s,
            start_frame=start_frame,
            end_frame=end_frame,
            n_frames=n,
            time_column=time_column,
            anchor_source=anchor_source,
            evidence=(
                f"按列 {ep_col} 的唯一值划分 episode（该 episode {n} 帧）"
                + (f"；时间列 {time_column}" if time_column else "；无时间列")
            ),
        ))
    return anchors


def _anchors_from_h5_nodes(context: RunContext) -> list[EpisodeAnchor]:
    """从流登记表的 h5 节点流构造锚点（每个节点流视为一个可标注单元）。

    设计取舍：HDF5 的"episode"概念因布局而异——帧布局（每帧一组）合并后的
    流是**按字段**组织的，没有天然的 episode 边界。因此本模块把**每个 h5
    节点流**作为一个锚点单元（而非强行猜 episode），证据里如实说明这一点。
    """
    from app.tools._readers import split_path_spec

    anchors: list[EpisodeAnchor] = []
    for s in context.meta.get("streams", []):
        if s.get("format") != "h5":
            continue
        file_part, node = split_path_spec(s.get("path", ""))
        if not node:
            continue
        n_frames = s.get("n_rows") if isinstance(s.get("n_rows"), int) else -1
        # 帧布局流已知帧数；逐帧纵向拼接后有 frame_index 列。
        if s.get("frame_layout") and isinstance(s.get("n_frames"), int):
            n_frames = int(s["n_frames"])
        anchors.append(EpisodeAnchor(
            episode_key=_sanitize_key(node),
            label=f"容器节点 {Path(file_part).stem}::{node}",
            start_s=None,
            end_s=None,
            start_frame=0 if n_frames > 0 else None,
            end_frame=(n_frames - 1) if n_frames > 0 else None,
            n_frames=n_frames,
            time_column=None,
            anchor_source=ANCHOR_H5_NODE,
            evidence=(
                f"h5 节点流（{Path(file_part).name} 的节点 {node}）"
                + ("；该节点为帧布局，已按字段合并" if s.get("frame_layout") else "")
            ),
        ))
    return anchors


def resolve_anchors(context: RunContext) -> dict[str, Any]:
    """解析当前数据集的 episode 锚点清单（**格式 → 锚点的唯一适配点**）。

    按以下优先级分派（顺序即证据强度）：

    1. **LeRobot**：源目录含 ``meta/info.json``（含 features/fps/codebase_version）
       → 用 ``episode_index``/episode 列划分，fps 来自 info.json；
    2. **多流目录**：数据表含 episode 列候选 → 按其唯一值划分；
    3. **HDF5**：流登记表含 h5 节点流 → **每个节点流一个锚点**（h5 无天然
       episode 边界，不强行猜）；
    4. **兜底**：以上都不成 → 整段视为一个 episode（``__all__``），并如实
       在 warnings 中说明"未找到 episode 划分线索"。

    Args:
        context: 运行时上下文（读 meta 与 df）。

    Returns:
        dict，含 ``success``、``anchors``（list[dict]，经 :meth:`EpisodeAnchor.to_dict`）、
        ``anchor_source``、``n_anchors``、``dataset``、``warnings``。
        未加载数据集时 ``success=False`` 且 ``error="no_data_loaded"``。
    """
    if context.dataset_id is None and context.df is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载任何数据集",
            "anchors": [],
            "n_anchors": 0,
            "user_message": "尚未加载任何数据集。请先调用 load_dataset 加载数据。",
        }

    warnings: list[str] = []
    df = context.df
    time_col = None
    if df is not None:
        time_col = _find_col(df.columns, _TIME_COL_CANDIDATES)

    anchors: list[EpisodeAnchor] = []
    anchor_source = ANCHOR_WHOLE

    # 1) LeRobot：fps 来自 meta/info.json。
    if _is_lerobot_dataset(context) and df is not None:
        fps = _lerobot_fps(context)
        anchors = _anchors_from_episode_column(
            df, time_column=time_col, fps=fps,
            anchor_source=ANCHOR_LEROBOT,
            evidence_prefix="episode",
        )
        if anchors:
            anchor_source = ANCHOR_LEROBOT
            if fps is None:
                warnings.append(
                    "LeRobot 布局已识别，但 meta/info.json 未给出可用 fps——"
                    "帧↔秒换算不可用（start_s/end_s 为 null）。"
                )
        else:
            warnings.append(
                "识别为 LeRobot 布局，但主表未找到 episode 列——"
                "已回退到其它锚点来源。"
            )

    # 2) 多流目录 / 通用：按 episode 列划分。
    if not anchors and df is not None:
        # fps 尝试从列或流登记表取得（缺省 None → 不做帧↔秒换算，不猜）。
        fps_col = _find_col(df.columns, _FPS_COL_CANDIDATES)
        fps: float | None = None
        if fps_col is not None:
            try:
                v = df[fps_col].dropna()
                if len(v) >= 1:
                    cand = float(v.iloc[0])
                    if cand > 0:
                        fps = cand
            except (ValueError, TypeError):
                fps = None
        if fps is None:
            for v in context.meta.get("video_meta", []) or []:
                cand = v.get("fps")
                if isinstance(cand, (int, float)) and cand > 0:
                    fps = float(cand)
                    break
        anchors = _anchors_from_episode_column(
            df, time_column=time_col, fps=fps,
            anchor_source=ANCHOR_EPISODE_COLUMN,
            evidence_prefix="episode",
        )
        if anchors:
            anchor_source = ANCHOR_EPISODE_COLUMN

    # 3) HDF5 节点流。
    if not anchors:
        anchors = _anchors_from_h5_nodes(context)
        if anchors:
            anchor_source = ANCHOR_H5_NODE
            warnings.append(
                "HDF5 容器：以**每个节点流**为标注单元（h5 无天然的 episode "
                "边界，未强行推断）。若你的语义单元是容器内子流，此锚点即正确单元。"
            )

    # 4) 兜底：整段一个 episode。
    if not anchors:
        n = int(df.shape[0]) if df is not None else -1
        anchors = [EpisodeAnchor(
            episode_key=WHOLE_DATASET_KEY,
            label="整段数据（单一单元）",
            start_s=None,
            end_s=None,
            start_frame=0 if n > 0 else None,
            end_frame=(n - 1) if n > 0 else None,
            n_frames=n,
            time_column=time_col,
            anchor_source=ANCHOR_WHOLE,
            evidence="未找到 episode 划分线索（无 episode 列、非容器布局）——整段视为一个单元",
        )]
        anchor_source = ANCHOR_WHOLE
        warnings.append(
            "未找到 episode 划分线索（无 episode 列，也非常见容器布局）——"
            "已把整段数据视为一个标注单元。若你的数据本应分 episode，"
            "请确认 episode 列名并写入数据集画像后重试。"
        )

    # 无时间信息时如实警告（不猜秒值）。
    if anchors and all(a.start_s is None for a in anchors):
        warnings.append(
            "锚点缺少可用时间列或 fps——start_s/end_s 为 null。"
            "切片标注的时间边界将无法与数据对齐，建议先经 "
            "find_timestamp_columns / inspect_streams 确认时间口径。"
        )

    return {
        "success": True,
        "dataset": context.dataset_id,
        "anchors": [a.to_dict() for a in anchors],
        "n_anchors": len(anchors),
        "anchor_source": anchor_source,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 写入前的规范化与校验
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float | None) -> str:
    """秒 → ``HH:MM:SS``（显示用）；None → 空串（诚实降级，不填 ``00:00:00``）。

    Args:
        seconds: 秒数（可为负/超 24 小时，按算术进位）。

    Returns:
        ``HH:MM:SS`` 字符串；None 时返回空串。
    """
    if seconds is None:
        return ""
    try:
        total = int(round(float(seconds)))
    except (ValueError, TypeError):
        return ""
    sign = "-" if total < 0 else ""
    total = abs(total)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}"


def parse_timestamp(text: str) -> float | None:
    """``HH:MM:SS``（或 ``MM:SS`` / 纯秒数）→ 秒；无法解析返回 None。

    与 :func:`format_timestamp` 互逆，供用户以人类可读格式输入边界。

    Args:
        text: 时间字符串。

    Returns:
        秒数；解析失败返回 None（不猜）。
    """
    s = str(text or "").strip()
    if not s:
        return None
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    parts = s.split(":")
    if len(parts) > 3:
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return -total if neg else total


def normalize_record(
    record: dict[str, Any], *, scope: str, episode_key: str | None = None,
    anchor: EpisodeAnchor | None = None,
) -> dict[str, Any]:
    """把一条标注规范化为落盘格式（补默认值、归一字段、给出时间戳双表示）。

    规范化规则（详见设计文档 §4.3）：

    - ``start_timestamp`` / ``end_timestamp`` 由 ``start_s`` / ``end_s``
      **派生**（显示用），数值字段才是权威；
    - 无 ``start_s`` 但给了 ``start_timestamp`` 时**解析回数值**（互为逆）；
    - 秒与帧同时保留（训练真值需精确帧号，元数据发布需人类可读）；
    - ``source`` / ``confidence`` 落在受限词表内，非法值**拒绝**（不静默改成默认值）。

    Args:
        record: 原始记录（可来自 LLM 工具参数或用户输入）。
        scope: ``task`` 或 ``segment``。
        episode_key: 覆盖记录的 episode_key（缺省取记录内字段）。
        anchor: 该 episode 的锚点（用于补齐/校验帧号）。

    Returns:
        dict，含 ``ok`` 与 ``record``（规范化结果）或 ``error`` / ``reason``。
    """
    if scope not in _VALID_SCOPES:
        return {"ok": False, "error": "invalid_scope",
                "reason": f"scope 必须是 {_VALID_SCOPES} 之一，收到 {scope!r}"}

    rec = dict(record or {})
    key = str(episode_key or rec.get("episode_key") or "").strip()
    if not key:
        # 锚点存在但记录未写 key 时，用锚点的 key 补齐（调用方已给出定位依据，
        # 属可确定信息，不是猜测）；完全没有锚点依据时才拒绝。
        if anchor is not None and anchor.episode_key:
            key = anchor.episode_key
        else:
            return {
                "ok": False, "error": "missing_episode_key",
                "reason": "记录缺少 episode_key（应取自 resolve_anchors 的返回）",
            }
    rec["episode_key"] = key
    rec["scope"] = scope

    # 帧号外推需要锚点：调用方未显式传入时，尝试从记录自带锚点信息补齐。
    # 设计取舍：**不**在这里静默改用 WHOLE_DATASET_KEY 兜底——"缺锚点"是
    # 真实缺口（说明调用方没先取锚点），必须让上层看到并修正。
    if anchor is None and rec.get("anchor") is not None:
        try:
            a = rec["anchor"]
            if isinstance(a, EpisodeAnchor):
                anchor = a
            elif isinstance(a, dict):
                anchor = EpisodeAnchor(**a)
        except TypeError:
            anchor = None

    # 来源与可信度：受限词表，非法值拒绝而非静默兜底。
    source = str(rec.get("source") or "").strip()
    if not source:
        rec["source"] = SOURCE_LLM  # 缺省按最保守处理（未确认的模型产出）
    elif source not in _VALID_SOURCES:
        return {
            "ok": False, "error": "invalid_source",
            "reason": f"source 必须是 {_VALID_SOURCES} 之一，收到 {source!r}",
        }
    confidence = str(rec.get("confidence") or "").strip().lower()
    if not confidence:
        rec["confidence"] = (
            "high" if rec["source"] == SOURCE_USER
            else "low" if rec["source"] == SOURCE_LLM else "medium"
        )
    elif confidence not in _VALID_CONFIDENCES:
        return {
            "ok": False, "error": "invalid_confidence",
            "reason": f"confidence 必须是 {_VALID_CONFIDENCES} 之一，收到 {confidence!r}",
        }
    else:
        rec["confidence"] = confidence

    if scope == SCOPE_SEGMENT:
        # 秒与 HH:MM:SS 互为逆：任一可得即补全另一个。
        start_s = rec.get("start_s")
        end_s = rec.get("end_s")
        if start_s is None:
            start_s = parse_timestamp(str(rec.get("start_timestamp") or ""))
        if end_s is None:
            end_s = parse_timestamp(str(rec.get("end_timestamp") or ""))
        if start_s is None or end_s is None:
            return {
                "ok": False, "error": "missing_time_bounds",
                "reason": (
                    "切片标注需要可解析的 start_s/end_s（或等价的 "
                    "start_timestamp/end_timestamp）——不接受缺时间的片段"
                ),
            }
        try:
            start_s = float(start_s)
            end_s = float(end_s)
        except (ValueError, TypeError):
            return {"ok": False, "error": "invalid_time_bounds",
                    "reason": "start/end 无法转为数值"}
        if end_s < start_s:
            return {
                "ok": False, "error": "reversed_time_bounds",
                "reason": f"end_s({end_s}) 小于 start_s({start_s})",
            }
        rec["start_s"] = round(start_s, 6)
        rec["end_s"] = round(end_s, 6)
        rec["start_timestamp"] = format_timestamp(start_s)
        rec["end_timestamp"] = format_timestamp(end_s)

        # 帧号：优先用给定值；缺省且锚点带 fps 信息时由秒推算。
        if rec.get("start_frame") is None and anchor is not None:
            if anchor.start_frame is not None and anchor.start_s is not None:
                # 用锚点自身的"秒→帧"比例外推，避免依赖未知单位的时间列。
                span_s = (anchor.end_s - anchor.start_s) if (
                    anchor.end_s is not None and anchor.start_s is not None
                ) else None
                span_f = (anchor.end_frame - anchor.start_frame) if (
                    anchor.end_frame is not None and anchor.start_frame is not None
                ) else None
                if span_s and span_f and span_s > 0:
                    ratio = span_f / span_s
                    rec["start_frame"] = anchor.start_frame + int(round(
                        (start_s - anchor.start_s) * ratio))
        if rec.get("id") is None:
            rec["id"] = 1
        # 噪声标记一致性：is_noise=True 必须有 cleaning_reason（硬约束）。
        if rec.get("is_noise") and not str(rec.get("cleaning_reason") or "").strip():
            return {
                "ok": False, "error": "noise_reason_required",
                "reason": "is_noise=true 时必须给出 cleaning_reason（标注规则一致性）",
            }
    else:
        # 任务级：必须有任务描述。
        if not str(rec.get("task") or "").strip():
            return {
                "ok": False, "error": "missing_task",
                "reason": "任务级标注需要 task（任务描述）字段",
            }

    if not rec.get("created_at"):
        rec["created_at"] = _now_iso()
    rec.setdefault("actor", rec.get("source"))
    return {"ok": True, "record": rec}


def check_source_for_use(
    records: list[dict[str, Any]], use: str
) -> dict[str, Any]:
    """校验标注来源是否允许用于指定用途（用户决策：三种用途都要）。

    分级规则（设计文档 §4.4）：

    - ``training``（下游训练真值）：**只允许** ``user_confirmed`` /
      ``imported``；``signal_derived`` 需先人工复核、``llm_proposed`` **禁止**；
    - ``publication``（数据集发布元数据）：允许前三者，但 ``llm_proposed``
      与 ``signal_derived`` 须声明为自动生成（返回 ``must_declare_generated``）；
    - ``audit``（内部清查）：全部允许。

    Args:
        records: 待校验的标注记录列表。
        use: 用途，取值见 ``USE_*`` 常量。

    Returns:
        dict，含 ``ok``、``use``、``violations``（违规清单，含 episode_key/id/
        source/reason）、``must_declare_generated``（需声明自动生成的记录数）。
    """
    if use not in _VALID_USES:
        return {
            "ok": False, "error": "invalid_use", "use": use,
            "reason": f"use 必须是 {_VALID_USES} 之一",
            "violations": [],
            "must_declare_generated": 0,
        }

    violations: list[dict[str, Any]] = []
    declare = 0
    for r in records:
        src = str(r.get("source") or "")
        ident = {"episode_key": r.get("episode_key"), "id": r.get("id")}
        if use == USE_TRAINING:
            if src == SOURCE_LLM:
                violations.append({
                    **ident, "source": src,
                    "reason": "llm_proposed 来源的标注禁止用于下游训练真值，"
                              "须先经人工复核并转为 user_confirmed",
                })
            elif src == SOURCE_SIGNAL:
                violations.append({
                    **ident, "source": src,
                    "reason": "signal_derived 来源的标注用于训练真值前须人工复核"
                              "（边界由信号算出，但语义未经确认）",
                })
        elif use == USE_PUBLICATION:
            if src in (SOURCE_LLM, SOURCE_SIGNAL):
                declare += 1

    return {
        "ok": not violations,
        "use": use,
        "violations": violations,
        "n_violations": len(violations),
        "must_declare_generated": declare,
    }


# ---------------------------------------------------------------------------
# 读写接口
# ---------------------------------------------------------------------------


def load_annotations(
    output_dir: str,
    dataset_id: str | None,
    *,
    scope: str | None = None,
    episode_key: str | None = None,
) -> dict[str, Any]:
    """读取标注（当前态）。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。
        scope: ``task`` / ``segment`` / None（None = 两者都读）。
        episode_key: 可选，只返回该 episode 的记录。

    Returns:
        dict，含 ``records``（list[dict]，含 scope 字段以便区分）、
        ``by_scope``（{task: n, segment: n}）、``bad_lines``（损坏行数，
        **如实报告，绝不静默**）、``paths``。
    """
    scopes = [scope] if scope in _VALID_SCOPES else list(_VALID_SCOPES)
    records: list[dict[str, Any]] = []
    bad_total = 0
    paths: dict[str, str] = {}
    by_scope: dict[str, int] = {}

    for sc in scopes:
        p = _scope_path(output_dir, dataset_id, sc)
        paths[sc] = str(p)
        rows, bad = _read_jsonl(p)
        bad_total += bad
        n = 0
        for r in rows:
            r = dict(r)
            r.setdefault("scope", sc)
            if episode_key is not None and r.get("episode_key") != episode_key:
                continue
            records.append(r)
            n += 1
        by_scope[sc] = n

    return {
        "records": records,
        "by_scope": by_scope,
        "n_records": len(records),
        "bad_lines": bad_total,
        "paths": paths,
    }


def append_annotations(
    output_dir: str,
    dataset_id: str | None,
    records: list[dict[str, Any]],
    *,
    actor: str = "",
    reason: str = "",
    session_tag: str = "",
) -> dict[str, Any]:
    """追加/更新标注（加锁 + 重读 + 去重合并 + 原子写 + 记修订日志）。

    去重键：``(scope, episode_key, id)``——同一片段重复提交视为**更新**
    （记录 ``before``/``after`` 进修订日志），而非追加重复行。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。
        records: 已规范化的记录（应经 :func:`normalize_record`）。
        actor: 本次变更发起者（缺省取记录的 source）。
        reason: 变更原因（进修订日志，供回溯）。
        session_tag: 会话标识（多会话并发时定位变更来源）。

    Returns:
        dict，含 ``saved`` / ``updated`` / ``skipped`` / ``skipped_detail`` /
        ``revision_logged`` / ``revision_total_lines``（超软上限时在
        ``warnings`` 中提示归档）／``paths``。
    """
    if not records:
        return {
            "saved": 0, "updated": 0, "skipped": 0, "skipped_detail": [],
            "revision_logged": 0, "revision_total_lines": 0, "warnings": [],
            "paths": {},
        }

    by_scope: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        sc = str(r.get("scope") or SCOPE_SEGMENT)
        if sc not in _VALID_SCOPES:
            by_scope.setdefault("__invalid__", []).append(r)
            continue
        by_scope.setdefault(sc, []).append(r)

    saved = updated = 0
    skipped_detail: list[dict[str, Any]] = []
    for r in by_scope.pop("__invalid__", []):
        skipped_detail.append({
            "record": r, "reason": "scope 非 task/segment，未落盘",
        })

    paths: dict[str, str] = {}
    revision_lines: list[str] = []

    for sc, recs in by_scope.items():
        path = _scope_path(output_dir, dataset_id, sc)
        paths[sc] = str(path)
        # 加锁 → 重读盘上最新 → 合并 → 原子写（防读-改-写竞态：两个会话
        # 同时对同一数据集写入时，后写的不得丢掉先写的）。
        with _file_lock(path):
            existing, bad = _read_jsonl(path)
            if bad:
                skipped_detail.append({
                    "reason": f"{path.name} 有 {bad} 行损坏，已跳过（原样保留）",
                })
            index: dict[tuple[str, str], dict[str, Any]] = {}
            for row in existing:
                k = (str(row.get("episode_key") or ""), str(row.get("id") or ""))
                index[k] = row

            for rec in recs:
                key = (str(rec.get("episode_key") or ""), str(rec.get("id") or ""))
                before = index.get(key)
                if before is not None:
                    updated += 1
                    # 只记录"实质变化"的字段，避免噪声（时间戳字段每次都会变）。
                    changed = {
                        f: [before.get(f), rec.get(f)]
                        for f in set(before) | set(rec)
                        if f not in ("created_at",) and before.get(f) != rec.get(f)
                    }
                    if changed:
                        revision_lines.append(json.dumps({
                            "ts": _now_iso(),
                            "op": "update",
                            "scope": sc,
                            "episode_key": rec.get("episode_key"),
                            "id": rec.get("id"),
                            "actor": actor or str(rec.get("source") or ""),
                            "session_tag": session_tag,
                            "before": {k: v[0] for k, v in changed.items()},
                            "after": {k: v[1] for k, v in changed.items()},
                            "reason": reason or "更新既有标注",
                        }, ensure_ascii=False))
                    else:
                        skipped_detail.append({
                            "episode_key": rec.get("episode_key"),
                            "id": rec.get("id"),
                            "reason": "内容与既有记录完全一致，未产生变更",
                        })
                else:
                    saved += 1
                    revision_lines.append(json.dumps({
                        "ts": _now_iso(),
                        "op": "create",
                        "scope": sc,
                        "episode_key": rec.get("episode_key"),
                        "id": rec.get("id"),
                        "actor": actor or str(rec.get("source") or ""),
                        "session_tag": session_tag,
                        "after": rec,
                        "reason": reason or "新增标注",
                    }, ensure_ascii=False))
                index[key] = rec

            # 写回当前态：按 (episode_key, id) 排序，保证文件可 diff、可人工阅读。
            ordered = sorted(
                index.values(),
                key=lambda r: (str(r.get("episode_key") or ""), str(r.get("id") or "")),
            )
            _atomic_write_text(
                path,
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered),
            )

    # 修订日志：原子追加。
    rev_logged = 0
    rev_total = 0
    warnings: list[str] = []
    if revision_lines:
        rev_path = _revisions_dir(output_dir, dataset_id) / _REVISIONS_FILENAME
        _atomic_append_lines(rev_path, revision_lines)
        rev_logged = len(revision_lines)
        existing_rev, _ = _read_jsonl(rev_path)
        rev_total = len(existing_rev)
        if rev_total > _REVISION_SOFT_LIMIT:
            warnings.append(
                f"修订日志已 {rev_total:,} 行，超过软上限 "
                f"{_REVISION_SOFT_LIMIT:,}——建议归档到 "
                f"revisions/archive/（**未自动处理**，避免静默丢失历史）。"
            )

    result: dict[str, Any] = {
        "saved": saved,
        "updated": updated,
        "skipped": len(skipped_detail),
        "skipped_detail": skipped_detail,
        "revision_logged": rev_logged,
        "revision_total_lines": rev_total,
        "warnings": warnings,
        "paths": paths,
    }
    return result


def delete_annotation(
    output_dir: str,
    dataset_id: str | None,
    *,
    scope: str,
    episode_key: str,
    id: Any,
    actor: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """删除单条标注（复核时剔除错误标注用），并把删除记入修订日志。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。
        scope: ``task`` / ``segment``。
        episode_key: 目标 episode。
        id: 目标片段 id（任务级通常为 1）。
        actor: 变更发起者。
        reason: 删除原因（进修订日志）。

    Returns:
        dict，含 ``deleted``（是否删除成功）、``before``（被删记录）、``paths``。
    """
    if scope not in _VALID_SCOPES:
        return {"deleted": False, "error": "invalid_scope",
                "reason": f"scope 必须是 {_VALID_SCOPES} 之一"}

    path = _scope_path(output_dir, dataset_id, scope)
    removed: dict[str, Any] | None = None
    with _file_lock(path):
        rows, _bad = _read_jsonl(path)
        kept: list[dict[str, Any]] = []
        for r in rows:
            if (str(r.get("episode_key") or "") == str(episode_key)
                    and str(r.get("id") or "") == str(id)):
                removed = r
                continue
            kept.append(r)
        if removed is not None:
            ordered = sorted(
                kept,
                key=lambda r: (str(r.get("episode_key") or ""), str(r.get("id") or "")),
            )
            _atomic_write_text(
                path,
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered),
            )

    if removed is not None:
        rev_path = _revisions_dir(output_dir, dataset_id) / _REVISIONS_FILENAME
        _atomic_append_lines(rev_path, [json.dumps({
            "ts": _now_iso(),
            "op": "delete",
            "scope": scope,
            "episode_key": episode_key,
            "id": id,
            "actor": actor,
            "session_tag": "",
            "before": removed,
            "after": None,
            "reason": reason or "删除标注",
        }, ensure_ascii=False)])

    return {
        "deleted": removed is not None,
        "before": removed,
        "paths": {scope: str(path)},
    }


# ---------------------------------------------------------------------------
# 版本管理（快照 / 历史 / diff）
# ---------------------------------------------------------------------------


def snapshot_annotations(
    output_dir: str,
    dataset_id: str | None,
    *,
    label: str = "",
    scopes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """把当前态冻结为快照（**交付/发布前手动触发**，不自动）。

    为什么需要：训练真值与数据集发布是**冻结语义**——"发给下游的那一版"
    必须可回溯。快照不做自动触发，避免交付语义被自动行为污染。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。
        label: 快照标签（进文件名，便于辨认，如 ``v1_for_release``）。
        scopes: 要快照的作用域；缺省两者都做。

    Returns:
        dict，含 ``paths``（各作用域快照文件路径）、``counts``、``ts``、
        ``label``；无任何标注时 ``success=False``（不产出空快照）。
    """
    scope_list = [s for s in (scopes or _VALID_SCOPES) if s in _VALID_SCOPES]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_label = _sanitize_key(label) if label else ""
    suffix = f"_{safe_label}" if safe_label else ""

    out_paths: dict[str, str] = {}
    counts: dict[str, int] = {}
    snap_dir = _snapshots_dir(output_dir, dataset_id)
    total = 0

    for sc in scope_list:
        src = _scope_path(output_dir, dataset_id, sc)
        rows, _bad = _read_jsonl(src)
        if not rows:
            counts[sc] = 0
            continue
        dst = snap_dir / f"{stamp}{suffix}_{sc}.jsonl"
        _atomic_write_text(
            dst,
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        )
        out_paths[sc] = str(dst)
        counts[sc] = len(rows)
        total += len(rows)

    if not out_paths:
        return {
            "success": False,
            "error": "no_annotations",
            "user_message": "当前数据集没有任何标注可快照（未产出空快照）。",
            "paths": {}, "counts": counts, "ts": stamp, "label": label,
        }

    return {
        "success": True,
        "paths": out_paths,
        "counts": counts,
        "n_records": total,
        "ts": stamp,
        "label": label,
        "user_message": (
            f"已冻结快照（{total} 条标注，作用域 {list(out_paths)}）。"
            "快照不会被后续编辑影响，可用于交付/发布溯源。"
        ),
    }


def list_snapshots(output_dir: str, dataset_id: str | None) -> dict[str, Any]:
    """列出已有快照（按时间倒序）。

    Returns:
        dict，含 ``snapshots``（[{name, path, ts, scope, n_records}]）。
    """
    snap_dir = _snapshots_dir(output_dir, dataset_id)
    items: list[dict[str, Any]] = []
    for p in sorted(snap_dir.glob("*.jsonl"), reverse=True):
        rows, _bad = _read_jsonl(p)
        stem = p.stem
        # 文件名形如 <ts>[_<label>]_<scope>。
        parts = stem.split("_")
        ts = parts[0] if parts else ""
        scope = parts[-1] if len(parts) > 1 else ""
        items.append({
            "name": p.name,
            "path": str(p),
            "ts": ts,
            "scope": scope,
            "n_records": len(rows),
        })
    return {"snapshots": items, "n_snapshots": len(items)}


def load_annotation_history(
    output_dir: str,
    dataset_id: str | None,
    *,
    scope: str | None = None,
    episode_key: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """读取修订历史（可按 scope / episode_key 过滤），供展示"这条改过几次"。

    Args:
        output_dir: 项目输出目录。
        dataset_id: 数据集标识名。
        scope: 可选，过滤作用域。
        episode_key: 可选，过滤 episode。
        limit: 最多返回的**最近**条目数（从尾部取；总数如实报告）。

    Returns:
        dict，含 ``entries``、``n_entries``（过滤后总数）、``total_lines``
        （日志总行数，用于判断是否被 limit 截断）、``truncated``。
    """
    rev_path = _revisions_dir(output_dir, dataset_id) / _REVISIONS_FILENAME
    rows, bad = _read_jsonl(rev_path)
    filtered: list[dict[str, Any]] = []
    for r in rows:
        if scope is not None and r.get("scope") != scope:
            continue
        if episode_key is not None and r.get("episode_key") != episode_key:
            continue
        filtered.append(r)

    n = len(filtered)
    truncated = n > limit
    entries = filtered[-limit:] if truncated else filtered
    return {
        "entries": entries,
        "n_entries": n,
        "total_lines": len(rows),
        "bad_lines": bad,
        "truncated": truncated,
        "path": str(rev_path),
    }


def diff_snapshots(path_a: str, path_b: str) -> dict[str, Any]:
    """对比两个快照，返回新增/修改/删除三类差异（"这版和上版差在哪"）。

    Args:
        path_a: 基线快照路径（较旧）。
        path_b: 对比快照路径（较新）。

    Returns:
        dict，含 ``added`` / ``modified`` / ``removed`` / ``unchanged`` 计数与
        明细（``items``）。两个路径都不可读时 ``success=False``。
    """
    a_rows, a_bad = _read_jsonl(Path(path_a))
    b_rows, b_bad = _read_jsonl(Path(path_b))
    if not Path(path_a).exists() and not Path(path_b).exists():
        return {
            "success": False,
            "error": "snapshot_not_found",
            "reason": f"两个快照都不存在：{path_a} / {path_b}",
        }

    def _key(r: dict[str, Any]) -> tuple[str, str]:
        return (str(r.get("episode_key") or ""), str(r.get("id") or ""))

    a_map = {_key(r): r for r in a_rows}
    b_map = {_key(r): r for r in b_rows}

    added = [
        {"episode_key": k[0], "id": k[1], "record": b_map[k]}
        for k in b_map if k not in a_map
    ]
    removed = [
        {"episode_key": k[0], "id": k[1], "record": a_map[k]}
        for k in a_map if k not in b_map
    ]
    modified: list[dict[str, Any]] = []
    unchanged = 0
    for k in a_map:
        if k not in b_map:
            continue
        before, after = a_map[k], b_map[k]
        changed = {
            f: [before.get(f), after.get(f)]
            for f in set(before) | set(after)
            if before.get(f) != after.get(f)
        }
        if changed:
            modified.append({**{"episode_key": k[0], "id": k[1]}, "changes": changed})
        else:
            unchanged += 1

    return {
        "success": True,
        "added": len(added),
        "modified": len(modified),
        "removed": len(removed),
        "unchanged": unchanged,
        "items": {"added": added, "modified": modified, "removed": removed},
        "bad_lines": {"a": a_bad, "b": b_bad},
    }
