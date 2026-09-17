"""标注写入工具：任务语义标注、切片标注落盘、标注自身质检。

设计依据：``docs/标注与质检能力设计.md`` §5.1 / §5.4。

本模块把前三个阶段的成果**串成闭环**：

    质检数据集（阶段二）→ 切分边界（阶段三）→ 人工命名 → 落盘（本模块）
                                                        ↓
                                              标注质检（本模块）→ 修正 → 快照

**三个工具**：

1. :func:`annotate_task` —— 任务级语义标注（整段任务描述），与
   ``propose_stream_semantics`` 同构的**三态闸门**（登记 → 用户确认 → 落盘）。
   候选来源全部确定性（LeRobot ``meta/tasks.jsonl``、语言列唯一值），
   取不到就请用户提供，**不由模型编造**。
2. :func:`save_annotations` —— 切片/任务标注落盘。含**来源标记**、
   **复核闸门**、**用途合规校验**（``llm_proposed`` 禁止进训练真值路径）。
3. :func:`check_annotation_qc` —— 标注自身质检（重叠/空隙/超界/规则不一致等），
   确定性最高、几乎零误报，是"标注可用性"的最后一道闸门。

**核心纪律**（贯穿三个工具）：

- **命名不得编造**：``atomic_action`` / ``action_description`` 若由模型提出，
  一律标 ``llm_proposed`` + ``confidence=low``，并**禁止**在用户确认前用于
  训练真值（用途合规校验强制拦截）。
- **闸门不可绕过**：``confirm=False`` 只落盘草稿状态？——不，本模块的设计是
  ``save_annotations`` 必须显式给 ``confirmed_by_user`` 才能把来源标记为
  ``user_confirmed``；否则来源保持调用方给定值。
- **只写 outputs/**：绝不改动源数据集。

本模块不 import streamlit。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agents import RunContextWrapper
from agents.decorators import tool
from pydantic import BaseModel, Field

from app.agent.context import RunContext
from app.config import get_settings
from app.tools import annotation_store as store
from app.tools.annotation_store import (
    SCOPE_SEGMENT,
    SCOPE_TASK,
    SOURCE_IMPORTED,
    SOURCE_LLM,
    SOURCE_SIGNAL,
    SOURCE_USER,
    USE_AUDIT,
    USE_PUBLICATION,
    USE_TRAINING,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 任务描述列候选（用于确定性提取任务语义）。
_TASK_COLS = (
    "task", "task_name", "instruction", "language", "language_instruction",
    "task_description", "prompt", "subtask", "skill",
)
# 交互手取值（用户裁决：统一枚举）。
_HAND_VALUES = ("left_hand", "right_hand", "both_hands", "none")

# 标注质检：可判 fail 的确定性规则（与 warn 型规则分离）。
_QC_FAIL_RULES = (
    "time_overlap",
    "bounds_exceeded",
    "non_monotonic_ids",
    "noise_reason_empty",
    "source_for_declared_use",
)
_QC_WARN_RULES = (
    "time_gap",
    "segment_too_short",
    "segment_too_long",
    "missing_hand",
    "action_vocab_offlist",
    "category_imbalance",
)


# ---------------------------------------------------------------------------
# 输入模型
# ---------------------------------------------------------------------------


class TaskAnnotation(BaseModel):
    """一条任务级语义标注。"""

    episode_key: str = Field(
        description="episode 标识（取自 resolve_anchors 返回的 episode_key）")
    task: str = Field(description="任务描述（用户给定或数据中的语言指令原文）")
    task_zh: str = Field(default="", description="中文描述（可选）")
    category: str = Field(
        default="", description="任务类别（自由文本，建议取自配置的动词表）")


class SegmentAnnotation(BaseModel):
    """一条切片标注（可选含动作名称）。"""

    episode_key: str = Field(description="episode 标识（取自 resolve_anchors）")
    id: int = Field(default=1, description="片段序号（同一 episode 内递增）")
    start_s: float | None = Field(default=None, description="起始秒")
    end_s: float | None = Field(default=None, description="结束秒")
    start_timestamp: str = Field(
        default="", description="起始时间 HH:MM:SS（可选，与 start_s 互为逆）")
    end_timestamp: str = Field(
        default="", description="结束时间 HH:MM:SS（可选，与 end_s 互为逆）")
    start_frame: int | None = Field(default=None, description="起始帧号（可选）")
    end_frame: int | None = Field(default=None, description="结束帧号（可选）")
    atomic_action: str = Field(
        default="",
        description="原子动作（受控词表、下划线风格，如 Pick_Up / Place；"
                    "**必须由用户提供或确认**，不得由你编造）")
    action_description: str = Field(
        default="", description="动作描述（中文；同上，需用户提供或确认）")
    action_description_en: str = Field(default="", description="英文描述（可选）")
    interacting_hand: str = Field(
        default="", description="交互手：left_hand/right_hand/both_hands/none")
    target_object_class: str = Field(default="", description="目标物体类别（可选）")
    is_noise: bool = Field(default=False, description="是否标记为噪声片段")
    cleaning_reason: str = Field(
        default="", description="标记为噪声时必须给出原因（is_noise=true 时必填）")


# ---------------------------------------------------------------------------
# 推断辅助
# ---------------------------------------------------------------------------


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """按候选名查找列（精确优先，再前缀匹配）。"""
    lowered = {str(c).lower().strip(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _lerobot_tasks(context: RunContext) -> list[dict[str, Any]]:
    """读取 LeRobot 的 ``meta/tasks.jsonl``（任务标注的权威来源）。

    LeRobot v2 把自然语言任务指令放在 ``meta/tasks.jsonl``（任务文本 →
    整数 ID 映射）；这比让模型猜测任务要可靠得多，因此**优先**用它。

    Returns:
        任务记录列表 [{task, task_index}]；读不到返回空列表。
    """
    src = str(context.meta.get("source", "") or "")
    if not src:
        return []
    p = Path(src) / "meta" / "tasks.jsonl"
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    task = obj.get("task") or obj.get("name")
                    if task:
                        out.append({
                            "task": str(task),
                            "task_index": obj.get("task_index"),
                        })
    except OSError:
        return []
    return out


def _lerobot_episode_tasks(context: RunContext) -> dict[str, Any]:
    """读取 LeRobot 每 episode 的任务映射（``meta/episodes.jsonl``）。

    Args:
        context: 运行时上下文。

    Returns:
        {episode_key: {"task": str, "task_index": int}}；读不到返回空 dict。
    """
    src = str(context.meta.get("source", "") or "")
    if not src:
        return {}
    p = Path(src) / "meta" / "episodes.jsonl"
    if not p.exists():
        return {}
    out: dict[str, Any] = {}
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                ep = obj.get("episode_index")
                if ep is None:
                    ep = obj.get("episode_id")
                if ep is None:
                    continue
                tasks = obj.get("tasks") or []
                if isinstance(tasks, str):
                    tasks = [tasks]
                if tasks:
                    out[str(ep)] = {
                        "task": str(tasks[0]),
                        "all_tasks": [str(t) for t in tasks],
                    }
    except OSError:
        return {}
    return out


def _task_candidates_from_columns(context: RunContext) -> dict[str, Any]:
    """从数据表的语言/任务列提取任务候选（确定性，逐 episode 取唯一值）。"""
    df = context.df
    if df is None or df.empty:
        return {"found": False, "reason": "无主数据表"}

    col = _find_col(df, _TASK_COLS)
    if col is None:
        return {
            "found": False,
            "reason": (
                f"主表未找到任务/指令列（候选名：{'、'.join(_TASK_COLS[:6])}…）"
            ),
        }

    ep_col = _find_col(df, ("episode_index", "episode", "ep", "episode_id",
                            "traj_id", "trajectory_id"))
    per_episode: dict[str, Any] = {}
    if ep_col is not None:
        for key, sub in df.groupby(ep_col, sort=True, dropna=False):
            vals = [str(v) for v in sub[col].dropna().unique() if str(v).strip()]
            if vals:
                per_episode[str(key)] = vals if len(vals) > 1 else vals[0]
    else:
        vals = [str(v) for v in df[col].dropna().unique() if str(v).strip()]
        if vals:
            per_episode[store.WHOLE_DATASET_KEY] = (
                vals if len(vals) > 1 else vals[0])

    return {
        "found": bool(per_episode),
        "column": col,
        "per_episode": per_episode,
        "reason": "" if per_episode else f"列 {col} 无有效非空值",
    }


# ---------------------------------------------------------------------------
# 标注自身质检（check_annotation_qc）
# ---------------------------------------------------------------------------


def check_annotation_qc_impl(
    context: RunContext,
    *,
    use: str = USE_AUDIT,
    scope: str | None = None,
    episode_key: str | None = None,
    settings=None,
) -> dict[str, Any]:
    """对已落盘的标注做质检（确定性规则为主，是标注可用性的最后闸门）。

    **与 ``check_dataset_quality`` 的分工**：那个查**数据**，这个查**标注**。
    本工具的规则确定性最高（重叠、空隙、超界、序号、噪声原因），几乎零误报，
    因此**允许判 fail**（不需要像数据质检那样把启发式指标压制为 warn）。

    规则分两层（与数据集质检同构，便于模型统一转述）：

    - **可判 fail**：``time_overlap``（片段重叠）、``bounds_exceeded``
      （超出 episode 时长）、``non_monotonic_ids``（序号重复/不递增）、
      ``noise_reason_empty``（标了噪声却无原因）、
      ``source_for_declared_use``（来源与声明的用途不兼容，如 ``llm_proposed``
      却要用于训练真值）。
    - **仅 warn**：``time_gap``（片段间有未覆盖空隙，可能漏标）、
      ``segment_too_short`` / ``segment_too_long``（粒度离群）、
      ``missing_hand``（有目标物体却未标交互手）、``action_vocab_offlist``
      （动作名不在词表内）、``category_imbalance``（某动作占比过高）。

    Args:
        context: 运行时上下文（取 output_dir / dataset_id）。
        use: 标注将要用于的用途（training/publication/audit），
            用于 ``source_for_declared_use`` 校验。
        scope: 只查某作用域（缺省两者都查）。
        episode_key: 只查某 episode。
        settings: 可选配置覆盖（测试注入）。

    Returns:
        dict，含 success、result（pass/warn/fail）、failures、warnings、
        per_episode（逐 episode 的片段连续性统计）、n_records、user_message。
    """
    settings = settings or get_settings()

    output_dir = context.output_dir or "outputs"
    loaded = store.load_annotations(
        output_dir, context.dataset_id, scope=scope, episode_key=episode_key)

    records = loaded["records"]
    # 阈值与来源信息**在所有返回路径上都必须齐备**（模型需要它们来解释判定，
    # 缺失会导致它无法说明"阈值未经该数据集验证"）。
    _thresholds = {
        "min_segment_s": settings.annotation_min_segment_s,
        "max_segment_s": settings.annotation_max_segment_s,
        "max_gap_s": settings.annotation_max_gap_s,
    }
    if not records:
        return {
            "success": True,
            "dataset": context.dataset_id,
            "check": "check_annotation_qc",
            "result": "pass",
            "use": use,
            "n_records": 0,
            "n_segments": 0,
            "n_tasks": 0,
            "failures": [],
            "warnings": [],
            "failed_rules": [],
            "warned_rules": [],
            "per_episode": {},
            "bad_lines": loaded["bad_lines"],
            "thresholds": _thresholds,
            "vocab_configured": False,
            "user_message": (
                "当前数据集没有任何标注可质检。请先产出并落盘标注"
                "（任务标注或切片标注）。"
            ),
        }

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    segs = [r for r in records if r.get("scope") == SCOPE_SEGMENT]
    tasks = [r for r in records if r.get("scope") == SCOPE_TASK]

    # ---- 逐 episode 检查切片 ----
    per_episode: dict[str, Any] = {}
    min_s = settings.annotation_min_segment_s
    max_s = settings.annotation_max_segment_s
    max_gap = settings.annotation_max_gap_s

    by_ep: dict[str, list[dict[str, Any]]] = {}
    for r in segs:
        by_ep.setdefault(str(r.get("episode_key") or ""), []).append(r)

    # 取锚点以校验"是否超出 episode 时长"（锚点不可得时跳过该规则）。
    anchor_map: dict[str, store.EpisodeAnchor] = {}
    try:
        anchors = store.resolve_anchors(context)
        if anchors.get("success"):
            for a in anchors["anchors"]:
                anchor_map[str(a["episode_key"])] = store.EpisodeAnchor(**a)
    except (TypeError, KeyError):
        anchor_map = {}

    for ep, rows in by_ep.items():
        rows_sorted = sorted(rows, key=lambda r: float(r.get("start_s") or 0.0))
        ids = [r.get("id") for r in rows_sorted]

        # 序号：重复或非递增。
        dup_ids = {i for i in ids if ids.count(i) > 1}
        if dup_ids:
            failures.append({
                "rule": "non_monotonic_ids",
                "episode_key": ep,
                "detail": f"片段 id 重复：{sorted(dup_ids, key=str)}",
            })
        elif ids != sorted(ids, key=lambda x: (x is None, x)):
            failures.append({
                "rule": "non_monotonic_ids",
                "episode_key": ep,
                "detail": f"片段 id 未按时间递增：{ids}",
            })

        # 重叠与空隙（相邻片段）。
        overlaps: list[str] = []
        gaps: list[str] = []
        for prev, cur in zip(rows_sorted, rows_sorted[1:]):
            pe = prev.get("end_s")
            cs = cur.get("start_s")
            if pe is None or cs is None:
                continue
            pe_f, cs_f = float(pe), float(cs)
            if cs_f < pe_f - 1e-9:
                overlaps.append(
                    f"id {prev.get('id')}(end {pe_f}) 与 id {cur.get('id')}"
                    f"(start {cs_f}) 重叠")
            elif cs_f > pe_f + max_gap:
                gaps.append(
                    f"id {prev.get('id')}→{cur.get('id')} 间隙 "
                    f"{round(cs_f - pe_f, 3)}s")

        if overlaps:
            failures.append({
                "rule": "time_overlap",
                "episode_key": ep,
                "detail": "；".join(overlaps[:5]),
                "n": len(overlaps),
            })
        if gaps:
            warnings.append({
                "rule": "time_gap",
                "episode_key": ep,
                "detail": "；".join(gaps[:5]),
                "n": len(gaps),
                "note": (
                    f"间隙超过 {max_gap}s，可能是漏标片段——"
                    "若该段时间确实不属于任何动作，可忽略"
                ),
            })

        # 超出 episode 实际时长。
        anchor = anchor_map.get(ep)
        if anchor is not None and anchor.duration_s is not None:
            dur = anchor.duration_s
            over = [
                f"id {r.get('id')} end_s={r.get('end_s')} > episode 时长 {dur}"
                for r in rows_sorted
                if r.get("end_s") is not None
                and float(r["end_s"]) > dur + max_gap
            ]
            if over:
                failures.append({
                    "rule": "bounds_exceeded",
                    "episode_key": ep,
                    "detail": "；".join(over[:5]),
                    "episode_duration_s": dur,
                })

        # 粒度离群。
        for r in rows_sorted:
            if r.get("start_s") is None or r.get("end_s") is None:
                continue
            d = float(r["end_s"]) - float(r["start_s"])
            if d < min_s:
                warnings.append({
                    "rule": "segment_too_short",
                    "episode_key": ep, "id": r.get("id"),
                    "value": round(d, 4), "threshold": min_s,
                    "detail": f"片段时长 {round(d, 3)}s 短于下限 {min_s}s",
                })
            elif d > max_s:
                warnings.append({
                    "rule": "segment_too_long",
                    "episode_key": ep, "id": r.get("id"),
                    "value": round(d, 4), "threshold": max_s,
                    "detail": f"片段时长 {round(d, 3)}s 超过上限 {max_s}s",
                })

        # 噪声原因缺失 + 语义一致性。
        for r in rows_sorted:
            if r.get("is_noise") and not str(r.get("cleaning_reason") or "").strip():
                failures.append({
                    "rule": "noise_reason_empty",
                    "episode_key": ep, "id": r.get("id"),
                    "detail": "is_noise=true 但 cleaning_reason 为空",
                })
            hand = str(r.get("interacting_hand") or "").strip()
            obj = r.get("target_object_class")
            if obj and (not hand or hand == "none"):
                warnings.append({
                    "rule": "missing_hand",
                    "episode_key": ep, "id": r.get("id"),
                    "detail": (
                        f"标注了目标物体 {obj!r} 但未标注交互手——"
                        "两者语义上通常应同时存在，请确认是否遗漏"
                    ),
                })
            if hand and hand not in _HAND_VALUES:
                warnings.append({
                    "rule": "missing_hand",
                    "episode_key": ep, "id": r.get("id"),
                    "detail": (
                        f"interacting_hand={hand!r} 不在统一枚举 "
                        f"{list(_HAND_VALUES)} 内"
                    ),
                })

        per_episode[ep] = {
            "n_segments": len(rows_sorted),
            "n_overlaps": len(overlaps),
            "n_gaps": len(gaps),
        }

    # ---- 动作词表与类别分布 ----
    vocab = _load_action_vocab(context, settings)
    if vocab:
        off = sorted({
            str(r.get("atomic_action"))
            for r in segs
            if r.get("atomic_action") and str(r["atomic_action"]) not in vocab
        })
        if off:
            warnings.append({
                "rule": "action_vocab_offlist",
                "detail": (
                    f"{len(off)} 个动作名不在当前词表内：{off[:10]}——"
                    "若确认应纳入，请扩词表；否则请统一命名"
                ),
                "off_list": off[:20],
            })

    if segs:
        counts: dict[str, int] = {}
        for r in segs:
            a = str(r.get("atomic_action") or "").strip()
            if a:
                counts[a] = counts.get(a, 0) + 1
        if counts:
            total = sum(counts.values())
            top, top_n = max(counts.items(), key=lambda kv: kv[1])
            if total >= 5 and top_n / total > 0.8:
                warnings.append({
                    "rule": "category_imbalance",
                    "detail": (
                        f"动作 {top!r} 占 {round(top_n / total * 100, 1)}% "
                        "（样本可能不平衡，训练时需注意）"
                    ),
                    "top_action": top, "ratio": round(top_n / total, 4),
                })

    # ---- 来源与用途合规（硬约束，可 fail）----
    use_check = store.check_source_for_use(records, use)
    if not use_check["ok"]:
        for v in use_check["violations"][:10]:
            failures.append({
                "rule": "source_for_declared_use",
                "episode_key": v.get("episode_key"),
                "id": v.get("id"),
                "detail": v.get("reason"),
            })

    # ---- 汇总 ----
    result = "fail" if failures else ("warn" if warnings else "pass")

    fail_rules = sorted({f["rule"] for f in failures})
    warn_rules = sorted({w["rule"] for w in warnings})
    parts = [
        f"标注质检判定：{result}（共 {len(records)} 条："
        f"切片 {len(segs)}、任务 {len(tasks)}）。"
    ]
    if failures:
        parts.append(
            f"**必须修正**的确定性问题 {len(failures)} 项"
            f"（{('、'.join(fail_rules))}）："
            + "；".join(f["detail"] for f in failures[:3]) + "。"
        )
    if warnings:
        parts.append(
            f"另有 {len(warnings)} 项提示（{('、'.join(warn_rules))}）——"
            "提示不等于错误，部分可能是数据本身特性（如片段间隔确实不属于"
            "任何动作），请人工判断。"
        )
    if not failures and not warnings:
        parts.append("未发现标注质量问题。")
    if loaded["bad_lines"]:
        parts.append(
            f"另有 {loaded['bad_lines']} 行标注文件损坏已被跳过（未计入统计）。"
        )
    parts.append(
        f"用途校验基准：{use}——"
        + (
            "该用途下所有来源均合规。"
            if use_check["ok"]
            else "存在来源与用途不兼容的记录（见上）。"
        )
    )

    return {
        "success": True,
        "dataset": context.dataset_id,
        "check": "check_annotation_qc",
        "result": result,
        "use": use,
        "n_records": len(records),
        "n_segments": len(segs),
        "n_tasks": len(tasks),
        "failures": failures,
        "warnings": warnings,
        "failed_rules": fail_rules,
        "warned_rules": warn_rules,
        "per_episode": per_episode,
        "bad_lines": loaded["bad_lines"],
        "thresholds": _thresholds,
        "vocab_configured": bool(vocab),
        "user_message": "".join(parts),
    }


def _load_action_vocab(context: RunContext, settings) -> set[str]:
    """加载原子动作词表（用于 ``action_vocab_offlist`` 检查）。

    来源优先级：数据集画像目录下的 ``action_vocab.json``（按数据集配置，
    符合用户裁决 #1/#3）→ 无则返回空集（此时不做词表检查，如实报告）。
    """
    from app.tools.output_paths import dataset_output_dir

    p = dataset_output_dir(context.output_dir or "outputs", context.dataset_id)
    f = p / "action_vocab.json"
    if not f.exists():
        return set()
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeDecodeError):
        return set()
    if isinstance(obj, dict):
        items = obj.get("actions") or obj.get("atomic_actions") or []
    elif isinstance(obj, list):
        items = obj
    else:
        return set()
    return {str(x) for x in items if str(x).strip()}


# ---------------------------------------------------------------------------
# 任务语义标注
# ---------------------------------------------------------------------------


def annotate_task_impl(
    context: RunContext,
    annotations: list[dict] | None = None,
    *,
    discover: bool = True,
    confirm: bool = False,
    settings=None,
) -> dict[str, Any]:
    """任务级语义标注：确定性发现候选 → 用户确认 → 落盘。

    **三态闸门**（与 ``propose_stream_semantics`` 同构）：

    1. 先在数据中**确定性**找任务候选（LeRobot ``meta/tasks.jsonl`` 与
       ``meta/episodes.jsonl``、数据表的语言/指令列唯一值）；
    2. 找不到就返回 ``no_task_signal`` 并请用户提供，**不由模型编造**；
    3. ``confirm=False`` 只登记不落盘；用户明确同意后以 ``confirm=True``
       重新调用才落盘，来源标 ``user_confirmed``。

    Args:
        context: 运行时上下文。
        annotations: 待落盘的任务标注（每条含 episode_key/task/…）。
        discover: 是否执行确定性候选发现（缺省 True）。
        confirm: 是否落盘。

    Returns:
        dict，含 success、discovered（候选）、validated、annotations、
        saved（落盘结果）、user_message。
    """
    settings = settings or get_settings()

    if context.dataset_id is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset。",
        }

    out: dict[str, Any] = {
        "success": True,
        "dataset": context.dataset_id,
    }

    # ---- 1) 确定性候选发现 ----
    if discover:
        candidates: dict[str, Any] = {}
        sources: list[str] = []

        lr_tasks = _lerobot_tasks(context)
        if lr_tasks:
            sources.append("meta/tasks.jsonl")
            candidates["lerobot_tasks"] = lr_tasks[:50]

        ep_tasks = _lerobot_episode_tasks(context)
        if ep_tasks:
            sources.append("meta/episodes.jsonl")
            candidates["lerobot_episode_tasks"] = dict(
                list(ep_tasks.items())[:50])

        col_cand = _task_candidates_from_columns(context)
        if col_cand.get("found"):
            sources.append(f"数据列 {col_cand['column']}")
            candidates["column_uniques"] = col_cand["per_episode"]

        out["discovered"] = candidates
        out["discover_sources"] = sources
        out["has_candidate"] = bool(candidates)

        if not candidates:
            out["success"] = False
            out["error"] = "no_task_signal"
            out["reason"] = col_cand.get("reason") or "未找到任何任务语义线索"
            out["user_message"] = (
                "未能在该数据中找到任务语义线索"
                "（已查：LeRobot 的 meta/tasks.jsonl、meta/episodes.jsonl、"
                f"数据表语言列）。原因：{col_cand.get('reason') or '无'}\n\n"
                "**请用户直接提供任务描述**（例如「把红色方块放进碗里」），"
                "我再登记并落盘。任务描述属于领域知识，**不能由我推测**——"
                "凭数据形态猜任务等于编造。"
            )
            out["suggested_tools"] = ["profile_data", "inspect_streams"]
            if not annotations:
                return out

    # ---- 2) 校验并规范化待落盘标注 ----
    results: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []

    # 取锚点以校验 episode_key 是否真实存在（避免把标注挂到不存在的 episode）。
    valid_keys: set[str] | None = None
    try:
        anchors = store.resolve_anchors(context)
        if anchors.get("success"):
            valid_keys = {str(a["episode_key"]) for a in anchors["anchors"]}
    except (TypeError, KeyError):
        valid_keys = None

    for item in (annotations or []):
        rec = dict(item)
        key = str(rec.get("episode_key") or "").strip()
        rec.setdefault("scope", SCOPE_TASK)
        # 落盘来源：经用户确认才标 user_confirmed。
        rec["source"] = SOURCE_USER if confirm else SOURCE_LLM
        rec["confidence"] = "high" if confirm else "low"

        norm = store.normalize_record(rec, scope=SCOPE_TASK, episode_key=key)
        if not norm["ok"]:
            results.append({
                "episode_key": key, "ok": False,
                "error": norm.get("error"), "reason": norm.get("reason"),
            })
            continue
        if valid_keys is not None and key not in valid_keys:
            results.append({
                "episode_key": key, "ok": False,
                "error": "unknown_episode_key",
                "reason": (
                    f"episode_key={key!r} 不在当前数据集的锚点清单内"
                    f"（可用：{sorted(valid_keys)[:8]}…）"
                ),
            })
            continue
        results.append({"episode_key": key, "ok": True, "task": norm["record"]["task"]})
        valid_records.append(norm["record"])

    out["results"] = results
    out["validated"] = len(valid_records)
    out["rejected"] = len(results) - len(valid_records)

    # ---- 3) 落盘（闸门）----
    if not confirm:
        out["saved"] = None
        out["user_message"] = (
            f"已登记 {len(valid_records)} 条任务标注，**尚未落盘**。"
            "请向用户转述候选与待落盘内容并请求确认；用户同意后以 "
            "confirm=True 重新调用即可持久化（来源将标记为 user_confirmed）。"
            if valid_records else
            "没有可落盘的有效任务标注（详见 results 中的拒绝原因）。"
        )
        if annotations and not valid_records:
            out["success"] = False
            out["error"] = "no_valid_annotation"
        return out

    if not valid_records:
        out["success"] = False
        out["error"] = "no_valid_annotation"
        out["saved"] = None
        out["user_message"] = (
            "confirm=True 但没有可落盘的有效任务标注——"
            "请先修正 results 中列出的拒绝原因。"
        )
        return out

    saved = store.append_annotations(
        context.output_dir or "outputs",
        context.dataset_id,
        valid_records,
        actor=SOURCE_USER,
        reason="任务级语义标注落盘（用户确认）",
        session_tag=context.session_tag or "",
    )
    out["saved"] = saved
    out["user_message"] = (
        f"已落盘 {saved['saved']} 条、更新 {saved['updated']} 条任务标注"
        f"（来源 user_confirmed）。"
        + (f"跳过 {saved['skipped']} 条。" if saved["skipped"] else "")
        + "这些标注可用于全部用途（含训练真值）。"
    )
    return out


# ---------------------------------------------------------------------------
# 切片/任务标注落盘
# ---------------------------------------------------------------------------


def save_annotations_impl(
    context: RunContext,
    segments: list[dict] | None = None,
    tasks: list[dict] | None = None,
    *,
    source: str = SOURCE_SIGNAL,
    confirmed_by_user: bool = False,
    use: str | None = None,
    actor: str = "",
    reason: str = "",
    dry_run: bool = False,
    settings=None,
) -> dict[str, Any]:
    """落盘切片/任务标注（来源标记 + 复核闸门 + 用途合规校验）。

    **来源与复核的语义（重要，避免误标）**：

    - ``confirmed_by_user=True`` → 来源强制为 ``user_confirmed``（人工确认过）；
    - ``confirmed_by_user=False`` → 来源用调用方给的 ``source``
      （``signal_derived`` 机械切分 或 ``llm_proposed`` 模型提出）；
    - ``use="training"`` 时若来源为 ``llm_proposed``/``signal_derived``，
      **拒绝落盘并说明原因**——这是"三种用途"落到代码里的强制机制。

    Args:
        context: 运行时上下文。
        segments: 切片标注清单（字段见 ``SegmentAnnotation``）。
        tasks: 任务级标注清单。
        source: 来源（当 ``confirmed_by_user=False`` 时生效）。
        confirmed_by_user: 是否已经用户确认。
        use: 声明的用途；给出时执行用途合规校验。
        actor: 变更发起者（进修订日志）。
        reason: 变更原因（进修订日志）。
        dry_run: 只校验不落盘。

    Returns:
        dict，含 success、validated、rejected、saved、use_check、
        user_message。
    """
    settings = settings or get_settings()

    if context.dataset_id is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset。",
        }

    eff_source = SOURCE_USER if confirmed_by_user else source
    if eff_source not in (SOURCE_USER, SOURCE_SIGNAL, SOURCE_LLM, SOURCE_IMPORTED):
        return {
            "success": False,
            "error": "invalid_source",
            "reason": f"source 非法：{source!r}",
            "user_message": (
                "来源标记非法。可用：user_confirmed（用户确认）、"
                "signal_derived（机械切分）、llm_proposed（模型提出）、"
                "imported（外部导入）。"
            ),
        }
    if use is not None and use not in (USE_TRAINING, USE_PUBLICATION, USE_AUDIT):
        return {
            "success": False,
            "error": "invalid_use",
            "reason": f"use 非法：{use!r}",
            "user_message": (
                "用途非法。可用：training（训练真值）、publication（发布元数据）、"
                "audit（内部清查）。"
            ),
        }

    # 锚点（校验 episode_key 与补齐帧号）。
    anchor_map: dict[str, store.EpisodeAnchor] = {}
    try:
        anchors = store.resolve_anchors(context)
        if anchors.get("success"):
            for a in anchors["anchors"]:
                anchor_map[str(a["episode_key"])] = store.EpisodeAnchor(**a)
    except (TypeError, KeyError):
        anchor_map = {}

    results: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []

    def _prepare(items: list[dict] | None, scope: str) -> None:
        for item in (items or []):
            rec = dict(item)
            key = str(rec.get("episode_key") or "").strip()
            rec["source"] = eff_source
            if confirmed_by_user:
                rec["confidence"] = rec.get("confidence") or "high"
            anchor = anchor_map.get(key)
            if anchor_map and anchor is None:
                results.append({
                    "scope": scope, "episode_key": key, "ok": False,
                    "error": "unknown_episode_key",
                    "reason": (
                        f"episode_key={key!r} 不在锚点清单内"
                        f"（可用：{sorted(anchor_map)[:8]}…）"
                    ),
                })
                continue
            # 交互手枚举归一（用户裁决 #2）。
            hand = str(rec.get("interacting_hand") or "").strip()
            if hand and hand not in _HAND_VALUES:
                results.append({
                    "scope": scope, "episode_key": key, "ok": False,
                    "error": "invalid_interacting_hand",
                    "reason": (
                        f"interacting_hand={hand!r} 不在统一枚举 "
                        f"{list(_HAND_VALUES)} 内（用户已裁决统一枚举）"
                    ),
                })
                continue

            norm = store.normalize_record(
                rec, scope=scope, episode_key=key, anchor=anchor)
            if not norm["ok"]:
                results.append({
                    "scope": scope, "episode_key": key, "ok": False,
                    "error": norm.get("error"), "reason": norm.get("reason"),
                })
                continue
            results.append({
                "scope": scope, "episode_key": key, "ok": True,
                "id": norm["record"].get("id"),
            })
            valid.append(norm["record"])

    _prepare(segments, SCOPE_SEGMENT)
    _prepare(tasks, SCOPE_TASK)

    # **同一批次内的重复 id 必须拒绝**（重要设计决定）：
    # 落盘的去重键是 (scope, episode_key, id)——重复 id 会被**静默合并**
    # （后一条覆盖前一条），导致用户以为写入了两个片段、实际只存了一个。
    # 这类"意图冲突"必须在入口拦下并如实报告，而不是让下游质检去发现一个
    # 在磁盘上永远不可能存在的状态。
    dup_keys: dict[tuple[str, str, str], list[int]] = {}
    for idx, r in enumerate(valid):
        k = (str(r.get("scope")), str(r.get("episode_key")), str(r.get("id")))
        dup_keys.setdefault(k, []).append(idx)
    conflict = {k: v for k, v in dup_keys.items() if len(v) > 1}
    if conflict:
        detail = [
            {
                "scope": k[0], "episode_key": k[1], "id": k[2],
                "reason": (
                    f"同一批次内 id={k[2]} 出现 {len(v)} 次——须为每个片段"
                    "分配唯一 id（同一 episode 内递增）"
                ),
            }
            for k, v in conflict.items()
        ]
        return {
            "success": False,
            "error": "duplicate_ids_in_batch",
            "dataset": context.dataset_id,
            "results": results + [{**d, "ok": False} for d in detail],
            "validated": len(valid),
            "rejected": len(results),
            "saved": None,
            "user_message": (
                f"**拒绝落盘**：同批次内存在 {len(conflict)} 个重复 id。"
                "落盘以 (episode_key, id) 为键，重复 id 会导致后一条静默覆盖"
                "前一条（数据丢失且无提示）。请为每个片段分配唯一 id。\n"
                + "；".join(d["reason"] for d in detail[:3])
            ),
        }

    if not valid:
        return {
            "success": False,
            "error": "no_valid_annotation",
            "dataset": context.dataset_id,
            "results": results,
            "validated": 0,
            "rejected": len(results),
            "saved": None,
            "user_message": (
                "没有可落盘的有效标注——请修正 results 中的拒绝原因。"
            ),
        }

    # ---- 用途合规校验（硬闸门）----
    use_check = None
    if use is not None:
        use_check = store.check_source_for_use(valid, use)
        if not use_check["ok"]:
            return {
                "success": False,
                "error": "source_use_incompatible",
                "dataset": context.dataset_id,
                "results": results,
                "validated": len(valid),
                "rejected": len(results) - len(valid),
                "use_check": use_check,
                "saved": None,
                "user_message": (
                    f"**拒绝落盘**：来源 {eff_source!r} 与用途 {use!r} 不兼容。\n"
                    + "；".join(
                        v.get("reason", "") for v in use_check["violations"][:5]
                    )
                    + "\n\n请先经人工复核（以 confirmed_by_user=True 重新调用，"
                    "来源将变为 user_confirmed），或改用 audit/publication 用途。"
                ),
            }

    if dry_run:
        return {
            "success": True,
            "dataset": context.dataset_id,
            "results": results,
            "validated": len(valid),
            "rejected": len(results) - len(valid),
            "use_check": use_check,
            "saved": None,
            "effective_source": eff_source,
            "dry_run": True,
            "user_message": (
                f"校验通过（{len(valid)} 条），dry_run 模式**未落盘**。"
                f"来源将为 {eff_source}。"
            ),
        }

    saved = store.append_annotations(
        context.output_dir or "outputs",
        context.dataset_id,
        valid,
        actor=actor or eff_source,
        reason=reason or f"标注落盘（来源 {eff_source}）",
        session_tag=context.session_tag or "",
    )

    gen_note = ""
    if use_check is not None and use_check.get("must_declare_generated"):
        gen_note = (
            f" 另有 {use_check['must_declare_generated']} 条为自动生成来源，"
            "发布时须声明 generated=true。"
        )

    # 落盘后自动跑一次标注质检（闭环：写完立即校验，问题当下暴露）。
    qc = None
    if not use_check or use_check.get("ok"):
        qc = check_annotation_qc_impl(
            context, use=use or USE_AUDIT, settings=settings)

    return {
        "success": True,
        "dataset": context.dataset_id,
        "results": results,
        "validated": len(valid),
        "rejected": len(results) - len(valid),
        "use_check": use_check,
        "effective_source": eff_source,
        "saved": saved,
        "post_save_qc": (
            {
                "result": qc["result"],
                "n_failures": len(qc["failures"]),
                "n_warnings": len(qc["warnings"]),
                "failed_rules": qc["failed_rules"],
                "user_message": qc["user_message"],
            } if qc else None
        ),
        "user_message": (
            f"已落盘 {saved['saved']} 条、更新 {saved['updated']} 条标注"
            f"（来源 {eff_source}）。"
            + (f"跳过 {saved['skipped']} 条。" if saved["skipped"] else "")
            + gen_note
            + (f" 落盘后自动质检：{qc['result']}。{qc['user_message']}" if qc else "")
        ),
    }


# ---------------------------------------------------------------------------
# @tool 包装
# ---------------------------------------------------------------------------


@tool
def annotate_task(
    wrapper: RunContextWrapper[RunContext],
    annotations: list[TaskAnnotation] | None = None,
    discover: bool = True,
    confirm: bool = False,
) -> dict:
    """登记/落盘任务级语义标注（整段任务描述）。

    **任务描述不得由你编造**：先在数据中确定性查找（LeRobot 的
    meta/tasks.jsonl、meta/episodes.jsonl、数据表语言列）；找到则作为候选
    转述给用户；**找不到就请用户提供**。凭数据形态猜任务等于编造（违反纪律 1）。

    使用纪律（闸门）：``confirm=False`` 只登记返回、**不落盘**；**必须在用户
    明确同意后**才可传 ``confirm=True``。

    Args:
        annotations: 任务级标注清单，每条含 episode_key（取自锚点清单）、
            task（任务描述）、task_zh（可选中文）、category（可选类别）。
            不传时仅执行候选发现。
        discover: 是否执行确定性候选发现（缺省 True）。
        confirm: 是否落盘。

    Returns:
        dict，含 discovered（确定性候选）、discover_sources、results（逐条
        校验结果）、validated、saved、user_message。无候选且无输入时
        success=False + error="no_task_signal"。
    """
    return annotate_task_impl(
        wrapper.context,
        [a.model_dump() for a in annotations] if annotations else None,
        discover=discover,
        confirm=confirm,
    )


@tool
def save_annotations(
    wrapper: RunContextWrapper[RunContext],
    segments: list[SegmentAnnotation] | None = None,
    tasks: list[TaskAnnotation] | None = None,
    source: str = SOURCE_SIGNAL,
    confirmed_by_user: bool = False,
    use: str | None = None,
    reason: str = "",
    dry_run: bool = False,
) -> dict:
    """落盘任务/切片标注（来源标记 + 复核闸门 + 用途合规校验）。

    **来源必须如实标记**：``segment_actions`` 的机械切分传
    ``source="signal_derived"``；由你提出的动作名称或描述传
    ``source="llm_proposed"``。用户已确认过的内容传
    ``confirmed_by_user=True``（来源将成为 user_confirmed）。

    **用途闸门**：若 ``use="training"``（下游训练真值），来源为
    ``llm_proposed`` 或 ``signal_derived`` 会被**拒绝落盘**——这两类必须先经
    人工复核。请如实告知用户这一限制，不要试图绕开。

    Args:
        segments: 切片标注清单（episode_key/id/start_s/end_s/atomic_action/
            action_description/interacting_hand/ 等）。动作名称若由你提出，
            必须先确认字段标为 llm_proposed，且提醒用户复合核。
        tasks: 任务级标注清单。
        source: 来源：signal_derived / llm_proposed / imported（
            confirmed_by_user=True 时此项被覆盖为 user_confirmed）。
        confirmed_by_user: 是否已获用户确认。
        use: 声明用途：training / publication / audit；给出时执行合规校验。
        reason: 变更原因（进修订日志，供回溯）。
        dry_run: 只校验不落盘。

    Returns:
        dict，含 validated、rejected、results、use_check、saved、
        post_save_qc（落盘后自动质检结果）、user_message。
    """
    return save_annotations_impl(
        wrapper.context,
        [s.model_dump() for s in segments] if segments else None,
        [t.model_dump() for t in tasks] if tasks else None,
        source=source,
        confirmed_by_user=confirmed_by_user,
        use=use,
        reason=reason,
        dry_run=dry_run,
    )


@tool
def check_annotation_qc(
    wrapper: RunContextWrapper[RunContext],
    use: str = "audit",
    scope: str | None = None,
    episode_key: str | None = None,
) -> dict:
    """对已落盘的标注做质检（重叠/空隙/超界/序号/噪声原因/来源用途合规）。

    **与 check_dataset_quality 的分工**：那个查**数据**，这个查**标注**。
    本工具的规则确定性高（片段重叠、超出 episode 时长、序号错乱、标了噪声
    却无原因、来源与用途不兼容），因此**重叠等确定性问题可判 fail**。

    Args:
        use: 标注将用于的用途（training/publication/audit）——用于校验
            "来源是否允许该用途"（如 llm_proposed 不得进训练真值）。
        scope: 只查某作用域（task/segment）；缺省两者都查。
        episode_key: 只查某 episode。

    Returns:
        dict，含 result（pass/warn/fail）、failures（必须修正的确定性问题）、
        warnings（提示，可能属正常）、per_episode、thresholds、user_message。
    """
    return check_annotation_qc_impl(
        wrapper.context, use=use, scope=scope, episode_key=episode_key)
