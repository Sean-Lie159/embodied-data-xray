"""流语义批量假设工具（第 3 层 LLM 假设通道的结构化出口）。

背景：对话中 LLM 对信封型数据集的分组推理清晰（IMU/手部跟踪/触觉/tf），
但结论生命周期止于对话——流清单是工具算的，下次加载照样 unknown；且既有
confirm_stream_semantic 从未注册为 @tool，prompt 所称"确认后落盘路径"实际
不存在（agent 调不了）。本工具给 LLM 的推理一个**结构化出口**：

    验证（本工具，确定性）→ 用户确认（对话）→ 落盘（confirm=True）

闸门不变：归类与命名的自由给足（kind/semantic_label 为自由文本），
判定变更必须经工具验证与用户确认。

验证器全部基于确定性特征（嵌套发现的 signal_fields / 展开视图的数值列），
不做任何猜测：证据不足一律 weak，交用户裁决。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from agents import RunContextWrapper
from agents.decorators import tool
from pydantic import BaseModel, Field

from app.agent.context import RunContext
from app.tools import profile_store


class StreamAssumption(BaseModel):
    """单条流语义假设（LLM 提交，工具验证）。"""

    file: str = Field(description="流文件名（如 left_glove_imu_data_palm.jsonl）")
    kind: str = Field(default="unknown", description="语义类型（自由文本：imu/pose/tf/tactile…）")
    semantic_label: str = Field(default="", description="展示标签（建议中文；缺省用 kind）")
    time_column: str | None = Field(
        default=None,
        description="传感器时间列的点分路径（如 data.header.timestamp_us），可选",
    )
    fields: list[str] = Field(
        default_factory=list,
        description="自定义 kind 的结构证据路径清单（可选）",
    )

# 单批假设条数上限（防滥用；26 流量级的数据集远够用）。
_MAX_ASSUMPTIONS = 64
# 验证用采样行数与展开列上限（浅验证，不读全量）。
_VERIFY_SAMPLE_ROWS = 20
_VERIFY_MAX_COLS = 128


def _locate_stream(context: RunContext, filename: str) -> dict[str, Any] | None:
    """按文件名（大小写不敏感）在流登记表定位流。

    h5 节点流（path 带 ::node）额外支持按**节点路径**匹配（如
    "meta/index_map/observation/camera/lf_chest_fisheye"）——Path().name
    对这类 path 只会取到末段，产生歧义（多个 meta/index_map 节点末段相同）。
    """
    name = filename.strip().lower()
    streams = context.meta.get("streams", [])
    for s in streams:
        if Path(s.get("path", "")).name.lower() == name:
            return s
    # h5 节点流：按节点路径（:: 之后部分）匹配。
    for s in streams:
        p = s.get("path", "")
        if "::" in p and p.partition("::")[2].lower() == name:
            return s
    return None


def _load_expanded(path: str, fmt: str) -> tuple[pd.DataFrame | None, str | None]:
    """读取并展开流内容，返回 (df, note)；读取失败返回 (None, None)。

    **经统一读取注册表**：格式分派与 ``"<file>::<node>"`` 解析收敛在
    ``_readers``（此前本函数自己分派 jsonl/json/h5/表格，是重复实现之一）；
    信封展开（object 列 → 点分扁平列）在此保留——验证器在扁平列上找特征。
    """
    from app.tools._readers import ReadRequest, read_stream

    result = read_stream(ReadRequest(
        path_spec=path, want="sample", fmt=fmt,
        limit=_VERIFY_SAMPLE_ROWS, expand=True,
    ))
    if not result.ok or result.frame is None:
        return None, None
    return result.frame, result.expand_note

def _quat_bases(columns: list[str]) -> dict[str, list[str]]:
    """找出展开列中的四元数组：同前缀的 .x/.y/.z/.w 四列。"""
    by_base: dict[str, list[str]] = {}
    for c in columns:
        if "." not in c:
            continue
        base, _, leaf = c.rpartition(".")
        if leaf in ("x", "y", "z", "w"):
            by_base.setdefault(base, []).append(c)
    return {
        b: [f"{b}.{s}" for s in ("x", "y", "z", "w")]
        for b, cols in by_base.items()
        if len(cols) == 4
    }


def _vec_bases(columns: list[str]) -> dict[str, list[str]]:
    """找出展开列中的三维向量组：同前缀的 .x/.y/.z 三列。"""
    by_base: dict[str, list[str]] = {}
    for c in columns:
        if "." not in c:
            continue
        base, _, leaf = c.rpartition(".")
        if leaf in ("x", "y", "z"):
            by_base.setdefault(base, []).append(c)
    return {
        b: [f"{b}.{s}" for s in ("x", "y", "z")]
        for b, cols in by_base.items()
        if len(cols) == 3
    }


def _norm_median(df: pd.DataFrame, cols: list[str]) -> float | None:
    """计算向量列的模长中位数（前 50 行，全数值才有效）。"""
    try:
        sub = df[cols].head(50).apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) == 0:
            return None
        norms = (sub ** 2).sum(axis=1).pow(0.5)
        return float(norms.median())
    except Exception:  # noqa: BLE001
        return None


def _verify_kind(
    kind: str,
    df: pd.DataFrame | None,
    fields: list[str],
) -> tuple[str, str]:
    """按假设的 kind 做确定性验证。

    Returns:
        (verified, evidence)。verified ∈ strong / weak / failed。
    """
    kind_l = (kind or "").strip().lower()

    if df is None:
        return "failed", "无法读取流内容做结构验证（格式不支持或读取失败）"

    columns = [str(c) for c in df.columns]
    quats = _quat_bases(columns)
    vecs = _vec_bases(columns)

    if kind_l == "imu":
        quat_ev = [
            b for b, cols in quats.items()
            if (n := _norm_median(df, cols)) is not None and 0.9 <= n <= 1.1
        ]
        accel_ev = [
            b for b, cols in vecs.items()
            if "accel" in b.lower()
            and (n := _norm_median(df, cols)) is not None
            and 0.5 * 9.8 <= n <= 2 * 9.8
        ]
        gyro_ev = [
            b for b, cols in vecs.items()
            if any(t in b.lower() for t in ("angular", "gyr", "gyro"))
        ]
        found = []
        if quat_ev:
            found.append(f"四元数组 {quat_ev[0]}（模长≈1）")
        if accel_ev:
            found.append(f"加速度向量 {accel_ev[0]}（模长在 0.5g~2g）")
        if gyro_ev:
            found.append(f"角速度向量 {gyro_ev[0]}")
        n_ev = len(found)
        if n_ev >= 2:
            return "strong", "；".join(found)
        if n_ev == 1:
            return "weak", f"仅 {found[0]}（单证据，IMU 通常需姿态+运动分量佐证）"
        return "failed", "未发现四元数/加速度/角速度中的任何 IMU 特征"

    if kind_l == "pose":
        quat_ev = [
            b for b, cols in quats.items()
            if (n := _norm_median(df, cols)) is not None and 0.9 <= n <= 1.1
        ]
        if quat_ev:
            return "strong", f"四元数组 {quat_ev[0]}（模长≈1）"
        return "failed", "未发现模长≈1 的四元数组"

    if kind_l == "tf":
        # 展开视图（expand_envelope）把 list[dict] 消化为扁平列
        # （data.transforms.0.parent_frame_id）——原始 list 结构不复存在，
        # 因此**主要依据展开列名**找 TF 特征（frame_id 键 + transforms 前缀）。
        tf_cols = [
            c for c in columns
            if "transforms" in c.lower() and "frame_id" in c.lower()
        ]
        if tf_cols:
            return "strong", f"展开列含坐标变换特征键（如 {tf_cols[0]}）"
        # 兜底：未展开时值本身是 list[dict] 且含 frame 键。
        for col in columns:
            series = df[col]
            for v in series:
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    keys = {str(k).lower() for k in v[0]}
                    if any("frame" in k for k in keys):
                        return "strong", f"{col} 为 frame 变换记录列表"
                    return "weak", f"{col} 为 dict 列表但无 frame 键"
        return "failed", "未发现 transforms 类结构（展开列或原始值中均无 frame_id 特征）"

    # 自定义/未知 kind：仅结构存在性检查（evidence 路径在展开列中存在）。
    # 设计约定：自定义语义永远 weak（结构对不等于语义对，交用户裁决）。
    hits = [f for f in fields if f in columns]
    if fields and hits:
        return "weak", f"字段存在：{hits[:3]}"
    if not fields:
        return "weak", "未提供 fields，仅登记假设（结构未验证）"
    return "failed", f"fields 均不存在于展开列（抽样 {len(columns)} 列）"


def propose_stream_semantics_impl(
    context: RunContext,
    assumptions: list[dict],
    confirm: bool = False,
) -> dict:
    """批量假设验证与落盘的核心实现（供 @tool 包装与测试直接调用）。"""
    if context.dataset_id is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset。",
        }
    if not assumptions or not isinstance(assumptions, list):
        return {
            "success": False,
            "error": "empty_assumptions",
            "user_message": "assumptions 为空：请提供至少一条 {file, kind, semantic_label} 假设。",
        }
    if len(assumptions) > _MAX_ASSUMPTIONS:
        return {
            "success": False,
            "error": "too_many_assumptions",
            "user_message": f"单批最多 {_MAX_ASSUMPTIONS} 条假设，当前 {len(assumptions)} 条。",
        }

    results: list[dict[str, Any]] = []
    confirmed: list[str] = []
    overrides: dict[str, dict[str, Any]] = {}

    for a in assumptions:
        fname = str(a["file"])
        kind = str(a.get("kind", "") or "unknown")
        label = str(a.get("semantic_label") or kind)
        time_column = a.get("time_column")
        fields = [str(f) for f in (a.get("fields") or [])]

        stream = _locate_stream(context, fname)
        if stream is None:
            results.append({
                "file": fname, "kind": kind, "verified": "failed",
                "evidence": "流登记表中不存在该文件名（未加载或拼写不符）",
            })
            continue

        # 媒体流（视频/音频）：无表格结构可核验——如实区分"媒体流无法结构
        # 验证"与"读取失败"（此前一律 failed + "格式不支持或读取失败"，
        # 对正常视频流是误导性证据）。
        if stream.get("format") in ("video", "mp4", "mov", "avi", "mkv",
                                    "webm", "audio", "m4a", "wav", "mp3"):
            results.append({
                "file": fname, "kind": kind, "semantic_label": str(
                    a.get("semantic_label") or kind),
                "verified": "weak",
                "evidence": (
                    "媒体流：无表格结构可核验（语义假设已登记，"
                    "待用户确认落盘；时间轴佐证见配对的索引表/metainfo）"
                ),
            })
            continue

        df, _note = _load_expanded(stream.get("path", ""), stream.get("format", ""))
        verified, evidence = _verify_kind(kind, df, fields)
        results.append({
            "file": fname, "kind": kind, "semantic_label": label,
            "verified": verified, "evidence": evidence,
            "time_column": time_column,
        })

        if confirm and verified in ("strong", "weak"):
            overrides[Path(stream["path"]).name] = {
                "kind": kind,
                "role": {"role": label, "confidence": "high",
                         "evidence": "用户确认（经 propose_stream_semantics 验证）"},
                "semantic_label": label,
                "label_evidence": (
                    f"用户确认；工具验证：{verified}（{evidence}）"
                ),
                "label_confidence": "high" if verified == "strong" else "low",
            }
            if time_column:
                overrides[Path(stream["path"]).name]["time_column"] = str(time_column)
            confirmed.append(fname)

    summary = {
        "strong": sum(1 for r in results if r["verified"] == "strong"),
        "weak": sum(1 for r in results if r["verified"] == "weak"),
        "failed": sum(1 for r in results if r["verified"] == "failed"),
    }

    out: dict[str, Any] = {
        "success": True,
        "results": results,
        "summary": summary,
        "confirmed": confirmed,
        "dataset": context.dataset_id,
    }
    if confirm:
        if overrides:
            profile_store.save_dataset_profile(
                context.output_dir, context.dataset_id,
                stream_overrides=overrides, pair_overrides=None,
            )
            # 立即应用覆盖到当前会话的流登记表（不用等重载），并清掉受影响流
            # 的采样率缓存——确认的 time_column 可能与缓存所用列不同（如容器
            # 批量写入时间的 ~58 万 Hz 荒谬采样率），重测后 UI/对话即用新口径。
            streams = context.meta.get("streams", [])
            for s in streams:
                fname = Path(s.get("path", "")).name
                if fname in overrides:
                    s["measured_rate"] = None
            context.meta["streams"] = profile_store.apply_profile_overrides(
                streams, profile_store.load_dataset_profile(
                    context.output_dir, context.dataset_id)
            )
        out["user_message"] = (
            f"已把 {len(confirmed)} 条通过验证的假设落盘为用户确认"
            f"（strong {summary['strong']} / weak {summary['weak']}，"
            f"failed {summary['failed']} 条未落盘）。下次加载该数据集时，"
            "流清单将直接显示确认后的语义标签。"
        )
    else:
        out["user_message"] = (
            f"验证完成：strong {summary['strong']} / weak {summary['weak']} / "
            f"failed {summary['failed']}。尚未落盘——请向用户转述验证结果并确认；"
            "用户同意后以 confirm=True 重新调用即可持久化。"
        )
    return out


@tool
def propose_stream_semantics(
    wrapper: RunContextWrapper[RunContext],
    assumptions: list[StreamAssumption],
    confirm: bool = False,
) -> dict:
    """批量提交流语义假设：先确定性验证；confirm=True 时把通过验证的假设落盘。

    使用纪律（闸门）：confirm=False 只验证不落盘；**必须在用户明确同意后**
    才可传 confirm=True——落盘是持久的，下次加载直接生效。

    Args:
        assumptions: 假设清单，每条含 file（文件名，必填）、kind（语义类型，
            自由文本如 imu/pose/tf/tactile）、semantic_label（展示标签，建议
            中文）、time_column（可选，传感器时间列的点分路径）、fields（可选，
            自定义 kind 的结构证据路径清单）。单批最多 64 条。
        confirm: 是否落盘。False=仅验证；True=把 strong/weak 条目写入用户
            确认画像（failed 条目不落盘）。

    Returns:
        dict，含 results（逐条 verified/evidence）、summary
        （strong/weak/failed 计数）、confirmed（confirm=True 时已落盘清单）。
    """
    context = wrapper.context
    # Pydantic 模型 → dict（impl 内部保持 dict 访问）。
    items: list[dict] = [
        {
            "file": a.file,
            "kind": a.kind,
            "semantic_label": a.semantic_label or a.kind,
            "time_column": a.time_column,
            "fields": list(a.fields),
        }
        for a in assumptions
    ]
    return propose_stream_semantics_impl(context, items, confirm)
