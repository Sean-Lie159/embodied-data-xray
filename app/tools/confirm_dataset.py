"""数据集画像确认：数据集类型 / 预期缺失 / 能力标签。

设计依据：``docs/指标单一来源与数据集画像设计.md`` 第二部分（B-3.2）。

## 要解决的问题（真实事故，2026-09-21）

用户在手套数据上确认了"这是双手位姿追踪数据""左右前臂本就不采集"，
但报告仍显示 ``推测类型: unknown``，质检仍把前臂整列判为不可通过。
用户质疑："很多是我们已经知道、做过语义确认的，**为什么还是 unknown**？"

根因：
1. 数据集类型（``guessed_type``）由 ``_sniffing`` 的硬编码 if-else 判定，
   **算一次即固定**，且**没有任何确认通道**——对比之下，**流的**语义标签
   有 ``label_source`` 支持用户覆盖，说明这套机制只做在流层面；
2. "左右前臂不采集"属**预期缺失**，但**无处记录**，导致质检每轮都报
   ``fail``，用户每轮都要口头解释一遍。

## 本模块

提供工具 :func:`confirm_dataset_profile`：把用户的确认写入既有画像文件
（``outputs/by_dataset/<净名>/profile.json``，复用 ``profile_store`` 的
原子写与跨会话锁——**不新造一套存储**）。写入后：

- ``load_dataset`` 读画像覆盖 ``guessed_type``（报告不再显示 unknown）；
- ``check_dataset_quality`` 的缺失值检查排除**已确认的预期缺失**，
  并如实标注"属预期缺失"（**不隐藏，只区分**）。

本模块不 import streamlit。
"""

from __future__ import annotations

from typing import Any

from agents import RunContextWrapper
from agents.decorators import tool
from pydantic import BaseModel, Field

from app.agent.context import RunContext
from app.tools import profile_store

# 是否把"数据集类型"列入受控词表？——不设。
# 理由：数据集类型是**领域知识**（如"手套位姿追踪数据"），强制词表会逼用户
# 往不合适的类别里塞。此处只要求是简短文本，并保留 auto_detected 供对照。


class ConfirmDatasetProfile(BaseModel):
    """一次数据集画像确认的输入。

    注意：``openai-agents`` 的工具参数走**严格 JSON Schema**，不接受
    ``dict[str, ...]``（会生成 ``additionalProperties``，被 SDK 拒绝）。
    因此复杂映射一律用 **JSON 字符串**表达，由本模块解析。
    """

    dataset_type: str = Field(
        default="",
        description=(
            "用户确认的数据集类型（如「双手手套位姿追踪数据」）。"
            "留空表示本次不确认类型。**必须来自用户**，不得由你推测填写。"
        ),
    )
    type_note: str = Field(
        default="", description="类型的补充说明（如「左右前臂不采集」）")
    expected_missing_json: str = Field(
        default="",
        description=(
            "**已确认的预期缺失**，JSON 字符串，键为文件名、值为列名模式数组，"
            "支持通配符。例如："
            '`{"tips_trajectory.csv": ["*forearm_pos_*"]}`。'
            "质检会把这些列排除在缺失值判定外，并标注为「预期缺失」。"
            "**必须由用户确认**该缺失是预期内的（如设备本就没有该传感器）。"
        ),
    )
    expected_missing_note: str = Field(
        default="", description="预期缺失的原因说明（如「手套无前臂传感器」）")
    capabilities_override_json: str = Field(
        default="",
        description=(
            "能力标签的人工纠正，JSON 字符串。例如 "
            '`{"has_force": false}`'
        ),
    )
    auto_detected_type: str = Field(
        default="",
        description="自动识别的类型（取自 load_dataset 的 guessed_type），仅供对照",
    )


def _parse_json_object(raw: str, *, field_name: str) -> tuple[dict[str, Any], str]:
    """解析 JSON 对象字符串；失败返回 (空 dict, 错误说明)。

    Args:
        raw: JSON 字符串（可为空）。
        field_name: 字段名（用于错误提示）。

    Returns:
        (解析结果, 错误信息)；解析成功时错误信息为空串。
    """
    import json

    s = (raw or "").strip()
    if not s:
        return {}, ""
    try:
        obj = json.loads(s)
    except ValueError as exc:
        return {}, f"{field_name} 不是合法 JSON：{exc}"
    if not isinstance(obj, dict):
        return {}, f"{field_name} 必须是 JSON 对象（而非数组或标量）"
    return obj, ""


def confirm_dataset_profile_impl(
    context: RunContext,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """写入用户确认的数据集画像（类型 / 预期缺失 / 能力覆盖）。

    Args:
        context: 运行时上下文（提供 output_dir / dataset_id）。
        payload: 见 :class:`ConfirmDatasetProfile`。

    Returns:
        dict，含 success、saved（是否落盘）、confirmed（本次确认的内容摘要）、
        dataset_type（当前生效值）、user_message。
    """
    output_dir = context.output_dir or "outputs"
    dataset_id = context.dataset_id

    if not dataset_id:
        return {
            "success": False,
            "error": "no_dataset",
            "reason": "尚未加载数据集，无法写入画像（画像按数据集隔离）",
            "user_message": "请先加载数据集，再确认其画像。",
        }

    dataset_type = str(payload.get("dataset_type") or "").strip()
    type_note = str(payload.get("type_note") or "").strip()
    em_note = str(payload.get("expected_missing_note") or "").strip()
    auto_type = str(payload.get("auto_detected_type") or "").strip() or None

    # 复杂映射以 JSON 字符串传入（SDK 严格 schema 不接受 dict 参数）。
    expected_missing, em_err = _parse_json_object(
        str(payload.get("expected_missing_json") or ""),
        field_name="expected_missing_json",
    )
    caps_override, caps_err = _parse_json_object(
        str(payload.get("capabilities_override_json") or ""),
        field_name="capabilities_override_json",
    )
    if em_err or caps_err:
        return {
            "success": False,
            "error": "invalid_json",
            "reason": "；".join(e for e in (em_err, caps_err) if e),
            "user_message": (
                "参数格式有误：" + "；".join(e for e in (em_err, caps_err) if e)
                + "。请提供合法的 JSON 对象。"
            ),
        }

    # 规范化 expected_missing：确保值是字符串列表（防止误传字符串）。
    clean_em: dict[str, list[str]] = {}
    for fname, pats in (expected_missing or {}).items():
        if isinstance(pats, str):
            pats = [pats]
        clean_em[str(fname)] = [str(p) for p in pats if str(p).strip()]

    if not (dataset_type or clean_em or caps_override):
        return {
            "success": False,
            "error": "nothing_to_confirm",
            "reason": "未提供任何可确认内容（类型 / 预期缺失 / 能力覆盖均为空）",
            "user_message": (
                "没有需要确认的内容。如果你要固定数据集类型，请提供类型名；"
                "如果要标注预期缺失，请给出文件名与列名模式。"
            ),
        }

    # 自动带上当前嗅探出的类型，便于日后对照"确认改了什么"。
    if auto_type is None:
        auto_type = context.meta.get("guessed_type")

    # 写入画像（复用 profile_store：原子写 + 跨会话锁 + 读-改-写合并）。
    profile_store.save_dataset_profile(
        output_dir,
        dataset_id,
        dataset_type=dataset_type or None,
        auto_detected_type=auto_type,
        dataset_type_note=type_note or None,
        expected_missing=clean_em or None,
        expected_missing_note=em_note or None,
        capabilities_override=caps_override or None,
    )

    # 即时生效：把确认结果写回当前会话的上下文（无需重新加载）。
    confirmed_parts: list[str] = []
    if dataset_type:
        context.meta["guessed_type"] = dataset_type
        context.meta["guessed_type_source"] = profile_store.SOURCE_USER
        if auto_type and auto_type != dataset_type:
            context.meta["guessed_type_auto_detected"] = auto_type
        if type_note:
            # 说明也要即时生效（报告会展示它），否则要重载才可见。
            context.meta["guessed_type_note"] = type_note
        confirmed_parts.append(f"数据集类型「{dataset_type}」")
    if clean_em:
        context.meta["expected_missing"] = {
            "patterns": clean_em,
            "source": profile_store.SOURCE_USER,
            "note": em_note,
        }
        n_cols = sum(len(v) for v in clean_em.values())
        confirmed_parts.append(
            f"{len(clean_em)} 个文件的预期缺失（{n_cols} 条列模式）")
    if caps_override:
        caps = context.meta.setdefault("capabilities", {})
        caps.update(caps_override)
        confirmed_parts.append(f"{len(caps_override)} 项能力标签")

    if not confirmed_parts:
        return {
            "success": False,
            "error": "nothing_to_confirm",
            "reason": "提供的确认内容均为空",
            "user_message": "没有需要确认的内容，未写入画像。",
        }

    em_hint = ""
    if clean_em:
        em_hint = (
            "  质检的缺失值检查会把这些列排除在判定外，并标注为「预期缺失」"
            "（**不隐藏**，只是与「意外缺失」区分开）。"
        )

    return {
        "success": True,
        "dataset": dataset_id,
        "saved": True,
        "confirmed": {
            "dataset_type": dataset_type or None,
            "auto_detected_type": auto_type,
            "expected_missing": clean_em or None,
            "capabilities_override": caps_override or None,
        },
        "dataset_type": context.meta.get("guessed_type"),
        "user_message": (
            "已写入数据集画像并**跨会话生效**："
            + "、".join(confirmed_parts)
            + "。之后对此数据集的报告与质检都会采用这些确认。"
            + em_hint
        ),
    }


@tool
def confirm_dataset_profile(
    wrapper: RunContextWrapper[RunContext],
    payload: ConfirmDatasetProfile,
) -> dict:
    """确认并固化数据集画像（类型 / 预期缺失 / 能力标签），跨会话生效。

    **必须由用户明确提供，不得由你推测填写**。典型用途：

    - 报告显示「推测类型 unknown」而用户知道它是什么 →
      传 ``dataset_type`` 固定下来（如「双手手套位姿追踪数据」）；
    - 用户说明某组列**本来就没有数据**（如"手套不含前臂传感器"）→
      传 ``expected_missing``（如 ``{"tips_trajectory.csv": ["*forearm_*"]}``），
      质检此后会把它们标为「预期缺失」而非「错误」。

    Args:
        payload: 见 ConfirmDatasetProfile 的字段说明。

    Returns:
        dict，含 saved、confirmed（本次确认摘要）、dataset_type（生效值）、
        user_message。
    """
    return confirm_dataset_profile_impl(wrapper.context, payload.model_dump())
