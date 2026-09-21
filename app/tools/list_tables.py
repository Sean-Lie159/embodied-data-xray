"""表清单工具（多表并列机制阶段 1）。

**为什么需要**（`docs/多表并列机制设计.md` §2.3、§4.1）：当前数据集的全部可分析
表早就登记在 ``meta["streams"]`` 里、``resolve_table_name`` 也早就支持按名读取，
但**模型没有任何确定性的入口知道自己有哪些表可选**：

1. 带 ``table`` 参数的 5 个工具，docstring 一律只写「缺省=主表」，把选择责任推给
   LLM 的隐含推断；
2. ``resolve_table_name`` 匹配失败时只返回 **3 个示例**，模型无从判断「是真的没有
   这张表，还是我写法不对」；
3. SYSTEM_PROMPT 里没有任何一条关于「当前表 / 如何换表」的纪律。

于是失败模式不是「存不下多张表」，而是「不知道有哪些表、不知道当前是哪张」。
本工具补的正是这个缺口——**只读**：不做任何 IO、不改任何状态、不测采样率。

业界参照：LangChain 的 ``sql_db_list_tables`` 只返回**裸表名**，模型靠名字猜错表、
拿到 schema 才发现必须重来（``langchain-community`` Issue #294，2025-02-07）。
故本工具在枚举阶段就给出**规模 + 语义标签 + 是否为当前缺省表**，而非只给名字。
"""

from __future__ import annotations

from typing import Any

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools._data_access import (
    main_table_candidates,
    resolve_default_table_name,
)

# 单次返回的表条目上限（体积护栏）。
#
# 为什么需要：真实数据集 ``aligned_joints.h5`` 的「每帧一组」布局在早期实现里
# 登记出 **84,810 条流**（见 docs/行为测试.md:623-630），虽已由
# ``_merge_frame_layout`` 合并到数十条，但 MCAP 多 topic、超大规模目录仍可能
# 让表数上百。本上限保证返回体积可控，超限时**按规模降序截断并如实声明**。
_MAX_TABLES = 80

# 剩余表的规模范围只在能取到至少一个有效行数时才给出。
_MAX_SEMANTIC_LABEL = 60


def _is_table_stream(stream: dict[str, Any]) -> bool:
    """判断流登记项是否计入「可分析的表」。

    口径与 ``inspect_streams`` 的 ``n_table_streams`` 摘要**必须一致**
    （``inspect_streams.py:767``：``kind != "video"``）。为什么复用同一判据而不是
    另立标准：两处口径不一致会让模型看到「inspect_streams 说有 12 条表流、
    list_tables 只列 9 张」，进而怀疑工具不可靠。视频**不是表**（不能传给
    ``table`` 参数做统计），故排除。

    Args:
        stream: 流登记项。

    Returns:
        True 表示可作为 table 参数传给分析工具。
    """
    return stream.get("kind") != "video"


def list_tables_impl(context: RunContext) -> dict[str, Any]:
    """列出当前数据集全部可分析的表（含规模、语义标签与当前缺省表标记）。

    Args:
        context: 运行时上下文。

    Returns:
        dict，含 success、dataset、default_table（当前缺省表名）、n_tables、
        tables（按规模降序的条目列表，每条含 table_name / rows / cols / kind /
        semantic_label / label_confidence / status / is_default）；表数超过
        ``_MAX_TABLES`` 时截断并置 truncated=True + truncation_note。
        未加载数据集时返回 no_data_loaded。
    """
    if context.dataset_id is None and context.df is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "reason": "尚未加载任何数据集",
            "user_message": (
                "尚未加载任何数据集。请先调用 load_dataset 加载数据，再查看表清单。"
            ),
        }

    # 命名口径**必须复用唯一生成点**（与 resolve_table_name 的接受集一致）。
    #
    # 为什么不能另写一套：2026-09-20 的真实事故正是「清单与容错自相矛盾」——
    # 错误提示说「请用流登记表的表名」，而 inspect_streams 当时只有 source
    # （完整绝对路径），照抄必失败，agent 因此误判「工具不支持读取该节点」。
    # 复用 _usable_table_name 保证本工具给出的表名**照抄即可用**。
    from app.tools.inspect_streams import _usable_table_name

    streams = context.meta.get("streams", []) or []

    # 缺省表名与规模：**统一经 _data_access 的共享实现**，不自带一套。
    #
    # 为什么必须共享（2026-09-21 实测踩中）：`meta["main_table"]["name"]` **不是
    # 可靠的表名**，且 candidates 的位置随加载路径而变（目录型在
    # `selection.candidates` 用 name/nrows/ncols；单文件型在顶层 `candidates`
    # 用 table_name/rows/cols）。若本工具另写一套解析，一旦与
    # `resolve_table_name` 的口径分叉，就会出现"清单给的 default_table 与
    # 不传 table 时实际分析的表不是同一张"的静默错位——比报错更危险。
    # 故统一调用 `resolve_default_table_name` / `main_table_candidates`。
    default_table = resolve_default_table_name(context)
    default_lower = default_table.strip().lower() if default_table else None

    # 规模来源：流登记项的 n_rows/n_cols **只在容器子流（h5 节点/mcap topic）上存在**，
    # 独立文件流并不写入这两个键（实测：目录型数据集的 csv 流 n_rows 恒为 None）。
    # 规模权威来自 candidates——load_dataset 用它排序选主表
    # （load_dataset.py:2113-2135 / :2616-2619），每一侧都给出了规模。
    #
    # 为什么必须从这里取：否则模型看到全部 rows=None，无法判断"哪张是明细表、
    # 哪张是 2 行的清单"，本工具"给出规模"的核心价值即归零。
    size_by_name: dict[str, tuple[Any, Any]] = {}
    for c in main_table_candidates(context):
        # 目录路径用 name（=文件名），单文件路径用 table_name（=stem::node）。
        key = c.get("table_name") or c.get("name")
        if key:
            size_by_name[str(key)] = (
                c.get("rows", c.get("nrows")), c.get("cols", c.get("ncols"))
            )

    entries: list[dict[str, Any]] = []
    for s in streams:
        if not _is_table_stream(s):
            continue
        table_name = _usable_table_name(s)
        if not table_name:
            continue
        # 规模：优先流登记项（容器子流有值），回退 candidates（独立文件流）。
        rows = s.get("n_rows")
        cols = s.get("n_cols")
        if rows is None or cols is None:
            cand = size_by_name.get(table_name)
            if cand:
                rows = rows if rows is not None else cand[0]
                cols = cols if cols is not None else cand[1]
        label = s.get("semantic_label")
        if isinstance(label, str) and len(label) > _MAX_SEMANTIC_LABEL:
            label = label[:_MAX_SEMANTIC_LABEL] + "…"
        entry: dict[str, Any] = {
            "table_name": table_name,
            "rows": rows,
            "cols": cols,
            "kind": s.get("kind"),
            "semantic_label": label,
            "label_confidence": s.get("label_confidence"),
            "status": s.get("status", "active"),
            # **消除歧义的核心字段**：模型据此知道「缺省分析的是哪张」。
            "is_default": bool(
                default_lower and table_name.strip().lower() == default_lower
            ),
        }
        if s.get("frame_layout"):
            # 帧布局流：透出帧数，让模型理解"行数是帧数 × 每帧行数"。
            entry["n_frames"] = s.get("n_frames")
        if s.get("status") == "empty":
            # 空流（行数 ≤ EMPTY_STREAM_MAX_ROWS，见 _sniffing.py:59）：不能作
            # 统计对象，显式标注以免模型对它白试一次。**仍列出**——用户需要知道
            # 这个文件存在（如 2 行的 tasks.csv 是合法但不适合统计的表）。
            entry["usable"] = False
        entries.append(entry)

    # 排序：规模降序（与主表选择判据同口径），并列时表名字母序（确定性）。
    def _size_key(e: dict[str, Any]) -> tuple[int, str]:
        rows = e.get("rows") or 0
        cols = e.get("cols") or 1
        try:
            size = int(rows) * max(1, int(cols))
        except (TypeError, ValueError):
            size = 0
        return (-size, str(e.get("table_name") or ""))

    entries.sort(key=_size_key)

    n_tables = len(entries)
    truncated = False
    truncation_note: str | None = None
    if n_tables > _MAX_TABLES:
        dropped = entries[_MAX_TABLES:]
        entries = entries[:_MAX_TABLES]
        truncated = True
        # 如实声明被省略部分的规模范围——否则模型会误以为数据集只有这些表。
        sizes = [e.get("rows") for e in dropped if isinstance(e.get("rows"), int)]
        size_hint = (
            f"被省略的表规模在 {min(sizes)} – {max(sizes)} 行之间；"
            if sizes else ""
        )
        truncation_note = (
            f"表数超过单次展示上限（{_MAX_TABLES} 条），已按规模降序截断，"
            f"省略 {len(dropped)} 条。{size_hint}如需查看完整清单，"
            "请缩小数据集范围或直接指定表名。"
        )

    # user_message：如实给出总量、缺省表与关键提示。
    msg_parts = [
        f"当前数据集 {context.dataset_id} 共有 {n_tables} 张可分析的表"
        f"（此处展示 {len(entries)} 张）。"
    ]
    if default_table:
        msg_parts.append(
            f"当前缺省表（不传 table 参数时分析的对象）是 {default_table}。"
        )
    else:
        # 纯媒体数据集「无有效主表」是合法状态，不得暗示"缺失"。
        msg_parts.append("当前数据集没有缺省表（不含表格数据）。")
    msg_parts.append(
        "分析其他表请在对应工具的 table 参数中传上表的 table_name"
        "（形如「<文件stem>::<节点>」或文件名，不含目录与扩展名）。"
    )
    if truncation_note:
        msg_parts.append(truncation_note)

    return {
        "success": True,
        "dataset": context.dataset_id,
        "default_table": default_table,
        "n_tables": n_tables,
        "tables_shown": len(entries),
        "tables": entries,
        "truncated": truncated,
        "truncation_note": truncation_note,
        "user_message": " ".join(msg_parts),
    }


@tool
def list_tables(wrapper: RunContextWrapper[RunContext]) -> dict:
    """列出当前数据集全部可分析的表（表名、规模、语义标签、当前缺省表）。

    适用：需要知道「这个数据集有哪些表可选」「当前缺省分析的是哪张表」
    「用户说的某张表叫什么名字」时。**先查清单再指定表名**，不要凭文件名猜测。

    与 inspect_streams 的分工：inspect_streams 给出完整设备清单（含视频/IMU/
    标定/时钟来源等能力画像，需要实测采样率）；本工具只给**可传给 table 参数的
    表清单**，零 IO、零副作用，用于「要分析哪张表」这一决策。

    Args:
        无（读取已加载数据集的流登记表）。

    Returns:
        dict，含 success、dataset、default_table（当前缺省表名）、n_tables
        （可分析表总数）、tables（按规模降序的条目：table_name / rows / cols /
        kind / semantic_label / label_confidence / is_default）、truncated 与
        truncation_note（表数超过展示上限时为 True 与说明），以及 user_message。
        未加载数据集时返回 no_data_loaded。
    """
    return list_tables_impl(wrapper.context)
