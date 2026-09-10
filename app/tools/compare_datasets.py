"""数据集对比：两个同构录制并排比较（行数 / 采样率 / 时间跨度 / 缺口）。

**为什么需要**：同一任务的多次录制（或两个批次）常需对比——"这次录制比上次
少了多少帧""哪路流两批不一致"。项目此前只能逐个加载再手工对照，且加载是
覆盖式的（新加载替换旧数据集），历史数字只能靠对话记忆（项目曾因此出过
"拿旧数字当当前数据"的事故）。

本工具**只读、不切换当前数据集**：给定两个路径，分别做轻量普查（流清单 +
逐流时间戳统计），产出并排对比表与差异摘要。

与既有机制的关系：
- 数据加载是覆盖式的（RunContext 单数据集语义），故对比必须独立于它；
- 读取复用统一读取注册表（read_stream）——目录/单文件/容器子流同一套；
- 不做深度统计（那是加载后各工具的职责），只做**结构级对比**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.tools._readers import ReadRequest, read_stream

# 对比时每侧最多枚举的流数（防大目录上下文爆炸）。
_MAX_STREAMS_PER_SIDE = 40


def _profile_side(path: str, max_streams: int = _MAX_STREAMS_PER_SIDE) -> dict[str, Any]:
    """轻量剖析一侧（目录或单文件），返回流清单与逐流时间统计。"""
    src = Path(path)
    if not src.exists():
        return {"ok": False, "error": "path_not_found", "path": path}

    if src.is_dir():
        from app.tools._sniffing import probe_directory, probe_full_paths

        probe = probe_directory(src)
        candidates = (
            probe_full_paths(probe, "tables")
            + probe_full_paths(probe, "videos")
            + probe_full_paths(probe, "others")
        )
        streams: list[dict[str, Any]] = []
        for p_str in candidates[:max_streams]:
            p = Path(p_str)
            if p.suffix.lower() in (".h5", ".hdf5", ".mcap"):
                # 容器：枚举子流（复用统一注册表的 candidates）。
                r = read_stream(ReadRequest(path_spec=str(p), want="candidates"))
                for sub in (r.candidates or [])[:max_streams]:
                    streams.append({"path": f"{p}::{sub}", "format": p.suffix.lstrip(".")})
            else:
                streams.append({"path": str(p), "format": p.suffix.lstrip(".").lower()})
        return {"ok": True, "path": str(src), "kind": "dir",
                "n_total_files": probe.get("total_files"), "streams": streams[:max_streams]}

    # 单文件
    r = read_stream(ReadRequest(path_spec=str(src), want="candidates"))
    if r.candidates:
        return {"ok": True, "path": str(src), "kind": "container",
                "streams": [{"path": f"{src}::{s}", "format": r.fmt}
                            for s in r.candidates[:max_streams]]}
    fmt = src.suffix.lstrip(".").lower()
    return {"ok": True, "path": str(src), "kind": "file",
            "streams": [{"path": str(src), "format": fmt}]}


def _timing_of(path_spec: str) -> dict[str, Any] | None:
    """取单流的时间戳统计（样本数 / 时长 / 采样率）。"""
    result = read_stream(ReadRequest(path_spec=path_spec, want="timestamp"))
    if not result.ok or result.timestamp is None or len(result.timestamp) < 2:
        return None
    import numpy as np

    from app.tools.timestamp_units import TIME_UNITS, infer_unit, to_ns

    ts = np.sort(np.asarray(result.timestamp, dtype=float))
    unit = infer_unit(ts, result.timestamp_column or "")["unit"]
    ns = to_ns(ts, unit) if unit in TIME_UNITS else ts
    span = float(ns[-1] - ns[0]) / 1e9
    return {
        "n": int(len(ts)),
        "span_s": round(span, 3),
        "rate_hz": round((len(ts) - 1) / span, 3) if span > 0 else None,
    }


def compare_datasets_impl(
    context: RunContext,
    path_a: str,
    path_b: str,
    max_streams: int = _MAX_STREAMS_PER_SIDE,
) -> dict[str, Any]:
    """并排对比两个数据集（或录制）的流清单与时间统计。

    Args:
        context: 运行时上下文（**不切换当前数据集**，仅作输出目录用）。
        path_a: 数据集 A 路径（目录或单文件）。
        path_b: 数据集 B 路径。
        max_streams: 每侧最多对比的流数。

    Returns:
        dict，含 a/b（各自流清单与统计）、matched（按流名匹配的对比行）、
        only_in_a / only_in_b（单侧独有的流）、summary。
    """
    a = _profile_side(path_a, max_streams)
    b = _profile_side(path_b, max_streams)
    if not a.get("ok"):
        return {"success": False, "error": "path_a_not_found",
                "user_message": f"数据集 A 路径不存在：{path_a}"}
    if not b.get("ok"):
        return {"success": False, "error": "path_b_not_found",
                "user_message": f"数据集 B 路径不存在：{path_b}"}

    def _index(side: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for s in side["streams"]:
            out[Path(s["path"].partition("::")[0]).name
                + ("::" + s["path"].partition("::")[2] if "::" in s["path"] else "")] = s
        return out

    ia, ib = _index(a), _index(b)
    matched: list[dict[str, Any]] = []
    for name in sorted(set(ia) & set(ib)):
        ta = _timing_of(ia[name]["path"])
        tb = _timing_of(ib[name]["path"])
        row: dict[str, Any] = {"stream": name}
        if ta and tb:
            row.update({
                "a_n": ta["n"], "b_n": tb["n"],
                "a_span_s": ta["span_s"], "b_span_s": tb["span_s"],
                "a_rate_hz": ta["rate_hz"], "b_rate_hz": tb["rate_hz"],
                "same_n": ta["n"] == tb["n"],
                "same_span": abs(ta["span_s"] - tb["span_s"]) < 0.5,
            })
        else:
            row["note"] = "一侧无可用时间戳，无法对比时间统计"
        matched.append(row)

    only_a = sorted(set(ia) - set(ib))
    only_b = sorted(set(ib) - set(ia))
    diff_n = [m["stream"] for m in matched if m.get("same_n") is False]
    diff_span = [m["stream"] for m in matched if m.get("same_span") is False]

    return {
        "success": True,
        "a": {"path": a["path"], "kind": a["kind"], "n_streams": len(ia)},
        "b": {"path": b["path"], "kind": b["kind"], "n_streams": len(ib)},
        "matched": matched,
        "only_in_a": only_a[:20],
        "only_in_b": only_b[:20],
        "summary": {
            "n_matched": len(matched),
            "n_only_a": len(only_a),
            "n_only_b": len(only_b),
            "n_diff_rows": len(diff_n),
            "n_diff_span": len(diff_span),
        },
        "user_message": (
            f"对比完成：A（{a['kind']}，{len(ia)} 条流）vs B（{b['kind']}，"
            f"{len(ib)} 条流）；同名流 {len(matched)} 条，其中行数不同 "
            f"{len(diff_n)} 条、时长差 >0.5s {len(diff_span)} 条；"
            f"A 独有 {len(only_a)} 条、B 独有 {len(only_b)} 条。"
        ),
    }


@tool
def compare_datasets(
    wrapper: RunContextWrapper[RunContext],
    path_a: str,
    path_b: str,
    max_streams: int = _MAX_STREAMS_PER_SIDE,
) -> dict:
    """并排对比两个数据集（或同任务的两次录制）的结构与时间统计。

    适用：同一任务多次录制 / 两个批次的差异排查——"这次比上次少了多少帧"
    "哪路流两批不一致"。**只读对比，不切换当前已加载的数据集**。

    Args:
        path_a: 数据集 A 路径（目录或单文件）。
        path_b: 数据集 B 路径。
        max_streams: 每侧最多对比的流数（默认 40，超出截断）。

    Returns:
        dict，含 a / b（各自流数与类型）、matched（同名流的行数/时长/采样率
        并排）、only_in_a / only_in_b（单侧独有的流）、summary、user_message。
    """
    return compare_datasets_impl(wrapper.context, path_a, path_b, max_streams)
