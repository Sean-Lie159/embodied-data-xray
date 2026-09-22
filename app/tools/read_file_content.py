"""文件内容查看（只读）：文档 / 配置 / 标定 / 元数据通用。

设计依据：``docs/文件内容查看能力设计.md``。

## 要解决的问题（真实场景，2026-09-22）

分析 ``origin-data-fullmodal-sample-dataset`` 时，用户问"这个数据集里有没有
数采设备信息（相机/手环型号）"。agent 回答"够不到，建议重新加载单文件"——
而这既不必要（会替换当前数据集），且其中一条路实际走不通（``.md`` 是
``unsupported_format``）。

**根因**：``probe_directory`` 把文件分六组，而 ``build_streams_registry``
只登记 tables / videos / audios / images：

- ``others`` 组（``.md``/``.yaml`` 等文档）**被完全丢弃**——普查看得到文件名，
  却没有任何读取路径；
- ``cals`` 组（标定/元数据 JSON）**刻意不进流登记表**——这本身正确
  （防止标定 JSON 被当数据表参与对齐），但副作用是**内容也不可读**，
  只剩指纹探测给出的键名。

**核心区分**：**"不参与数据分析" ≠ "内容不可查看"**。
标定文件确实不该进表清单，但用户完全有理由查看它的内容（相机内参、
投影矩阵）。本模块补上这条**只读查看通道**。

## 三条纪律（硬性）

1. **只读**：绝不写入、不修改源文件；
2. **不做内容理解**：只返回**原文**（或键值摘要），不解析、不总结、不抽字段
   ——含义由模型阅读后判断。工具替模型"理解"必然引入幻觉；
3. **不进主表语义**：读取结果不写入 ``context.df``、不进 ``findings``、
   不参与对齐（不污染确定性分析）。

本模块不 import streamlit。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext
from app.config import get_settings

# 可读的纯文本扩展名（按内容读取的类型）。
_TEXT_EXTS = frozenset({
    ".md", ".markdown", ".txt", ".text", ".rst", ".yaml", ".yml",
    ".log", ".info", ".ini", ".cfg", ".conf", ".toml", ".json5",
})
# JSON 类（走"配置模式"：键值摘要）。
_JSON_EXTS = frozenset({".json"})
# 表格类（走"引导模式"：指向数据工具，不在本工具重复实现）。
_TABLE_EXTS = frozenset({".csv", ".parquet", ".h5", ".hdf5", ".mcap"})


def _dataset_root(context: RunContext) -> Path | None:
    """取当前数据集的根目录（用于把相对路径解析为绝对，并限制越权访问）。

    注意：**不能只依赖 ``file_survey``**——目录很大时它会被
    ``_compress_file_survey`` 压缩成"只有计数、没有路径"。因此以
    ``meta["source"]`` 为准（目录型数据集在加载时写入）。
    """
    src = str(context.meta.get("source", "") or "")
    if not src:
        # 单文件数据集：source 是文件路径，取其父目录。
        return None
    p = Path(src)
    return p if p.is_dir() else p.parent


def _is_within(path: Path, root: Path) -> bool:
    """判断 path 是否位于 root 之内（防越权读取数据集目录外的文件）。"""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _candidate_names(context: RunContext) -> list[str]:
    """收集可读文件的候选名（供"按文件名定位"与错误提示）。

    来源（合并去重）：
    1. 流登记表（``meta["streams"]`` 的 path）；
    2. ``file_survey`` 各组路径（可能被压缩，故不依赖）；
    3. **磁盘扫描**（以数据集根为界，递归）——这是唯一**必然可用**的来源，
       覆盖 ``others`` 组（文档类）与 ``cals`` 组（标定类）。
    """
    names: set[str] = set()
    for s in context.meta.get("streams", []):
        p = str(s.get("path", "") or "")
        if p:
            names.add(p)

    survey = context.meta.get("file_survey") or {}
    for key in ("tables", "cals", "others", "videos", "audios", "images"):
        view = survey.get(key)
        if isinstance(view, dict):
            for p in view.get("paths") or []:
                names.add(str(p))

    root = _dataset_root(context)
    if root is not None and root.is_dir():
        try:
            for p in root.rglob("*"):
                if p.is_file():
                    names.add(str(p))
        except OSError:
            pass
    return sorted(names)


def _resolve_path(context: RunContext, raw: str) -> tuple[Path | None, dict]:
    """把用户/模型给的 path 解析为**数据集内的**真实文件。

    解析顺序（**先限定在数据集内，再看绝对路径**）：

    1. **数据集根为基准**的解析（最有价值，也是"只给文件名"的主路径）：
       ``root/name`` 存在 → 用它（并限界在 root 内）；
    2. 与候选清单（磁盘扫描 + 流登记表）按**完整路径**匹配（大小写不敏感）；
    3. 与候选清单按**文件名**匹配（多命中时取第一个并提示）；
    4. **最后**才把入参当绝对/相对 CWD 的路径试（仅供显式绝对路径场景）。

    **为什么顺序如此（2026-09-22 实测缺陷）**：若先做 ``Path(s).is_file()``，
    则相对文件名会**相对 CWD** 解析——而 CWD 是**项目根**，那里恰有
    ``README.md``！于是"看数据集的 README"会**静默命中项目自己的 README**，
    再因不在数据集内而被拒（表现为"明明文件存在却报越权"）。
    故必须先以**数据集根**为基准解析。

    Returns:
        (路径 或 None, 说明 dict)。说明 dict 在失败时含 error / available。
    """
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        return None, {
            "error": "empty_path",
            "reason": "path 为空",
            "user_message": "请提供要查看的文件名或路径。",
        }

    root = _dataset_root(context)
    cands = _candidate_names(context)
    low = s.lower()
    note: dict[str, Any] = {}

    # **越权前置检查（安全优先）**：只对**绝对路径**做。
    #
    # 判据为什么只认"绝对路径"（2026-09-22 实测两轮修正）：
    # - 用 `Path(s).is_file()` 判：会相对 **CWD=项目根** 解析，而项目根恰有
    #   README.md → 正常的"看数据集 README"被误判为外部文件；
    # - 用"含 / 或 \\"判：会误伤**数据集内相对路径**（如
    #   ``lerobot/meta/info.json``，这是完全合法的用法）；
    # - 故只有**绝对路径**才当作"可能越权"，其余一律先按数据集根解析。
    #
    # 为什么必须前置：绝对外部路径若交给后面的"按文件名匹配"分支，其 basename
    # 可能与数据集内文件同名（如都有 README.md），于是**静默命中同名的内部
    # 文件**——既不报越权、结果来源也错乱。
    if Path(s).is_absolute():
        raw_path = Path(s)
        if root is None or not _is_within(raw_path, root):
            return None, {
                "error": "path_outside_dataset",
                "reason": f"{raw_path} 不在当前数据集目录内",
                "user_message": (
                    f"只能查看当前数据集目录内的文件（{root}）。"
                    "越权路径已被拒绝——请用数据集内的文件名（如 README.md）。"
                ),
            }
        if raw_path.is_file():
            return raw_path, {}

    if root is not None and root.is_dir():
        # 1) 以数据集根为基准（支持 "README.md" 与 "lerobot/meta/info.json"）。
        rel = root / s
        if rel.is_file():
            return rel, {}

    # 2) 完整路径匹配。
    for c in cands:
        if c.lower() == low:
            return Path(c), {}
    # 3) 文件名匹配（仅当入参**不含路径分隔符**时才做，避免路径串误命中）。
    if "/" not in s and "\\" not in s:
        by_name = [c for c in cands if Path(c).name.lower() == low]
        if by_name:
            hit = Path(by_name[0])
            if len(by_name) > 1:
                note["ambiguous_match"] = {
                    "n_matched": len(by_name),
                    "used": by_name[0],
                    "others": [Path(c).name for c in by_name[1:4]],
                    "note": "同名文件有多个，已取第一个；如需精确请用完整路径",
                }
            return hit, note
        # 3b) 子串匹配（宽松兜底，仅唯一命中时接受；同样限"纯文件名"入参）。
        partial = [c for c in cands if low in c.lower()]
        if len(partial) == 1:
            return Path(partial[0]), {
                "matched_by": "substring",
                "used": partial[0],
            }

    # 失败：给出可用文件清单（按扩展名分组，便于模型自我纠正）。
    readable = [
        c for c in cands
        if Path(c).suffix.lower() in (_TEXT_EXTS | _JSON_EXTS)
    ]
    return None, {
        "error": "file_not_found",
        "reason": f"未找到文件 {raw!r}",
        "available_documents": [Path(c).name for c in readable[:30]],
        "n_available_documents": len(readable),
        "user_message": (
            f"未找到文件 {raw!r}。"
            f"当前数据集内可查看的文本/配置文件共 {len(readable)} 个"
            + (
                f"，例如：{'、'.join(Path(c).name for c in readable[:8])}"
                if readable else ""
            )
            + "。"
        ),
    }


def _read_text(
    path: Path, *, max_chars: int, start_line: int | None, keyword: str | None,
) -> dict[str, Any]:
    """文本模式：读原文（支持起始行与关键词过滤）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "success": False, "error": "read_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "user_message": f"读取 {path.name} 失败：{exc}",
        }

    lines = text.splitlines()
    n_total = len(lines)
    basis = "全文"

    if keyword:
        kw = keyword.strip().lower()
        # 命中行 + 前后各 1 行（单行常缺上下文，见设计 §7 决策点 4）。
        keep: list[int] = []
        for i, ln in enumerate(lines):
            if kw in ln.lower():
                for j in (i - 1, i, i + 1):
                    if 0 <= j < n_total and j not in keep:
                        keep.append(j)
        keep.sort()
        picked = [(i + 1, lines[i]) for i in keep]
        basis = f"关键词 {keyword!r} 命中 {len(keep)} 行（含前后各 1 行）"
        body = "\n".join(f"{no:>5}| {ln}" for no, ln in picked)
        if not picked:
            return {
                "success": True, "path": str(path), "kind": "document",
                "content": "", "n_lines_total": n_total,
                "n_lines_returned": 0, "truncated": False,
                "content_basis": basis,
                "user_message": (
                    f"{path.name} 中未找到包含 {keyword!r} 的行"
                    f"（已搜索全文 {n_total} 行）。"
                ),
            }
    else:
        start = max(0, (start_line or 1) - 1)
        picked = [(i + 1, lines[i]) for i in range(start, n_total)]
        body = "\n".join(f"{no:>5}| {ln}" for no, ln in picked)
        if start:
            basis = f"第 {start + 1}-{n_total} 行"

    truncated = len(body) > max_chars
    if truncated:
        body = body[:max_chars]
    n_returned = body.count("\n") + 1 if body else 0
    note = ""
    if truncated:
        note = (
            f"（内容被截断至 {max_chars} 字符；全文共 {n_total} 行。"
            "可用 start_line 从后续行继续读取，或用 keyword 定位。）"
        )
    return {
        "success": True,
        "path": str(path),
        "kind": "document",
        "content": body,
        "n_lines_total": n_total,
        "n_lines_returned": n_returned,
        "truncated": truncated,
        "content_basis": basis + ("（已截断）" if truncated else ""),
        "note": note,
        # **不做内容理解**：明确声明原文交付，含义由模型判断。
        "content_not_interpreted": True,
        "user_message": (
            f"已读取 {path.name}（{basis}，共 {n_total} 行）。"
            + (f" {note}" if note else "")
        ),
    }


def _read_config(path: Path, *, max_chars: int) -> dict[str, Any]:
    """配置模式：JSON 返回**键值摘要**（复用既有 _compact_config）。"""
    from app.tools.load_dataset import _compact_config, _read_json_object

    obj = _read_json_object(path)
    if obj is None:
        # 不是单对象（如行列表或非法 JSON）→ 回退文本模式（原文可能有价值）。
        return _read_text(
            path, max_chars=max_chars, start_line=None, keyword=None)

    compact = _compact_config(obj)
    body = json.dumps(compact, ensure_ascii=False, indent=2)
    truncated = len(body) > max_chars
    if truncated:
        body = body[:max_chars]
    return {
        "success": True,
        "path": str(path),
        "kind": "config",
        "content": body,
        "n_keys": len(obj),
        "keys": [str(k) for k in list(obj.keys())[:60]],
        "truncated": truncated,
        "content_basis": f"配置键值摘要（{len(obj)} 个顶层键）",
        "note": (
            "（配置值已压缩：标量原样、长值截断、嵌套结构只报规模。"
            "如需某个键的完整内容请告知。）" if not truncated else
            f"（摘要被截断至 {max_chars} 字符。）"
        ),
        "content_not_interpreted": True,
        "user_message": (
            f"已读取配置文件 {path.name} 的键值摘要（{len(obj)} 个顶层键）。"
        ),
    }


def read_file_content_impl(
    context: RunContext,
    path: str,
    *,
    max_chars: int | None = None,
    start_line: int | None = None,
    keyword: str | None = None,
) -> dict[str, Any]:
    """查看数据集内任意文本文件的内容（只读）。

    Args:
        context: 运行时上下文。
        path: 文件路径或文件名（大小写不敏感）。
        max_chars: 返回字符上限；缺省取配置。
        start_line: 起始行（1-based）。
        keyword: 只返回含该关键词的行（含前后各 1 行，带行号）。

    Returns:
        见模块 docstring 与设计文档 §3.1。
    """
    settings = get_settings()
    limit = int(max_chars or getattr(
        settings, "read_content_max_chars", 4000))

    if context.dataset_id is None and not context.meta:
        return {
            "success": False,
            "error": "no_data_loaded",
            "user_message": "尚未加载任何数据集。请先调用 load_dataset。",
        }

    resolved, note = _resolve_path(context, path)
    if resolved is None:
        return {"success": False, **note}

    ext = resolved.suffix.lower()

    # 表格类：**不在本工具重复实现**（职责分离，见设计 §3.2）。
    if ext in _TABLE_EXTS:
        return {
            "success": False,
            "error": "is_data_table",
            "path": str(resolved),
            "reason": f"{resolved.name} 是数据表（{ext}），应使用数据分析工具",
            "suggested_tools": ["profile_data", "compute_stats", "plot_chart"],
            "user_message": (
                f"{resolved.name} 是可分析的数据表，请用数据概况/统计工具查看"
                "（如 profile_data，可指定 table 参数）。"
                "本工具只用于查看文档与配置文件。"
            ),
        }

    if ext in _JSON_EXTS:
        result = _read_config(resolved, max_chars=limit)
    elif ext in _TEXT_EXTS:
        result = _read_text(
            resolved, max_chars=limit, start_line=start_line, keyword=keyword)
    else:
        # 未知扩展名：先按文本试读（很多格式就是纯文本），失败再如实说明。
        result = _read_text(
            resolved, max_chars=limit, start_line=start_line, keyword=keyword)

    if note:
        result.update(note)
    result["dataset"] = context.dataset_id
    return result


@tool
def read_file_content(
    wrapper: RunContextWrapper[RunContext],
    path: str,
    max_chars: int = 4000,
    start_line: int | None = None,
    keyword: str | None = None,
) -> dict:
    """查看数据集内任意文本文件的内容（只读；文档/配置/标定/元数据通用）。

    用于回答"这个数据集是干什么的""采集设备是什么""字段怎么定义""标定参数
    是多少"这类问题——**这些信息通常写在 README、文档、meta/info.json、
    标定 JSON 里，而它们不属于可分析的数据表**，此前没有任何读取路径。

    **不做内容理解**：本工具只返回**原文**（JSON 返回键值摘要），不解析、
    不总结、不抽取语义——含义由你阅读后判断。请勿引用本工具未返回的内容。

    Args:
        path: 文件路径，或数据集内的文件名（大小写不敏感；如 "README.md"）。
        max_chars: 返回的最大字符数（默认 4000，防止长文档撑爆上下文）。
        start_line: 可选，从第几行开始读（1-based）；用于分段读取长文件。
        keyword: 可选，只返回包含该关键词的行（含前后各 1 行，带行号）。
            用于在长文档中定位设备型号、字段定义等——**逐行匹配，不猜语义**。

    Returns:
        dict，含 success、path、kind（document / config）、content、
        n_lines_total、n_lines_returned、truncated、content_basis（来源说明，
        如"第 1-120 行"或"关键词命中 3 行"）；失败时含结构化 error
        与 user_message（找不到文件时附 available_documents 清单）。
    """
    return read_file_content_impl(
        wrapper.context, path,
        max_chars=max_chars, start_line=start_line, keyword=keyword,
    )
