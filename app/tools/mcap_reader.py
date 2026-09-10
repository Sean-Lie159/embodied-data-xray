"""MCAP 容器读取内核（原型，纯逻辑，可独立测试）。

背景与定位（详见 research/research_report_mcap_unpack_assessment.md）：

MCAP 是 Foxglove 开源的、序列化无关的时序消息容器，物理结构为
``<Magic><Header><Data><Summary?><Footer><Magic>``，核心 record 含
Schema / Channel / Message / Chunk。**单文件内含多路 topic**，这一点与
项目既有的 HDF5「单文件多节点」形态**同构**，因此沿用 h5 节点流范式：
流登记的 ``path`` 记为 ``"<mcap 路径>::<topic>"``。

本原型**只处理 JSON 编码的 MCAP**（``channel.message_encoding == "json"``）：
此类消息 ``message.data`` 直接就是 JSON 字节流，``json.loads`` 即得 dict，
**无需 schema 反序列化**——这是最低风险的接入面。ROS2 CDR 编码
（``message_encoding == "cdr"``，需按 ros2msg/ros2idl schema 反序列化）
列为第二期能力，本模块遇到时**如实标注 unsupported，不硬解**。

设计纪律（延续项目既定契约）：

- **只读发现**：``probe_mcap`` 只读 summary（channels / schemas / statistics），
  不逐条解码消息；``read_mcap_topic`` 才按 topic 读消息。
- **不硬猜**：非 JSON 编码的 channel 在清单中标 ``decodable=False`` 并给出
  原因，绝不用错误的解码器产出看似合理的伪值。
- **时间戳单位显式**：MCAP 的 ``log_time`` / ``publish_time`` 均为 uint64
  纳秒；导出为 DataFrame 时列名带 ``_ns`` 后缀，杜绝单位误读
  （项目曾在 ``duration_s`` 实为纳秒上出过事故，见 docs/技术债.md 第 4 条）。
- **解码失败可恢复**：单条消息解码失败只计数并跳过，不中断整个读取。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

# 本模块只支持 JSON 编码的消息；其余编码如实标注，不硬解。
_JSON_ENCODINGS = ("json",)

# 读取单条消息的字节上限保护：MCAP 规范默认记录长度上限 4GiB，
# 防损坏文件触发 OOM（社区实践，见评估报告）。
MAX_RECORD_BYTES = 4 * 1024 * 1024 * 1024


class McapDependencyError(RuntimeError):
    """运行环境缺少读取 MCAP 所需的依赖（mcap 包）。

    与「文件损坏/格式不对」严格区分：缺依赖时根本没读到文件内容，
    不得给出「可能损坏」之类误导性措辞（项目既有纪律，见
    load_dataset.MissingDependencyError 同款处理）。
    """

    def __init__(self) -> None:
        super().__init__("读取 MCAP 需要 mcap 包（pip install mcap），当前环境未安装")

    def user_hint(self) -> str:
        """可直接转达给用户的中文修复指引。"""
        return (
            "读取 .mcap 文件需要环境安装 mcap 依赖（pip 包名 mcap），"
            "当前环境未安装——文件未被读取，并非文件损坏。"
            "请在运行环境执行 `pip install mcap` 后重新加载。"
        )


def _import_mcap() -> Any:
    """延迟导入 mcap 包（保持项目模块导入期对可选依赖无硬依赖）。

    Returns:
        mcap.reader 模块对象。

    Raises:
        McapDependencyError: 环境未安装 mcap。
    """
    try:
        from mcap import reader as mcap_reader  # noqa: PLC0415

        return mcap_reader
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise McapDependencyError() from exc


def probe_mcap(path: str) -> dict[str, Any]:
    """读 MCAP 的 summary，返回 topic / schema / 统计概览（不逐条解码）。

    与项目「只读发现」纪律一致：本函数只解析容器尾部 summary 段，
    拿到 channels / schemas / statistics 即返回，**不读取消息体**，
    因此对 GB 级文件也是廉价的。

    Args:
        path: .mcap 文件路径。

    Returns:
        dict：
        - success: 是否成功；
        - topics: [{topic, message_encoding, schema_name, message_count,
          decodable, decode_note}]；decodable 表示本原型能否解码该 topic
          （仅 JSON 编码为 True）；
        - message_count: 全部 topic 的消息总数（来自 statistics，可能为 None）；
        - time_range_ns: {start_ns, end_ns}（来自 statistics，可能为 None）；
        - n_topics: topic 数；
        - file_size_bytes: 文件字节数；
        - error / user_message: 失败时的结构化错误。

    Raises:
        不直接抛出；缺依赖时抛 McapDependencyError 由调用方转结构化错误
        （与 load_dataset 的缺依赖契约一致），其余异常转 success=False。
    """
    src = Path(path)
    reader_mod = _import_mcap()  # 缺依赖 → McapDependencyError（调用方处理）

    if not src.exists():
        return {
            "success": False,
            "error": "file_not_found",
            "user_message": f"文件 {path} 不存在，请检查路径是否正确。",
        }

    try:
        with src.open("rb") as f:
            reader = reader_mod.make_reader(f)
            summary = reader.get_summary()
    except Exception as exc:  # noqa: BLE001 - 容器级解析失败转结构化错误
        return {
            "success": False,
            "error": "mcap_parse_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "user_message": (
                f"MCAP 文件 {src.name} 解析失败，可能不是有效的 MCAP 容器或已损坏。"
            ),
        }

    channels = list(getattr(summary, "channels", {}).values()) if summary else []
    schemas = getattr(summary, "schemas", {}) if summary else {}
    stats = getattr(summary, "statistics", None) if summary else None

    # 逐 topic 汇总消息数（statistics.channel_message_counts：channel_id → 计数）。
    counts: dict[int, int] = {}
    if stats is not None:
        raw_counts = getattr(stats, "channel_message_counts", None) or {}
        counts = {int(k): int(v) for k, v in raw_counts.items()}

    topics: list[dict[str, Any]] = []
    for ch in channels:
        enc = (getattr(ch, "message_encoding", "") or "").lower()
        decodable = enc in _JSON_ENCODINGS
        schema = schemas.get(getattr(ch, "schema_id", 0))
        topics.append({
            "topic": getattr(ch, "topic", ""),
            "message_encoding": enc or "unknown",
            "schema_name": getattr(schema, "name", None) if schema else None,
            "message_count": counts.get(int(getattr(ch, "id", -1))),
            "decodable": decodable,
            "decode_note": (
                None if decodable
                else f"本原型仅支持 JSON 编码的 MCAP；该 topic 编码为 {enc or 'unknown'}，"
                     "需 ROS2 CDR 解码（第二期能力），暂不解包"
            ),
        })
    # 稳定排序：消息数降序（主 topic 在首），并列按 topic 字母序。
    topics.sort(key=lambda t: (-(t["message_count"] or 0), t["topic"]))

    time_range = None
    if stats is not None:
        start = getattr(stats, "message_start_time", None)
        end = getattr(stats, "message_end_time", None)
        if start is not None and end is not None:
            time_range = {"start_ns": int(start), "end_ns": int(end)}

    total = getattr(stats, "message_count", None) if stats is not None else None

    return {
        "success": True,
        "topics": topics,
        "n_topics": len(topics),
        "message_count": int(total) if total is not None else None,
        "time_range_ns": time_range,
        "file_size_bytes": src.stat().st_size,
    }


def _iter_decoded_messages(
    path: str, topic: str
) -> Iterator[tuple[int, int, dict[str, Any], int]]:
    """遍历指定 topic 的 JSON 消息，产出 (log_time_ns, publish_time_ns, payload, seq)。

    单条解码失败（非法 JSON / 非 dict）只计数并跳过，不中断整体读取——
    与项目「错误可恢复、不因单条坏数据打崩」的纪律一致。

    Args:
        path: .mcap 文件路径。
        topic: 目标 topic 名。

    Yields:
        (log_time_ns, publish_time_ns, payload_dict, sequence)。
    """
    reader_mod = _import_mcap()
    with Path(path).open("rb") as f:
        reader = reader_mod.make_reader(f)
        for _schema, channel, message in reader.iter_messages(topics=[topic]):
            enc = (getattr(channel, "message_encoding", "") or "").lower()
            if enc not in _JSON_ENCODINGS:
                continue  # 非 JSON 编码：交由 probe 层标注，不在此硬解
            try:
                payload = json.loads(message.data)
            except (ValueError, TypeError):
                continue  # 单条非法 JSON：跳过，不中断
            if not isinstance(payload, dict):
                continue  # 非 dict 载荷（数组/标量）暂不作为表格行
            yield (
                int(message.log_time),
                int(message.publish_time),
                payload,
                int(getattr(message, "sequence", 0)),
            )


def read_mcap_topic(
    path: str,
    topic: str,
    max_messages: int | None = None,
) -> dict[str, Any]:
    """按 topic 读取 MCAP 消息为扁平 DataFrame（复用项目既有扁平化约定）。

    输出列结构（与 jsonl 信封流一致，便于复用 expand_envelope 等既有工具）：
    - ``mcap_log_time_ns`` / ``mcap_publish_time_ns``：容器时间（uint64 纳秒，
      来自 Message record，**不是**消息体内的传感器时间）；
    - ``mcap_sequence``：消息序号；
    - ``data``：消息体（object 列，嵌套 dict）——与 wujiGlove 导出形态同名，
      使 ``expand_envelope`` / ``discover_nested_fields`` 语义无缝衔接。

    Args:
        path: .mcap 文件路径。
        topic: 目标 topic 名。
        max_messages: 可选，最多读取的消息条数（防大文件撑爆内存）；None 为全量。

    Returns:
        dict：
        - success、df（DataFrame 或 None）、topic、n_rows、n_cols、
          columns、decoded、skipped（解码失败跳过的条数）；
        - 缺依赖 → 抛 McapDependencyError（调用方转结构化错误）；
        - topic 不存在 → success=False 且 error="topic_not_found"。
    """
    import pandas as pd  # 局部导入：保持模块导入期轻量

    rows: list[dict[str, Any]] = []
    skipped = 0
    for log_ns, pub_ns, payload, seq in _iter_decoded_messages(path, topic):
        rows.append({
            "mcap_log_time_ns": log_ns,
            "mcap_publish_time_ns": pub_ns,
            "mcap_sequence": seq,
            "data": payload,
        })
        if max_messages is not None and len(rows) >= max_messages:
            break

    if not rows:
        # 区分「topic 不存在」与「存在但无可解码消息」。
        probe = probe_mcap(path)
        known = {t["topic"] for t in probe.get("topics", [])} if probe.get("success") else set()
        if topic not in known:
            return {
                "success": False,
                "error": "topic_not_found",
                "topic": topic,
                "df": None,
                "user_message": (
                    f"MCAP 中不存在 topic {topic}。可用 topic 见 probe_mcap 的 topics 清单。"
                ),
            }
        return {
            "success": True,
            "df": None,
            "topic": topic,
            "n_rows": 0,
            "n_cols": 0,
            "columns": [],
            "decoded": 0,
            "skipped": 0,
            "user_message": f"topic {topic} 存在但没有可解码的 JSON 消息。",
        }

    df = pd.DataFrame(rows)
    return {
        "success": True,
        "df": df,
        "topic": topic,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "decoded": len(rows),
        "skipped": skipped,
    }


# --- 解包落盘 ---------------------------------------------------------------

# 解包支持的输出格式 → 扩展名。
UNPACK_FORMATS: dict[str, str] = {
    "jsonl": ".jsonl",
    "json": ".json",
    "csv": ".csv",
}


def _safe_filename(topic: str) -> str:
    """把 topic 名转成安全的文件名（去掉前导斜杠、替换路径分隔符）。"""
    cleaned = topic.strip().strip("/").replace("/", "__").replace("\\", "__")
    # 兜底：替换 Windows 非法文件名字符。
    for ch in '<>:"|?*':
        cleaned = cleaned.replace(ch, "_")
    return cleaned or "topic"


def unpack_mcap_to_dir(
    path: str,
    output_dir: str,
    topics: list[str] | None = None,
    fmt: str = "jsonl",
    max_messages: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """把 MCAP 的指定 topic 解包为文件落盘（格式自选）。

    落盘策略（社区共识 + 项目纪律）：
    - **按 topic 分文件**——异构 topic 结构差异大，合表会列爆炸；
    - 落在调用方指定的 ``output_dir`` 下，**绝不写入数据集源目录**
      （延续 docs/四层语义识别架构.md 的「不污染原始数据」纪律）；
    - 文件名为 topic 的安全化形式（``/imu`` → ``imu.jsonl``）。

    各格式语义：
    - ``jsonl``：每消息一行（``{mcap_log_time_ns, mcap_publish_time_ns,
      mcap_sequence, data}``）——**推荐默认**，保留嵌套、可流式追加；
    - ``json``：整体一个数组；
    - ``csv``：扁平化成宽表（``data`` 列序列化为 JSON 字符串，防止丢字段）。

    Args:
        path: .mcap 文件路径。
        output_dir: 落盘目录（调用方负责给出，如 outputs/mcap_unpack/<id>）。
        topics: 要解包的 topic 列表；None 表示全部「可解码」的 topic。
        fmt: 输出格式（jsonl / json / csv）。
        max_messages: 每 topic 最多解包的消息条数；None 为全量。
        progress: 可选回调（每完成一个 topic 调用一次），供 UI 展示进度。

    Returns:
        dict：success、written（[{topic, file, format, n_rows}]）、
        skipped_topics（[{topic, reason}]，如非 JSON 编码）、output_dir、
        user_message；fmt 非法 → success=False。
    """
    import pandas as pd  # 局部导入

    if fmt not in UNPACK_FORMATS:
        return {
            "success": False,
            "error": "unsupported_format",
            "user_message": (
                f"不支持的解包格式 {fmt}，可选：{'、'.join(UNPACK_FORMATS)}。"
            ),
        }

    probe = probe_mcap(path)
    if not probe.get("success"):
        return probe

    decodable = {t["topic"]: t for t in probe["topics"] if t["decodable"]}
    skipped: list[dict[str, str]] = [
        {"topic": t["topic"], "reason": t["decode_note"] or "不可解码"}
        for t in probe["topics"] if not t["decodable"]
    ]

    target_topics = topics if topics is not None else list(decodable)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict[str, Any]] = []
    for topic in target_topics:
        if topic not in decodable:
            skipped.append({
                "topic": topic,
                "reason": "不在可解码 topic 清单中（不存在或非 JSON 编码）",
            })
            continue
        res = read_mcap_topic(path, topic, max_messages=max_messages)
        df: "pd.DataFrame | None" = res.get("df")
        if df is None:
            skipped.append({"topic": topic, "reason": "无可解码消息"})
            continue
        out_path = out_dir / f"{_safe_filename(topic)}{UNPACK_FORMATS[fmt]}"
        _write_payload(df, out_path, fmt)
        written.append({
            "topic": topic,
            "file": str(out_path),
            "format": fmt,
            "n_rows": int(df.shape[0]),
        })
        if progress is not None:
            progress(topic)

    return {
        "success": True,
        "written": written,
        "skipped_topics": skipped,
        "output_dir": str(out_dir),
        "n_written": len(written),
        "user_message": (
            f"已解包 {len(written)} 个 topic 到 {out_dir}"
            + (f"，跳过 {len(skipped)} 个（非 JSON 编码或不存在）。" if skipped else "。")
        ),
    }


# --- topic 语义分类（第 1 层命名线索 + 第 2 层展开样本指纹）-------------------
#
# 设计依据见 docs/MCAP支持设计说明.md 第 4 节。核心问题：既有
# classify_table_stream 依赖「文件名 + 顶层列名」，而 MCAP 的语义信息在两个
# 既有通道之外——① topic 名（/imu 不含 accel/gyro 字样）；② 信号藏在 data
# 嵌套 dict 内（对顶层列词典不可见）。故此处给既有识别架构补上 MCAP 特有的
# 两个输入通道：topic 名（伪文件名线索）+ 展开样本（指纹输入）。

# topic 名特征 → (kind, 语义标签) 的显式命名线索映射（可审计）。
_TOPIC_NAME_HINTS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("tf_static", "static_tf"), "static", "静态坐标变换（不随时间变化）"),
    (("imu", "accel", "gyro", "gyr"), "imu", "IMU（加速度/角速度）"),
    (("tactile", "touch", "force", "torque", "wrench"), "tactile", "触觉/力觉"),
    (("joint", "cmd", "action", "ctrl"), "actions", "关节状态/动作"),
    (("pose", "odom", "tf"), "pose", "位姿（6DoF）"),
    (("camera", "image", "rgb", "depth", "color"), "image", "图像流"),
)


def _hint_from_topic(topic: str) -> tuple[str, str] | None:
    """按 topic 名给出第 1 层命名线索（kind, 语义标签）；未命中返回 None。

    Args:
        topic: MCAP topic 名（如 "/imu"）。

    Returns:
        (kind, semantic_label)；无命名线索返回 None（交由指纹层判定，不硬猜）。
    """
    lower = topic.lower()
    for keys, kind, label in _TOPIC_NAME_HINTS:
        if any(k in lower for k in keys):
            return kind, label
    return None


def classify_mcap_topic(
    topic: str,
    name: str,
    columns: list[str],
    sample: "Any | None",
    nrows: int,
) -> dict[str, Any]:
    """对 MCAP topic 给出语义标签（命名线索优先，展开样本指纹兜底）。

    与既有 `_sniffing.classify_table_stream` 的分工：本函数**先**用 topic 名
    给出命名线索（MCAP 特有通道），命中即返回并附证据；未命中则**委托**
    既有分类器（传入展开后的样本，使四元数/加速度指纹可达），保持判定口径统一。

    Args:
        topic: MCAP topic 名。
        name: 伪文件名（供既有分类器做命名匹配与证据展示）。
        columns: 顶层列名。
        sample: 样本 DataFrame（**建议传 expand_envelope 展开后的视图**，
            使嵌套信号对指纹层可见）；可为 None。
        nrows: 该 topic 消息数。

    Returns:
        dict，与 classify_table_stream 同结构（kind / semantic_label /
        label_evidence / label_confidence / status / channels /
        timestamp_column / quaternion_groups / imu_axes），额外含
        label_source 标注判定来源（topic_name_hint / content_fingerprint）。
    """
    hint = _hint_from_topic(topic)
    if hint is not None:
        kind, label = hint
        return {
            "kind": kind,
            "semantic_label": label,
            "label_evidence": f"topic 名命名线索（{topic}）",
            "label_confidence": "medium",
            "label_source": "topic_name_hint",
            "status": "active",
            "channels": columns,
            "timestamp_column": None,
            "quaternion_groups": [],
            "imu_axes": None,
        }

    # 未命中命名线索 → 委托既有分类器（传入展开样本，让指纹层够得着嵌套信号）。
    from app.tools._sniffing import classify_table_stream  # 局部导入避免循环

    klass = classify_table_stream(name, columns, sample, nrows)
    klass["label_source"] = f"content_fingerprint/{klass.get('label_source', 'classify')}"
    return klass


def unpack_mcap_tool_impl(
    context: "Any",
    topics: "list[str] | None" = None,
    fmt: str = "jsonl",
    max_messages: "int | None" = None,
) -> dict[str, Any]:
    """`unpack_mcap` 工具的实现体（落在当前数据集的 outputs 子目录下）。

    落盘目录固定为 ``<output_dir>/mcap_unpack/<dataset_id>/``——集中管理、
    不写数据集源目录（延续 docs/四层语义识别架构.md 第 4 层纪律）。

    Args:
        context: RunContext（取 output_dir / dataset_id / meta）。
        topics: 要解包的 topic 列表；省略时解包全部可解码 topic。
        fmt: 输出格式（jsonl / json / csv）。
        max_messages: 每 topic 最多解包的消息条数；省略为全量。

    Returns:
        见 unpack_mcap；额外含 dataset 字段标注来源数据集。
    """
    from pathlib import Path as _Path

    source = (context.meta or {}).get("source")
    # 仅当当前数据集是 mcap 时才允许解包（避免对非 mcap 数据误落盘）。
    if not source or str(source).lower().endswith(".mcap") is False:
        # 也兼容目录加载场景：从流登记表找 mcap 文件。
        mcap_streams = [
            s for s in (context.meta or {}).get("streams", [])
            if s.get("format") == "mcap"
        ]
        if mcap_streams:
            source = mcap_streams[0]["path"].split("::")[0]
        else:
            return {
                "success": False,
                "error": "no_mcap_loaded",
                "user_message": (
                    "当前数据集不是 MCAP 文件，无法解包。请先 load_dataset 加载一个 .mcap 文件。"
                ),
            }

    dataset_id = context.dataset_id or _Path(str(source)).stem
    out_dir = _Path(context.output_dir) / "mcap_unpack" / str(dataset_id)
    result = unpack_mcap_to_dir(
        str(source), str(out_dir), topics=topics, fmt=fmt,
        max_messages=max_messages,
    )
    result["dataset"] = dataset_id
    result["source"] = str(source)
    return result


def _write_payload(df: "pd.DataFrame", out_path: Path, fmt: str) -> None:
    """按格式把 topic 表写入文件（UTF-8）。"""
    if fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif fmt == "json":
        out_path.write_text(
            json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif fmt == "csv":
        export = df.copy()
        # data 为 object 列（嵌套 dict）：序列化为 JSON 字符串，避免丢字段。
        if "data" in export.columns:
            export["data"] = export["data"].map(
                lambda v: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list)) else v
            )
        export.to_csv(out_path, index=False, encoding="utf-8")
    else:  # pragma: no cover - 已被上游校验拦截
        raise ValueError(f"未实现的解包格式：{fmt}")


# --- Agent 工具包装 ---------------------------------------------------------

try:  # pragma: no cover - 装饰器注册在 import 期完成，行为由下游测试覆盖
    from agents import RunContextWrapper
    from agents.decorators import tool

    from app.agent.context import RunContext

    @tool
    def unpack_mcap(
        wrapper: RunContextWrapper[RunContext],
        topics: "list[str] | None" = None,
        fmt: str = "jsonl",
        max_messages: "int | None" = None,
    ) -> dict:
        """把当前 MCAP 数据集的指定 topic 解包落盘（格式可自选）。

        仅对 MCAP 编码为 JSON 的 topic 有效；ROS2 CDR 编码的 topic 本期不解包
        （返回中 skipped_topics 会如实说明原因）。文件落在 outputs/mcap_unpack/
        <数据集名>/ 下，不写入数据集源目录。

        Args:
            topics: 要解包的 topic 列表（如 ["/imu", "/joint_states"]）；
                省略时解包全部可解码 topic。
            fmt: 输出格式，可选 jsonl（默认，每消息一行、保留嵌套）、
                json（整体数组）、csv（扁平宽表，data 列序列化为 JSON 字符串）。
            max_messages: 每个 topic 最多解包的消息条数（大文件时用）；省略为全量。

        Returns:
            dict，含 success、written（逐 topic 的文件路径与行数）、
            skipped_topics（未解包的 topic 及原因）、output_dir、dataset、
            user_message；当前数据集非 MCAP 时返回结构化错误。
        """
        return unpack_mcap_tool_impl(
            wrapper.context, topics=topics, fmt=fmt, max_messages=max_messages,
        )

except ImportError:  # pragma: no cover - 无 agents 环境（如独立跑内核测试）
    unpack_mcap = None  # type: ignore[assignment]
