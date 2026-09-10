"""JSON 编码 MCAP 读取内核的原型验证测试。

验证链路：合成 MCAP（make_mcap）→ probe_mcap（只读 summary）→
read_mcap_topic（按 topic 读为 DataFrame）→ unpack_mcap（落盘 jsonl/json/csv）。

覆盖要点：
1. probe 只读 summary，正确列出 topic / 编码 / 消息数 / 可解码性；
2. 双时钟：容器 mcap_log_time_ns（纳秒，带 _ns 后缀）与消息体内
   header.timestamp_us 并存，且不被混淆；
3. 非 JSON 编码 topic 如实标注 decodable=False，不硬解（诚实降级）；
4. 落盘三格式（jsonl/json/csv）内容正确、按 topic 分文件；
5. 落盘不写数据集源目录（调用方指定 output_dir）；
6. 缺依赖时抛结构化 McapDependencyError（与既有缺依赖契约一致）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools._data_access import resolve_table_name
from app.tools.mcap_reader import (
    McapDependencyError,
    probe_mcap,
    read_mcap_topic,
    unpack_mcap_to_dir,
)
from tests.fixtures.make_mcap import build_mcap, build_mcap_with_cdr_topic

_BASE_NS = 1_787_294_456_000_000_000


@pytest.fixture()
def demo_mcap(tmp_path: Path) -> Path:
    return build_mcap(tmp_path / "demo.mcap", n_imu=100, n_joint=50, n_tf_static=3)


# --- 1. probe：只读 summary -------------------------------------------------


def test_probe_lists_topics_sorted_by_count(demo_mcap: Path) -> None:
    """probe 列出全部 topic，按消息数降序（主 topic 在首）。"""
    r = probe_mcap(str(demo_mcap))
    assert r["success"] is True
    assert r["n_topics"] == 3
    topics = [t["topic"] for t in r["topics"]]
    assert topics[0] == "/imu"  # 100 条，最多
    assert set(topics) == {"/imu", "/joint_states", "/tf_static"}
    counts = {t["topic"]: t["message_count"] for t in r["topics"]}
    assert counts == {"/imu": 100, "/joint_states": 50, "/tf_static": 3}


def test_probe_reports_encoding_and_decodable(demo_mcap: Path) -> None:
    """JSON 编码 topic 标 decodable=True，且无解码警示。"""
    r = probe_mcap(str(demo_mcap))
    for t in r["topics"]:
        assert t["message_encoding"] == "json"
        assert t["decodable"] is True
        assert t["decode_note"] is None


def test_probe_time_range_and_total(demo_mcap: Path) -> None:
    """statistics 透出消息总数与时间范围（纳秒）。"""
    r = probe_mcap(str(demo_mcap))
    assert r["message_count"] == 153  # 100 + 50 + 3
    tr = r["time_range_ns"]
    assert tr is not None
    assert tr["start_ns"] == _BASE_NS
    assert tr["end_ns"] > tr["start_ns"]


def test_probe_missing_file(tmp_path: Path) -> None:
    """文件不存在 → 结构化错误，不抛异常。"""
    r = probe_mcap(str(tmp_path / "nope.mcap"))
    assert r["success"] is False and r["error"] == "file_not_found"


# --- 2. 双时钟：容器纳秒 vs 消息体传感器时间 --------------------------------


def test_read_topic_container_time_is_ns_suffixed(demo_mcap: Path) -> None:
    """容器时间列名带 _ns 后缀（杜绝单位误读），数值为纳秒。"""
    r = read_mcap_topic(str(demo_mcap), "/imu")
    assert r["success"] is True
    df = r["df"]
    assert df.shape[0] == 100
    assert "mcap_log_time_ns" in df.columns
    assert "mcap_publish_time_ns" in df.columns
    assert "mcap_sequence" in df.columns
    assert "data" in df.columns
    # 容器时间为纳秒量级（epoch ns），非秒。
    assert int(df["mcap_log_time_ns"].iloc[0]) == _BASE_NS


def test_read_topic_nested_sensor_time_preserved(demo_mcap: Path) -> None:
    """消息体内的 header.timestamp_us 保留在 data 列，与容器时间并存不混淆。"""
    r = read_mcap_topic(str(demo_mcap), "/imu")
    df = r["df"]
    first = df["data"].iloc[0]
    assert isinstance(first, dict)
    assert "header" in first and "timestamp_us" in first["header"]
    # 传感器时间（us）与容器时间（ns）是同一轴的不同表示，量级相差 1000。
    assert abs((_BASE_NS // 1000) - first["header"]["timestamp_us"]) <= 20


def test_read_topic_quaternion_in_data(demo_mcap: Path) -> None:
    """四元数与加速度向量完整保留在嵌套 data 中（可被 expand_envelope 消费）。"""
    df = read_mcap_topic(str(demo_mcap), "/imu")["df"]
    d = df["data"].iloc[0]
    assert set(d["orientation"]) == {"x", "y", "z", "w"}
    assert "linear_acceleration" in d


def test_read_topic_missing_returns_structured(demo_mcap: Path) -> None:
    """不存在的 topic → 结构化 topic_not_found。"""
    r = read_mcap_topic(str(demo_mcap), "/no_such_topic")
    assert r["success"] is False and r["error"] == "topic_not_found"


def test_read_topic_max_messages(demo_mcap: Path) -> None:
    """max_messages 限制读取条数（防大文件内存）。"""
    r = read_mcap_topic(str(demo_mcap), "/imu", max_messages=10)
    assert r["df"].shape[0] == 10


# --- 3. 非 JSON 编码：诚实降级，不硬解 --------------------------------------


def test_non_json_topic_marked_not_decodable(tmp_path: Path) -> None:
    """CDR 编码 topic 标 decodable=False + decode_note，不硬解。"""
    p = build_mcap_with_cdr_topic(tmp_path / "mixed.mcap")
    r = probe_mcap(str(p))
    by_topic = {t["topic"]: t for t in r["topics"]}
    assert by_topic["/scan"]["decodable"] is False
    assert "cdr" in by_topic["/scan"]["message_encoding"]
    assert by_topic["/scan"]["decode_note"]
    assert by_topic["/imu"]["decodable"] is True


def test_unpack_skips_non_json_topic(tmp_path: Path) -> None:
    """解包时跳过非 JSON 编码 topic，并如实说明原因。"""
    p = build_mcap_with_cdr_topic(tmp_path / "mixed.mcap")
    out = tmp_path / "out"
    r = unpack_mcap_to_dir(str(p), str(out), fmt="jsonl")
    assert r["success"] is True
    assert {w["topic"] for w in r["written"]} == {"/imu"}
    assert any(s["topic"] == "/scan" for s in r["skipped_topics"])


# --- 4. 解包落盘：三格式 ----------------------------------------------------


def test_unpack_jsonl_one_line_per_message(demo_mcap: Path, tmp_path: Path) -> None:
    """jsonl：每消息一行，行数等于消息数，含容器时间与 data。"""
    out = tmp_path / "out"
    r = unpack_mcap_to_dir(str(demo_mcap), str(out), topics=["/imu"], fmt="jsonl")
    assert r["success"] is True
    f = Path(r["written"][0]["file"])
    assert f.name == "imu.jsonl"
    lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 100
    rec = json.loads(lines[0])
    assert rec["mcap_log_time_ns"] == _BASE_NS
    assert rec["data"]["orientation"]["w"] == 1.0


def test_unpack_json_array(demo_mcap: Path, tmp_path: Path) -> None:
    """json：整体一个数组。"""
    out = tmp_path / "out"
    r = unpack_mcap_to_dir(str(demo_mcap), str(out), topics=["/joint_states"], fmt="json")
    f = Path(r["written"][0]["file"])
    arr = json.loads(f.read_text(encoding="utf-8"))
    assert isinstance(arr, list) and len(arr) == 50


def test_unpack_csv_serializes_nested_data(demo_mcap: Path, tmp_path: Path) -> None:
    """csv：data 列序列化为 JSON 字符串（不丢嵌套字段）。"""
    import pandas as pd

    out = tmp_path / "out"
    r = unpack_mcap_to_dir(str(demo_mcap), str(out), topics=["/imu"], fmt="csv")
    f = Path(r["written"][0]["file"])
    df = pd.read_csv(f)
    assert df.shape[0] == 100
    assert "data" in df.columns
    d = json.loads(df["data"].iloc[0])  # 可从字符串反解析回 dict
    assert d["orientation"]["w"] == 1.0


def test_unpack_all_topics_by_default(demo_mcap: Path, tmp_path: Path) -> None:
    """topics=None → 解包全部可解码 topic，每个 topic 一个文件。"""
    out = tmp_path / "out"
    r = unpack_mcap_to_dir(str(demo_mcap), str(out), fmt="jsonl")
    assert r["n_written"] == 3
    names = {Path(w["file"]).name for w in r["written"]}
    assert names == {"imu.jsonl", "joint_states.jsonl", "tf_static.jsonl"}


def test_unpack_writes_to_given_dir_not_source(tmp_path: Path) -> None:
    """落盘写入调用方指定目录，绝不写入数据集源目录（不污染原始数据）。"""
    # 源目录与输出目录分离（模拟真实纪律：outputs/ 与 data/ 不同处）。
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    demo = build_mcap(source_dir / "demo.mcap", n_imu=20, n_joint=5, n_tf_static=2)
    out = tmp_path / "outputs" / "unpacked"

    r = unpack_mcap_to_dir(str(demo), str(out), fmt="jsonl")
    assert Path(r["output_dir"]) == out
    for w in r["written"]:
        assert Path(w["file"]).parent == out
        assert Path(w["file"]).parent != source_dir
    # 源目录未被污染：除原 mcap 外无新增文件。
    assert list(source_dir.glob("*")) == [demo]


def test_unpack_invalid_format(demo_mcap: Path, tmp_path: Path) -> None:
    """非法格式 → 结构化错误。"""
    r = unpack_mcap_to_dir(str(demo_mcap), str(tmp_path / "o"), fmt="xml")
    assert r["success"] is False and r["error"] == "unsupported_format"


def test_unpack_unknown_topic_reported(demo_mcap: Path, tmp_path: Path) -> None:
    """请求不存在的 topic → 计入 skipped，不中断其余 topic。"""
    r = unpack_mcap_to_dir(
        str(demo_mcap), str(tmp_path / "o"), topics=["/imu", "/ghost"], fmt="jsonl",
    )
    assert r["n_written"] == 1
    assert any(s["topic"] == "/ghost" for s in r["skipped_topics"])


# --- 5. 缺依赖契约 ----------------------------------------------------------


def test_missing_dependency_error_message() -> None:
    """McapDependencyError 的 user_hint 必须明确「非文件损坏」并给出修复指令。"""
    err = McapDependencyError()
    hint = err.user_hint()
    assert "mcap" in hint
    # 明确澄清「并非文件损坏」，且给出可执行的 pip 修复指令（不误导用户）。
    assert "并非文件损坏" in hint
    assert "pip install mcap" in hint
    assert "未安装" in hint


# --- 6. topic 名 → 语义分类（设计说明第 4 节核心决策）-----------------------


def test_classify_topic_name_hints() -> None:
    """topic 名命名线索生效：/imu→imu、/joint_states→actions、/tf_static→static。"""
    from app.tools.mcap_reader import classify_mcap_topic

    assert classify_mcap_topic("/imu", "x::/imu", [], None, 0)["kind"] == "imu"
    assert classify_mcap_topic(
        "/joint_states", "x::/joint_states", [], None, 0
    )["kind"] == "actions"
    assert classify_mcap_topic(
        "/tf_static", "x::/tf_static", [], None, 0
    )["kind"] == "static"
    assert classify_mcap_topic("/camera/rgb", "x::/camera/rgb", [], None, 0)["kind"] == "image"


def test_classify_unknown_topic_not_guessed() -> None:
    """无命名线索且无样本 → 如实 unknown，不硬猜（交由假设/确认层）。"""
    from app.tools.mcap_reader import classify_mcap_topic

    r = classify_mcap_topic("/mystery", "x::/mystery", ["a_ns", "v"], None, 10)
    assert r["kind"] == "unknown"


# --- 7. 接线：load_dataset 加载 .mcap + topic 登记为流 ----------------------


def test_load_mcap_registers_topic_streams(demo_mcap: Path, tmp_path: Path) -> None:
    """加载 .mcap → 全部 topic 登记为流，主 topic 标 is_main，kind 由 topic 名判定。"""
    from app.agent.context import RunContext
    from app.tools.load_dataset import load_dataset_impl

    ctx = RunContext(output_dir=str(tmp_path / "outputs"))
    r = load_dataset_impl(ctx, str(demo_mcap))
    assert r["success"] is True
    assert r["mcap_main_topic"] == "/imu"
    streams = ctx.meta["streams"]
    kinds = {s["path"].split("::")[-1]: s["kind"] for s in streams}
    assert kinds == {"/imu": "imu", "/joint_states": "actions", "/tf_static": "static"}
    assert sum(1 for s in streams if s.get("is_main")) == 1


def test_resolve_mcap_topic_by_name(demo_mcap: Path, tmp_path: Path) -> None:
    """按流名（stem::topic 或裸 topic）读取非主 topic，主表不被替换。"""
    from app.agent.context import RunContext
    from app.tools.load_dataset import load_dataset_impl

    ctx = RunContext(output_dir=str(tmp_path / "outputs"))
    load_dataset_impl(ctx, str(demo_mcap))

    r1 = resolve_table_name(ctx, "demo::/joint_states")
    assert r1["success"] is True and r1["source"] == "mcap_topic"
    assert r1["df"].shape == (50, 4)
    r2 = resolve_table_name(ctx, "/tf_static")
    assert r2["success"] is True and r2["df"].shape == (3, 4)
    # 主表仍是 /imu（100 条），未被替换。
    assert ctx.df.shape == (100, 4)


def test_resolve_mcap_topic_with_expand(demo_mcap: Path, tmp_path: Path) -> None:
    """expand 对 mcap topic 生效：嵌套 data 展开为点分列。"""
    from app.agent.context import RunContext
    from app.tools.load_dataset import load_dataset_impl

    ctx = RunContext(output_dir=str(tmp_path / "outputs"))
    load_dataset_impl(ctx, str(demo_mcap))
    r = resolve_table_name(ctx, "/imu", expand=True)
    assert r["success"] is True
    assert "data.orientation.w" in r["df"].columns


def test_resolve_mcap_missing_topic(demo_mcap: Path, tmp_path: Path) -> None:
    """不存在的 topic → 结构化 table_not_found。"""
    from app.agent.context import RunContext
    from app.tools.load_dataset import load_dataset_impl

    ctx = RunContext(output_dir=str(tmp_path / "outputs"))
    load_dataset_impl(ctx, str(demo_mcap))
    r = resolve_table_name(ctx, "/ghost")
    assert r["success"] is False and r["error"] == "table_not_found"


def test_unpack_mcap_tool_impl_writes_under_outputs(tmp_path: Path) -> None:
    """unpack_mcap 工具实现：落盘到 outputs/mcap_unpack/<id>/，不碰源目录。"""
    from app.agent.context import RunContext
    from app.tools.load_dataset import load_dataset_impl
    from app.tools.mcap_reader import unpack_mcap_tool_impl

    # 源目录与 outputs 分离（模拟真实纪律：data/ 与 outputs/ 不同处）。
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    demo = build_mcap(source_dir / "demo.mcap", n_imu=20, n_joint=8, n_tf_static=2)
    out = tmp_path / "outputs"

    ctx = RunContext(output_dir=str(out))
    load_dataset_impl(ctx, str(demo))
    r = unpack_mcap_tool_impl(ctx, fmt="jsonl")
    assert r["success"] is True and r["n_written"] == 3
    assert Path(r["output_dir"]) == out / "mcap_unpack" / "demo"
    assert Path(r["written"][0]["file"]).parent == out / "mcap_unpack" / "demo"
    # 源目录未被污染。
    assert list(source_dir.glob("*")) == [demo]


def test_unpack_mcap_tool_impl_guards_non_mcap(tmp_path: Path) -> None:
    """当前数据集非 MCAP → 结构化错误，不误落盘。"""
    from app.agent.context import RunContext
    from app.tools.mcap_reader import unpack_mcap_tool_impl

    ctx = RunContext(output_dir=str(tmp_path / "outputs"))
    r = unpack_mcap_tool_impl(ctx)
    assert r["success"] is False and r["error"] == "no_mcap_loaded"
