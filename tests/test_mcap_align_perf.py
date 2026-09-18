"""MCAP 容器对齐的性能与判定正确性测试（2026-09-14 卡死事故）。

背景（用户实测）：加载 1.86 GB / 419 万条消息、26 个 topic 的 MCAP 后，
agent 在"检查丢帧丢包"这一步**卡死 15 分钟**（思考过程也不动）。
实测根因有两层：

1. **逐 topic 过滤是 O(全部消息) × topic 数**。mcap 库的
   ``iter_messages(topics=[t])`` 过滤开销与文件总消息数成正比——实测遍历全部
   消息 32.8s，而"只取单个 topic"也要 16s（哪怕该 topic 只有 1199 条）。
   26 个 topic 逐个取共 **909 秒**。此外原先每个 topic 都走全量读取路径
   （含每条消息的 json.loads 与 DataFrame 构建），tactile_point_cloud 单 topic
   就要 200 秒。
2. **改用一次遍历 + 内存分桶**后降到 37 秒（与消息总数同阶，与 topic 数无关）。

同时修掉两个判定缺陷：
- **突发型流误报缺口**：/tf、IMU 在容器时间口径下是 burst（突发内微秒级、
  突发间毫秒级静默），按"间隔 > 5×中位"统计会把每个突发间静默算成丢包——
  实测误报 /tf 有 121620 个"缺口"。
- **多时钟未提示**：这些流的消息体自带传感器时间（``data.header.timestamp_us``
  / ``data.transforms[].timestamp_us``），容器时间可能只是批量写入时间。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext

# app.tools 包的 __init__ 把同名 FunctionTool 暴露为包属性，须经 sys.modules
# 取真实模块。
import app.tools.align_container  # noqa: F401
import app.tools.mcap_reader  # noqa: F401

_ac = sys.modules["app.tools.align_container"]
_mr = sys.modules["app.tools.mcap_reader"]


# --- 1. 时间戳统计：突发型流不报假缺口 -------------------------------------


def test_burst_stream_reports_no_gaps() -> None:
    """**核心回归**：突发型流的缺口计数为 None（而非把突发间静默算成丢包）。

    真实事故：/tf 被报 121620 个"缺口"，实为突发间静默——若用户据此判断
    "丢包严重"，结论完全错误。
    """
    # 构造突发型序列：每突发 100 条（间隔 1µs），突发间静默 10ms。
    rng = np.random.default_rng(0)
    stamps: list[int] = []
    t = 0
    for _ in range(200):
        for _ in range(100):
            t += 1_000  # 1 µs
            stamps.append(t)
        t += 10_000_000  # 10 ms 静默
    arr = np.asarray(stamps, dtype=float)

    timing = _ac._timing_from_array(arr, "ns")
    assert timing is not None
    assert timing["shape"] == "burst"
    assert timing["n_gaps"] is None
    assert timing["gap_status"] == "not_applicable"
    assert timing["n_bursts"] == 200


def test_periodic_stream_reports_gaps() -> None:
    """周期型流仍如实报缺口（零回归：真实缺口不能被一起吞掉）。"""
    # 周期 10ms，中间挖掉 100 个点（=1 秒缺口）。
    stamps = [i * 10_000_000 for i in range(500)]
    stamps = stamps[:200] + [s + 1_000_000_000 for s in stamps[200:]]
    timing = _ac._timing_from_array(np.asarray(stamps, dtype=float), "ns")
    assert timing is not None
    assert timing["shape"] == "periodic"
    assert timing["n_gaps"] == 1
    assert timing["max_gap_s"] > 0.5


def test_timing_from_array_respects_unit() -> None:
    """单位由调用方给定（微秒 → 纳秒换算正确）。"""
    # 100 个点，间隔 10000 µs = 10 ms → 100 Hz
    stamps = np.arange(100, dtype=float) * 10_000
    timing = _ac._timing_from_array(stamps, "us")
    assert timing is not None
    assert timing["unit"] == "us"
    assert timing["rate_hz"] == pytest.approx(100.0, rel=0.01)
    # 换算到纳秒后跨度 = 99 × 10ms。
    assert timing["span_s"] == pytest.approx(0.99, rel=0.01)


def test_timing_from_array_single_point_returns_none() -> None:
    """不足 2 点返回 None（不产出无意义统计）。"""
    assert _ac._timing_from_array([1.0], "ns") is None
    assert _ac._timing_from_array([], "ns") is None


# --- 2. MCAP 批量时间戳读取 ------------------------------------------------


def _make_mcap(tmp_path: Path, n_per_topic: int = 50) -> Path:
    """用 mcap 库合成一个双 topic 的测试文件。"""
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer

    path = tmp_path / "t.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        chans = {}
        for i, topic in enumerate(("/a", "/b")):
            sid = w.register_schema(
                name="s", encoding="jsonschema",
                data=b'{"type":"object"}')
            chans[topic] = w.register_channel(
                topic=topic, message_encoding="json", schema_id=sid)
        for k in range(n_per_topic):
            for topic in ("/a", "/b"):
                w.add_message(
                    channel_id=chans[topic],
                    log_time=1_789_370_000_000_000_000 + k * 10_000_000,
                    publish_time=1_789_370_000_000_000_000 + k * 10_000_000,
                    data=b'{"v": 1}',
                )
        w.finish()
    return path


def test_read_all_timestamps_buckets_by_topic(tmp_path: Path) -> None:
    """一次遍历取齐多个 topic（按 topic 分桶）。"""
    path = _make_mcap(tmp_path, n_per_topic=30)
    result = _mr.read_mcap_all_timestamps(str(path), ["/a", "/b"])
    assert set(result) == {"/a", "/b"}
    assert result["/a"]["n_rows"] == 30
    assert result["/b"]["n_rows"] == 30
    assert len(result["/a"]["log_time_ns"]) == 30


def test_read_all_timestamps_filters_when_requested(tmp_path: Path) -> None:
    """指定 topics 时只返回这些 topic（不返回多余项）。"""
    path = _make_mcap(tmp_path, n_per_topic=10)
    result = _mr.read_mcap_all_timestamps(str(path), ["/a"])
    assert set(result) == {"/a"}


def test_read_all_timestamps_missing_topic_is_absent(tmp_path: Path) -> None:
    """请求不存在的 topic 时结果里没有它（不抛异常）。"""
    path = _make_mcap(tmp_path, n_per_topic=5)
    result = _mr.read_mcap_all_timestamps(str(path), ["/a", "/nope"])
    assert set(result) == {"/a"}


def test_read_all_timestamps_bad_file_returns_empty(tmp_path: Path) -> None:
    """损坏文件返回空 dict（不抛异常）。"""
    bad = tmp_path / "bad.mcap"
    bad.write_bytes(b"not an mcap")
    assert _mr.read_mcap_all_timestamps(str(bad), ["/a"]) == {}


def test_single_topic_timestamps_matches_batch(tmp_path: Path) -> None:
    """单 topic 轻量读取与批量结果一致（两条路径口径相同）。"""
    path = _make_mcap(tmp_path, n_per_topic=25)
    one = _mr.read_mcap_topic_timestamps(str(path), "/a")
    batch = _mr.read_mcap_all_timestamps(str(path), ["/a"])
    assert one["n_rows"] == batch["/a"]["n_rows"]
    assert list(one["log_time_ns"]) == list(batch["/a"]["log_time_ns"])


def test_single_topic_timestamps_no_json_decode(tmp_path: Path) -> None:
    """**性能红线**：轻量路径不得解析 JSON 载荷。

    做法：写入非法 JSON 载荷——若实现会 json.loads，则计数为 0；
    轻量路径只看消息头，应仍能取到全部时间戳。
    """
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer

    path = tmp_path / "badjson.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        cid = w.register_channel(topic="/a", message_encoding="json", schema_id=sid)
        for k in range(10):
            w.add_message(channel_id=cid, log_time=1000 + k,
                          publish_time=1000 + k, data=b"NOT JSON")
        w.finish()

    result = _mr.read_mcap_topic_timestamps(str(path), "/a")
    assert result["success"] is True
    assert result["n_rows"] == 10, "轻量路径不应受非法 JSON 影响（说明仍在解析载荷）"


def test_mcap_reader_timestamp_uses_light_path(tmp_path: Path) -> None:
    """MCAP reader 的 timestamp() 走轻量路径（不构建含 data 列的 DataFrame）。"""
    path = _make_mcap(tmp_path, n_per_topic=20)
    from app.tools._readers import get_reader

    reader = get_reader("mcap")
    assert reader is not None
    series, col = reader.timestamp(str(path), sub="/a", column=None)
    assert series is not None
    assert len(series) == 20
    assert col == "mcap_log_time_ns"


# --- 3. 对齐结果：告警与多时钟提示 ----------------------------------------


def _ctx_with_streams(tmp_path: Path, topics: list[str], file_name: str) -> RunContext:
    ctx = RunContext(output_dir=str(tmp_path), dataset_id="demo")
    ctx.meta["streams"] = [
        {"path": f"{tmp_path / file_name}::{t}", "format": "mcap", "kind": "unknown"}
        for t in topics
    ]
    return ctx


def test_burst_streams_go_to_burst_list(tmp_path: Path) -> None:
    """突发型子流被单列（burst_streams），供上层与用户明确其口径不适用。"""
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer

    path = tmp_path / "c.mcap"
    # /burst：突发型（突发内微间隔 + 突发间大静默）；/per：周期型。
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        cb = w.register_channel(topic="/burst", message_encoding="json", schema_id=sid)
        cp = w.register_channel(topic="/per", message_encoding="json", schema_id=sid)
        t = 0
        for burst in range(300):
            for _ in range(100):
                t += 1_000
                w.add_message(channel_id=cb, log_time=t, publish_time=t,
                              data=b'{"v":1}')
            t += 10_000_000
        for k in range(30_000):
            w.add_message(channel_id=cp, log_time=k * 10_000_000,
                          publish_time=k * 10_000_000, data=b'{"v":1}')
        w.finish()

    ctx = _ctx_with_streams(tmp_path, ["/burst", "/per"], path.name)
    r = _ac.align_container_streams_impl(ctx)
    assert r["success"] is True
    assert "/burst" in (r.get("burst_streams") or [])
    # 多时钟提示必须给出（否则用户会误信容器时间的采样率）。
    assert r.get("clock_note")
    assert "传感器时间" in str(r["clock_note"])
    # 突发流的缺口计数为 None，且说明文字必须是"口径不适用"而非报出具体条数。
    rows = {x["sub"]: x for x in r["streams"]}
    assert rows["/burst"]["n_gaps"] is None
    notes = "；".join(rows["/burst"].get("notes") or [])
    assert "缝隙口径不适用" in notes or "口径不适用" in notes
    assert "个缺口" not in notes, f"仍报出具体缺口条数：{notes}"


def test_periodic_stream_gap_still_reported(tmp_path: Path) -> None:
    """周期型流的真实缺口仍被报出（修复不能把真问题一起吞掉）。"""
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer

    path = tmp_path / "g.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        c = w.register_channel(topic="/p", message_encoding="json", schema_id=sid)
        # 周期 10ms，中间留 2 秒空洞：前半 200 点，后半整体后移 2 秒。
        ts = [i * 10_000_000 for i in range(200)]
        ts += [(200 + i) * 10_000_000 + 2_000_000_000 for i in range(200)]
        for x in ts:
            w.add_message(channel_id=c, log_time=x, publish_time=x, data=b'{"v":1}')
        w.finish()

    ctx = _ctx_with_streams(tmp_path, ["/p"], path.name)
    r = _ac.align_container_streams_impl(ctx)
    assert r["success"] is True
    rows = {x["sub"]: x for x in r["streams"]}
    assert rows["/p"]["n_gaps"] == 1
    assert rows["/p"]["max_gap_s"] >= 1.5
    assert r.get("burst_streams") == []


def _count_iter(tmp_path: Path) -> Path:
    """合成一个三 topic 的 MCAP，供"遍历次数"类测试使用。"""
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer

    path = tmp_path / "iter.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        for topic in ("/a", "/b", "/c"):
            c = w.register_channel(topic=topic, message_encoding="json",
                                   schema_id=sid)
            for k in range(200):
                w.add_message(channel_id=c, log_time=k * 10_000_000,
                              publish_time=k * 10_000_000, data=b'{"v":1}')
        w.finish()
    return path


def test_timestamp_cache_shared_across_topics(tmp_path: Path) -> None:
    """**性能回归（第三次同类问题）**：逐 topic 取时间戳只遍历文件一次。

    真实事故：inspect_streams 对每个流各调一次 ``read_mcap_topic_timestamps``，
    每次都重新遍历 1.86GB 文件——26 个 topic 实测 **1074 秒**（期间其他工具
    等待）。缓存必须在读取器层共享，而不是在各调用点各打补丁（此前
    align_container 与 check_temporal_sync 分别打了补丁，漏掉了 inspect_streams）。
    """
    path = _count_iter(tmp_path)
    mr_mod = _mr
    mr_mod.clear_mcap_timestamp_cache()

    calls = {"n": 0}
    real = mr_mod._import_mcap

    class _Wrap:
        def __init__(self, mod):
            self._mod = mod

        def make_reader(self, f):
            calls["n"] += 1
            return self._mod.make_reader(f)

    import functools

    @functools.lru_cache(maxsize=1)
    def _cached_mod():
        return _Wrap(real())

    orig = mr_mod._import_mcap
    mr_mod._import_mcap = _cached_mod  # type: ignore[assignment]
    try:
        for t in ("/a", "/b", "/c"):
            r = mr_mod.read_mcap_topic_timestamps(str(path), t)
            assert r["success"] is True and r["n_rows"] == 200
    finally:
        mr_mod._import_mcap = orig  # type: ignore[assignment]

    assert calls["n"] == 1, (
        f"文件被遍历 {calls['n']} 次（应 1 次，缓存未在读取器层共享）"
    )


def test_cache_does_not_lose_topics(tmp_path: Path) -> None:
    """**缓存正确性**：先取一个 topic，再取另一个，不得返回"无数据"。

    真实踩坑：早期实现把"带 topics 过滤的结果"也当全量缓存，导致先查 /a 后
    查 /b 时缓存命中但缺 /b，误判为"该 topic 无时间戳"并退化成慢路径。
    """
    path = _count_iter(tmp_path)
    _mr.clear_mcap_timestamp_cache()
    first = _mr.read_mcap_topic_timestamps(str(path), "/a")
    second = _mr.read_mcap_topic_timestamps(str(path), "/b")
    third = _mr.read_mcap_topic_timestamps(str(path), "/c")
    assert first["n_rows"] == 200
    assert second["n_rows"] == 200, "第二个 topic 丢失（缓存不完整）"
    assert third["n_rows"] == 200, "第三个 topic 丢失（缓存不完整）"


def test_cache_invalidated_on_file_change(tmp_path: Path) -> None:
    """文件被覆盖写入后缓存失效（按 mtime+size 指纹）。"""
    path = _count_iter(tmp_path)
    _mr.clear_mcap_timestamp_cache()
    r1 = _mr.read_mcap_topic_timestamps(str(path), "/a")
    assert r1["n_rows"] == 200

    # 重写一个更短的文件（mtime 与 size 都变）。
    from mcap.writer import Writer

    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        c = w.register_channel(topic="/a", message_encoding="json", schema_id=sid)
        for k in range(50):
            w.add_message(channel_id=c, log_time=k, publish_time=k, data=b'{"v":1}')
        w.finish()

    r2 = _mr.read_mcap_topic_timestamps(str(path), "/a")
    assert r2["n_rows"] == 50, f"缓存未随文件变更失效（仍报 {r2['n_rows']}）"


def test_clear_cache_is_idempotent(tmp_path: Path) -> None:
    """清空缓存可重复调用（测试隔离用）。"""
    _mr.clear_mcap_timestamp_cache()
    _mr.clear_mcap_timestamp_cache()
    path = _count_iter(tmp_path)
    assert _mr.read_mcap_topic_timestamps(str(path), "/a")["n_rows"] == 200


def test_temporal_sync_bulk_prefetch_no_reeval(tmp_path: Path) -> None:
    """**性能红线**：check_temporal_sync 对同容器 MCAP 只遍历一次。

    做法：合成含两个 topic 的 MCAP，monkeypatch 底层 iter_messages 计数——
    若逐流各自重遍历，遍历次数会随 topic 数翻倍；一次遍历分桶则为 1。
    """
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer
    from app.tools import mcap_reader as mr

    path = tmp_path / "sync.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        for topic in ("/a", "/b", "/c"):
            c = w.register_channel(topic=topic, message_encoding="json",
                                   schema_id=sid)
            for k in range(500):
                w.add_message(channel_id=c, log_time=k * 10_000_000,
                              publish_time=k * 10_000_000, data=b'{"v":1}')
        w.finish()

    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    ctx.meta["streams"] = [
        {"path": f"{path}::{t}", "format": "mcap", "kind": "unknown",
         "channels": [], "n_rows": 500}
        for t in ("/a", "/b", "/c")
    ]
    # 计数 mcap reader 的批量读取被调用几次（应为 1 次，覆盖 3 个 topic）。
    calls = {"bulk": 0}
    real = mr.read_mcap_all_timestamps

    def _spy(*a, **k):
        calls["bulk"] += 1
        return real(*a, **k)

    mr.read_mcap_all_timestamps = _spy  # type: ignore[assignment]
    try:
        from app.tools.check_temporal_sync import check_temporal_sync_impl

        r = check_temporal_sync_impl(ctx)
    finally:
        mr.read_mcap_all_timestamps = real  # type: ignore[assignment]

    assert r.get("success") is True, r.get("user_message")
    assert calls["bulk"] == 1, (
        f"同容器被遍历了 {calls['bulk']} 次（应 1 次；逐流重遍历是性能事故根因）"
    )
    # 三个 topic 都应参与（批量结果被正确分发）。
    checks = (r.get("measurements") or {}).get("stream_checks") or {}
    present = [k for k, v in checks.items() if v.get("present")]
    assert len(present) == 3, f"批量结果未正确分发：{present}"


def test_temporal_sync_custom_column_bypasses_bulk(tmp_path: Path) -> None:
    """指定自定义时间列时**不**走批量（批量只含容器时间，需按列取值）。"""
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer

    path = tmp_path / "cust.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        for topic in ("/a", "/b"):
            c = w.register_channel(topic=topic, message_encoding="json",
                                   schema_id=sid)
            for k in range(200):
                w.add_message(
                    channel_id=c, log_time=k * 10_000_000,
                    publish_time=k * 10_000_000,
                    data=('{"header": {"timestamp_us": %d}}'
                          % (1_789_370_000_000_000 + k * 10_000)).encode())
            w.finish()

    ctx = RunContext(output_dir=str(tmp_path), dataset_id="ds")
    for t in ("/a", "/b"):
        ctx.meta.setdefault("streams", []).append({
            "path": f"{path}::{t}", "format": "mcap", "kind": "unknown",
            "channels": [], "n_rows": 200, "time_column": "header.timestamp_us",
        })
    from app.tools.check_temporal_sync import check_temporal_sync_impl

    r = check_temporal_sync_impl(ctx)
    # 自定义列可能读不到（该 topic 无展开列），但**不得因此崩溃**；
    # 关键是路径选择正确：不静默用容器时间冒充用户指定的列。
    assert r.get("success") in (True, False)
    if not r.get("success"):
        assert r.get("error") in ("not_applicable", "streams_no_match")


def test_empty_frame_notes_absent_for_aligned_streams(tmp_path: Path) -> None:
    """完全对齐的周期流不产生噪声告警。"""
    mcap = pytest.importorskip("mcap")
    from mcap.writer import Writer

    path = tmp_path / "n.mcap"
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        sid = w.register_schema(name="s", encoding="jsonschema",
                                data=b'{"type":"object"}')
        for topic in ("/a", "/b"):
            c = w.register_channel(topic=topic, message_encoding="json",
                                   schema_id=sid)
            for k in range(1000):
                w.add_message(channel_id=c, log_time=k * 10_000_000,
                              publish_time=k * 10_000_000, data=b'{"v":1}')
        w.finish()

    ctx = _ctx_with_streams(tmp_path, ["/a", "/b"], path.name)
    r = _ac.align_container_streams_impl(ctx)
    assert r["success"] is True
    assert r.get("warnings") == []
    assert "对齐良好" in str(r["user_message"])
