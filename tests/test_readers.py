"""统一读取注册表（_readers.py）的契约测试。

背景：项目的格式分派点散落 15 处（7 文件）、复合路径特判 4 处、同一能力
2~3 份重复实现——这是"工具层总是暴露边界"的机制性根因。本模块把"能不能读"
收敛为 read_stream 单一入口 + register_reader 插件。

本测试锁定三件事：
  1. 六类内置格式经统一入口读取的行为与既有实现一致（零回归前提）；
  2. 复合路径 ``"<file>::<sub>"`` 的解析与分派只有一处（split_path_spec）；
  3. **抽象成立性**：新格式只需 register_reader，不改任何工具即可被支持。
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from app.tools._readers import (
    ReadRequest,
    ReadResult,
    get_reader,
    read_stream,
    register_reader,
    resolve_fmt,
    split_path_spec,
    supported_formats,
)

T0_MS = 1_788_426_767_625


# --- 1. 路径解析与格式推断 --------------------------------------------------


def test_split_path_spec() -> None:
    """复合路径拆分：唯一解析点（h5 节点 / mcap topic 共用）。"""
    assert split_path_spec("a/b.csv") == ("a/b.csv", None)
    assert split_path_spec("a/d.h5::pose/x/y") == ("a/d.h5", "pose/x/y")
    assert split_path_spec("a/x.mcap::/imu/data") == ("a/x.mcap", "/imu/data")
    # 空 sub 视为无子流。
    assert split_path_spec("a/b.csv::") == ("a/b.csv", None)


def test_resolve_fmt_variants() -> None:
    """格式推断：显式 fmt（可带点）优先；否则按扩展名；未知返回空串。"""
    assert resolve_fmt("x.csv") == "csv"
    assert resolve_fmt("x.jsonl") == "jsonl"
    assert resolve_fmt("x.h5::node") == "h5"
    assert resolve_fmt("x.mcap::/t") == "mcap"
    assert resolve_fmt("x.csv", "CSV") == "csv"
    assert resolve_fmt("x.dat", ".parquet") == "parquet"
    assert resolve_fmt("x.unknown") == ""


def test_supported_formats_contains_six() -> None:
    """六类内置 reader 均已注册。"""
    fmts = supported_formats()
    for f in ("csv", "parquet", "json", "jsonl", "h5", "mcap"):
        assert f in fmts, f"{f} 未注册"


# --- 2. 内置格式经统一入口读取 ----------------------------------------------


@pytest.fixture()
def mixed_ds(tmp_path: Path) -> Path:
    """建含 csv/json/jsonl/parquet/h5 的目录（mcap 由专用测试覆盖）。"""
    d = tmp_path / "ds"
    d.mkdir()
    pd.DataFrame({"a": [1, 2, 3], "ts": [10, 20, 30]}).to_csv(d / "t.csv", index=False)
    (d / "t.json").write_text(json.dumps([{"a": 1, "ts": 10}, {"a": 2, "ts": 20}]),
                              encoding="utf-8")
    with (d / "t.jsonl").open("w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"a": i, "ts": i * 10}) + "\n")
    pd.DataFrame({"a": [1, 2], "ts": [5, 6]}).to_parquet(d / "t.parquet")
    n = 20
    comp = np.zeros(n, dtype=[("value", "<f4"), ("timestamp", "<f8")])
    comp["timestamp"] = T0_MS + np.arange(n) * 10.0
    with h5py.File(d / "t.h5", "w") as hf:
        hf.create_group("action/feedback").create_dataset("motor_command", data=comp)
    return d


@pytest.mark.parametrize("fname,want_rows", [
    ("t.csv", 3), ("t.json", 2), ("t.jsonl", 4), ("t.parquet", 2),
])
def test_frame_reads_all_table_formats(mixed_ds: Path, fname: str, want_rows: int) -> None:
    """frame：四类表格格式统一入口读取（行数正确）。"""
    r = read_stream(ReadRequest(path_spec=str(mixed_ds / fname), want="frame"))
    assert r.ok is True, r.reason
    assert r.frame.shape[0] == want_rows
    assert r.fmt == fname.rsplit(".", 1)[1]


@pytest.mark.parametrize("fname,want", [
    ("t.csv", ["a", "ts"]), ("t.json", ["a", "ts"]),
    ("t.jsonl", ["a", "ts"]), ("t.parquet", ["a", "ts"]),
])
def test_columns_reads_all_table_formats(mixed_ds: Path, fname: str, want: list[str]) -> None:
    """columns：四类表格格式列名读取（此前有 2 份重复实现）。"""
    r = read_stream(ReadRequest(path_spec=str(mixed_ds / fname), want="columns"))
    assert r.ok is True, r.reason
    assert set(want).issubset(set(r.columns))


def test_nrows_all_table_formats(mixed_ds: Path) -> None:
    """nrows：行数读取（与 frame 行数一致）。"""
    for fname, want in (("t.csv", 3), ("t.json", 2), ("t.jsonl", 4), ("t.parquet", 2)):
        r = read_stream(ReadRequest(path_spec=str(mixed_ds / fname), want="nrows"))
        assert r.ok is True, fname
        assert r.nrows == want, fname


def test_sample_limit(mixed_ds: Path) -> None:
    """sample：limit 生效（前 N 行）。"""
    r = read_stream(ReadRequest(path_spec=str(mixed_ds / "t.csv"), want="sample",
                                limit=2))
    assert r.ok is True and r.frame.shape[0] == 2


def test_expand_flag(mixed_ds: Path) -> None:
    """expand：信封展开（jsonl data 列 → 点分列）。"""
    p = mixed_ds / "env.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"t": i, "data": {"v": i * 2}}) + "\n")
    r = read_stream(ReadRequest(path_spec=str(p), want="frame", expand=True))
    assert r.ok is True
    assert "data.v" in r.frame.columns


# --- 3. 容器型：子流清单与按子流读取 ----------------------------------------


def test_h5_candidates_and_sub_read(mixed_ds: Path) -> None:
    """h5：candidates 列子流；按 ``::node`` 读取（复合路径统一解析）。"""
    h5p = str(mixed_ds / "t.h5")
    r = read_stream(ReadRequest(path_spec=h5p, want="candidates"))
    assert r.ok is True
    assert "action/feedback/motor_command" in r.candidates

    r2 = read_stream(ReadRequest(
        path_spec=f"{h5p}::action/feedback/motor_command", want="frame"))
    assert r2.ok is True
    assert r2.frame.shape[0] == 20
    assert r2.sub == "action/feedback/motor_command"


def test_h5_sub_timestamp(mixed_ds: Path) -> None:
    """h5 子流时间戳：compound timestamp 字段（此前有独立特判实现）。"""
    h5p = str(mixed_ds / "t.h5")
    r = read_stream(ReadRequest(
        path_spec=f"{h5p}::action/feedback/motor_command", want="timestamp"))
    assert r.ok is True
    assert r.timestamp_column == "timestamp"
    assert len(r.timestamp) == 20


def test_h5_nrows_and_columns(mixed_ds: Path) -> None:
    """h5 子流行数 / 列名。"""
    h5p = str(mixed_ds / "t.h5")
    sub = f"{h5p}::action/feedback/motor_command"
    assert read_stream(ReadRequest(path_spec=sub, want="nrows")).nrows == 20
    assert set(read_stream(ReadRequest(path_spec=sub, want="columns")).columns) == {
        "value", "timestamp"}


# --- 4. 错误语义 ------------------------------------------------------------


def test_unsupported_format() -> None:
    """未注册格式 → unsupported_format（不是崩溃）。"""
    r = read_stream(ReadRequest(path_spec="x.unknownfile", want="frame"))
    assert r.ok is False and r.error == "unsupported_format"


def test_missing_file() -> None:
    """文件不存在 → read_failed。"""
    r = read_stream(ReadRequest(path_spec="no/such/file.csv", want="frame"))
    assert r.ok is False and r.error == "read_failed"


def test_non_container_candidates_empty(mixed_ds: Path) -> None:
    """非容器格式的 candidates 为空清单（不是错误）。"""
    r = read_stream(ReadRequest(path_spec=str(mixed_ds / "t.csv"), want="candidates"))
    assert r.ok is True and r.candidates == []


# --- 5. 抽象成立性：新格式只需注册 ------------------------------------------


def test_registering_new_format_needs_no_tool_change(tmp_path: Path) -> None:
    """**抽象成立性演练**：注册一个假格式 reader，无需改任何工具即可读。

    这是本重构的核心验收——新格式（如 rosbag2）= 注册一个 reader。
    """
    class _FakeReader:
        fmt = "fake"
        extensions = [".fake"]  # 注册时自动并入格式推断表

        def frame(self, path, *, sub, limit):
            return pd.DataFrame({"x": [1, 2, 3], "fake_ts": [0.1, 0.2, 0.3]})

        def columns(self, path, *, sub):
            return ["x", "fake_ts"]

        def nrows(self, path, *, sub):
            return 3

        def sub_streams(self, path):
            return ["s1", "s2"]

        def timestamp(self, path, *, sub, column):
            return (np.array([0.1, 0.2, 0.3]), "fake_ts")

    p = tmp_path / "x.fake"
    p.write_bytes(b"whatever")
    register_reader(_FakeReader())
    try:
        assert "fake" in supported_formats()
        assert get_reader("fake") is not None
        r = read_stream(ReadRequest(path_spec=str(p), want="frame"))
        assert r.ok is True and r.fmt == "fake"
        assert r.frame.shape == (3, 2)
        assert read_stream(ReadRequest(path_spec=str(p), want="nrows")).nrows == 3
        assert read_stream(ReadRequest(path_spec=str(p), want="candidates")).candidates == [
            "s1", "s2"]
        ts = read_stream(ReadRequest(path_spec=str(p), want="timestamp"))
        assert ts.ok is True and ts.timestamp_column == "fake_ts"
    finally:
        from app.tools import _readers

        _readers._REGISTRY.pop("fake", None)


def test_read_result_fail_factory() -> None:
    """ReadResult.fail 统一构造失败结果。"""
    r = ReadResult.fail("read_failed", "原因", fmt="csv", sub="s")
    assert r.ok is False and r.error == "read_failed" and r.reason == "原因"
    assert r.fmt == "csv" and r.sub == "s"
