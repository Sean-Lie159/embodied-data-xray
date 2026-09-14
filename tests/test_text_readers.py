"""纯文本数据读取器（.txt 时间戳清单 / .INFO 日志）与目录登记测试。

背景（用户实测 2026-09-14）：数据集 2655849 的时间戳**就在**
``camera/<相机>/<相机>.txt`` 里（14142 行 ``<纳秒时间戳> <帧状态>``、30 Hz、
跨度 471 秒），但 ``.txt`` 不在支持格式内，目录加载只把路径归入 ``others``
且**从不打开**——agent 只能回答"工具根本没看这些文件"（诚实但存在能力缺口）。

本次扩展：新增 txt/log 两种读取器（注册进统一读取表，不改各工具），并在目录
加载时"先打开确认、再登记为流"，使现有时间对齐能力直接可用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.agent.context import RunContext
from app.tools._readers import ReadRequest, read_stream, supported_formats
from app.tools._text_readers import parse_log_lines, parse_timestamp_lines

# --- 1. 时间戳清单解析 ------------------------------------------------------


def test_parse_timestamp_with_status(tmp_path: Path) -> None:
    """``<时间戳> <状态字母>`` 形态：首列识别为 timestamp，第二列识别为 status。"""
    f = tmp_path / "cam.txt"
    lines = [f"{1756265284737924649 + i * 33_333_333} P" for i in range(20)]
    lines[0] = lines[0].replace(" P", " I")
    f.write_text("\n".join(lines), encoding="utf-8")

    df = parse_timestamp_lines(str(f))
    assert df is not None
    assert list(df.columns) == ["timestamp", "status"]
    assert df.shape == (20, 2)
    assert df["timestamp"].iloc[0] == 1756265284737924649
    assert set(df["status"].unique()) == {"I", "P"}


def test_parse_timestamp_with_frame_index(tmp_path: Path) -> None:
    """``<时间戳> <递增整数>`` 形态：第二列识别为 frame_index。"""
    f = tmp_path / "depth.txt"
    lines = [f"{1756265284805200809 + i * 33_333_333} {1235886 + i}" for i in range(10)]
    f.write_text("\n".join(lines), encoding="utf-8")

    df = parse_timestamp_lines(str(f))
    assert df is not None
    assert list(df.columns) == ["timestamp", "frame_index"]


def test_parse_rejects_prose(tmp_path: Path) -> None:
    """**拒读**：说明文本（首列非数值）不得被解析为数据表。"""
    f = tmp_path / "readme.txt"
    f.write_text("这是一个说明文件\n没有数值列\n", encoding="utf-8")
    assert parse_timestamp_lines(str(f)) is None


def test_parse_rejects_keyvalue_config(tmp_path: Path) -> None:
    """**拒读**：键值配置（YAML 风格）不得被解析为数据表。"""
    f = tmp_path / "cfg.txt"
    f.write_text("type: G2A\nAID: X1\nstatus: ''\n", encoding="utf-8")
    assert parse_timestamp_lines(str(f)) is None


def test_parse_rejects_ragged_rows(tmp_path: Path) -> None:
    """**拒读**：列数严重不一致的文件不猜（不产出错误的表）。"""
    f = tmp_path / "ragged.txt"
    rows = ["1000000000 a", "1000000001 a b c d e", "1000000002 a"]
    f.write_text("\n".join(rows), encoding="utf-8")
    assert parse_timestamp_lines(str(f)) is None


def test_parse_skips_blank_lines(tmp_path: Path) -> None:
    """空行被跳过而非导致失败。"""
    f = tmp_path / "with_blank.txt"
    f.write_text("1000000000 P\n\n1000000001 P\n\n", encoding="utf-8")
    df = parse_timestamp_lines(str(f))
    assert df is not None and df.shape[0] == 2


def test_parse_limit_is_honored(tmp_path: Path) -> None:
    """limit 生效（大文件按需读取）。"""
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"{10**18 + i} P" for i in range(100)), encoding="utf-8")
    df = parse_timestamp_lines(str(f), limit=10)
    assert df is not None and df.shape[0] == 10


def test_parse_does_not_invent_column_names(tmp_path: Path) -> None:
    """**不臆造语义**：非时间戳量级的首列不得命名为 timestamp。

    真实风险：若把任意数值列都叫 timestamp，会让它被错当作时间轴参与对齐。
    """
    f = tmp_path / "small.txt"
    f.write_text("\n".join(f"{i} {i * 2}" for i in range(20)), encoding="utf-8")
    df = parse_timestamp_lines(str(f))
    assert df is not None
    assert "timestamp" not in df.columns, list(df.columns)


# --- 2. 日志解析（glog 风格）---------------------------------------------


def _glog(level: str, md: str, hms: str, frac: str, tid: int, src: str, msg: str) -> str:
    return f"{level}{md} {hms}.{frac} {tid} {src}] {msg}"


def test_parse_log_lines_basic(tmp_path: Path) -> None:
    """glog 行被解析为结构化表（时间键/级别/线程/来源/消息）。"""
    f = tmp_path / "hal.INFO"
    f.write_text("\n".join([
        "Log file created at: 2026/03/20 09:37:20",
        _glog("I", "0320", "09:37:20", "200228", 610336,
              "dylog_impl.cpp:641", "[DYLOG] set level to INFO"),
        _glog("W", "0320", "09:37:21", "123456", 610337,
              "foo.cpp:12", "some warning"),
        _glog("E", "0320", "09:37:22", "000001", 610338, "bar.cpp:9", "an error"),
    ]), encoding="utf-8")

    df = parse_log_lines(str(f))
    assert df is not None
    assert list(df.columns) == ["time_of_day_us", "level", "thread", "source", "message"]
    assert set(df["level"].dropna().unique()) == {"I", "W", "E"}
    # 非日志行保留在 message 中（不丢弃信息），level 为空。
    assert df["level"].isna().sum() == 1


def test_parse_log_time_of_day_is_sortable(tmp_path: Path) -> None:
    """时间键可排序、可算间隔（当日微秒数）。"""
    f = tmp_path / "t.INFO"
    f.write_text("\n".join([
        _glog("I", "0320", "09:00:00", "000000", 1, "a.cc:1]", "first"),
        _glog("I", "0320", "09:00:01", "500000", 1, "a.cc:2]", "second"),
    ]), encoding="utf-8")
    df = parse_log_lines(str(f))
    assert df is not None
    t = df["time_of_day_us"].dropna().to_numpy()
    assert len(t) == 2
    assert np.all(np.diff(t) > 0)
    # 1.5 秒 = 1_500_000 微秒
    assert int(t[1] - t[0]) == 1_500_000


def test_parse_log_declares_year_missing(tmp_path: Path) -> None:
    """**诚实标注**：日志时间为当日时刻（glog 不含年份），不得冒充绝对时间戳。"""
    f = tmp_path / "t.INFO"
    f.write_text(_glog("I", "0320", "09:00:00", "000000", 1, "a.cc:1]", "m"),
                 encoding="utf-8")
    df = parse_log_lines(str(f))
    assert df is not None
    assert df.attrs.get("clock_scope") == "time_of_day"
    assert "不含年份" in str(df.attrs.get("clock_note"))
    # 列名本身也应体现"当日时刻"而非 timestamp。
    assert "time_of_day_us" in df.columns


def test_parse_log_rejects_non_log(tmp_path: Path) -> None:
    """**拒读**：不含任何 glog 行的文本不解析。"""
    f = tmp_path / "plain.INFO"
    f.write_text("hello world\njust text\n", encoding="utf-8")
    assert parse_log_lines(str(f)) is None


# --- 3. 读取器注册与统一入口 ----------------------------------------------


def test_formats_registered() -> None:
    """txt / log 已注册进统一读取表（新增格式=注册一次）。"""
    fmts = supported_formats()
    assert "txt" in fmts and "log" in fmts


def test_read_stream_txt_frame(tmp_path: Path) -> None:
    """经统一入口读 txt 全表。"""
    f = tmp_path / "a.txt"
    f.write_text("\n".join(f"{10**18 + i * 33333333} P" for i in range(10)),
                 encoding="utf-8")
    r = read_stream(ReadRequest(path_spec=str(f), want="frame"))
    assert r.ok and r.frame is not None
    assert r.fmt == "txt"
    assert "timestamp" in r.frame.columns


def test_read_stream_txt_timestamp(tmp_path: Path) -> None:
    """经统一入口直取时间戳（这是接入时间对齐链路的关键）。"""
    f = tmp_path / "b.txt"
    f.write_text("\n".join(f"{10**18 + i * 33333333} P" for i in range(10)),
                 encoding="utf-8")
    r = read_stream(ReadRequest(path_spec=str(f), want="timestamp"))
    assert r.ok and r.timestamp is not None
    assert r.timestamp_column == "timestamp"
    assert len(r.timestamp) == 10


def test_read_stream_txt_columns_and_nrows(tmp_path: Path) -> None:
    """列名与行数读取可用。"""
    f = tmp_path / "c.txt"
    f.write_text("\n".join(f"{10**18 + i} P" for i in range(7)), encoding="utf-8")
    rc = read_stream(ReadRequest(path_spec=str(f), want="columns"))
    rn = read_stream(ReadRequest(path_spec=str(f), want="nrows"))
    assert rc.ok and "timestamp" in rc.columns
    assert rn.ok and rn.nrows == 7


def test_read_stream_log_timestamp(tmp_path: Path) -> None:
    """日志的时间键可经统一入口取得（列名为 time_of_day_us，非绝对时间）。"""
    f = tmp_path / "d.INFO"
    f.write_text(_glog("I", "0320", "09:00:00", "000000", 1, "a.cc:1]", "m"),
                 encoding="utf-8")
    r = read_stream(ReadRequest(path_spec=str(f), want="timestamp"))
    assert r.ok and r.timestamp is not None
    assert r.timestamp_column == "time_of_day_us"


def test_read_stream_rejects_bad_txt(tmp_path: Path) -> None:
    """非数据表文本经统一入口读取失败（不返回误导性的空表）。"""
    f = tmp_path / "prose.txt"
    f.write_text("这是说明\n不是数据\n", encoding="utf-8")
    r = read_stream(ReadRequest(path_spec=str(f), want="frame"))
    assert r.ok is False


# --- 4. 目录加载：登记为流 ------------------------------------------------


def _make_dataset(tmp_path: Path) -> Path:
    """构造一个含 camera/*.txt 与 logs/*.INFO 的最小数据集。"""
    ds = tmp_path / "ds"
    (ds / "camera" / "cam_a").mkdir(parents=True)
    (ds / "camera" / "cam_b").mkdir(parents=True)
    (ds / "logs").mkdir(parents=True)
    for i, cam in enumerate(("cam_a", "cam_b")):
        rows = [f"{1756265284737924649 + k * 33_333_333 + i * 1000} P"
                for k in range(60)]
        (ds / "camera" / cam / f"{cam}.txt").write_text(
            "\n".join(rows), encoding="utf-8")
    (ds / "logs" / "hal.INFO").write_text("\n".join([
        "Log file created at: 2026/03/20 09:37:20",
        _glog("I", "0320", "09:37:20", "200228", 610336, "a.cc:1]", "started"),
        _glog("W", "0320", "09:37:21", "1", 610337, "b.cc:2]", "warn"),
    ]), encoding="utf-8")
    return ds


def test_directory_registers_text_streams(tmp_path: Path) -> None:
    """**核心回归**：目录内的 txt/INFO 被登记为流（此前只列路径、从不打开）。"""
    from app.tools.load_dataset import load_dataset_impl

    ds = _make_dataset(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    result = load_dataset_impl(ctx, str(ds))
    assert result["success"] is True

    streams = ctx.meta.get("streams") or []
    txt_streams = [s for s in streams if s.get("format") == "txt"]
    log_streams = [s for s in streams if s.get("format") == "log"]
    assert len(txt_streams) == 2, f"txt 流未登记：{[s.get('path') for s in streams]}"
    assert len(log_streams) == 1
    # 登记条目须含时间列与实测采样率（供 UI 与对齐分析直接用）。
    assert all(s.get("time_column") for s in txt_streams)
    assert all(s.get("n_rows") == 60 for s in txt_streams)


def test_registered_text_stream_is_alignable(tmp_path: Path) -> None:
    """**端到端**：登记后的 txt 流能被时间同步检查读取并参与对齐。

    这是本次扩展的价值证明——时间戳从"工具根本没看"变成"可用于对齐判定"。
    """
    from app.tools.check_temporal_sync import check_temporal_sync_impl
    from app.tools.load_dataset import load_dataset_impl

    ds = _make_dataset(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(ds))
    res = check_temporal_sync_impl(ctx)
    assert res.get("success") is True, res.get("user_message")
    checks = (res.get("measurements") or {}).get("stream_checks") or {}
    present = [k for k, v in checks.items() if v.get("present")]
    assert len(present) >= 2, f"可对齐流不足：{present}"
    # 两路相机的采样率应被实测出来（约 30 Hz）。
    for k in present:
        if k.endswith(".txt"):
            assert checks[k].get("actual_rate_hz") == pytest.approx(30.0, rel=0.05)


def test_prose_txt_not_registered(tmp_path: Path) -> None:
    """**不误登记**：目录里的说明文本不得被当成数据流。"""
    from app.tools.load_dataset import load_dataset_impl

    ds = tmp_path / "ds2"
    ds.mkdir()
    (ds / "readme.txt").write_text("说明文档\n无数据\n", encoding="utf-8")
    (ds / "data.txt").write_text(
        "\n".join(f"{10**18 + i} P" for i in range(30)), encoding="utf-8")
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(ds))
    streams = ctx.meta.get("streams") or []
    names = [Path(s["path"]).name for s in streams if s.get("format") == "txt"]
    assert "data.txt" in names
    assert "readme.txt" not in names, "说明文本被误登记为数据流"
