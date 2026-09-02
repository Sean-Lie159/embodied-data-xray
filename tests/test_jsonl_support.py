""".jsonl（JSON Lines）原生支持的单元测试。

覆盖全链路：单文件加载 → 目录嗅探登记 → 主表选择 → 概况统计 → 合理性检查，
以及 .jsonl 与 .json 严格区分（lines=True）的纪律。

合成数据刻意包含三类"刁难特征"（对应真实采集数据的常见形态）：
  - 嵌套数组列（如关节角列表）→ object dtype；
  - 嵌套对象列（如 {"mode": ...}）→ object dtype；
  - 缺失字段行 → NaN（非空计数必须能统计）；
  - 时间戳列（纳秒 epoch）→ 供采样率/对齐识别。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.agent.context import RunContext
from app.tools import _data_access, _sniffing
from app.tools.check_sensor_sanity import _constant_columns, check_sensor_sanity_impl
from app.tools.load_dataset import load_dataset_impl
from app.tools.profile_data import profile_data_impl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """把行记录写为 JSONL（每行一个 JSON 对象）。"""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _sample_rows(n: int = 30, *, zero_head: int = 0) -> list[dict]:
    """构造样本行：时间戳 + 嵌套数组 + 嵌套对象 + 数值列（第 5 行缺字段）。"""
    rows: list[dict] = []
    for i in range(n):
        row: dict = {
            "timestamp_ns": 1_700_000_000_000_000_000 + i * 10_000_000,
            "joints": [0.0, 0.0, 0.0] if i < zero_head else [float(i)] * 3,
            "meta": {"mode": "auto", "id": i},
            "accel_x": float(i),
        }
        if i == 5:
            row.pop("accel_x")  # 缺字段 → NaN 行
        rows.append(row)
    return rows


# --- 1. 单文件加载 ----------------------------------------------------------


def test_single_jsonl_loads_with_nested_and_nan(tmp_path: Path) -> None:
    """单文件 .jsonl：嵌套值 → object 列，缺字段 → NaN，行列数正确。"""
    _write_jsonl(tmp_path / "g.jsonl", _sample_rows(30))
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(tmp_path / "g.jsonl"))

    assert r["success"] is True, r.get("reason")
    assert r["format"] == "jsonl"
    assert r["n_rows"] == 30
    assert set(r["columns"]) == {"timestamp_ns", "joints", "meta", "accel_x"}

    df = ctx.df
    assert df is not None
    # 嵌套列表/对象 → object dtype（不被静默展开或丢弃）。
    assert df["joints"].dtype == object
    assert df["meta"].dtype == object
    # 缺失字段行 → NaN，非空计数正确（30 行 - 1 缺失 = 29）。
    assert int(df["accel_x"].isna().sum()) == 1
    assert int(df["accel_x"].notna().sum()) == 29


def test_jsonl_nrows_matches_loaded_rows(tmp_path: Path) -> None:
    """read_table_nrows 的 jsonl 行数与全量装载行数一致（主表评分可信）。"""
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, _sample_rows(12))
    nrows = _data_access.read_table_nrows(str(p), "jsonl")
    assert nrows == 12
    df = _data_access.read_stream_full(str(p), "jsonl")
    assert df is not None and df.shape[0] == nrows


def test_jsonl_skips_malformed_lines_without_crash(tmp_path: Path) -> None:
    """单行非法 JSON：跳过该行而非整体崩溃（真实采集常含截断行）。"""
    p = tmp_path / "g.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write('{"a": 1, "b": 2}\n')
        f.write("这不是 JSON\n")            # 非法行
        f.write("\n")                        # 空行
        f.write('{"a": 3, "b": 4}\n')
    rows = _data_access.read_jsonl_rows(str(p), limit=None)
    assert len(rows) == 2
    assert [r["a"] for r in rows] == [1, 3]
    # 全量装载走 lines=True，不应因非法行崩掉整个加载。
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(p))
    assert r["success"] is True or r["error"] == "parse_failed"  # 不裸抛异常


# --- 2. 目录嗅探：进入表格候选集 -------------------------------------------


def test_jsonl_enters_table_candidates_and_registers(tmp_path: Path) -> None:
    """目录内 26 个 .jsonl 全部登记为流，主表正确选出（不因扩展名被忽略）。"""
    d = tmp_path / "ds"
    d.mkdir()
    for k in range(26):
        _write_jsonl(d / f"g_{k:02d}.jsonl", _sample_rows(30))

    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(d))
    assert r["success"] is True, r.get("reason")

    survey = r["file_survey"]
    assert survey["total_files"] == 26
    # 全部登记为表格流（.jsonl 属表格扩展名，不进标定组、不被当系统文件）。
    streams = ctx.meta["streams"]
    assert len(streams) == 26
    assert all(Path(s["path"]).suffix == ".jsonl" for s in streams)
    # 主表选出且为 .jsonl。
    main = (r.get("main_table") or {}).get("name")
    assert main is not None and main.endswith(".jsonl")


def test_jsonl_is_table_ext_not_calib_ext() -> None:
    """.jsonl 属表格扩展名、不属标定扩展名（避免被当标定候选而漏登记）。"""
    assert ".jsonl" in _sniffing._TABLE_EXTS
    assert ".jsonl" not in _sniffing._CALIB_EXTS


def test_jsonl_columns_sniffed_cheaply(tmp_path: Path) -> None:
    """嗅探期只读首行即得列名（不读全量）。"""
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, _sample_rows(50))
    cols = _sniffing._read_table_columns_cheap(str(p), "jsonl")
    assert cols is not None
    assert set(cols) == {"timestamp_ns", "joints", "meta", "accel_x"}


# --- 3. object 列：profile_data 统计 ---------------------------------------


def test_profile_covers_object_columns(tmp_path: Path) -> None:
    """object 列被 profile_data 统计：dtype 如实、非空计数、样例值齐全。"""
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / "g.jsonl", _sample_rows(30))
    ctx = RunContext(output_dir=str(tmp_path))
    load_dataset_impl(ctx, str(d))

    p = profile_data_impl(ctx)
    assert p["success"] is True, p.get("reason")
    by_name = {c["name"]: c for c in p["columns"]}

    # 嵌套数组列：object dtype、非空计数、样例值均有。
    joints = by_name["joints"]
    assert joints["dtype"] == "object"
    assert joints["n_missing"] == 0
    assert joints["sample_values"], "object 列必须有样例值"
    # 嵌套对象列同样覆盖。
    meta = by_name["meta"]
    assert meta["dtype"] == "object"
    assert meta["sample_values"]
    # 含缺失字段的数值列：缺失被如实计入。
    assert by_name["accel_x"]["n_missing"] == 1


# --- 4. object 列：check_sensor_sanity 全零/恒定检测 ------------------------


def test_constant_detection_covers_nested_object_columns() -> None:
    """嵌套数组列：全零判恒定（掉线）；起始段全零但整体有值不误判。"""
    all_zero = np.array([[0.0, 0.0, 0.0] for _ in range(20)], dtype=object)
    assert "joints" in _constant_columns({"joints": all_zero}, 1e-6)

    partial = np.array(
        [[0.0, 0.0, 0.0] for _ in range(10)] + [[1.0, 2.0, 3.0] for _ in range(10)],
        dtype=object,
    )
    assert "joints" not in _constant_columns({"joints": partial}, 1e-6)


def test_sanity_runs_on_jsonl_stream_and_finds_constant(tmp_path: Path) -> None:
    """端到端：JSONL 目录执行 check_sensor_sanity，恒定通道被如实检出。"""
    d = tmp_path / "ds"
    d.mkdir()
    rows = []
    for i in range(40):
        rows.append({
            "timestamp_ns": 1_700_000_000_000_000_000 + i * 10_000_000,
            "accel_x": 0.1 * i,
            "accel_y": 0.2 * i,
            "accel_z": 9.8,  # 恒定 → 应被判恒定通道
        })
    _write_jsonl(d / "g.jsonl", rows)

    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    s = check_sensor_sanity_impl(ctx)
    assert s["success"] is True, s.get("reason")
    # 恒定通道被检出（JSONL 流的数值通道确实被读取并参与判定）。
    assert "accel_z" in s.get("constant_channels", [])


# --- 5. 纪律：.jsonl 与 .json 严格区分（不得混用）--------------------------


def test_jsonl_and_json_not_interchangeable(tmp_path: Path) -> None:
    """.jsonl 与 .json 分别读取：互按对方格式解析时不得静默产出错误数据。"""
    # JSONL 文件（每行一个对象）。
    jl = tmp_path / "data.jsonl"
    _write_jsonl(jl, [{"a": 1}, {"a": 2}])
    # JSON 文件（整体一个数组）。
    js = tmp_path / "data.json"
    js.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")

    ctx1 = RunContext(output_dir=str(tmp_path))
    r1 = load_dataset_impl(ctx1, str(jl))
    assert r1["success"] is True and r1["format"] == "jsonl" and r1["n_rows"] == 2

    ctx2 = RunContext(output_dir=str(tmp_path))
    r2 = load_dataset_impl(ctx2, str(js))
    assert r2["success"] is True and r2["format"] == "json" and r2["n_rows"] == 2

    # 显式把 JSONL 当 JSON 加载：不是有效 JSON 值 → 结构化失败，不静默给错数据。
    ctx3 = RunContext(output_dir=str(tmp_path))
    r3 = load_dataset_impl(ctx3, str(jl), fmt="json")
    assert r3["success"] is False and r3["error"] == "parse_failed"

    # 反向（JSON 数组当 JSONL）：pandas 的 lines=True 对单行数组较宽容，会产出
    # 1×2 的垃圾表（单元格是 dict）而非正确的 2×1 表。此处固化该已知行为——
    # 断言它**不会**产出看似正确的数据，避免把宽容解析误当作可用结果。
    ctx4 = RunContext(output_dir=str(tmp_path))
    r4 = load_dataset_impl(ctx4, str(js), fmt="jsonl")
    if r4["success"]:
        assert not (r4["n_rows"] == 2 and r4["n_cols"] == 1), (
            "错误格式不得产出与正确解析一致的结果（pandas lines=True 宽容解析）"
        )


def test_jsonl_registered_in_supported_formats() -> None:
    """.jsonl 已登记进 _SUPPORTED_FORMATS（错误提示会列出它）。"""
    from app.tools.load_dataset import _SUPPORTED_FORMATS

    assert ".jsonl" in _SUPPORTED_FORMATS


def test_jsonl_roundtrip_preserves_nested_values(tmp_path: Path) -> None:
    """JSONL 装载后嵌套值结构保持（list/dict 未被压平成字符串或丢失）。"""
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, [{"v": [1.0, 2.0], "o": {"k": "x"}}, {"v": [3.0, 4.0], "o": {"k": "y"}}])
    df = _data_access.read_stream_full(str(p), "jsonl")
    assert df is not None
    first = df["v"].iloc[0]
    assert isinstance(first, (list, np.ndarray)), f"嵌套数组应保持为容器，实际 {type(first)}"
    assert isinstance(df["o"].iloc[0], dict)


def test_jsonl_empty_file_does_not_crash(tmp_path: Path) -> None:
    """空 .jsonl 文件：按空流处理，不崩。"""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    d = tmp_path / "ds"
    d.mkdir()
    _write_jsonl(d / "empty.jsonl", [])
    _write_jsonl(d / "g.jsonl", _sample_rows(10))
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(d))
    assert r["success"] is True, r.get("reason")
    empty = [s for s in ctx.meta["streams"] if Path(s["path"]).name == "empty.jsonl"]
    assert empty and empty[0]["status"] == "empty"


def test_jsonl_frame_index_column_not_treated_as_table(tmp_path: Path) -> None:
    """回归护栏：JSONL 的 object 列不参与数值统计分支（无 numeric 字段但不报错）。"""
    p = tmp_path / "g.jsonl"
    _write_jsonl(p, _sample_rows(10))
    df = pd.read_json(p, lines=True)
    # object 列走向量/非零分支，不进入数值描述统计，且整体流程不抛异常。
    from app.tools.profile_data import _vector_column_stats

    stats = _vector_column_stats(df["joints"])
    assert stats is None or isinstance(stats, dict)
