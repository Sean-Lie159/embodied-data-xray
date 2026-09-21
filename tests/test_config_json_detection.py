"""配置型 JSON 识别测试（**真实缺陷回归**，2026-09-21）。

事故：Forsense-G7 手套数据的 ``session.json``（322 字节、9 个字段）被
``classify_table_stream`` 判为「未使用/空流」、``status="empty"``，
导致关键录制参数（``nominal_hz``=120、``hand_mode``=both）对下游完全不可见，
模型只能看到"空"——用户随即质疑"明明有内容，为何显示空"。

根因链：
1. ``_JsonReader.columns`` 对"顶层是 dict 但无行列表键"的 JSON 返回 ``[]``；
2. ``read_table_nrows`` 走 ``_json_row_list`` → 返回 0 行；
3. ``classify_table_stream`` 的"空流检测"（``nrows <= 2``）据此判定为空流。

修复要点：**区分"没有内容"与"不是逐行数据"**——
有顶层键但 0 行 → 配置型 JSON（有内容）；无键且 0 行 → 真·空流。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.tools._readers import ReadRequest, read_stream
from app.tools._sniffing import classify_table_stream
from app.tools.load_dataset import _compact_config, _read_json_object


# 用户实际提供的 session.json 内容。
SESSION = {
    "delay_ns": 25000000,
    "files": [
        "joints_position.csv",
        "joints_orientation.csv",
        "tips_trajectory.csv",
    ],
    "format_version": "3",
    "hand_mode": "both",
    "nominal_hz": 120,
    "position_decimal_places": 6,
    "quaternion_decimal_places": 6,
    "started_wall_local": "2026-09-21T15:19:03",
}


@pytest.fixture
def session_json(tmp_path: Path) -> Path:
    p = tmp_path / "session.json"
    p.write_text(json.dumps(SESSION, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 列名读取层
# ---------------------------------------------------------------------------


def test_json_reader_columns_returns_top_level_keys(session_json: Path) -> None:
    """配置型 JSON 的列名读取须返回顶层键（而非空列表）。"""
    res = read_stream(ReadRequest(path_spec=str(session_json), want="columns"))
    assert res.ok is True
    assert res.columns == list(SESSION.keys())


def test_explicit_empty_data_json_still_returns_no_columns(
    tmp_path: Path,
) -> None:
    """**显式空数据文件**（含行列表键但为空）仍返回空列名。

    这类文件是在声明"没有数据行"（如 ``{"data": []}``），不属配置——
    否则会把空数据表误标成配置（实测：格式骨架里的 schema/camera JSON）。
    """
    p = tmp_path / "schema.json"
    p.write_text(json.dumps({"data": []}), encoding="utf-8")
    res = read_stream(ReadRequest(path_spec=str(p), want="columns"))
    assert res.ok is True
    assert res.columns == []


# ---------------------------------------------------------------------------
# 分类层
# ---------------------------------------------------------------------------


def test_config_json_classified_as_config_not_empty(session_json: Path) -> None:
    """配置型 JSON 须被标为 config（而非 empty），且透出顶层键。"""
    obj = json.loads(session_json.read_text(encoding="utf-8"))
    cols = list(obj.keys())
    res = classify_table_stream("session.json", cols, None, 0, fmt="json")

    assert res["kind"] == "config"
    assert res["status"] == "config"
    assert res["semantic_label"] == "配置文件（非逐行数据）"
    # 关键：键必须透出，否则下游仍看不到内容。
    assert res["config_keys"] == list(SESSION.keys())
    # 证据须可转述。
    assert "nominal_hz" in res["label_evidence"]


def test_config_json_evidence_is_informative(session_json: Path) -> None:
    """证据文字须说明"为何不是空流"（否则用户仍会困惑）。"""
    obj = json.loads(session_json.read_text(encoding="utf-8"))
    res = classify_table_stream(
        "session.json", list(obj.keys()), None, 0, fmt="json")
    ev = res["label_evidence"]
    assert "单个 JSON 对象" in ev
    assert "不是逐行表格" in ev or "非逐行" in ev


def test_csv_with_few_rows_still_empty(tmp_path: Path) -> None:
    """**回归守护**：只有表头/极少行的 CSV 仍判空流，不得误判为配置。

    此前若只按"有列名"判定，2 行的 CSV 会被误标为配置文件
    （实测 `controller_poses.csv` 出现该回归）。
    """
    res = classify_table_stream(
        "controller_poses.csv", ["pose_x", "pose_y"], None, 1, fmt="csv")
    assert res["status"] == "empty"
    assert res["kind"] == "unknown"


def test_csv_with_few_rows_and_no_fmt_hint_still_empty() -> None:
    """未给 fmt 提示时，按扩展名判定（.csv 不进入 config 分支）。"""
    res = classify_table_stream(
        "controller_poses.csv", ["pose_x", "pose_y"], None, 1)
    assert res["status"] == "empty"


def test_empty_json_without_keys_still_empty(tmp_path: Path) -> None:
    """空对象 JSON（``{}``）无键无行 → 仍判空流。"""
    res = classify_table_stream("empty.json", [], None, 0, fmt="json")
    assert res["status"] == "empty"
    assert res["kind"] == "unknown"


def test_json_with_row_list_key_but_empty_is_empty() -> None:
    """含行列表键但为空的 JSON（``{"data": []}``）判空流，不判配置。"""
    res = classify_table_stream(
        "schema.json", [], None, 0, fmt="json")
    assert res["status"] == "empty"


# ---------------------------------------------------------------------------
# 配置值透出层
# ---------------------------------------------------------------------------


def test_read_json_object_extracts_dict(session_json: Path) -> None:
    """能读出顶层 dict。"""
    obj = _read_json_object(session_json)
    assert obj is not None
    assert obj["nominal_hz"] == 120
    assert obj["hand_mode"] == "both"


def test_read_json_object_returns_none_for_list(tmp_path: Path) -> None:
    """顶层是数组时返回 None（不当作配置）。"""
    p = tmp_path / "arr.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert _read_json_object(p) is None


def test_read_json_object_returns_none_for_invalid(tmp_path: Path) -> None:
    """非法 JSON 返回 None（不抛异常）。"""
    p = tmp_path / "bad.json"
    p.write_text("{ 不是合法 JSON", encoding="utf-8")
    assert _read_json_object(p) is None


def test_compact_config_keeps_scalars_and_lists() -> None:
    """标量与短列表原样保留（这是配置的价值所在）。"""
    out = _compact_config(SESSION)
    assert out["nominal_hz"] == 120
    assert out["hand_mode"] == "both"
    assert out["delay_ns"] == 25000000
    assert out["files"] == SESSION["files"]


def test_compact_config_truncates_long_values() -> None:
    """超长字符串/列表被截断并标注（防撑爆上下文）。"""
    cfg = {
        "long_str": "x" * 500,
        "long_list": list(range(100)),
        "nested": {"a": 1, "b": 2},
        "deep_list": [{"a": 1}],
    }
    out = _compact_config(cfg)
    assert out["long_str"].endswith("…")
    assert len(out["long_str"]) <= 201
    assert any("共 100 项" in str(x) for x in out["long_list"])
    assert "2 个键" in out["nested"]
    assert "嵌套列表" in out["deep_list"]


def test_compact_config_caps_key_count() -> None:
    """键数超上限时标注截断数量（不静默丢键）。"""
    cfg = {f"k{i}": i for i in range(100)}
    out = _compact_config(cfg)
    assert out["_truncated_keys"] == 100 - 60


# ---------------------------------------------------------------------------
# 端到端：目录加载与设备清单
# ---------------------------------------------------------------------------


def test_directory_load_surfaces_config_values(tmp_path: Path) -> None:
    """端到端：目录加载后，session.json 的配置值必须对下游可见。

    这是本次事故的核心验收点——用户之所以质疑，正是因为
    ``nominal_hz`` 等参数在工具返回里根本看不到。
    """
    import numpy as np
    import pandas as pd

    from app.agent.context import RunContext
    from app.tools.load_dataset import load_dataset_impl

    d = tmp_path / "20260921_151733"
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps(SESSION, indent=2), encoding="utf-8")

    n = 50
    t = np.arange(n) * (1e9 / 120)
    pd.DataFrame({
        "timestamp_ns": t.astype("int64"),
        "L_j00_quat_w": np.sin(np.arange(n) * 0.1),
        "L_j00_quat_x": np.cos(np.arange(n) * 0.1),
        "R_j00_quat_w": np.sin(np.arange(n) * 0.12),
        "R_j00_quat_x": np.cos(np.arange(n) * 0.12),
    }).to_csv(d / "joints_orientation.csv", index=False)

    ctx = RunContext()
    res = load_dataset_impl(ctx, str(d))
    assert res.get("success") is True

    entry = next(
        (ti for ti in res["table_info"] if ti["name"] == "session.json"), None)
    assert entry is not None, "session.json 未出现在表格清单中"
    # 不得再被判为空流。
    assert entry["sniff"]["status"] == "config"
    assert entry["sniff"]["semantic_label"] == "配置文件（非逐行数据）"
    # 关键：配置值必须可见。
    assert entry["config"]["nominal_hz"] == 120
    assert entry["config"]["hand_mode"] == "both"


def test_inspect_streams_lists_config_files_separately(tmp_path: Path) -> None:
    """设备清单须把配置文件与空流分开列出（语义相反，不可混）。"""
    import numpy as np
    import pandas as pd

    from app.agent.context import RunContext
    from app.tools.inspect_streams import inspect_streams_impl
    from app.tools.load_dataset import load_dataset_impl

    d = tmp_path / "ds"
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps(SESSION), encoding="utf-8")
    n = 50
    pd.DataFrame({
        "timestamp_ns": (np.arange(n) * 1e6).astype("int64"),
        "L_j00_quat_w": np.sin(np.arange(n) * 0.1),
        "L_j00_quat_x": np.cos(np.arange(n) * 0.1),
        "R_j00_quat_w": np.sin(np.arange(n) * 0.12),
        "R_j00_quat_x": np.cos(np.arange(n) * 0.12),
    }).to_csv(d / "joints_orientation.csv", index=False)

    ctx = RunContext()
    load_dataset_impl(ctx, str(d))
    ir = inspect_streams_impl(ctx)

    cfg_names = {Path(c["source"]).name for c in ir.get("config_files", [])}
    empty_names = {Path(s["source"]).name for s in ir.get("empty_streams", [])}
    assert "session.json" in cfg_names
    assert "session.json" not in empty_names, "配置文件不得再被列为空流"
    assert ir["summary"]["n_config_files"] == 1
    # 用户消息须点明"不是空流"。
    assert "不是空流" in ir["user_message"]
