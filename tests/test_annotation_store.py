"""annotation_store 测试：锚点解析（三格式）、读写、来源分级、版本管理。

设计依据：``docs/标注与质检能力设计.md`` 第 4 节。

覆盖要点：
- 三种目标格式（LeRobot v2 / 多流采集目录 / HDF5）各自解析出正确锚点；
- 无 episode 线索时诚实降级为整段单元（**不猜**）；
- JSONL 追加/更新去重/删除；
- 来源分级与用途合规校验（``llm_proposed`` 禁止进训练真值）；
- 版本管理：修订日志记 before/after、快照冻结、diff 三分类；
- 时间戳格式化与解析互逆。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools import annotation_store as ann


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def output_dir(tmp_path: Path) -> str:
    """独立的输出目录（避免污染真实 outputs/）。"""
    d = tmp_path / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _ctx_with_df(df: pd.DataFrame, dataset_id: str = "demo", **meta) -> RunContext:
    """构造带主表的 RunContext（meta 可追加）。"""
    ctx = RunContext()
    ctx.df = df
    ctx.dataset_id = dataset_id
    ctx.meta = {"n_rows": len(df), "columns": [str(c) for c in df.columns], **meta}
    return ctx


def _rec(data: dict, *, scope: str, **kw) -> dict:
    """规范化一条记录并断言成功，直接返回 record（测试便捷函数）。"""
    res = ann.normalize_record(data, scope=scope, **kw)
    assert res["ok"] is True, f"normalize_record 失败：{res}"
    return res["record"]


# ---------------------------------------------------------------------------
# 锚点解析：多流采集目录（episode 列）
# ---------------------------------------------------------------------------


def test_anchors_from_episode_column_with_timestamps() -> None:
    """含 episode 列与时间列时，按唯一值划分且取到真实起止秒。"""
    df = pd.DataFrame({
        "episode": [0, 0, 0, 1, 1],
        "timestamp": [0.0, 0.1, 0.2, 0.0, 0.1],
        "joint0": [0.0, 0.1, 0.2, 0.3, 0.4],
    })
    res = ann.resolve_anchors(_ctx_with_df(df))

    assert res["success"] is True
    assert res["anchor_source"] == ann.ANCHOR_EPISODE_COLUMN
    assert res["n_anchors"] == 2
    keys = [a["episode_key"] for a in res["anchors"]]
    assert keys == ["0", "1"]
    ep0 = res["anchors"][0]
    assert ep0["n_frames"] == 3
    assert ep0["start_s"] == 0.0
    assert ep0["end_s"] == pytest.approx(0.2)
    assert ep0["time_column"] == "timestamp"
    # 证据必须可转述（供模型如实说明锚点来源）。
    assert "episode" in ep0["evidence"]


def test_anchors_falls_back_to_whole_dataset_without_episode_column() -> None:
    """无 episode 线索时降级为整段单元，并**如实警告**（不猜、不静默）。"""
    df = pd.DataFrame({"timestamp": [0.0, 0.1, 0.2], "v": [1, 2, 3]})
    res = ann.resolve_anchors(_ctx_with_df(df))

    assert res["success"] is True
    assert res["anchor_source"] == ann.ANCHOR_WHOLE
    assert res["n_anchors"] == 1
    assert res["anchors"][0]["episode_key"] == ann.WHOLE_DATASET_KEY
    assert any("未找到 episode 划分线索" in w for w in res["warnings"])


def test_anchors_no_time_column_reports_null_and_warns() -> None:
    """无时间列时 start_s/end_s 为 None，且明确警告（不造 0 值冒充）。"""
    df = pd.DataFrame({"episode": [0, 0], "joint0": [1.0, 2.0]})
    res = ann.resolve_anchors(_ctx_with_df(df))

    ep = res["anchors"][0]
    assert ep["start_s"] is None
    assert ep["end_s"] is None
    assert any("start_s/end_s 为 null" in w for w in res["warnings"])


def test_anchors_fps_derives_frames_and_seconds() -> None:
    """有 fps 列时按帧率推算秒（帧↔秒换算可用）。"""
    df = pd.DataFrame({
        "episode": [0] * 10 + [1] * 5,
        "fps": [10.0] * 15,
        "joint0": range(15),
    })
    res = ann.resolve_anchors(_ctx_with_df(df))

    ep0 = res["anchors"][0]
    assert ep0["start_frame"] == 0
    assert ep0["end_frame"] == 9
    assert ep0["start_s"] == pytest.approx(0.0)
    assert ep0["end_s"] == pytest.approx(1.0)  # (9+1)/10


def test_anchors_unloaded_dataset_returns_structured_error() -> None:
    """未加载数据集时返回结构化错误（不抛异常）。"""
    res = ann.resolve_anchors(RunContext())
    assert res["success"] is False
    assert res["error"] == "no_data_loaded"
    assert res["n_anchors"] == 0


# ---------------------------------------------------------------------------
# 锚点解析：LeRobot v2
# ---------------------------------------------------------------------------


def test_anchors_lerobot_layout_uses_info_fps(tmp_path: Path) -> None:
    """LeRobot 布局：识别 meta/info.json，用其 fps 做帧↔秒换算。"""
    root = tmp_path / "lerobot_ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 20, "codebase_version": "v2.0", "features": {}}),
        encoding="utf-8",
    )
    df = pd.DataFrame({
        "episode_index": [0] * 21 + [1] * 11,
        "frame_index": list(range(21)) + list(range(11)),
        "action": [0.0] * 32,
    })
    res = ann.resolve_anchors(_ctx_with_df(df, source=str(root)))

    assert res["anchor_source"] == ann.ANCHOR_LEROBOT
    assert res["n_anchors"] == 2
    ep0 = res["anchors"][0]
    assert ep0["n_frames"] == 21
    assert ep0["end_s"] == pytest.approx(21 / 20)  # (20+1)/20
    assert "episode_index" in ep0["evidence"]


def test_anchors_lerobot_without_fps_warns(tmp_path: Path) -> None:
    """LeRobot 布局但 info.json 无 fps：帧↔秒换算不可用，须警告。"""
    root = tmp_path / "lerobot_nofps"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v2.0", "features": {}}), encoding="utf-8"
    )
    df = pd.DataFrame({"episode_index": [0, 0, 0], "action": [0.0, 1.0, 2.0]})
    res = ann.resolve_anchors(_ctx_with_df(df, source=str(root)))

    assert res["anchor_source"] == ann.ANCHOR_LEROBOT
    assert any("未给出可用 fps" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# 锚点解析：HDF5
# ---------------------------------------------------------------------------


def test_anchors_from_h5_node_streams() -> None:
    """HDF5：以每个节点流为标注单元，并说明"h5 无天然 episode 边界"。"""
    ctx = RunContext()
    ctx.dataset_id = "h5ds"
    ctx.meta = {
        "streams": [
            {"path": "a.h5::action/end/orientation", "format": "h5", "kind": "actions"},
            {"path": "a.h5::obs/joint", "format": "h5", "kind": "state",
             "frame_layout": True, "n_frames": 100},
            {"path": "imu.csv", "format": "csv"},  # 非 h5，应被忽略
        ],
    }
    res = ann.resolve_anchors(ctx)

    assert res["anchor_source"] == ann.ANCHOR_H5_NODE
    assert res["n_anchors"] == 2
    keys = sorted(a["episode_key"] for a in res["anchors"])
    assert keys == ["action_end_orientation", "obs_joint"]
    by_key = {a["episode_key"]: a for a in res["anchors"]}
    assert by_key["obs_joint"]["n_frames"] == 100
    assert by_key["obs_joint"]["end_frame"] == 99
    assert any("每个节点流" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# 时间戳格式化与解析
# ---------------------------------------------------------------------------


def test_format_and_parse_timestamp_are_inverse() -> None:
    """格式化与解析互为逆（往返一致）。"""
    assert ann.format_timestamp(16) == "00:00:16"
    assert ann.format_timestamp(3661) == "01:01:01"
    assert ann.format_timestamp(None) == ""  # 诚实降级：不填 00:00:00
    assert ann.format_timestamp(0) == "00:00:00"

    assert ann.parse_timestamp("00:00:16") == 16.0
    assert ann.parse_timestamp("01:01:01") == 3661.0
    assert ann.parse_timestamp("00:16") == 16.0
    assert ann.parse_timestamp("16") == 16.0
    assert ann.parse_timestamp("") is None
    assert ann.parse_timestamp("bad") is None


def test_parse_timestamp_rejects_ambiguous_input() -> None:
    """四级及以上冒号分隔视为非法（不猜）。"""
    assert ann.parse_timestamp("1:2:3:4") is None


# ---------------------------------------------------------------------------
# normalize_record：规范化与硬校验
# ---------------------------------------------------------------------------


def _anchor() -> ann.EpisodeAnchor:
    return ann.EpisodeAnchor(
        episode_key="0", label="episode 0",
        start_s=0.0, end_s=10.0, start_frame=0, end_frame=200,
        n_frames=201, time_column="timestamp",
    )


def test_normalize_segment_derives_timestamp_from_seconds() -> None:
    """秒为权威，HH:MM:SS 由秒派生（两者都要）。"""
    res = ann.normalize_record(
        {"id": 1, "start_s": 16.0, "end_s": 20.0, "atomic_action": "Place"},
        scope=ann.SCOPE_SEGMENT, anchor=_anchor(),
    )
    assert res["ok"] is True
    rec = res["record"]
    assert rec["start_timestamp"] == "00:00:16"
    assert rec["end_timestamp"] == "00:00:20"
    assert rec["start_s"] == 16.0
    # 缺省来源按最保守处理（未确认的模型产出）。
    assert rec["source"] == ann.SOURCE_LLM
    assert rec["confidence"] == "low"


def test_normalize_segment_accepts_timestamp_string_input() -> None:
    """只给 HH:MM:SS 时能反解出数值秒（互为逆）。"""
    res = ann.normalize_record(
        {"id": 2, "start_timestamp": "00:00:16", "end_timestamp": "00:00:20"},
        scope=ann.SCOPE_SEGMENT, anchor=_anchor(),
    )
    assert res["ok"] is True
    assert res["record"]["start_s"] == 16.0


def test_normalize_segment_derives_frames_from_anchor_ratio() -> None:
    """帧号缺省时按锚点的秒↔帧比例外推（而非猜未知单位的时间列）。"""
    res = ann.normalize_record(
        {"id": 1, "start_s": 5.0, "end_s": 10.0},
        scope=ann.SCOPE_SEGMENT, anchor=_anchor(),
    )
    # 锚点：0..10s 对应 0..200 帧 → 5s 应对应 100 帧。
    assert res["record"]["start_frame"] == 100


def test_normalize_segment_rejects_missing_time_bounds() -> None:
    """缺时间边界的片段**拒绝落盘**（不接受无时间的切片）。"""
    res = ann.normalize_record({"id": 1}, scope=ann.SCOPE_SEGMENT, anchor=_anchor())
    assert res["ok"] is False
    assert res["error"] == "missing_time_bounds"


def test_normalize_segment_rejects_reversed_bounds() -> None:
    """end < start 拒绝（数据自相矛盾）。"""
    res = ann.normalize_record(
        {"id": 1, "start_s": 20.0, "end_s": 10.0},
        scope=ann.SCOPE_SEGMENT, anchor=_anchor(),
    )
    assert res["ok"] is False
    assert res["error"] == "reversed_time_bounds"


def test_normalize_noise_requires_reason() -> None:
    """is_noise=true 必须给 cleaning_reason（标注规则一致性硬约束）。"""
    res = ann.normalize_record(
        {"id": 1, "start_s": 0.0, "end_s": 5.0, "is_noise": True},
        scope=ann.SCOPE_SEGMENT, anchor=_anchor(),
    )
    assert res["ok"] is False
    assert res["error"] == "noise_reason_required"

    ok = ann.normalize_record(
        {"id": 1, "start_s": 0.0, "end_s": 5.0, "is_noise": True,
         "cleaning_reason": "机器人暂停等待指令"},
        scope=ann.SCOPE_SEGMENT, anchor=_anchor(),
    )
    assert ok["ok"] is True


def test_normalize_rejects_invalid_source_and_confidence() -> None:
    """来源/可信度非法值**拒绝**，不静默改成默认值。"""
    bad_src = ann.normalize_record(
        {"episode_key": "0", "id": 1, "start_s": 0.0, "end_s": 1.0,
         "source": "guessed"},
        scope=ann.SCOPE_SEGMENT,
    )
    assert bad_src["ok"] is False
    assert bad_src["error"] == "invalid_source"

    bad_conf = ann.normalize_record(
        {"episode_key": "0", "id": 1, "start_s": 0.0, "end_s": 1.0,
         "confidence": "very_high"},
        scope=ann.SCOPE_SEGMENT,
    )
    assert bad_conf["ok"] is False
    assert bad_conf["error"] == "invalid_confidence"


def test_normalize_task_requires_task_text() -> None:
    """任务级标注必须有 task 描述。"""
    res = ann.normalize_record({"episode_key": "0"}, scope=ann.SCOPE_TASK)
    assert res["ok"] is False
    assert res["error"] == "missing_task"

    ok = ann.normalize_record(
        {"task": "把红色方块放进碗里"}, scope=ann.SCOPE_TASK, episode_key="0",
    )
    assert ok["ok"] is True
    assert ok["record"]["scope"] == ann.SCOPE_TASK


def test_normalize_requires_episode_key() -> None:
    """缺 episode_key 拒绝（无法定位到数据）。"""
    res = ann.normalize_record({"task": "x"}, scope=ann.SCOPE_TASK)
    assert res["ok"] is False
    assert res["error"] == "missing_episode_key"


# ---------------------------------------------------------------------------
# 来源分级与用途合规（用户决策：三种用途都要）
# ---------------------------------------------------------------------------


def test_check_source_blocks_llm_for_training() -> None:
    """llm_proposed 与 signal_derived 都禁止直接进训练真值。"""
    recs = [
        {"episode_key": "0", "id": 1, "source": ann.SOURCE_LLM},
        {"episode_key": "0", "id": 2, "source": ann.SOURCE_SIGNAL},
        {"episode_key": "0", "id": 3, "source": ann.SOURCE_USER},
    ]
    res = ann.check_source_for_use(recs, ann.USE_TRAINING)
    assert res["ok"] is False
    assert res["n_violations"] == 2
    sources = {v["source"] for v in res["violations"]}
    assert sources == {ann.SOURCE_LLM, ann.SOURCE_SIGNAL}


def test_check_source_allows_user_for_training() -> None:
    """user_confirmed 可用于训练真值。"""
    recs = [{"episode_key": "0", "id": 1, "source": ann.SOURCE_USER}]
    assert ann.check_source_for_use(recs, ann.USE_TRAINING)["ok"] is True


def test_check_source_publication_requires_generated_declaration() -> None:
    """发布用途允许自动来源，但须声明为自动生成。"""
    recs = [
        {"episode_key": "0", "id": 1, "source": ann.SOURCE_LLM},
        {"episode_key": "0", "id": 2, "source": ann.SOURCE_SIGNAL},
        {"episode_key": "0", "id": 3, "source": ann.SOURCE_USER},
    ]
    res = ann.check_source_for_use(recs, ann.USE_PUBLICATION)
    assert res["ok"] is True
    assert res["must_declare_generated"] == 2


def test_check_source_audit_allows_everything() -> None:
    """内部清查用途全部允许。"""
    recs = [
        {"episode_key": "0", "id": 1, "source": ann.SOURCE_LLM},
        {"episode_key": "0", "id": 2, "source": ann.SOURCE_SIGNAL},
    ]
    res = ann.check_source_for_use(recs, ann.USE_AUDIT)
    assert res["ok"] is True
    assert res["n_violations"] == 0


def test_check_source_invalid_use_returns_error() -> None:
    """非法用途返回结构化错误。"""
    res = ann.check_source_for_use([], "guess")
    assert res["ok"] is False
    assert res["error"] == "invalid_use"


# ---------------------------------------------------------------------------
# 读写：追加 / 更新去重 / 删除
# ---------------------------------------------------------------------------


@pytest.fixture
def seg_records() -> list[dict]:
    """两条已规范化的切片记录。"""
    return [
        _rec(
            {"id": 1, "start_s": 0.0, "end_s": 16.0, "atomic_action": "Place",
             "source": ann.SOURCE_LLM},
            scope=ann.SCOPE_SEGMENT, episode_key="0",
        ),
        _rec(
            {"id": 2, "start_s": 16.0, "end_s": 20.0, "atomic_action": "Walk_Forward",
             "source": ann.SOURCE_LLM},
            scope=ann.SCOPE_SEGMENT, episode_key="0",
        ),
    ]


def test_append_and_load_round_trip(output_dir: str, seg_records: list[dict]) -> None:
    """落盘后能读回，字段完整（含 scope 标注）。"""
    res = ann.append_annotations(output_dir, "demo", seg_records)
    assert res["saved"] == 2
    assert res["updated"] == 0

    loaded = ann.load_annotations(output_dir, "demo", scope=ann.SCOPE_SEGMENT)
    assert loaded["n_records"] == 2
    assert loaded["bad_lines"] == 0
    acts = [r["atomic_action"] for r in loaded["records"]]
    assert acts == ["Place", "Walk_Forward"]


def test_append_updates_existing_instead_of_duplicating(
    output_dir: str, seg_records: list[dict]
) -> None:
    """同一 (episode_key, id) 重复提交视为更新，不产生重复行。"""
    ann.append_annotations(output_dir, "demo", seg_records)

    changed = _rec(
        {"id": 1, "start_s": 0.0, "end_s": 15.8, "atomic_action": "Pick_Up",
         "source": ann.SOURCE_USER, "confidence": "high"},
        scope=ann.SCOPE_SEGMENT, episode_key="0",
    )
    res = ann.append_annotations(
        output_dir, "demo", [changed], actor=ann.SOURCE_USER, reason="人工复核修正",
    )
    assert res["updated"] == 1
    assert res["saved"] == 0

    loaded = ann.load_annotations(output_dir, "demo", scope=ann.SCOPE_SEGMENT)
    assert loaded["n_records"] == 2  # 仍是两行，未重复
    row1 = next(r for r in loaded["records"] if r["id"] == 1)
    assert row1["atomic_action"] == "Pick_Up"
    assert row1["source"] == ann.SOURCE_USER


def test_append_identical_record_reports_skipped(
    output_dir: str, seg_records: list[dict]
) -> None:
    """内容完全一致时不产生变更，并如实计入 skipped（不假装更新）。"""
    ann.append_annotations(output_dir, "demo", seg_records)
    res = ann.append_annotations(output_dir, "demo", [seg_records[0]])
    assert res["saved"] == 0
    assert res["updated"] == 1  # 命中既有键
    assert res["skipped"] == 1
    assert "内容与既有记录完全一致" in res["skipped_detail"][0]["reason"]


def test_append_records_for_both_scopes(output_dir: str) -> None:
    """任务级与切片级分别落不同文件。"""
    task = _rec(
        {"task": "把蔬菜夹到碗里", "source": ann.SOURCE_USER},
        scope=ann.SCOPE_TASK, episode_key="0",
    )
    seg = _rec(
        {"id": 1, "start_s": 0.0, "end_s": 16.0},
        scope=ann.SCOPE_SEGMENT, episode_key="0",
    )
    ann.append_annotations(output_dir, "demo", [task, seg])

    loaded = ann.load_annotations(output_dir, "demo")
    assert loaded["by_scope"] == {ann.SCOPE_TASK: 1, ann.SCOPE_SEGMENT: 1}
    d = ann.annotation_dir(output_dir, "demo")
    assert (d / "tasks.jsonl").exists()
    assert (d / "segments.jsonl").exists()


def test_load_filters_by_episode_key(output_dir: str) -> None:
    """可按 episode_key 过滤。"""
    recs = [
        _rec({"id": 1, "start_s": 0.0, "end_s": 1.0},
             scope=ann.SCOPE_SEGMENT, episode_key="0"),
        _rec({"id": 1, "start_s": 0.0, "end_s": 1.0},
             scope=ann.SCOPE_SEGMENT, episode_key="1"),
    ]
    ann.append_annotations(output_dir, "demo", recs)
    loaded = ann.load_annotations(output_dir, "demo", episode_key="1")
    assert loaded["n_records"] == 1
    assert loaded["records"][0]["episode_key"] == "1"


def test_load_annotations_on_empty_dir_returns_empty(output_dir: str) -> None:
    """未标注时返回空清单而非报错。"""
    loaded = ann.load_annotations(output_dir, "never_annotated")
    assert loaded["n_records"] == 0
    assert loaded["bad_lines"] == 0


def test_load_reports_corrupted_lines(output_dir: str) -> None:
    """损坏行**如实报告**行数，绝不静默忽略。"""
    ann.append_annotations(output_dir, "demo", [
        _rec({"id": 1, "start_s": 0.0, "end_s": 1.0},
             scope=ann.SCOPE_SEGMENT, episode_key="0")
    ])
    p = ann.annotation_dir(output_dir, "demo") / "segments.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "{ 这不是合法 JSON\n",
                 encoding="utf-8")

    loaded = ann.load_annotations(output_dir, "demo", scope=ann.SCOPE_SEGMENT)
    assert loaded["n_records"] == 1
    assert loaded["bad_lines"] == 1


def test_delete_annotation_removes_only_target(
    output_dir: str, seg_records: list[dict]
) -> None:
    """删除只影响目标记录，其余保留。"""
    ann.append_annotations(output_dir, "demo", seg_records)
    res = ann.delete_annotation(
        output_dir, "demo", scope=ann.SCOPE_SEGMENT, episode_key="0", id=1,
        actor=ann.SOURCE_USER, reason="边界错误，剔除",
    )
    assert res["deleted"] is True
    assert res["before"]["atomic_action"] == "Place"

    loaded = ann.load_annotations(output_dir, "demo", scope=ann.SCOPE_SEGMENT)
    assert loaded["n_records"] == 1
    assert loaded["records"][0]["id"] == 2


def test_delete_nonexistent_returns_false(output_dir: str) -> None:
    """删除不存在的记录返回 deleted=False（不抛异常）。"""
    res = ann.delete_annotation(
        output_dir, "demo", scope=ann.SCOPE_SEGMENT, episode_key="0", id=999,
    )
    assert res["deleted"] is False


def test_append_rejects_invalid_scope_record(output_dir: str) -> None:
    """scope 非法的记录被跳过并如实报告（不静默丢弃）。"""
    res = ann.append_annotations(
        output_dir, "demo", [{"scope": "bogus", "episode_key": "0", "id": 1}]
    )
    assert res["skipped"] == 1
    assert "scope 非 task/segment" in res["skipped_detail"][0]["reason"]


# ---------------------------------------------------------------------------
# 版本管理：修订日志 / 快照 / diff
# ---------------------------------------------------------------------------


def test_revision_log_records_create_and_update(
    output_dir: str, seg_records: list[dict]
) -> None:
    """修订日志记录 create 与 update 两类操作，update 含 before/after。"""
    ann.append_annotations(output_dir, "demo", seg_records)

    changed = _rec(
        {"id": 1, "start_s": 0.0, "end_s": 15.8, "atomic_action": "Pick_Up"},
        scope=ann.SCOPE_SEGMENT, episode_key="0",
    )
    ann.append_annotations(
        output_dir, "demo", [changed], actor=ann.SOURCE_USER,
        reason="人工复核修正边界", session_tag="s-1a2b",
    )

    hist = ann.load_annotation_history(output_dir, "demo")
    ops = [e["op"] for e in hist["entries"]]
    assert ops.count("create") == 2
    assert ops.count("update") == 1

    upd = next(e for e in hist["entries"] if e["op"] == "update")
    assert upd["reason"] == "人工复核修正边界"
    assert upd["session_tag"] == "s-1a2b"
    assert upd["before"]["atomic_action"] == "Place"
    assert upd["after"]["atomic_action"] == "Pick_Up"
    # 时间戳字段不应出现在变更清单（每次都会变，是噪声）。
    assert "created_at" not in upd["before"]


def test_revision_log_records_delete(
    output_dir: str, seg_records: list[dict]
) -> None:
    """删除也进修订日志（可回溯"这条为什么没了"）。"""
    ann.append_annotations(output_dir, "demo", seg_records)
    ann.delete_annotation(
        output_dir, "demo", scope=ann.SCOPE_SEGMENT, episode_key="0", id=1,
        reason="重复标注",
    )
    hist = ann.load_annotation_history(
        output_dir, "demo", scope=ann.SCOPE_SEGMENT, episode_key="0",
    )
    ops = [e["op"] for e in hist["entries"]]
    assert "delete" in ops
    deleted = next(e for e in hist["entries"] if e["op"] == "delete")
    assert deleted["reason"] == "重复标注"


def test_history_filters_and_reports_truncation(
    output_dir: str, seg_records: list[dict]
) -> None:
    """历史可按 episode_key 过滤；limit 截断时如实报告 truncated。"""
    ann.append_annotations(output_dir, "demo", seg_records)
    hist = ann.load_annotation_history(output_dir, "demo", limit=1)
    assert hist["truncated"] is True
    assert hist["n_entries"] == 2
    assert len(hist["entries"]) == 1  # 取最近的


def test_snapshot_freezes_current_state(
    output_dir: str, seg_records: list[dict]
) -> None:
    """快照冻结当前态，后续编辑不影响快照内容。"""
    ann.append_annotations(output_dir, "demo", seg_records)
    snap = ann.snapshot_annotations(
        output_dir, "demo", label="v1_for_release", scopes=[ann.SCOPE_SEGMENT],
    )
    assert snap["success"] is True
    assert snap["counts"][ann.SCOPE_SEGMENT] == 2
    snap_path = Path(snap["paths"][ann.SCOPE_SEGMENT])
    assert snap_path.exists()

    # 后续编辑当前态。
    ann.delete_annotation(
        output_dir, "demo", scope=ann.SCOPE_SEGMENT, episode_key="0", id=1,
    )
    # 快照不受影响（冻结语义）。
    snap_rows, _ = ann._read_jsonl(snap_path)
    assert len(snap_rows) == 2


def test_snapshot_without_annotations_fails_cleanly(output_dir: str) -> None:
    """无标注时不产出空快照（如实失败）。"""
    res = ann.snapshot_annotations(output_dir, "empty_ds")
    assert res["success"] is False
    assert res["error"] == "no_annotations"


def test_list_snapshots(output_dir: str, seg_records: list[dict]) -> None:
    """能列出已有快照及其记录数。"""
    ann.append_annotations(output_dir, "demo", seg_records)
    ann.snapshot_annotations(
        output_dir, "demo", label="first", scopes=[ann.SCOPE_SEGMENT])
    res = ann.list_snapshots(output_dir, "demo")
    assert res["n_snapshots"] == 1
    assert res["snapshots"][0]["n_records"] == 2


def test_diff_snapshots_classifies_three_ways(
    output_dir: str, seg_records: list[dict]
) -> None:
    """diff 正确分出新增/修改/删除三类。"""
    ann.append_annotations(output_dir, "demo", seg_records)
    snap_a = ann.snapshot_annotations(
        output_dir, "demo", label="a", scopes=[ann.SCOPE_SEGMENT])

    # 改动：1 号改动作名、2 号删除；新增 3 号。
    changed = _rec(
        {"id": 1, "start_s": 0.0, "end_s": 16.0, "atomic_action": "Pick_Up"},
        scope=ann.SCOPE_SEGMENT, episode_key="0",
    )
    ann.append_annotations(output_dir, "demo", [changed])
    ann.delete_annotation(
        output_dir, "demo", scope=ann.SCOPE_SEGMENT, episode_key="0", id=2)
    added = _rec(
        {"id": 3, "start_s": 20.0, "end_s": 25.0},
        scope=ann.SCOPE_SEGMENT, episode_key="0",
    )
    ann.append_annotations(output_dir, "demo", [added])

    snap_b = ann.snapshot_annotations(
        output_dir, "demo", label="b", scopes=[ann.SCOPE_SEGMENT])

    diff = ann.diff_snapshots(
        snap_a["paths"][ann.SCOPE_SEGMENT], snap_b["paths"][ann.SCOPE_SEGMENT])
    assert diff["success"] is True
    assert diff["added"] == 1
    assert diff["modified"] == 1
    assert diff["removed"] == 1
    mod = diff["items"]["modified"][0]
    assert mod["changes"]["atomic_action"] == ["Place", "Pick_Up"]


def test_diff_snapshots_missing_files() -> None:
    """两个快照都不存在时返回结构化错误。"""
    res = ann.diff_snapshots("no_such_a.jsonl", "no_such_b.jsonl")
    assert res["success"] is False
    assert res["error"] == "snapshot_not_found"


# ---------------------------------------------------------------------------
# 路径与隔离
# ---------------------------------------------------------------------------


def test_annotation_dir_uses_dataset_subdir(output_dir: str) -> None:
    """标注目录落在 by_dataset/<净名>/annotations 下（与其它产物同源）。"""
    d = ann.annotation_dir(output_dir, "my_dataset")
    assert d.name == "annotations"
    assert d.parent.parent.name == "by_dataset"
    assert d.exists()


def test_annotation_dir_falls_back_to_misc(output_dir: str) -> None:
    """dataset_id 为空时走 _misc 兜底（不抛异常）。"""
    d = ann.annotation_dir(output_dir, None)
    assert d.exists()
    assert "_misc" in str(d)


def test_two_datasets_do_not_share_annotations(output_dir: str) -> None:
    """不同数据集的标注互相隔离。"""
    rec = _rec(
        {"id": 1, "start_s": 0.0, "end_s": 1.0},
        scope=ann.SCOPE_SEGMENT, episode_key="0",
    )
    ann.append_annotations(output_dir, "ds_a", [rec])
    assert ann.load_annotations(output_dir, "ds_b")["n_records"] == 0
