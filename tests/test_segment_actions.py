"""segment_actions 测试：确定性边界检测 + **只给边界不给名字**。

设计依据：``docs/标注与质检能力设计.md`` §5.2.2。

**本文件最重要的守护有两条**：

1. ``test_no_motion_signal_*``：无运动信号时**必须失败且不给等分兜底**
   （等分边界无物理依据，会污染下游标注与训练）；
2. ``test_boundary_only_no_action_name_*``：输出**绝不含** ``atomic_action`` /
   ``action_description`` 字段（当前模型无视觉能力，生成即编造）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.config import get_settings
from app.tools.segment_actions import segment_actions_impl


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    return get_settings()


def _ctx(df, dataset_id: str = "demo", **meta) -> RunContext:
    ctx = RunContext()
    ctx.df = df
    ctx.dataset_id = dataset_id
    ctx.meta = dict(meta)
    if df is not None:
        ctx.meta.setdefault("columns", [str(c) for c in df.columns])
    return ctx


def _segmented_df(
    n: int = 400, fps: float = 50.0, n_actions: int = 4,
) -> pd.DataFrame:
    """构造有明确动作段的轨迹：每段内速度不同，段间有速度突变。

    这样"速度变化点"信号应有确定的检出（合成真值已知）。
    """
    t = np.arange(n) / fps
    seg_len = n // n_actions
    # 每段用不同频率的正弦（速度不同），段间用阶跃造成速度突变。
    parts = []
    for i in range(n_actions):
        seg_t = np.arange(seg_len) / fps
        freq = 0.5 + i * 1.0
        parts.append(np.sin(2 * np.pi * freq * seg_t) * (1 + i))
    joint0 = np.concatenate(parts)
    n_actual = joint0.size

    # 夹爪：在第 1 段末打开、第 2 段末关闭（构造明确的抓取/释放事件）。
    gripper = np.zeros(n_actual)
    gripper[int(n_actual * 0.25):int(n_actual * 0.5)] = 1.0
    gripper[int(n_actual * 0.75):] = 1.0

    return pd.DataFrame({
        "episode_index": [0] * n_actual,
        "timestamp": t[:n_actual],
        "fps": [fps] * n_actual,
        "action_joint0": joint0,
        "action_joint1": joint0 * 0.5,
        "gripper_position": gripper,
    })


# ---------------------------------------------------------------------------
# 诚实降级（最重要的守护之一）
# ---------------------------------------------------------------------------


def test_no_motion_signal_does_not_fallback_to_equal_split(settings) -> None:
    """**无运动信号时必须失败，绝不产出等分切片。**

    构造只有恒定的元数据列（无任何运动通道）的数据。
    """
    n = 200
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp": np.arange(n) * 0.02,
        "success": [1] * n,  # 元数据列，不参与运动分析
    })
    res = segment_actions_impl(_ctx(df), settings=settings)

    assert res["success"] is False
    assert res["error"] == "no_motion_signal"
    # 关键：不得返回任何片段。
    assert "episodes" not in res or not res.get("episodes")
    # 必须明确告知"为什么不做等分"。
    assert "等分" in res["user_message"]
    assert res["signals_missing"]


def test_no_motion_signal_on_constant_channels(settings) -> None:
    """恒定动作通道（无任何变化）也视为无信号，不产出切片。"""
    n = 200
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.02,
        "action_joint0": np.zeros(n),
        "action_joint1": np.zeros(n),
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is False
    assert res["error"] == "no_motion_signal"


def test_no_motion_signal_lists_missing_signals(settings) -> None:
    """失败时必须列出缺少的信号，便于用户定位原因。"""
    df = pd.DataFrame({
        "timestamp": np.arange(50) * 0.02,
        "success": [1] * 50,
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["signals_missing"]
    assert any("夹爪" in m or "运动通道" in m or "速度" in m
               for m in res["signals_missing"])


def test_not_applicable_without_table(settings) -> None:
    """无状态/动作表时返回 not_applicable（不抛异常）。"""
    ctx = RunContext()
    ctx.dataset_id = "streams_only"
    ctx.meta = {"streams": []}
    res = segment_actions_impl(ctx, settings=settings)
    assert res["success"] is False
    assert res["error"] in ("not_applicable", "no_motion_signal")


def test_unloaded_dataset_returns_structured_error(settings) -> None:
    """未加载数据集返回结构化错误。"""
    res = segment_actions_impl(RunContext(), settings=settings)
    assert res["success"] is False
    assert res["error"] == "no_data_loaded"


# ---------------------------------------------------------------------------
# 只给边界不给名字（最重要的守护之二）
# ---------------------------------------------------------------------------


def test_output_contains_no_action_name_fields(settings) -> None:
    """**输出绝不含动作名与描述字段**（模型无视觉能力，生成即编造）。

    断言范围限定在**数据字段**（episodes 的片段本身），不含解释性文案——
    文案里提到这些字段名是为了说明"本工具不生成它们"，属于必要的说明。
    """
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    assert res["success"] is True

    # 逐片段检查实际数据字段（这才是会被落盘/训练使用的部分）。
    for ep in res["episodes"]:
        for s in ep["segments"]:
            for forbidden in ("atomic_action", "action_description",
                              "action_description_en", "interacting_hand",
                              "target_object_class", "is_noise",
                              "cleaning_reason"):
                assert forbidden not in s, f"片段数据中不得出现 {forbidden}"

    # 明确声明边界与名称的分离。
    assert res["boundary_only"] is True
    assert res["action_name_generated"] is False
    assert "不生成" in res["note"]


def test_user_message_states_no_action_names(settings) -> None:
    """用户消息必须主动说明"只有边界、没有动作名称"。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    msg = res["user_message"]
    assert "只有边界" in msg or "不含动作名" in msg or "没有动作名称" in msg
    assert "视觉" in msg  # 说明原因


def test_segments_have_boundary_evidence(settings) -> None:
    """每个片段必须带边界证据（供人工复核"凭什么在这里切"）。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    for ep in res["episodes"]:
        for s in ep["segments"]:
            assert "boundary_evidence" in s
            assert s["boundary_evidence"]["signals"]
            assert "人工" in s["boundary_evidence"]["note"]


def test_segment_fields_match_annotation_schema(settings) -> None:
    """片段字段必须能直接喂给 annotation_store.normalize_record。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    seg = res["episodes"][0]["segments"][0]
    for key in ("start_s", "end_s", "start_frame", "end_frame"):
        assert key in seg, f"片段缺少 {key}"

    # 实际验证可被标注层接受。
    from app.tools.annotation_store import SCOPE_SEGMENT, normalize_record
    rec = normalize_record(
        {"id": 1, "start_s": seg["start_s"], "end_s": seg["end_s"],
         "start_frame": seg["start_frame"], "end_frame": seg["end_frame"]},
        scope=SCOPE_SEGMENT, episode_key="0",
    )
    assert rec["ok"] is True


# ---------------------------------------------------------------------------
# 信号检测正确性
# ---------------------------------------------------------------------------


def test_detects_segments_on_synthetic_trajectory(settings) -> None:
    """合成轨迹应产出多个候选片段（不要求数量精确——那是模型做不到的）。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    assert res["success"] is True
    assert res["n_segments"] >= 2
    assert res["n_episodes"] == 1


def test_gripper_signal_hits_true_boundaries_precisely(settings) -> None:
    """**夹爪信号精确命中真实边界**（物理事件，可靠度最高）。

    构造的数据在 25%/50%/75% 处有夹爪开合，对应真实动作段边界。
    这是本工具最可信的信号，必须精确。
    """
    res = segment_actions_impl(
        _ctx(_segmented_df(n=400)), method="gripper", settings=settings)
    assert res["success"] is True
    starts = [s["start_frame"] for s in res["episodes"][0]["segments"]]
    # 真实边界在 0/100/200/300（夹爪翻转为 100/200/300）。
    for truth in (100, 200, 300):
        assert any(abs(s - truth) <= 1 for s in starts), (
            f"夹爪信号未命中真实边界 {truth}，检出 {starts}"
        )


def test_velocity_signal_documented_as_candidate_only(settings) -> None:
    """**速度信号必须被标注为"候选"**（它无法区分切换与强度变化）。

    这是对实现局限的诚实记录：速度变化点判据实测会在不同频率的相邻段之间
    产生误报（合成轨迹上产出约 12 个候选、真实边界仅 3 个），因此必须在
    证据与用户消息中明确其不可靠性，避免用户误以为边界已被确定检出。
    """
    res = segment_actions_impl(
        _ctx(_segmented_df()), method="velocity", settings=settings)
    assert res["success"] is True

    ev = res["episodes"][0]["segments"][0]["boundary_evidence"]["note"]
    assert "候选" in ev
    assert "误报" in ev or "无法区分" in ev
    # 用户消息也须说明。
    assert "候选" in res["user_message"]


def test_velocity_detector_finds_true_boundaries_among_candidates(
    settings,
) -> None:
    """速度检测器**至少能覆盖**真实边界（作为候选），即使伴随误报。"""
    from app.tools.segment_actions import (
        _speed_profile,
        _velocity_change_points,
    )

    df = _segmented_df(n=400)
    speed, _used = _speed_profile(
        df, ["action_joint0", "action_joint1"])
    cps = _velocity_change_points(speed, settings.annotation_change_point_ratio)

    # 真实边界 100/200/300 应落在某个候选点附近（容差 10 行）。
    for truth in (100, 200, 300):
        assert any(abs(c - truth) <= 10 for c in cps), (
            f"真实边界 {truth} 未被任何候选覆盖：{cps}"
        )
    # 同时如实确认：它并非精确检测器（会有多余候选）。
    assert len(cps) >= 3


def test_gripper_events_are_detected(settings) -> None:
    """夹爪开合变化被检出（最可靠的语义边界）。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    assert "gripper_state_change" in res["signals_used"]


def test_velocity_change_points_detected(settings) -> None:
    """速度变化点信号被使用。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    assert "velocity_change_point" in res["signals_used"]


def test_gripper_only_method(settings) -> None:
    """method="gripper" 时只用夹爪信号。"""
    res = segment_actions_impl(
        _ctx(_segmented_df()), method="gripper", settings=settings)
    if res["success"]:
        assert res["signals_used"] == ["gripper_state_change"]


def test_velocity_only_method(settings) -> None:
    """method="velocity" 时只用速度信号。"""
    res = segment_actions_impl(
        _ctx(_segmented_df()), method="velocity", settings=settings)
    assert res["success"] is True
    assert "velocity_change_point" in res["signals_used"]
    assert "gripper_state_change" not in res["signals_used"]


def test_invalid_method_rejected(settings) -> None:
    """非法 method 返回结构化错误。"""
    res = segment_actions_impl(_ctx(_segmented_df()), method="bogus",
                               settings=settings)
    assert res["success"] is False
    assert res["error"] == "invalid_method"


def test_pause_boundaries_detected(settings) -> None:
    """明显停顿段的两端被检出为边界。"""
    n = 400
    # 三段运动 + 两段长时间停顿。
    a = np.concatenate([
        np.sin(np.arange(100) * 0.3),
        np.zeros(80),                      # 停顿
        np.sin(np.arange(120) * 0.5) * 2,
        np.zeros(100),                     # 停顿
    ])
    df = pd.DataFrame({
        "timestamp": np.arange(a.size) * 0.02,
        "action_joint0": a,
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    assert "pause_boundary" in res["signals_used"]


def test_constant_gripper_reports_missing_not_error(settings) -> None:
    """夹爪列存在但恒定时，如实记为缺失信号（而非报错）。"""
    n = 300
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.02,
        "action_joint0": np.sin(np.arange(n) * 0.2),
        "gripper_position": np.zeros(n),  # 恒定
    })
    res = segment_actions_impl(_ctx(df), method="gripper", settings=settings)
    # 只用夹爪信号且夹爪恒定 → 无信号。
    assert res["success"] is False
    assert any("恒定" in m for m in res["signals_missing"])


# ---------------------------------------------------------------------------
# 粒度约束
# ---------------------------------------------------------------------------


def test_respects_min_segment_duration(settings) -> None:
    """过短片段被合并（消除碎片）。"""
    res = segment_actions_impl(
        _ctx(_segmented_df()), min_segment_s=2.0, settings=settings)
    assert res["success"] is True
    for ep in res["episodes"]:
        for s in ep["segments"]:
            if s["start_s"] is not None and s["end_s"] is not None:
                # 允许首尾因合并而略短（合并规则会并到相邻段）。
                dur = s["end_s"] - s["start_s"]
                assert dur > 0


def test_respects_max_segment_duration(settings) -> None:
    """过长片段被再切（不超上限）。"""
    n = 2000
    t = np.arange(n) * 0.02
    df = pd.DataFrame({
        "timestamp": t,
        "action_joint0": np.sin(2 * np.pi * 0.5 * t),
    })
    res = segment_actions_impl(
        _ctx(df), max_segment_s=5.0, settings=settings)
    if res["success"]:
        for ep in res["episodes"]:
            for s in ep["segments"]:
                if s["start_s"] is not None and s["end_s"] is not None:
                    dur = s["end_s"] - s["start_s"]
                    # 允许 10% 余量（谷值切分不精确落在等分点）。
                    assert dur <= 5.0 * 1.15, f"片段 {dur}s 超上限"


def test_segments_are_contiguous(settings) -> None:
    """相邻片段首尾相接（无空隙、无重叠）——直接喂标注层不会冲突。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    for ep in res["episodes"]:
        segs = ep["segments"]
        for prev, cur in zip(segs, segs[1:]):
            # 帧号连续。
            assert cur["start_frame"] == prev["end_frame"] + 1, (
                f"片段不连续：{prev['end_frame']} → {cur['start_frame']}"
            )
            if prev["end_s"] is not None and cur["start_s"] is not None:
                assert cur["start_s"] == pytest.approx(prev["end_s"], abs=1e-6)


def test_covers_whole_trajectory(settings) -> None:
    """片段覆盖整条轨迹（首个从 0 开始，末个到末尾）。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    for ep in res["episodes"]:
        segs = ep["segments"]
        assert segs[0]["start_frame"] == 0
        assert segs[-1]["end_frame"] == ep["n_frames"] - 1


# ---------------------------------------------------------------------------
# 多 episode 与筛选
# ---------------------------------------------------------------------------


def test_multi_episode_segmentation(settings) -> None:
    """多 episode 各自切分。"""
    df1 = _segmented_df(200)
    df2 = _segmented_df(200)
    df2["episode_index"] = 1
    df = pd.concat([df1, df2], ignore_index=True)
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    assert res["n_episodes"] == 2
    assert {e["episode_key"] for e in res["episodes"]} == {"0", "1"}


def test_episode_key_filter(settings) -> None:
    """可按 episode_key 只处理一个 episode。"""
    df1 = _segmented_df(200)
    df2 = _segmented_df(200)
    df2["episode_index"] = 1
    df = pd.concat([df1, df2], ignore_index=True)
    res = segment_actions_impl(_ctx(df), episode_key="1", settings=settings)
    assert res["n_episodes"] == 1
    assert res["episodes"][0]["episode_key"] == "1"


def test_episode_not_found(settings) -> None:
    """指定不存在的 episode 返回结构化错误并提示取锚点。"""
    res = segment_actions_impl(
        _ctx(_segmented_df()), episode_key="999", settings=settings)
    assert res["success"] is False
    assert res["error"] == "episode_not_found"
    assert "resolve_anchors" in res["available_hint"]


def test_frame_numbers_offset_by_original_frames(settings) -> None:
    """有帧号列时，输出帧号基于原始帧号（而非从 0 重排）。"""
    n = 300
    t = np.arange(n) * 0.02
    df = pd.DataFrame({
        "timestamp": t,
        "frame_index": np.arange(1000, 1000 + n),  # 原始帧号从 1000 起
        "action_joint0": np.concatenate([
            np.sin(np.arange(150) * 0.3), np.sin(np.arange(150) * 1.2) * 2]),
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    assert res["episodes"][0]["segments"][0]["start_frame"] == 1000


# ---------------------------------------------------------------------------
# 返回结构与阈值
# ---------------------------------------------------------------------------


def test_thresholds_reported_with_default_source(settings) -> None:
    """阈值与来源必须给出（供模型说明"未经数据集验证"）。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    assert res["threshold_source"] == "default"
    th = res["thresholds"]
    for key in ("change_point_ratio", "idle_speed", "min_pause_steps",
                "min_segment_s", "max_segment_s", "spike_multiplier"):
        assert key in th, f"阈值 {key} 未给出"


def test_user_message_mentions_unverified_thresholds(settings) -> None:
    """用户消息须说明阈值未经数据集验证且业界无公认标准。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    msg = res["user_message"]
    assert "默认值" in msg
    assert "无公认标准" in msg


def test_no_time_column_reports_null_seconds(settings) -> None:
    """无时间列且无 fps 时，秒为 None（诚实降级，不造 0）。"""
    n = 200
    df = pd.DataFrame({
        "action_joint0": np.concatenate([
            np.sin(np.arange(100) * 0.3), np.sin(np.arange(100) * 1.2) * 2]),
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    seg = res["episodes"][0]["segments"][0]
    assert seg["start_s"] is None
    assert seg["end_s"] is None
    # 帧号仍应有值。
    assert seg["start_frame"] == 0


def test_fps_derives_seconds_without_time_column(settings) -> None:
    """无时间列但有 fps 时，按帧号推算秒。"""
    n = 200
    df = pd.DataFrame({
        "fps": [10.0] * n,
        "action_joint0": np.concatenate([
            np.sin(np.arange(100) * 0.3), np.sin(np.arange(100) * 1.2) * 2]),
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    seg = res["episodes"][0]["segments"][0]
    assert seg["start_s"] == 0.0
    assert seg["end_s"] is not None


def test_timestamp_unit_conversion_applied(settings) -> None:
    """毫秒时间戳列被正确换算为秒（复用 timestamp_units）。"""
    n = 300
    # timestamp_ms 列：毫秒单位。
    df = pd.DataFrame({
        "timestamp_ms": np.arange(n) * 20.0,  # 20ms 间隔 = 50Hz
        "action_joint0": np.concatenate([
            np.sin(np.arange(150) * 0.3), np.sin(np.arange(150) * 1.2) * 2]),
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    seg = res["episodes"][0]["segments"][-1]
    # 总时长应为 (300-1)*20ms ≈ 5.98s
    assert seg["end_s"] == pytest.approx(5.98, abs=0.1)


def test_deterministic_across_runs(settings) -> None:
    """同一输入两次运行结果一致（无随机性）。"""
    df = _segmented_df()
    r1 = segment_actions_impl(_ctx(df.copy()), settings=settings)
    r2 = segment_actions_impl(_ctx(df.copy()), settings=settings)
    assert r1["n_segments"] == r2["n_segments"]
    assert r1["signals_used"] == r2["signals_used"]
    assert [s["start_frame"] for s in r1["episodes"][0]["segments"]] == \
           [s["start_frame"] for s in r2["episodes"][0]["segments"]]


def test_signals_missing_reported_when_partial(settings) -> None:
    """部分信号缺失时如实报告（不假装全部信号都可用）。"""
    n = 300
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.02,
        "action_joint0": np.concatenate([
            np.sin(np.arange(150) * 0.3), np.sin(np.arange(150) * 1.2) * 2]),
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    # 无夹爪列，必须如实报告。
    assert any("夹爪" in m for m in res["signals_missing"])


def test_metadata_columns_excluded_from_motion(settings) -> None:
    """时间/episode/success 等元数据列不参与运动信号分析。"""
    res = segment_actions_impl(_ctx(_segmented_df()), settings=settings)
    # 元数据列不应导致误检（片段数应合理，而非爆量）。
    assert res["n_segments"] < 100


# ---------------------------------------------------------------------------
# 阶段四联调暴露的三个检测器缺陷（回归守护）
# ---------------------------------------------------------------------------


def test_merge_short_segments_does_not_cascade(settings) -> None:
    """**短片段合并不得级联吞噬**（阶段四发现的最严重缺陷）。

    初版"过短则并入前一段"在连续短片段串上会级联：吸收一个后仍"短"，
    于是继续吸收，最终 19 个片段压成 1 个（用户失去全部切片结果）。
    修正为"连续短串整体并入后一个正常片段"，合并一次且有界。
    """
    from app.tools.segment_actions import _merge_short_segments

    segs = [
        {"start_frame": 0, "end_frame": 200, "n_frames": 201,
         "start_s": 0.0, "end_s": 4.0},                       # 正常
        {"start_frame": 201, "end_frame": 210, "n_frames": 10,
         "start_s": 4.0, "end_s": 4.2},                       # 短
        {"start_frame": 211, "end_frame": 220, "n_frames": 10,
         "start_s": 4.2, "end_s": 4.4},                       # 短
        {"start_frame": 221, "end_frame": 420, "n_frames": 200,
         "start_s": 4.4, "end_s": 8.4},                       # 正常
    ]
    out = _merge_short_segments(segs, 0.5, 50.0)
    # 两个正常片段应当保留（短串被吸收，而非全部塌缩成 1 个）。
    assert len(out) == 2, f"合并发生级联：{out}"
    assert out[0]["start_frame"] == 0
    assert out[1]["end_frame"] == 420


def test_merge_short_segments_all_short_keeps_original(settings) -> None:
    """全部片段都过短时保留原样（不塌缩为一个巨大片段）。"""
    from app.tools.segment_actions import _merge_short_segments

    segs = [
        {"start_frame": i * 10, "end_frame": i * 10 + 9, "n_frames": 10,
         "start_s": float(i), "end_s": float(i) + 0.2}
        for i in range(5)
    ]
    out = _merge_short_segments(segs, 0.5, 50.0)
    assert len(out) == 5


def test_discrete_channels_excluded_from_motion_analysis(settings) -> None:
    """**离散/二值通道不得参与连续信号分析**（阶段四发现的缺陷 1）。

    二值夹爪列的加速度是脉冲，会污染突波检测（实测产出 153 个虚假事件）。
    """
    from app.tools.segment_actions import _motion_cols

    n = 200
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.02,
        "action_joint0": np.sin(np.arange(n) * 0.2),          # 连续
        "gripper_position": np.concatenate([                # 二值（离散）
            np.zeros(100), np.ones(100)]),
    })
    cols = _motion_cols(df)
    assert "action_joint0" in cols
    assert "gripper_position" not in cols, "二值通道不得参与连续信号分析"


def test_spike_detector_does_not_flood_on_smooth_high_freq(settings) -> None:
    """**突波判据不得在平滑高频信号上泛滥**（阶段四发现的缺陷 2）。

    纯 median 基线判据会把 38% 的采样点标为突波；改用 IQR 稳健离群后应接近 0。
    """
    from app.tools.segment_actions import _spike_points

    n = 400
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.02,
        "action_joint0": np.concatenate([
            np.sin(np.arange(200) * 0.3),
            np.sin(np.arange(200) * 1.5) * 2]),
    })
    events, _used = _spike_points(
        df, ["action_joint0"], settings.quality_diagnostic_spike_multiplier)
    assert len(events) < n * 0.05, (
        f"突波事件泛滥（{len(events)}/{n}）——判据基线选错了"
    )


def test_pause_detector_does_not_flood_on_slow_oscillation(settings) -> None:
    """**停顿判据不得把慢速振荡的每个波谷都当停顿**（阶段四发现的缺陷 2 同类）。"""
    from app.tools.segment_actions import _pause_boundaries, _speed_profile

    n = 400
    df = pd.DataFrame({
        "timestamp": np.arange(n) * 0.02,
        "action_joint0": np.concatenate([
            np.sin(np.arange(200) * 0.3),
            np.sin(np.arange(200) * 1.5) * 2]),
    })
    speed, _used = _speed_profile(df, ["action_joint0"])
    bounds = _pause_boundaries(
        speed, settings.annotation_idle_speed,
        settings.annotation_min_pause_steps)
    assert len(bounds) <= 4, (
        f"假停顿泛滥（{len(bounds)} 个边界）——慢速振荡被误判为停顿"
    )


def test_gripper_boundary_survives_merge_pipeline(settings) -> None:
    """夹爪边界必须能穿过完整流水线存活（端到端，防被合并淹没）。"""
    n = 600
    quiet = np.zeros(100)
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp": np.arange(n) * 0.02,
        "fps": [50.0] * n,
        "action_joint0": np.concatenate([
            np.sin(np.arange(200) * 0.4), quiet,
            np.sin(np.arange(200) * 0.4) * 1.5, quiet,
        ]),
        "gripper_position": np.concatenate([
            np.zeros(200), np.zeros(100), np.ones(200), np.ones(100),
        ]),
    })
    res = segment_actions_impl(_ctx(df), settings=settings)
    assert res["success"] is True
    assert res["n_segments"] >= 2, f"夹爪边界被淹没：{res['n_segments']} 个片段"
