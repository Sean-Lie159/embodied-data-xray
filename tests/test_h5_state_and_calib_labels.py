"""h5 state/* 遥测流与 extrinsic 标定的语义标签（2026-09-21 缺陷 A+B）。

背景（真实数据集 ``2655849``，实测 106 条流的 kind 分布）：

    {timestamp_index: 16, unknown: 49, actions: 8, video: 8, log: 24, pose: 1}

排查后确认 ``log`` 24 条正确、``unknown`` 里 20 条空流正确，**真问题是 29 条**：

- **缺陷 A（24 条）**：h5 的 ``state/*`` 遥测流全部误报"未知（无法分类）"。
  根因：帧布局下这些节点是**裸 ndarray**，``_classify_h5_leaf`` 为其返回
  ``fields=[]``，于是 ``_classify_h5_node`` 里所有依赖 ``fl`` 的判据
  （``any("quat" in f ...)`` 等）**恒为 False**；state 路径又不含
  ``action``/``pose``/``imu`` 等既有词根 → 落到 unknown。
  反证：同数据的 ``action/joint/position`` 因路径含 ``action`` 被正确识别。
- **缺陷 B（5 条）**：``parameters/sensor/extrinsic_end_T_*_aligned.json``
  是相机外参变换矩阵，语义在文件名上很明确，但走的是
  ``_sniffing.classify_table_stream``（**非 h5 节点路径**），且文件名不含
  ``calibration`` 字样 → 落 unknown。

修复后 ``unknown`` 49 → 20（**且 20 条全是 status="empty" 的合法空流**，
无真正的未分类流），新增 ``joint_state``/``effort``/``status``/``wrench``
与 ``calibration``。

另附一处**隐患修复**：``state/end/wrench`` 最初被我归入 ``kind="force"``，
但 ``force`` 在 ``inspect_streams`` 里是**单槽汇总**（单个 dict，后写覆盖先写），
且该槽期望列名清单而裸数组 channels 恒为空 → 实测产出
``present=True, n_channels=0`` 的自相矛盾结果。改用独立 ``kind="wrench"``。
"""

from __future__ import annotations

import sys

import pytest

import app.tools.load_dataset  # noqa: F401  确保模块已加载

_ld = sys.modules["app.tools.load_dataset"]
_classify_h5_node = _ld._classify_h5_node
_classify_state_leaf = _ld._classify_state_leaf
_is_calibration_extrinsic = _ld._is_calibration_extrinsic

# 真实数据集实测的 state 流 → 期望 kind（作为测试锚点）。
_EXPECTED_STATE_KINDS = {
    "state/joint/position": "joint_state",
    "state/joint/velocity": "joint_state",
    "state/joint/effort": "effort",
    "state/joint/mode": "status",
    "state/waist/position": "joint_state",
    "state/waist/velocity": "joint_state",
    "state/waist/effort": "effort",
    "state/waist/mode": "status",
    "state/head/position": "joint_state",
    "state/head/velocity": "joint_state",
    "state/head/effort": "effort",
    "state/head/mode": "status",
    "state/robot/position": "joint_state",
    "state/robot/orientation": "pose",
    "state/end/position": "joint_state",
    "state/end/velocity": "joint_state",
    "state/end/pose": "pose",
    "state/end/orientation": "pose",
    "state/end/arm_position": "pose",
    "state/end/arm_orientation": "pose",
    "state/end/wrench": "wrench",
    "state/end/mode": "status",
    "state/end/errcode": "status",
    "state/left_effector/position": "joint_state",
    "state/right_effector/position": "joint_state",
}


# --- 缺陷 A：state/* 归类 ----------------------------------------------------


@pytest.mark.parametrize(("path", "kind"), sorted(_EXPECTED_STATE_KINDS.items()))
def test_state_leaf_classified(path: str, kind: str) -> None:
    """**核心**：24 条真实 state 路径全部归类（此前全为 unknown）。

    注意 fields 传 ``[]``——这正是真实情形（帧布局下裸 ndarray 无列名），
    也是原缺陷的触发条件。
    """
    got_kind, label = _classify_h5_node([], path)
    assert got_kind == kind, f"{path} 期望 {kind}，实际 {got_kind}（{label}）"
    assert label and "未知" not in label


def test_state_leaf_handles_empty_fields() -> None:
    """**根因回归**：fields 为空时也必须能归类（缺陷的直接成因）。"""
    # 有 fields 时能识别（旧行为）：
    assert _classify_h5_node(["quat_w", "quat_x"], "state/end/orientation")[0] == "pose"
    # 无 fields 时（真实情形）也必须识别——此前这里返回 unknown。
    assert _classify_h5_node([], "state/end/orientation")[0] == "pose"
    assert _classify_h5_node([], "state/joint/position")[0] == "joint_state"


def test_state_leaf_unclassifiable_returns_none() -> None:
    """无法归类的 state 叶子返回 None（交由调用方标 unknown，不硬猜）。

    注意中间段兜底：``state/joint/<未知名>`` 会因中间段 ``joint`` 命中而判为
    关节状态流（**这是有意行为**——"joint" 段已足以说明是关节类数据）。
    只有连中间段也无语义词时才返回 None。
    """
    # 无任何语义词段 → None（不硬猜）。
    assert _classify_state_leaf("state/whatever") is None
    assert _classify_state_leaf("state/foo/bar") is None
    # 中间段有语义词 → 按中间段归类（有意行为，非缺陷）。
    assert _classify_state_leaf("state/joint/whatever")[0] == "joint_state"


def test_state_rule_order_wrench_before_position() -> None:
    """**顺序敏感回归**：``wrench`` 必须先于通用位置规则被判出。

    ``state/end/wrench`` 若被 ``position`` 类规则先吃掉会误标关节状态。
    """
    assert _classify_state_leaf("state/end/wrench")[0] == "wrench"
    assert _classify_state_leaf("state/end/torque")[0] == "wrench"
    # effort 不能被 position 抢。
    assert _classify_state_leaf("state/joint/effort")[0] == "effort"
    assert _classify_state_leaf("state/head/effort")[0] == "effort"


# --- 回归：既有分类不被抢占 --------------------------------------------------


def test_action_streams_not_captured_by_state_rules() -> None:
    """**零回归**：``action/*`` 仍为动作流（state 规则不得抢占）。"""
    for p in ("action/joint/position", "action/end/orientation",
              "action/waist/effort", "action/end/wrench"):
        kind, label = _classify_h5_node([], p)
        assert kind == "actions", f"{p} 被抢占为 {kind}（{label}）"


def test_timestamp_streams_not_captured() -> None:
    """**零回归**：时间戳流仍为 timestamp_index（state 规则不得抢占）。"""
    assert _classify_h5_node([], "main_timestamp")[0] == "timestamp_index"
    assert _classify_h5_node(
        [], "timestamp/camera/head_color")[0] == "timestamp_index"


def test_state_timestamp_like_leaf_not_treated_as_timestamp_stream() -> None:
    """``state/`` 路径下的时间戳词叶子不得被判为时间戳流（路径白名单）。"""
    assert _classify_h5_node([], "state/joint/timestamp")[0] != "timestamp_index"


def test_wrench_does_not_use_force_kind() -> None:
    """**隐患回归**：wrench 不得用 ``kind="force"``。

    ``force`` 在 inspect_streams 是单槽汇总（后写覆盖先写），且期望列名清单；
    裸数组 channels 恒为空 → 会产出 present=True 但 n_channels=0 的矛盾结果。
    """
    kind, _ = _classify_h5_node([], "state/end/wrench")
    assert kind == "wrench", "wrench 不得归入单槽的 force kind"
    # 多条 wrench 都应是同一独立 kind（可在列表中共存、互不覆盖）。
    kinds = {_classify_h5_node([], f"state/{part}/wrench")[0]
             for part in ("end", "left_effector", "right_effector")}
    assert kinds == {"wrench"}


# --- 缺陷 B：标定文件 --------------------------------------------------------


@pytest.mark.parametrize("name", [
    "extrinsic_end_T_hand_left_rgbd_aligned.json",
    "extrinsic_end_T_hand_right_rgbd_aligned.json",
    "extrinsic_end_T_head_front_rgbd_aligned.json",
    "extrinsic_end_T_head_left_stereo_aligned.json",
    "extrinsic_end_T_head_right_stereo_aligned.json",
])
def test_extrinsic_files_recognized_as_calibration(name: str) -> None:
    """**核心**：5 条真实外参文件名全部识别为标定（此前 unknown）。"""
    assert _is_calibration_extrinsic(name) is True


def test_calibration_label_distinguishes_extrinsic_intrinsic() -> None:
    """外参与内参给不同标签（不笼统标"标定数据"）。"""
    assert "外参" in _ld._calibration_label("extrinsic_end_T_head_rgbd.json")
    assert "内参" in _ld._calibration_label("intrinsic_head_front_rgb.json")
    # ``<A>_T_<B>`` 变换矩阵命名惯例 → 外参。
    assert "外参" in _ld._calibration_label("end_T_head_rgbd.json")


def test_non_calibration_names_not_matched() -> None:
    """**防误报**：普通文件名不得命中标定判据。"""
    for name in ("imu_data.json", "episodes.jsonl", "camera_dlb.INFO",
                 "struct_chassis.json", "head_manifest.json"):
        assert _is_calibration_extrinsic(name) is False, f"{name} 被误判为标定"


def test_plain_calibration_dir_still_works() -> None:
    """既有行为不破：路径含 calibration 仍判标定（通用标签）。"""
    kind, label = _classify_h5_node([], "calibration/foo")
    assert kind == "calibration"
    assert label == "标定数据"


def test_short_calibration_file_keeps_empty_label() -> None:
    """**设计决策回归**：≤2 行的标定文件保持"未使用/空流"，不贴标定标签。

    空流判定在 ``classify_table_stream`` 中位于标定分支**之前**。这是刻意的
    「空流优先」：给 0~2 行的文件贴"相机外参"会夸大其可用性（用户可能以为
    那是可用的外参数据）。真实数据集里 ``intrinsic_*.json`` 正是这种情形。
    """
    from app.tools._sniffing import classify_table_stream

    res = classify_table_stream(
        "extrinsic_end_T_head_front_rgbd_aligned.json", [], None, 1)
    assert res["kind"] == "unknown"
    assert res["semantic_label"] == "未使用/空流"
    assert res["status"] == "empty"


def test_nonempty_extrinsic_json_classified_as_calibration() -> None:
    """多行外参 json 走标定分支（不落 unknown）。"""
    from app.tools._sniffing import classify_table_stream

    res = classify_table_stream(
        "extrinsic_end_T_head_front_rgbd_aligned.json", [], None, 4)
    assert res["kind"] == "calibration"
    assert "外参" in res["semantic_label"]
    assert res["label_confidence"] == "medium"  # 命名线索，非内容验证


def test_intrinsic_json_classified_as_intrinsic() -> None:
    """内参 json 走标定分支且标签区分为内参。"""
    from app.tools._sniffing import classify_table_stream

    res = classify_table_stream("intrinsic_head_front_rgb.json", [], None, 4)
    assert res["kind"] == "calibration"
    assert "内参" in res["semantic_label"]


# --- 端到端 -----------------------------------------------------------------


def test_end_to_end_unknown_drops_to_only_empty(tmp_path) -> None:
    """**端到端**：构造真实形态数据集，确认 unknown 只剩空流。

    构造：h5（含 state/* 帧内裸数组 + main_timestamp）、camera/ 同名 txt、
    以及 parameters/sensor/extrinsic_*.json。
    """
    import numpy as np

    h5py = pytest.importorskip("h5py")
    from pathlib import Path

    from app.agent.context import RunContext

    root = tmp_path / "ds"
    rec = root / "record"
    rec.mkdir(parents=True)
    n_frames = _ld._FRAME_LAYOUT_MIN_GROUPS + 5
    T0 = 1_756_265_284_805_200_809

    with h5py.File(rec / "aligned_joints.h5", "w") as f:
        for i in range(n_frames):
            g = f.create_group(str(i))
            g.create_dataset("main_timestamp", data=np.uint64(T0 + i * 33_447_424))
            g.create_dataset("state/joint/position", data=np.arange(14, dtype=float) + i)
            g.create_dataset("state/end/wrench", data=np.arange(12, dtype=float) + i)
            g.create_dataset("state/joint/effort", data=np.arange(5, dtype=float) + i)
            g.create_dataset("state/head/mode", data=np.uint8(i % 3))

    sensor = root / "parameters" / "sensor"
    sensor.mkdir(parents=True)
    # **与真实文件同形**：真实 ``extrinsic_*.json`` 是 4.1MB / 14135 行的
    # list（逐帧变换矩阵），故非空、走标定分支。
    #
    # 注意行数必须 > EMPTY_STREAM_MAX_ROWS(2)：标定分支在 classify_table_stream
    # 里位于空流判定**之后**，这是刻意的「空流优先」——真实 ``intrinsic_*.json``
    # 是 192 字节的单 dict（0 行），被判空流而非"相机内参"，避免给 0 行文件贴
    # 有意义的标签、夸大其可用性。
    (sensor / "extrinsic_end_T_head_front_rgbd_aligned.json").write_text(
        "[" + ",".join(
            '{"rotation": [[1,0,0],[0,1,0],[0,0,1]], "translation": [0, 0, %d]}' % i
            for i in range(6)
        ) + "]",
        encoding="utf-8")

    ctx = RunContext(output_dir=str(tmp_path), dataset_id=None)
    assert _ld.load_dataset_impl(ctx, str(root))["success"] is True

    by_name = {str(s.get("path")).replace("\\", "/"): s
               for s in ctx.meta["streams"]}

    # state/* 不再 unknown。
    joint = next(v for k, v in by_name.items() if k.endswith("::state/joint/position"))
    assert joint["kind"] == "joint_state"
    wrench = next(v for k, v in by_name.items() if k.endswith("::state/end/wrench"))
    assert wrench["kind"] == "wrench"
    effort = next(v for k, v in by_name.items() if k.endswith("::state/joint/effort"))
    assert effort["kind"] == "effort"
    mode = next(v for k, v in by_name.items() if k.endswith("::state/head/mode"))
    assert mode["kind"] == "status"

    # extrinsic 不再 unknown。
    ext = next(v for k, v in by_name.items() if "extrinsic" in k)
    assert ext["kind"] == "calibration"

    # 剩余 unknown：只允许存在"空流"这一类（status="empty"）。
    # 本 fixture 的文件都非空，故实际应为 **0 条** unknown——即全部流都拿到
    # 了有意义的标签。这比"只剩空流"更强，故同时断言无未知流。
    leftovers = [s for s in ctx.meta["streams"] if s.get("kind") == "unknown"]
    assert all(s.get("status") == "empty" for s in leftovers), (
        f"仍有非空 unknown：{[s.get('path') for s in leftovers]}"
    )
    assert leftovers == [], (
        f"本 fixture 无空文件，不应有 unknown：{[s.get('path') for s in leftovers]}"
    )

    # 空流在真实数据里确实存在（intrinsic_*.json 单 dict 0 行）——那类保持
    # "未使用/空流"标签（见 test_short_calibration_file_keeps_empty_label）。
    # 本用例单独锚定"非空文件不给空流标签"这一侧。


def test_path_based_labels_carry_semantic_assumption_caveat() -> None:
    """**诚实性回归**：路径判定的标签须在 evidence 里声明是语义假设。

    按 AGENTS.md §3.4「LLM 假设不得替代工具验证」——标签仅供理解清单，
    不得据此做数值结论。
    """
    import numpy as np

    h5py = pytest.importorskip("h5py")
    from app.agent.context import RunContext

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        rec = root / "record"
        rec.mkdir(parents=True)
        n_frames = _ld._FRAME_LAYOUT_MIN_GROUPS + 5
        with h5py.File(rec / "a.h5", "w") as f:
            for i in range(n_frames):
                g = f.create_group(str(i))
                g.create_dataset("state/joint/position", data=np.arange(14, dtype=float))
        ctx = RunContext(output_dir=td, dataset_id=None)
        assert _ld.load_dataset_impl(ctx, str(root))["success"] is True

        s = next(x for x in ctx.meta["streams"]
                 if str(x.get("path")).endswith("::state/joint/position"))
        ev = str(s.get("label_evidence"))
        assert "语义假设" in ev or "未经内容验证" in ev, ev
        assert s.get("label_source") == "h5_node_scan_path", s.get("label_source")
