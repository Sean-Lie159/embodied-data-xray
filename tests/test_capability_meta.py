"""能力标签注册表与模态矩阵动态化测试。

设计依据：``docs/指标单一来源与数据集画像设计.md`` 第二部分（B-3.1）。

**本文件最重要的守护**：``test_every_capability_has_chinese_label``——
遍历 ``capabilities`` 的全部键，断言都在注册表中。这防止将来新增能力标签时
漏配中文名，导致模态矩阵静默丢掉一行（**这正是 has_pose 消失的原因**）。
"""

from __future__ import annotations

from app.tools.capability_meta import (
    absent_note,
    build_capability_matrix,
    capability_label,
    capability_what,
    is_detail,
)


# 生产代码中 capabilities 的实际键集合（_sniffing.py:2238-2249）。
_PRODUCTION_CAPABILITY_KEYS = {
    "has_video_streams",
    "has_audio",
    "has_imu",
    "imu_axes",
    "has_force",
    "has_calibration",
    "has_actions",
    "has_hand_tracking",
    "has_pose",
    "dataset_format",
}


def test_every_capability_has_chinese_label() -> None:
    """**守护**：capabilities 的每个实际键都必须有中文名。

    这是防止"模态矩阵漏行"这类缺陷复发的核心测试——has_pose 曾因未出现在
    硬编码列表里而在报告中消失。
    """
    for key in _PRODUCTION_CAPABILITY_KEYS:
        label = capability_label(key)
        assert label != key, f"能力标签 {key} 未登记中文名"
        assert capability_what(key), f"能力标签 {key} 缺少白话说明"


def test_has_pose_is_registered() -> None:
    """**has_pose 必须被登记**（本次缺陷的主角）。"""
    assert capability_label("has_pose") == "位姿"
    assert not is_detail("has_pose")


def test_matrix_includes_pose_row() -> None:
    """模态矩阵必须包含「位姿」行（回归守护）。"""
    caps = {"has_video_streams": False, "has_imu": False, "has_pose": True}
    m = build_capability_matrix(caps)
    labels = [r["label"] for r in m["rows"]]
    assert "位姿" in labels, f"矩阵缺少位姿行：{labels}"
    pose = next(r for r in m["rows"] if r["label"] == "位姿")
    assert pose["present"] is True


def test_matrix_iterates_all_capabilities_not_hardcoded() -> None:
    """矩阵必须遍历**全部**键（未来新增的也要出现，而非硬编码子集）。"""
    caps = dict.fromkeys(_PRODUCTION_CAPABILITY_KEYS, False)
    caps["has_pose"] = True
    m = build_capability_matrix(caps)

    matrix_keys = {r["key"] for r in m["rows"]} | {d["key"] for d in m["details"]}
    assert matrix_keys == _PRODUCTION_CAPABILITY_KEYS


def test_matrix_details_hold_non_boolean_capabilities() -> None:
    """附注型（轴数/格式名）不占矩阵行，进 details。"""
    caps = {
        "has_imu": True,
        "imu_axes": 6,
        "dataset_format": "lerobot",
        "has_pose": True,
    }
    m = build_capability_matrix(caps)
    row_keys = {r["key"] for r in m["rows"]}
    detail_keys = {d["key"] for d in m["details"]}
    assert "imu_axes" not in row_keys
    assert "dataset_format" not in row_keys
    assert "imu_axes" in detail_keys
    assert "dataset_format" in detail_keys


def test_matrix_preserves_none_as_undetected() -> None:
    """None（未探测）与 False（确认没有）必须区分——三态不可混同。"""
    caps = {"has_imu": False, "has_force": None}
    m = build_capability_matrix(caps)
    by_key = {r["key"]: r for r in m["rows"]}
    assert by_key["has_imu"]["present"] is False
    assert by_key["has_force"]["present"] is None


def test_absent_note_explains_missing_is_not_defect() -> None:
    """缺失说明须能安抚"没有≠缺陷"（避免用户误读）。"""
    assert absent_note("has_video_streams")
    caps = {"has_video_streams": False}
    m = build_capability_matrix(caps)
    assert m["rows"][0]["absent_note"]
    # 存在时不给 absent_note（不无谓地解释）。
    m2 = build_capability_matrix({"has_video_streams": True})
    assert m2["rows"][0]["absent_note"] == ""


def test_unregistered_key_is_surfaced_not_hidden() -> None:
    """未登记的键必须**显式暴露**（而非静默丢弃）。"""
    caps = {"has_pose": True, "brand_new_capability": True}
    m = build_capability_matrix(caps)
    assert "brand_new_capability" in m["unregistered"]
    # 仍出一行，用原键名（不编造中文名）。
    labels = [r["label"] for r in m["rows"]]
    assert "brand_new_capability" in labels


def test_matrix_handles_empty_and_invalid_input() -> None:
    """空/非法输入返回空结构（不抛异常）。"""
    assert build_capability_matrix({})["rows"] == []
    assert build_capability_matrix(None)["rows"] == []  # type: ignore[arg-type]


def test_capability_label_falls_back_to_key() -> None:
    """未登记时原样返回键名（不猜中文名）。"""
    assert capability_label("unknown_key_xyz") == "unknown_key_xyz"
    assert capability_what("unknown_key_xyz") == ""


# ---------------------------------------------------------------------------
# 报告集成
# ---------------------------------------------------------------------------


def test_report_matrix_contains_pose_and_no_phantom_language_row() -> None:
    """报告模态矩阵：须含位姿、且不得出现幽灵行「语言标注」。

    初版硬编码列表里有 ``("语言标注", caps.get("has_language"))``——
    但 ``has_language`` **根本不在 capabilities 里**（永远是 None → 显示 ✗），
    属"凭空多出的假行"。改为动态生成后该行自然消失。
    """
    from app.agent.context import RunContext
    from app.tools.generate_report import _build_dataset_overview

    ctx = RunContext()
    ctx.dataset_id = "demo"
    ctx.meta = {
        "capabilities": {
            "has_video_streams": False,
            "has_imu": False,
            "has_force": False,
            "has_calibration": False,
            "has_actions": False,
            "has_pose": True,
            "has_hand_tracking": True,
            "has_audio": False,
            "imu_axes": None,
            "dataset_format": "unknown",
        },
        "streams": [],
        "format": "csv",
        "n_rows": 100,
    }
    text = _build_dataset_overview(ctx)

    assert "位姿" in text, "模态矩阵缺少位姿行"
    assert "手部追踪" in text
    assert "语言标注" not in text, "不应出现 capabilities 中不存在的幽灵行"
    # 位姿应为 ✓。
    pose_line = next(
        (ln for ln in text.splitlines() if "位姿" in ln and ln.startswith("|")), "")
    assert "✓" in pose_line


def test_report_surfaces_unregistered_capability_warning() -> None:
    """报告须提示未登记的能力标签（让遗漏可见）。"""
    from app.agent.context import RunContext
    from app.tools.generate_report import _build_dataset_overview

    ctx = RunContext()
    ctx.dataset_id = "demo"
    ctx.meta = {
        "capabilities": {"has_pose": True, "totally_new_thing": True},
        "streams": [],
        "format": "csv",
        "n_rows": 1,
    }
    text = _build_dataset_overview(ctx)
    assert "未登记中文名" in text
    assert "totally_new_thing" in text
