"""app/tools/generate_report 工具的单元测试。

覆盖：空 findings 返回结构化错误、完整 findings+qc 时报告各章节齐全、图片引用
为相对路径、局限性章节包含 skipped/unknown 项。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agents.tool import FunctionTool

from app.agent.context import RunContext
from app.tools.generate_report import generate_report, generate_report_impl


def _ctx(output_dir: str, findings=None, qc=None, meta=None) -> RunContext:
    meta = meta or {}
    if qc is not None:
        meta["qc"] = qc
    return RunContext(dataset_id="demo", df=None, meta=meta,
                      findings=findings or [], output_dir=output_dir)


def test_generate_report_is_registered() -> None:
    assert isinstance(generate_report, FunctionTool)
    assert generate_report.name == "generate_report"


def test_empty_findings_returns_error(tmp_path: Path) -> None:
    ctx = _ctx(output_dir=str(tmp_path))
    r = generate_report_impl(ctx)
    assert r["success"] is False
    assert r["error"] == "no_findings"


def test_full_report_all_sections(tmp_path: Path) -> None:
    findings = [
        {"tool": "compute_stats", "type": "stat", "metric": "success_rate",
         "n_episodes": 5, "summary": "5 个 episode；成功率 1.0"},
        {"tool": "plot_chart", "type": "chart", "file_path": "outputs/demo_line_1.png",
         "title": "line chart", "description": "Line of value",
         "plot_spec": {"x_axis": "timestamp", "y_axis": ["value"], "grouped_by": None, "n_series": 1}},
    ]
    qc = {
        "check_temporal_sync": {"result": "warn", "verification_level": "timestamp_consistency"},
        "check_sensor_sanity": {"result": "pass", "constant_channels": []},
    }
    meta = {"capabilities": {"has_imu": True, "imu_axes": 6},
            "streams": [{"path": "imu.csv", "format": "csv", "kind": "imu", "channels": []}],
            "guessed_type": "IMU 数据"}
    ctx = _ctx(findings=findings, qc=qc, output_dir=str(tmp_path), meta=meta)

    r = generate_report_impl(ctx)

    assert r["success"] is True
    # 报告文件落盘。
    md_path = Path(r["file_path"])
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    # 各章节齐全。
    assert "数据集概况" in content
    assert "质检结果" in content
    assert "任务级统计" in content
    assert "图表" in content
    assert "局限性说明" in content
    # 章节条目数。
    assert r["section_counts"]["qc"] == 2
    assert r["section_counts"]["stats"] == 1
    assert r["section_counts"]["charts"] == 1


def test_image_relative_path(tmp_path: Path) -> None:
    findings = [
        {"tool": "plot_chart", "type": "chart", "file_path": "outputs/demo_line_1.png",
         "title": "line chart", "description": "Line of value",
         "plot_spec": {"x_axis": "timestamp", "y_axis": ["value"], "grouped_by": None, "n_series": 1}},
    ]
    ctx = _ctx(findings=findings, output_dir=str(tmp_path))

    r = generate_report_impl(ctx)
    content = Path(r["file_path"]).read_text(encoding="utf-8")

    # 图片引用为相对路径（含 ../outputs/，非绝对路径）。
    assert "![](../outputs/demo_line_1.png)" in content
    assert "C:" not in content.split("![](../outputs/")[0]  # 无绝对路径前缀


def test_limitations_contains_skipped_unknown(tmp_path: Path) -> None:
    findings = [
        {"tool": "compute_stats", "type": "stat", "metric": "success_rate",
         "n_episodes": 1, "summary": "success 聚合规则为推测；无法确定单位"},
    ]
    qc = {
        "check_temporal_sync": {"result": "fail", "verification_level": "timestamp_consistency"},
    }
    meta = {"guessed_type": "unknown"}
    ctx = _ctx(findings=findings, qc=qc, output_dir=str(tmp_path), meta=meta)

    r = generate_report_impl(ctx)
    content = Path(r["file_path"]).read_text(encoding="utf-8")

    # 局限性章节应包含 skipped/unknown/推测 相关项。
    assert r["section_counts"]["limitations"] >= 1
    assert "推测" in content or "无法确定" in content or "unknown" in content


def test_findings_append_report(tmp_path: Path) -> None:
    findings = [
        {"tool": "compute_stats", "type": "stat", "metric": "success_rate",
         "n_episodes": 5, "summary": "5 个 episode"},
    ]
    ctx = _ctx(findings=findings, output_dir=str(tmp_path))

    generate_report_impl(ctx)

    # findings 追加了 type=report 条目。
    assert ctx.findings[-1]["type"] == "report"


def test_report_stats_section_contains_semantic_notes(tmp_path: Path) -> None:
    """报告统计章节应列出语义注释（推测标注）。"""
    findings = [
        {"tool": "compute_stats", "type": "stat", "metric": "success_rate",
         "n_episodes": 5, "summary": "5 个 episode；成功率 1.0",
         "semantic_notes": ["success 聚合规则为推测，建议人工确认"]},
    ]
    ctx = _ctx(findings=findings, output_dir=str(tmp_path))

    r = generate_report_impl(ctx)
    content = Path(r["file_path"]).read_text(encoding="utf-8")

    # 统计章节含"推测"标注。
    assert "success 聚合规则为推测" in content


def test_limitations_generalizes_all_semantic_notes(tmp_path: Path) -> None:
    """局限性章节应汇总所有 findings 的 semantic_notes（通用化，不只 stat）。"""
    findings = [
        {"tool": "compute_stats", "type": "stat", "metric": "success_rate",
         "n_episodes": 5, "summary": "5 个 episode",
         "semantic_notes": ["success 聚合规则为推测"]},
        {"tool": "plot_chart", "type": "chart", "file_path": "outputs/x.png",
         "title": "chart", "description": "desc",
         "plot_spec": {"x_axis": "x", "y_axis": ["y"], "grouped_by": None, "n_series": 1},
         "semantic_notes": ["该图表基于推测的时间戳对齐"]},
    ]
    ctx = _ctx(findings=findings, output_dir=str(tmp_path))

    r = generate_report_impl(ctx)
    content = Path(r["file_path"]).read_text(encoding="utf-8")

    # 局限性章节应包含两个工具（stat 与 chart）的 semantic_notes。
    assert "success 聚合规则为推测" in content
    assert "该图表基于推测的时间戳对齐" in content


def test_report_profile_contains_stream_details(tmp_path: Path) -> None:
    """数据集画像章节应含流明细表（角色/格式/采样率/来源文件）。"""
    findings = [
        {"tool": "compute_stats", "type": "stat", "metric": "success_rate",
         "n_episodes": 5, "summary": "5 个 episode"},
    ]
    meta = {
        "capabilities": {"has_imu": True, "imu_axes": 6, "has_force": True, "has_actions": True},
        "streams": [
            {"path": "imu.csv", "format": "csv", "kind": "imu",
             "role": {"role": "IMU 传感器"}, "measured_rate": {"sample_rate_hz": 100.0}},
            {"path": "ft.csv", "format": "csv", "kind": "force",
             "role": {"role": "力/力矩传感器"}, "measured_rate": None},
        ],
        "guessed_type": "IMU 数据",
    }
    ctx = _ctx(output_dir=str(tmp_path), findings=findings, meta=meta)

    r = generate_report_impl(ctx)
    content = Path(r["file_path"]).read_text(encoding="utf-8")

    # 画像含流明细表（流名/角色/采样率）。
    assert "流明细" in content
    assert "IMU 传感器" in content
    assert "100.0 Hz" in content or "100 Hz" in content
    # 模态矩阵。
    assert "模态矩阵" in content
    assert "视频流" in content


def test_report_qc_section_contains_measurements(tmp_path: Path) -> None:
    """质检章节应含测量值与阈值（检查明细表）。"""
    findings = [
        {"tool": "compute_stats", "type": "stat", "metric": "success_rate",
         "n_episodes": 5, "summary": "5 个 episode"},
    ]
    qc = {
        "check_temporal_sync": {
            "result": "warn", "verification_level": "timestamp_consistency",
            "detail": {
                "stream_checks": {
                    "imu.csv": {"present": True, "disorder_count": 0, "duplicate_count": 0,
                                "frame_loss_ratio": 0.01, "actual_rate_hz": 100.0},
                },
                "residuals": {"ft.csv": {"residual_max_ms": 5.0, "residual_mean_ms": 5.0}},
                "drift": {"imu.csv": {"drift_slope_ms_per_s": 2.0, "drift_detected": False}},
                "thresholds": {"frame_loss_ratio": 0.02, "residual_threshold_ms": 5.0},
            },
        },
    }
    ctx = _ctx(output_dir=str(tmp_path), findings=findings, qc=qc)

    r = generate_report_impl(ctx)
    content = Path(r["file_path"]).read_text(encoding="utf-8")

    # 质检章节含测量值（丢帧率、残差）与阈值。
    assert "0.01" in content  # 丢帧率测量值
    assert "5.0 ms" in content  # 残差测量值
    assert "残差" in content
