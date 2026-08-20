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
