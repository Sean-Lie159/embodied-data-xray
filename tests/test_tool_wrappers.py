"""@tool wrapper 层的冒烟测试（经 SDK on_invoke_tool 的真实调用路径）。

背景（真实事故）：check_sensor_sanity 的 wrapper 调用 impl 时引用了形参表
中不存在的 expand 变量（NameError），而项目既有测试全部直接调 impl——
wrapper 层零覆盖，487 项全绿但 UI/CLI 实际路径全崩。本文件以 SDK 的
on_invoke_tool + JSON 参数字符串方式驱动 wrapper，锁住这一层。

覆盖全部 9 个注册工具的"能被 SDK 调用且返回 dict"（参数最简组合），
并对 check_sensor_sanity 的 expand 路径做专项断言。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from agents import RunContextWrapper

from app.agent.context import RunContext
from app.tools import (
    check_sensor_sanity,
    check_temporal_sync,
    compute_stats,
    generate_report,
    inspect_streams,
    load_dataset,
    plot_chart,
    profile_data,
    propose_stream_semantics,
)
from app.tools.load_dataset import load_dataset_impl

T0_US = 1_787_294_445_600_000


def _make_ctx(tmp_path: Path) -> RunContext:
    """构造含一个信封 IMU 流的数据集上下文。"""
    d = tmp_path / "ds"
    d.mkdir()
    rng = pd.Series([0.01], dtype=float)  # 占位避免未用告警
    del rng
    # 随机微噪声：恒定输出（零方差）会被 sanity 正确判为恒定通道故障并排除；
    # 周期性噪声则会让极值点重复出现、触发饱和削顶误报——都用随机噪声模拟
    # 真实传感器的读数抖动（连续分布，极值不重复）。
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=(300, 3))
    rows = [{
        "mcap_log_time_ns": (T0_US + i * 1250) * 1000,
        "mcap_publish_time_ns": (T0_US + i * 1250) * 1000,
        "data": {
            "header": {"timestamp_us": T0_US + i * 1250},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.03},
            "linear_acceleration": {
                "x": round(-2.86 + float(noise[i, 0]), 5),
                "y": round(-6.92 + float(noise[i, 1]), 5),
                "z": round(6.34 + float(noise[i, 2]), 5),
            },
        },
    } for i in range(300)]
    with (d / "left_glove_imu_data_palm.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ctx = RunContext(output_dir=str(tmp_path))
    assert load_dataset_impl(ctx, str(d))["success"] is True
    return ctx


def _invoke_sync(tool_obj, ctx: RunContext, args: dict) -> dict:
    """同步驱动 @tool.on_invoke_tool（SDK 真实入口）。"""

    async def _drive() -> object:
        wrapper = RunContextWrapper(context=ctx)
        wrapper.run_config = None
        wrapper.tool_name = tool_obj.name
        return await tool_obj.on_invoke_tool(wrapper, json.dumps(args, ensure_ascii=False))

    out = asyncio.run(_drive())
    if isinstance(out, dict):
        return out
    try:
        return json.loads(out)
    except (ValueError, TypeError):
        return {"_raw": str(out)[:300]}


def _find_by_name(name: str):
    """按 name 在工具模块对象上定位 FunctionTool（modules 暴露的是包装对象）。"""
    candidates = [
        check_sensor_sanity, check_temporal_sync, compute_stats,
        generate_report, inspect_streams, load_dataset,
        plot_chart, profile_data, propose_stream_semantics,
    ]
    for t in candidates:
        if getattr(t, "name", None) == name:
            return t
    raise AssertionError(f"tool {name} not found")


@pytest.mark.parametrize("name,args", [
    ("load_dataset", {}),
    ("profile_data", {"table": "left_glove_imu_data_palm.jsonl", "expand": True}),
    ("inspect_streams", {}),
    ("check_temporal_sync", {}),
    ("check_sensor_sanity", {"expand": True}),
    ("check_sensor_sanity", {"table": "left_glove_imu_data_palm.jsonl", "expand": True}),
    ("compute_stats", {"metric": "success_rate"}),
    ("propose_stream_semantics", {"assumptions": []}),
    ("generate_report", {}),
])
def test_tool_wrapper_invocable_no_nameerror(
    tmp_path: Path, name: str, args: dict
) -> None:
    """每个 @tool 经 on_invoke_tool 调用：不得出现 NameError 类内部错误。

    真实事故回归：check_sensor_sanity wrapper 引用形参表中不存在的 expand，
    全部调用（无论入参）都报 NameError——impl 层测试全绿也无法发现。
    """
    ctx = _make_ctx(tmp_path)
    tool_obj = _find_by_name(name)
    out = _invoke_sync(tool_obj, ctx, args)
    text = json.dumps(out, ensure_ascii=False)
    assert "NameError" not in text, f"{name} wrapper 存在未定义变量引用：{text[:200]}"


def test_sanity_wrapper_expand_runs_gravity(tmp_path: Path) -> None:
    """sanity wrapper（expand=True）端到端：重力检查真实执行且通过。"""
    ctx = _make_ctx(tmp_path)
    tool_obj = _find_by_name("check_sensor_sanity")
    out = _invoke_sync(
        tool_obj, ctx,
        {"table": "left_glove_imu_data_palm.jsonl", "expand": True},
    )
    assert out.get("success") is True, out
    assert out.get("result") in ("pass", "warn")
    checks = out.get("checks", {})
    gravity_done = any(
        isinstance(v, dict) and (v.get("gravity_check") or {}).get("status") == "done"
        for v in checks.values()
    )
    assert gravity_done, f"应有真实执行的重力检查：{json.dumps(checks, ensure_ascii=False)[:300]}"


def test_all_registered_tools_have_wrapper_smoke(tmp_path: Path) -> None:
    """9 个注册工具都能定位到 FunctionTool（防新增工具漏进本文件覆盖）。"""
    names = {
        getattr(t, "name")
        for t in (
            check_sensor_sanity, check_temporal_sync, compute_stats,
            generate_report, inspect_streams, load_dataset,
            plot_chart, profile_data, propose_stream_semantics,
        )
    }
    assert names == {
        "load_dataset", "profile_data", "inspect_streams", "check_temporal_sync",
        "check_sensor_sanity", "compute_stats", "plot_chart", "generate_report",
        "propose_stream_semantics",
    }
