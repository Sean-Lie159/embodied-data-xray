"""工具返回体积护栏（_output_guard / guard_tools）的单元测试。

覆盖：三档渐进降级、**计数守恒**、truncated + truncation_note 必带、
未超预算时不动原值、护栏异常不吞掉工具结果。
"""

from __future__ import annotations

from app.agent.agent import _TOOL_DROPPABLE, guard_tools
from app.tools._output_guard import (
    _MAX_LIST_ITEMS,
    _measure_data as data_tokens,
    enforce_output_limit,
    measure_tokens,
)


def _big(n: int = 2000) -> dict:
    """构造超预算的返回（含次要字段、长列表、结论字段）。"""
    return {
        "success": True,
        "dataset": "demo",
        "result": "pass",
        "subdirs": [f"dir_{i}" for i in range(n)],
        "ext_dist": {f".ext{i}": i for i in range(200)},
        "streams": [{"path": f"s{i}", "kind": "imu"} for i in range(300)],
        "user_message": "这是一段说明" * 50,
    }


# --- 1. 未超预算：不动原值 --------------------------------------------------


def test_no_change_when_within_budget() -> None:
    """未超预算时原样返回，不加任何噪声字段（含不加 truncated）。"""
    r = {"success": True, "n_rows": 10}
    out = enforce_output_limit(r, budget_tokens=100_000)
    assert out == r
    assert "truncated" not in out
    assert "truncation_note" not in out


def test_non_dict_passthrough() -> None:
    """非 dict 原样返回（护栏只处理结构化返回）。"""
    assert enforce_output_limit("plain string", budget_tokens=1) == "plain string"
    assert enforce_output_limit(None, budget_tokens=1) is None


def test_invalid_budget_passthrough() -> None:
    """预算 <=0（配置异常）时不做任何降级，避免把结果压成空。"""
    r = _big(50)
    assert enforce_output_limit(r, budget_tokens=0) == r


# --- 2. 三档渐进降级 --------------------------------------------------------


def test_tier1_drops_declared_fields_first() -> None:
    """档 1：按 droppable 顺序丢弃次要字段，保留结论字段。"""
    r = _big(20)
    out = enforce_output_limit(r, budget_tokens=3_000, droppable=("subdirs", "ext_dist"))
    # 结论字段必须还在
    assert out["success"] is True
    assert out["dataset"] == "demo"
    assert out["result"] == "pass"
    # 被声明可丢的字段已丢
    assert "subdirs" not in out or "ext_dist" not in out
    assert out.get("truncated") is True
    assert out.get("truncation_note")


def test_tier1_stops_as_soon_as_within_budget() -> None:
    """达标即停：预算宽松时不得把 droppable 全丢光。"""
    r = _big(20)
    # 预算足够大到只需丢一个字段
    tokens_full = measure_tokens(r)
    out = enforce_output_limit(
        r, budget_tokens=int(tokens_full * 0.9), droppable=("subdirs", "ext_dist", "streams")
    )
    # 至少还有一个 droppable 字段保留（未全丢）
    assert any(k in out for k in ("subdirs", "ext_dist", "streams"))


def test_tier2_truncates_lists_but_keeps_counts() -> None:
    """档 2：长列表被截断，但**总数计数完整保留**（绝不静默抽样）。"""
    long_list = [{"i": i} for i in range(500)]
    r = {"success": True, "detail": long_list}
    out = enforce_output_limit(r, budget_tokens=200)
    inner = out.get("detail")
    assert isinstance(inner, dict), "长列表应被压为带计数的结构"
    assert inner["total"] == 500, "总数必须完整保留"
    assert inner["shown"] == min(_MAX_LIST_ITEMS, 500)
    assert inner["truncated"] is True


def test_tier3_keeps_conclusion_keys() -> None:
    """档 3：极端压缩后仍保留结论字段（success/result/关键数字）。"""
    r = _big(3000)
    out = enforce_output_limit(r, budget_tokens=50)
    assert out["success"] is True
    # 要么保留结论，要么给出结构化"过大"错误（二者都不得是裸明细）
    if out.get("error") == "output_too_large":
        assert out["truncated"] is True
        assert "缩小" in out["user_message"]
    else:
        assert "result" in out
        assert out["truncated"] is True


# --- 3. 通用纪律 ------------------------------------------------------------


def test_truncation_always_marked_and_noted() -> None:
    """任何降级都必带 truncated=True 与 truncation_note（防静默失真）。"""
    r = _big(500)
    out = enforce_output_limit(r, budget_tokens=100)
    assert out.get("truncated") is True
    assert isinstance(out.get("truncation_note"), str)
    assert out["truncation_note"]


def test_original_dict_not_mutated() -> None:
    """降级不得污染调用方持有的原对象（深拷贝语义）。"""
    r = _big(100)
    snapshot = dict(r)
    enforce_output_limit(r, budget_tokens=100)
    assert set(r.keys()) == set(snapshot.keys())


def test_measure_tokens_failsafe_on_unserializable() -> None:
    """不可序列化对象视为超限（fail-safe），强制走压缩分支。"""
    class Weird:
        def __repr__(self) -> str:
            raise RuntimeError("boom")

    n = measure_tokens({"k": Weird()})
    assert n > 1_000_000  # 极大值 → 必然触发压缩


def test_extremely_tight_budget_never_crashes() -> None:
    """极小预算不崩，且产出可用信息。"""
    out = enforce_output_limit(_big(100), budget_tokens=1)
    assert isinstance(out, dict)
    assert out.get("truncated") is True


# --- 4. guard_tools 包装 ----------------------------------------------------


def test_guard_tools_preserves_schema_and_count() -> None:
    """包装后工具数量与 schema 不变（只替换 on_invoke_tool）。"""
    from agents.tool import FunctionTool

    async def _noop(ctx, input_json):
        return {"success": True, "big": ["x" * 100] * 500}

    t = FunctionTool(
        name="fake_tool",
        description="测试用工具",
        params_json_schema={"type": "object", "properties": {}, "required": [],
                            "additionalProperties": False},
        on_invoke_tool=_noop,
        strict_json_schema=False,
    )
    guarded = guard_tools([t], budget_tokens=100)
    assert len(guarded) == 1
    g = guarded[0]
    assert isinstance(g, FunctionTool)
    assert g.name == "fake_tool"
    assert g.description == "测试用工具"
    assert g.params_json_schema == t.params_json_schema


def test_guard_tools_applies_limit_at_invoke() -> None:
    """包装后的工具在调用时真的被限流（端到端验证 on_invoke_tool 替换生效）。

    本项目未装 pytest-asyncio，故用 asyncio.run 同步驱动协程。
    """
    import asyncio

    from agents.tool import FunctionTool

    async def _big_result(ctx, input_json):
        return {"success": True, "detail": ["y" * 100] * 500}

    t = FunctionTool(
        name="big_out",
        description="返回很大的工具",
        params_json_schema={"type": "object", "properties": {}, "required": [],
                            "additionalProperties": False},
        on_invoke_tool=_big_result,
        strict_json_schema=False,
    )
    guarded = guard_tools([t], budget_tokens=100)[0]

    async def _drive() -> dict:
        return await guarded.on_invoke_tool(None, "{}")

    out = asyncio.run(_drive())
    assert out["truncated"] is True
    # 达标判定按**数据体积**（不含 truncation_note 元信息）——见 _measure_data
    # 的说明：把 note 计入会导致"加了说明反而多降一档"的降级螺旋。
    assert data_tokens(out) <= 100 or out.get("error") == "output_too_large"
    assert out.get("truncation_note"), "降级必须附说明"


def test_guard_tools_passes_through_non_function_tool() -> None:
    """非 FunctionTool 原样透传（不因包装而丢失）。"""
    sentinel = object()
    assert guard_tools([sentinel], budget_tokens=100)[0] is sentinel


def test_droppable_table_covers_all_tools() -> None:
    """可丢弃字段表覆盖全部 8 个注册工具（防漏配）。"""
    expected = {
        "load_dataset", "profile_data", "inspect_streams", "check_temporal_sync",
        "check_sensor_sanity", "compute_stats", "plot_chart", "generate_report",
    }
    assert expected.issubset(set(_TOOL_DROPPABLE))


def test_droppable_covers_all_registered_tools() -> None:
    """可丢弃字段表必须覆盖**当前实际注册的每一个工具**（防新增工具漏配）。

    为什么从"全员注册表"取名单而不是硬编码：此前测试只校验 8 个历史工具，
    工具数已增至 19，中间新增的（如 `compare_table_columns`）**全部漏配而测试
    不报**——漏配的工具会跳过档 1，超预算时按体积整体硬截断，把结论字段与长
    明细一起砍掉（2026-09-21 发现并修复）。

    注意：不在 `_ALL_TOOLS` 中的工具（如 CLI 独有项）不参与本断言——
    它们由各自的注册路径保证。
    """
    from app.services.chat_service import _ALL_TOOLS

    registered = {getattr(t, "name", None) for t in _ALL_TOOLS}
    registered.discard(None)
    missing = sorted(registered - set(_TOOL_DROPPABLE))
    assert not missing, (
        f"以下已注册工具在 _TOOL_DROPPABLE 中漏配（会跳过档 1 降级）：{missing}"
    )


def test_droppable_compare_table_columns_keeps_conclusions() -> None:
    """跨表运算的降级必须保留对齐口径与计数，先丢明细。

    回归（2026-09-21）：`compare_table_columns` 此前未登记，超预算时走通用
    截断——`aligned_on`/`n_matched`/`unmatched` 这些**结论字段**可能先于
    `samples` 一起被砍。现要求降级后这些字段仍在。
    """
    droppable = _TOOL_DROPPABLE["compare_table_columns"]
    assert "samples" in droppable, "样例行属可再生明细，应优先丢弃"
    assert "result" in droppable or "samples" in droppable
    # 结论字段绝不能出现在可丢清单里。
    for conclusion_key in ("aligned_on", "n_a", "n_b", "n_matched", "unmatched",
                           "columns_used", "user_message", "success"):
        assert conclusion_key not in droppable, (
            f"{conclusion_key} 是结论字段，不得列为可丢"
        )

    # 端到端：超预算时先丢 samples，结论字段留存。
    r = {
        "success": True,
        "aligned_on": "frame_index",
        "n_a": 28270, "n_b": 28270, "n_matched": 28270,
        "unmatched": {"keys_only_in_a": 0, "keys_only_in_b": 0},
        "result": {"pos_x（A−B）": {"n": 28270, "mean": 0.001, "std": 0.02}},
        "samples": [{"frame_index": i, "pos_x_A": 0.1, "pos_x_B": 0.1}
                    for i in range(500)],
        "user_message": "已按 frame_index 对齐，对齐后 28270 行。",
    }
    out = enforce_output_limit(r, budget_tokens=300, droppable=droppable,
                               tool_name="compare_table_columns")
    assert out.get("truncated") is True
    assert out["aligned_on"] == "frame_index"
    assert out["n_matched"] == 28270
    assert "samples" not in out, "应先丢弃样例行"
