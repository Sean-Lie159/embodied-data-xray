"""工具注册一致性回归：CLI 与 UI 必须注册同一套工具。

**为什么需要这条测试**（真实缺陷，2026-09-16 发现）：
CLI（``main.py`` 的 ``_build_main_agent``）与 UI（``chat_service._ALL_TOOLS``）
各自硬编码工具列表，且 CLI 长期漏注册 ``unpack_mcap``——表现为"UI 能解包
MCAP 容器，CLI 不能"，CLI 横幅还把工具清单硬编码成字符串，与真实注册不同步，
导致这个缺陷很久无人察觉。

本测试把"两处工具集必须一致"固化为回归：新增工具时若只改了一处，立即失败。
这是对 ``docs/标注与质检能力设计.md`` §8 阶段一交付项 4 的自动化守护。
"""

from __future__ import annotations

from app.services.chat_service import _ALL_TOOLS


def test_cli_and_ui_register_same_tools() -> None:
    """CLI 与 UI 注册的工具名集合必须完全一致。"""
    from main import _build_main_agent  # noqa: PLC0415 - 延迟导入避免顶层副作用

    # _build_main_agent 会构造真实 Model（需 .env 配置）；这里只取工具名，
    # 因此用 try 兜底：配置缺失时退化为直接读其源码级工具列表。
    try:
        _agent, cli_tool_names = _build_main_agent()
    except Exception:  # noqa: BLE001 - 无 .env 环境（CI）时走静态比对
        cli_tool_names = _cli_tool_names_from_source()

    ui_tool_names = [t.name for t in _ALL_TOOLS]

    assert set(cli_tool_names) == set(ui_tool_names), (
        "CLI 与 UI 的工具集不一致——新增/删减工具时必须同时改两处。\n"
        f"仅在 UI：{sorted(set(ui_tool_names) - set(cli_tool_names))}\n"
        f"仅在 CLI：{sorted(set(cli_tool_names) - set(ui_tool_names))}"
    )


def test_cli_tool_count_matches_ui() -> None:
    """数量一致（更直白的断言，失败信息更易读）。"""
    try:
        _agent, cli_tool_names = _build_main_agent()
    except Exception:  # noqa: BLE001
        cli_tool_names = _cli_tool_names_from_source()

    assert len(cli_tool_names) == len(_ALL_TOOLS)


def test_unpack_mcap_registered_in_both() -> None:
    """unpack_mcap 必须在两处都注册（本缺陷的直接回归用例）。"""
    try:
        _agent, cli_tool_names = _build_main_agent()
    except Exception:  # noqa: BLE001
        cli_tool_names = _cli_tool_names_from_source()

    assert "unpack_mcap" in cli_tool_names
    assert "unpack_mcap" in [t.name for t in _ALL_TOOLS]


def test_static_cli_tool_names_in_sync_with_ui() -> None:
    """main.py 的静态兜底清单也必须与 UI 一致。

    为什么单独测：`_CLI_TOOL_NAMES` 是 `chat_loop` 在拿不到真实注册结果时的
    展示兜底。它若与真实注册漂移，用户看到的工具清单就是错的——这正是
    `unpack_mcap` 缺陷当初的隐藏形态（横幅硬编码、与真实注册不同步）。
    """
    import main as main_module

    assert set(main_module._CLI_TOOL_NAMES) == {t.name for t in _ALL_TOOLS}


def _cli_tool_names_from_source() -> list[str]:
    """无 .env 配置时，从 main.py 源码静态提取 CLI 的工具名清单。

    为什么不直接 import 常量：CLI 的工具列表是函数内的局部变量（历史写法），
    没有模块级常量可读。改为解析源码中的 import 与列表字面量，可覆盖"工具集
    被改动"这一唯一关注点。
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "main.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    # 收集 from app.tools import (...) 的本地名 → 原名的映射。
    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "app.tools":
            for a in node.names:
                alias_map[a.asname or a.name] = a.name

    # 定位 _build_main_agent 内的 tools = [...] 列表字面量。
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_build_main_agent":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Assign):
                continue
            targets = [t.id for t in sub.targets if isinstance(t, ast.Name)]
            if "tools" not in targets or not isinstance(sub.value, ast.List):
                continue
            names: list[str] = []
            for elt in sub.value.elts:
                if isinstance(elt, ast.Name):
                    names.append(alias_map.get(elt.id, elt.id))
            if names:
                return names
    return []
