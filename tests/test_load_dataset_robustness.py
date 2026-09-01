"""load_dataset 嗅探器健壮性（畸形输入不崩）的单元测试。

覆盖：空目录、无扩展名文件、畸形文件名、以及单文件探测失败被记录为 probe_error
并继续其余文件（不中断加载）。不依赖真实网络。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl


def _make_malformed_dir(tmp_path: Path, name: str = "weird") -> Path:
    """构造含畸形文件名 / 无扩展名 / 正常文件的目录。"""
    root = tmp_path / name
    root.mkdir()
    # 畸形文件名（含特殊字符、无扩展名）。
    (root / "pipeline_Device").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    (root / "no_ext").write_text("x,y\n1,2\n", encoding="utf-8")
    # 无扩展名但内容是 CSV 的畸形文件。
    (root / "data 2 [v1]").write_text("p,q\n1,2\n", encoding="utf-8")
    # 正常 CSV（确保加载不因畸形文件而完全失败）。
    pd.DataFrame({"episode": [0, 1], "qpos1": [0.1, 0.2]}).to_csv(root / "state.csv", index=False)
    return root


def test_empty_directory_does_not_crash(tmp_path: Path) -> None:
    """空目录加载不应崩溃，返回 success。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(empty))
    assert r["success"] is True
    assert "probe_errors" in r  # 失败清单字段存在（空目录则无失败）
    assert r.get("probe_errors", []) == []


def test_no_extension_file_does_not_crash(tmp_path: Path) -> None:
    """无扩展名文件不应让加载崩溃（被归入 others/忽略，或探测失败被记录）。"""
    root = tmp_path / "noext"
    root.mkdir()
    (root / "noext_file").write_text("a,b\n1,2\n", encoding="utf-8")
    pd.DataFrame({"episode": [0], "qpos1": [0.1]}).to_csv(root / "state.csv", index=False)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True


def test_malformed_filenames_do_not_crash(tmp_path: Path) -> None:
    """畸形文件名（特殊字符/空格/中括号/无扩展名）不崩，正常文件仍被加载。"""
    root = _make_malformed_dir(tmp_path)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is True
    # 正常 state.csv 应被识别为主表或至少进入 table_info。
    table_names = [Path(t["file"]).name for t in r.get("table_info", [])]
    assert "state.csv" in table_names


def test_single_file_probe_error_does_not_stop_load(tmp_path: Path, monkeypatch) -> None:
    """单文件探测抛异常 → 记录 probe_error，其余文件仍被处理（不中断）。"""
    import app.tools.load_dataset as ld
    import app.tools._sniffing as sn

    root = tmp_path / "mix"
    root.mkdir()
    pd.DataFrame({"episode": [0, 1], "qpos1": [0.1, 0.2]}).to_csv(root / "good.csv", index=False)
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(root / "bad.csv", index=False)

    real_classify = sn.classify_table_stream

    def _boom_classify(name, *a, **k):
        if name == "bad.csv":
            raise IndexError("list index out of range")
        return real_classify(name, *a, **k)

    monkeypatch.setattr(sn, "classify_table_stream", _boom_classify)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    # 加载不中断，success=True。
    assert r["success"] is True
    # bad.csv 被记录到 probe_errors，且 reason 带文件名与阶段。
    assert r["probe_errors"], "应有探测失败记录"
    err = r["probe_errors"][0]
    assert err["file"].endswith("bad.csv")
    assert err["phase"] == "fingerprint"
    # good.csv 仍被正常登记（未被单文件失败拖垮）。
    table_names = [Path(t["file"]).name for t in r.get("table_info", [])]
    assert "good.csv" in table_names


def test_directory_probe_crash_becomes_structured_error(tmp_path: Path, monkeypatch) -> None:
    """目录级探测整体异常 → 结构化 error（directory_probe_failed），不裸抛。"""
    import importlib
    import sys

    # app.tools.load_dataset 模块属性会被 __init__.py 导出的 load_dataset FunctionTool
    # 遮蔽，必须经 sys.modules 取真实模块再 patch。
    ld = sys.modules["app.tools.load_dataset"]
    root = tmp_path / "boom"
    root.mkdir()
    pd.DataFrame({"a": [1]}).to_csv(root / "a.csv", index=False)

    def _boom_dir(*a, **k):
        raise RuntimeError("list index out of range")

    monkeypatch.setattr(ld, "_load_directory_impl", _boom_dir)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))
    assert r["success"] is False
    assert r["error"] == "directory_probe_failed"
    assert "list index out of range" in r["reason"]  # 裸异常被包装进 reason，不原样冒出
    assert str(root) in r["reason"]  # reason 带目录
