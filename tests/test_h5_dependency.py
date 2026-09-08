"""HDF5 加载的依赖区分与真实解析测试。

背景（用户实测）：h5 文件加载报 "Missing optional dependency 'pytables'"，
但 user_message 兜底措辞说"可能不是有效的 h5 数据，或文件已损坏"——误导。
且 requirements.txt 只装了 h5py（另一个 HDF5 库），没装 pandas 读 HDF5
实际需要的 tables（pytables）。

两类修复的回归：
  - 缺依赖 → error="missing_dependency"，user_message 给出修复指令、
    **不含"损坏"措辞**（诚实降级：文件根本没被读过）；
  - 装齐依赖后 → 合成 h5 正常加载；坏 h5 报 parse_failed（此时"可能损坏"
    措辞才是恰当的）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools.load_dataset import (
    MissingDependencyError,
    _load_hdf5,
    load_dataset_impl,
)


def test_missing_dependency_error_message_has_no_corrupt_hint() -> None:
    """缺依赖异常的 user_hint 不含"损坏"措辞，含可执行修复指令。"""
    exc = MissingDependencyError("pytables", "tables", "HDF5 (.h5)")
    hint = exc.user_hint()
    assert "pip install tables" in hint
    # "并非文件损坏"里的"损坏"字样是澄清用语，不算误导；误导性表述是
    # "可能……已损坏"这类可能性兜底——确保不存在。
    assert "并非文件损坏" in hint
    assert "可能" not in hint


def test_missing_dependency_structured_error(tmp_path: Path, monkeypatch) -> None:
    """mock 缺依赖场景：load_dataset 返回 missing_dependency 结构化错误。

    回归用户实测：此前该场景落 parse_failed 兜底，user_message 说
    "可能不是有效的 h5 数据，或文件已损坏"——误导。
    """
    import sys

    # app/tools/__init__.py 导出的 load_dataset FunctionTool 遮蔽了模块名，
    # 必须经 sys.modules 取真实模块再 patch（与 test_load_dataset_robustness 同款）。
    ld = sys.modules["app.tools.load_dataset"]

    def _boom(*a, **k):
        raise ImportError(
            "Missing optional dependency 'pytables'. Use pip or conda to install pytables."
        )

    monkeypatch.setattr(ld.pd, "HDFStore", _boom)
    p = tmp_path / "dataset.h5"
    p.write_bytes(b"fake")
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(p))
    assert r["success"] is False
    assert r["error"] == "missing_dependency"
    assert "pip install tables" in r["user_message"]
    # 禁的是误导性兜底（"可能……已损坏"），澄清句"并非文件损坏"是要求保留的。
    assert "可能" not in r["user_message"]
    assert "并非文件损坏" in r["user_message"]


def test_real_h5_loads_after_dependency_fixed(tmp_path: Path) -> None:
    """装齐依赖后：合成 h5（pandas HDFStore 格式）正常加载。"""
    p = tmp_path / "ok.h5"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    df.to_hdf(p, key="data", mode="w")
    out = _load_hdf5(str(p))
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 3


def test_corrupt_h5_reports_parse_failed(tmp_path: Path) -> None:
    """真坏的 h5（装齐依赖后仍读不了）→ parse_failed，此时"可能损坏"措辞恰当。"""
    p = tmp_path / "bad.h5"
    p.write_bytes(b"not a hdf5 file at all")
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(p))
    assert r["success"] is False
    assert r["error"] == "parse_failed"
    assert "损坏" in r["user_message"]


def test_requirements_includes_tables() -> None:
    """requirements.txt 必须含 tables（h5 是宣称支持格式，缺它必踩）。"""
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    text = req.read_text(encoding="utf-8")
    assert "tables" in text, "requirements.txt 缠 tables——.h5 宣称支持但读不了"
