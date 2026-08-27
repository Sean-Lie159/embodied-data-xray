"""数据格式骨架回归测试（参数化）。

对 tests/fixtures/skeletons/*.yaml 每份清单自动生成一个测试：加载该骨架目录，
断言加载不崩、返回结构化结果、文件登记数量与清单一致、空流/系统文件标注符合预期、
主表选择符合预期。新增格式 = 往 skeletons/ 丢一份清单，本测试自动多一条。

另含骨架确定性测试（同清单两次生成逐字节一致）与匿名化元检查。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.fixtures.make_skeleton import _SKELETONS_DIR, build_skeleton
from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl

_SKELETONS = sorted(p.stem for p in _SKELETONS_DIR.glob("*.yaml"))


def _manifest(name: str) -> dict:
    return yaml.safe_load((_SKELETONS_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("skeleton_name", _SKELETONS)
def test_skeleton_loads_and_registers(tmp_path: Path, skeleton_name: str) -> None:
    """每份骨架清单：加载不崩、返回结构化结果、登记数量/空流/系统文件/主表符合预期。"""
    manifest = _manifest(skeleton_name)
    root = build_skeleton(tmp_path, skeleton_name)
    ctx = RunContext(output_dir=str(tmp_path))
    r = load_dataset_impl(ctx, str(root))

    # 加载不崩、返回结构化结果。
    assert r.get("success") is True, f"{skeleton_name} 加载失败: {r.get('reason')}"
    assert "file_survey" in r

    # 文件登记数量与清单一致（减去系统文件）。
    exp_total = manifest["expectations"]["total_files"]
    assert r["file_survey"]["total_files"] == exp_total, (
        f"{skeleton_name} 登记文件数 {r['file_survey']['total_files']} != 期望 {exp_total}"
    )

    # 空流标注符合预期。
    empty_names = {Path(s["path"]).name for s in ctx.meta["streams"] if s.get("status") == "empty"}
    exp_empty = set(manifest["expectations"].get("empty_streams", []))
    assert exp_empty.issubset(empty_names), (
        f"{skeleton_name} 空流缺失: 期望 {exp_empty - empty_names} 未标注空流"
    )

    # 系统文件被跳过（不在 streams 中）。
    for sys_file in ("desktop.ini", "Thumbs.db", ".DS_Store"):
        assert not any(s["path"].endswith(sys_file) for s in ctx.meta["streams"]), (
            f"{skeleton_name} 系统文件 {sys_file} 不应出现在流登记表"
        )

    # 主表选择符合预期。
    exp_main = manifest["expectations"].get("main_table")
    actual_main = (r.get("main_table") or {}).get("name")
    if exp_main is None:
        assert actual_main is None, f"{skeleton_name} 主表应为 null，实际 {actual_main}"
    else:
        assert actual_main == exp_main, f"{skeleton_name} 主表 {actual_main} != 期望 {exp_main}"

    # 数据集格式能力符合预期（若清单声明了）。
    exp_format = manifest["expectations"].get("dataset_format")
    if exp_format is not None:
        actual_format = ctx.meta.get("capabilities", {}).get("dataset_format")
        assert actual_format == exp_format, (
            f"{skeleton_name} dataset_format {actual_format} != 期望 {exp_format}"
        )


def test_skeleton_deterministic(tmp_path: Path) -> None:
    """同一骨架两次生成逐字节一致（确定性）。"""
    for sk in _SKELETONS:
        a = build_skeleton(tmp_path / "a", sk)
        b = build_skeleton(tmp_path / "b", sk)
        files_a = {p.relative_to(a).as_posix(): p.read_bytes() for p in a.rglob("*") if p.is_file()}
        files_b = {p.relative_to(b).as_posix(): p.read_bytes() for p in b.rglob("*") if p.is_file()}
        assert files_a == files_b, f"{sk} 两次生成不一致"


def test_manifests_anonymized_and_have_tricky_features() -> None:
    """清单匿名化（无真实项目名/设备序列号/日期）且 worldcode 含刁难特征。"""
    for sk in _SKELETONS:
        manifest = _manifest(sk)
        joined = yaml.safe_dump(manifest).lower()
        # 匿名化检查：不得含真实设备/项目名、序列号、日期模式。
        assert "01350f7c" not in joined
        assert "vlta_reorg" not in joined
        assert "2026-" not in joined
    # worldcode 骨架必须包含刁难特征（结构指纹）。
    wc = _manifest("worldcode_nuscenes")
    names = [f["name"] for f in wc["files"]]
    assert "ego_pose.json" in names             # 2 字节空 JSON
    assert "desktop.ini" in names               # 系统文件
    assert "camera-a.index.parquet" in names    # .index.parquet 双扩展名
    assert "camera-a_480.mp4" in names          # 多分辨率版本组
    # 含方位词但无角色命中的文件名（曾打崩 infer_role 的根因）。
    assert any("right" in n and "camera" in n and "mipi" in n for n in names)
