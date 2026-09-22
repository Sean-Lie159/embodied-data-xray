"""文件内容查看测试（**真实场景回归**，2026-09-22）。

场景：分析 ``origin-data-fullmodal-sample-dataset`` 时用户问"数据集里有没有
数采设备信息（相机/手环型号）"，agent 回答"够不到，建议重新加载单文件"。

根因：``probe_directory`` 把文件分六组，而 ``build_streams_registry`` 只登记
tables/videos/audios/images —— ``others`` 组（.md 等文档）**被完全丢弃**、
``cals`` 组（标定/元数据 JSON）**刻意不进流登记表**，于是这两类文件的**内容
没有任何读取路径**。

核心修复：**"不参与数据分析" ≠ "内容不可查看"**。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.agent.context import RunContext
from app.tools.load_dataset import load_dataset_impl
from app.tools.read_file_content import read_file_content_impl

# 复刻 origin-data 的关键文件构成。
README_TEXT = """# OriginData Sample Dataset

OriginData is a full-modal dataset released by OriginFlow. By synchronously
capturing sEMG, first-person-view video, IMU, and other signals, it provides
multi-modal data such as pose + fingertip contact force.

## Fields

- `{side}.force_values`: Fz of thumb, index, middle, ring, little, in mN.
- Force is an estimate inferred from EMG, not ground truth from a hardware
  force sensor.
"""

DOC_TEXT = """# LeRobot v3 Dataset Documentation

## 2. Camera Calibration

Read `calibration_id` from `episodes.jsonl`, obtain `P_left_from_joints`
from `meta/camera_calibration.json`.
"""

CALIB = {
    "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
    "distortion": {"k1": 0.01},
    "P_left_from_joints": [[1.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4],
}

INFO = {
    "fps": 30,
    "robot_type": "origin_flow",
    "features": {"observation.images.head_left": {"dtype": "video"}},
}


@pytest.fixture
def ds_root(tmp_path: Path) -> Path:
    """构造与 origin-data 同构的目录。"""
    root = tmp_path / "origin-data-fullmodal-sample-dataset"
    (root / "lerobot" / "meta").mkdir(parents=True)
    (root / "README.md").write_text(README_TEXT, encoding="utf-8")
    (root / "leRobotv3DatasetDoc.md").write_text(DOC_TEXT, encoding="utf-8")
    (root / "lerobot" / "meta" / "camera_calibration.json").write_text(
        json.dumps(CALIB), encoding="utf-8")
    (root / "lerobot" / "meta" / "info.json").write_text(
        json.dumps(INFO), encoding="utf-8")
    pd.DataFrame({
        "timestamp": range(20),
        "v": range(20),
    }).to_parquet(root / "file-007.parquet")
    return root


@pytest.fixture
def ctx(ds_root: Path) -> RunContext:
    """已加载该数据集的上下文。"""
    c = RunContext()
    res = load_dataset_impl(c, str(ds_root))
    assert res.get("success") is True, res
    return c


# ---------------------------------------------------------------------------
# 核心：文档类文件（others 组）现在可读
# ---------------------------------------------------------------------------


def test_readme_is_readable(ctx: RunContext) -> None:
    """**事故核心验收**：README.md 的内容可读。"""
    res = read_file_content_impl(ctx, "README.md")
    assert res["success"] is True, res
    assert res["kind"] == "document"
    assert "OriginData" in res["content"]
    assert "OriginFlow" in res["content"]
    assert res["n_lines_total"] > 0


def test_keyword_locates_emg_lines_with_line_numbers(ctx: RunContext) -> None:
    """**关键词定位**：能定位到 emg 相关行并带行号。

    这正是当初该问的问题——agent 若能读文档，就能回答"力值是从 EMG 推算的"。
    """
    res = read_file_content_impl(ctx, "README.md", keyword="emg")
    assert res["success"] is True
    assert "EMG" in res["content"]
    assert "sEMG" in res["content"]
    # 必须带行号（便于引用与续读）。
    assert "|" in res["content"]
    assert "关键词" in res["content_basis"]


def test_keyword_no_hit_is_honest(ctx: RunContext) -> None:
    """关键词无命中时如实说明（不编造、不返回空而装成功）。"""
    res = read_file_content_impl(ctx, "README.md", keyword="zzz-not-exist")
    assert res["success"] is True
    assert res["content"] == ""
    assert res["n_lines_returned"] == 0
    assert "未找到" in res["user_message"]


def test_second_document_readable(ctx: RunContext) -> None:
    """另一份文档同样可读（证明不是特例）。"""
    res = read_file_content_impl(ctx, "leRobotv3DatasetDoc.md")
    assert res["success"] is True
    assert "LeRobot v3" in res["content"]


def test_read_by_relative_path(ctx: RunContext) -> None:
    """支持相对路径（含子目录）。"""
    res = read_file_content_impl(ctx, "lerobot/meta/info.json")
    assert res["success"] is True
    assert "robot_type" in res["content"]


# ---------------------------------------------------------------------------
# 配置 / 标定（cals 组）现在可读
# ---------------------------------------------------------------------------


def test_calibration_json_content_is_readable(ctx: RunContext) -> None:
    """**标定 JSON 内容可读**（此前只剩指纹探测给的键名）。

    这解决了"标定文件不进表清单是对的，但内容也不可读"的副作用。

    注意：``_compact_config`` 对**嵌套结构只报规模**（防深层膨胀），
    故断言"键在、且嵌套被如实标注"，而非断言嵌套内的具体数值
    （那是刻意的压缩行为，不是缺陷）。
    """
    res = read_file_content_impl(ctx, "camera_calibration.json")
    assert res["success"] is True
    assert res["kind"] == "config"
    # 顶层键必须可见。
    assert "intrinsics" in res["keys"]
    assert "P_left_from_joints" in res["keys"]
    # 嵌套结构被压缩但**如实标注规模**（不静默丢弃）。
    assert "嵌套对象" in res["content"] or "嵌套列表" in res["content"]


def test_calibration_readable_by_relative_path(ctx: RunContext) -> None:
    """标定文件也可用相对路径读取（含子目录）。"""
    res = read_file_content_impl(ctx, "lerobot/meta/camera_calibration.json")
    assert res["success"] is True
    assert "intrinsics" in res["keys"]


def test_metadata_json_content_is_readable(ctx: RunContext) -> None:
    """meta/info.json 的 schema 信息可读。"""
    res = read_file_content_impl(ctx, "info.json")
    assert res["success"] is True
    assert "robot_type" in res["content"]
    assert "origin_flow" in res["content"]


def test_config_returns_key_summary_not_raw(ctx: RunContext) -> None:
    """配置模式返回**键值摘要**（复用 _compact_config），并声明未做理解。"""
    res = read_file_content_impl(ctx, "camera_calibration.json")
    assert res["n_keys"] >= 3
    assert "intrinsics" in res["keys"]
    # 必须声明"未做内容理解"（原文交付，含义由模型判断）。
    assert res.get("content_not_interpreted") is True


# ---------------------------------------------------------------------------
# 表格类：引导而非重复实现（职责分离）
# ---------------------------------------------------------------------------


def test_table_is_redirected_not_read(ctx: RunContext) -> None:
    """**表格请求返回引导**，不在本工具重复实现读表。"""
    res = read_file_content_impl(ctx, "file-007.parquet")
    assert res["success"] is False
    assert res["error"] == "is_data_table"
    assert "profile_data" in res["suggested_tools"]
    assert "数据表" in res["user_message"]


# ---------------------------------------------------------------------------
# 容量与截断
# ---------------------------------------------------------------------------


def test_max_chars_truncates_and_reports(ctx: RunContext) -> None:
    """超长内容被截断且**如实标注**（不静默）。"""
    res = read_file_content_impl(ctx, "README.md", max_chars=80)
    assert res["success"] is True
    assert len(res["content"]) <= 80
    assert res["truncated"] is True
    assert res["n_lines_total"] > res["n_lines_returned"]
    assert "截断" in res["content_basis"]
    assert "start_line" in res["note"]


def test_start_line_resumes_reading(ctx: RunContext) -> None:
    """``start_line`` 可续读后续行。"""
    full = read_file_content_impl(ctx, "README.md")
    tail = read_file_content_impl(ctx, "README.md", start_line=6)
    assert tail["success"] is True
    assert "第 6-" in tail["content_basis"]
    # 续读内容应是全文的后半部分（行数更少）。
    assert tail["n_lines_returned"] < full["n_lines_returned"]


# ---------------------------------------------------------------------------
# 路径解析与安全
# ---------------------------------------------------------------------------


def test_file_not_found_lists_available(ctx: RunContext) -> None:
    """找不到时给出**可用文件清单**（供自我纠正）。"""
    res = read_file_content_impl(ctx, "nope.md")
    assert res["success"] is False
    assert res["error"] == "file_not_found"
    assert "README.md" in res["available_documents"]
    assert res["n_available_documents"] >= 3


def test_outside_dataset_path_rejected(ctx: RunContext) -> None:
    """**越权路径被拒绝**（只能读数据集目录内的文件）。

    用**项目根**下的文件做"外部路径"——它确实存在且不在数据集目录内。
    （早期写法用 pytest 的 tmp_path，但那与 ds_root 同源、并非"外部"，
    导致测试未真正验证越权路径。）
    """
    outside = Path(__file__).resolve().parent.parent / "README.md"
    assert outside.is_file(), "测试前置：项目 README 应存在"
    res = read_file_content_impl(ctx, str(outside))
    assert res["success"] is False
    assert res["error"] == "path_outside_dataset"
    assert "拒绝" in res["user_message"]


def test_project_readme_not_hijacked_by_relative_name(ctx: RunContext) -> None:
    """**关键回归**：按文件名 "README.md" 必须命中**数据集内**的 README。

    实测缺陷（2026-09-22）：若先做 ``Path("README.md").is_file()``，该相对路径
    会相对 **CWD（项目根）** 解析——而项目根恰有 README.md！于是"看数据集的
    README"会静默命中**项目自己的 README**，再因不在数据集内被拒
    （表现为"文件明明存在却报越权"）。修正为**先以数据集根为基准解析**。
    """
    res = read_file_content_impl(ctx, "README.md")
    assert res["success"] is True, res
    # 必须是**数据集内**那份（含 OriginData），而不是项目自己的 README。
    assert "OriginData" in res["content"]
    assert "Embodied-data-Xray" not in res["content"], (
        "命中了项目自身的 README —— 相对路径解析顺序有误"
    )


def test_empty_path_rejected(ctx: RunContext) -> None:
    """空路径返回结构化错误。"""
    res = read_file_content_impl(ctx, "   ")
    assert res["success"] is False
    assert res["error"] == "empty_path"


def test_no_dataset_returns_structured_error() -> None:
    """未加载数据集时返回结构化错误。"""
    res = read_file_content_impl(RunContext(), "README.md")
    assert res["success"] is False
    assert res["error"] == "no_data_loaded"


def test_case_insensitive_filename(ctx: RunContext) -> None:
    """文件名匹配大小写不敏感。"""
    res = read_file_content_impl(ctx, "readme.MD")
    assert res["success"] is True
    assert "OriginData" in res["content"]


# ---------------------------------------------------------------------------
# 只读保证
# ---------------------------------------------------------------------------


def test_reading_does_not_modify_file(ctx: RunContext, ds_root: Path) -> None:
    """**只读保证**：读取前后文件字节完全一致。"""
    p = ds_root / "README.md"
    before = p.read_bytes()
    read_file_content_impl(ctx, "README.md")
    assert p.read_bytes() == before


def test_reading_does_not_pollute_context(ctx: RunContext) -> None:
    """**不污染上下文**：读取不写入 df / findings / 能力标签。"""
    df_before = ctx.df
    findings_before = len(ctx.findings)
    caps_before = json.dumps(ctx.meta.get("capabilities", {}), sort_keys=True)

    read_file_content_impl(ctx, "README.md")

    assert ctx.df is df_before, "读取文档不得替换主表"
    assert len(ctx.findings) == findings_before, "读取文档不得写入 findings"
    assert json.dumps(
        ctx.meta.get("capabilities", {}), sort_keys=True) == caps_before
