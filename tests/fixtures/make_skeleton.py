"""数据格式骨架生成器（测试辅助代码，非 agent 工具）。

读取 tests/fixtures/skeletons/*.yaml 清单，在目标目录生成对应的合成数据目录，
用于"数据格式骨架"回归测试（test_format_skeletons.py）。

生成确定性：固定内容、固定种子，同一清单两次生成逐字节一致（可用哈希验证）。
生成的是"结构骨架"（能通过内容指纹/不崩即可），不是真实数据——验证的是"不崩/
登记数量正确"，不是"认得准"。

用法（测试内调用）：
    from tests.fixtures.make_skeleton import build_skeleton
    root = build_skeleton(tmp_path / "sk", "tobii_ego")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# skeletons 清单目录。
_SKELETONS_DIR = Path(__file__).parent / "skeletons"


def build_skeleton(dest: Path, skeleton_name: str) -> Path:
    """按清单在 dest 下生成骨架目录，返回目录路径。

    Args:
        dest: 目标父目录（生成的骨架子目录在此之下）。
        skeleton_name: 清单文件名（不含 .yaml，如 "tobii_ego"）。

    Returns:
        生成的骨架目录路径（dest / <skeleton_name>）。
    """
    manifest_path = _SKELETONS_DIR / f"{skeleton_name}.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到骨架清单：{manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = dest / skeleton_name
    root.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        _write_file(root, entry)
    return root


def _write_file(root: Path, entry: dict) -> None:
    """按条目的 type 生成对应文件内容（确定性）。"""
    name = entry["name"]
    typ = entry["type"]
    path = root / name
    if typ == "csv":
        path.write_text(_csv_content(name), encoding="utf-8")
    elif typ == "json":
        path.write_text(_json_content(entry.get("generate", "")), encoding="utf-8")
    elif typ == "parquet":
        _parquet_content(name).to_parquet(path)
    elif typ == "system":
        path.write_text("[ViewState] skeleton placeholder\n", encoding="utf-8")
    else:  # video / audio / image：0 字节占位（探测失败走 probe_error 也算正确行为）。
        path.write_bytes(b"")


def _csv_content(name: str) -> str:
    """生成确定性 CSV（时间戳单调递增，能过内容指纹）。"""
    base = name.split(".")[0]
    ts = [0, 1_000_000, 2_000_000]  # 单调递增纳秒
    # 位姿列（含四元数模长≈1）与 IMU/动作列。
    rows = []
    for i, t in enumerate(ts):
        # 归一化四元数（模长≈1）。
        q = [1.0, 0.0, 0.0, 0.0]
        rows.append(f"{t},{i}.0,{i}.0,{i}.0,{q[0]},{q[1]},{q[2]},{q[3]}")
    header = "timestamp_ns,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w"
    # 特殊表：controller_poses 仅表头（空流）。
    if name == "controller_poses.csv":
        return "pose_x,pose_y,pose_z\n"
    # 特殊表：metainfo（帧序号/曝光时间戳）。
    if name.endswith("_metainfo.csv"):
        header = "frame_index,pts_us,capture_utc_ns"
        rows = ["0,0,1", "1,1,2", "2,2,3"]
    return header + "\n" + "\n".join(rows) + "\n"


def _json_content(generate: str) -> str:
    """按生成规则生成最小合法 JSON（确定性）。"""
    gen = (generate or "").lower()
    # 标定 JSON。
    if "intrinsics" in gen or "extrinsic" in gen or "calib" in gen:
        return json.dumps({"intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}}, ensure_ascii=False)
    # 2 字节空 JSON。
    if gen.strip().startswith("{}"):
        return "{}"
    # 最小合法 dict（nuScenes schema 表族 / 相机同名 json）。
    return json.dumps({"data": []}, ensure_ascii=False)


def _parquet_content(name: str) -> pd.DataFrame:
    """生成确定性最小 parquet（含 timestamp_ns 列）。"""
    return pd.DataFrame({
        "timestamp_ns": [0, 1_000_000],
        "idx": [0, 1],
    })
