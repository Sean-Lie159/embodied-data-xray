"""真实数据集暴露的嗅探缺陷修复验证（第一部分）。

用真实采集目录的 18 个文件名构造合成目录，验证：
1. 目录普查返回完整文件清单（按类型分组，全部 18 个路径，不抽样、不省略）；
2. 取消随机抽样后，流登记表覆盖全部表格文件（读头部判类型），视频/音频/图片
   只登记路径与格式；
3. 确定性：同一目录两次加载，file_survey 与 streams 完全一致。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.tools._sniffing import probe_directory
from app.tools.load_dataset import load_dataset_impl

# 真实采集目录的文件名清单（内容可合成，文件名必须一致）。
_REAL_FILES = [
    "accel.csv", "audio.m4a", "audio_metainfo.csv", "camera_params_ctrl.json",
    "camera_params_rgb.json", "camera_params_tracking.json", "controller_poses.csv",
    "ctrl.mp4", "ctrl_metainfo.csv", "gyro.csv", "hand_tracking.csv", "head_pose.csv",
    "imu_calibration.json", "rgb.mp4", "rgb_metainfo.csv", "tracking.mp4",
    "tracking_metainfo.csv", "信息.jpg",
]

# 各 csv 的表头 + 数据行（保证列名嗅探可读；内容无关紧要）。
# 除 controller_poses.csv 仅为表头（验证空流检测）外，其余表格均含 ≥3 行数据，
# 避免触发空流检测，从而能验证 accel/gyro 被识别为 IMU。
_CSV_HEADERS = {
    "accel.csv": "timestamp_ns,accel_x,accel_y,accel_z\n0,1,2,3\n4,5,6,7\n8,9,10,11\n",
    "audio_metainfo.csv": "packet_index,pts_us,capture_utc_ns\n0,0,1\n1,1,2\n2,2,3\n",
    "controller_poses.csv": "pose_x,pose_y,pose_z\n",  # 仅表头 → 空流
    "ctrl_metainfo.csv": "frame_index,pts_us\n0,0\n1,1\n2,2\n",
    "gyro.csv": "timestamp_ns,gyro_x,gyro_y,gyro_z\n0,1,2,3\n4,5,6,7\n8,9,10,11\n",
    "hand_tracking.csv": "frame_number,left_joint0_id,left_joint0_name,left_active\n0,0,PALM,1\n1,0,PALM,1\n2,0,PALM,1\n",
    "head_pose.csv": "timestamp_ns,ee_pos_x,ee_pos_y,ee_pos_z\n0,1,2,3\n4,5,6,7\n8,9,10,11\n",
    "rgb_metainfo.csv": "frame_index,pts_us\n0,0\n1,1\n2,2\n",
    "tracking_metainfo.csv": "frame_index,pts_us\n0,0\n1,1\n2,2\n",
}

# 各 json 的标定内容（含标定关键键）。
_JSON_CALIB = {
    "camera_params_ctrl.json": {"intrinsics": {"fx": 1, "fy": 1}},
    "camera_params_rgb.json": {"intrinsics": {"fx": 1, "fy": 1}},
    "camera_params_tracking.json": {"extrinsics": {"matrix": [[1, 0, 0, 0]]}},
    "imu_calibration.json": {"matrix": [[1, 0, 0], [0, 1, 0]]},
}


def _make_real_like_dir(tmp_path: Path, name: str = "real_dataset") -> Path:
    """用真实文件名构造合成目录，内容可合成。"""
    root = tmp_path / name
    root.mkdir()
    for fname in _REAL_FILES:
        p = root / fname
        if fname.endswith(".csv"):
            p.write_text(_CSV_HEADERS.get(fname, "a,b,c\n"), encoding="utf-8")
        elif fname.endswith(".json"):
            p.write_text(
                json.dumps(_JSON_CALIB.get(fname, {"note": "x"})),
                encoding="utf-8",
            )
        elif fname.endswith(".mp4"):
            p.write_bytes(b"fakemovie")
        elif fname.endswith(".m4a"):
            p.write_bytes(b"fakeaudio")
        elif fname.endswith(".jpg"):
            p.write_bytes(b"fakeimage")
        else:
            p.write_bytes(b"other")
    return root


def test_probe_directory_lists_all_18_files_grouped() -> None:
    """probe_directory 应返回完整分组清单，覆盖全部 18 个文件且不抽样。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = _make_real_like_dir(Path(td))
        probe = probe_directory(root)

    # 总数应为 18。
    assert probe["total_files"] == 18

    # 所有 18 个文件名应出现在某个分组列表中（路径完整，取 basename 比较）。
    all_listed = (
        probe["tables"] + probe["videos"] + probe["audios"]
        + probe["images"] + probe["cals"] + probe["others"]
    )
    listed_names = {Path(p).name for p in all_listed}
    assert listed_names == set(_REAL_FILES)

    # 按类型分组的期望分布。
    assert Path(probe["tables"][0]).name if probe["tables"] else None  # 列表非空
    assert len(probe["videos"]) == 3   # ctrl/rgb/tracking.mp4
    assert len(probe["audios"]) == 1   # audio.m4a
    assert len(probe["images"]) == 1   # 信息.jpg
    assert len(probe["cals"]) == 4     # 4 个 json（含标定键）
    # csv 共 9 个表格。
    assert len(probe["tables"]) == 9


def test_load_directory_survey_complete_and_streams_full() -> None:
    """目录加载应返回完整清单，流登记表覆盖全部表格+视频+音频+图片。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = _make_real_like_dir(Path(td))
        ctx = RunContext()
        result = load_dataset_impl(ctx, str(root))

    assert result["success"] is True
    survey = result["file_survey"]
    assert survey["total_files"] == 18

    # file_survey 各分组非空且覆盖全部文件。
    listed = (
        survey["tables"] + survey["videos"] + survey["audios"]
        + survey["images"] + survey["cals"] + survey["others"]
    )
    assert {Path(p).name for p in listed} == set(_REAL_FILES)

    # 流登记表应覆盖：9 表格 + 3 视频 + 1 音频 + 1 图片 = 14 条。
    streams = ctx.meta["streams"]
    assert len(streams) == 14
    kinds = [s["kind"] for s in streams]
    assert kinds.count("video") == 3
    assert kinds.count("audio") == 1
    assert kinds.count("image") == 1
    assert kinds.count("imu") >= 1  # accel+gyro 表应被识别为 imu

    # 音频/图片流只登记路径与格式，不含 channels。
    audio_stream = next(s for s in streams if s["kind"] == "audio")
    assert audio_stream["path"].endswith("audio.m4a")
    assert audio_stream["channels"] == []


def test_load_directory_deterministic_across_two_loads() -> None:
    """同一目录两次加载，file_survey 与 streams 必须完全一致。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = _make_real_like_dir(Path(td))
        ctx_a = RunContext()
        ctx_b = RunContext()
        res_a = load_dataset_impl(ctx_a, str(root))
        res_b = load_dataset_impl(ctx_b, str(root))

    # file_survey 深度相等（json 序列化比较，避免对象引用差异）。
    assert json.dumps(res_a["file_survey"], sort_keys=True, ensure_ascii=False) == \
        json.dumps(res_b["file_survey"], sort_keys=True, ensure_ascii=False)
    # streams 也需一致。
    assert json.dumps(ctx_a.meta["streams"], sort_keys=True, ensure_ascii=False) == \
        json.dumps(ctx_b.meta["streams"], sort_keys=True, ensure_ascii=False)
    # probe_directory 本身也应确定性。
    probe_a = probe_directory(root)
    probe_b = probe_directory(root)
    assert json.dumps(probe_a, sort_keys=True, ensure_ascii=False) == \
        json.dumps(probe_b, sort_keys=True, ensure_ascii=False)
