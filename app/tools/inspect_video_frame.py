"""视频帧抽查：按时刻抽取单帧存图，供确认相机语义/遮挡/画面内容。

**为什么需要**：项目对视频只有 ffprobe 元数据（分辨率/帧率/帧数/时长），
"这路相机对着哪里""画面有没有被遮挡"这类问题无法回答——而这正是判断相机
语义（手部跟踪 / 第一人称 / 正面）的关键证据。完整视频内容理解成本高且不必
要，**抽一帧看图**即可覆盖绝大多数确认需求。

设计边界（与项目既有纪律一致）：
- **只读抽取**：用 ffmpeg 抽单帧存到 ``outputs/`` 下，绝不改动源视频；
- **不做内容理解**：本工具只产出图片路径与元数据，画面含义由用户/模型看图判断；
- **依赖可降级**：ffmpeg 不可用时返回结构化提示（与 ffprobe 同款处理）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from agents import RunContextWrapper
from agents.decorators import tool

from app.agent.context import RunContext

# ffmpeg 单次调用超时（秒）：抽单帧正常应在数秒内完成。
_FFMPEG_TIMEOUT = 60


def _ffmpeg_available() -> bool:
    """真正尝试调用一次 ffmpeg（比 shutil.which 更可靠，与 ffprobe 同款）。"""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True,
            timeout=10, encoding="utf-8", errors="replace",
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# 无容器索引的裸码流扩展名：**不支持输入侧 -ss**（无索引可 seek），
# 必须用输出侧 seek（`-i` 在前、`-ss` 在后），由解码器边解码边丢弃到目标时刻。
_RAW_VIDEO_EXTS = {".h265", ".hevc", ".h264"}


def _is_raw_stream(path: Path) -> bool:
    """判断是否为无容器裸码流（决定抽帧策略与时长来源）。"""
    return path.suffix.lower() in _RAW_VIDEO_EXTS


def _probe_duration(path: Path) -> float | None:
    """用 ffprobe 取视频时长（秒）；失败返回 None。

    **裸码流的 ``format.duration`` 为 N/A**（无容器索引），此时改用
    "实测帧数 ÷ 帧率"推算（与 ``_sniffing.probe_video`` 同口径）——否则抽帧
    的越界夹取失效（用户给 1000 秒也不会被夹回有效范围）。
    """
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            raw = (data.get("format") or {}).get("duration")
            if raw not in (None, "", "N/A"):
                return float(raw)
    except Exception:  # noqa: BLE001
        pass
    # 回退：裸流经帧数/帧率推算时长。
    if _is_raw_stream(path):
        try:
            from app.tools._sniffing import probe_video

            meta = probe_video(str(path))
            dur = meta.get("duration_s")
            if isinstance(dur, (int, float)) and dur > 0:
                return float(dur)
        except Exception:  # noqa: BLE001
            return None
    return None


def _estimate_duration_from_registry(context: RunContext, src: Path) -> float | None:
    """从上下文中的视频元数据/同名时间戳清单估算视频时长（裸流兜底）。

    裸码流的 ``format.duration`` 为 N/A，但时长通常可从**同目录的逐帧时间戳
    清单**（真实采集的标准形态：``<相机>.txt`` 每行一个时间戳）精确得到——
    首末时间戳之差即录制跨度。这比"全量解码数帧"廉价得多（读文本 vs 解码
    563 MB 视频）。

    Args:
        context: 运行时上下文（取 meta.video_meta）。
        src: 视频文件路径。

    Returns:
        估算时长（秒）；无法估算返回 None。
    """
    # 途径 1：同名 .txt 时间戳清单的首末差（最准且最省）。
    txt = src.parent / f"{src.stem}.txt"
    if txt.exists():
        try:
            lines = txt.read_text(encoding="utf-8", errors="replace").splitlines()
            stamps: list[float] = []
            for ln in lines[:1] + lines[-1:]:  # 只解析首末两行，避免全量解析
                parts = ln.split()
                if parts:
                    try:
                        stamps.append(float(parts[0]))
                    except ValueError:
                        pass
            if len(stamps) == 2 and stamps[1] > stamps[0]:
                delta = stamps[1] - stamps[0]
                # 量级判定：>=1e12 视为纳秒，否则按微秒/毫秒处理过于冒险——
                # 只在明显是纳秒（19 位 epoch）时换算，其余不猜。
                if stamps[0] >= 1e17:
                    return delta / 1e9
                if stamps[0] >= 1e14:
                    return delta / 1e6
                if stamps[0] >= 1e11:
                    return delta / 1e3
        except Exception:  # noqa: BLE001
            pass
    # 途径 2：目录探测已登记的 video_meta（含帧数/帧率时换算）。
    try:
        for v in context.meta.get("video_meta", []) or []:
            if Path(str(v.get("file", ""))).name != src.name:
                continue
            dur = v.get("duration_s")
            if isinstance(dur, (int, float)) and dur > 0:
                return float(dur)
            nf, fps = v.get("nb_frames"), v.get("fps")
            if isinstance(nf, (int, float)) and isinstance(fps, (int, float)) and fps > 0:
                return float(nf) / float(fps)
    except Exception:  # noqa: BLE001
        pass
    return None


def inspect_video_frame_impl(
    context: RunContext,
    video: str,
    at_seconds: float = 0.0,
    output_name: str | None = None,
    count_frames: bool = False,
) -> dict[str, Any]:
    """抽取视频指定时刻的单帧存为图片，返回图片路径与元数据。

    Args:
        context: 运行时上下文（用 output_dir 落盘）。
        video: 视频文件路径或文件名（可为流登记表中的视频流文件名）。
        at_seconds: 抽取时刻（秒，从 0 起）；越界时自动夹到有效范围。
        output_name: 可选，输出图片文件名（不含扩展名）；缺省自动生成。
        count_frames: 默认 False。置 True 时额外**全量解码统计精确帧数**
            （裸码流无索引，探测阶段拿不到帧数）。代价高：563MB/14142 帧实测
            约 90 秒，故默认关闭；仅在需要核对"视频帧数是否与时间戳逐帧对应"
            时按需开启。

    Returns:
        dict，含 success、frame_path（相对 output_dir 的路径）、at_seconds、
        video、duration_s、width/height；ffmpeg 不可用时含 degraded 提示。
        count_frames=True 时含 nb_frames_measured（实测帧数）。
    """
    # 解析视频路径：支持直接路径，也支持流登记表里的文件名。
    src = Path(video)
    if not src.exists():
        name_lower = Path(video).name.lower()
        for s in context.meta.get("streams", []):
            p = s.get("path", "")
            if s.get("kind") == "video" and Path(p).name.lower() == name_lower:
                src = Path(p)
                break
    if not src.exists():
        return {
            "success": False,
            "error": "video_not_found",
            "user_message": (
                f"视频文件不存在：{video}。可用视频见 inspect_streams 的视频流清单。"
            ),
        }

    if not _ffmpeg_available():
        return {
            "success": False,
            "error": "ffmpeg_unavailable",
            "user_message": (
                "未检测到可用的 ffmpeg，无法抽帧。请安装 ffmpeg 并确保其 bin 目录"
                "在 PATH 中（用 `ffmpeg -version` 验证）。"
            ),
        }

    duration = _probe_duration(src)
    at = max(0.0, float(at_seconds))
    clamped = False
    if duration is not None and at > duration:
        at = max(0.0, duration - 0.1)  # 夹到末尾前 0.1s
        clamped = True
    elif duration is None:
        # **裸流无时长信息时的兜底**（2026-09-14）：duration 为 None 会让越界
        # 夹取失效——用户给 1000s（超出 471s 的录制）时，ffmpeg 会一直解码到
        # 文件末尾才发现没有更多帧，实测耗时 19s 且最终失败。
        # 这里用**流登记表的实测帧数 ÷ 帧率**估算时长（若目录已探测到），
        # 从而把越界值夹回有效范围；估算不可用则保持原值（由超时兜底）。
        est = _estimate_duration_from_registry(context, src)
        if est is not None and est > 0 and at > est:
            at = max(0.0, est - 0.1)
            duration = est
            clamped = True

    out_dir = context.output_path()
    stem = output_name or f"{src.stem}_frame_{at:g}s"
    out_path = out_dir / f"{stem}.jpg"

    # 抽帧命令：**裸码流必须用输出侧 seek**（2026-09-14 实测）。
    #
    # 裸 HEVC 无容器索引，输入侧 `-ss`（在 -i 之前）会报
    # "could not seek to position 0.000" 且**以 returncode=0 退出但不产出任何
    # 帧**（out 为 0 字节）——静默失败。改成 `-i` 在前、`-ss` 在后，由解码器
    # 边解码边丢弃到目标时刻，实测 0.28s 即出图（远快于全量解码）。
    # 容器格式（mp4 等）两种都行，统一用输出侧 seek 以收敛为一条路径。
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if at > 0:
        cmd += ["-ss", f"{at}"]
    cmd += ["-frames:v", "1", "-q:v", "3", str(out_path)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "ffmpeg_timeout",
            "user_message": (
                f"抽帧超时（>{_FFMPEG_TIMEOUT}s）：{src.name}。"
                "裸码流无索引，抽取越靠后的时刻耗时越长；"
                "可改抽更早的时刻（如 at_seconds=0）重试。"
            ),
        }

    # **不可只看 returncode**：实测裸流 seek 失败时 rc=0 但产物为空——
    # 必须同时校验产物存在且非空，否则会把"静默失败"当成成功。
    produced = out_path.exists() and out_path.stat().st_size > 0
    if proc.returncode != 0 or not produced:
        stderr = (proc.stderr or "")
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink(missing_ok=True)  # 清理 0 字节残留，避免误当作成果
        hint = ""
        if "could not seek" in stderr or "nothing was encoded" in stderr:
            hint = "（该文件为无索引的裸码流，已自动改用输出侧 seek；若仍失败请改抽更早时刻）"
        return {
            "success": False,
            "error": "frame_extract_failed",
            "reason": stderr.strip().splitlines()[-1][:200] if stderr.strip() else "",
            "user_message": (
                f"抽帧失败：{src.name}（第 {at:g}s）{hint}。"
                "请检查该视频编码是否受当前 ffmpeg 支持。"
            ),
        }

    # 图片尺寸（用 matplotlib 读，避免引入新依赖；失败不影响主流程）。
    width = height = None
    try:
        import matplotlib.image as mpimg

        img = mpimg.imread(str(out_path))
        height, width = int(img.shape[0]), int(img.shape[1])
    except Exception:  # noqa: BLE001
        pass

    rel = out_path.relative_to(out_dir).as_posix() if out_dir in out_path.parents else out_path.name
    result: dict[str, Any] = {
        "success": True,
        "video": src.name,
        "at_seconds": round(at, 3),
        "duration_s": round(duration, 3) if duration is not None else None,
        "frame_path": rel,
        "width": width,
        "height": height,
        "raw_stream": _is_raw_stream(src),
        "user_message": (
            f"已抽取 {src.name} 第 {at:g}s 的单帧，存为 {rel}"
            f"（{width}×{height}）。可据此确认该路相机的画面内容"
            "（朝向/遮挡/目标物体），但画面含义需人工判读，工具不做内容理解。"
        ),
    }

    # 按需实测帧数（默认不做——全量解码代价高，见 docstring）。
    if count_frames:
        from app.tools._sniffing import _count_frames_raw

        measured = _count_frames_raw(str(src))
        if measured is None:
            result["nb_frames_measured"] = None
            result["count_frames_note"] = (
                "帧数实测失败（解码异常或超时）。若视频较大，全量解码可能超过"
                "内部超时限制。"
            )
        else:
            n = measured["nb_frames"]
            result["nb_frames_measured"] = n
            # 与同名时间戳清单核对（相机采集的典型形态：逐帧时间戳 .txt）。
            txt = src.parent / f"{src.stem}.txt"
            if txt.exists():
                try:
                    n_ts = sum(1 for ln in txt.read_text(
                        encoding="utf-8", errors="replace").splitlines() if ln.strip())
                    result["timestamp_rows"] = n_ts
                    result["frames_match_timestamps"] = (n == n_ts)
                    result["user_message"] += (
                        f" 实测帧数 {n:,}，同名时间戳清单 {txt.name} 有 {n_ts:,} 行"
                        + ("——**逐帧一一对应**。" if n == n_ts
                           else "——两者**不一致**，可能存在丢帧或多余帧。")
                    )
                except Exception:  # noqa: BLE001
                    result["user_message"] += f" 实测帧数 {n:,}。"
            else:
                result["user_message"] += f" 实测帧数 {n:,}。"
    return result


@tool
def inspect_video_frame(
    wrapper: RunContextWrapper[RunContext],
    video: str,
    at_seconds: float = 0.0,
    output_name: str | None = None,
    count_frames: bool = False,
) -> dict:
    """抽取视频指定时刻的单帧并保存为图片，用于确认相机画面内容。

    适用：确认某路相机的朝向/遮挡/画面语义（如"这路是不是对着手部"），
    或抽查录制中段的画质。只读抽取，不改动源视频；不做画面内容理解——
    工具只产出图片与元数据，画面含义由用户判读。

    **支持裸码流**（``.h265``/``.hevc``/``.h264``）：这类文件无容器索引，
    抽帧自动改用输出侧 seek（内部处理，无需用户关心）。注意裸流的时长与帧数
    在目录探测阶段不可知（需全量解码），若用户问"这个视频多少帧/多长"，
    可用 ``count_frames=True`` 实测（代价高，先告知用户预计耗时）。

    Args:
        video: 视频文件路径或文件名（流登记表中的视频流）。
        at_seconds: 抽取时刻（秒，从 0 起）；越界自动夹到有效范围。
        output_name: 可选，输出图片文件名（不含扩展名）。
        count_frames: 默认 False。置 True 时额外全量解码统计**精确帧数**，
            并与同名时间戳清单核对是否逐帧对应。**代价高**（563MB 裸流实测
            约 90 秒），仅在用户明确要求核对帧数时使用。

    Returns:
        dict，含 success、frame_path（相对 outputs/ 的图片路径）、at_seconds、
        duration_s、width/height；count_frames=True 时含 nb_frames_measured
        与 frames_match_timestamps；失败时含结构化 error 与 user_message。
    """
    return inspect_video_frame_impl(
        wrapper.context, video, at_seconds, output_name, count_frames)
