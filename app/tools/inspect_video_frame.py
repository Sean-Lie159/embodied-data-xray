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


def _probe_duration(path: Path) -> float | None:
    """用 ffprobe 取视频时长（秒），用于时刻越界校验；失败返回 None。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        return float(data["format"]["duration"])
    except Exception:  # noqa: BLE001
        return None


def inspect_video_frame_impl(
    context: RunContext,
    video: str,
    at_seconds: float = 0.0,
    output_name: str | None = None,
) -> dict[str, Any]:
    """抽取视频指定时刻的单帧存为图片，返回图片路径与元数据。

    Args:
        context: 运行时上下文（用 output_dir 落盘）。
        video: 视频文件路径或文件名（可为流登记表中的视频流文件名）。
        at_seconds: 抽取时刻（秒，从 0 起）；越界时自动夹到有效范围。
        output_name: 可选，输出图片文件名（不含扩展名）；缺省自动生成。

    Returns:
        dict，含 success、frame_path（相对 output_dir 的路径）、at_seconds、
        video、duration_s、width/height；ffmpeg 不可用时含 degraded 提示。
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
    if duration is not None and at > duration:
        at = max(0.0, duration - 0.1)  # 夹到末尾前 0.1s

    out_dir = context.output_path()
    stem = output_name or f"{src.stem}_frame_{at:g}s"
    out_path = out_dir / f"{stem}.jpg"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{at}", "-i", str(src),
             "-frames:v", "1", "-q:v", "3", str(out_path)],
            capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "ffmpeg_timeout",
            "user_message": f"抽帧超时（>{_FFMPEG_TIMEOUT}s）：{src.name}。",
        }

    if proc.returncode != 0 or not out_path.exists():
        return {
            "success": False,
            "error": "frame_extract_failed",
            "reason": (proc.stderr or "").strip()[:200],
            "user_message": f"抽帧失败：{src.name}（第 {at:g}s）。该视频可能损坏或编码不受支持。",
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
    return {
        "success": True,
        "video": src.name,
        "at_seconds": round(at, 3),
        "duration_s": round(duration, 3) if duration is not None else None,
        "frame_path": rel,
        "width": width,
        "height": height,
        "user_message": (
            f"已抽取 {src.name} 第 {at:g}s 的单帧，存为 {rel}"
            f"（{width}×{height}）。可据此确认该路相机的画面内容"
            "（朝向/遮挡/目标物体），但画面含义需人工判读，工具不做内容理解。"
        ),
    }


@tool
def inspect_video_frame(
    wrapper: RunContextWrapper[RunContext],
    video: str,
    at_seconds: float = 0.0,
    output_name: str | None = None,
) -> dict:
    """抽取视频指定时刻的单帧并保存为图片，用于确认相机画面内容。

    适用：确认某路相机的朝向/遮挡/画面语义（如"这路是不是对着手部"），
    或抽查录制中段的画质。只读抽取，不改动源视频；不做画面内容理解——
    工具只产出图片与元数据，画面含义由用户判读。

    Args:
        video: 视频文件路径或文件名（流登记表中的视频流）。
        at_seconds: 抽取时刻（秒，从 0 起）；越界自动夹到有效范围。
        output_name: 可选，输出图片文件名（不含扩展名）。

    Returns:
        dict，含 success、frame_path（相对 outputs/ 的图片路径）、at_seconds、
        duration_s、width/height；失败时含结构化 error 与 user_message。
    """
    return inspect_video_frame_impl(wrapper.context, video, at_seconds, output_name)
