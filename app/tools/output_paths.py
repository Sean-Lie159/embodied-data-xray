"""输出路径集中管理：按数据集建子目录，产物结构分明。

设计（docs/UI优化总纲与输出目录改造设计.md 第 3 节）：

    outputs/
      by_dataset/<数据集名净化>/
        charts/   <session_tag 前缀><dataset_id>_<chart_type>_<ts>.png
        reports/  <session_tag 前缀><dataset_id>_report_<ts>.md
        profile.json      ← 该数据集的语义确认画像（见 profile_store）
      _misc/charts|reports/   ← 未加载数据集时产生的产物
      uploads/  sample_dataset/  mcap_unpack/   ← 保持顶层（非数据集产物）

关键设计：

- **文件名规则完全不变**（仍含 session_tag 前缀与 dataset_id），多会话输出隔离
  契约零改动（docs/多会话设计.md 3.3）；子目录只是**多一层归类**。
- **dataset_id 本身不被修改**：子目录名是净化后的显示名，仅用于"人类查找"；
  程序定位产物以 context.dataset_id 与 findings 的 file_path 为准。
- 集中在本模块：4 处落盘点需要同一套净化与目录规则，否则报告与图表可能
  落到不同目录；集中后单测只需覆盖一个模块。

本模块为纯 Python（不 import streamlit），可单测。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# 数据集子目录所在的顶层目录名。
_BY_DATASET_DIR = "by_dataset"

# 未加载数据集时的兜底子目录名（不参与净化，固定值）。
_MISC_DIR = "_misc"

# 子目录名长度上限（字符）。超长 dataset_id 截断于此，再附短哈希防碰撞。
_MAX_DIR_NAME = 60

# 产物类型子目录。
_CHARTS_SUBDIR = "charts"
_REPORTS_SUBDIR = "reports"


def sanitize_dataset_dir_name(dataset_id: str | None) -> str:
    """把 dataset_id 净化为安全的子目录名（截断 + 短哈希防碰撞）。

    规则（顺序执行，幂等）：
    1. 去危险字符：路径分隔符、控制字符、Windows 保留字符 ``<>:"|?*`` 与空白
       → ``_``；中文与常规符号保留（与 upload_store 同源思路）；
    2. 超过 60 字符则截断到 60；
    3. **无条件**追加 ``-<sha1(dataset_id) 前 6 位>``——为什么不只在截断时才加：
       两个不同的超长 dataset_id 截断到 60 字符后前 60 位可能相同（如带不同
       时间戳后缀的同批数据），仅截断不加哈希会**串目录**。无条件加使映射稳定。
    4. dataset_id 为 None/空 → 固定 ``_misc``（兜底，"未加载数据集"）。

    Args:
        dataset_id: 数据集标识名（可能含 ``:``、超长、为空）。

    Returns:
        安全的子目录名，如 ``lerobot-9f3a2c``、``_misc``。
    """
    raw = (dataset_id or "").strip()
    if not raw:
        return _MISC_DIR
    # 1. 危险字符与空白 → 下划线（保留中文、字母数字、点、连字符、下划线）。
    cleaned = re.sub(r"[^\w.\-\u4e00-\u9fff]+", "_", raw, flags=re.UNICODE)
    cleaned = cleaned.strip("._") or "dataset"
    # 2. 截断。
    if len(cleaned) > _MAX_DIR_NAME:
        cleaned = cleaned[:_MAX_DIR_NAME]
    # 3. 短哈希（基于原始 id，保证不同 id 映射到不同目录）。
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
    return f"{cleaned}-{digest}"


def dataset_output_dir(output_dir: str, dataset_id: str | None) -> Path:
    """返回某数据集的产物根目录（``outputs/by_dataset/<净名>/``），不存在则创建。

    Args:
        output_dir: 项目输出目录（settings.output_dir 或 RunContext.output_dir）。
        dataset_id: 数据集标识名；None/空 → ``outputs/_misc/``。

    Returns:
        数据集产物目录（Path）。
    """
    base = Path(output_dir) if output_dir else Path("outputs")
    d = base / _BY_DATASET_DIR / sanitize_dataset_dir_name(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def chart_dir(output_dir: str, dataset_id: str | None) -> Path:
    """图表子目录（``.../by_dataset/<净名>/charts/``），不存在则创建。"""
    d = dataset_output_dir(output_dir, dataset_id) / _CHARTS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(output_dir: str, dataset_id: str | None) -> Path:
    """报告子目录（``.../by_dataset/<净名>/reports/``），不存在则创建。"""
    d = dataset_output_dir(output_dir, dataset_id) / _REPORTS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d
