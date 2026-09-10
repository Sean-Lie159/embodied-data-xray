"""Agent 运行时上下文。

``RunContext`` 是单 Agent + 工具循环中在 Python 进程内共享的状态对象。它通过
``Runner.run(context=...)`` 注入，各工具经 ``RunContextWrapper`` 访问。上下文
对象**不会序列化给 LLM**，因此可安全地持有 DataFrame 等非序列化对象。

**单数据集语义（重要）**：任一时刻 `RunContext` 只持有**一个**当前数据集
（``df`` 与 ``dataset_id``），新加载会覆盖旧加载。此前的数据集不再可被工具操作，
其数字只能来自对话历史中工具真实返回过的结果（必须标注出处）。

字段约定：
- ``df``：当前已加载的数据集（未加载时为 None）。
- ``dataset_id``：当前数据集的标识名（取自加载路径的文件名，无扩展名）。
- ``meta``：数据集元信息（来源、格式、行/列数等），供工具按需查询。
- ``output_dir``：图表与报告的保存目录。
- ``findings``：各工具累积的分析结果摘要，供 generate_report 汇总。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class RunContext:
    """单次 Agent 运行共享的可变上下文。

    注意：单数据集语义——``load_dataset`` 新加载会覆盖旧的 ``df`` 与
    ``dataset_id``，旧数据集不再可被工具操作。
    """

    df: pd.DataFrame | None = None
    dataset_id: str | None = None
    meta: dict = field(default_factory=dict)
    output_dir: str = "outputs"
    findings: list = field(default_factory=list)
    # 最近一次历史压缩的统计（未压缩过为 None）；由 run_turn 在自动压缩时写入，
    # 供 UI/CLI 展示"已压缩 N 条、节省约 X token"。
    last_compaction: dict | None = None
    # 会话标识（UI 多会话时由 ChatService 生成）：仅用作**输出文件名前缀**，
    # 避免多个会话分析同一数据集时输出互相覆盖。缺省空串 → 文件名与单会话
    # 场景完全一致（零回归）。
    session_tag: str = ""

    def output_path(self) -> Path:
        """返回输出目录的绝对路径，目录不存在时自动创建。"""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
