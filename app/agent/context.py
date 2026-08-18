"""Agent 运行时上下文。

``RunContext`` 是单 Agent + 工具循环中在 Python 进程内共享的状态对象。它通过
``Runner.run(context=...)`` 注入，各工具经 ``RunContextWrapper`` 访问。上下文
对象**不会序列化给 LLM**，因此可安全地持有 DataFrame 等非序列化对象。

字段约定：
- ``df``：当前已加载的数据集（未加载时为 None）。
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
    """单次 Agent 运行共享的可变上下文。"""

    df: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)
    output_dir: str = "outputs"
    findings: list = field(default_factory=list)

    def output_path(self) -> Path:
        """返回输出目录的绝对路径，目录不存在时自动创建。"""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
