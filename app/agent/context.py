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

    def output_path(self) -> Path:
        """返回输出目录的绝对路径，目录不存在时自动创建。"""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
