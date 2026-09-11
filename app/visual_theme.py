"""跨层视觉常量（UI 层与工具层共用；**不 import streamlit**）。

为什么独立成模块（docs/UI视觉优化设计.md 5.2）：
配色需要被两处引用——`.streamlit/config.toml` 的 ``chartCategoricalColors``
（Streamlit 内置图表）与 ``app/tools/plot_chart.py`` 的 matplotlib 图表。
若把常量放 ``app/ui/``，工具层 import 它会造成"工具层依赖 UI 层"的依赖方向
错觉，违背 ARCHITECTURE.md"业务逻辑与 UI 完全解耦"的设计目标（且将来谁在
``app/ui/constants.py`` 里加一行 ``import streamlit`` 就会真的引入违规依赖）。
故本模块置于 ``app/`` 顶层，不属 ui/ 也不属 tools/，明确表达"跨层的视觉定义"。

变更纪律：
- ``CHART_COLORS`` 若修改，**必须同步** ``.streamlit/config.toml`` 的
  ``chartCategoricalColors``（有单测 ``test_visual_theme.py`` 断言两者相等）；
- 颜色数量必须正好 10（Streamlit 对 ``chartCategoricalColors`` 的硬性要求）。
"""

from __future__ import annotations

# 图表分类配色（10 色，与主色 #2E6F8E 同族的冷色序列）。
# 与 .streamlit/config.toml 的 chartCategoricalColors 必须逐值一致。
CHART_COLORS: list[str] = [
    "#2E6F8E", "#4F9BBF", "#7BC0D9", "#A8D5E2", "#C9E4EA",
    "#8FB8A8", "#B7C9A8", "#D9CBA8", "#C9A8B7", "#9E9E9E",
]

# 图表文字/刻度颜色：**中性灰**。
#
# 为什么不用黑或白（docs/UI视觉优化设计.md 5.1）：图表以 png 落盘，是静态图片，
# 无法随页面主题重绘。深色页面上深色文字不可读、浅色页面上白色文字不可读，
# 故只能取中间明度——这样同一张图在浅色底与深色底上都可辨识，避免"生成两份图"。
CHART_INK: str = "#6B7C86"

# 图表网格线颜色：比文字更淡，避免抢视线。
CHART_GRID: str = "#B8C4CB"

# 品牌主色（与 config.toml 的 primaryColor 一致）；供需要"应用色"的图表元素引用。
BRAND_PRIMARY: str = "#2E6F8E"
