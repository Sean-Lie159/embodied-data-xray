"""图表中文字体解析与注册（跨平台，不 import streamlit）。

为什么需要独立模块（2026-09-14 改造）：`app/tools/plot_chart.py` 此前把图表内
文字**一律锁成英文**，理由是"matplotlib 默认字体缺中文字形会渲染成方框"。实测
（本机 matplotlib 3.11 + Windows）该前提已不成立：系统装有微软雅黑 / 黑体 /
思源黑体等中文字体，配置后渲染零缺字警告；而锁英文的代价是**用户看到
"line chart"、"histogram chart" 这类无信息量的标题**——模型即便传了中文标题
"关节角度随时间的响应曲线"，也会被 `_safe_title` 静默丢弃并回退英文。

设计要点：
- **按平台探测候选字体**，注册第一个真实存在的（Windows / macOS / Linux 各自
  的中文字体清单），全部找不到时回退到不含中文的默认族（此时图表标题会自动
  降级为不含中文的形式，见 plot_chart）；
- **不装字体、不下载**：只使用系统已有字体，零新增依赖；
- 结果是"尽力而为"：找不到中文字体不报错，只是中文会显示为方框——因此
  `has_cjk_font()` 供上层判断要不要生成中文文案。
"""

from __future__ import annotations

from functools import lru_cache

# 候选中文字体族（按优先级）。每一族名都是 matplotlib 可识别的 family 名，
# 命中判定用 font_manager.findfont(fallback_to_default=False)——比"猜文件名"
# 可靠（字体文件名与族名常常不一致，如 msyh.ttc 的族名是 Microsoft YaHei）。
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    # Windows
    "Microsoft YaHei",      # 微软雅黑（Win7+ 标配）
    "Microsoft YaHei UI",
    "SimHei",               # 黑体
    "DengXian",             # 等线
    "SimSun",               # 宋体
    "KaiTi",                # 楷体
    "Noto Sans SC",         # 部分环境随 Office / 浏览器装上
    "Source Han Serif SC",
    # macOS
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    # Linux
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "AR PL UMing CN",
    "Droid Sans Fallback",
)


@lru_cache(maxsize=1)
def resolve_cjk_font() -> str | None:
    """返回本机第一个可用的中文字体族名；找不到返回 None。

    结果缓存（字体环境在进程生命周期内不变，探测涉及文件系统扫描，不宜重复）。

    Returns:
        字体族名（可直接放进 ``rcParams["font.sans-serif"]``）；无可用中文字体
        返回 None（调用方据此决定文案是否使用中文）。
    """
    from matplotlib import font_manager

    for family in _CJK_FONT_CANDIDATES:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
        except Exception:  # noqa: BLE001 - 找不到会抛 ValueError，属预期
            continue
        if path:
            return family
    return None


@lru_cache(maxsize=1)
def has_cjk_font() -> bool:
    """本机是否有可用中文字体（供上层决定文案语言）。"""
    return resolve_cjk_font() is not None


@lru_cache(maxsize=1)
def apply_chart_fonts() -> str | None:
    """把中文字体写进 matplotlib rcParams（含负号修正）。

    为什么必须同时设 ``axes.unicode_minus=False``：中文字体的 U+2212（数学减号）
    字形常常缺失或与 ASCII 连字符宽度不同，负数刻度会出现方框或错位；关掉数学
    减号后 matplotlib 用 ASCII ``-``，所有字体都能显示。

    Returns:
        生效的中文字体族名（供日志/测试核对）；无可用中文字体返回 None
        （此时不改动 rcParams，保持 matplotlib 默认，避免引入不存在的族名）。
    """
    import matplotlib

    family = resolve_cjk_font()
    if family is None:
        return None
    existing = list(matplotlib.rcParams.get("font.sans-serif", []))
    # 中文字体置首，其余族名保留作兜底（缺字时逐级回退）。
    rest = [f for f in existing if f != family]
    matplotlib.rcParams["font.sans-serif"] = [family, *rest]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return family
