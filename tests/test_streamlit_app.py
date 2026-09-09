"""streamlit_app 的 AppTest 冒烟测试。

用 Streamlit 的 AppTest 框架加载 app 并运行，验证 UI 组件能渲染且不崩溃
（不触发真实 agent 调用——无 chat_input 输入时不会执行 reply）。
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(_APP), default_timeout=15)


def test_streamlit_app_runs_without_error() -> None:
    """AppTest 加载 app 应成功，无未处理异常。"""
    at = _app()
    at.run()
    # 页面应渲染出标题。
    titles = [t.value for t in at.title]
    assert any("Embodied-data-Xray" in t for t in titles)
    # 三个 tab 应存在。
    tab_labels = [t.label for t in at.tabs]
    assert "图表" in tab_labels
    assert "Findings/报告" in tab_labels
    assert "数据概况" in tab_labels


def test_streamlit_app_no_agent_call_without_input() -> None:
    """无用户输入时，不应触发 agent 执行（messages 为空）。"""
    at = _app()
    at.run()
    # 无 chat_input 输入 → 不应产生 assistant 消息。
    chat_msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert len(chat_msgs) == 0


def test_streamlit_app_user_input_renders_chat() -> None:
    """模拟一次用户输入：应无异常渲染，且聊天记录出现（assistant 消息）。

    这是 UI 改动的回归保障：验证用户输入后，agent 回复能在聊天区正常渲染。
    会触发一次真实模型调用（轻量"你好"），依赖 .env 配置的模型可用。
    """
    at = _app()
    at.run()
    # 模拟用户输入。
    at.chat_input[0].set_value("你好").run()
    # 无未处理异常。
    assert not at.exception
    # 用户与助手消息都应出现。
    user_msgs = [m for m in at.chat_message if m.name == "user"]
    assistant_msgs = [m for m in at.chat_message if m.name == "assistant"]
    assert len(user_msgs) >= 1
    assert len(assistant_msgs) >= 1
    # 助手消息有非空内容。
    assert any((m.markdown[0].value if m.markdown else "") for m in assistant_msgs)

def test_chat_input_outside_scroll_container() -> None:
    """输入框必须在滚动容器之外（钉底固定）——回归：此前在容器内会随聊天
    记录一起滚走，无法同时回顾历史与输入。

    静态断言：chat_input 调用行的缩进应为 8（left 列内层级）；若 >8 说明
    它被放回了滚动容器内部（真实事故形态）。
    """
    import inspect

    import streamlit_app

    src = inspect.getsource(streamlit_app)
    lines = src.splitlines()
    input_line = next(
        i for i, ln in enumerate(lines, 1) if "= st.chat_input(" in ln
    )
    indent = len(lines[input_line - 1]) - len(lines[input_line - 1].lstrip())
    assert indent == 8, (
        f"st.chat_input 缩进为 {indent}，若 >8 说明它被放回了滚动容器内"
    )
    # 输入框应在滚动容器之后（先建容器、后放输入框）。
    container_line = next(
        i for i, ln in enumerate(lines, 1)
        if "st.container(height=_SCROLL_HEIGHT)" in ln
    )
    assert input_line > container_line
