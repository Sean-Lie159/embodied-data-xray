"""多会话（多开对话）相关能力的单元测试。

覆盖三处多实例化障碍的修复：
  1. 历史压缩预算从模块级全局改为**按会话传入**（多会话各有各的预算）；
  2. `RunContext.session_tag` 用于输出文件名隔离（缺省空串 → 零回归）；
  3. `profile_store` 的并发写入保护（合并而非替换）。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.context import RunContext
from app.services.chat_service import ChatService, _split_turns


# --- 1. 会话隔离（RunContext / ChatService 天然多实例）---------------------


def test_two_contexts_are_isolated() -> None:
    """两个 RunContext 互不影响（每会话一份 → 单数据集语义升级为每会话单数据集）。"""
    a = RunContext(output_dir="outputs", dataset_id="ds_a")
    b = RunContext(output_dir="outputs", dataset_id="ds_b")
    a.meta["streams"] = [{"path": "a.csv"}]
    b.meta["streams"] = [{"path": "b.csv"}]
    a.findings.append("a finding")
    assert b.dataset_id == "ds_b"
    assert b.meta["streams"] == [{"path": "b.csv"}]
    assert b.findings == []


def test_two_services_have_independent_state() -> None:
    """两个 ChatService 实例的历史与上下文互相独立。"""
    a = ChatService.__new__(ChatService)
    b = ChatService.__new__(ChatService)
    a.context, b.context = RunContext(dataset_id="a"), RunContext(dataset_id="b")
    a.history_input = [{"role": "user", "content": "A 的问题"}]
    b.history_input = [{"role": "user", "content": "B 的问题"}]
    assert a.context.dataset_id != b.context.dataset_id
    assert len(_split_turns(a.history_input)) == 1
    assert a.history_input[0]["content"] == "A 的问题"
    assert b.history_input[0]["content"] == "B 的问题"


# --- 2. 压缩预算按会话传入（不再互相覆盖）----------------------------------


def test_history_budget_is_per_turn_not_global() -> None:
    """压缩参数经 run_turn 参数传入：不同预算互不覆盖（模块级全局已移除）。"""
    import inspect

    import app.agent.agent as ag

    sig = inspect.signature(ag.run_turn)
    assert "history_budget_tokens" in sig.parameters
    assert "history_keep_recent_turns" in sig.parameters
    # 模块级全局已改名为默认值（不再是运行期被覆盖的可变状态）。
    assert not hasattr(ag, "_history_budget_tokens")


def test_service_holds_own_budget() -> None:
    """ChatService 持有自己的压缩预算（不依赖全局）。"""
    from app.services.chat_service import ChatService as CS

    src = inspect_src(CS._configure_compaction)
    assert "self._history_budget" in src
    assert "self._keep_recent_turns" in src
    run_src = inspect_src(CS.areply)
    assert "history_budget_tokens=self._history_budget" in run_src


def inspect_src(fn) -> str:
    import inspect

    return inspect.getsource(fn)


def test_run_turn_uses_passed_budget(tmp_path: Path) -> None:
    """run_turn 用传入预算判定压缩（预算 0 → 不压缩）。

    直接验证参数生效（构造超长历史 + 预算 0，历史应原样保留）。
    """
    import asyncio

    import app.agent.agent as ag

    captured: dict = {}
    real_compact = None

    # 构造一个假 agent 会太重；这里直接验证参数解析路径：
    # 预算 0 时不应进入压缩分支（通过 monkeypatch compact_history 观察是否调用）。
    import app.agent.history_compaction as hc

    real = hc.compact_history

    def _spy(*a, **k):
        captured["called"] = True
        return real(*a, **k)

    hc.compact_history = _spy
    try:
        # 用最小可运行路径：Runner.run 会失败，但压缩判定在 run 之前执行。
        class _FakeAgent:  # noqa: D401
            pass

        long_history = [{"role": "user", "content": "x" * 200_000}]

        async def _drive() -> None:
            try:
                await ag.run_turn(
                    _FakeAgent(), RunContext(), "hi", long_history,
                    history_budget_tokens=0,  # 关闭压缩
                )
            except Exception:  # noqa: BLE001 - 预期在 Runner.run 处失败
                pass

        asyncio.run(_drive())
        assert "called" not in captured, "预算 0 不应触发压缩"
    finally:
        hc.compact_history = real


# --- 3. session_tag（输出文件名隔离）下个 commit 覆盖 ------------------------

# --- 3. session_tag：输出文件名隔离（多会话分析同一数据集不互相覆盖）-------


def test_session_tag_default_empty_zero_regression() -> None:
    """缺省 session_tag 为空串 → 文件名与单会话场景完全一致（零回归）。"""
    from app.tools._data_access import output_prefix

    ctx = RunContext(dataset_id="demo")
    assert ctx.session_tag == ""
    assert output_prefix(ctx) == ""


def test_session_tag_prefix_applied() -> None:
    """session_tag 非空 → 输出前缀含会话标识。"""
    from app.tools._data_access import output_prefix

    ctx = RunContext(dataset_id="demo", session_tag="s-1a2b")
    assert output_prefix(ctx) == "s-1a2b_"


def test_plot_and_report_names_include_session_tag(tmp_path: Path) -> None:
    """图表与报告文件名含会话前缀（多会话分析同一数据集不覆盖）。"""
    from app.tools.generate_report import _output_report_path
    from app.tools.plot_chart import _output_path

    ctx_a = RunContext(output_dir=str(tmp_path), dataset_id="same",
                       session_tag="s-aaaa")
    ctx_b = RunContext(output_dir=str(tmp_path), dataset_id="same",
                       session_tag="s-bbbb")
    pa, pb = _output_path(ctx_a, "line"), _output_path(ctx_b, "line")
    assert pa.name.startswith("s-aaaa_same_line_")
    assert pb.name.startswith("s-bbbb_same_line_")
    assert pa != pb  # 同一数据集、同一秒内也不冲突

    ra = _output_report_path(ctx_a)
    rb = _output_report_path(ctx_b)
    assert ra.name.startswith("s-aaaa_same_report_")
    assert rb.name.startswith("s-bbbb_same_report_")
    assert ra != rb


def test_new_session_tag_unique_and_prefixed() -> None:
    """自动生成的会话标识唯一且有前缀。"""
    from app.services.chat_service import _new_session_tag

    tags = {_new_session_tag() for _ in range(20)}
    assert len(tags) == 20
    assert all(t.startswith("s-") and len(t) == 6 for t in tags)


def test_chat_service_accepts_session_tag() -> None:
    """ChatService 可显式指定 / 关闭会话标识（不连真实模型，只验构造路径）。"""
    import inspect

    from app.services.chat_service import ChatService

    sig = inspect.signature(ChatService.__init__)
    assert "session_tag" in sig.parameters

# --- 4. profile 并发保护（多会话同时确认不丢）-------------------------------


def test_concurrent_confirmations_do_not_lose_updates(tmp_path: Path) -> None:
    """**并发合并**：两会话同时确认同一数据集的不同流 → 两条都在。

    这是多会话引入的真实新风险（单会话不存在）：无锁 + 无重读时，
    后写的会整体覆盖先写的。
    """
    import threading

    from app.tools.profile_store import load_profile, save_dataset_profile

    out = str(tmp_path)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _confirm(fname: str, label: str) -> None:
        try:
            barrier.wait(timeout=5)
            save_dataset_profile(
                out, "ds", stream_overrides={fname: {"kind": "x",
                                                     "semantic_label": label}}
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=_confirm, args=(f"a{i}.csv", f"流{i}"))
          for i in range(2)]
    for th in ts:
        th.start()
    for th in ts:
        th.join(timeout=10)

    assert not errors, errors
    streams = load_profile(out)["datasets"]["ds"]["streams"]
    assert "a0.csv" in streams and "a1.csv" in streams, (
        f"并发确认丢失：{list(streams)}"
    )


def test_atomic_write_leaves_no_partial_file(tmp_path: Path) -> None:
    """原子写：写完后无残留临时文件，且 JSON 完整可解析。"""
    import json

    from app.tools.profile_store import _profile_path, save_dataset_profile

    save_dataset_profile(str(tmp_path), "ds",
                         stream_overrides={"a.csv": {"kind": "k"}})
    path = _profile_path(str(tmp_path))
    assert path.exists()
    json.loads(path.read_text(encoding="utf-8"))  # 完整可解析
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert leftovers == [], f"残留临时文件：{leftovers}"


def test_lock_released_after_write(tmp_path: Path) -> None:
    """写入后锁文件被释放（不留残留锁阻塞后续写入）。"""
    from app.tools.profile_store import _profile_path, save_dataset_profile

    save_dataset_profile(str(tmp_path), "ds",
                         stream_overrides={"a.csv": {"kind": "k"}})
    lock = _profile_path(str(tmp_path)).with_name(".dataset_profile.json.lock")
    assert not lock.exists(), "锁文件未释放"
    # 再次写入应正常（锁未残留）。
    save_dataset_profile(str(tmp_path), "ds",
                         stream_overrides={"b.csv": {"kind": "k"}})


def test_merge_preserves_other_sessions_entries(tmp_path: Path) -> None:
    """合并语义：会话 A 确认流 X、会话 B 确认流 Y → 两次写入后都在。

    （现有实现已是按文件名 update；本测试锁定该行为不被改为整体替换。）
    """
    from app.tools.profile_store import load_profile, save_dataset_profile

    out = str(tmp_path)
    save_dataset_profile(out, "ds", stream_overrides={"x.csv": {"kind": "kx"}})
    save_dataset_profile(out, "ds", stream_overrides={"y.csv": {"kind": "ky"}})
    streams = load_profile(out)["datasets"]["ds"]["streams"]
    assert set(streams) == {"x.csv", "y.csv"}
    assert streams["x.csv"]["source"] == "user_confirmed"

# --- 5. UI 多会话（AppTest：新建 / 切换 / 关闭 / 状态保留）-------------------


def _app():
    from pathlib import Path as _P

    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(
        str(_P(__file__).resolve().parents[1] / "streamlit_app.py"),
        default_timeout=20,
    )


def test_ui_creates_one_session_by_default() -> None:
    """首次打开自动创建一个会话（保证始终有 active 会话）。"""
    at = _app()
    at.run()
    assert not at.exception
    assert len(at.session_state.sessions) == 1
    assert at.session_state.active_session in at.session_state.sessions


def test_ui_new_session_button_adds_session() -> None:
    """点"＋ 新建对话" → 会话数 +1 且新会话为 active、messages 为空。"""
    at = _app()
    at.run()
    before = len(at.session_state.sessions)
    btn = [b for b in at.button if b.key == "new_session"][0]
    btn.click().run()
    assert not at.exception
    assert len(at.session_state.sessions) == before + 1
    active = at.session_state.active_session
    assert at.session_state.sessions[active]["messages"] == []


def test_ui_switch_session_preserves_state() -> None:
    """切换标签页：各会话状态独立保留（切换后 messages 仍属于各自会话）。"""
    at = _app()
    at.run()
    first = at.session_state.active_session
    # 给第一个会话塞一条消息（模拟已对话）。
    at.session_state.sessions[first]["messages"] = [
        {"role": "user", "content": "会话1的消息"}]
    # 新建并切到第二个。
    [b for b in at.button if b.key == "new_session"][0].click().run()
    second = at.session_state.active_session
    assert second != first
    assert at.session_state.sessions[second]["messages"] == []
    # 切回第一个：消息仍在（状态保留）。
    [b for b in at.button if b.key == f"tab_{first}"][0].click().run()
    assert at.session_state.active_session == first
    assert at.session_state.sessions[first]["messages"][0]["content"] == "会话1的消息"


def test_ui_close_session_removes_it() -> None:
    """关闭会话：会话数 -1；关到最后一个时自动新建（始终有会话）。"""
    at = _app()
    at.run()
    [b for b in at.button if b.key == "new_session"][0].click().run()
    assert len(at.session_state.sessions) == 2
    target = list(at.session_state.sessions)[0]
    [b for b in at.button if b.key == f"close_{target}"][0].click().run()
    assert not at.exception
    assert len(at.session_state.sessions) == 1
    # 关掉最后一个 → 自动新建。
    last = list(at.session_state.sessions)[0]
    [b for b in at.button if b.key == f"close_{last}"][0].click().run()
    assert len(at.session_state.sessions) == 1


def test_ui_sessions_have_independent_services() -> None:
    """两会话各有独立 ChatService 与 RunContext（数据集互不干扰）。"""
    at = _app()
    at.run()
    [b for b in at.button if b.key == "new_session"][0].click().run()
    sessions = at.session_state.sessions
    services = [s["service"] for s in sessions.values()]
    assert len(services) == 2
    assert services[0] is not services[1]
    assert services[0].context is not services[1].context
    # 会话标识不同 → 输出文件名不冲突。
    tags = {s["service"].context.session_tag for s in sessions.values()}
    assert len(tags) == 2
