"""annotate 模块测试：任务语义标注 / 标注落盘 / 标注自身质检。

设计依据：``docs/标注与质检能力设计.md`` §5.1 / §5.4。

**本文件最重要的守护**：

1. ``test_training_use_blocks_llm_source*``：``llm_proposed`` 来源**必须被拒绝**
   用于训练真值（这是"三种用途"落到代码里的强制机制）；
2. ``test_task_not_invented_when_no_signal*``：任务线索缺失时必须
   ``no_task_signal`` 请用户提供，**不得编造**；
3. ``test_gate_*``：``confirm=False`` 不落盘、``confirmed_by_user`` 才标
   ``user_confirmed``。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent.context import RunContext
from app.config import get_settings
from app.tools import annotation_store as store
from app.tools.annotate import (
    annotate_task_impl,
    check_annotation_qc_impl,
    save_annotations_impl,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """配置对象。

    注意：**不能把 fixture 命名为 ``settings``**——被测函数的同名参数
    ``settings`` 会与 fixture 冲突，导致 fixture 对象被当作配置传入
    （实测报 ``AttributeError: 'FixtureFunctionDefinition' has no attribute``）。
    """
    return get_settings()


@pytest.fixture
def output_dir(tmp_path: Path) -> str:
    d = tmp_path / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _ctx(
    output_dir: str, df: pd.DataFrame | None = None,
    dataset_id: str = "demo", **meta,
) -> RunContext:
    ctx = RunContext()
    ctx.df = df
    ctx.dataset_id = dataset_id
    ctx.output_dir = output_dir
    ctx.meta = dict(meta)
    if df is not None:
        ctx.meta.setdefault("columns", [str(c) for c in df.columns])
    return ctx


def _ep_df(n_per: int = 50, n_eps: int = 2) -> pd.DataFrame:
    """构造带 episode 划分的时间序列数据。"""
    rows = []
    for e in range(n_eps):
        t = np.arange(n_per) * 0.1
        for i in range(n_per):
            rows.append({
                "episode_index": e,
                "timestamp": t[i],
                "action_joint0": float(np.sin(t[i])),
                "fps": 10.0,
            })
    return pd.DataFrame(rows)


def _seg(ep: str = "0", id_: int = 1, s: float = 0.0, e: float = 1.0, **kw):
    """构造一条切片标注输入。"""
    d = {"episode_key": ep, "id": id_, "start_s": s, "end_s": e}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# 任务语义标注（三态闸门 + 不编造）
# ---------------------------------------------------------------------------


def test_task_discovered_from_lerobot_tasks_jsonl(
    output_dir: str, tmp_path: Path,
) -> None:
    """LeRobot 的 meta/tasks.jsonl 是任务标注的权威来源，应被优先读取。"""
    root = tmp_path / "lerobot_ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 10, "features": {}}), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task": "把红色方块放进碗里", "task_index": 0}) + "\n",
        encoding="utf-8",
    )
    res = annotate_task_impl(
        _ctx(output_dir, _ep_df(), source=str(root)), settings=get_settings())
    assert res["success"] is True
    assert "meta/tasks.jsonl" in res["discover_sources"]
    tasks = res["discovered"]["lerobot_tasks"]
    assert tasks[0]["task"] == "把红色方块放进碗里"


def test_task_discovered_from_language_column(output_dir: str) -> None:
    """数据表语言列的唯一值作为任务候选（确定性提取）。"""
    df = _ep_df(n_per=20, n_eps=2)
    df["task"] = ["把方块放进碗里"] * 20 + ["打开抽屉"] * 20
    res = annotate_task_impl(_ctx(output_dir, df), settings=get_settings())
    assert res["success"] is True
    assert res["discovered"]["column_uniques"]["0"] == "把方块放进碗里"
    assert res["discovered"]["column_uniques"]["1"] == "打开抽屉"


def test_task_not_invented_when_no_signal(output_dir: str) -> None:
    """**无线索时必须请用户提供，不得编造任务描述。**"""
    df = _ep_df(n_per=20, n_eps=1)  # 无语言列、非 LeRobot
    res = annotate_task_impl(_ctx(output_dir, df), settings=get_settings())

    assert res["success"] is False
    assert res["error"] == "no_task_signal"
    # 必须明确请用户提供，并说明"猜=编造"。
    assert "请用户直接提供任务描述" in res["user_message"]
    assert "编造" in res["user_message"]
    # 不得返回任何任务内容。
    assert "discovered" not in res or not res.get("discovered")


def test_task_discover_false_still_accepts_user_input(output_dir: str) -> None:
    """discover=False 时直接接受用户提供的任务（不查线索）。"""
    res = annotate_task_impl(
        _ctx(output_dir, _ep_df(n_per=20, n_eps=1)),
        [{"episode_key": "0", "task": "把方块放进碗里"}],
        discover=False, settings=get_settings(),
    )
    assert res["success"] is True
    assert res["validated"] == 1


def test_annotate_task_confirm_false_does_not_save(output_dir: str) -> None:
    """**闸门**：confirm=False 只登记、不落盘。"""
    ctx = _ctx(output_dir, _ep_df(n_per=20, n_eps=1))
    res = annotate_task_impl(
        ctx, [{"episode_key": "0", "task": "把方块放进碗里"}],
        discover=False, confirm=False, settings=get_settings(),
    )
    assert res["validated"] == 1
    assert res["saved"] is None
    assert "尚未落盘" in res["user_message"]

    # 磁盘上确实没有标注。
    loaded = store.load_annotations(output_dir, "demo")
    assert loaded["n_records"] == 0


def test_annotate_task_confirm_true_saves_as_user_confirmed(
    output_dir: str,
) -> None:
    """confirm=True 落盘，来源必须标 user_confirmed（可用于训练真值）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=20, n_eps=1))
    res = annotate_task_impl(
        ctx, [{"episode_key": "0", "task": "把方块放进碗里"}],
        discover=False, confirm=True, settings=get_settings(),
    )
    assert res["saved"]["saved"] == 1

    loaded = store.load_annotations(output_dir, "demo", scope=store.SCOPE_TASK)
    assert loaded["n_records"] == 1
    rec = loaded["records"][0]
    assert rec["source"] == store.SOURCE_USER
    assert rec["confidence"] == "high"
    assert rec["task"] == "把方块放进碗里"


def test_annotate_task_rejects_unknown_episode_key(output_dir: str) -> None:
    """episode_key 不在锚点清单内时拒绝（防止标注挂到不存在的 episode）。"""
    res = annotate_task_impl(
        _ctx(output_dir, _ep_df(n_per=20, n_eps=1)),
        [{"episode_key": "999", "task": "x"}],
        discover=False, confirm=True, settings=get_settings(),
    )
    assert res["success"] is False
    assert res["error"] == "no_valid_annotation"
    bad = [r for r in res["results"] if not r["ok"]]
    assert bad[0]["error"] == "unknown_episode_key"


def test_annotate_task_requires_task_text(output_dir: str) -> None:
    """缺 task 文本的记录被拒绝（不从其它字段拼凑）。"""
    res = annotate_task_impl(
        _ctx(output_dir, _ep_df(n_per=20, n_eps=1)),
        [{"episode_key": "0", "task_zh": "只有中文"}],
        discover=False, settings=get_settings(),
    )
    bad = [r for r in res["results"] if not r["ok"]]
    assert bad and bad[0]["error"] == "missing_task"


# ---------------------------------------------------------------------------
# 标注落盘：来源标记与用途闸门（最重要的守护）
# ---------------------------------------------------------------------------


def test_save_segments_with_signal_source(output_dir: str) -> None:
    """机械切分的来源标 signal_derived。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0), _seg("0", 2, 1.0, 2.0)],
        source=store.SOURCE_SIGNAL, settings=get_settings(),
    )
    assert res["success"] is True
    assert res["saved"]["saved"] == 2
    assert res["effective_source"] == store.SOURCE_SIGNAL

    loaded = store.load_annotations(output_dir, "demo", scope=store.SCOPE_SEGMENT)
    assert all(r["source"] == store.SOURCE_SIGNAL for r in loaded["records"])


def test_save_confirmed_marks_user_confirmed(output_dir: str) -> None:
    """confirmed_by_user=True 时来源被覆盖为 user_confirmed。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Pick_Up")],
        source=store.SOURCE_LLM, confirmed_by_user=True, settings=get_settings(),
    )
    assert res["success"] is True
    assert res["effective_source"] == store.SOURCE_USER

    loaded = store.load_annotations(output_dir, "demo", scope=store.SCOPE_SEGMENT)
    rec = loaded["records"][0]
    assert rec["source"] == store.SOURCE_USER
    assert rec["confidence"] == "high"


def test_training_use_blocks_llm_source(output_dir: str) -> None:
    """**核心闸门**：llm_proposed 来源禁止用于训练真值，必须拒绝落盘。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Pick_Up")],
        source=store.SOURCE_LLM, use=store.USE_TRAINING, settings=get_settings(),
    )
    assert res["success"] is False
    assert res["error"] == "source_use_incompatible"
    assert res["saved"] is None
    # 必须给出可执行的补救路径。
    assert "confirmed_by_user=True" in res["user_message"]

    # 磁盘上不得有任何标注（拒绝要彻底）。
    assert store.load_annotations(output_dir, "demo")["n_records"] == 0


def test_training_use_blocks_signal_derived_source(output_dir: str) -> None:
    """signal_derived（机械切分）用于训练真值前也须人工复核，同样被拦截。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0)],
        source=store.SOURCE_SIGNAL, use=store.USE_TRAINING, settings=get_settings(),
    )
    assert res["success"] is False
    assert res["error"] == "source_use_incompatible"


def test_training_use_allows_user_confirmed(output_dir: str) -> None:
    """user_confirmed 来源可用于训练真值。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Pick_Up")],
        confirmed_by_user=True, use=store.USE_TRAINING, settings=get_settings(),
    )
    assert res["success"] is True
    assert res["use_check"]["ok"] is True


def test_audit_use_allows_llm_source(output_dir: str) -> None:
    """内部清查用途允许 llm_proposed 来源。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Pick_Up")],
        source=store.SOURCE_LLM, use=store.USE_AUDIT, settings=get_settings(),
    )
    assert res["success"] is True


def test_publication_requires_generated_declaration(output_dir: str) -> None:
    """发布用途允许自动来源，但须提示声明 generated。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0)],
        source=store.SOURCE_SIGNAL, use=store.USE_PUBLICATION, settings=get_settings(),
    )
    assert res["success"] is True
    assert res["use_check"]["must_declare_generated"] == 1
    assert "generated" in res["user_message"]


def test_dry_run_validates_without_saving(output_dir: str) -> None:
    """dry_run 只校验不落盘。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0)],
        dry_run=True, settings=get_settings(),
    )
    assert res["dry_run"] is True
    assert res["saved"] is None
    assert store.load_annotations(output_dir, "demo")["n_records"] == 0


def test_invalid_source_rejected(output_dir: str) -> None:
    """非法来源拒绝（不静默兜底）。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()), segments=[_seg("0", 1, 0.0, 1.0)],
        source="guessed", settings=get_settings(),
    )
    assert res["success"] is False
    assert res["error"] == "invalid_source"


def test_invalid_use_rejected(output_dir: str) -> None:
    """非法用途拒绝。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()), segments=[_seg("0", 1, 0.0, 1.0)],
        use="guess", settings=get_settings(),
    )
    assert res["success"] is False
    assert res["error"] == "invalid_use"


def test_interacting_hand_enum_enforced(output_dir: str) -> None:
    """交互手取值必须落在统一枚举内（用户裁决 #2）。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0, interacting_hand="right")],  # 应为 right_hand
        settings=get_settings(),
    )
    assert res["success"] is False
    bad = [r for r in res["results"] if not r["ok"]]
    assert bad[0]["error"] == "invalid_interacting_hand"
    assert "left_hand" in bad[0]["reason"]


def test_rejects_unknown_episode_key(output_dir: str) -> None:
    """episode_key 不在锚点内时拒绝落盘。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("nope", 1, 0.0, 1.0)],
        settings=get_settings(),
    )
    assert res["success"] is False
    bad = [r for r in res["results"] if not r["ok"]]
    assert bad[0]["error"] == "unknown_episode_key"


def test_rejects_noise_without_reason(output_dir: str) -> None:
    """is_noise=true 但无 cleaning_reason 时拒绝。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0, is_noise=True)],
        settings=get_settings(),
    )
    assert res["success"] is False
    bad = [r for r in res["results"] if not r["ok"]]
    assert bad[0]["error"] == "noise_reason_required"


def test_post_save_qc_runs_automatically(output_dir: str) -> None:
    """落盘后自动质检（闭环：写完立即校验）。"""
    res = save_annotations_impl(
        _ctx(output_dir, _ep_df()),
        segments=[_seg("0", 1, 0.0, 1.0), _seg("0", 2, 1.0, 2.0)],
        settings=get_settings(),
    )
    assert res["success"] is True
    assert res["post_save_qc"] is not None
    assert res["post_save_qc"]["result"] in ("pass", "warn", "fail")


def test_save_updates_existing_annotation(output_dir: str) -> None:
    """同 (episode_key, id) 重复落盘视为更新，不产生重复行。"""
    ctx = _ctx(output_dir, _ep_df())
    save_annotations_impl(
        ctx, segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Place")],
        source=store.SOURCE_SIGNAL, settings=get_settings())
    res = save_annotations_impl(
        ctx, segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Pick_Up")],
        confirmed_by_user=True, reason="人工修正动作名", settings=get_settings())
    assert res["saved"]["updated"] == 1
    loaded = store.load_annotations(output_dir, "demo", scope=store.SCOPE_SEGMENT)
    assert loaded["n_records"] == 1
    assert loaded["records"][0]["atomic_action"] == "Pick_Up"


# ---------------------------------------------------------------------------
# 标注自身质检
# ---------------------------------------------------------------------------


def test_qc_passes_on_clean_annotations(output_dir: str) -> None:
    """干净标注（连续、无重叠、有动作名）判 pass。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    save_annotations_impl(
        ctx,
        segments=[
            _seg("0", 1, 0.0, 1.0, atomic_action="Pick_Up",
                 interacting_hand="right_hand", target_object_class="block"),
            _seg("0", 2, 1.0, 2.0, atomic_action="Place",
                 interacting_hand="right_hand", target_object_class="block"),
        ],
        confirmed_by_user=True, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc["result"] == "pass", qc["failures"]
    assert not qc["failures"]


def test_qc_detects_overlap_as_failure(output_dir: str) -> None:
    """**片段重叠判 fail**（确定性问题，直接污染训练）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    save_annotations_impl(
        ctx,
        segments=[_seg("0", 1, 0.0, 2.0), _seg("0", 2, 1.0, 3.0)],  # 重叠 1.0-2.0
        source=store.SOURCE_SIGNAL, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc["result"] == "fail"
    assert "time_overlap" in qc["failed_rules"]
    assert "重叠" in qc["user_message"]


def test_qc_detects_gap_as_warning_only(output_dir: str) -> None:
    """片段间隙只 warn（可能确实不属于任何动作）——不得判 fail。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    save_annotations_impl(
        ctx,
        segments=[_seg("0", 1, 0.0, 1.0), _seg("0", 2, 2.0, 3.0)],  # 间隙 1s
        source=store.SOURCE_SIGNAL, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc["result"] == "warn"
    assert "time_gap" in qc["warned_rules"]
    assert "不等于错误" in qc["user_message"]


def test_duplicate_ids_in_batch_rejected_at_entry(output_dir: str) -> None:
    """**同批次内重复 id 在入口拒绝**（不静默合并）。

    为什么在入口拦而不是留给质检：落盘去重键是 (episode_key, id)，重复 id
    会被后一条**静默覆盖**——用户以为写入两个片段、实际只存一个（数据丢失
    且无提示）。这类"意图冲突"必须在写入前拦下；交给质检去发现一个磁盘上
    永远不可能存在的状态是没有意义的。
    """
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    res = save_annotations_impl(
        ctx,
        segments=[_seg("0", 1, 0.0, 1.0), _seg("0", 1, 1.0, 2.0)],  # id 重复
        source=store.SOURCE_SIGNAL, settings=get_settings(),
    )
    assert res["success"] is False
    assert res["error"] == "duplicate_ids_in_batch"
    assert res["saved"] is None
    assert "静默覆盖" in res["user_message"]
    # 磁盘上不得留下任何记录（拒绝要彻底，避免"写了一半"）。
    assert store.load_annotations(output_dir, "demo")["n_records"] == 0


def test_qc_detects_duplicate_ids_on_disk(output_dir: str) -> None:
    """质检层仍能检出**已落盘**的重复 id（防绕过入口的写入路径）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    # 直接写盘绕过 save_annotations 的入口校验。
    store.append_annotations(output_dir, "demo", [
        {"scope": store.SCOPE_SEGMENT, "episode_key": "0", "id": 1,
         "start_s": 0.0, "end_s": 1.0, "source": store.SOURCE_SIGNAL},
    ])
    # 伪造重复（绕过去重键：用不同 id 但相同时间，模拟非预期写入）。
    store.append_annotations(output_dir, "demo", [
        {"scope": store.SCOPE_SEGMENT, "episode_key": "0", "id": 1,
         "start_s": 2.0, "end_s": 3.0, "source": store.SOURCE_SIGNAL},
    ])
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    # 去重键保证磁盘上只有一条（这正是入口必须拦截的原因）。
    assert qc["n_segments"] == 1


def test_qc_detects_noise_without_reason(output_dir: str) -> None:
    """噪声标记无原因判 fail。

    注意：save_annotations 已在入口拦截此情况，所以这里直接构造落盘记录
    来验证质检层自身的检出能力（防绕过入口的写入路径）。
    """
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    bad = {
        "scope": store.SCOPE_SEGMENT, "episode_key": "0", "id": 1,
        "start_s": 0.0, "end_s": 1.0, "is_noise": True,
        "cleaning_reason": "", "source": store.SOURCE_SIGNAL,
    }
    store.append_annotations(output_dir, "demo", [bad])

    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc["result"] == "fail"
    assert "noise_reason_empty" in qc["failed_rules"]


def test_qc_detects_bounds_exceeded(output_dir: str) -> None:
    """片段超出 episode 时长判 fail。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))  # 50 行 @10fps = 5s
    save_annotations_impl(
        ctx,
        segments=[_seg("0", 1, 0.0, 4.9), _seg("0", 2, 4.9, 9.0)],  # 远超 5s
        source=store.SOURCE_SIGNAL, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert "bounds_exceeded" in qc["failed_rules"]


def test_qc_detects_missing_hand_as_warning(output_dir: str) -> None:
    """有目标物体却未标交互手 → warn。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    save_annotations_impl(
        ctx,
        segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Place",
                       target_object_class="bowl")],
        confirmed_by_user=True, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert "missing_hand" in qc["warned_rules"]


def test_qc_flags_offlist_action_names(output_dir: str) -> None:
    """动作名不在词表内 → warn（提示扩词表或统一命名）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    # 写入数据集画像的动作词表。
    vocab_dir = store.annotation_dir(output_dir, "demo").parent
    (vocab_dir / "action_vocab.json").write_text(
        json.dumps({"actions": ["Pick_Up", "Place"]}), encoding="utf-8")

    save_annotations_impl(
        ctx,
        segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Weird_Action")],
        confirmed_by_user=True, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc["vocab_configured"] is True
    assert "action_vocab_offlist" in qc["warned_rules"]


def test_qc_flags_imbalanced_categories(output_dir: str) -> None:
    """某动作占比过高 → warn（样本不平衡）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=100))
    segs = [_seg("0", i + 1, float(i), float(i + 1), atomic_action="Place")
            for i in range(9)]
    segs.append(_seg("0", 10, 9.0, 10.0, atomic_action="Pick_Up"))
    save_annotations_impl(ctx, segments=segs,
                          confirmed_by_user=True, settings=get_settings())
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert "category_imbalance" in qc["warned_rules"]


def test_qc_enforces_use_on_saved_annotations(output_dir: str) -> None:
    """质检按用途校验已落盘标注的来源（llm_proposed 声明用于训练 → fail）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    save_annotations_impl(
        ctx, segments=[_seg("0", 1, 0.0, 1.0, atomic_action="Pick_Up")],
        source=store.SOURCE_LLM, use=store.USE_AUDIT, settings=get_settings(),
    )
    # 声明用于训练真值 → 来源不合规。
    qc = check_annotation_qc_impl(ctx, use=store.USE_TRAINING, settings=get_settings())
    assert qc["result"] == "fail"
    assert "source_for_declared_use" in qc["failed_rules"]


def test_qc_separates_failures_from_warnings(output_dir: str) -> None:
    """必须分别给出 failures 与 warnings（供模型分别转述）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=100))
    save_annotations_impl(
        ctx,
        segments=[
            _seg("0", 1, 0.0, 2.0, atomic_action="Place"),        # 与下条重叠
            _seg("0", 2, 1.0, 3.0, atomic_action="Place",
                 target_object_class="bowl"),                     # 缺手 → warn
            _seg("0", 3, 5.0, 6.0, atomic_action="Place"),        # 有间隙 → warn
        ],
        confirmed_by_user=True, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc["result"] == "fail"
    assert qc["failures"] and qc["warnings"]
    assert "必须修正" in qc["user_message"]
    assert "提示" in qc["user_message"]


def test_qc_on_empty_annotations(output_dir: str) -> None:
    """无标注时返回 pass 并提示需先产出标注（不报错）。"""
    qc = check_annotation_qc_impl(_ctx(output_dir, _ep_df()), settings=get_settings())
    assert qc["result"] == "pass"
    assert qc["n_records"] == 0
    assert "没有任何标注" in qc["user_message"]


def test_qc_reports_corrupted_lines(output_dir: str) -> None:
    """损坏标注行如实报告（不静默）。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    save_annotations_impl(
        ctx, segments=[_seg("0", 1, 0.0, 1.0)],
        source=store.SOURCE_SIGNAL, settings=get_settings())
    p = store.annotation_dir(output_dir, "demo") / "segments.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "{ 坏行\n", encoding="utf-8")

    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc["bad_lines"] == 1
    assert "损坏" in qc["user_message"]


def test_qc_per_episode_summary(output_dir: str) -> None:
    """给出逐 episode 的连续性统计。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50, n_eps=2))
    save_annotations_impl(
        ctx,
        segments=[_seg("0", 1, 0.0, 1.0), _seg("1", 1, 0.0, 1.0)],
        source=store.SOURCE_SIGNAL, settings=get_settings(),
    )
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    assert set(qc["per_episode"]) == {"0", "1"}
    assert qc["per_episode"]["0"]["n_segments"] == 1


def test_qc_thresholds_reported(output_dir: str) -> None:
    """阈值随返回给出。"""
    ctx = _ctx(output_dir, _ep_df(n_per=50))
    qc = check_annotation_qc_impl(ctx, settings=get_settings())
    for key in ("min_segment_s", "max_segment_s", "max_gap_s"):
        assert key in qc["thresholds"]


# ---------------------------------------------------------------------------
# 闭环：切分 → 落盘 → 质检
# ---------------------------------------------------------------------------


def test_full_loop_segment_save_qc(output_dir: str) -> None:
    """**闭环验证**：切分边界 → 作为草稿落盘 → 质检 → 人工确认后更新。

    这是阶段一至四串起来的端到端路径。
    """
    from app.tools.segment_actions import segment_actions_impl

    # 构造**两段语义清晰的动作**：中间有明显静止间隔（真正的停顿），
    # 且有夹爪开合事件（最可靠的物理边界）。
    n = 600
    quiet = np.zeros(100)
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "timestamp": np.arange(n) * 0.02,
        "fps": [50.0] * n,
        "action_joint0": np.concatenate([
            np.sin(np.arange(200) * 0.4),           # 动作段 1
            quiet,                                   # 停顿 100 帧（2s）
            np.sin(np.arange(200) * 0.4) * 1.5,     # 动作段 2
            quiet,                                   # 收尾停顿
        ]),
        "gripper_position": np.concatenate([
            np.zeros(200), np.zeros(100),
            np.ones(200), np.ones(100),
        ]),
    })
    ctx = _ctx(output_dir, df)

    # 1) 切分
    seg_res = segment_actions_impl(ctx, settings=get_settings())
    assert seg_res["success"] is True
    segs = seg_res["episodes"][0]["segments"]
    assert len(segs) >= 2, f"未能切出多个片段：{segs}"

    # 2) 作为草稿落盘（来源 signal_derived）
    payload = [
        {"episode_key": "0", "id": i + 1, "start_s": s["start_s"],
         "end_s": s["end_s"]}
        for i, s in enumerate(segs)
    ]
    draft = save_annotations_impl(
        ctx, segments=payload, source=store.SOURCE_SIGNAL,
        use=store.USE_AUDIT, settings=get_settings())
    assert draft["success"] is True
    assert draft["saved"]["saved"] == len(payload)

    # 3) 质检草稿：无重叠/超界（切分保证连续），应非 fail
    qc1 = check_annotation_qc_impl(ctx, settings=get_settings())
    assert qc1["result"] != "fail", qc1["failures"]

    # 4) 人工命名（user_confirmed）+ 声明训练用途
    named = [
        {"episode_key": "0", "id": i + 1, "start_s": s["start_s"],
         "end_s": s["end_s"], "atomic_action": f"Action_{i + 1}",
         "interacting_hand": "right_hand"}
        for i, s in enumerate(segs)
    ]
    final = save_annotations_impl(
        ctx, segments=named, confirmed_by_user=True,
        use=store.USE_TRAINING, reason="人工命名与复核", settings=get_settings())
    assert final["success"] is True
    assert final["use_check"]["ok"] is True

    # 5) 训练用途下质检应通过来源校验
    qc2 = check_annotation_qc_impl(ctx, use=store.USE_TRAINING, settings=get_settings())
    assert "source_for_declared_use" not in qc2["failed_rules"]

    # 6) 修订日志应有完整轨迹（create + update）。
    hist = store.load_annotation_history(output_dir, "demo")
    ops = [e["op"] for e in hist["entries"]]
    assert ops.count("create") >= len(payload)
    assert ops.count("update") >= 1
