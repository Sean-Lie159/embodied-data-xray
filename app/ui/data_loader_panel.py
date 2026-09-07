"""数据加载侧栏面板：路径输入（主）+ 单文件上传（辅）+ 示例数据集（可选）。

与 Agent 的状态一致性（设计文档 3.3）：加载直接调 load_dataset_impl 写入
service.context（所有工具共享），Agent 下一轮工具调用自然读到新数据集；
同时向对话流追加一条 assistant 说明消息（决策 3：进对话流，保可溯源），
成功含数据集名与概况一句话，失败如实转达工具的 user_message（纪律 4）。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config.settings import get_settings
from app.tools.load_dataset import load_dataset_impl
from app.ui.upload_store import ALLOWED_SUFFIXES, save_upload

# 单文件上传大小上限（MB）：仅辅助通道，目录型数据集请走路径输入。
_MAX_UPLOAD_MB = 200


def _append_message(messages: list[dict], content: str) -> None:
    """向对话流追加一条 assistant 说明消息（不触发 agent，无工具轨迹）。"""
    messages.append({"role": "assistant", "content": content})


def _load_path(service, messages: list[dict], path: str) -> bool:
    """按路径加载数据集并写对话说明；返回是否成功。"""
    try:
        result = load_dataset_impl(service.context, path)
    except Exception as exc:  # noqa: BLE001  # 工具层已尽量结构化，这里兜底转达
        _append_message(messages, f"数据加载失败：{exc}")
        return False
    if result.get("success"):
        name = Path(path).name
        n_files = result.get("n_streams")
        _append_message(
            messages,
            f"已通过侧栏『数据加载』面板加载数据集 **{name}**"
            + (f"（{n_files} 条流）" if n_files else "")
            + "。你可以直接提问，例如：这个数据集概况如何？成功率多少？",
        )
        return True
    _append_message(
        messages,
        f"数据加载失败（{Path(path).name}）：{result.get('user_message', '未知原因')}",
    )
    return False


def render_data_loader(service, messages: list[dict]) -> None:
    """渲染侧栏"数据加载"折叠面板（调用方置于 st.sidebar 内）。"""
    with st.expander("数据加载"):
        # 主路径：粘贴绝对路径（文件或目录均可）。
        path = st.text_input(
            "数据集绝对路径",
            key="dataset_path_input",
            placeholder=r"如 C:\Users\me\data\robot_session_01",
            help="支持单个数据文件（csv/parquet/json/jsonl）或数据集目录。",
        )
        if st.button("加载路径数据集", key="btn_load_path", type="primary"):
            if not path.strip():
                st.warning("请先粘贴数据集路径。")
            else:
                with st.spinner("加载中……"):
                    ok = _load_path(service, messages, path.strip())
                if ok:
                    st.success("加载成功，详见对话区说明。")
                    st.rerun()
                else:
                    st.error("加载失败，详见对话区说明。")

        st.divider()
        # 辅路径：单文件上传（目录型数据集不适用）。
        st.caption(
            f"单文件上传（≤{_MAX_UPLOAD_MB}MB，{'/'.join(sorted(ALLOWED_SUFFIXES))}）；"
            "多文件数据集请用上方路径输入。"
        )
        upload = st.file_uploader(
            "上传单个数据文件",
            type=[s.lstrip(".") for s in sorted(ALLOWED_SUFFIXES)],
            key="dataset_uploader",
        )
        if upload is not None and st.button("保存并加载上传文件", key="btn_load_upload"):
            if upload.size > _MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"文件超过 {_MAX_UPLOAD_MB}MB 上限，请改用路径输入。")
            else:
                try:
                    uploads_dir = Path(get_settings().output_path()) / "uploads"
                    saved = save_upload(
                        upload.getvalue(), upload.name, uploads_dir
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    return
                with st.spinner("加载中……"):
                    ok = _load_path(service, messages, str(saved))
                if ok:
                    st.success(f"已保存并加载：{saved.name}")
                    st.rerun()
                else:
                    st.error("加载失败，详见对话区说明。")

        # 示例数据集按钮由 Commit 4（app/ui/sample_dataset.py）加入本面板底部。
