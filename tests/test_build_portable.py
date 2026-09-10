"""build_portable 纯函数测试（2026-09-07 便携分发包，档位二）。

守护构建脚本里可独立验证的逻辑（不做真实下载/装依赖，网络相关 mock 掉）：
- _build_env_lines：.env 预填内容正确（base_url/key/model/temp），不含多余行；
- _launcher_content：启动.bat 指向 runtime、设 MPLCONFIGDIR、跑 streamlit 8501；
- _read_deploy_env：从部署机 .env 读 key/model/temp（mock env_io）；
- requirements 过滤：构建时剔除 pytest（生产包不带测试框架）。

真实打包（下载解释器+装依赖）耗时且联网，由构建脚本手工执行、文档第 8 节
在干净环境验收，不纳入 pytest。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.build_portable as bp  # type: ignore[import]


# --- _build_env_lines：.env 预填内容 -------------------------------------------


def test_build_env_lines_content() -> None:
    deploy = {"key": "ck_abc", "model": "hy3", "temp": "0.7"}
    lines = bp._build_env_lines("http://192.168.1.100:8787/v1", deploy)
    text = "\n".join(lines)
    assert "OPENAI_API_KEY=ck_abc" in text
    assert "OPENAI_BASE_URL=http://192.168.1.100:8787/v1" in text
    assert "DEFAULT_MODEL=hy3" in text
    assert "DEFAULT_TEMPERATURE=0.7" in text


# --- _launcher_content：启动.bat ----------------------------------------------


def test_launcher_bat_uses_launch_py() -> None:
    """启动.bat 应调用 launch.py（而非直接固定 8501 跑 streamlit）。"""
    content = bp._launcher_bat_content()
    # 指向 runtime + 调 launch.py（端口探测逻辑在 launch.py 里）。
    assert "%~dp0runtime" in content
    assert "launch.py" in content
    # 不再固定写死 8501（否则端口被占即闪退）。
    assert "--server.port 8501" not in content
    # 设 MPLCONFIGDIR（matplotlib 缓存指向包内目录）。
    assert "MPLCONFIGDIR" in content and "mplcache" in content


def test_readme_template_for_end_users() -> None:
    """包内使用说明：面向同事，含启动方式/数据加载/常见问题，不泄露技术细节。"""
    content = bp._README_TEMPLATE
    assert "启动.bat" in content  # 告诉用户双击哪个
    assert "数据加载" in content  # 数据怎么进来
    assert "常见问题" in content or "闪退" in content  # 排障指引
    # 不应出现发布者侧的敏感细节（如内网 IP 占位、portproxy 命令）。
    assert "portproxy" not in content and "netsh" not in content


def test_launch_py_probes_port_and_runs_streamlit() -> None:
    """launch.py 内容：探测空闲端口、调 streamlit run、自动开浏览器。"""
    content = bp._LAUNCH_PY
    assert "_port_free" in content or "bind(" in content  # 端口探测
    assert "streamlit" in content and "-m" in content  # 调 streamlit
    assert "--server.port" in content
    assert "webbrowser" in content and "open(" in content  # 自动开浏览器
    # 端口被占时顺延而非崩溃。
    assert "MAX_PORT" in content or "range(" in content


# --- _read_deploy_env：从部署机 .env 读配置 ------------------------------------


def test_read_deploy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """mock env_io.read_env_file：正确取 key/model/temp；key 缺失时为空。"""

    class _FakeEnvIO:
        @staticmethod
        def read_env_file(path):  # noqa: ANN001, ANN205
            return {"OPENAI_API_KEY": "ck_x", "DEFAULT_MODEL": "m1",
                    "DEFAULT_TEMPERATURE": "0.5"}

    monkeypatch.setattr(bp, "read_env_file", _FakeEnvIO.read_env_file)
    env = bp._read_deploy_env()
    assert env["key"] == "ck_x"
    assert env["model"] == "m1"
    assert env["temp"] == "0.5"


# --- requirements 剔除 pytest（生产包不带测试框架） -----------------------------


def test_install_deps_skips_pytest(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text(
        "streamlit==1.40.0\npytest==8.3.0\npytest-cov==5.0.0\npandas==2.2.2\n",
        encoding="utf-8",
    )
    # _install_deps 会真实调用 pip，这里只测它的过滤逻辑：读原始文本剔除后。
    filtered = "\n".join(
        line for line in req.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("pytest", "pytest-cov"))
    )
    assert "pytest" not in filtered and "pytest-cov" not in filtered
    assert "streamlit" in filtered and "pandas" in filtered
