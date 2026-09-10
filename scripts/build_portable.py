"""build_portable.py —— 构建"解压双击即用"的便携分发包（档位二）。

设计文档：docs/便携分发包设计.md。

产物（构建后输出到 ``dist_portable/``，该目录已被 .gitignore 排除）::

    EmbodiedDataAgent_<日期>.zip/
    ├── runtime/              # 内嵌 Python 3.12 + 全部依赖（占大头，几乎不变）
    ├── app/ scripts/ streamlit_app.py main.py requirements.txt
    ├── .env                  # 预填：代理地址 + key + hy3（构建时从部署机 .env 注入）
    ├── .streamlit/credentials.toml   # 免首次 email 问卷
    └── 启动.bat              # 双击入口

代码与环境分离：runtime/ 依赖锁版本，代码更新只需覆盖包根小文件。

用法（部署机运行）::

    python scripts/build_portable.py
        --base-url http://<内网IP>:8787/v1   # 缺省时尝试自动检测本机内网 IPv4
        --model hy3                          # 缺省从部署机 .env 读 DEFAULT_MODEL
        --temp 0.7                           # 缺省从部署机 .env 读 DEFAULT_TEMPERATURE

key 始终从部署机 .env 的 OPENAI_API_KEY 读取注入（脚本不硬编码、key 不入库）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# 使脚本可被 `python scripts/build_portable.py` 直接运行（sys.path[0]=scripts 时
# 也能 import 项目内的 app 包，与 make_sample_dataset.py 同款处理）。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.env_io import read_env_file  # type: ignore[import]
# python-build-standalone 的发布信息（install_only 版，Windows x64）。
_PYBS_INDEX = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"

# 打包进分发包的代码路径（不含 tests/outputs/data/reference/.git/.env）。
_CODE_ITEMS = ["app", "scripts", "streamlit_app.py", "main.py", "requirements.txt"]
# 分发包内预建 matplotlib 缓存目录（启动.bat 设 MPLCONFIGDIR）。
_MPL_CACHE_REL = "runtime/mplcache"


def _detect_lan_ipv4() -> str | None:
    """探测本机局域网 IPv4（尽力而为）：连一个外网 UDP 不回包，取其源地址。

    Returns:
        局域网 IPv4（如 192.168.1.100）；探测失败返回 None。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def _download_python_install_only(dest: Path) -> None:
    """下载并解包 python-build-standalone 的 Windows x64 install_only 版到 dest。

    Args:
        dest: 目标目录（作为 runtime/ 的上层，解包后含 python.exe）。

    Raises:
        RuntimeError: 无法从 GitHub API 定位到匹配的发布资产。
    """
    with urllib.request.urlopen(_PYBS_INDEX, timeout=30) as resp:  # noqa: S310
        release = json.load(resp)
    asset_url = None
    for asset in release.get("assets", []):
        name = asset["name"]
        # 例：cpython-3.12.8+20250115-x86_64-pc-windows-shared_install_only.tar.gz
        if name.startswith("cpython-3.12") and "x86_64-pc-windows" in name \
                and name.endswith("install_only.tar.gz"):
            asset_url = asset["browser_download_url"]
            break
    if not asset_url:
        raise RuntimeError("未能从 python-build-standalone 定位 3.12 Windows x64 install_only 包")

    print(f"[1/4] 下载内嵌 Python: {Path(asset_url).name}")
    with tempfile.TemporaryDirectory() as td:
        arc = Path(td) / "py.tar.gz"
        urllib.request.urlretrieve(asset_url, arc)  # noqa: S310
        print("[1/4] 解包到 runtime/ ...")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(arc), str(dest), format="gztar")
        # python-build-standalone 的 tarball 顶层结构随版本而异：
        # 新版（2026 起）用 python/ 子目录包裹（python/python.exe），
        # 旧版是扁平（顶层直接 python.exe）。统一把实际可执行文件所在层
        # 上移到 dest/ 根，保证 runtime/python.exe 恒存在。
        nested = dest / "python"
        if nested.is_dir() and (nested / "python.exe").exists():
            for item in list(nested.iterdir()):
                shutil.move(str(item), str(dest / item.name))
            nested.rmdir()


def _install_deps(python_exe: Path, requirements: Path) -> None:
    """用内嵌解释器安装依赖到其自身 site-packages。

    Args:
        python_exe: runtime/python.exe。
        requirements: requirements.txt 路径。
    """
    print("[2/4] 安装依赖（跳过 pytest，生产包不带测试框架）...")
    req = requirements.read_text(encoding="utf-8")
    req = "\n".join(
        line for line in req.splitlines()
        if not re.match(r"^\s*(pytest|pytest-cov)\s*[<=>]?", line)
    )
    tmp_req = requirements.parent / "_portable_requirements.txt"
    tmp_req.write_text(req, encoding="utf-8")
    try:
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "--no-input",
             "-r", str(tmp_req)],
            check=True,
        )
    finally:
        tmp_req.unlink(missing_ok=True)


def _prewarm_matplotlib(python_exe: Path, cache_dir: Path) -> None:
    """预热 matplotlib 字体缓存，避免同事首次启动在只读/受限目录报错。

    Args:
        python_exe: runtime/python.exe。
        cache_dir: 包内 mplcache 目录（启动.bat 通过 MPLCONFIGDIR 指向它）。
    """
    print("[3/4] 预热 matplotlib 字体缓存 ...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(cache_dir)
    subprocess.run(
        [str(python_exe), "-c", "import matplotlib.pyplot"],
        env=env, check=True,
    )


def _read_deploy_env() -> dict[str, str]:
    """读取部署机 .env 的关键配置（key 由此注入分发包，脚本不硬编码）。"""
    env = read_env_file(REPO_ROOT / ".env")
    return {
        "key": env.get("OPENAI_API_KEY", ""),
        "model": env.get("DEFAULT_MODEL", "hy3"),
        "temp": env.get("DEFAULT_TEMPERATURE", "0.2"),
    }


def _build_env_lines(base_url: str, deploy: dict[str, str]) -> list[str]:
    """拼分发包 .env 内容（不含注释以外的多余行，避免含内网细节之外信息）。"""
    return [
        "# 便携分发包预置配置（由 scripts/build_portable.py 生成）",
        f"OPENAI_API_KEY={deploy['key']}",
        f"OPENAI_BASE_URL={base_url}",
        f"DEFAULT_MODEL={deploy['model']}",
        f"DEFAULT_TEMPERATURE={deploy['temp']}",
    ]


def _launcher_bat_content() -> str:
    """启动.bat 内容：指向 runtime，调用 launch.py 找空闲端口再跑 streamlit。

    bat 用 ANSI/GBK 保存（Windows 中文代码页），故这里不写非 ASCII 注释。
    """
    return (
        "@echo off\r\n"
        "cd /d %~dp0\r\n"
        'set "PATH=%~dp0runtime;%~dp0runtime\\Scripts;%PATH%"\r\n'
        'set "MPLCONFIGDIR=%~dp0runtime\\mplcache"\r\n'
        "set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false\r\n"
        '"%~dp0runtime\\python.exe" "%~dp0launch.py"\r\n'
        "if errorlevel 1 pause\r\n"
    )


_README_TEMPLATE = """\
# 具身智能数据分析 Agent（便携版）

## 怎么用

1. 把本文件夹**解压到本地磁盘**（如 D 盘），不要直接在压缩包里或网络共享盘里运行；
2. 双击 **启动.bat**；
3. 稍等几秒，浏览器会自动打开分析界面（地址形如 http://localhost:8501）；
4. 在左侧栏「数据加载」里：
   - **粘贴数据集路径**（推荐）：把数据集文件夹的完整路径粘进去；
   - 或**上传单个数据文件**（csv / parquet / json / jsonl）；
   - 想先看看效果：点「加载示例数据集」，然后在对话框里提问试试；
5. 在底部对话框用自然语言提问，例如：
   - 这个数据集概况如何？
   - 时间同步检查一下，缺口发生在哪？
   - 成功率多少，哪些 episode 离群？

## 环境要求

- Windows 10/11 64 位；**无需安装 Python 或任何依赖**（已内置）；
- 首次启动若提示缺少 DLL，安装一次
  "Microsoft Visual C++ 2015-2022 Redistributable (x64)" 即可。

## 常见问题

- **双击没反应 / 闪退**：多等几秒；若仍失败，右键本文件夹在
  终端里运行 `启动.bat` 查看报错并反馈给发布者；
- **界面打不开**：检查浏览器地址栏端口号（若 8501 被占用，程序会自动改用
  8502 等端口，以命令行窗口里打印的地址为准）；
- **提问报错、模型不可用**：本版本的模型接口由发布者统一提供，需要发布者的
  机器与网络在线；请联系发布者。

## 注意

- 模型密钥已内置在本包 `.env` 中，请勿把本包转发给无关人员；
- 数据只在你本机处理，不会上传。
"""


_LAUNCH_PY = r'''"""launch.py —— 分发包启动器：探测空闲端口后启动 streamlit，自动开浏览器。

固定 8501 端口在用户机器上可能已被占用（真实事故：双击闪退、无浏览器）。
故先探测 8501，被占则顺延找第一个空闲端口，再以该端口运行 streamlit 并
自动打开浏览器；进程退出前暂停窗口以便用户看到报错。
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = 8501
MAX_PORT = 8600


def _port_free(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> None:
    port = DEFAULT_PORT
    for p in range(DEFAULT_PORT, MAX_PORT):
        if _port_free(p):
            port = p
            break
    if port != DEFAULT_PORT:
        print(f"端口 {DEFAULT_PORT} 被占用，改用 {port}。")
    python = os.path.join(BASE, "runtime", "python.exe")
    cmd = [python, "-m", "streamlit", "run", "streamlit_app.py",
           "--server.port", str(port), "--server.headless", "false"]
    url = f"http://localhost:{port}"
    # 稍候浏览器打开（等服务就绪）。
    proc = subprocess.Popen(cmd, cwd=BASE)
    time.sleep(4)
    webbrowser.open(url)
    proc.wait()
    print("\n服务已停止。按任意键关闭窗口。")
    input()


if __name__ == "__main__":
    main()
'''


def _make_zip(build_dir: Path, out_dir: Path) -> Path:
    """把 build_dir 打成 zip（顶层目录名 = 包根 build_dir.name，便于解压即用）。

    Args:
        build_dir: 已构建好的分发包目录（名如 EmbodiedDataAgent_20260909）。
        out_dir: 产物输出目录（dist_portable）。

    Returns:
        生成的 zip 路径（build_dir.name.zip，置于 out_dir）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{build_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in build_dir.rglob("*"):
            if file.is_file():
                zf.write(file, str(file.relative_to(build_dir.parent)))
    return zip_path


def build(base_url: str, model: str | None, temp: str | None) -> Path:
    """执行完整构建流程，返回产出 zip 路径。"""
    deploy = _read_deploy_env()
    if not deploy["key"]:
        raise SystemExit("部署机 .env 缺 OPENAI_API_KEY，无法构建分发包（key 需预置）。")
    model = model or deploy["model"]
    temp = temp or deploy["temp"]

    date_stamp = __import__("time").strftime("%Y%m%d")
    build_dir = REPO_ROOT / "dist_portable" / f"EmbodiedDataAgent_{date_stamp}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    (build_dir / "runtime").mkdir(parents=True)

    _download_python_install_only(build_dir / "runtime")

    python_exe = build_dir / "runtime" / "python.exe"
    if not python_exe.exists():
        raise RuntimeError("内嵌 Python 解包后未找到 python.exe")
    _install_deps(python_exe, REPO_ROOT / "requirements.txt")

    # 复制代码（不含 tests/outputs/data/reference/.git）。
    print("[4/4] 复制项目代码 ...")
    for item in _CODE_ITEMS:
        src = REPO_ROOT / item
        dst = build_dir / item
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.exists():
            shutil.copy2(src, dst)
    # 预建 matplotlib 缓存目录（_prewarm_matplotlib 已建，这里确保存在）。
    (build_dir / _MPL_CACHE_REL).mkdir(parents=True, exist_ok=True)

    # .env
    (build_dir / ".env").write_text(
        "\n".join(_build_env_lines(base_url, deploy)) + "\n", encoding="utf-8"
    )
    # .streamlit/credentials.toml 防首次 email 问卷
    (build_dir / ".streamlit").mkdir(parents=True, exist_ok=True)
    (build_dir / ".streamlit" / "credentials.toml").write_text(
        '[general]\nemail = ""\n', encoding="utf-8"
    )
    # 启动器（launch.py，UTF-8）：探测空闲端口后跑 streamlit（修复 8501 被占
    # 时双击闪退）。随分发包分发，不在仓库 scripts/ 下重复维护，故构建时写入。
    (build_dir / "launch.py").write_text(_LAUNCH_PY, encoding="utf-8")
    # 面向终端用户（同事）的简短说明，随包分发（发布者视角的详细文档见
    # docs/便携分发包设计.md，不进包）。
    (build_dir / "使用说明.md").write_text(_README_TEMPLATE, encoding="utf-8")
    # 启动.bat（GBK/ANSI 编码，与 Windows 中文代码页一致）：调 launch.py。
    (build_dir / "启动.bat").write_bytes(
        _launcher_bat_content().encode("gbk", errors="replace")
    )

    zip_path = _make_zip(build_dir, REPO_ROOT / "dist_portable")
    print(f"\n构建完成：{zip_path}")
    print(f"包体约 {zip_path.stat().st_size / 1024 / 1024:.0f}MB；"
          f"同事解压后双击「启动.bat」即可。")
    return zip_path


def main() -> None:
    """解析参数并执行构建。"""
    parser = argparse.ArgumentParser(description="构建便携分发包（档位二）")
    parser.add_argument(
        "--base-url", default=None,
        help="代理地址，如 http://192.168.1.100:8787/v1；缺省自动检测本机内网 IP",
    )
    parser.add_argument("--model", default=None, help="模型名（缺省读部署机 .env）")
    parser.add_argument("--temp", default=None, help="温度（缺省读部署机 .env）")
    args = parser.parse_args()

    base_url = args.base_url
    if not base_url:
        ip = _detect_lan_ipv4()
        if not ip:
            raise SystemExit(
                "无法自动检测内网 IP。请用 --base-url http://<部署机内网IP>:8787/v1 显式指定。"
            )
        base_url = f"http://{ip}:8787/v1"
        print(f"自动检测内网 IP：{ip}，预填 {base_url}")
    build(base_url, args.model, args.temp)


if __name__ == "__main__":
    main()
