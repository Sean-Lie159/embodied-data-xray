@echo off
rem 局域网启动脚本（实例模式）：同事经 http://<本机内网IP>:8501 访问。
rem 前置条件：本机 .env 已完成模型配置（含 INSTANCE_MODE=1 时访问者不可见配置）。
rem 首次使用请放行 Windows 防火墙 8501 端口（见 README「实例模式」章节）。

cd /d "%~dp0.."
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
