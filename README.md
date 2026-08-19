# 具身智能数据分析 Agent

一个**本地运行、面向具身智能（Embodied AI）领域**的数据分析 Agent。通过自然语言对话完成：加载机器人数据集 → 统计分析 → 可视化 → 生成解读报告。

支持的数据格式：LeRobot 数据集、HDF5、Parquet/CSV，以及**原始采集数据目录**（自动文件普查 + 能力嗅探）。

## 技术栈

Python 3.12 / openai-agents / pandas / numpy / pyarrow / h5py / matplotlib / plotly / Streamlit / pytest

## 安装

1. 准备 Python 3.12，创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

2. 配置模型：复制 `.env.example` 为 `.env`，填入你的模型服务商密钥与端点。

## 配置

在 `.env` 中配置（参考 `.env.example`）：

| 配置项 | 说明 |
|---|---|
| `OPENAI_API_KEY` | 模型 API 密钥（DeepSeek / Kimi / OpenAI 任一兼容服务） |
| `OPENAI_BASE_URL` | 模型服务端点（DeepSeek `https://api.deepseek.com`，Kimi `https://api.moonshot.cn/v1`） |
| `DEFAULT_MODEL` | 默认模型名（如 `deepseek-v4-flash`） |
| `DEFAULT_TEMPERATURE` | 默认温度（0.0–2.0） |

## 使用

### 命令行对话

```bash
python main.py
```

在终端输入自然语言指令，Agent 会自主调用工具（加载数据、生成概况、统计分析等）。输入 `exit` / `quit` 退出。

## ffmpeg / ffprobe 说明（可选）

加载**含视频的原始采集目录**时，Agent 会用 `ffprobe` 读取视频元数据（帧率、分辨率、帧数、时长、编码）。

- `ffprobe` 属于 [ffmpeg](https://ffmpeg.org/)，需要单独安装。
- **未安装 ffprobe 时不影响整体功能**：目录加载会跳过视频元数据嗅探，其余（文件普查、表格列名嗅探、标定检测）正常执行，并返回结构化降级提示。

### Windows 安装方式

**方式一：winget（推荐）**

```bash
winget install Gyan.FFmpeg
```

安装后重新打开终端，或在当前会话中把 `C:\ffmpeg\bin` 加入 `PATH`。

**方式二：手动安装**

1. 到 [ffmpeg 官网下载页](https://ffmpeg.org/download.html) 选择 Windows 版本（如 gyan.dev 或 BtbN 构建）。
2. 解压到一个固定目录（如 `C:\ffmpeg`）。
3. 把 `C:\ffmpeg\bin` 加入系统 `PATH`（系统属性 → 环境变量 → Path）。
4. 重启终端，验证 `ffprobe -version` 有输出。

### macOS / Linux

```bash
# macOS（Homebrew）
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

验证安装：`ffprobe -version`。

## 目录数据集嗅探

`load_dataset` 支持传入**目录**，会执行：

- 递归文件普查（扩展名分布、目录结构）；
- 表格类（csv/json/parquet）读列名，推断 IMU（6/9 轴）、位姿、状态/动作；
- json/yaml 标定文件检测（intrinsic/extrinsic/K/D 等键）；
- 视频类 ffprobe 元数据（可用时）；
- 生成能力标签（`has_imu` / `has_video_streams` / `has_calibration` / `has_actions` 等）与推测类型（含置信度），写入 `RunContext.meta`。

## 测试

```bash
python -m pytest
```

真实模型连通性测试默认跳过，需设置 `RUN_LLM_TESTS=1` 后运行。

## 许可证

MIT（待定）。
