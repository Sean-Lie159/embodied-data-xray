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

### Streamlit 图形界面

```bash
streamlit run streamlit_app.py
```

浏览器打开后即可使用图形界面：
- **左侧对话区**：输入自然语言指令，与 Agent 对话；每轮回复下方可展开"查看执行过程"查看工具调用轨迹。
- **右侧展示区**：三个标签页——「图表」（展示生成的图片）、「Findings/报告」（分析结果列表 + 报告下载按钮）、「数据概况」（当前数据集的能力标签与流清单）。
- 会话状态（已加载数据、对话历史）在刷新前持续保留。

## ffmpeg / ffprobe 说明（可选）

加载**含视频的原始采集目录**时，Agent 会用 `ffprobe` 读取视频元数据（帧率、分辨率、帧数、时长、编码）。

- `ffprobe` 属于 [ffmpeg](https://ffmpeg.org/)，需要单独安装。
- **未安装 ffprobe 时不影响整体功能**：目录加载会跳过视频元数据嗅探，其余（文件普查、表格列名嗅探、标定检测）正常执行，并返回结构化降级提示。

### Windows 安装方式

**方式一：winget（推荐）**

```bash
winget install Gyan.FFmpeg
```

winget 默认不会把 ffmpeg 装到 `C:\ffmpeg\bin`，而是装在类似
`C:\Users\<你>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-<版本>-full_build\bin`
的目录。**请务必用下面命令确认实际路径，再把它加入 PATH**：

```powershell
# 确认 ffprobe 是否在 PATH 中、以及实际安装路径
where ffprobe
ffprobe -version
```

- 若 `where ffprobe` 有输出：说明已在 PATH 中，直接重新打开终端加载数据集即可。
- 若 `where ffprobe` 无输出：找到上面 winget 安装目录里的 `bin` 文件夹，
  把该完整路径加入系统 `PATH`（系统属性 → 环境变量 → Path），或在本会话临时：

  ```powershell
  $env:PATH += ";C:\Users\<你>\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-<版本>-full_build\bin"
  ```

  然后重开终端，用 `ffprobe -version` 验证有输出。

> 注意：本项目检测 ffprobe 时会**真正尝试调用一次** `ffprobe -version`，
> 只要 PATH 里能找到可执行文件即视为可用，不依赖某个固定路径约定。

**方式二：手动安装**

1. 到 [ffmpeg 官网下载页](https://ffmpeg.org/download.html) 选择 Windows 版本（如 gyan.dev 或 BtbN 构建）。
2. 解压到一个固定目录（如 `C:\ffmpeg`）。
3. 把解压后目录里的 `bin` 子目录（如 `C:\ffmpeg\bin`）加入系统 `PATH`。
4. 重启终端，验证 `ffprobe -version` 有输出。

### macOS / Linux

```bash
# macOS（Homebrew）
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

验证安装：`ffprobe -version`（有版本信息输出即表示可用）。

## 支持的数据布局清单

当前**已验证**的数据布局（每种布局的结构骨架已固化为回归测试，见 `tests/fixtures/skeletons/`）：

| 布局 | 特征 | 状态 |
|---|---|---|
| 单文件 | CSV / JSON / Parquet / HDF5 单表 | ✅ 支持 |
| Tobii 式 Ego 目录 | CSV 传感器流 + `*_metainfo.csv` + JSON 相机标定 + MP4/M4A/JPG | ✅ 支持 |
| worldcode / nuScenes 重组目录 | nuScenes schema JSON 表族 + `.index.parquet` + 多分辨率视频版本组 + 同名 json↔mp4 | ✅ 支持 |
| LeRobot v2 | `meta/info.json` + `data/chunk-*` + `videos/chunk-*` | ✅ 支持 |

**清单外格式**：agent 会尝试加载（普查/嗅探尽力而为），但识别可能受限——未知语义角色
会被如实标注为 unknown，而不是硬猜。请优先使用上方已验证布局；新布局建议先提取结构
骨架进测试套件（见行为测试.md 的元规则），再进入分析流程。

## 目录数据集嗅探

`load_dataset` 支持传入**目录**，会执行：

- 递归文件普查（扩展名分布、目录结构）；
- 表格类（csv/json/parquet）读列名，推断 IMU（6/9 轴）、位姿、状态/动作；
- 数据识别按**语义角色**匹配（时间戳伴随表、元数据文件、索引列），不绑定具体后缀或
  固定列名；词表未命中时回退内容指纹（数值列单调递增 + 差分均匀 + 量级符合时间单位）；
- json/yaml 标定文件检测（intrinsic/extrinsic/K/D 等键）；小尺寸配置型 JSON
  （fps/features 等）归数据集元数据角色，不进流清单、不参与对齐；
- 视频类 ffprobe 元数据（可用时）；
- 生成能力标签（`has_imu` / `has_video_streams` / `has_calibration` / `has_actions` 等）
  与推测类型（含置信度），写入 `RunContext.meta`。

## 测试

```bash
python -m pytest
```

真实模型连通性测试默认跳过，需设置 `RUN_LLM_TESTS=1` 后运行。

## 许可证

MIT（待定）。
