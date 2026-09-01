# embodied-data-xray

**X-ray your robot datasets** — structural profiling & analysis agent for embodied-AI collection data.

一个**本地运行、面向具身智能（Embodied AI）领域**的数据分析 Agent：像 X 光一样透视机器人采集数据的**结构**——文件构成、时间戳伴随表、骨骼位姿块、元数据角色——再通过自然语言对话完成统计分析、可视化与解读报告。

> 为什么叫 x-ray？它看的是数据的**结构骨架**，而不是内容本身。所有已支持格式的结构骨架都固化为回归测试（`tests/fixtures/skeletons/`），保证"新格式来了不再崩"。

## 它能做什么

- **目录透视**：递归普查 + 语义角色识别（时间戳伴随表 / 元数据文件 / 骨骼位姿块 / 索引列），生成能力标签与推测类型
- **确定性分析**：数据概况、任务级统计、骨骼位姿范围（按数据集声明的 N×M 块分解）、时间同步与漂移检查、传感器合理性检查——全部由 pandas 确定性计算，LLM 只负责解读
- **对话式操作**：自然语言驱动（openai-agents SDK），CLI 与 Streamlit 双入口；工具调用轨迹、token 消耗全程可见
- **诚实降级**：识别不了就如实标注 unknown 并附证据，绝不硬猜；数值计算不交给 LLM

## 快速开始

```bash
# 1) Python 3.12 + 依赖（版本已按实测环境锁定）
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2) 配置模型：复制 .env.example 为 .env，填入密钥与端点
copy .env.example .env

# 3) 没有数据？用内置结构骨架生成一个示例目录（零数据体验全流程）
python -c "from tests.fixtures.make_skeleton import build_skeleton; \
import pathlib; \
print(build_skeleton(pathlib.Path('data'), 'tobii_ego'))"

# 4) 启动
python main.py                      # CLI
streamlit run streamlit_app.py      # 或 Streamlit 图形界面
```

在对话中输入 `加载 data/tobii_ego`，即可走通「加载 → 质检 → 统计 → 绘图 → 报告」全链路。
也可换 `nuscenes_reorg` 或 `lerobot_v2` 骨架，体验不同格式的识别。

## 支持的数据布局清单

当前**已验证**的数据布局（每种布局的结构骨架已固化为回归测试）：

| 布局 | 特征 | 状态 |
|---|---|---|
| 单文件 | CSV / JSON / Parquet / HDF5 单表 | ✅ 支持 |
| 多流采集目录 | CSV 传感器流 + 每路视频的时间戳伴随表 + 相机标定 JSON + 媒体文件 | ✅ 支持 |
| 帧索引 + 视频目录 | 每路视频的帧索引表（`.index.parquet`）+ 多分辨率/预览视频变体 + 同名 sidecar JSON | ✅ 支持 |
| LeRobot v2 | `meta/info.json` + `data/chunk-*` + `videos/chunk-*` | ✅ 支持 |

**清单外格式**：agent 仍会尝试加载（普查/嗅探尽力而为），但识别可能受限——未知语义
角色会被如实标注为 `unknown`，而不是硬猜。新布局建议先把结构骨架加入测试套件，
再进入分析流程。

## 数据识别如何工作

识别规则绑定**语义角色**，不绑定物理形态（后缀/目录名/固定列名）：

- **时间戳伴随表**：与媒体同 stem 的表格 + 帧级时间戳列（`frame_*`/`exposure_*`/`pts`），行数与视频帧数相符——无论它是 `*_metainfo.csv` 还是 `camera-x.index.parquet`
- **元数据文件**：小尺寸 JSON + 配置型键（fps/features 等）→ 不进流清单、不参与对齐
- **骨骼位姿块**：数据集声明 `names=xxx_NxM` 且 N×M==shape → 按块分解统计位姿范围与四元数范数
- **时间戳列**：词表命中优先，未命中回退内容指纹（单调递增 + 量级符合时间单位）；单位推断带自我纠正（采样率超物理区间自动换单位重算）

### 架构文档

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 分层设计、数据识别架构（语义角色优先 / 支持清单 / LLM 假设边界）
- [docs/行为测试.md](docs/行为测试.md) — 行为回归用例（含数据识别三层防御的终身回归）
- [docs/技术债.md](docs/技术债.md) — 已知限制与决策记录

## ffmpeg / ffprobe（可选）

加载含视频的目录时，会用 `ffprobe` 读取视频元数据（帧率/分辨率/帧数/时长/编码）。
未安装不影响其余功能，视频元数据会以结构化降级提示呈现。

Windows 推荐 `winget install Gyan.FFmpeg`；安装后用 `where ffprobe` 确认在 PATH 中
（winget 的实际安装目录在 `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_...`，
与常见的 `C:\ffmpeg\bin` 约定不同）。macOS / Linux：`brew install ffmpeg` 或
`apt install ffmpeg`。验证：`ffprobe -version`。

## 测试

```bash
python -m pytest
```

285 项测试（含多种格式骨架的结构回归）；真实模型连通性测试默认跳过，需设置
`RUN_LLM_TESTS=1` 后运行。

## 许可证

[MIT](LICENSE) © 2026 Sean-Lie159
