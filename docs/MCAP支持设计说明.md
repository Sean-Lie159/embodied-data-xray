# MCAP 支持设计说明（JSON 编码，2026-09-10）

> 本文遵循 AGENTS.md 第 4 条"先文档后代码"，动工前已经过 owner 确认。
> 背景调研与选型依据见 `research/research_report_mcap_unpack_assessment.md`（不入库）。
> 原型验证产物：`app/tools/mcap_reader.py` + `tests/test_mcap_reader.py`（19 用例全绿）。

---

## 1. 目标与范围

**目标**：让 Agent 支持 `.mcap` 文件的分析（加载 → 概况 → 质检 → 统计 → 绘图 → 报告），
并把指定 topic 解包落盘（csv / json / jsonl 自选）。

**本期范围（第一期）**：**仅 JSON 编码的 MCAP**（`channel.message_encoding == "json"`）。
此类消息 `message.data` 直接是 JSON 字节流，`json.loads` 即得 dict，无需 schema 反序列化——
风险最低，且覆盖项目已有的真实数据形态（`wujiGlove_data` 即 MCAP 导出的 JSON 信封流）。

**明确不做（第二期）**：

- ROS2 CDR 编码（`message_encoding == "cdr"`）的解码——需 `rosbags` 反序列化，
  且会遇到项目从未处理的 `sensor_msgs/Image`（原始 bytes，非表格）、点云等类型；
- 图像 / 点云 / 音频消息的二进制落盘；
- 物理互相关对齐（项目既有 v2 范畴，与 MCAP 无关）。

---

## 2. 容器结构与关键事实（决定实现难度）

MCAP 物理结构：`<Magic><Header><Data><Summary?><Footer><Magic>`，record 统一为
`<opcode(1B)><length(uint64)><content>`，小端序。核心 record：

| record | 关键字段 | 对本项目的意义 |
|---|---|---|
| Schema | `id` / `name` / `encoding` / `data` | 判据：`jsonschema` 或空 = 自描述 |
| Channel | `id` / `topic` / `message_encoding` / `schema_id` | **多 topic 模型**，同构于 h5 多节点 |
| Message | `channel_id` / `log_time` / `publish_time` / `data` | 时间戳均 **uint64 纳秒** |
| Chunk | `compression`（zstd/lz4/空） | 压缩由 `mcap` 包透明处理 |

两条与项目历史直接相关的结论：

1. **`message_encoding` 决定解码难度**。`json` → 直接可用；`cdr` → 需 schema 反序列化。
   项目此前只接触过前者（导出端已把 CDR 解成 JSON），**从未解过真正的 CDR**——这是本期
   与第二期的分界线。
2. **`log_time` vs `publish_time` 的默认选择沿用既有决策**。
   `docs/时间对齐能力改造设计.md` 第 9 节决策 4 已拍板：默认 `log_time`
   （录制侧统一时钟，跨流可比）；`publish_time` 来自各设备时钟域。本设计**继承**该决策，
   不重新论证。

---

## 3. 与既有架构的接入方式（核心：复用 h5 节点流范式）

MCAP「单文件多 topic」与 HDF5「单文件多节点」**完全同构**。因此沿用 h5 节点流机制，
不新建范式：

| 环节 | h5 现状 | MCAP 对应 |
|---|---|---|
| 流登记 `path` | `"<h5 路径>::<node>"` | `"<mcap 路径>::<topic>"` |
| `format` | `"h5"` | `"mcap"` |
| 取数 | `resolve_table_name` → `read_hdf5_node` | `resolve_table_name` → `read_mcap_topic` |
| 登记函数 | `register_h5_node_streams` | `register_mcap_topic_streams` |
| 主表标记 | 信息量最大节点 `is_main=True` | 消息数最多 topic `is_main=True` |
| 概览透出 | `meta["h5_structure"]` | `meta["mcap_summary"]` |

**可直接复用的既有设施（零改动）**——已由原型实证：

- `expand_envelope`：MCAP topic 表（含 `data` object 列）能被直接展开为
  `data.orientation.x`、`data.linear_acceleration.z` 等点分列；
- `find_timestamp_columns`：靠内容指纹认出 `mcap_log_time_ns`（`source: fingerprint`）；
- 质检层（`check_temporal_sync` / `check_sensor_sanity`）、统计层、绘图、报告全部不改。

---

## 4. 关键设计决策：topic 名如何参与语义分类

### 4.1 问题（原型实测暴露）

原型验证发现：MCAP topic 表交给既有 `classify_table_stream` 会判成 `unknown`。
实测证据（`/imu` topic）：

```
raw cols: ['mcap_log_time_ns', 'mcap_publish_time_ns', 'mcap_sequence', 'data']
kind: unknown | label: 未知（无法分类） | conf: low（无法判定）
evidence: 判不出，未做硬猜；已排查：词典线索（动作/IMU/位姿/力/手部跟踪均未命中）；
          内容指纹（时间戳：列 mcap_log_time_ns 单调递增…；四元数：未命中；力：列名无 force/torque…）
```

**根因**：`classify_table_stream` 的第 1 层词典依赖**文件名**（`accel.csv` → 文件名含
`accel`），第 2 层指纹依赖**顶层列名**（`quat_x` 等）。而 MCAP 的语义信息在**两个既有
通道之外**：① topic 名（`/imu`）不含 `accel/gyro` 字样；② 信号藏在 `data` 嵌套 dict 内
（`data.orientation`），对顶层列词典不可见。

这正是 `docs/多时钟与嵌套信封支持设计.md` 记录的 **F1 失效**（wujiGlove 26/26 全 unknown）
在 MCAP 层的同型复现。**若不正视，MCAP 接入会直接继承这个已知缺陷。**

### 4.2 决策：topic 名作为「第 1 层命名线索」注入，不做新分类器

**方案（选定）**：把 topic 名改写为**伪文件名**送给 `classify_table_stream`，
并同时传入**展开后的样本**作为第 2 层指纹输入。这样：

- topic `/imu` → 伪文件名 `imu.mcap::/imu`（含 `imu` 线索）；
- topic `/joint_states` → 含 `joint` 线索 → 命中 `_ACTION_STATE_COLS` 的关节前缀；
- topic `/tf_static` → 含 `static` 线索 → 可判为静态流；
- 样本用 `expand_envelope` 展开后传入 → 四元数 / 加速度指纹**可见**。

即：**不新增分类器、不新增层级**，只是给既有的四层识别架构**补上 MCAP 特有的两个输入
通道**（topic 名 + 展开样本）。这与 `docs/四层语义识别架构.md` 的「语义角色优先、内容
指纹裁判」原则一致——MCAP 的 topic 名恰恰是最强的**命名线索**，而展开样本让指纹层够得着。

### 4.3 命名线索映射表（显式，可审计）

| topic 名特征 | 伪文件名线索 | 预期 kind | 依据 |
|---|---|---|---|
| `imu` / `accel` / `gyro` | 含 `imu` | `imu` | 第 1 层命名线索 |
| `joint` / `cmd` / `action` | 含 `joint` | `actions` | `_ACTION_STATE_COLS` 关节前缀 |
| `tf_static` / `static` | 含 `static` | `static`（不参与对齐） | `docs/时间对齐能力改造设计.md` 4.2 排除集 |
| `pose` / `odom` | 含 `pose` | `pose` | 第 1 层命名线索 |
| `tactile` / `touch` | 含 `tactile` | `tactile` | 自由 kind，展示用 |

**未命中命名线索的 topic**：走第 2 层展开样本指纹；仍判不出则**如实标 `unknown`**，
交由第 3/4 层（`propose_stream_semantics` 假设 → 用户确认）处理——**绝不硬猜**
（延续既有契约）。

### 4.4 不做什么（防范围蔓延）

- **不改 `classify_table_stream` 的签名与既有语义**——只是调用方（MCAP 分支）多传信息；
- **不把 topic 名写死进 `_sniffing` 词典**——伪文件名机制让既有规则自然生效，避免新增
  一份需要长期维护的 MCAP 专用词表；
- **不在第一期做 CDR topic 的语义识别**——非 JSON 编码 topic 在流清单中标
  `decodable=False` + 原因，不参与分类。

---

## 5. 时间戳与数据模型约定

### 5.1 列结构（与 wujiGlove 导出形态对齐）

`read_mcap_topic` 产出的 DataFrame 列：

| 列名 | 含义 | 单位（字段名自带） |
|---|---|---|
| `mcap_log_time_ns` | Message 记录时刻（容器） | **纳秒** |
| `mcap_publish_time_ns` | Message 发布时刻（容器） | **纳秒** |
| `mcap_sequence` | 消息序号 | 计数 |
| `data` | 消息体（object，嵌套 dict） | — |

`data` 列名**刻意与 wujiGlove 导出形态一致**，使 `expand_envelope` /
`discover_nested_fields` 的既有语义无缝衔接（原型已实证）。

### 5.2 单位纪律（沿用既有强制约定）

列名带 `_ns` 后缀，杜绝把纳秒当秒解读（项目曾因 `duration_s` 实为纳秒出过事故，
见 `docs/技术债.md` 第 4 条）。传感器时间藏在 `data.header.timestamp_us` 内，
与容器时间并存——**双时钟并列透出**由既有 `discover_nested_fields` +
`clock_artifact_suspected` 机制处理，本期不新增逻辑。

---

## 6. 解包落盘设计

### 6.1 落盘纪律（硬性）

- **按 topic 分文件**——异构 topic 结构差异大，合表会列爆炸；
- 落在调用方指定的 `output_dir` 下（工具默认 `outputs/mcap_unpack/<dataset_id>/`），
  **绝不写入数据集源目录**（延续 `docs/四层语义识别架构.md` 第 4 层"不污染原始数据"）；
- `outputs/` 已在 `.gitignore`，符合 AGENTS.md 第 2 条。

### 6.2 格式语义

| 格式 | 扩展名 | 语义 |
|---|---|---|
| `jsonl`（默认） | `.jsonl` | 每消息一行，保留嵌套、可流式追加 |
| `json` | `.json` | 整体一个数组 |
| `csv` | `.csv` | 扁平化宽表；`data` 列序列化为 JSON 字符串防丢字段 |

### 6.3 工具形态（有副作用，需显式 + 可预览）

与项目"工具默认只读"基调不同，落盘是**写操作**。设计为显式工具 `unpack_mcap`：

- 参数：`topics`（默认全部可解码）、`fmt`（默认 jsonl）、`output_dir`（可选）、
  `max_messages`（可选，防大文件）；
- 返回中给 `written`（文件清单 + 行数）与 `skipped_topics`（原因），
  **不把消息内容灌进上下文**（AGENTS.md 第 5 条上下文节俭）；
- 单条消息解码失败只计数跳过，不中断。

---

## 7. 依赖

| 包 | 用途 | 必要性 |
|---|---|---|
| `mcap` | 容器读取（JSON 编码） | **必需**（本期） |
| `zstandard` / `lz4` | chunk 解压 | 可选（压缩文件需要） |

`requirements.txt` 需补 `mcap`（及 `zstandard`/`lz4`）。环境实测版本：
`mcap 1.4.0` / `zstandard 0.25.0` / `lz4 4.4.5`。

**缺依赖契约**：与 `load_dataset.MissingDependencyError` 同款——`McapDependencyError.user_hint()`
必须明确"并非文件损坏"并给出 `pip install mcap`，**不得**用"可能损坏"兜底措辞。

---

## 8. 改动范围与 commit 拆分

| 文件 | 改动 |
|---|---|
| `app/tools/mcap_reader.py` | **已建（原型）**：probe / read / unpack |
| `app/tools/load_dataset.py` | `_SUPPORTED_FORMATS` 加 `.mcap`；新增 `register_mcap_topic_streams`；单文件分支 + 目录 `.mcap` 登记 |
| `app/tools/_data_access.py` | `resolve_table_name` 加 `mcap` 分支（按 `::` 拆 topic 读取） |
| `app/tools/mcap_tools.py` | **新增**：`unpack_mcap` 的 `@tool` 包装（或并入 mcap_reader） |
| `app/tools/__init__.py` + `app/services/chat_service.py` | 注册 `unpack_mcap` 工具 + `_TOOL_DROPPABLE` |
| `app/agent/agent.py` | SYSTEM_PROMPT 增 MCAP 说明（可选，简短） |
| `requirements.txt` | 补 `mcap` / `zstandard` / `lz4` |
| `tests/` | 已建 `test_mcap_reader.py`（19 用例）；补接线用例 + 骨架 |

**commit 拆分建议**：

- **Commit A**：`mcap_reader.py` 内核 + 测试（**已完成，可直接提交**）；
- **Commit B**：`load_dataset` / `_data_access` 接线（`.mcap` 可加载、topic 登记为流、
  按名取数）；
- **Commit C**：`unpack_mcap` 注册为工具 + 落盘 + prompt/依赖 + 测试。

---

## 9. 风险与取舍

| 风险 | 应对 |
|---|---|
| topic 名判不出语义（同 F1） | 伪文件名 + 展开样本双通道；仍判不出则如实 unknown，交假设/确认层 |
| 上下文爆炸（几十 topic × 数万消息） | 返回只给清单/计数/落盘路径，消息内容不进上下文 |
| 大文件内存 | `max_messages` + 逐条迭代；不用非 seekable 流的 `log_time_order` |
| 落盘副作用 | 显式工具 + 返回文件清单；落在 `outputs/` 不碰源目录 |
| 非 JSON 编码被误当可解 | `probe` 标 `decodable=False` + `decode_note`；解包时跳过并如实说明 |
| 依赖体积（开源分发） | 本期仅 `mcap`（极轻）；CDR 的 `rosbags` 留第二期可选依赖 |

---

## 10. 验收锚点

**合成数据（pytest，已绿）**：`tests/test_mcap_reader.py` 19 用例——probe 只读 summary、
双时钟并存、非 JSON 诚实降级、三格式落盘、不污染源目录、缺依赖契约。

**接线后新增**（Commit B/C）：

- `load_dataset` 加载 .mcap 成功，`meta["streams"]` 含 `<path>::<topic>` 条目；
- `resolve_table_name(ctx, "demo.mcap::/imu")` 返回该 topic 表；
- topic 名线索生效：`/imu` → `kind=imu`（不再是 unknown）；
- `unpack_mcap` 工具可被 agent 调用并落盘。

**回归**：全量套件不退化（接线前基线 550 passed, 1 skipped）。

---

_状态：Commit A（内核 + 测试）、Commit B（load_dataset / _data_access 接线）、
Commit C（unpack_mcap 工具注册 + 依赖）均已实施。测试 559 passed, 1 skipped。_
