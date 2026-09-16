# 研究计划：embodied-data-agent 新增「任务语义标注 + 任务切片标注 + 质检」功能可行性评估

> 生成时间：2026-09-16
> 查询类型：**Depth-first（深度优先）**——核心问题单一（"这个 agent 能否加标注与质检"），
> 但需要从"现有代码能力边界""具身智能标注规范""VLM 自动分割技术可行性""具身数据质检标准"
> 四个不同视角切入，最后综合成一个可执行的可信结论。

---

## 1. 问题拆解

用户提出三件事，隐含一条共同主线：

| # | 需求 | 本质 | 与现有 agent 的关系 |
|---|---|---|---|
| A | **任务语义标注** | 给数据集/episode 标注任务描述、语言指令、任务类别 | 与既有 `propose_stream_semantics` → `profile.json` 落盘模式**同构** |
| B | **任务切片标注** | 把连续轨迹按原子动作切成 `{start_timestamp, end_timestamp, atomic_action, ...}` 列表 | **全新能力**——需要边界定位（数值）+ 动作命名（语义） |
| C | **质检** | 基于具体数据集标准的质量检查 | 现有 `check_temporal_sync` / `check_sensor_sanity` 已是雏形，属**扩展** |

关键区分（本评估的核心洞察）：
- A、C 主要靠**确定性计算 + 既有交互范式**，工程量可控；
- B 内部还要劈成两半：
  - **B1 边界定位**（"在第几秒切"）——可用机器人本体信号确定性计算；
  - **B2 动作命名**（"这段叫什么"）——必须靠视觉理解或人工，**这才是真瓶颈**。

## 2. 需验证的关键事实（已通过勘察与研究闭合）

| 待验证 | 结论 | 证据来源 |
|---|---|---|
| agent 是否已有任何写入/标注能力 | **无**。13 个工具全部只读，`RunContext` 无标注字段 | `app/tools/`、`app/services/chat_service.py:58` |
| 是否有"LLM 生成 → 用户确认 → 落盘"既有范式可复用 | **有**，`propose_semantics.py` + `profile_store.py`，落盘 `profile.json` | `app/tools/profile_store.py:217` |
| LLM 路径能否接收图像 | **不能**。`OpenAIChatCompletionsModel` + 文本 message，工具返回只有图片路径 | `app/llm/factory.py:80`、`inspect_video_frame.py` |
| 是否有抽帧基础设施 | **有**，ffmpeg 抽单帧（缺"均匀批量抽帧"） | `inspect_video_frame.py:224` |
| 质检是否已有统一返回格式与可调阈值 | **有**，pass/warn/fail + measurements/thresholds/affected_episodes，阈值在 settings.py | `docs/ARCHITECTURE.md` 6.1、`app/config/settings.py:90` |
| 现有质检规则是否覆盖"任务级/动作级/标注级"缺陷 | **不覆盖**，只覆盖时间同步与传感器信号 | 同上 |
| 用户给定的 schema 字段是否有行业依据 | **大部分有**，verb 表、手别、目标物体、noise 标记均可对标 | 见调研报告一 |
| VLM 能否可靠做秒级自动切分 | **不可靠**。最佳 Segment F1 仅 0.306（IoU≥0.75） | WGO-Bench 现场报告 |
| 是否有非视觉的确定性切分方案 | **有**，基于本体信号/速度/接触/夹爪状态 | 调研报告二 |
| 具身数据质检有无公开标准 | **无单一标准**，但有分层架构（RDA L1/L2/L3）与可量化公式可循 | 调研报告三 |

## 3. 研究执行方案（已执行）

### 3.1 本地代码勘察（主线，1 个 code-explorer subagent + 主线精读）
覆盖：目录结构 → 13 个工具签名与返回结构 → agent prompt 与工具注册 → RunContext 字段 →
services 编排 → UI 交互 → output_paths/profile_store 落盘机制 → 测试体系 → AGENTS.md 纪律 →
ARCHITECTURE.md 未来演进清单。

### 3.2 外部研究（3 个并行 research_subagent）
1. **原子动作标注规范**：RT-1 / BridgeData V2 / AgiBot World / Open X-Embodiment 的 verb 表、
   切分粒度标准、`interacting_hand` / `target_object_class` 取值体系、noise 标记实践。
2. **VLM 自动分割技术可行性**：T-PIVOT / UniTime / TimeLens / RoboSubtaskNet 等论文，
   WGO-Bench 实测指标，token 成本，非 VLM 的确定性替代方案，开源标注工具。
3. **具身数据质检标准**：DROID / AgiBot World / RoboMIND 的 QC 流程，RDA 四层框架，
   HF GIGO 量化公式，失败类型清单与具体阈值。

## 4. 报告结构（预设）

1. 执行摘要 —— 直接回答"能加/不能加/怎么加"
2. 现有 agent 的能力边界与"标注"性质的冲突（关键技术判断）
3. 需求 A：任务语义标注 —— 可做，复用既有范式
4. 需求 B：任务切片标注 —— 难在哪，B1/B2 拆解，三条路线对比与推荐
5. 需求 C：质检 —— 最容易，给出规则包设计与阈值来源
6. 落地路线图（分阶段，含验证方式）
7. 风险与边界（哪些必须诚实降级、哪些不该做）
8. 结论（逐条回答用户三个问题）
9. 局限性与待澄清事项
10. 参考文献

## 5. 终止准则
研究已满足终止条件：四个视角的事实均已闭合，关键数字（F1 0.306、F1@50 79.5、
碰撞阈值 15×median、饱和 7°、过暗 (50-μ)/50 等）均有可点击来源，无需追加 subagent。
