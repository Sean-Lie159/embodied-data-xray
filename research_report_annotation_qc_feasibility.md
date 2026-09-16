# 具身智能数据分析 Agent 新增「任务语义标注 / 任务切片标注 / 质检」功能可行性研究

> 研究对象：`embodied-data-agent`（Embodied-data-Xray，Python 3.12 + openai-agents + Streamlit）
> 研究日期：2026 年 9 月 16 日
> 研究方式：全仓库代码勘察（结构、工具签名、上下文契约、落盘机制、纪律文档）+ 三路并行外部调研（标注规范 / VLM 分割技术 / 质检标准）

---

## 执行摘要

结论是**三件事都能做，但难度并不相同，而且难点和你的预期不一样**。任务语义标注最好做，本质上与项目已有的"语义假设 → 工具验证 → 用户确认 → 落盘画像"机制完全同构，只需把落盘对象从"流的语义标签"换成"任务/原子动作词表"，工程量约等于新增一个工具。质检也确实是数据分析的延伸，且比预想的更有章法——业界已有成熟的分层架构（硬门禁判 fail、诊断项仅 warn）和一批可直接搬用的量化公式，现有的 `check_temporal_sync` / `check_sensor_sanity` 已经把统一返回格式和可调阈值这套地基打好了，缺的是"任务级/动作级/标注级"这几类规则。

真正需要你拍板的是**任务切片标注**。这里必须把它劈成两半看：**"在第几秒切开"**（边界定位）和**"这一段叫什么名字"**（动作命名）。边界定位可以用机器人本体信号（关节速度、末端速度、夹爪开合状态变化）做确定性计算，这条路可靠、可测、和你项目"数值不交给 LLM"的纪律一致。但动作命名本质上需要看懂画面，而当前这个 agent 的模型链路**只支持纯文本**——`OpenAIChatCompletionsModel` 配的是文本 message，`inspect_video_frame` 只返回图片路径、明确声明"不做内容理解"。所以自动生成你 schema 里 `atomic_action` 和 `action_description` 这两个字段，在当前架构下**做不了**，除非升级到视觉模型。

即便升级到视觉模型，外部实测数据也不乐观：目前最贴近这个任务的公开基准（WGO-Bench，743 个人工标注片段）显示，最佳自动子任务分割的 Segment F1 只有 **0.306**（要求 IoU ≥ 0.75 的严格匹配），端到端标 F1 仅 **0.168**。这意味着自动标注只能当"草稿"，必须配人在回路校验。因此我建议的路线是：**确定性的边界切分 + 人工/LLM 辅助命名 + 结构化落盘 + 标注复核质检**，把"自动生成的标注"和"人工确认的标注"用 `source` 字段严格区分——这恰好是项目已有的 `user_confirmed` / `content_fingerprint` 来源标记思路的直接延伸。

---

## 一、现有 Agent 的能力边界，以及它与"标注"的根本性张力

理解这个项目首先要抓住一个事实：**它是被设计成只读的，而且这一点贯穿了每一层**。13 个注册工具（`load_dataset`、`profile_data`、`inspect_streams`、`check_temporal_sync`、`check_sensor_sanity`、`compute_stats`、`plot_chart`、`generate_report`、`propose_stream_semantics`、`unpack_mcap`、`align_container_streams`、`inspect_video_frame`、`compare_datasets`）全部只做"读 → 算 → 出结论"，没有一个工具会写回数据集。`RunContext`（`app/agent/context.py`）持有的字段是 `df`、`dataset_id`、`meta`、`output_dir`、`findings`、`session_tag`，其中没有任何与标注相关的槽位。系统提示词里那句"你通过调用工具完成数据处理，自己不具备直接读写数据的权限"，就是这个定位的宣示。

这个定位带来一个必须正视的张力：**"标注"在生产流程上是一种数据写入行为**。但项目其实已经解决了这个张力，只是用在另一个场景上。看 `propose_stream_semantics` 这条链路：模型从流的内容与嵌套发现里推断出语义分组（比如"这些是触觉流"），先用工具验证，把"假设 + 证据 + 未经验证"的标注转述给你，你明确同意后再以 `confirm=True` 落盘到 `outputs/by_dataset/<数据集名>/profile.json`，**跨会话永久生效**。落盘的每个映射都带 `label_source` 标记（`user_confirmed` / `content_fingerprint` / `dictionary`），`load_dataset` 下次加载时会优先读取画像覆盖自动识别结果。

这套机制是新增标注功能最重要的可复用资产，因为它精确地回答了"一个不允许自己编数字的 Agent，如何合法地产出标注"——**把 LLM 的产出限定为"待确认的假设"，把"确定性计算"和"人类裁决"设为它变成事实的两道闸门**。你 schema 里的 `atomic_action` / `action_description` 完全属于"LLM 可以提，但必须被确认"的这一类，而 `start_timestamp` / `end_timestamp` 属于"必须由确定性计算得出"的那一类。这条界线画清楚了，标注功能在架构上就没有原则性障碍。

另外要注意两个现成的落盘基础设施：`app/tools/output_paths.py` 统一管理 `outputs/by_dataset/<净名-短哈希>/` 下的 `charts/`、`reports/`、`profile.json`，文件名带 `session_tag` 前缀做多会话隔离；`app/tools/profile_store.py` 已经实现了原子写（临时文件 + `os.replace`）和基于锁文件的跨会话互斥，并做了"加锁 → 重读 → 合并 → 原子写"的读-改-写竞态防护。标注文件如果有多个并发会话在编辑，这套保护可以直接借用。还有一个细节值得注意：`profile_store.py` 的 `_PROFILE_FILENAME` 是单文件，如果标注要按 episode 分文件（比如 100 个 episode 就 100 个 JSON），需要设计新的目录结构而不是硬塞进 `profile.json`。

最后，有一条测试体系上的有利条件：`tests/fixtures/skeletons/` 按"数据格式结构骨架"组织回归，有 `build_skeleton` 生成器和参数化测试。新增标注功能可以先把"带标注文件的数据集骨架"固化进去，让标注的读写有可回归的结构契约。

---

## 二、需求一：任务语义标注（可做，工程量小）

任务语义标注指的是给数据集的 episode 或整体打上任务描述、语言指令、任务类别。主流数据集的做法可以直接对标：LeRobot 把任务指令放在 `meta/tasks.jsonl`（自然语言 → 整数 ID 映射），并在 v3 引入 `language_persistent`（全局指令）和 `language_events`（逐帧子任务标签）两个列；RT-1 用 **744 条指令**覆盖 **9 类技能**（pick object 130 条、move object near object 337 条、place object into receptacle 84 条等），但只标一整句自然语言指令、不标切分点；BridgeData V2 采用**事后众包标注**而非采集时标注，标注者被明确要求"描述机器人的任务并特别强调被移动物体的最终位置"，覆盖 **13 个 skill**（pick-and-place、pushing、wiping、sweeping、stacking、folding cloths、开关抽屉/门/纸箱翻盖、twisting knobs、flipping switches、拉拉链、turning levers）。

这里要纠正一个数字：BridgeData V2 是 **13 个 skill + 24 个环境**，业界常见的"24 类 skill"说法对不上原文。AgiBot World 则是目前唯一有公开"任务-技能两级切分"说明的大规模真机数据集，规模为 **100 万+ 轨迹、217 个任务、87 项技能、106 个场景**，明确提供 item / scene / skill（`sub-task segmented`）/ task 四级标注，并强调"每个原子技能至少由 100 条轨迹支撑"。

落地建议是把任务语义标注设计成一个与 `propose_stream_semantics` 同构的工具，比如叫 `annotate_task_semantics`。它做三件事：从数据现有痕迹（LeRobot 的 `task` 字段、文件名、已有语言标注列）中提取候选任务描述；在缺失时由模型基于已有证据提出待确认的假设；经你确认后落盘到数据集画像的新分片（建议 `outputs/by_dataset/<净名>/tasks.json`，与 `profile.json` 并列，避免单文件无限膨胀）。这个工具本身不需要新的数值能力，工作量主要在词表管理、落盘结构和与 `load_dataset` 的回读衔接上。

关于词表，外部调研的一个重要发现是：**Open X-Embodiment 并没有公开的原子动作动词表**。它聚合了 60 个数据集、527 种 skill、160,266 个 task，但论文明确没有统一改写各数据集的原始语言标签，只用 PaLM 从指令中抽取 objects 与 behaviors 做统计。所以不要指望能找到一份权威动词表直接抄。可行做法是**内置一个可配置的动词表**（以 RT-1 的 9 类技能 + BridgeData V2 的 13 个 skill 为骨架起步），并把词表放进 `app/config/` 或数据集的画像文件里，允许按数据集覆盖——这与你项目"阈值集中在 config、按数据集类型调整"的既有做法一致。

---

## 三、需求二：任务切片标注（核心难点，必须拆成两个子问题）

这是需要你重点裁决的部分。我把它拆成边界定位（B1）和动作命名（B2），因为二者的技术可行性差距极大。

### 3.1 为什么 B2（动作命名）在当前架构下做不了

`app/llm/factory.py` 的 `build_model` 构造的是 `OpenAIChatCompletionsModel(model=..., openai_client=AsyncOpenAI(...))`，全项目没有任何地方向模型发送图像内容块。`inspect_video_frame` 虽然能抽帧，但它的 docstring 和返回体都写得很清楚："**不做内容理解**——本工具只产出图片路径与元数据，画面含义由用户/模型看图判断"。而在当前链路里，"模型看图"这一步是不存在的：模型拿到的只是一行文件路径。

这不只是接口问题，还牵涉到项目的分层纪律——要让模型真的"看图"，得在 LLM 层或工具层引入图像内容块（比如工具返回 `{"type": "image_url", ...}`），这会触及 `RunContext` 不能序列化给模型的设计约定，也要确认你所用的网关（DeepSeek / Kimi / 混元）是否支持视觉输入以及多模态 message 的格式。这是**架构级改动**，不是加个参数就能解决的。

### 3.2 即使升级到视觉模型，外部实测数据也不支持"自动出标注"

外部调研找到了一个与本任务高度贴合、且极其有说服力的公开基准：**Macrodata 的 WGO-Bench 现场报告**（100 个片段、743 个人工标注片段、62 个任务、71.5 分钟；评分要求 Segment F1 的 IoU ≥ 0.75，是严格匹配）。关键数字如下。

| 指标 | 数值 |
|---|---|
| 最佳子任务分割 Segment F1 | **0.306** |
| 最佳子任务标注准确率 | 61.0% |
| 最佳端到端 F1 | **0.168** |
| 最优模型 | Gemini 3.5 Flash |
| 批量成本 | $2.64 / 小时视频，约为人工的 **1/19** |
| 实验规模 | 54 次分割实验，F1 范围 0.007–0.306 |

各类方法的分割 F1 对比同样值得注意：固定长度基线（5.77 秒均分）0.070，字幕嵌入相似度分割 0.081，逐帧时间戳图像 0.193，视觉时间戳 + 接触表 0.263。也就是说，**连"每 5.77 秒切一刀"这种完全不用模型的基线都能拿到 0.070**，而精心设计的视觉方案也只把 F1 推到 0.306。这个量级意味着自动结果无法直接入库。

学术侧的数据也一致。微软的 T-PIVOT（arXiv:2408.17422）用 GPT-4o 做开放词汇时序定位，在 Breakfast 上最好成绩是 MoF 60.1 / IoU 40.8 / F1 51.2，仍**逊于学习型 SOTA**（MoF 65.1 / IoU 52.1 / F1 54.6），且作者自述其方法假设了"已知动作序列"、动作步数越多性能越差。南京大学与腾讯 ARC 的 TimeLens（arXiv:2512.14698）指出更根本的问题：**Charades-STA 基准中 20.6% 的标注违反查询唯一性、34.9% 存在标注精度问题**——也就是说连"标准答案"本身都不够干净，这直接解释了为什么自动切分的评测数字上不去。综述类工作（arXiv:2508.10922）总结的四大局限也很契合：精确时序定位需要密集高分辨率输入，与 LLM 的 token 限制直接冲突；压缩策略会丢弃关键时序线索。

有监督方法能好得多——RoboSubtaskNet（arXiv:2602.10015，注意力增强 I3D + 改进 MS-TCN）在 GTEA 上达到 F1@50 **79.5** / Edit 88.6 / Acc 78.9，但在 Breakfast 上掉到 30.4 / 52.0 / 53.5，跨数据集泛化极差，而且需要训练。

### 3.3 B1（边界定位）：这条路可靠，且符合项目纪律

既然视觉命名走不通，那切分点从哪来？答案是从**机器人自己的本体信号**里来。这是本报告最推荐的技术路线，理由是它与项目"确定性计算优先、数值不交给 LLM"的原则完全一致，而且可测、可回归。

可用的信号包括：关节位置/速度/加速度的突变点（action discontinuity，RDA 用 MAD 尖峰检测）、末端执行器速度曲线的变化点、夹爪开合状态的变化（抓取与释放的物理标志）、以及基于速度阈值的停顿段识别。学术界对应的方法族包括变化点检测（change point detection）、贝叶斯在线变点检测（BOCPD）、以及操作任务的子阶段分割方法（SPOT、UPN、LASER 一类）。这里有几个公开的量化公式可以直接搬用：HF 的 GIGO 工具用 `θ > 15 × median(|a|)` 判定碰撞/突波，加速度代理为 `a_t = (q_{t+1} - 2q_t + q_{t-1}) / (t_{t+1} - t_t)²`；用速度阈值 **0.1** 识别空闲步；用致动器饱和判据 `|a_t - q_{t+1}| > 7°`；用路径效率 `clip(D/L, 0, 1)`（D 为直线距离、L 为实际路径长度）识别犹豫绕路。

值得特别指出的是，RDA（Robot Data Audit）的实测发现给"人工干预边界"提供了直接支持：**在 AgiBot 官方人工接管（teleop takeover）的边界处，动作不连续尖峰富集 3.1 倍**。这意味着基于速度突变点的切分不仅可行，而且恰好能捕捉到语义上重要的事件——接管边界本身就是天然的切片点。这可以作为本方案的起点证据。

需要提醒的是粒度问题。你 schema 里给的示例是 16 秒和 4 秒两段，而外部资料对这个量级的"合理性"没有任何权威界定：DROID 的轨迹约 5–20 秒，Open X v1.0 多数不足 5 秒，AgiBot World 是 30–60 秒（部分超 2 分钟），RoboMIND 按时间步区分（Franka/UR 短任务 <200 步，Tien Kung/AgileX 长任务 >500 步）。这些是**整段演示**的长度，不是原子动作的长度。所以"一个原子动作应该多长"没有公认标准，建议设成按数据集可配置的参数（比如最小片段时长、最大片段时长、最短合并阈值），而不是硬编码。

### 3.4 三条路线对比与推荐

| 路线 | 边界怎么来 | 动作名怎么来 | 可行性 | 代价 |
|---|---|---|---|---|
| **A. 确定性切分 + 人工命名**（推荐） | 本体信号变化点检测，可测可回归 | 用户在 UI/对话里命名，或从词表选择 | **高** | 需要新工具 + 标注编辑界面 |
| **B. 升级视觉模型做自动命名** | 同 A 或视觉 | VLM 看图生成 | 中（架构改动 + F1 仅 0.3，必须人工复核） | 需确认网关多模态支持，LLM 层要改 |
| **C. 纯 LLM 离线生成标注** | LLM 猜 | LLM 写 | **低**——模型看不到画面，等于编造，违反项目纪律 | 不可接受 |

我明确推荐 **A 作为第一阶段**，B 作为后续可选增强（且必须配人在回路），C 应当排除——因为它恰好踩中项目最核心的纪律红线："LLM 假设不得用于数值计算""不得把假设当作事实陈述"。用纯文本模型去生成 `action_description`，产出的就是无证据的编造，这与 `docs/ARCHITECTURE.md` 3.3.1 节确立的三层架构（语义角色规则管正确性、支持清单管范围、LLM 假设管盲区且必须显式标注待确认）直接冲突。

### 3.5 落盘格式与 schema 建议

你给的 schema 设计得不错，字段划分（时间区间 / 动词 / 双语描述 / 手别 / 目标物体 / 噪声标记）与 AgiBot World 和 RoboMIND 的实践能对上。几点具体建议：

第一，**给每个片段加 `source` 与 `confidence` 字段**，取值参照 `profile_store.py` 的既有约定（`user_confirmed` / `signal_derived` / `llm_proposed`），这样"哪些是算出来的、哪些是人确认的、哪些是模型猜的"在数据里是显式可查的，与你 `label_source` 的做法一致。

第二，**`start_timestamp` / `end_timestamp` 建议同时保留帧号与秒值**。你示例里的 `"00:00:16"` 字符串格式对人类友好但不利于程序精确对齐；建议落盘形如 `{"start_s": 16.0, "end_s": 20.0, "start_frame": 400, "end_frame": 500}`，显示层再格式化成 `HH:MM:SS`。

第三，**`atomic_action` 的命名风格需要统一**。你的示例是 `Place`、`Walk_Forward`（首字母大写 + 下划线），而 RT-1 与 BridgeData 用的是自然语言小写短语并带物体（`pick iced tea can`、`stack the blocks`）。两者各有道理——前者适合做机器可读的受控词表，后者信息量更大。建议 `atomic_action` 用受控词表（下划线风格，如 `Pick_Up` / `Walk_Forward`），把自然语言细节放进 `action_description`，这样 `atomic_action` 可以直接用于分组统计和质检。

第四，**`target_object_class` 建议关联一个可扩展的物体类别表**。外部调研没有找到具身领域统一的物体类别表（EPIC-KITCHENS 有自己的 noun 体系，GRASP 有另一套，彼此不通用），所以同样建议按数据集配置。

第五，关于 `interacting_hand`：调研明确显示 **AgiBot World 没有公开的 `left/right` 手别逐帧标签**，只有末端执行器类型的区分（gripper 夹爪 / dexterous hand 灵巧手）。所以这个字段在你的数据集上是自定义的，建议取值枚举定为 `left_hand` / `right_hand` / `both_hands` / `none`，并注意示例里的 `"none"` 与其它取值风格不一致（其余带 `_hand` 后缀），建议统一。

---

## 四、需求三：质检（最容易，且有现成章法可循）

你的判断是对的，质检确实是数据分析的延伸，而且它比标注更容易落地。先说一个重要的定性结论：**具身智能领域目前没有一份正式发布的、行业公认的"数据集质检标准"**。存在的是四类可依据的资源，各自的严谨度不同。

第一类是论文里描述的定性 QC 流程。DROID（arXiv:2403.12945）的做法是把约 16k 条标记为 unsuccessful 的轨迹在训练时排除（但仍发布），并用众包做语言标注、每条 episode 最多 3 条独立标注做冗余校验，场景从 2,080 个去重到 564 个——但它**没给出任何定量质量指标**。AgiBot World（arXiv:2503.06669）用三阶段流程（预采集可行性验证 → 正式采集含本地丢帧检查 → 后处理逐条人工验证），并给出了一个很有说服力的量化证据：在 "Wipe Table" 任务上，**528 条人工验证轨迹 vs 482 条未验证轨迹，验证后子集完成分数提升 0.18**，说明验证比数量更重要。它还保留约 1% 的失败轨迹并手动标注失败原因与时间戳。

第二类是最有价值的一份——**RoboMIND（arXiv:2412.13877）定义了 8 项质量保证标准**，而且全部是从视频中标注的：Touch Excess（不必要的接触）、Movement not Smooth（抖动或中断）、Secondary Grabbing（失败后重复抓取）、Mechanical Arm Shaking（异常振动）、Collision before Grabbing（抓取前碰撞）、Image Distortion（图像失真）、Failed Placement（放置位置不正确）、Gripper out of the Camera（夹爪超出画面）。它的三步流程是初步检查（快速浏览，确认无掉帧卡顿）→ 详细检查（逐帧/慢动作核对 8 项）→ 过滤与问题记录（记录**具体时间戳 + 描述**并分类）。这套"记录时间戳"的做法与标注功能天然衔接。

第三类是格式规范的硬约束。LeRobot 的 `meta/info.json` 定义 schema（feature 名、dtype、shape、fps、codebase_version），源码里有 `fps <= 0` 抛 ValueError 的硬校验、版本兼容性校验、以及推送前必须调 `finalize()` 的强制约束。但需要注意，跨 episode 的 feature 一致性、时间戳结构、episode 长度约束这些，**官方文档没有展开**，实现细节在源码里，而"时间戳/帧连续性检查不在 utils.py 中"——也就是说 LeRobot 本身并不提供完整的质检脚本。

第四类是社区工具，这是最接近你要的"可配置质检规则"的参考。**RDA（Robot Data Audit）采用了最值得借鉴的四层架构**：L1 完整性闸门（9 项硬检查，可判 PASS / REVIEW / EXCLUDE：missing_dropout、invalid_values、schema_consistency、temporal_validity、joint_limit、video_frame_integrity、video_freeze、video_timestamp_alignment、video_stream_presence），L2 轨迹诊断（8 项，仅报告不判定：sensor_sync、sampling_jitter、velocity_acceleration、action_discontinuity、visual_quality 等），L3 数据集画像（4 项：idle_ratio、distribution、coverage、temporal_structure），L4 汇总 P10/P50/P90。它的盲测成绩是精确率 **1.000**、召回率 **0.800**（在 pusht 数据集注入 50 个缺陷 episode）。

RDA 有一条设计哲学对你的设计极其重要：**只有 L1 硬完整性检查能触发 EXCLUDE，L2/L3 诊断永不自动升级为判定**。作者披露早期版本（v0.5.x）因为让 idle_ratio 自动升级判定，导致在 libero_10 上产生 **65% 的误报**，改成新架构后误报减少 97–100%。原因很直白——**阈值必须任务相关，不存在通用阈值**：70% 的空闲比对 push 任务正常，对 lift 任务就可疑；33 个尖峰对脚本生成数据正常，对平滑遥操作数据就该报警。这正是你说的"要基于针对具体数据集的质检标准"在工程上的确切含义。

另外还有一个可直接搬用的公式集（来自 HF RobotData 的 GIGO 工具）：过暗惩罚 `max(0, (50.0 - μ) / 50.0)`（只惩罚平均亮度低于 50/255 的过暗，刻意不惩罚过曝以避免误报）、模糊度用拉普拉斯方差 `Var(ΔI)`（核 `[0,1,0; 1,-4,1; 0,1,0]`，方差越低越模糊）、碰撞用 `θ > 15 × median(|a|)`、致动器饱和用 `|a_t - q_{t+1}| > 7°`、空闲用速度阈值 0.1、路径效率用 `clip(D/L, 0, 1)`。

### 4.1 与现有代码的衔接

现有质检工具的返回格式已经完全对齐业界做法。`check_temporal_sync` 返回 `{result: pass/warn/fail, measurements, thresholds, affected_episodes, baseline_recommendation, unit_warnings}`，阈值集中在 `app/config/settings.py`（`sync_*` / `sanity_*` / `stats_*` 前缀分组）。新质检规则只需要按同一模板扩展，并且建议**把 RDA 的分层思想引入 `result` 字段的语义**：明确区分"硬门禁（可 fail，如 NaN/Inf、时间戳非单调、丢帧、schema 不一致）"与"诊断项（仅 warn，永不自动 fail，如空闲比、抖动、路径效率）"。

现有覆盖的空白也很清楚。已经有的：时间同步（`check_temporal_sync`）、传感器合理性（`check_sensor_sanity`，含 NaN/Inf、恒定通道、量程饱和）。缺的是三类：**动作级**（动作抖动/尖峰、动作不连续、轨迹平滑度、犹豫绕路、空闲段比例）、**任务级**（episode 时长分布离群、任务失败、演示不完整）、**标注级**（任务标签缺失、标注粒度不均、时间区间重叠或空隙、类别不平衡、标注与数据时长不匹配）。这三类恰好可以合并成一个新的质检工具，并且**标注级质检是标注功能的天然搭档**——一旦有了切片标注，就能自动检查"相邻片段是否重叠""是否有未覆盖的时间空隙""片段时长是否在合理区间"，这类检查确定性极高、几乎零误报。

### 4.2 一个应该避免的坑

调研里明确列出了一批"未找到公开出处"的阈值：时间戳容差的统一 ms 值、采样率不稳/数据缺口的百分比阈值、动作 jitter 的绝对数值阈值、多相机同步偏移的 ms 容差、episode 时长的合理区间硬阈值、标注类别不平衡的量化阈值。特别是**动作 jitter**——"Consistency Matters"（arXiv:2412.14309）明确提出反对用绝对值判定，改用 K-means 聚类分组。所以这些阈值应该设计成**按数据集可配置 + 默认值仅供参考**，并且在返回里带 `thresholds` 与来源说明，绝不能让模型把兜底假设的数值当成物理事实转述——这一点你项目的 `unit_warnings` 机制已经提供了现成范式。

---

## 五、落地路线图建议

考虑到项目有"先文档后代码""小步交付，一次只实现一个工具或模块"的硬性纪律，我建议分四个阶段，每阶段都能独立验收。

**阶段一：质检规则扩展（先做这个，风险最低、复用度最高）。** 新增一个任务级/动作级质检工具，规则分硬门禁与诊断项两层，阈值全部进 `settings.py` 并标注来源。复用 `check_temporal_sync` 的返回模板。验收方式是合成数据注入已知缺陷后断言检出（可参照 RDA 的盲测做法，以及项目 `tests/fixtures/skeletons/` 的骨架机制）。

**阶段二：任务语义标注。** 新增 `annotate_task_semantics` 工具，形态与 `propose_stream_semantics` 同构；落盘到 `outputs/by_dataset/<净名>/tasks.json`；动词表可配置、以 RT-1 9 类 + BridgeData 13 个 skill 为初始骨架。验收方式是确认后重载数据集，画像能正确回读并标注 `label_source=user_confirmed`。

**阶段三：切片边界自动检测。** 新增基于本体信号的变化点检测工具，输出候选切片点（带证据指标，如速度突变幅度、夹爪状态变化位置），**只给边界不给名字**。这是纯确定性计算，可以完全按现有工具的方式单测。同时要把"带标注文件的数据集骨架"加进 `tests/fixtures/skeletons/`。

**阶段四：标注编辑与落盘 + 标注质检。** 在这一步才需要 UI 参与：一个时间轴式的标注审阅/编辑界面（Streamlit 侧栏或独立 tab），让你对候选切片命名、修正边界、标记 `is_noise` 与 `cleaning_reason`；落盘时带 `source` 与 `confidence`；随后由标注质检工具校验重叠、空隙、时长合理性。如果你希望自动命名，这一步再评估视觉模型接入（需要你先与所用网关确认多模态支持）。

---

## 六、风险与边界

**最大的风险是让模型编造 `action_description`。** 当前链路模型看不到画面，任何自动生成的动作描述都是无证据的推理产物。这与你项目系统提示词第 1 条（"只能基于工具返回的真实结果进行分析和表述，不得假设、推测或编造"）和第 9 条（"假设不得伪装成结论"）直接冲突。所以在拿到视觉能力之前，`action_description` 必须来自人工输入或明确标注为 `llm_proposed` 的待确认假设。

**第二个风险是阈值误报。** RDA 的 65% 误报事故是明确的先例。缓解办法就是它验证过的那条：硬门禁与诊断项分离，诊断项只 warn 不 fail，并且把阈值做成按数据集配置。

**第三个风险是标注文件规模。** 如果每个 episode 都产出一个 JSON，一个 1,000 episode 的数据集就是 1,000 个文件。建议落盘结构按数据集聚合（如 `annotations/<split>.jsonl`，每行一个片段），而不是每 episode 一个文件——JSONL 逐行读写的模式项目里已有实现（`_data_access.read_jsonl_rows`）。

**第四点需要提醒的是上下文预算。** 标注结果如果很长，会成为工具返回的一部分送进模型。你项目有 `_TOOL_DROPPABLE` 机制（`app/agent/agent.py` 里按工具配置"优先丢弃字段"）和 `_output_guard.enforce_output_limit` 三层防御，新工具的返回也应该登记 `_TOOL_DROPPABLE`，把逐片段明细列为可丢字段、只保留计数与摘要。

---

## 七、结论

**任务语义标注可以加。** 它与项目已有的"LLM 提假设 → 工具验证 → 用户确认 → 落盘画像"机制完全同构，主要工作在词表管理、落盘结构和回读衔接上，不需要新的数值能力。业界可对标 RT-1 的 9 类技能、BridgeData V2 的 13 个 skill、AgiBot World 的 217 任务/87 技能两级结构。

**质检可以加，而且是三件事里最容易的。** 你的直觉正确。现有 `check_temporal_sync` / `check_sensor_sanity` 已经把"测量值 + 阈值 + pass/warn/fail + affected_episodes"的统一格式和 config 化阈值打好了地基，缺的只是动作级、任务级、标注级三类规则。建议照搬 RDA 的四层架构与"硬门禁可 fail、诊断项仅 warn"的设计哲学，以及一批有公开公式的量化规则（过暗 `max(0,(50-μ)/50)`、模糊 `Var(ΔI)`、碰撞 `15×median(|a|)`、饱和 `7°`、空闲速度 0.1、路径效率 `D/L`）。

**任务切片标注可以做，但必须变通，且关键瓶颈不在切分而在命名。** 边界定位可以用机器人本体信号（速度突变、夹爪状态变化、动作连续性）做确定性计算——这条路可靠、可测，且 RDA 实测发现人工接管边界的动作不连续尖峰富集 3.1 倍，说明信号驱动切分能捕捉到语义上重要的事件。但 `atomic_action` 与 `action_description` 的自动生成在当前架构下**做不了**：模型链路是纯文本的（`OpenAIChatCompletionsModel` + 文本 message），`inspect_video_frame` 明确声明不做画面理解。即便升级到视觉模型，公开实测也显示最佳自动子任务分割 Segment F1 仅 **0.306**（IoU ≥ 0.75 严格匹配）、端到端 F1 仅 **0.168**，只能当草稿并必须配人工复核。

因此我建议的目标形态是：**确定性边界切分 + 人工（或视觉模型辅助）命名 + 结构化落盘（带 source/confidence 溯源）+ 标注复核质检**，四件事串成一条闭环，而质检既服务于数据集本身，也服务于你自己产出的标注。

---

## 八、局限性与待澄清事项

本报告有三个需要你补充才能进一步收窄的点。第一，**你的目标数据集究竟是哪一类**——是 LeRobot v2 格式（有现成的 `meta/tasks.jsonl` 和 episode 划分）、还是自定义的多流采集目录（`app/tools/_data_access.py` 里的 `episode` / `traj_id` 列候选能识别）、还是 HDF5 容器？这直接决定标注锚定在哪一层，也决定本体信号是否可得。你 schema 示例里出现的"筷子、锅、蔬菜、左手端的碗"看起来是双臂灵巧手数据，这类数据通常有丰富的关节与力觉信号，对边界检测很有利，但也可能没有 episode 划分列。

第二，**你用的模型网关是否支持视觉输入**。这决定路线 B（自动命名）是否具备前提。如果支持，还需要评估多模态 message 如何与 `openai-agents` 的 `RunContext` 设计（上下文对象不序列化给模型）共存。

第三，**标注的最终用途是什么**——是作为下游策略训练的真值、作为数据集发布时附带的元数据、还是仅作为内部数据清查的辅助？用途决定了对精度和落盘格式的严苛程度：训练真值需要严格的人工复核流程，内部清查则草稿级别可能就够了。

另外需要明确，本报告对质检阈值的引用中，有一批数字在公开资料里找不到出处（时间戳容差的统一 ms 值、缺口百分比阈值、jitter 绝对阈值、多相机同步偏移容差、episode 时长合理区间）。这些都已在上文注明，在实现时应当作为"待标注来源的可配置默认值"处理，而不是当作权威标准使用。

---

## 参考文献

1. [RT-1: Robotics Transformer for Real-World Control at Scale (arXiv:2212.06817)](https://arxiv.org/html/2212.06817)
2. [Open X-Embodiment: Robotic Learning Datasets and RT-X Models (arXiv:2310.08864v9)](https://arxiv.org/html/2310.08864v9)
3. [BridgeData V2: A Dataset for Robot Learning at Scale (arXiv:2308.12952)](https://arxiv.org/html/2308.12952)
4. [BridgeData V2 项目主页 (RAIL Berkeley)](https://rail-berkeley.github.io/bridgedata/)
5. [AgiBot World Colosseo: A Large-scale Manipulation Platform (arXiv:2503.06669v3)](https://arxiv.org/html/2503.06669v3)
6. [AgiBot-World GitHub (OpenDriveLab)](https://github.com/OpenDriveLab/Agibot-World)
7. [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (arXiv:2403.12945)](https://arxiv.org/html/2403.12945v1)
8. [RoboMIND: Benchmark on Multi-embodiment Intelligence Normative Data (arXiv:2412.13877v3)](https://arxiv.org/html/2412.13877v3)
9. [RoboMIND 项目主页](https://x-humanoid-robomind.github.io/)
10. [T-PIVOT: Open-vocabulary Temporal Action Localization using VLMs (arXiv:2408.17422)](https://arxiv.org/abs/2408.17422)
11. [UniTime: Universal Video Temporal Grounding with Generative Multi-modal LLMs (arXiv:2506.18883)](https://arxiv.org/abs/2506.18883)
12. [TimeLens: Rethinking Video Temporal Grounding (arXiv:2512.14698)](https://arxiv.org/abs/2512.14698)
13. [A Survey on Video Temporal Grounding with Multimodal Large Language Model (arXiv:2508.10922)](https://arxiv.org/abs/2508.10922)
14. [RoboSubtaskNet: Subtask Segmentation for Robot Manipulation (arXiv:2602.10015)](https://arxiv.org/abs/2602.10015)
15. [Macrodata: Segmenting Robot Video into Actionable Subtasks (WGO-Bench field report)](https://macrodata.co/blog/annotating-robot-video-subtasks)
16. [macrodata-labs/refiner (开源分割管线)](https://github.com/macrodata-labs/refiner)
17. [Consistency Matters: Defining Demonstration Data Quality Metrics in Robot Learning from Demonstration (arXiv:2412.14309)](https://arxiv.org/html/2412.14309v1)
18. [Garbage In, Garbage Out: The Case For Better Robot Data Understanding (HuggingFace RobotData, 2025-10-26)](https://huggingface.tw/blog/robotdata/gigo)
19. [score_lerobot_episodes (HuggingFace RobotData 质检工具)](https://github.com/RoboticsData/score_lerobot_episodes)
20. [RDA — Robot Data Audit (GitHub)](https://github.com/liesliy/rda)
21. [RDA: We Audited 13 Public Robot Datasets With One Tool](https://dev.to/liesliy/we-audited-13-public-robot-datasets-with-one-tool-and-zero-tuning-heres-what-the-numbers-actually-1f41)
22. [lerobot-doctor (GitHub)](https://github.com/jashshah999/lerobot-doctor)
23. [LeRobot Dataset Format and Structure (DeepWiki)](https://deepwiki.com/huggingface/lerobot/2.1-dataset-format-and-structure)
24. [LeRobot datasets/utils.py 源码](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/datasets/utils.py)
25. [Robot Data Curation with Mutual Information Estimators (arXiv:2502.08623)](https://arxiv.org/abs/2502.08623)
26. [Timeseries-QC: Timestamp Validation](https://nagusubra.github.io/timeseries-qc/timestamp-validation/)
27. [Open X-Embodiment GitHub (google-deepmind)](https://github.com/google-deepmind/open_x_embodiment)
