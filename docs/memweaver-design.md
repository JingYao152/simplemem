# MemWeaver 算法框架设计（LoCoMo 版定稿）

> 状态：设计定稿，待实现（P0 → P1 → P2）。
> 前提：本方案基于两项前置分析——(1) 所有组织决策由 LLM 语义判断做出（替代数值阈值）；
> (2) LoCoMo 数据实测结构（10 对话；每对话 19–32 个 session，均值 27.2；每 session
> 中位 20 轮、最大 47 轮；每轮均值 22.7 词；恰好 2 名说话人；仅 session 级日期，
> 格式如 "1:45 pm on 6 August, 2022"；QA 无 question_date；evidence 均值 1.42/题）。

---

## 1. 核心原理

**组织 ⇄ 表示协同演化（co-evolution of memory organization and representation）**：
记忆库是一个写入时持续自组织的层级织物（fabric）。三个不变式：

1. 每条记忆的向量表示**继承其在织物中位置的上下文**（所属线程的活体摘要）；
2. 新写入通过**类型化编织操作**（supersede / refine / bridge）重组织物；
3. 所有组织决策由 **LLM 语义判断**做出——方案不含任何需调优的数值阈值。

与相近工作的划界：

| 相关工作 | 差异 |
|---|---|
| SeCom / LightMem（话题分段构建） | 它们是一次性静态分段；MemWeaver 线程跨 session 存续、可被唤醒 |
| RAPTOR（层级摘要） | 静态一次性建树；MemWeaver 摘要是随写入被**重写**的活文档 |
| A-Mem（Zettelkasten 链接） | 只链接不管理生命周期；MemWeaver 编织操作类型化且带有效期语义 |
| Zep/Graphiti（双时态知识图谱） | 需要图基础设施；MemWeaver 在**扁平向量库**上实现时点一致性 |
| Anthropic Contextual Retrieval | 静态上下文前缀；MemWeaver 上下文随摘要重写而更新（表示协同演化） |

## 2. 数据模型

在现有 `MemoryEntry`（lossless_restatement / keywords / timestamp / location /
persons / entities / topic）上扩展：

```text
kind:           "fact" | "thread_summary" | "entity_profile"
thread_id:      所属线程 id
valid_from:     生效时间。缺省 = 事实自身时间戳，否则 session 日期。
                天级分辨率（LoCoMo 仅有 session 级日期 → 结构决定，非参数）
valid_until:    "" = 至今有效；被 supersede 时闭合为接替事实的 session 日期
superseded_by:  接替者 entry_id（形成事实演化链）
links:          ["refine:<id>", "bridge:<id>"]（双向写入）
context_digest: 嵌入时使用的线程摘要指纹（P1 重上下文化用）
```

固定 ID 约定：线程摘要 `entry_id = "thread::<tid>"`；实体档案
`entry_id = "profile::<name>"`。LoCoMo 每对话恰好 2 名说话人 → 每对话 2 条档案，
档案层无容量参数。

存储后端（LanceDB）需补三个能力：`get_by_ids`、`update_metadata(entry_id, fields)`、
`delete_by_ids`（摘要/档案更新 = 删旧行 + 插新行）。

## 3. 写入管线（session 为原子单位）

Session 边界由数据自带（Dialogue.timestamp 变化即切换）。实测最大 session 47 轮
（约 1000 词），单 session 必然装进一个 prompt——无 flush 容量参数。

```text
对每个 session S（日期 d，解析为 ISO）:

  ── Call A: 线程分配（1 次 LLM 调用，temperature=0.7）──
  输入:  S 的全部对话轮（编号 1..n）
         + 全部现存线程的一行摘要（实测线程数量级 20–60，全量入 prompt 可行；
           无休眠机制、无预筛参数）
  输出:  [{turns: [编号...], thread: "existing:<tid>" | "new:<标题>"}]
  兜底:  解析失败 / 引用不存在线程 → 整个 session 归入一个新线程（确定性降级）

  ── Call B: 逐线程处理（每个被触及线程 1 次调用；线程间并行、线程内串行）──
  输入:  线程当前摘要
         + 线程现存全部事实（编号列出；编织候选 = 同线程全量，结构有界，无 top-k）
         + 分配到本线程的对话轮 + session 日期 d
  输出:  {
           facts: [{restatement, keywords, timestamp, location, persons,
                    entities, topic,
                    source_turn_ids: [来源 dialogue_id...],
                    weave: {op: none|supersede|refine|bridge, target: 候选编号}}],
           summary: "重写后的线程摘要",
           summary_impact: "none" | "minor" | "major",
           outdated_facts: [候选编号...],     // P1 重上下文化的语义触发信号
           exempt_turn_ids: [仅寒暄/确认的 dialogue_id...]
         }

  ── P3 覆盖债务审计（ENABLE_COVERAGE_DEBT_SCHEDULER）──
  对每个被分配的 dialogue turn，检查其是否出现在任一 facts.source_turn_ids
  或 exempt_turn_ids。缺失映射的内容性 turn 写入独立 CoverageDebt JSON
  记录，保存原始 turn、相邻 turn、线程、实体与主题。债务不写入向量库。
  后续会话再次触及同一线程或实体时，以债务 turn、相邻 turn 与相关事实
  发起一次小范围补抽；finalize() 对剩余债务补抽一次。补抽成功的条目经
  schema 校验后作为普通事实写入 Fabric，解析失败的债务保留 pending 状态。

  ── 编织执行（确定性代码，非 LLM）──
  supersede:  旧条目.valid_until = d; 旧条目.superseded_by = 新条目 id
  refine/bridge: 双方 links 追加对应类型边
  summary_impact != "none": 更新 "thread::<tid>" 摘要条目
  涉及说话人: 更新 "profile::<speaker>" 档案条目

finalize():
  跨线程补编织扫描（必要组件，非保险——knowledge-update 型旧事实可能被
  Call A 分进不同线程）:
    对每条 valid_until == "" 的事实，本地嵌入检索全库 top-3 跨线程近邻（零 LLM 成本），
    高相似候选对交 LLM 批量判定漏掉的 supersede
  → backend.optimize()
```

## 4. 表示层（P1：上下文继承嵌入）

双层设计——**文本层保持 SimpleMem 原味，向量层叠加织物上下文**：

- 文本层：`lossless_restatement` 不变，说话人/绝对时间/指代消解仍写在句子
  内部（Φ_coref + Φ_time），且 Call B 抽取时可见线程活体摘要，跨 session
  指代消解质量优于原"前一窗口 3 条"的弱上下文。事实文本**不可变**。
- 向量层：插入时嵌入文本 = `"[<线程摘要一句话版>] <lossless_restatement>"`。
  前缀**严格只进向量，不进任何存储文本**——回答端看到的仍是干净事实。
- 动机：无损语句的自包含是**句子级**的，不携带**主题级**语境（"Melanie
  买了水彩画笔"的单句向量不知道自己属于延续 8 个 session 的绘画爱好线索），
  开放域/单跳查询（cat4/cat1）的漏检主要来自这里。
- 与 Anthropic Contextual Retrieval 的区别：(a) 上下文来自活体线程摘要而非
  静态文档，随写入演化；(b) 有更新语义——摘要重写后由 `outdated_facts`
  点名的条目重嵌入（本地计算，零 API 成本）；(c) 不把上下文写进事实文本，
  避免 LLM 重写导致"无损"语句漂移。
- 可证伪性：`ENABLE_RECONTEXT` 关闭即退化为纯 SimpleMem 单句嵌入，
  P1 消融直接量化线程语境前缀的贡献。

## 5. 读取管线

设计原则：**检索基座 = SimpleMem 原生（与 baseline 完全共享、原样不动）；
MemWeaver 只叠加织物衍生原语（C3）+ 一个通用重排组件**。RRF 融合、符号路
改造、题型格式化 prompt、充分性门控均不纳入（属 EvolveMem 已声称成果或
既有文献族；后续读侧如有新想法再议——候选清单见文末）。

```text
anchor = 记忆库最大 session 日期        // LoCoMo 无 question_date → 结构规则

检索基座（SimpleMem 原生 hybrid_retriever，未改动）:
  多查询规划 + 语义/词法/符号三路 + 去重合并    // SimpleMem 论文 3.3 节原生架构
  唯一叠加: 语义路池含 fact + thread_summary + entity_profile 三粒度，
            且非 when 类问题加 as-of 过滤:
              valid_until == "" OR valid_until >= anchor      ← C3 织物衍生
            when 类问题不过滤（问历史需要旧事实）

→ 一跳扩展（起点 = 候选集中每个 kind="fact" 条目；摘要/档案不向外扩展）:
    ① supersede 链双向（superseded_by 及反向）→ 事实演化链成员    // cat2
    ② weave 边双向（links: refine/bridge）→ 跨线程桥接事实        // cat3
    ③ 结构归属边（thread_id → 活体摘要条目）→ 主题语境            // cat4
  仅一跳、入池去重；每个新条目记录 provenance（锚点 + 边类型）
  （evidence 实测均值 1.42 → 链接度数结构性低，无截断参数）← C3 织物衍生
→ 交叉编码器平铺精排（BAAI/bge-reranker-v2-m3，本地推理）:
  扩展后全池逐条打分，取 top RERANK_TOP_K=20 作为回答上下文。
  扩展条目的打分输入带 provenance 短前缀（如 "[bridge of: <锚点句>]"）——
  bridge 事实与 query 直接相似度低（正是其未被检索命中的原因），
  裸文本精排会再次排掉它们；锚点候选自身裸文本打分
          ← 唯一保留的通用读侧组件（继承标注，不进贡献声明）
→ 生成: SimpleMem 原生回答 prompt（不加题型格式化）；
        supersede 链成员相邻排布并标注:
        "[SUPERSEDED on <d> by Context N]"（时间题常问"之前是什么"）← C3
```

暂缓的读侧候选（后续有新想法再议）：RRF 融合、符号路日期窗口化与
person 过滤禁用、题型格式化 prompt、充分性门控（cat5 由评测 harness
现有的对抗 MCQ 流程兜底）。

## 6. LLM 决策点清单（替代超参数的全部位置）

| 决策 | 所在调用 | 替代的原参数 | 确定性兜底 |
|---|---|---|---|
| 对话轮归属哪个线程 | Call A | THREAD_ATTACH_THRESHOLD | 整 session 入新线程 |
| 摘要是否需要重写 | Call B summary_impact | SUMMARY_REWRITE_MIN_NEW_FACTS | 不重写 |
| 哪些旧事实过期需重上下文化 | Call B outdated_facts | RECONTEXT_DRIFT_THRESHOLD | 不重嵌 |
| 新旧事实的编织关系 | Call B weave | WEAVE_CANDIDATE_TOP_K（候选=同线程全量） | op=none |

全部 LLM 调用统一 temperature=0.7（config 项 `LLM_TEMPERATURE`，全局生效）；
每类兜底触发率记录为系统健康指标。

## 7. 参数账本

**调优参数：0 个。** MemWeaver 新增容量常数仅 1 个（fixed、不调优）：

| 常数 | 值 | 性质 |
|---|---|---|
| RERANK_TOP_K | 20 | 精排后进入回答上下文的条目数，容量常数（初版固定） |

检索基座沿用 SimpleMem 原生配置（SEMANTIC_TOP_K=25 / KEYWORD_TOP_K=5 /
STRUCTURED_TOP_K=5 / MAX_REFLECTION_ROUNDS=2），与 baseline 逐项相同，
作为对照控制变量不动。

被数据结构消灭的参数：THREAD_FLUSH_SIZE（session 边界）、线程候选池预筛
（线程数结构有界）、WEAVE_CANDIDATE_TOP_K（同线程全量）、BUNDLE_MAX_NEIGHBORS
（一跳闭包）。被移出方案的参数（随组件暂缓）：RRF_K、CONTEXT_TOP_K、
门控硬上限、符号路窗口化相关规则。

## 8. 阶段划分与 LoCoMo 验证映射

| 阶段 | 内容 | 主要受益类别（实测题数） |
|---|---|---|
| P0 | 数据模型 + 后端三能力 + Call A/B 写管线 + supersede + as-of 检索 + 兜底扫描 | cat2 时间题(321) |
| P1 | 上下文继承嵌入 + outdated_facts 重嵌入 + 实体档案入池 | cat1 单跳(282)、cat4 开放域(841) |
| P2 | 一跳扩展 + 交叉编码器平铺精排 | cat3 多跳(96) + 全局 |
| P3 | 事实来源映射 + 覆盖债务登记与延后补抽 | cat1 单跳、cat3 多跳的事实覆盖 |

验证方法：每阶段同 harness A/B（test_locomo10.py）；另用 QA 的 evidence 字段
直接量**检索命中率**（session/dia_id 级），不必等端到端分数。

消融开关（config，每项对应论文消融表一行）：
ENABLE_MEMWEAVER / ENABLE_WEAVING / ENABLE_SWEEP / ENABLE_RECONTEXT /
ENABLE_EXPAND_RERANK（一跳扩展+精排整体开关）。
P3 使用 `ENABLE_COVERAGE_DEBT_SCHEDULER`，默认关闭；债务记录保存在
`<LANCEDB_PATH>/<MEMORY_TABLE_NAME>_coverage_debts.json`，不会进入语义、
词法、符号检索或回答上下文。

## 9. 成本与风险

- **写时调用量**：每对话 ≈ 27 session × (1 + 平均 2–3 个被触及线程) ≈ 80–110 次
  LLM 调用，对比基线固定窗口 ≈ 15 次（约 6 倍）。换取读时上下文更小更准。
  论文核心效率图：写读合计的 accuracy-per-token Pareto 曲线。
- **风险与对策**：
  1. Call A 分配质量是全楼地基 → 实现后先在 1–2 个样本上人工审计线程划分；
  2. 跨线程 supersede 漏检 → 兜底扫描独立消融开关，单独量化贡献；
  3. LLM 决策随机性 → temperature=0.7 下决策方差高于贪心解码：必须报 3 次
     运行的均值±方差；线程分配 / supersede 等结构性决策若观察到不稳定，
     可对单次决策做 3 采样多数投票（成本 ×3，仅在审计发现不稳定时启用）；
  4. 解析失败 → 每个决策点有确定性兜底，兜底率入日志。

## 10. 与 EvolveMem 的关系（机制/策略分层）

代码层面无冲突：MemWeaver 全部实现于 `simplemem/core/`，EvolveMem
（`simplemem/evolver/`）有独立的 retriever/store/benchmark runner，
不共享检索代码。语义层面的主从关系定义如下：

1. **机制 vs 策略**：MemWeaver 定义写侧机制（线程/编织/有效期）与织物
   衍生读取原语；检索基座保持 SimpleMem 原生，EvolveMem 的检索数值
   动作空间不受影响。若将来进化循环作用于 MemWeaver，其动作空间是
   MemWeaver 的消融开关与 Call A/B prompt 变体——EvolveMem 进化策略，
   不触碰织物机制本身。
2. **问题类型处理**：MemWeaver 不引入题型格式化 prompt（EvolveMem
   已声称的成果）；读侧仅存的题型判断是"when 类问题跳过 as-of 过滤"
   这一条结构规则。EvolveMem 的 gold-category 旗标 prompt 仅作为其
   进化循环内部工具，不进核心管线。
3. **时间机制互斥（硬性规定）**：as-of 有效期过滤取代
   `time_decay_half_life_days` 软衰减。time_decay 是无事实生命周期时
   对 knowledge-update 的启发式补偿，与 supersede 同开会双重惩罚旧事实，
   且在 when 类问题上与"不过滤"策略直接冲突。MemWeaver 启用时
   time_decay 必须禁用。
4. **实验归因**：MemWeaver 主表与消融在纯 core 管线上跑（evolver 不参与）；
   "EvolveMem 外环进化 MemWeaver 策略"留作扩展实验。

**创新性边界（论文贡献声明的画法）**：RRF 融合、题型格式化 prompt、
充分性门控等 EvolveMem 已声称/文献已有的读侧方法**已整体移出方案**
（见第 5 节暂缓清单），从根源上消除撞车。系统内唯一保留的通用读侧
组件是交叉编码器重排器，标注为继承组件、不进贡献声明。声称的贡献
严格限于：C1 写时自组织织物（线程/活体摘要/类型化编织）、C2 组织⇄
表示协同演化（上下文继承嵌入 + 语义触发重嵌入）、C3 织物衍生的读取
原语（as-of 时点检索与一跳证据扩展——强调其存在依赖织物结构：无
valid_until 即无 as-of，无 weave 边即无一跳扩展）。时间锚本身不新，新的是
锚作用于写时编织产生的有效期而非分数软衰减。
**Baseline parity 纪律**：所有对照系统配备同一个重排器（唯一的通用
读侧组件）后再比较，检索基座与回答 prompt 均为 SimpleMem 原生且各系统
一致，使主表增量只能归因于 C1–C3。

## 11. LongMemEval 适配备注（暂缓，规则已定）

- anchor 改用数据自带 question_date；`_abs` 弃答题需要弃答机制——届时
  重新评估充分性门控（现已移出方案）或等读侧新想法；
- `_m`（~500 session）规模下线程摘要全量入 prompt 会爆预算 → 启用嵌入预筛
  （scaling rule：摘要总量超 prompt 预算才启用，预筛量为容量常数）；
- 跨线程兜底扫描在 knowledge-update 上从"保险"升级为"必要组件"。
