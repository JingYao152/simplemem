# MemWeaver 实现说明（P0 + P1 + P2）

对应 `docs/memweaver-design.md` 第 8 节的三个阶段：

- **P0**：数据模型 + 后端三能力 + Call A/B 写管线 + supersede 执行 + as-of 检索
  + finalize 兜底扫描（目标类别：cat2 时间题 321 题）
- **P1**：上下文继承嵌入 + `outdated_facts` 重嵌入 + 实体档案入池
  （目标类别：cat1 单跳 282 题、cat4 开放域 841 题）
- **P2**：一跳证据扩展 + 交叉编码器平铺精排 + supersede 链标注
  （目标类别：cat3 多跳 96 题 + 全局）

本文只记录"设计 → 代码"的落点、实现期做的判断，以及验证方式；设计本身以
`memweaver-design.md` 为准。

## 1. 代码落点

| 设计条目 | 代码位置 |
|---|---|
| 数据模型扩展（kind/thread_id/valid_from/valid_until/superseded_by/links/context_digest）+ 固定 ID 约定 | `simplemem/core/models/memory_entry.py` |
| 存储 schema（新增 7 列）+ 旧表迁移 | `simplemem/core/database/vector_store_backend.py` `_init_table` / `_migrate_table` |
| 后端三能力 `get_by_ids` / `update_metadata` / `delete_by_ids` | 同上（Protocol + LanceDB 实现）、`vector_store.py`（facade） |
| as-of 过滤所需的比较/析取谓词 | `vector_store_backend.py` `FieldPredicate` / `AnyOf` |
| session 日期解析（天级分辨率） | `simplemem/core/memweaver/dates.py` |
| 织物快照（线程/事实/线程号分配） | `simplemem/core/memweaver/fabric.py` `load_fabric` |
| 编织执行（supersede 闭合有效期、refine/bridge 双向边） | `fabric.py` `execute_supersede` / `execute_link` |
| Call A / Call B / 兜底扫描 prompt | `simplemem/core/memweaver/prompts.py` |
| 写管线（session 切分、Call A、Call B、编织执行、活体摘要、跨线程扫描） | `simplemem/core/memweaver/writer.py` |
| 时间锚 + when 类问题规则 + as-of 谓词 | `simplemem/core/memweaver/asof.py` |
| **P1** 上下文前缀 / 嵌入文本 / `context_digest` 指纹 | `simplemem/core/memweaver/context.py` |
| **P1** 嵌入文本覆写（前缀只进向量）+ 原地重嵌入 | `vector_store.py` `add_entries(embed_texts=...)` / `reembed_entries`；后端 `update_vector` |
| **P1** `outdated_facts` 触发的重上下文化 | `writer.py` `_recontextualize` |
| **P1** 实体档案构建（确定性）与人物→线程归属 | `fabric.py` `build_entity_profile_entry` / `speaker_threads`；`writer.py` `_update_profiles` |
| **P2** 一跳扩展（三类边 + provenance） | `simplemem/core/memweaver/expansion.py` |
| **P2** 反向边查询（谁指向我） | `vector_store_backend.py` `find_by_field` |
| **P2** 交叉编码器精排（继承组件，不在 memweaver/ 内） | `simplemem/core/reranker.py` |
| **P2** supersede 链相邻排布 + `[SUPERSEDED ...]` 标注 | `simplemem/core/answer_generator.py` |
| 检索侧 as-of 叠加 / 扩展 / 精排装配 | `simplemem/core/hybrid_retriever.py` |
| 消融开关 / `LLM_TEMPERATURE` | `simplemem/core/settings.py`、`simplemem/core/config_default.py` |
| 系统装配（写路由 + finalize） | `main.py` |
| 评测：A/B 同 harness + evidence 检索命中率 | `test_locomo10.py`、`locomo_evidence.py` |

单元测试：`tests/test_memweaver.py`（91 例，全部 LLM 调用脚本化与交叉编码器
桩化，覆盖每个决策点的确定性兜底、P1 的前缀不落库/指纹幂等/档案重写、
P2 的三类边/一跳边界/provenance 打分/精排降级/链式排布）、
`tests/test_evidence_retrieval.py`（12 例）。

## 2. 参数账本

新增**调优参数 0 个**（与设计的核心承诺一致）。代码中出现的常数都是结构常数、
格式常数或评测常数，没有一个会改变任何组织决策：

| 常数 | 值 | 性质 |
|---|---|---|
| `SWEEP_NEIGHBOURS` | 3 | 设计第 3 节"top-3 跨线程近邻"，结构常数 |
| `SWEEP_SCAN_DEPTH` | 12 | 为拿到 3 个跨线程近邻而多取的行数，纯实现细节 |
| `SWEEP_BATCH_SIZE` | 20 | 判定 prompt 的分批大小；所有候选对都会被判定，不改变任何决策 |
| `CONTEXT_PREFIX_MAX_CHARS` | 200 | P1 上下文前缀的渲染长度上限，格式常数（同 `ThreadState.one_line()` 的 240） |
| `RERANK_TOP_K` | 20 | **设计第 7 节唯一的容量常数**：精排后进入回答上下文的条目数 |
| `RERANK_BATCH_SIZE` | 32 | 交叉编码器每次前向的对数，纯 plumbing，不改变任何打分 |
| `_ABBREVIATION_MAX_LEN` | 3 | 判定"句号属于缩写而非句末"的词长阈值，格式启发式 |
| `EVIDENCE_HIT_THRESHOLD` | 0.3 | **评测侧**命中判定阈值，不属于系统参数 |

扫描没有引入"相似度阈值"：候选对由"top-3 跨线程近邻"结构性给出，避免新增
可调参数。线程间并行度沿用既有 `MAX_PARALLEL_WORKERS`。

## 3. 实现期做的判断

1. **as-of 过滤覆盖三路检索**。设计第 5 节把 as-of 叠加写在语义路上；实现里
   语义路做**前置过滤**（下推到 LanceDB，top-k 预算不被过期条目吃掉），词法路
   与符号路对结果做**同一谓词的后置过滤**（全文检索无法前置过滤）。动机是让
   进入回答的候选池整体时点一致——否则 top-5 的词法路会把刚被 supersede 的
   旧事实重新带回来。谓词本身与设计逐字一致：
   `valid_until == "" OR valid_until >= anchor`（天级分辨率下边界闭合：恰好在
   锚点当天被闭合的事实仍可见）。
2. **时间锚的来源**。锚 = 记忆库最大 session 日期，取自被写侧打上 session 日期
   的条目：线程摘要的 `valid_from`（每次重写都落在当次 session 日期）与事实的
   `valid_until`（闭合日期）。事实的 `valid_from` 可能是对话中提到的**未来**
   日期（"下月要去露营"），只作最后兜底，避免锚被未来事件推高。
3. **when 类问题的判定**。实测 cat2 中 246/321 题含 "when"，其余是
   "How long ago…" / "How many weeks passed…" 这类同样需要旧事实的时间题，
   故结构规则为一条正则：`\bwhen\b | how long | how many (days|weeks|months|years)`。
   这是读侧唯一的题型判断（设计第 10.2 节允许的那一条）。
4. **实体档案**：`kind="entity_profile"` 与 `profile::<name>` ID 约定已就位，但
   **不生成**档案条目——设计第 8 节把"实体档案入池"排在 P1。P0 的语义池含
   fact + thread_summary 两种粒度。
5. **Call B 整体失败的兜底**：设计给出了各决策点的兜底（weave→none、摘要→不重写），
   未规定整个 Call B 失败时怎么办。为不丢信息，实现为"退化到 SimpleMem 原生抽取"
   （复用 `MemoryBuilder._generate_memory_entries`），产出的事实挂在该线程上、
   `op=none`，并计入 `call_b_fallback`。
6. **Call A 部分覆盖**：引用不存在线程或解析失败 → 整 session 入新线程（设计原文）。
   若只是**漏掉**了部分轮次，则保留已有分配、把漏掉的轮次放进一个新线程
   （`call_a_partial_fallback`），同样避免丢信息。
7. **线程状态不缓存**：每个 session 处理前从存储重建织物快照（`load_fabric`）。
   代价是每 session 一次全表读，收益是 harness 每个样本 `vector_store.clear()`
   之后写侧自动归零，不会出现内存态与存储态漂移。
8. **P1 前缀与摘要不可能不一致**：事实的上下文前缀与本 session 实际落库的
   摘要来自同一个函数 `_resolve_summary`（返回"本 session 之后该线程的摘要 +
   是否重写"）。若 `summary_impact == "none"`，前缀取**旧摘要**（因为落库的仍
   是旧摘要），这样 `context_digest` 永远指向真实存在的那段摘要，重嵌入的幂等
   判断才可靠。
9. **重嵌入用原地改向量，而非删旧插新**：设计给 P0 的三能力里"摘要/档案更新 =
   删旧行 + 插新行"，但重嵌入是高频操作（每次摘要重写都可能点名若干事实），
   删插之间进程中断会丢事实。因此后端补了第 4 个能力 `update_vector`
   （只换向量 + `context_digest`，事实文本不动，全文索引因此无需重建）。
10. **重上下文化是选择性的，不是全量重建索引**：只有 Call B 在 `outdated_facts`
   里点名的事实会被重嵌入。实测（6 session 样本）108 条事实里 42 条的指纹等于
   当前活体摘要，其余仍挂在更早版本的摘要上——这是设计的语义触发语义，不是
   缺陷。想要"全量重上下文化"是另一个机制，未实现。
11. **实体档案由确定性代码生成，不额外调 LLM**：设计第 3 节把档案更新列在
   "编织执行（确定性代码，非 LLM）"块内，成本模型（第 9 节）也只算 Call A/B。
   实现为人物级织物摘要：该说话人参与的每个线程一行（标题 + 一句话摘要），
   按线程最近更新时间排序，随摘要演化自动更新。参与关系读自存储数据
   （事实的 `persons`），外加本 session 该说话人实际发言的线程。
   档案不设容量参数（LoCoMo 每对话恰好 2 名说话人）。
   **风险**：档案文本较长（数 KB），其向量是人物级混合，可能在 top-25 里挤掉
   具体事实；用 `ENABLE_ENTITY_PROFILES` 单独消融即可量化。
12. **消融开关比设计多一个**：设计第 8 节的 5 个开关把 P1 的三个机制压在
   一行里。前缀嵌入与实体档案是两个独立机制、贡献可分离，故拆成
   `ENABLE_RECONTEXT`（前缀 + 重嵌入）与 `ENABLE_ENTITY_PROFILES`（档案入池）
   两个开关。两者都为 True 时等于设计描述的 P1 全量行为。
13. **P2 反向边需要一个新的存储原语**：weave 边在 P0 就双向写入，所以顺着
   `links` 用 `get_by_ids` 就够；线程摘要 ID 固定（`thread::<tid>`）也不用查询。
   唯一缺的是"谁 supersede 了我"——`superseded_by` 只在旧条目上单向存在，而设计
   要求 supersede 链**双向**遍历（时间题问"之前是什么"正需要反向）。因此后端补
   `find_by_field(field, values)`。facade 层会丢掉空值：空串代表"没有这条边"，
   拿空串去查会命中所有没有该边的条目（基线数据全中）。
14. **扩展后的条目不再过 as-of 过滤**：as-of 的作用域是检索基座产出的候选池；
   一跳扩展沿 supersede 链把历史**故意**取回来，此时再套一遍有效期过滤就把扩展
   本身抵消了。这带来一个 P0 性质的**有意变更**：非 when 类问题现在也可能看到
   已闭合的事实，但它们带 provenance 进池、并在回答上下文里被标注为
   `[SUPERSEDED on <d> by Context N]`。P0 阶段那条"状态题看不到过期事实"的
   测试因此被限定到"关闭 P2 时"的语义，另有 P2 测试钉住新语义。
15. **精排器放在 `memweaver/` 之外**：设计第 10 节把它标为继承组件、不进贡献
   声明，且 baseline parity 要求所有对照系统都配同一个精排器。因此
   `simplemem/core/reranker.py` 完全不认识织物，`ENABLE_EXPAND_RERANK` 也**不**
   受 `ENABLE_MEMWEAVER` 约束——基线 arm 打开它就是"基线 + 精排"，因为基线数据
   没有织物边可走，扩展自然是空操作。**主表两个 arm 都应打开它**；纯 SimpleMem
   的参照数字是 `--no-memweaver --no-expand-rerank`。
16. **精排不可用时确定性降级**：模型加载失败（无 sentence-transformers、模型主机
   不可达）只报告一次，之后保持检索顺序并仍然截断到 `RERANK_TOP_K`——这样容量
   常数在两条路径下都成立，跑批不会因为环境缺模型而中断。
17. **supersede 链会汇聚，排布要按块**：兜底扫描可以用同一条新事实闭合多条旧
   事实，所以一个后继可能有多个前驱。排布实现为"把在场的前驱（递归）全部排在
   后继之前"，于是整条链是连续块、后继在块尾，标注的 `Context N` 指向真实位置。
   实测样本上 20 条上下文里有 6 个连续块。环形链（写侧不产生）保持原序、不递归
   死循环。
18. **全文索引失效修复**（`vector_store_backend.py`）：LanceDB 的 FTS 索引不随写入
   增量更新，原实现只在首次 insert 时建索引。基线一次性批量写入时无感，但
   MemWeaver 每 session 写一次，会导致只有第 1 个 session 的事实进入词法路。
   改为"写入/更新/删除标脏 → 下次词法检索前重建"，两个 arm 共享此修复，
   parity 不受影响。

## 4. 消融开关

### 4.0 阶段 ↔ 开关的对应关系（含两处不是 1:1 的地方）

| 阶段 | 开关 | 能否单独关掉 |
|---|---|---|
| P0 写侧机制 | `ENABLE_MEMWEAVER`（总）、`ENABLE_WEAVING`、`ENABLE_SWEEP` | 编织与兜底扫描各自独立；P0 本身即总开关 |
| P1 表示协同演化 | `ENABLE_RECONTEXT`、`ENABLE_ENTITY_PROFILES` | 两个机制各自独立 |
| P2 读侧原语 | `ENABLE_EXPAND_RERANK`（总）、`ENABLE_EXPANSION`、`ENABLE_RERANK` | 扩展与精排各自独立 |

两处**结构性**的不可分：

1. **P1 依赖 P0**，不存在"关掉 P0、留着 P1"的配置：上下文继承嵌入的前缀来自活体
   线程摘要，实体档案来自线程归属——没有写侧织物就没有这两样东西的输入。
   `ENABLE_MEMWEAVER=False` 时 P1 的两个开关自然失效。
2. **P2 不依赖 P0/P1**，可以单独开：基线数据上一跳扩展找不到织物边（空操作），
   精排照常工作。这正是 parity 配置（见 4.2）。

`ENABLE_EXPANSION` 与 `ENABLE_RERANK` 之所以必须分开：设计第 10 节把精排器标为
**继承组件、不进贡献声明**，而一跳扩展是 **C3 声明的贡献**。两者共用一个开关就
无法回答"增量来自 C3 还是来自那个通用精排器"——这是审稿人会问的第一个问题。

### 4.1 开关清单

`config.py`（或环境变量）中：

| 开关 | 默认 | 状态 |
|---|---|---|
| `ENABLE_MEMWEAVER` | `False` | P0：总开关（写管线 + as-of）。`False` = 纯 SimpleMem 基线 |
| `ENABLE_WEAVING` | `True` | P0：类型化编织（supersede/refine/bridge）。关掉只保留线程 + 活体摘要 |
| `ENABLE_SWEEP` | `True` | P0：finalize 跨线程 supersede 扫描 |
| `ENABLE_RECONTEXT` | `True` | P1：上下文继承嵌入 + `outdated_facts` 重嵌入。关掉 = 纯 SimpleMem 单句嵌入 |
| `ENABLE_ENTITY_PROFILES` | `True` | P1：实体档案（`profile::<name>`）入池 |
| `ENABLE_EXPAND_RERANK` | `True` | P2 阶段总开关（复合）：关掉则扩展与精排都不做 |
| `ENABLE_EXPANSION` | `True` | P2 之一：一跳扩展 + supersede 链标注 = **贡献 C3**（论文声明的部分） |
| `ENABLE_RERANK` | `True` | P2 之二：交叉编码器 = **继承组件、不进贡献声明**；parity 要求两个 arm 都开 |
| `RERANK_TOP_K` | `20` | 精排后进入回答上下文的条目数（容量常数） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 本地交叉编码器；CPU 上可换 `bge-reranker-base` 降成本 |
| `LLM_TEMPERATURE` | `0.7` | MemWeaver 全部 LLM 调用统一温度 |

启用 MemWeaver 时必须禁用 EvolveMem 的 `time_decay_half_life_days`
（设计第 10.3 节：时间机制互斥）。

### 4.2 主表与 parity

主表两个 arm 都开 `ENABLE_RERANK`（`ENABLE_EXPANSION` 在基线上是空操作）：

| 配置 | 命令 | 用途 |
|---|---|---|
| 基线 + 精排 | `--no-memweaver` | 主表 arm A |
| MemWeaver 全量 | `--memweaver` | 主表 arm B |
| 纯 SimpleMem | `--no-memweaver --no-expand-rerank` | 论文里的"原版 SimpleMem"参照 |
| C3 归因 | `--memweaver --no-expansion` | arm B 减去 C3 读侧原语，精排仍在 |
| 精排贡献 | `--memweaver --no-rerank` | arm B 减去继承组件 |

## 5. 跑 A/B

```bash
python scripts/fetch_locomo10.py                      # -> test_ref/locomo10.json

# Arm A：基线 + 同一个精排器（parity 要求，设计第 10 节）
python test_locomo10.py --no-memweaver --llm-judge --parallel-questions \
    --result-file results/baseline_run1.json

# Arm B：MemWeaver（P0 + P1 + P2）
python test_locomo10.py --memweaver --llm-judge --parallel-questions \
    --result-file results/memweaver_run1.json

# 纯 SimpleMem 参照数字（不配精排器）
python test_locomo10.py --no-memweaver --no-expand-rerank --llm-judge \
    --parallel-questions --result-file results/simplemem_pure.json

# 消融（每项对应论文消融表一行）
python test_locomo10.py --memweaver --no-weaving   --result-file results/mw_no_weaving.json
python test_locomo10.py --memweaver --no-sweep     --result-file results/mw_no_sweep.json
python test_locomo10.py --memweaver --no-recontext --result-file results/mw_no_recontext.json
python test_locomo10.py --memweaver --no-profiles  --result-file results/mw_no_profiles.json
python test_locomo10.py --memweaver --no-expansion --result-file results/mw_no_expansion.json
python test_locomo10.py --memweaver --no-rerank    --result-file results/mw_no_rerank.json
python test_locomo10.py --memweaver --no-expand-rerank --result-file results/mw_no_p2.json

# 对比（温度 0.7 → 每 arm 跑 3 次，脚本按 arm 聚合 mean±std）
python scripts/compare_locomo_results.py \
    results/baseline_run1.json results/baseline_run2.json results/baseline_run3.json \
    --memweaver results/memweaver_run1.json results/memweaver_run2.json results/memweaver_run3.json
```

两个 arm 共享：检索基座配置（`SEMANTIC_TOP_K=25 / KEYWORD_TOP_K=5 /
STRUCTURED_TOP_K=5 / MAX_REFLECTION_ROUNDS=2`）、回答 prompt、cat5 对抗流程、
评测指标。差异只有写管线与 as-of 过滤。

结果 JSON 的 `summary.write_stats` 记录该次运行的写侧健康指标（各类兜底触发
次数、编织计数），对应设计第 6 节"兜底触发率入日志"。

## 6. 检索命中率的定义（evidence 字段）

`locomo_evidence.py`。记忆条目是 LLM 改写句、不带轮次 ID，无法按 ID 对齐
evidence，因此按词法对齐，并按**对话自身的 IDF** 加权：某条 evidence 轮次算
"被检出"，当且仅当某个被检索条目覆盖了该轮次 ≥ `EVIDENCE_HIT_THRESHOLD`
的高区分度内容。IDF 加权是关键——未加权的词重叠被寒暄词主导，任何条目与任何
轮次都有大量重叠。

指标（会自动进入 harness 的按类别聚合，因此可以直接读 cat2 的检索命中率）：

- `retrieval_hit_any` / `retrieval_hit_all`：dia_id 级，命中任一 / 全部 evidence 轮次
- `retrieval_coverage`：dia_id 级平均覆盖率，**无阈值**，最稳的连续量
- `retrieval_session_hit_all`：session 级，把每个被检索条目归因到其最匹配轮次
  所属 session，再问 evidence 涉及的 session 是否都被触达
- `retrieval_evidence_count`：可用 evidence 轮次数（实测均值 1.42）

阈值 0.3 的标定用 LoCoMo 自带的 `observation` 字段（其条目自带来源轮次
指针，是"记忆条目"的现成代理）：引用了某轮次的陈述句达到 0.3 的比例为 49%，
而随机抽取的 25 条未引用陈述句达到 0.3 的比例为 1.6%。因此命中率是
**arm 间的相对量**——完美检索也不会得 1.0，因为忠实的改写句会保留事实、
丢掉寒暄。绝对水平请读 `retrieval_coverage`。

## 7. 写侧健康指标（P1 新增）

`summary.write_stats` 里除 P0 的兜底计数外新增：

| 指标 | 含义 |
|---|---|
| `recontext_reembedded` | 因摘要重写而重嵌入的事实数（本地计算，零 API 成本） |
| `recontext_up_to_date` | 被点名但指纹已是当前上下文、因此跳过的事实数（幂等命中） |
| `profiles_written` | 实际写入的档案行数 |
| `profiles_unchanged` | 该说话人参与的线程本 session 没变、因此跳过重写的次数 |

## 8. 读侧成本（P2 引入，跑之前要知道）

精排是本方案唯一的重本地计算：候选池 = 基座输出（多查询 + 反思，实测 25–100 条）
+ 一跳扩展（实测 +15~20 条），全池逐条过交叉编码器。设计明确要求"扩展后全池
逐条打分"，所以没有截断参数；代价是每题一次 40–150 对的 cross-encoder 前向。

- 4 核 CPU、无 GPU 时 `bge-reranker-v2-m3`（568M）每题约数秒；196 题 × 2 arm
  会显著拉长墙钟时间。降成本的**唯一**旋钮是换小模型
  （`RERANKER_MODEL=BAAI/bge-reranker-base`），它改成本不改方法。
- 环境拿不到模型时自动降级为"保持检索顺序 + 截断到 top-20"，日志里会说一次；
  此时 P2 的贡献只剩一跳扩展。

## 9. 尚未实现

设计第 8 节的三个阶段已全部落地。设计中明确"暂缓/移出"的读侧候选仍未实现，
按设计保持移出：RRF 融合、符号路日期窗口化与 person 过滤禁用、题型格式化
prompt、充分性门控（见设计第 5 节暂缓清单）。第 11 节的 LongMemEval 适配同样
未开始（question_date 锚、`_abs` 弃答、`_m` 规模的摘要预筛）。
