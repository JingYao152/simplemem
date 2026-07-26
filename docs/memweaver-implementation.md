# MemWeaver 实现说明（P0 + P1）

对应 `docs/memweaver-design.md` 第 8 节的前两个阶段：

- **P0**：数据模型 + 后端三能力 + Call A/B 写管线 + supersede 执行 + as-of 检索
  + finalize 兜底扫描（目标类别：cat2 时间题 321 题）
- **P1**：上下文继承嵌入 + `outdated_facts` 重嵌入 + 实体档案入池
  （目标类别：cat1 单跳 282 题、cat4 开放域 841 题）

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
| 检索侧 as-of 叠加 | `simplemem/core/hybrid_retriever.py` |
| 消融开关 / `LLM_TEMPERATURE` | `simplemem/core/settings.py`、`simplemem/core/config_default.py` |
| 系统装配（写路由 + finalize） | `main.py` |
| 评测：A/B 同 harness + evidence 检索命中率 | `test_locomo10.py`、`locomo_evidence.py` |

单元测试：`tests/test_memweaver.py`（63 例，全部 LLM 调用脚本化，覆盖每个
决策点的确定性兜底、P1 的前缀不落库/指纹幂等/档案重写）、
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
13. **全文索引失效修复**（`vector_store_backend.py`）：LanceDB 的 FTS 索引不随写入
   增量更新，原实现只在首次 insert 时建索引。基线一次性批量写入时无感，但
   MemWeaver 每 session 写一次，会导致只有第 1 个 session 的事实进入词法路。
   改为"写入/更新/删除标脏 → 下次词法检索前重建"，两个 arm 共享此修复，
   parity 不受影响。

## 4. 消融开关

`config.py`（或环境变量）中：

| 开关 | 默认 | 状态 |
|---|---|---|
| `ENABLE_MEMWEAVER` | `False` | P0：总开关（写管线 + as-of）。`False` = 纯 SimpleMem 基线 |
| `ENABLE_WEAVING` | `True` | P0：类型化编织（supersede/refine/bridge）。关掉只保留线程 + 活体摘要 |
| `ENABLE_SWEEP` | `True` | P0：finalize 跨线程 supersede 扫描 |
| `ENABLE_RECONTEXT` | `True` | P1：上下文继承嵌入 + `outdated_facts` 重嵌入。关掉 = 纯 SimpleMem 单句嵌入 |
| `ENABLE_ENTITY_PROFILES` | `True` | P1：实体档案（`profile::<name>`）入池 |
| `ENABLE_EXPAND_RERANK` | `False` | **P2 占位**，当前无实现 |
| `LLM_TEMPERATURE` | `0.7` | MemWeaver 全部 LLM 调用统一温度 |

启用 MemWeaver 时必须禁用 EvolveMem 的 `time_decay_half_life_days`
（设计第 10.3 节：时间机制互斥）。

## 5. 跑 A/B

```bash
python scripts/fetch_locomo10.py                      # -> test_ref/locomo10.json

# Arm A：纯 SimpleMem 基线
python test_locomo10.py --no-memweaver --llm-judge --parallel-questions \
    --result-file results/baseline_run1.json

# Arm B：MemWeaver（P0 + P1）
python test_locomo10.py --memweaver --llm-judge --parallel-questions \
    --result-file results/memweaver_run1.json

# 消融（每项对应论文消融表一行）
python test_locomo10.py --memweaver --no-weaving   --result-file results/mw_no_weaving.json
python test_locomo10.py --memweaver --no-sweep     --result-file results/mw_no_sweep.json
python test_locomo10.py --memweaver --no-recontext --result-file results/mw_no_recontext.json
python test_locomo10.py --memweaver --no-profiles  --result-file results/mw_no_profiles.json

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

## 8. 尚未实现（按设计留给 P2）

- 一跳证据扩展（supersede 链双向、weave 边双向、结构归属边）；
- 交叉编码器平铺精排（`RERANK_TOP_K=20`）与扩展条目的 provenance 前缀打分；
- supersede 链在回答上下文中的相邻排布与 `[SUPERSEDED on <d> by Context N]` 标注。
