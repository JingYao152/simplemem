# MemWeaver P0 实现说明

对应 `docs/memweaver-design.md` 第 8 节 P0 阶段：数据模型 + 后端三能力 +
Call A/B 写管线 + supersede 执行 + as-of 检索 + finalize 兜底扫描。
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
| 检索侧 as-of 叠加 | `simplemem/core/hybrid_retriever.py` |
| 消融开关 / `LLM_TEMPERATURE` | `simplemem/core/settings.py`、`simplemem/core/config_default.py` |
| 系统装配（写路由 + finalize） | `main.py` |
| 评测：A/B 同 harness + evidence 检索命中率 | `test_locomo10.py`、`locomo_evidence.py` |

单元测试：`tests/test_memweaver.py`（45 例，全部 LLM 调用脚本化，覆盖每个
决策点的确定性兜底）、`tests/test_evidence_retrieval.py`（12 例）。

## 2. 参数账本（与设计一致）

新增**调优参数 0 个**。代码中出现的常数都是结构常数或评测常数：

| 常数 | 值 | 性质 |
|---|---|---|
| `SWEEP_NEIGHBOURS` | 3 | 设计第 3 节"top-3 跨线程近邻"，结构常数 |
| `SWEEP_SCAN_DEPTH` | 12 | 为拿到 3 个跨线程近邻而多取的行数，纯实现细节 |
| `SWEEP_BATCH_SIZE` | 20 | 判定 prompt 的分批大小；所有候选对都会被判定，不改变任何决策 |
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
8. **全文索引失效修复**（`vector_store_backend.py`）：LanceDB 的 FTS 索引不随写入
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
| `ENABLE_RECONTEXT` | `False` | **P1 占位**，当前无实现 |
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

# Arm B：MemWeaver P0
python test_locomo10.py --memweaver --llm-judge --parallel-questions \
    --result-file results/memweaver_run1.json

# 消融
python test_locomo10.py --memweaver --no-weaving --result-file results/mw_no_weaving.json
python test_locomo10.py --memweaver --no-sweep    --result-file results/mw_no_sweep.json

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

## 7. 尚未实现（按设计留给后续阶段）

- P1：上下文继承嵌入（`context_digest` 字段已在数据模型里预留）、
  `outdated_facts` 触发的重嵌入（P0 只把该信号计数进 `outdated_signals`）、
  实体档案生成与入池；
- P2：一跳证据扩展、交叉编码器平铺精排（`RERANK_TOP_K=20`）、
  supersede 链在回答上下文中的相邻排布与 `[SUPERSEDED on <d> by Context N]` 标注。
