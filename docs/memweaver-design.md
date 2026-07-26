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

  ── Call A: 线程分配（1 次 LLM 调用，temperature=0）──
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
                    weave: {op: none|supersede|refine|bridge, target: 候选编号}}],
           summary: "重写后的线程摘要",
           summary_impact: "none" | "minor" | "major",
           outdated_facts: [候选编号...]      // P1 重上下文化的语义触发信号
         }

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

- 插入时嵌入文本 = `"[<线程摘要一句话版>] <lossless_restatement>"`。
  **上下文只进向量，不改文本字段**——回答端看到的仍是干净事实。
- 重上下文化触发 = Call B 的 `outdated_facts` 显式点名（语义定位，
  无漂移阈值），仅对被点名条目用新摘要重拼、重嵌入（本地计算，零 API 成本）。

## 5. 读取管线

```text
anchor = 记忆库最大 session 日期        // LoCoMo 无 question_date → 结构规则
qtype  = 表面形式分类（when 类 / 计数 / yes-no / either-or / 多项 / 默认）

三路检索（沿用现有多查询规划）:
  语义:  SEMANTIC_TOP_K=25；池含 fact + thread_summary + entity_profile 三粒度
         非 when 类问题加 as-of 过滤: valid_until == "" OR valid_until >= anchor
         when 类问题不过滤（问历史需要旧事实）
  词法:  BM25，KEYWORD_TOP_K=5
  符号:  仅时间/实体条件，返回日期窗口内全量（结构有界，无 top-k）
         person 过滤禁用 ← 2 人对话上是退化操作（几乎全库命中）

→ RRF 融合（RRF_K=60；截断 CONTEXT_TOP_K=30）
→ 束扩展: 每个 fact 候选 + 其线程摘要 + links/superseded_by 完整一跳闭包
          （evidence 实测均值 1.42 → 链接度数结构性低，无截断参数）
→ [P2] 交叉编码器束重排（BAAI/bge-reranker-v2-m3，本地推理）
→ 充分性门控喂入: 按重排分逐批喂给 LLM，判"证据已足够"即停（硬上限 12 束）
   同时是 cat5 对抗题的弃答依据: 门控判不充分 → 允许 "Not mentioned"

生成: 题型感知格式化（计数拼写形式 / 人类可读日期 / 裸 Yes-No / 选项文本 /
      多项逗号列表）；temporal 题渲染 supersede 链:
      "[SUPERSEDED on <d> by Context N]"（时间题常问"之前是什么"）
```

## 6. LLM 决策点清单（替代超参数的全部位置）

| 决策 | 所在调用 | 替代的原参数 | 确定性兜底 |
|---|---|---|---|
| 对话轮归属哪个线程 | Call A | THREAD_ATTACH_THRESHOLD | 整 session 入新线程 |
| 摘要是否需要重写 | Call B summary_impact | SUMMARY_REWRITE_MIN_NEW_FACTS | 不重写 |
| 哪些旧事实过期需重上下文化 | Call B outdated_facts | RECONTEXT_DRIFT_THRESHOLD | 不重嵌 |
| 新旧事实的编织关系 | Call B weave | WEAVE_CANDIDATE_TOP_K（候选=同线程全量） | op=none |
| 检索何时停止 | 充分性门控 | RERANK_TOP_K（调优义务 → 硬上限） | 喂满上限 |

全部决策调用 temperature=0；每类兜底触发率记录为系统健康指标。

## 7. 参数账本

**调优参数：0 个。** 容量常数 5 个（全部 fixed、不调优）：

| 常数 | 值 | 性质 |
|---|---|---|
| RRF_K | 60 | 文献标准值 |
| SEMANTIC_TOP_K | 25 | 容量（约为单对话记忆库的 ~10%） |
| KEYWORD_TOP_K | 5 | 容量 |
| CONTEXT_TOP_K | 30 | 容量 |
| 门控硬上限 | 12 束 | 安全上限，非调优目标 |

被数据结构消灭的参数：THREAD_FLUSH_SIZE（session 边界）、线程候选池预筛
（线程数结构有界）、WEAVE_CANDIDATE_TOP_K（同线程全量）、BUNDLE_MAX_NEIGHBORS
（一跳闭包）、STRUCTURED_TOP_K（日期窗口有界）。

## 8. 阶段划分与 LoCoMo 验证映射

| 阶段 | 内容 | 主要受益类别（实测题数） |
|---|---|---|
| P0 | 数据模型 + 后端三能力 + Call A/B 写管线 + supersede + as-of 检索 + 兜底扫描 | cat2 时间题(321)、cat5 对抗(446，经门控) |
| P1 | 上下文继承嵌入 + outdated_facts 重嵌入 + 实体档案入池 | cat1 单跳(282)、cat4 开放域(841) |
| P2 | RRF + 束扩展 + 交叉编码器重排 + 充分性门控 | cat3 多跳(96) + 全局 |

验证方法：每阶段同 harness A/B（test_locomo10.py）；另用 QA 的 evidence 字段
直接量**检索命中率**（session/dia_id 级），不必等端到端分数。

消融开关（config，每项对应论文消融表一行）：
ENABLE_MEMWEAVER / ENABLE_WEAVING / ENABLE_SWEEP / ENABLE_RECONTEXT /
ENABLE_BUNDLE_RERANK / ENABLE_SUFFICIENCY_GATE。

## 9. 成本与风险

- **写时调用量**：每对话 ≈ 27 session × (1 + 平均 2–3 个被触及线程) ≈ 80–110 次
  LLM 调用，对比基线固定窗口 ≈ 15 次（约 6 倍）。换取读时上下文更小更准。
  论文核心效率图：写读合计的 accuracy-per-token Pareto 曲线。
- **风险与对策**：
  1. Call A 分配质量是全楼地基 → 实现后先在 1–2 个样本上人工审计线程划分；
  2. 跨线程 supersede 漏检 → 兜底扫描独立消融开关，单独量化贡献；
  3. LLM 决策随机性 → temperature=0 + 报 3 次运行方差；
  4. 解析失败 → 每个决策点有确定性兜底，兜底率入日志。

## 10. LongMemEval 适配备注（暂缓，规则已定）

- anchor 改用数据自带 question_date；`_abs` 弃答题由充分性门控天然处理；
- `_m`（~500 session）规模下线程摘要全量入 prompt 会爆预算 → 启用嵌入预筛
  （scaling rule：摘要总量超 prompt 预算才启用，预筛量为容量常数）；
- 跨线程兜底扫描在 knowledge-update 上从"保险"升级为"必要组件"。
