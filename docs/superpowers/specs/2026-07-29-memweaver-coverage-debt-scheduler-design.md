# MemWeaver 覆盖债务调度器设计

**状态**：设计已确认，待实现。

## 目标

在会话事实抽取完成后，识别具有信息量却未写入可检索事实的 dialogue turn，并以小范围补抽恢复遗漏事实。模块服务于写入完整性，不改变现有检索排序、回答提示或事实真值规则。

## 边界

覆盖债务记录不是 `MemoryEntry`，不进入向量库、BM25 索引或回答上下文。只有补抽成功并完成既有字段校验的内容，才以普通事实条目写入 Fabric。

当前实现范围仅覆盖 LoCoMo 文本会话。内容性 turn 的判定、事实与 turn 的映射、补抽状态维护均在 MemWeaver 写入侧完成，不读取 QA 标注或参考答案。

## 数据模型

新增独立的 `CoverageDebt` 记录：

```text
debt_id:             稳定标识
session_id:          来源会话标识
thread_id:           Call A 分配后的所属线程标识
source_turn_ids:     未被事实覆盖的 dialogue 编号
nearby_turn_ids:     用于补抽的相邻上下文编号
entities:            已解析出的实体名
topic:               会话或线程主题
status:              pending | repaired | exempt
attempt_count:       补抽尝试次数
repair_fact_ids:     成功补抽生成的事实条目 ID
created_at:          登记时间
resolved_at:         修复或豁免时间
```

`MemoryEntry` 新增 `source_turn_ids: List[int]`，使每条事实均可追溯到一条或多条原始对话轮。Call B 输出另含 `exempt_turn_ids: List[int]`，只允许标记寒暄、重复确认和无事实内容。覆盖审计以会话全部 turn 减去事实的 `source_turn_ids` 和 `exempt_turn_ids` 得到债务集合；模型解析失败时，全部未映射 turn 均进入债务集合。

## 写入过程

```text
Call A 线程分配
  -> Call B 事实抽取、摘要更新和编织
  -> 事实解析与来源编号校验
  -> 覆盖审计
  -> CoverageDebt 持久化
  -> 延后补抽
  -> 校验后的普通事实写入 Fabric
```

覆盖审计在每个会话的事实写入后运行。对于每个内容性 turn，审计器检查其是否出现在至少一条事实的 `source_turn_ids` 中。缺失映射的 turn 连同相邻上下文、当前已写入事实和识别出的实体组成一笔 `pending` 债务。

模型输出无法解析、字段校验失败或事实数组为空时，相关内容进入债务记录；当前会话仍可继续处理，避免单个格式异常阻塞整段会话。

## 调度规则

初版采用固定规则，避免为补抽引入题目相关调参。

1. 会话结束时登记债务，不在主写入调用中持续重试。
2. 后续会话触及相同 `thread_id` 或存在实体交集时，处理相关 `pending` 债务；每次触发对每笔债务仅执行一次补抽。
3. `finalize()` 对未修复债务执行一次批量补抽。
4. 补抽输入仅含债务 turn、相邻 turn、相关已写入事实和来源会话日期，不输入完整线程历史。
5. 补抽结果通过现有事实 schema、时间规范化和来源编号校验后，方可写入 Fabric。
6. 多次补抽均失败的内容继续保留为 `pending`，供审计统计使用；只有 Call B 明确列入 `exempt_turn_ids` 的内容才可标记为 `exempt`。

## 与现有 MemWeaver 的关系

模块位于 `MemWeaver._process_session()` 中 Call B 更新应用之后，并通过独立存储维护债务记录。现有线程分配、事实编织、摘要更新、有效期处理、一跳扩展和交叉编码器重排保持原有语义。

补抽成功产生的事实与普通 Call B 事实使用相同的 `kind`、`thread_id`、`valid_from`、`links` 和向量写入流程。债务文本本身没有向量表示，也不会被检索器返回。

## 失败处理与可观测性

记录以下健康指标：

```text
contentful_turns
covered_turns
pending_debts
repaired_debts
exempt_turns
repair_attempts
repair_parse_failures
```

补抽请求失败时保留债务，不覆盖已有事实。补抽产生与既有事实语义重复的内容时，复用现有去重和编织规则。解析失败日志仅记录债务标识、会话标识和错误类别，避免输出 API 凭据或完整敏感上下文。

## 验证

离线单元测试覆盖：

1. 内容性 turn 缺少来源映射时创建债务。
2. 寒暄或重复确认被标为 `exempt`。
3. 补抽成功后写入普通事实并清除 `pending` 状态。
4. 补抽解析失败后债务仍可追踪，既有事实保持不变。
5. 债务记录不会进入向量检索或回答上下文。

Sample9 对比应分别记录事实溯源覆盖率、债务修复率、五类 HitAny、HitAll 和端到端 F1、BLEU、LLM Judge。端到端指标与检索覆盖分开报告。

## 范围外内容

本设计不包含 QA 标注驱动的补抽、全局图检索、答案格式化、摘要重上下文化和检索参数搜索。这些功能保持为独立研究方向，避免混入本模块的实验归因。
