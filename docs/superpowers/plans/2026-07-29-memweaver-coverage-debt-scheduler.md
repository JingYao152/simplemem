# MemWeaver 覆盖债务调度器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MemWeaver 会话写入后登记未被事实覆盖的内容性 dialogue turn，并在后续相关会话或 `finalize()` 中以小范围补抽恢复事实。

**Architecture:** `MemoryEntry.source_turn_ids` 将事实与原始 dialogue 建立持久映射。新的 `CoverageDebtStore` 以与 LanceDB 库同目录的原子 JSON 文件保存待补抽记录，债务记录不进入向量库。`MemWeaver` 在 Call B 应用后审计覆盖情况，并在后续相关会话及 `finalize()` 调用小范围修复提示；修复结果通过现有向量库作为普通事实写入。

**Tech Stack:** Python 3.10、Pydantic、LanceDB、pytest、现有 `LLMClient` 与 `VectorStore`。

## Global Constraints

`CoverageDebt` 不是 `MemoryEntry`，不得写入向量库、BM25 索引或回答上下文。

事实条目必须携带 `source_turn_ids`，且每个编号属于其 Call B 分配到的 turn 集合。

`exempt_turn_ids` 仅标记寒暄、重复确认和无事实内容；格式异常或解析失败产生 `pending` 债务，不能自动豁免。

补抽输入只能包含债务 turn、相邻 turn、来源会话日期和该线程已写入的相关事实；不得输入完整会话历史或 QA 标注。

默认开关为关闭，既有 MemWeaver 运行语义与测试保持不变；Sample9 专项实验通过环境变量显式开启。

所有测试离线运行，使用现有 `ScriptedLLM` 与确定性嵌入器；不得调用外部 API。

---

### Task 1: 事实来源编号的模型与向量持久化

**Files:**

- Modify: `simplemem/core/models/memory_entry.py`
- Modify: `simplemem/core/database/vector_store.py`
- Modify: `simplemem/core/database/vector_store_backend.py`
- Modify: `tests/test_memweaver.py`

**Interfaces:**

- Produces: `MemoryEntry.source_turn_ids: List[int]`
- Produces: `VectorStore.METADATA_FIELDS`、元数据序列化与反序列化包含 `source_turn_ids`
- Produces: LanceDB 表新增 `source_turn_ids: list<int64>`，旧表迁移时以空列表填充

- [ ] **Step 1: Write the failing source-persistence test**

```python
def test_source_turn_ids_roundtrip_through_vector_store(store):
    entry = MemoryEntry(
        entry_id="source-fact",
        lossless_restatement="Alice drinks oat milk coffee.",
        source_turn_ids=[7, 9],
    )

    store.add_entries([entry])

    restored = store.get_by_ids(["source-fact"])[0]
    assert restored.source_turn_ids == [7, 9]
```

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run: `python -m pytest tests/test_memweaver.py::test_source_turn_ids_roundtrip_through_vector_store -q`

Expected: FAIL because `MemoryEntry` has no `source_turn_ids` field or the field is discarded by storage.

- [ ] **Step 3: Add the field and persistence mapping**

```python
# MemoryEntry
source_turn_ids: List[int] = Field(default_factory=list)

# VectorStore metadata
"source_turn_ids": entry.source_turn_ids,

# Restored entry
source_turn_ids=list(metadata.get("source_turn_ids") or []),
```

Add the corresponding `pa.field("source_turn_ids", pa.list_(pa.int64()))` to the LanceDB schema. In `_migrate_table`, add an empty list default for this list column so pre-existing rows remain readable.

- [ ] **Step 4: Run the focused test and the storage subset**

Run: `python -m pytest tests/test_memweaver.py::test_source_turn_ids_roundtrip_through_vector_store tests/test_memweaver.py::test_lancedb_table_migrates_pre_memweaver_schema -q`

Expected: PASS.

- [ ] **Step 5: Commit the source-provenance task**

```bash
git add simplemem/core/models/memory_entry.py simplemem/core/database/vector_store.py simplemem/core/database/vector_store_backend.py tests/test_memweaver.py
git commit -m "feat(memweaver): persist fact source turn ids"
```

### Task 2: 覆盖债务记录与持久化存储

**Files:**

- Create: `simplemem/core/memweaver/coverage.py`
- Modify: `tests/test_memweaver.py`

**Interfaces:**

- Produces: `CoverageDebt` dataclass with `debt_id`, `session_id`, `thread_id`, `source_turns`, `nearby_turns`, `entities`, `topic`, `status`, `attempt_count`, `repair_fact_ids`, `created_at`, and `resolved_at`
- Produces: `CoverageDebtStore(path: str)` with `record`, `pending`, `pending_related`, `mark_repaired`, `increment_attempt`, and `mark_exempt`
- Produces: `audit_turn_coverage(turns, facts, exempt_turn_ids) -> List[Dialogue]`

- [ ] **Step 1: Write the failing audit and persistence tests**

```python
def test_coverage_audit_returns_only_unmapped_non_exempt_turns():
    source = turns("1:00 pm on 1 May, 2023", "coffee", "okay", start=11)
    fact_entry = MemoryEntry(
        lossless_restatement="Alice drinks coffee.",
        source_turn_ids=[11],
    )

    uncovered = audit_turn_coverage(source, [fact_entry], exempt_turn_ids=[12])

    assert uncovered == []


def test_debt_store_survives_reopen(tmp_path):
    path = tmp_path / "coverage-debts.json"
    store = CoverageDebtStore(str(path))
    debt = store.record(
        session_id="2023-05-01",
        thread_id="t1",
        source_turns=turns("2023-05-01T13:00:00", "coffee", start=11),
        nearby_turns=[],
        entities=["Alice"],
        topic="Coffee",
    )

    reopened = CoverageDebtStore(str(path))
    assert reopened.pending()[0].debt_id == debt.debt_id
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run: `python -m pytest tests/test_memweaver.py::test_coverage_audit_returns_only_unmapped_non_exempt_turns tests/test_memweaver.py::test_debt_store_survives_reopen -q`

Expected: FAIL because `coverage.py` and its public interfaces do not exist.

- [ ] **Step 3: Implement the isolated coverage module**

```python
@dataclass
class CoverageDebt:
    debt_id: str
    session_id: str
    thread_id: str
    source_turns: List[Dialogue]
    nearby_turns: List[Dialogue]
    entities: List[str]
    topic: str
    status: str = "pending"
    attempt_count: int = 0
    repair_fact_ids: List[str] = field(default_factory=list)
```

Serialize dialogue fields explicitly to JSON. Use a temporary sibling file and `Path.replace()` for atomic writes. Compute debt identity from session, thread and source dialogue identifiers so repeated audits update one record rather than creating duplicates.

- [ ] **Step 4: Run the focused tests and the full coverage-module test selection**

Run: `python -m pytest tests/test_memweaver.py -q -k "coverage or debt or source_turn_ids"`

Expected: PASS.

- [ ] **Step 5: Commit the debt-storage task**

```bash
git add simplemem/core/memweaver/coverage.py tests/test_memweaver.py
git commit -m "feat(memweaver): track persistent coverage debts"
```

### Task 3: Call B 来源映射、会话审计与延后补抽

**Files:**

- Modify: `simplemem/core/memweaver/prompts.py`
- Modify: `simplemem/core/memweaver/writer.py`
- Modify: `simplemem/core/settings.py`
- Modify: `tests/test_memweaver.py`

**Interfaces:**

- Produces: `ThreadUpdate.exempt_turn_ids: List[int]`
- Produces: the `MemWeaver` constructor accepts `enable_coverage_debt_scheduler: Optional[bool] = None` and `coverage_debt_store: Optional[CoverageDebtStore] = None` after its existing parameters
- Produces: `ENABLE_COVERAGE_DEBT_SCHEDULER=False`
- Produces: `MemWeaver._audit_coverage(assignments: Sequence[ThreadAssignment], updates: Sequence[ThreadUpdate], session_date: str) -> None`
- Produces: `MemWeaver._repair_debts(debts: Sequence[CoverageDebt], session_date: str) -> None`
- Produces: `MemWeaver._repair_debt(debt: CoverageDebt, session_date: str) -> None`

- [ ] **Step 1: Write the failing integration tests**

```python
def repair_update(facts, exempt_turn_ids=None):
    return json.dumps({
        "facts": facts,
        "exempt_turn_ids": exempt_turn_ids or [],
    })


# Extend ScriptedLLM test support before executing the tests.
# Its scripts and prompts dictionaries gain a "repair" queue, and _classify()
# returns "repair" when the prompt contains "[Coverage debt]".

def test_coverage_scheduler_records_unmapped_turn_after_call_b(store, tmp_path):
    llm = ScriptedLLM(
        assignment=[assignment(([1, 2], "new:Coffee"))],
        thread_update=[thread_update([
            fact("Alice drinks coffee.", source_turn_ids=[1]),
        ])],
    )
    debt_store = CoverageDebtStore(str(tmp_path / "debts.json"))
    weaver = build_weaver(
        store, llm,
        enable_coverage_debt_scheduler=True,
        coverage_debt_store=debt_store,
    )

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee", "oat milk"))
    weaver.process_remaining()

    debt = debt_store.pending()[0]
    assert [turn.dialogue_id for turn in debt.source_turns] == [2]
    assert weaver.stats["coverage_debts_created"] == 1


def test_related_later_session_repairs_existing_debt(store, tmp_path):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1, 2], "new:Coffee")),
            assignment(([3], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee.", source_turn_ids=[1])]),
            thread_update([fact("Alice visits a cafe.", source_turn_ids=[3])]),
        ],
        repair=[repair_update([
            fact("Alice prefers oat milk.", source_turn_ids=[2]),
        ])],
    )
    debt_store = CoverageDebtStore(str(tmp_path / "debts.json"))
    weaver = build_weaver(
        store, llm,
        enable_coverage_debt_scheduler=True,
        coverage_debt_store=debt_store,
    )

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee", "oat milk"))
    weaver.add_dialogues(turns("1:00 pm on 8 May, 2023", "cafe", start=3))
    weaver.process_remaining()

    assert by_text(store, "Alice prefers oat milk.").source_turn_ids == [2]
    assert debt_store.pending() == []
```

- [ ] **Step 2: Run the first integration test and verify the expected failure**

Run: `python -m pytest tests/test_memweaver.py::test_coverage_scheduler_records_unmapped_turn_after_call_b -q`

Expected: FAIL because Call B facts do not parse `source_turn_ids` and the scheduler interfaces do not exist.

- [ ] **Step 3: Extend Call B parsing and prompts**

Add `source_turn_ids` to every Call B fact JSON object and add a top-level `exempt_turn_ids` array. Render Call B turn labels from `Dialogue.dialogue_id`, validate all returned identifiers against the assigned turn set, and ignore invalid identifiers. Keep Call A numbering unchanged.

Add `build_coverage_repair_prompt()` that renders only the debt source turns, nearby turns, session date and selected thread facts. Its output format contains `facts` and `exempt_turn_ids`; no summary rewrite or weave judgement is requested.

- [ ] **Step 4: Implement scheduler hooks and repair writes**

At the start of `_process_session`, select debts related to the current assignments. After `_apply_session`, audit current `ThreadUpdate` facts and record uncovered turn groups. Then repair only the debts selected before the current audit, so a newly created debt never blocks its source session.

Repair facts must retain the debt thread id, use the repair source turn identifiers, receive the current thread context embedding prefix, and enter `VectorStore.add_entries()` as ordinary facts. A successful repair calls `mark_repaired`; an exception calls `increment_attempt`. `finalize()` repairs all remaining pending debts once before the sweep.

- [ ] **Step 5: Run the scheduler integration selection**

Run: `python -m pytest tests/test_memweaver.py -q -k "coverage_scheduler or repairs_existing_debt or source_turn_ids"`

Expected: PASS.

- [ ] **Step 6: Commit the scheduler task**

```bash
git add simplemem/core/memweaver/prompts.py simplemem/core/memweaver/writer.py simplemem/core/settings.py tests/test_memweaver.py
git commit -m "feat(memweaver): repair uncovered conversation turns"
```

### Task 4: 配置、文档与回归验证

**Files:**

- Modify: `docs/memweaver-design.md`
- Modify: `docs/superpowers/specs/2026-07-29-memweaver-coverage-debt-scheduler-design.md`
- Test: `tests/test_memweaver.py`

**Interfaces:**

- Produces: documented environment switch `ENABLE_COVERAGE_DEBT_SCHEDULER`
- Produces: specification fields that include persisted source and nearby dialogue content required for later repair

- [ ] **Step 1: Write the failing default-off regression test**

```python
def test_coverage_scheduler_is_disabled_by_default(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Coffee"))],
        thread_update=[thread_update([fact("Alice drinks coffee.")])],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.process_remaining()

    assert weaver.coverage_debt_store is None
    assert "coverage_debts_created" not in weaver.stats
```

- [ ] **Step 2: Run the test and verify the expected failure before the default-off implementation exists**

Run: `python -m pytest tests/test_memweaver.py::test_coverage_scheduler_is_disabled_by_default -q`

Expected: FAIL because `coverage_debt_store` does not exist on `MemWeaver`.

- [ ] **Step 3: Finalize default-off behavior and documentation**

Set the settings default to `False`. Document the write-side position, debt-file location, repair trigger rules, source turn persistence and the fact that debts never enter retrieval. Update the specification to list serialized source and nearby dialogue content as persistence requirements.

- [ ] **Step 4: Run focused and complete regression tests**

Run: `python -m pytest tests/test_memweaver.py -q`

Expected: PASS.

Run: `python -m pytest tests/test_api_key_rotation.py tests/test_memweaver.py -q`

Expected: PASS.

- [ ] **Step 5: Commit documentation and verification task**

```bash
git add docs/memweaver-design.md docs/superpowers/specs/2026-07-29-memweaver-coverage-debt-scheduler-design.md tests/test_memweaver.py
git commit -m "docs(memweaver): document coverage debt scheduling"
```
