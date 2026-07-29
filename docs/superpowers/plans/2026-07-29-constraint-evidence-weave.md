# Constraint Evidence Weave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently controlled read-side selector that preserves distinct, source-backed evidence for multi-requirement questions within the existing answer-context budget.

**Architecture:** The selector operates after candidate retrieval, one-hop expansion, and all-candidate cross-encoder scoring. It activates only for plans with at least two non-low-priority requirements and at least two planned searches. It assigns distinct fact entries to requirements, retains a typed bridge/refine anchor with its selected fact, and fills the remaining fixed budget in global rerank order. It adds no LLM call, no memory-store field, and no use of LoCoMo gold evidence.

**Tech Stack:** Python 3.10, Pydantic models, existing `CrossEncoderReranker`, pytest.

## Global Constraints

`ENABLE_CONSTRAINT_EVIDENCE_WEAVE` defaults to `False`.

The feature is available only when MemWeaver and reranking are active; when one-hop expansion is also active, typed bridge/refine anchors are preserved with their primary fact.

The selected answer context never exceeds `RERANK_TOP_K`.

When disabled, when the plan is single-requirement, or when reranker scoring is unavailable, retrieval returns the pre-feature flat rerank order.

The selector uses only current query plans, candidate facts, fabric provenance, and existing fact metadata.

---

### Task 1: Define the proof-weave selector with failing tests

**Files:**

- Create: `simplemem/core/memweaver/proof_weave.py`
- Create: `tests/test_constraint_evidence_weave.py`

**Interfaces:**

- Produces: `plan_requires_evidence_weave(information_plan: Mapping) -> bool`
- Produces: `select_constraint_evidence_weave(required_info, ranked_entries, reranker, to_text, provenance, top_k) -> List[MemoryEntry]`

- [x] **Step 1: Write failing tests**

```python
def test_constraint_evidence_weave_assigns_distinct_facts_to_requirements():
    selected = select_constraint_evidence_weave(
        required_info=[
            {"priority": "high", "info_type": "place", "description": "San Francisco"},
            {"priority": "high", "info_type": "event", "description": "car workshop"},
        ],
        ranked_entries=[generic, place_fact, event_fact],
        reranker=keyword_reranker,
        to_text=lambda entry: entry.lossless_restatement,
        provenance={},
        top_k=2,
    )
    assert [entry.entry_id for entry in selected] == [place_fact.entry_id, event_fact.entry_id]
```

```python
def test_constraint_evidence_weave_keeps_a_typed_anchor_next_to_primary():
    selected = select_constraint_evidence_weave(
        required_info=[first_requirement, second_requirement],
        ranked_entries=[primary, anchor, other_primary],
        reranker=keyword_reranker,
        to_text=lambda entry: entry.lossless_restatement,
        provenance={primary.entry_id: bridge_provenance(anchor)},
        top_k=3,
    )
    assert [entry.entry_id for entry in selected] == [primary.entry_id, anchor.entry_id, other_primary.entry_id]
```

```python
def test_plan_requires_evidence_weave_only_for_multi_requirement_plans():
    assert plan_requires_evidence_weave({
        "minimal_queries_needed": 2,
        "required_info": [high_requirement, medium_requirement],
    })
    assert not plan_requires_evidence_weave({
        "minimal_queries_needed": 1,
        "required_info": [high_requirement],
    })
```

- [x] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_constraint_evidence_weave.py -q`

Expected: collection failure because `proof_weave.py` is absent.

- [x] **Step 3: Implement the selector**

Implement `plan_requires_evidence_weave` using non-low-priority requirement count and `minimal_queries_needed`. Implement distinct greedy assignment over requirement-local cross-encoder scores. Preserve a primary fact's `bridge` or `refine` provenance anchor immediately after the primary. Fill unused capacity in incoming rank order. Return incoming rank order when local scoring is unavailable.

- [x] **Step 4: Run selector tests**

Run: `python -m pytest tests/test_constraint_evidence_weave.py -q`

Expected: all tests pass.

### Task 2: Add the independent runtime setting and retriever integration

**Files:**

- Modify: `simplemem/core/settings.py`
- Modify: `simplemem/core/hybrid_retriever.py`
- Modify: `tests/test_constraint_evidence_weave.py`

**Interfaces:**

- Consumes: `ENABLE_CONSTRAINT_EVIDENCE_WEAVE: bool`
- Consumes: `select_constraint_evidence_weave(...)`
- Produces: `HybridRetriever.enable_constraint_evidence_weave: bool`

- [x] **Step 1: Write failing integration tests**

```python
def test_retriever_applies_constraint_evidence_weave_only_when_enabled():
    enabled = make_retriever(enable_constraint_evidence_weave=True)
    disabled = make_retriever(enable_constraint_evidence_weave=False)
    plan = {"minimal_queries_needed": 2, "required_info": [high_requirement, medium_requirement]}

    assert entry_ids(enabled._expand_and_rerank("question", entries, information_plan=plan)) == expected_weave_ids
    assert entry_ids(disabled._expand_and_rerank("question", entries, information_plan=plan)) == expected_flat_ids
```

- [x] **Step 2: Run the integration test to verify failure**

Run: `python -m pytest tests/test_constraint_evidence_weave.py::test_retriever_applies_constraint_evidence_weave_only_when_enabled -q`

Expected: constructor rejects `enable_constraint_evidence_weave`.

- [x] **Step 3: Implement the runtime switch**

Add `ENABLE_CONSTRAINT_EVIDENCE_WEAVE = False` to settings. Add an optional constructor override to `HybridRetriever`; require MemWeaver plus reranking. In `_expand_and_rerank`, select the proof weave only when the setting is enabled, reranking succeeded, and the plan passes `plan_requires_evidence_weave`. Preserve existing requirement-bundle behavior when the new switch is disabled.

- [x] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_constraint_evidence_weave.py tests/test_requirement_bundles.py -q`

Expected: all tests pass.

### Task 3: Verify regressions and document the switch

**Files:**

- Modify: `docs/memweaver-design.md`
- Test: `tests/test_memweaver.py`

- [x] **Step 1: Document the read-side condition and safety boundaries**

Document that the selector is query-time only, accepts only multiple non-low-priority requirements, preserves the fixed answer budget, and uses `source_turn_ids` solely as provenance already attached to facts.

- [x] **Step 2: Run the regression suite**

Run: `python -m pytest tests/test_constraint_evidence_weave.py tests/test_requirement_bundles.py tests/test_memweaver.py -q`

Expected: all tests pass.

- [x] **Step 3: Commit the feature**

```bash
git add simplemem/core/memweaver/proof_weave.py simplemem/core/settings.py simplemem/core/hybrid_retriever.py tests/test_constraint_evidence_weave.py docs/memweaver-design.md docs/superpowers/plans/2026-07-29-constraint-evidence-weave.md
git commit -m "feat(memweaver): add constraint evidence weave"
```
