from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.memweaver.expansion import EDGE_BRIDGE, Provenance
from simplemem.core.models.memory_entry import KIND_FACT, MemoryEntry
from simplemem.core.reranker import CrossEncoderReranker


class _KeywordCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, document in pairs:
            query_tokens = set(query.lower().split())
            document_tokens = set(document.lower().split())
            scores.append(len(query_tokens & document_tokens))
        return scores


def _reranker(top_k=3):
    return CrossEncoderReranker(
        top_k=top_k,
        model_factory=lambda _: _KeywordCrossEncoder(),
    )


def _fact(entry_id, text):
    return MemoryEntry(
        entry_id=entry_id,
        lossless_restatement=text,
        kind=KIND_FACT,
    )


def _multi_requirement_plan():
    return {
        "minimal_queries_needed": 2,
        "required_info": [
            {
                "priority": "high",
                "info_type": "location",
                "description": "San Francisco",
            },
            {
                "priority": "medium",
                "info_type": "event",
                "description": "car workshop",
            },
        ],
    }


def test_plan_requires_evidence_weave_only_for_multi_requirement_plans():
    from simplemem.core.memweaver.proof_weave import plan_requires_evidence_weave

    assert plan_requires_evidence_weave(_multi_requirement_plan())
    assert not plan_requires_evidence_weave(
        {
            "minimal_queries_needed": 1,
            "required_info": _multi_requirement_plan()["required_info"][:1],
        }
    )


def test_constraint_evidence_weave_assigns_distinct_facts_to_requirements():
    from simplemem.core.memweaver.proof_weave import select_constraint_evidence_weave

    location = _fact("location", "Dave returned from San Francisco")
    event = _fact("event", "Dave attended a car workshop.")
    generic = _fact("generic", "Dave shared notes.")

    selected = select_constraint_evidence_weave(
        required_info=_multi_requirement_plan()["required_info"],
        ranked_entries=[generic, location, event],
        reranker=_reranker(top_k=2),
        to_text=lambda entry: entry.lossless_restatement,
        provenance={},
        top_k=2,
    )

    assert [entry.entry_id for entry in selected] == ["location", "event"]


def test_constraint_evidence_weave_keeps_typed_anchor_next_to_primary():
    from simplemem.core.memweaver.proof_weave import select_constraint_evidence_weave

    bridge = _fact("bridge", "Dave returned from San Francisco.")
    anchor = _fact("anchor", "Dave attended a car workshop.")
    companion = _fact("companion", "Dave learned restoration techniques.")

    selected = select_constraint_evidence_weave(
        required_info=_multi_requirement_plan()["required_info"],
        ranked_entries=[bridge, companion, anchor],
        reranker=_reranker(top_k=3),
        to_text=lambda entry: entry.lossless_restatement,
        provenance={
            bridge.entry_id: Provenance(
                edge=EDGE_BRIDGE,
                anchor_id=anchor.entry_id,
                anchor_text=anchor.lossless_restatement,
            )
        },
        top_k=3,
    )

    assert [entry.entry_id for entry in selected] == ["bridge", "anchor", "companion"]


def test_constraint_evidence_weave_fills_remaining_budget_with_new_source_turns():
    from simplemem.core.memweaver.proof_weave import select_constraint_evidence_weave

    location = _fact("location", "Dave returned from San Francisco")
    location.source_turn_ids = [1]
    event = _fact("event", "Dave attended a car workshop.")
    event.source_turn_ids = [2]
    duplicate = _fact("duplicate", "Dave visited San Francisco again.")
    duplicate.source_turn_ids = [1]
    independent = _fact("independent", "Dave learned restoration techniques.")
    independent.source_turn_ids = [3]

    selected = select_constraint_evidence_weave(
        required_info=_multi_requirement_plan()["required_info"],
        ranked_entries=[location, event, duplicate, independent],
        reranker=_reranker(top_k=3),
        to_text=lambda entry: entry.lossless_restatement,
        provenance={},
        top_k=3,
    )

    assert [entry.entry_id for entry in selected] == ["location", "event", "independent"]


def test_constraint_evidence_weave_does_not_repeat_residual_source_turns():
    from simplemem.core.memweaver.proof_weave import select_constraint_evidence_weave

    location = _fact("location", "Dave returned from San Francisco")
    location.source_turn_ids = [1]
    event = _fact("event", "Dave attended a car workshop.")
    event.source_turn_ids = [2]
    first_residual = _fact("first-residual", "Dave shared travel notes.")
    first_residual.source_turn_ids = [3]
    duplicate_residual = _fact("duplicate-residual", "Dave shared more travel notes.")
    duplicate_residual.source_turn_ids = [3]
    second_residual = _fact("second-residual", "Dave booked a museum visit.")
    second_residual.source_turn_ids = [4]

    selected = select_constraint_evidence_weave(
        required_info=_multi_requirement_plan()["required_info"],
        ranked_entries=[
            location,
            event,
            first_residual,
            duplicate_residual,
            second_residual,
        ],
        reranker=_reranker(top_k=4),
        to_text=lambda entry: entry.lossless_restatement,
        provenance={},
        top_k=4,
    )

    assert [entry.entry_id for entry in selected] == [
        "location",
        "event",
        "first-residual",
        "second-residual",
    ]


def test_retriever_applies_constraint_evidence_weave_only_when_enabled():
    enabled = HybridRetriever(
        llm_client=None,
        vector_store=object(),
        enable_memweaver=True,
        enable_expand_rerank=True,
        enable_expansion=False,
        enable_rerank=True,
        enable_constraint_evidence_weave=True,
        reranker=_reranker(top_k=2),
    )
    disabled = HybridRetriever(
        llm_client=None,
        vector_store=object(),
        enable_memweaver=True,
        enable_expand_rerank=True,
        enable_expansion=False,
        enable_rerank=True,
        enable_constraint_evidence_weave=False,
        reranker=_reranker(top_k=2),
    )
    location = _fact("location", "Dave returned from San Francisco.")
    event = _fact("event", "Dave attended a car workshop.")
    generic = _fact("generic", "Dave shared notes.")

    enabled_selected = enabled._expand_and_rerank(
        "question",
        [generic, location, event],
        information_plan=_multi_requirement_plan(),
    )
    disabled_selected = disabled._expand_and_rerank(
        "question",
        [generic, location, event],
        information_plan=_multi_requirement_plan(),
    )

    assert [entry.entry_id for entry in enabled_selected] == ["location", "event"]
    assert [entry.entry_id for entry in disabled_selected] == ["generic", "location"]
