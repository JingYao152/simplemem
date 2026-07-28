from simplemem.core.memweaver.expansion import EDGE_BRIDGE, Provenance
from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.models.memory_entry import KIND_FACT, MemoryEntry
from simplemem.core.reranker import CrossEncoderReranker


class _KeywordCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, document in pairs:
            wanted = set(query.lower().split())
            scores.append(len(wanted & set(document.lower().split())))
        return scores


def test_requirement_bundle_keeps_a_bridged_fact_with_its_anchor():
    from simplemem.core.memweaver.bundles import select_requirement_bundles

    reranker = CrossEncoderReranker(
        top_k=2,
        model_factory=lambda _: _KeywordCrossEncoder(),
    )
    unrelated = MemoryEntry(
        entry_id="unrelated",
        lossless_restatement="Alice bought a blue notebook.",
        kind=KIND_FACT,
    )
    anchor = MemoryEntry(
        entry_id="anchor",
        lossless_restatement="Alice began preparing for a chess tournament.",
        kind=KIND_FACT,
    )
    bridged = MemoryEntry(
        entry_id="bridged",
        lossless_restatement="Alice hired a chess coach for the tournament.",
        kind=KIND_FACT,
    )

    selected = select_requirement_bundles(
        required_info=[
            {
                "priority": "high",
                "info_type": "cause",
                "description": "chess coach",
            }
        ],
        ranked_entries=[unrelated, bridged, anchor],
        reranker=reranker,
        to_text=lambda entry: entry.lossless_restatement,
        provenance={
            bridged.entry_id: Provenance(
                edge=EDGE_BRIDGE,
                anchor_id=anchor.entry_id,
                anchor_text=anchor.lossless_restatement,
            )
        },
        top_k=2,
    )

    assert [entry.entry_id for entry in selected] == ["bridged", "anchor"]


def test_retriever_applies_requirement_bundle_only_when_enabled():
    reranker = CrossEncoderReranker(
        top_k=2,
        model_factory=lambda _: _KeywordCrossEncoder(),
    )
    retriever = HybridRetriever(
        llm_client=None,
        vector_store=object(),
        enable_memweaver=True,
        enable_expand_rerank=True,
        enable_expansion=False,
        enable_rerank=True,
        enable_requirement_bundles=True,
        reranker=reranker,
    )
    unrelated = MemoryEntry(
        entry_id="unrelated",
        lossless_restatement="Alice bought a blue notebook.",
        kind=KIND_FACT,
    )
    anchor = MemoryEntry(
        entry_id="anchor",
        lossless_restatement="Alice began preparing for a chess tournament.",
        kind=KIND_FACT,
    )
    bridged = MemoryEntry(
        entry_id="bridged",
        lossless_restatement="Alice hired a chess coach for the tournament.",
        kind=KIND_FACT,
    )

    selected = retriever._expand_and_rerank(
        "question",
        [unrelated, bridged, anchor],
        information_plan={
            "required_info": [
                {
                    "priority": "high",
                    "info_type": "cause",
                    "description": "chess coach",
                }
            ]
        },
        provenance={
            bridged.entry_id: Provenance(
                edge=EDGE_BRIDGE,
                anchor_id=anchor.entry_id,
                anchor_text=anchor.lossless_restatement,
            )
        },
    )

    assert [entry.entry_id for entry in selected] == ["bridged", "anchor"]
