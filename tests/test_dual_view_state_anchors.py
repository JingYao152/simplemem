"""Regression coverage for versioned dual-view state-anchor retrieval."""

import json

import numpy as np

from simplemem.core.database.vector_store import VectorStore
from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.memweaver.dual_view import state_anchor_text
from simplemem.core.memweaver.writer import MemWeaver
from simplemem.core.models.memory_entry import KIND_FACT, Dialogue, MemoryEntry
from simplemem.core.utils.llm_client import LLMClient


class TopicEmbedder:
    """Small deterministic embedder that distinguishes topic-bearing text."""

    dimension = 3

    def __init__(self):
        self.documents = []

    def encode_documents(self, texts):
        self.documents.extend(texts)
        return np.stack([self._encode(text) for text in texts])

    def encode_single(self, text, is_query=False):
        return self._encode(text)

    @staticmethod
    def _encode(text):
        lowered = text.lower()
        if "painting" in lowered or "paints" in lowered:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "coffee" in lowered:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)


class ScriptedWriterLLM:
    """Minimal Call A/B double; index writes and embeddings remain real."""

    extract_json = LLMClient.extract_json
    _clean_json_string = LLMClient._clean_json_string
    _extract_balanced_json = LLMClient._extract_balanced_json

    def chat_completion(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Assign every turn of this session" in prompt:
            return json.dumps(
                {"assignments": [{"turns": [1], "thread": "new:Painting"}]}
            )
        if "[Thread]" in prompt:
            return json.dumps(
                {
                    "facts": [
                        {
                            "lossless_restatement": "Melanie bought brushes on 1 May 2023.",
                            "keywords": ["brushes"],
                            "timestamp": "2023-05-01T13:00:00",
                            "location": None,
                            "persons": ["Melanie"],
                            "entities": [],
                            "topic": "painting",
                            "weave": {"op": "none", "target": None},
                        }
                    ],
                    "summary": "Melanie discussed painting supplies.",
                    "summary_impact": "major",
                    "outdated_facts": [],
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")


def _store(tmp_path, name, embedder):
    return VectorStore(
        db_path=str(tmp_path / "lancedb"),
        table_name=name,
        embedding_model=embedder,
    )


def test_state_anchor_text_exposes_auditable_state_fields():
    entry = MemoryEntry(
        entry_id="fact-1",
        lossless_restatement="Alex started working in Berlin.",
        persons=["Alex"],
        entities=["Berlin"],
        topic="employment",
        thread_id="t7",
        valid_from="2023-05-01",
        valid_until="2023-06-10",
    )

    assert state_anchor_text(entry) == (
        "State anchor | thread: t7 | topic: employment | persons: Alex | "
        "entities: Berlin | valid from: 2023-05-01 | valid until: 2023-06-10 | "
        "fact: Alex started working in Berlin."
    )


def test_dual_view_semantic_search_keeps_anchor_only_candidate(tmp_path):
    primary = _store(tmp_path, "facts", TopicEmbedder())
    anchors = _store(tmp_path, "state_anchors", TopicEmbedder())
    target = MemoryEntry(
        entry_id="target",
        lossless_restatement="Mary attends it every Friday.",
        topic="painting lesson",
        kind=KIND_FACT,
        thread_id="t1",
        valid_from="2023-05-01",
    )
    distractor = MemoryEntry(
        entry_id="distractor",
        lossless_restatement="Mary paints abstract art every Friday.",
        topic="art hobby",
        kind=KIND_FACT,
        thread_id="t2",
        valid_from="2023-05-01",
    )
    primary.add_entries([target, distractor])
    anchors.add_entries(
        [target, distractor],
        embed_texts=[state_anchor_text(target), state_anchor_text(distractor)],
    )

    retriever = HybridRetriever(
        llm_client=None,
        vector_store=primary,
        state_anchor_store=anchors,
        enable_memweaver=True,
        enable_dual_view_state_anchors=True,
        semantic_top_k=1,
        enable_expand_rerank=False,
        enable_planning=False,
        enable_reflection=False,
        enable_parallel_retrieval=False,
    )

    assert {entry.entry_id for entry in retriever._semantic_search("painting")} == {
        "target",
        "distractor",
    }


def test_memweaver_dual_view_writes_bare_facts_and_anchor_index(tmp_path):
    fact_embedder = TopicEmbedder()
    anchor_embedder = TopicEmbedder()
    primary = _store(tmp_path, "facts", fact_embedder)
    anchors = _store(tmp_path, "state_anchors", anchor_embedder)
    weaver = MemWeaver(
        llm_client=ScriptedWriterLLM(),
        vector_store=primary,
        state_anchor_store=anchors,
        enable_dual_view_state_anchors=True,
        enable_recontext=True,
        enable_sweep=False,
        enable_entity_profiles=False,
        max_parallel_workers=1,
    )

    weaver.add_dialogues(
        [
            Dialogue(
                dialogue_id=1,
                speaker="Melanie",
                content="I bought brushes.",
                timestamp="1:00 pm on 1 May, 2023",
            )
        ]
    )
    weaver.finalize()

    assert "Melanie bought brushes on 1 May 2023." in fact_embedder.documents
    assert not any(
        text.startswith("[") for text in fact_embedder.documents
        if "Melanie bought brushes" in text
    )
    assert len(anchors.get_all_entries()) == 1
    assert anchors.get_all_entries()[0].context_digest.startswith("state-anchor-v1:")
    assert any("topic: painting" in text for text in anchor_embedder.documents)
