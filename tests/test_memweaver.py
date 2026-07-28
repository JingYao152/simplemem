"""Tests for MemWeaver P0: fabric data model, backend capabilities, Call A/B
write pipeline, weaving execution, as-of retrieval and the finalize sweep.

Every LLM call is scripted, so the tests pin the deterministic behaviour around
the LLM decision points (including each fallback) without any API access.
"""

import json

import numpy as np
import pytest

from simplemem.core.database.vector_store import VectorStore
from simplemem.core.database.vector_store_backend import (
    AnyOf,
    FieldPredicate,
    LanceDBVectorStoreBackend,
)
from simplemem.core.answer_generator import AnswerGenerator
from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.memweaver import MemWeaver
from simplemem.core.memweaver.asof import (
    apply_asof,
    asof_filters,
    compute_anchor,
    is_temporal_history_question,
)
from simplemem.core.memweaver.context import (
    CONTEXT_PREFIX_MAX_CHARS,
    context_digest,
    context_prefix,
    contextual_embed_text,
)
from simplemem.core.memweaver.dates import parse_session_datetime, to_day
from simplemem.core.memweaver.expansion import (
    EDGE_SUPERSEDED_BY,
    EDGE_SUPERSEDES,
    EDGE_THREAD,
    expand_one_hop,
    scoring_text,
)
from simplemem.core.memweaver.fabric import ThreadState, load_fabric, speaker_threads
from simplemem.core.models.memory_entry import (
    KIND_ENTITY_PROFILE,
    KIND_FACT,
    KIND_THREAD_SUMMARY,
    Dialogue,
    MemoryEntry,
)
from simplemem.core.reranker import CrossEncoderReranker
from simplemem.core.utils.llm_client import LLMClient


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class KeywordEmbedder:
    """Deterministic embedder: one axis per topic keyword."""

    dimension = 4
    _AXES = (
        ("coffee", 0),
        ("espresso", 0),
        ("paint", 1),
        ("watercolor", 1),
        ("pottery", 2),
    )

    def encode_documents(self, texts):
        return np.stack([self._encode(text) for text in texts])

    def encode_single(self, text, is_query=False):
        return self._encode(text)

    @classmethod
    def _encode(cls, text):
        vector = np.zeros(cls.dimension, dtype=np.float32)
        lowered = text.lower()
        for keyword, axis in cls._AXES:
            if keyword in lowered:
                vector[axis] = 1.0
        if not vector.any():
            vector[3] = 1.0
        return vector


class RecordingEmbedder(KeywordEmbedder):
    """Records what text actually reached the embedder, and marks prefixes."""

    def __init__(self):
        self.documents = []

    def encode_documents(self, texts):
        self.documents.extend(texts)
        return np.stack([self._encode(text) for text in texts])

    @classmethod
    def _encode(cls, text):
        vector = super()._encode(text)
        # A prefixed text lands on its own axis, so a re-embed is observable
        # through search as well as through the recorded texts.
        if text.startswith("["):
            vector = vector.copy()
            vector[3] = 0.5
        return vector


class ScriptedLLM:
    """LLM stub that answers per call type, reusing the real JSON extractor."""

    extract_json = LLMClient.extract_json
    _clean_json_string = LLMClient._clean_json_string
    _extract_balanced_json = LLMClient._extract_balanced_json

    def __init__(self, assignment=None, thread_update=None, sweep=None, extraction=None):
        self.scripts = {
            "assignment": list(assignment or []),
            "thread_update": list(thread_update or []),
            "sweep": list(sweep or []),
            "extraction": list(extraction or []),
        }
        self.prompts = {key: [] for key in self.scripts}
        self.temperatures = []

    def chat_completion(
        self,
        messages,
        temperature=0.2,
        response_format=None,
        max_retries=3,
    ):
        prompt = messages[-1]["content"]
        kind = self._classify(prompt)
        self.prompts[kind].append(prompt)
        self.temperatures.append(temperature)

        queue = self.scripts[kind]
        if not queue:
            raise AssertionError(f"No scripted {kind} response for: {prompt[:120]}")
        # The last scripted response repeats, so retries can be exercised.
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        return response(prompt) if callable(response) else response

    @staticmethod
    def _classify(prompt):
        if "Assign every turn of this session" in prompt:
            return "assignment"
        if "[Pairs]" in prompt:
            return "sweep"
        if "[Thread]" in prompt:
            return "thread_update"
        return "extraction"


class StubExtractor:
    """Stands in for the plain-SimpleMem fallback extractor."""

    def __init__(self, restatements):
        self.restatements = restatements
        self.calls = 0

    def _generate_memory_entries(self, dialogues):
        self.calls += 1
        return [
            MemoryEntry(lossless_restatement=text, keywords=["fallback"])
            for text in self.restatements
        ]


def assignment(*groups):
    return json.dumps({"assignments": [
        {"turns": list(turns), "thread": thread} for turns, thread in groups
    ]})


def thread_update(facts, summary="Thread summary", impact="major", outdated=None):
    return json.dumps(
        {
            "facts": facts,
            "summary": summary,
            "summary_impact": impact,
            "outdated_facts": outdated or [],
        }
    )


def fact(restatement, op="none", target=None, **kwargs):
    payload = {
        "lossless_restatement": restatement,
        "keywords": kwargs.get("keywords", []),
        "timestamp": kwargs.get("timestamp"),
        "location": kwargs.get("location"),
        "persons": kwargs.get("persons", []),
        "entities": kwargs.get("entities", []),
        "topic": kwargs.get("topic", "topic"),
        "weave": {"op": op, "target": target},
    }
    return payload


def turns(session_stamp, *texts, start=1, speaker="Alice"):
    return [
        Dialogue(
            dialogue_id=start + offset,
            speaker=speaker,
            content=text,
            timestamp=session_stamp,
        )
        for offset, text in enumerate(texts)
    ]


@pytest.fixture
def store(tmp_path):
    return VectorStore(
        db_path=str(tmp_path / "lancedb"),
        table_name="entries",
        embedding_model=KeywordEmbedder(),
    )


@pytest.fixture
def recording_store(tmp_path):
    return VectorStore(
        db_path=str(tmp_path / "lancedb-recording"),
        table_name="entries",
        embedding_model=RecordingEmbedder(),
    )


def build_weaver(store, llm, **kwargs):
    kwargs.setdefault("fallback_extractor", StubExtractor([]))
    kwargs.setdefault("max_parallel_workers", 1)
    return MemWeaver(llm_client=llm, vector_store=store, **kwargs)


def facts_of(store):
    return [
        entry
        for entry in store.get_all_entries()
        if entry.kind == KIND_FACT
    ]


def by_text(store, needle):
    for entry in store.get_all_entries():
        if needle.lower() in entry.lossless_restatement.lower():
            return entry
    raise AssertionError(f"No entry containing {needle!r}")


# ----------------------------------------------------------------------
# Data model + storage backend (the three new capabilities)
# ----------------------------------------------------------------------


def test_entry_defaults_keep_pure_simplemem_entries_valid():
    entry = MemoryEntry(lossless_restatement="Alice drinks espresso.")

    assert entry.kind == KIND_FACT
    assert (entry.thread_id, entry.valid_from, entry.valid_until) == ("", "", "")
    assert entry.superseded_by == "" and entry.links == []
    assert entry.is_open
    assert MemoryEntry.thread_summary_id("t3") == "thread::t3"
    assert MemoryEntry.entity_profile_id("Melanie") == "profile::Melanie"


def test_source_turn_ids_roundtrip_through_vector_store(store):
    entry = MemoryEntry(
        entry_id="source-fact",
        lossless_restatement="Alice drinks oat milk coffee.",
        source_turn_ids=[7, 9],
    )

    store.add_entries([entry])

    restored = store.get_by_ids(["source-fact"])[0]
    assert restored.source_turn_ids == [7, 9]


def test_weave_targets_reads_typed_edges():
    entry = MemoryEntry(
        lossless_restatement="x",
        links=["refine:a", "bridge:b", "refine:c"],
    )

    assert entry.weave_targets("refine") == ["a", "c"]
    assert entry.weave_targets("bridge") == ["b"]


def test_backend_get_update_delete_roundtrip(store):
    store.add_entries(
        [
            MemoryEntry(
                entry_id="f1",
                lossless_restatement="Alice drinks coffee every morning",
                thread_id="t1",
                valid_from="2023-05-01",
            ),
            MemoryEntry(
                entry_id="f2",
                lossless_restatement="Alice quit coffee",
                thread_id="t1",
                valid_from="2023-06-01",
            ),
        ]
    )

    fetched = store.get_by_ids(["f2", "f1", "absent"])
    assert [entry.entry_id for entry in fetched] == ["f2", "f1"]
    assert fetched[0].thread_id == "t1"

    store.update_metadata("f1", {"valid_until": "2023-06-01", "superseded_by": "f2"})
    store.update_metadata("f2", {"links": ["refine:f1"]})
    closed, successor = store.get_by_ids(["f1", "f2"])
    assert (closed.valid_until, closed.superseded_by) == ("2023-06-01", "f2")
    assert closed.is_open is False
    assert successor.links == ["refine:f1"]

    store.delete_by_ids(["f1"])
    assert [entry.entry_id for entry in store.get_all_entries()] == ["f2"]


def test_update_metadata_rejects_unknown_and_immutable_fields(store):
    store.add_entries([MemoryEntry(entry_id="f1", lossless_restatement="a")])

    with pytest.raises(ValueError, match="Unknown metadata fields"):
        store.update_metadata("f1", {"not_a_field": "x"})
    with pytest.raises(ValueError, match="cannot be updated in place"):
        store.backend.update_metadata("f1", {"entry_id": "other"})
    with pytest.raises(ValueError, match="Invalid metadata field"):
        store.backend.update_metadata("f1", {"kind = 'x' OR TRUE": "y"})


def test_asof_prefilter_excludes_closed_entries(store):
    store.add_entries(
        [
            MemoryEntry(
                entry_id="old",
                lossless_restatement="Alice drinks coffee",
                valid_from="2023-05-01",
                valid_until="2023-06-01",
            ),
            MemoryEntry(
                entry_id="new",
                lossless_restatement="Alice drinks coffee without sugar",
                valid_from="2023-06-01",
            ),
        ]
    )

    visible = store.semantic_search(
        "coffee", top_k=10, filters=asof_filters("2023-06-15")
    )
    assert [entry.entry_id for entry in visible] == ["new"]
    assert len(store.semantic_search("coffee", top_k=10)) == 2


def test_summary_rewrite_is_delete_plus_insert(store):
    summary_id = MemoryEntry.thread_summary_id("t1")
    store.add_entries(
        [
            MemoryEntry(
                entry_id=summary_id,
                lossless_restatement="Old summary",
                kind=KIND_THREAD_SUMMARY,
                thread_id="t1",
            )
        ]
    )

    store.delete_by_ids([summary_id])
    store.add_entries(
        [
            MemoryEntry(
                entry_id=summary_id,
                lossless_restatement="New summary",
                kind=KIND_THREAD_SUMMARY,
                thread_id="t1",
            )
        ]
    )

    rows = store.get_all_entries()
    assert len(rows) == 1
    assert rows[0].lossless_restatement == "New summary"


def test_lancedb_table_migrates_pre_memweaver_schema(tmp_path):
    import lancedb
    import pyarrow as pa

    db_path = str(tmp_path / "legacy")
    db = lancedb.connect(db_path)
    legacy_schema = pa.schema(
        [
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), KeywordEmbedder.dimension)),
        ]
    )
    table = db.create_table("entries", schema=legacy_schema)
    table.add(
        [
            {
                "entry_id": "legacy",
                "lossless_restatement": "Alice drinks coffee",
                "keywords": ["coffee"],
                "timestamp": "",
                "location": "",
                "persons": [],
                "entities": [],
                "topic": "",
                "vector": [1.0, 0.0, 0.0, 0.0],
            }
        ]
    )

    store = VectorStore(
        db_path=db_path,
        table_name="entries",
        embedding_model=KeywordEmbedder(),
    )
    assert isinstance(store.backend, LanceDBVectorStoreBackend)

    migrated = store.get_by_ids(["legacy"])[0]
    assert migrated.kind == KIND_FACT
    assert migrated.valid_until == "" and migrated.links == []

    store.add_entries(
        [
            MemoryEntry(
                entry_id="fresh",
                lossless_restatement="Alice quit coffee",
                thread_id="t1",
                valid_from="2023-06-01",
            )
        ]
    )
    assert store.get_by_ids(["fresh"])[0].thread_id == "t1"


def test_failed_fts_build_is_retried_per_mutation_not_per_query(store, monkeypatch):
    """A broken full-text index must not be rebuilt once per lexical query."""
    store.add_entries(
        [MemoryEntry(entry_id="f1", lossless_restatement="Alice drinks coffee")]
    )

    attempts = []

    def failing_create_fts_index(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("tantivy unavailable")

    monkeypatch.setattr(
        store.backend.table, "create_fts_index", failing_create_fts_index
    )

    for _ in range(3):
        assert store.keyword_search(["coffee"], top_k=5) == []
    assert len(attempts) == 1, "index build retried on every query"

    # A write marks the index stale again, so a transient failure can recover.
    store.add_entries(
        [MemoryEntry(entry_id="f2", lossless_restatement="Alice quit coffee")]
    )
    store.keyword_search(["coffee"], top_k=5)
    store.keyword_search(["coffee"], top_k=5)
    assert len(attempts) == 2


def test_filter_expression_rejects_bad_operator():
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        FieldPredicate("LIKE", "x")
    with pytest.raises(ValueError, match="at least one predicate"):
        AnyOf([])


# ----------------------------------------------------------------------
# Session boundaries
# ----------------------------------------------------------------------


def test_sessions_split_on_timestamp_change_and_last_stays_buffered(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1, 2], "new:First"))],
        thread_update=[thread_update([fact("Alice mentioned coffee.")])],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(
        turns("1:00 pm on 1 May, 2023", "a", "b")
        + turns("2:00 pm on 8 May, 2023", "c", "d", start=3)
    )

    # Only the first session is processed; the trailing one waits for a flush.
    assert weaver.stats["sessions"] == 1
    assert [dialogue.dialogue_id for dialogue in weaver.dialogue_buffer] == [3, 4]

    weaver.process_remaining()
    assert weaver.stats["sessions"] == 2
    assert weaver.dialogue_buffer == []


def test_all_locomo_session_date_formats_parse():
    assert parse_session_datetime("1:56 pm on 8 May, 2023") == (
        "2023-05-08",
        "2023-05-08T13:56:00",
    )
    assert parse_session_datetime("12:30 am on 24 December 2022")[0] == "2022-12-24"
    assert parse_session_datetime("2025-11-15T14:30:00") == (
        "2025-11-15",
        "2025-11-15T14:30:00",
    )
    assert parse_session_datetime("8 May, 2023") == ("2023-05-08", "")
    assert parse_session_datetime(None) == ("", "")
    assert to_day("1:56 pm on 8 May, 2023") == "2023-05-08"


# ----------------------------------------------------------------------
# Call A - thread assignment
# ----------------------------------------------------------------------


def test_call_a_attaches_later_session_to_existing_thread(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Coffee habit")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee every morning.")]),
            thread_update([fact("Alice switched to decaf coffee.")]),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "I love coffee"))
    weaver.add_dialogues(turns("1:00 pm on 8 May, 2023", "decaf now", start=2))
    weaver.process_remaining()

    stored_facts = facts_of(store)
    assert {entry.thread_id for entry in stored_facts} == {"t1"}
    assert weaver.stats["call_a_fallback"] == 0
    # The second Call A prompt carries the living thread catalogue.
    assert "[t1] Coffee habit" in llm.prompts["assignment"][1]


def test_call_a_unknown_thread_reference_falls_back_to_one_new_thread(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "existing:t9"))],
        thread_update=[thread_update([fact("Alice drinks coffee.")])],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.process_remaining()

    assert weaver.stats["call_a_fallback"] == 1
    assert {entry.thread_id for entry in facts_of(store)} == {"t1"}


def test_call_a_unparsable_output_falls_back(store):
    llm = ScriptedLLM(
        assignment=["not json at all"],
        thread_update=[thread_update([fact("Alice drinks coffee.")])],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.process_remaining()

    assert weaver.stats["call_a_fallback"] == 1
    assert len(facts_of(store)) == 1


def test_call_a_partial_coverage_parks_leftover_turns(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Coffee"))],
        thread_update=[
            thread_update([fact("Alice drinks coffee.")]),
            thread_update([fact("Alice paints on weekends.")]),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee", "painting"))
    weaver.process_remaining()

    assert weaver.stats["call_a_partial_fallback"] == 1
    assert {entry.thread_id for entry in facts_of(store)} == {"t1", "t2"}


def test_call_a_merges_repeated_targets_and_ignores_duplicate_turns(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1, 2], "new:Coffee"), ([2, 3], "new:coffee"))
        ],
        thread_update=[thread_update([fact("Alice drinks coffee.")])],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "a", "b", "c"))
    weaver.process_remaining()

    # Same title (case-insensitive) = same thread, and turn 2 is claimed once.
    assert len(llm.prompts["thread_update"]) == 1
    assert weaver.stats["threads_created"] == 1
    prompt = llm.prompts["thread_update"][0]
    assert prompt.count("[Alice]") == 3


# ----------------------------------------------------------------------
# Call B - facts, weaving, living summary
# ----------------------------------------------------------------------


def test_call_b_writes_fabric_fields_and_uses_session_temperature(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Coffee"))],
        thread_update=[
            thread_update(
                [
                    fact("Alice drinks coffee every morning.", persons=["Alice"]),
                    fact(
                        "Alice will visit the coffee fair.",
                        timestamp="2023-06-20T10:00:00",
                    ),
                ],
                summary="Alice's coffee routine",
            )
        ],
    )
    weaver = build_weaver(store, llm, temperature=0.7)

    weaver.add_dialogues(turns("1:56 pm on 8 May, 2023", "coffee talk"))
    weaver.process_remaining()

    stored = {entry.lossless_restatement: entry for entry in facts_of(store)}
    routine = stored["Alice drinks coffee every morning."]
    assert (routine.kind, routine.thread_id) == (KIND_FACT, "t1")
    # No own timestamp -> session date; own timestamp -> its day.
    assert routine.valid_from == "2023-05-08"
    assert routine.valid_until == "" and routine.superseded_by == ""
    assert stored["Alice will visit the coffee fair."].valid_from == "2023-06-20"

    summary = store.get_by_ids([MemoryEntry.thread_summary_id("t1")])[0]
    assert summary.kind == KIND_THREAD_SUMMARY
    assert summary.lossless_restatement == "Alice's coffee routine"
    assert summary.topic == "Coffee"
    assert summary.valid_from == "2023-05-08"
    assert summary.timestamp == "2023-05-08T13:56:00"

    assert set(llm.temperatures) == {0.7}


def test_supersede_closes_the_old_fact_at_the_session_date(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Coffee")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee every morning.")]),
            thread_update(
                [fact("Alice quit coffee entirely.", op="supersede", target=1)],
                summary="Alice quit coffee",
            ),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "quitting", start=2))
    weaver.process_remaining()

    old = by_text(store, "drinks coffee every morning")
    new = by_text(store, "quit coffee entirely")
    assert old.valid_until == "2023-06-08"
    assert old.superseded_by == new.entry_id
    assert new.is_open
    assert weaver.stats["weave_supersede"] == 1

    # Candidate numbering is visible to Call B, with validity state.
    assert "1. Alice drinks coffee every morning." in llm.prompts["thread_update"][1]


def test_refine_and_bridge_write_bidirectional_edges(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Coffee")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee every morning.")]),
            thread_update(
                [
                    fact("Alice drinks her coffee black.", op="refine", target=1),
                    fact("Alice buys coffee beans nearby.", op="bridge", target=1),
                ]
            ),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "more coffee", start=2))
    weaver.process_remaining()

    anchor = by_text(store, "drinks coffee every morning")
    refinement = by_text(store, "coffee black")
    bridge = by_text(store, "coffee beans nearby")

    assert refinement.weave_targets("refine") == [anchor.entry_id]
    assert bridge.weave_targets("bridge") == [anchor.entry_id]
    assert set(anchor.weave_targets("refine")) == {refinement.entry_id}
    assert set(anchor.weave_targets("bridge")) == {bridge.entry_id}
    assert anchor.is_open  # refine/bridge never close a fact
    assert weaver.stats["weave_refine"] == 1
    assert weaver.stats["weave_bridge"] == 1


def test_invalid_weave_target_degrades_to_none(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Coffee"))],
        thread_update=[
            thread_update(
                [fact("Alice drinks coffee.", op="supersede", target=99)]
            )
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.process_remaining()

    assert weaver.stats["weave_invalid_target"] == 1
    assert weaver.stats["weave_supersede"] == 0
    assert facts_of(store)[0].is_open


def test_weaving_disabled_keeps_threads_but_writes_no_edges(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Coffee")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee every morning.")]),
            thread_update(
                [fact("Alice quit coffee entirely.", op="supersede", target=1)]
            ),
        ],
    )
    weaver = build_weaver(store, llm, enable_weaving=False)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "quitting", start=2))
    weaver.process_remaining()

    assert by_text(store, "drinks coffee every morning").is_open
    assert weaver.stats["weave_supersede"] == 0
    assert {entry.thread_id for entry in facts_of(store)} == {"t1"}
    assert store.get_by_ids([MemoryEntry.thread_summary_id("t1")])


def test_summary_impact_none_keeps_the_previous_summary(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Coffee")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee.")], summary="First summary"),
            thread_update(
                [fact("Alice mentioned coffee prices.")],
                summary="Rewritten summary",
                impact="none",
            ),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "prices", start=2))
    weaver.process_remaining()

    summary = store.get_by_ids([MemoryEntry.thread_summary_id("t1")])[0]
    assert summary.lossless_restatement == "First summary"
    assert weaver.stats["summary_skipped"] == 1
    assert weaver.stats["summary_rewrites"] == 1


def test_summary_rewrite_replaces_the_row_in_place(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Coffee")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee.")], summary="First summary"),
            thread_update([fact("Alice quit coffee.")], summary="Second summary"),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "quit", start=2))
    weaver.process_remaining()

    summaries = [
        entry
        for entry in store.get_all_entries()
        if entry.kind == KIND_THREAD_SUMMARY
    ]
    assert len(summaries) == 1
    assert summaries[0].entry_id == MemoryEntry.thread_summary_id("t1")
    assert summaries[0].lossless_restatement == "Second summary"
    # Living summary is the thread state Call B sees next session.
    assert "First summary" in llm.prompts["thread_update"][1]


def test_call_b_failure_falls_back_to_plain_extraction(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Coffee"))],
        thread_update=["}{ not json"],
    )
    extractor = StubExtractor(["Alice drinks coffee (fallback extraction)."])
    weaver = build_weaver(store, llm, fallback_extractor=extractor)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.process_remaining()

    assert weaver.stats["call_b_fallback"] == 1
    assert extractor.calls == 1
    assert len(llm.prompts["thread_update"]) == 3  # three parse attempts
    fallback_fact = facts_of(store)[0]
    assert fallback_fact.thread_id == "t1"
    assert fallback_fact.valid_from == "2023-05-01"


def test_outdated_facts_are_recorded_as_p1_signal(store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Coffee")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Alice drinks coffee.")]),
            thread_update([fact("Alice quit coffee.")], outdated=[1, 7]),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "quit", start=2))
    weaver.process_remaining()

    # Out-of-range candidate numbers are dropped; P0 only records the signal.
    assert weaver.stats["outdated_signals"] == 1


def test_fabric_snapshot_survives_store_clear(store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Coffee")), assignment(([1], "new:Coffee"))],
        thread_update=[thread_update([fact("Alice drinks coffee.")])],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "coffee"))
    weaver.process_remaining()
    store.clear()

    # Thread state is re-derived from storage, so ids restart cleanly.
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "coffee", start=2))
    weaver.process_remaining()

    assert {entry.thread_id for entry in facts_of(store)} == {"t1"}
    assert len(facts_of(store)) == 1


def test_load_fabric_reads_threads_and_orders_candidates(store):
    store.add_entries(
        [
            MemoryEntry(
                entry_id=MemoryEntry.thread_summary_id("t1"),
                lossless_restatement="Coffee thread summary",
                topic="Coffee",
                kind=KIND_THREAD_SUMMARY,
                thread_id="t1",
                valid_from="2023-06-01",
            ),
            MemoryEntry(
                entry_id="late",
                lossless_restatement="Alice quit coffee",
                thread_id="t1",
                valid_from="2023-06-01",
            ),
            MemoryEntry(
                entry_id="early",
                lossless_restatement="Alice drinks coffee",
                thread_id="t1",
                valid_from="2023-05-01",
            ),
        ]
    )

    snapshot = load_fabric(store)
    assert list(snapshot.threads) == ["t1"]
    assert snapshot.threads["t1"].title == "Coffee"
    assert [entry.entry_id for entry in snapshot.facts("t1")] == ["early", "late"]
    assert snapshot.next_thread_id() == "t2"
    assert "[t1] Coffee - Coffee thread summary" == snapshot.threads["t1"].one_line()


# ----------------------------------------------------------------------
# finalize() - cross-thread sweep
# ----------------------------------------------------------------------


def _seed_cross_thread_facts(store):
    store.add_entries(
        [
            MemoryEntry(
                entry_id="old",
                lossless_restatement="Alice drinks coffee every morning",
                thread_id="t1",
                valid_from="2023-05-01",
            ),
            MemoryEntry(
                entry_id="new",
                lossless_restatement="Alice stopped drinking coffee",
                thread_id="t2",
                valid_from="2023-06-01",
            ),
        ]
    )


def test_cross_thread_sweep_closes_missed_supersede(store):
    _seed_cross_thread_facts(store)
    llm = ScriptedLLM(
        sweep=[json.dumps({"judgements": [{"index": 1, "supersedes": True}]})]
    )
    weaver = build_weaver(store, llm)

    weaver.finalize()

    closed = store.get_by_ids(["old"])[0]
    assert closed.valid_until == "2023-06-01"
    assert closed.superseded_by == "new"
    assert store.get_by_ids(["new"])[0].is_open
    assert weaver.stats["sweep_supersedes"] == 1
    assert weaver.stats["sweep_pairs_judged"] == 1
    # Pairs are presented earlier-first; direction comes from dates, not the LLM.
    prompt = llm.prompts["sweep"][0]
    assert prompt.index("Alice drinks coffee every morning") < prompt.index(
        "Alice stopped drinking coffee"
    )


def test_cross_thread_sweep_respects_a_negative_judgement(store):
    _seed_cross_thread_facts(store)
    llm = ScriptedLLM(
        sweep=[json.dumps({"judgements": [{"index": 1, "supersedes": False}]})]
    )
    weaver = build_weaver(store, llm)

    weaver.finalize()

    assert store.get_by_ids(["old"])[0].is_open
    assert weaver.stats["sweep_supersedes"] == 0


def test_sweep_skips_same_thread_and_same_day_pairs(store):
    store.add_entries(
        [
            MemoryEntry(
                entry_id="a",
                lossless_restatement="Alice drinks coffee",
                thread_id="t1",
                valid_from="2023-05-01",
            ),
            MemoryEntry(
                entry_id="b",
                lossless_restatement="Alice drinks more coffee",
                thread_id="t1",
                valid_from="2023-06-01",
            ),
            MemoryEntry(
                entry_id="c",
                lossless_restatement="Bob drinks coffee too",
                thread_id="t2",
                valid_from="2023-05-01",
            ),
        ]
    )
    llm = ScriptedLLM(sweep=[json.dumps({"judgements": []})])
    weaver = build_weaver(store, llm)

    weaver.finalize()

    # a/b share a thread; a/c share a date -> nothing is orderable but b/c.
    assert weaver.stats["sweep_unordered_pairs"] >= 1
    assert all(entry.is_open for entry in facts_of(store) if entry.entry_id != "b")


def test_sweep_counts_each_unorderable_pair_once(store):
    """Same-day cross-thread pairs are skipped, and counted per pair."""
    store.add_entries(
        [
            MemoryEntry(
                entry_id=entry_id,
                lossless_restatement=f"{entry_id} drinks coffee",
                thread_id=thread_id,
                valid_from="2023-05-01",
            )
            for entry_id, thread_id in (("a", "t1"), ("b", "t2"), ("c", "t3"))
        ]
    )
    llm = ScriptedLLM(sweep=[json.dumps({"judgements": []})])
    weaver = build_weaver(store, llm)

    weaver.finalize()

    # Three distinct pairs among three facts, each seen from both endpoints.
    assert weaver.stats["sweep_unordered_pairs"] == 3
    assert weaver.stats["sweep_supersedes"] == 0
    assert all(entry.is_open for entry in facts_of(store))


def test_sweep_disabled_leaves_the_fabric_untouched(store):
    _seed_cross_thread_facts(store)
    llm = ScriptedLLM()
    weaver = build_weaver(store, llm, enable_sweep=False)

    weaver.finalize()

    assert store.get_by_ids(["old"])[0].is_open
    assert llm.prompts["sweep"] == []


def test_sweep_parse_failure_is_a_no_op(store):
    _seed_cross_thread_facts(store)
    llm = ScriptedLLM(sweep=["nonsense"])
    weaver = build_weaver(store, llm)

    weaver.finalize()

    assert store.get_by_ids(["old"])[0].is_open
    assert weaver.stats["sweep_parse_failures"] == 1


# ----------------------------------------------------------------------
# As-of retrieval (read side)
# ----------------------------------------------------------------------


def test_compute_anchor_prefers_session_dated_entries():
    entries = [
        MemoryEntry(
            lossless_restatement="summary",
            kind=KIND_THREAD_SUMMARY,
            valid_from="2023-06-08",
        ),
        MemoryEntry(
            lossless_restatement="closed fact",
            valid_from="2023-05-01",
            valid_until="2023-06-08",
        ),
        # A fact about a future plan must not become the anchor.
        MemoryEntry(lossless_restatement="future plan", valid_from="2024-01-01"),
    ]

    assert compute_anchor(entries) == "2023-06-08"
    assert compute_anchor([]) == ""
    assert compute_anchor(
        [MemoryEntry(lossless_restatement="only fact", valid_from="2023-03-03")]
    ) == "2023-03-03"


@pytest.mark.parametrize(
    "question",
    [
        "When did Caroline go to the LGBTQ support group?",
        "How long has Melanie been practicing art?",
        "How many weeks passed between Maria adopting Coco and Shadow?",
    ],
)
def test_temporal_history_questions_are_detected(question):
    assert is_temporal_history_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "What does Melanie do to relax?",
        "Where does Caroline work now?",
        "Who is Nate's neighbour?",
    ],
)
def test_non_temporal_questions_are_not_detected(question):
    assert not is_temporal_history_question(question)


def test_apply_asof_filters_result_lists():
    open_entry = MemoryEntry(lossless_restatement="open", valid_from="2023-05-01")
    closed_early = MemoryEntry(
        lossless_restatement="closed early",
        valid_from="2023-01-01",
        valid_until="2023-03-01",
    )
    closed_after_anchor = MemoryEntry(
        lossless_restatement="closed later",
        valid_from="2023-01-01",
        valid_until="2023-07-01",
    )
    entries = [open_entry, closed_early, closed_after_anchor]

    assert apply_asof(entries, "2023-06-01") == [open_entry, closed_after_anchor]
    assert apply_asof(entries, "") == entries


def _retriever(store, **kwargs):
    kwargs.setdefault("enable_planning", False)
    kwargs.setdefault("enable_reflection", False)
    kwargs.setdefault("enable_parallel_retrieval", False)
    kwargs.setdefault("enable_memweaver", True)
    # These tests are about the retrieval base; P2 expansion/rerank is opted into
    # explicitly by the tests that cover it.
    kwargs.setdefault("enable_expand_rerank", False)
    return HybridRetriever(
        llm_client=ScriptedLLM(), vector_store=store, **kwargs
    )


def _seed_superseded_pair(store, last_session_date="2023-07-05"):
    """A closed fact, its successor, and a later session that sets the anchor."""
    store.add_entries(
        [
            MemoryEntry(
                entry_id="old",
                lossless_restatement="Alice drinks coffee every morning",
                kind=KIND_FACT,
                thread_id="t1",
                valid_from="2023-05-01",
                valid_until="2023-06-08",
                superseded_by="new",
            ),
            MemoryEntry(
                entry_id="new",
                lossless_restatement="Alice stopped drinking coffee",
                kind=KIND_FACT,
                thread_id="t1",
                valid_from="2023-06-08",
            ),
            MemoryEntry(
                entry_id=MemoryEntry.thread_summary_id("t1"),
                lossless_restatement="Alice used to drink coffee and stopped",
                kind=KIND_THREAD_SUMMARY,
                thread_id="t1",
                valid_from=last_session_date,
            ),
        ]
    )


def test_state_question_only_sees_facts_valid_at_the_anchor(store):
    _seed_superseded_pair(store)
    retriever = _retriever(store, semantic_top_k=10)

    results = retriever.retrieve("Does Alice drink coffee?")

    assert "old" not in {entry.entry_id for entry in results}
    assert "new" in {entry.entry_id for entry in results}


def test_when_question_still_sees_superseded_facts(store):
    _seed_superseded_pair(store)
    retriever = _retriever(store, semantic_top_k=10)

    results = retriever.retrieve("When did Alice drink coffee every morning?")

    assert {"old", "new"} <= {entry.entry_id for entry in results}


def test_memweaver_disabled_never_filters(store):
    _seed_superseded_pair(store)
    retriever = _retriever(store, semantic_top_k=10, enable_memweaver=False)

    results = retriever.retrieve("Does Alice drink coffee?")

    assert {"old", "new"} <= {entry.entry_id for entry in results}


def test_fact_closed_on_the_anchor_day_stays_visible(store):
    """Design doc section 5 predicate: valid_until == "" OR valid_until >= anchor.

    At day resolution a fact closed on the anchor day was still valid during
    part of it, so the boundary is inclusive - the filter only hides facts
    closed strictly before the anchor.
    """
    _seed_superseded_pair(store, last_session_date="2023-06-08")
    retriever = _retriever(store, semantic_top_k=10)

    assert retriever._current_anchor() == "2023-06-08"
    results = retriever.retrieve("Does Alice drink coffee?")
    assert {"old", "new"} <= {entry.entry_id for entry in results}


def test_keyword_and_structured_paths_apply_the_same_predicate(store):
    _seed_superseded_pair(store)
    retriever = _retriever(store)
    anchor = retriever._current_anchor()
    assert anchor == "2023-07-05"

    keyword_hits = retriever._keyword_search(
        "coffee", {"keywords": ["coffee"]}, as_of=anchor
    )
    assert "old" not in {entry.entry_id for entry in keyword_hits}

    store.update_metadata("old", {"persons": ["Alice"]})
    store.update_metadata("new", {"persons": ["Alice"]})
    structured_hits = retriever._structured_search(
        {"persons": ["Alice"]}, as_of=anchor
    )
    assert {entry.entry_id for entry in structured_hits} == {"new"}


def test_thread_summaries_are_part_of_the_semantic_pool(store):
    _seed_superseded_pair(store)
    retriever = _retriever(store, semantic_top_k=10)

    results = retriever.retrieve("Does Alice drink coffee?")

    kinds = {entry.kind for entry in results}
    assert KIND_THREAD_SUMMARY in kinds and KIND_FACT in kinds


# ----------------------------------------------------------------------
# P1 - context-inheriting embeddings (design doc section 4)
# ----------------------------------------------------------------------


def test_context_prefix_is_the_first_sentence_of_the_living_summary():
    assert context_prefix(
        "Melanie paints watercolours every weekend. She started in 2021."
    ) == "Melanie paints watercolours every weekend."
    # Falls back to the title, then to nothing at all.
    assert context_prefix("", "Painting hobby") == "Painting hobby"
    assert context_prefix("", "") == ""
    assert context_prefix(None) == ""

    long_summary = "word " * 200
    prefix = context_prefix(long_summary)
    assert len(prefix) <= CONTEXT_PREFIX_MAX_CHARS + 3
    assert prefix.endswith("...")


def test_contextual_embed_text_and_digest_are_consistent():
    assert contextual_embed_text("Painting hobby", "Melanie bought brushes.") == (
        "[Painting hobby] Melanie bought brushes."
    )
    assert contextual_embed_text("", "Melanie bought brushes.") == (
        "Melanie bought brushes."
    )
    assert context_digest("Painting hobby") == context_digest("Painting hobby")
    assert context_digest("Painting hobby") != context_digest("Pottery class")
    assert context_digest("") == ""


def test_thread_context_reaches_the_embedder_but_no_stored_field(recording_store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Painting hobby"))],
        thread_update=[
            thread_update(
                [fact("Melanie bought watercolor brushes.")],
                summary="Melanie has painted watercolours for years.",
            )
        ],
    )
    weaver = build_weaver(recording_store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "paint talk"))
    weaver.process_remaining()

    embedded = [text for text in recording_store.embedding_model.documents
                if "watercolor brushes" in text]
    assert embedded == [
        "[Melanie has painted watercolours for years.] "
        "Melanie bought watercolor brushes."
    ]

    stored = by_text(recording_store, "watercolor brushes")
    # The text layer stays pure SimpleMem: no prefix in any stored field.
    assert stored.lossless_restatement == "Melanie bought watercolor brushes."
    assert "[" not in stored.lossless_restatement
    assert stored.topic == "topic" and stored.keywords == []
    # The vector layer records which context it was built under.
    assert stored.context_digest == context_digest(
        "Melanie has painted watercolours for years."
    )


def test_recontext_disabled_embeds_the_bare_sentence(recording_store):
    llm = ScriptedLLM(
        assignment=[assignment(([1], "new:Painting hobby"))],
        thread_update=[
            thread_update(
                [fact("Melanie bought watercolor brushes.")],
                summary="Melanie has painted watercolours for years.",
            )
        ],
    )
    weaver = build_weaver(recording_store, llm, enable_recontext=False)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "paint talk"))
    weaver.process_remaining()

    assert "Melanie bought watercolor brushes." in (
        recording_store.embedding_model.documents
    )
    assert not any(
        text.startswith("[") for text in recording_store.embedding_model.documents
    )
    assert by_text(recording_store, "watercolor brushes").context_digest == ""


def test_facts_inherit_the_kept_summary_when_it_is_not_rewritten(recording_store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Painting hobby")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Melanie paints on weekends.")], summary="First summary."),
            thread_update(
                [fact("Melanie mentioned paint prices.")],
                summary="Ignored rewrite.",
                impact="none",
            ),
        ],
    )
    weaver = build_weaver(recording_store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "paint"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "prices", start=2))
    weaver.process_remaining()

    # The stored summary was kept, so the context prefix is the kept one - the
    # digest can never disagree with what is stored.
    summary = recording_store.get_by_ids([MemoryEntry.thread_summary_id("t1")])[0]
    assert summary.lossless_restatement == "First summary."
    later = by_text(recording_store, "paint prices")
    assert later.context_digest == context_digest("First summary.")
    assert "[First summary.] Melanie mentioned paint prices." in (
        recording_store.embedding_model.documents
    )


# ----------------------------------------------------------------------
# P1 - semantically triggered re-embedding
# ----------------------------------------------------------------------


def _recontext_llm():
    return ScriptedLLM(
        assignment=[
            assignment(([1], "new:Painting hobby")),
            assignment(([1], "existing:t1")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Melanie bought watercolor brushes.")],
                          summary="Melanie is starting to paint."),
            thread_update([fact("Melanie sold a painting.")],
                          summary="Melanie now sells her watercolour paintings.",
                          outdated=[1]),
            thread_update([fact("Melanie framed a painting.")],
                          summary="Melanie now sells her watercolour paintings.",
                          impact="minor",
                          outdated=[1]),
        ],
    )


def test_outdated_facts_are_reembedded_under_the_rewritten_summary(recording_store):
    weaver = build_weaver(recording_store, _recontext_llm())

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "brushes"))
    weaver.process_remaining()
    first_digest = by_text(recording_store, "watercolor brushes").context_digest

    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "sold", start=2))
    weaver.process_remaining()

    old_fact = by_text(recording_store, "watercolor brushes")
    assert weaver.stats["recontext_reembedded"] == 1
    assert old_fact.context_digest != first_digest
    assert old_fact.context_digest == context_digest(
        "Melanie now sells her watercolour paintings."
    )
    # Fact text is immutable - only the vector moved.
    assert old_fact.lossless_restatement == "Melanie bought watercolor brushes."
    assert (
        "[Melanie now sells her watercolour paintings.] "
        "Melanie bought watercolor brushes."
    ) in recording_store.embedding_model.documents


def test_summary_digest_change_reembeds_all_open_facts_without_outdated_signal(
    recording_store,
):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Painting hobby")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update(
                [
                    fact("Melanie bought watercolor brushes."),
                    fact("Melanie joined a painting class."),
                ],
                summary="Melanie is starting to paint.",
            ),
            thread_update(
                [fact("Melanie sold a painting.")],
                summary="Melanie now sells her watercolour paintings.",
            ),
        ],
    )
    weaver = build_weaver(
        recording_store,
        llm,
        enable_thread_wide_recontext=True,
    )

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "brushes"))
    weaver.process_remaining()
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "sold", start=2))
    weaver.process_remaining()

    digest = context_digest("Melanie now sells her watercolour paintings.")
    assert weaver.stats["recontext_reembedded"] == 2
    assert by_text(recording_store, "watercolor brushes").context_digest == digest
    assert by_text(recording_store, "painting class").context_digest == digest


def test_reembedding_is_idempotent_via_the_context_digest(recording_store):
    weaver = build_weaver(recording_store, _recontext_llm())

    for stamp, text, start in (
        ("1:00 pm on 1 May, 2023", "brushes", 1),
        ("1:00 pm on 8 June, 2023", "sold", 2),
        ("1:00 pm on 9 July, 2023", "framed", 3),
    ):
        weaver.add_dialogues(turns(stamp, text, start=start))
        weaver.process_remaining()

    # Third session names the same fact again, but the summary is unchanged, so
    # its vector is already current.
    assert weaver.stats["recontext_reembedded"] == 1
    assert weaver.stats["recontext_up_to_date"] == 1


def test_recontext_disabled_never_reembeds(recording_store):
    weaver = build_weaver(recording_store, _recontext_llm(), enable_recontext=False)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "brushes"))
    weaver.process_remaining()
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "sold", start=2))
    weaver.process_remaining()

    assert weaver.stats["recontext_reembedded"] == 0
    assert weaver.stats["outdated_signals"] == 1  # the signal is still recorded


def test_reembed_entries_rejects_mismatched_texts(store):
    store.add_entries([MemoryEntry(entry_id="f1", lossless_restatement="a")])
    entries = store.get_by_ids(["f1"])

    with pytest.raises(ValueError, match="embed_texts has"):
        store.reembed_entries(entries, ["one", "two"], "digest")
    with pytest.raises(ValueError, match="embed_texts has"):
        store.add_entries(entries, embed_texts=[])


def test_update_vector_moves_the_entry_in_the_index(store):
    store.add_entries(
        [
            MemoryEntry(entry_id="f1", lossless_restatement="Alice drinks coffee"),
            MemoryEntry(entry_id="f2", lossless_restatement="Melanie paints"),
        ]
    )
    assert store.semantic_search("paint", top_k=1)[0].entry_id == "f2"

    # Re-embed f1 under a painting context; it must now win the paint query.
    store.reembed_entries(
        store.get_by_ids(["f1"]), ["[Painting hobby] Alice drinks coffee"], "digest1"
    )
    reembedded = store.get_by_ids(["f1"])[0]
    assert reembedded.context_digest == "digest1"
    assert reembedded.lossless_restatement == "Alice drinks coffee"
    assert "f1" in {
        entry.entry_id for entry in store.semantic_search("paint", top_k=2)
    }

    with pytest.raises(ValueError, match="cannot be updated through fields"):
        store.backend.update_vector("f1", [0.0] * 4, {"vector": [1.0]})
    with pytest.raises(ValueError, match="Invalid metadata field"):
        store.backend.update_vector("f1", [0.0] * 4, {"kind = 'x' OR TRUE": "y"})


# ----------------------------------------------------------------------
# P1 - entity profiles in the retrieval pool
# ----------------------------------------------------------------------


def _profile_llm():
    return ScriptedLLM(
        assignment=[
            assignment(([1, 2], "new:Painting hobby")),
            assignment(([1], "existing:t1"), ([2], "new:Pottery class")),
        ],
        thread_update=[
            thread_update(
                [fact("Melanie paints watercolours.", persons=["Melanie"])],
                summary="Melanie paints watercolours every weekend.",
            ),
            thread_update(
                [fact("Melanie sold a painting.", persons=["Melanie"])],
                summary="Melanie sells her watercolour paintings now.",
            ),
            thread_update(
                [fact("Caroline signed up for pottery.", persons=["Caroline"])],
                summary="Caroline joined a pottery class.",
            ),
        ],
    )


def test_entity_profiles_are_written_for_the_session_speakers(store):
    llm = _profile_llm()
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(
        [
            Dialogue(dialogue_id=1, speaker="Melanie", content="I paint",
                     timestamp="1:00 pm on 1 May, 2023"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="nice",
                     timestamp="1:00 pm on 1 May, 2023"),
        ]
    )
    weaver.process_remaining()

    profiles = {
        entry.entry_id: entry
        for entry in store.get_all_entries()
        if entry.kind == KIND_ENTITY_PROFILE
    }
    assert set(profiles) == {
        MemoryEntry.entity_profile_id("Melanie"),
        MemoryEntry.entity_profile_id("Caroline"),
    }
    melanie = profiles[MemoryEntry.entity_profile_id("Melanie")]
    assert melanie.persons == ["Melanie"]
    assert melanie.topic == "Melanie"
    assert melanie.valid_from == "2023-05-01" and melanie.is_open
    assert "Painting hobby" in melanie.lossless_restatement
    assert "Melanie paints watercolours every weekend." in melanie.lossless_restatement
    assert weaver.stats["profiles_written"] == 2


def test_profiles_are_rewritten_in_place_as_threads_evolve(store):
    llm = _profile_llm()
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(
        [
            Dialogue(dialogue_id=1, speaker="Melanie", content="I paint",
                     timestamp="1:00 pm on 1 May, 2023"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="nice",
                     timestamp="1:00 pm on 1 May, 2023"),
        ]
    )
    weaver.add_dialogues(
        [
            Dialogue(dialogue_id=3, speaker="Melanie", content="sold one",
                     timestamp="2:00 pm on 8 June, 2023"),
            Dialogue(dialogue_id=4, speaker="Caroline", content="pottery for me",
                     timestamp="2:00 pm on 8 June, 2023"),
        ]
    )
    weaver.process_remaining()

    profiles = [
        entry
        for entry in store.get_all_entries()
        if entry.kind == KIND_ENTITY_PROFILE
    ]
    assert len(profiles) == 2, "one row per speaker, rewritten not appended"

    melanie = store.get_by_ids([MemoryEntry.entity_profile_id("Melanie")])[0]
    caroline = store.get_by_ids([MemoryEntry.entity_profile_id("Caroline")])[0]
    assert "sells her watercolour paintings" in melanie.lossless_restatement
    assert melanie.valid_from == "2023-06-08"
    # Caroline's own thread shows up in her profile; Melanie's does not.
    assert "Pottery class" in caroline.lossless_restatement
    assert "Pottery class" not in melanie.lossless_restatement


def test_profiles_join_the_semantic_pool(store):
    llm = _profile_llm()
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(
        [
            Dialogue(dialogue_id=1, speaker="Melanie", content="I paint",
                     timestamp="1:00 pm on 1 May, 2023"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="nice",
                     timestamp="1:00 pm on 1 May, 2023"),
        ]
    )
    weaver.process_remaining()

    retriever = _retriever(store, semantic_top_k=10)
    kinds = {entry.kind for entry in retriever.retrieve("What does Melanie paint?")}

    assert KIND_ENTITY_PROFILE in kinds
    assert KIND_FACT in kinds


def test_profiles_disabled_writes_no_profile_rows(store):
    llm = _profile_llm()
    weaver = build_weaver(store, llm, enable_entity_profiles=False)

    weaver.add_dialogues(
        [
            Dialogue(dialogue_id=1, speaker="Melanie", content="I paint",
                     timestamp="1:00 pm on 1 May, 2023"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="nice",
                     timestamp="1:00 pm on 1 May, 2023"),
        ]
    )
    weaver.process_remaining()

    assert not [
        entry
        for entry in store.get_all_entries()
        if entry.kind == KIND_ENTITY_PROFILE
    ]
    assert weaver.stats["profiles_written"] == 0


def test_profiles_count_as_session_dated_for_the_anchor():
    entries = [
        MemoryEntry(
            lossless_restatement="profile",
            kind=KIND_ENTITY_PROFILE,
            persons=["Melanie"],
            valid_from="2023-08-01",
        ),
        MemoryEntry(
            lossless_restatement="summary",
            kind=KIND_THREAD_SUMMARY,
            valid_from="2023-06-08",
        ),
        MemoryEntry(lossless_restatement="future plan", valid_from="2024-01-01"),
    ]

    assert compute_anchor(entries) == "2023-08-01"


def test_speaker_threads_orders_by_recency():
    threads = {
        "t1": ThreadState("t1", "Painting", "s1", updated_on="2023-05-01"),
        "t2": ThreadState("t2", "Pottery", "s2", updated_on="2023-07-01"),
        "t3": ThreadState("t3", "Running", "s3", updated_on="2023-06-01"),
    }
    facts_by_thread = {
        "t1": [MemoryEntry(lossless_restatement="a", persons=["Melanie"])],
        "t2": [MemoryEntry(lossless_restatement="b", persons=["Caroline"])],
        "t3": [MemoryEntry(lossless_restatement="c", persons=["Melanie"])],
    }

    ordered = speaker_threads(threads, facts_by_thread, "Melanie")
    assert [state.thread_id for state in ordered] == ["t3", "t1"]

    # A thread the speaker just spoke in counts even before its facts name them.
    with_extra = speaker_threads(threads, facts_by_thread, "Melanie", ["t2"])
    assert [state.thread_id for state in with_extra] == ["t2", "t3", "t1"]


# ----------------------------------------------------------------------
# P1 review fixes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "summary, expected",
    [
        # Abbreviations must not be read as sentence ends.
        ("Melanie moved to St. Louis and paints there.",
         "Melanie moved to St. Louis and paints there."),
        ("Nate served in the U.S. Army before teaching.",
         "Nate served in the U.S. Army before teaching."),
        ("Dr. Kim treats Caroline's knee.", "Dr. Kim treats Caroline's knee."),
        # Real sentence boundaries still cut.
        ("Melanie paints watercolours. She sells them now.",
         "Melanie paints watercolours."),
        ("Does Caroline still run? She stopped in May.", "Does Caroline still run?"),
        ("Melanie won! She was thrilled.", "Melanie won!"),
        # A trailing period is not a boundary to cut at.
        ("Melanie paints watercolours.", "Melanie paints watercolours."),
    ],
)
def test_context_prefix_survives_abbreviations(summary, expected):
    assert context_prefix(summary) == expected


def test_repeated_outdated_indices_are_reembedded_once(recording_store):
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Painting hobby")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Melanie bought watercolor brushes.")],
                          summary="Melanie is starting to paint."),
            thread_update([fact("Melanie sold a painting.")],
                          summary="Melanie sells her watercolour paintings now.",
                          outdated=[1, 1, 1]),
        ],
    )
    weaver = build_weaver(recording_store, llm)

    weaver.add_dialogues(turns("1:00 pm on 1 May, 2023", "brushes"))
    weaver.add_dialogues(turns("1:00 pm on 8 June, 2023", "sold", start=2))
    weaver.process_remaining()

    assert weaver.stats["recontext_reembedded"] == 1
    assert weaver.stats["recontext_up_to_date"] == 0


def test_profile_text_carries_no_internal_thread_ids(store):
    llm = _profile_llm()
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(
        [
            Dialogue(dialogue_id=1, speaker="Melanie", content="I paint",
                     timestamp="1:00 pm on 1 May, 2023"),
            Dialogue(dialogue_id=2, speaker="Caroline", content="nice",
                     timestamp="1:00 pm on 1 May, 2023"),
        ]
    )
    weaver.process_remaining()

    profile = store.get_by_ids([MemoryEntry.entity_profile_id("Melanie")])[0]
    assert "Painting hobby" in profile.lossless_restatement
    assert "[t1]" not in profile.lossless_restatement
    # Call A still needs the id in its catalogue.
    assert load_fabric(store).threads["t1"].one_line().startswith("[t1] ")


def test_unchanged_profile_is_not_rewritten(store):
    """A speaker whose threads did not change costs no profile re-embedding."""
    llm = ScriptedLLM(
        assignment=[
            assignment(([1], "new:Painting hobby")),
            assignment(([1], "existing:t1")),
        ],
        thread_update=[
            thread_update([fact("Melanie paints.", persons=["Melanie"])],
                          summary="Melanie paints watercolours."),
            # Second session changes nothing about the thread summary.
            thread_update([fact("Melanie mentioned paint prices.", persons=["Melanie"])],
                          summary="ignored",
                          impact="none"),
        ],
    )
    weaver = build_weaver(store, llm)

    weaver.add_dialogues(
        [Dialogue(dialogue_id=1, speaker="Melanie", content="I paint",
                  timestamp="1:00 pm on 1 May, 2023")]
    )
    weaver.add_dialogues(
        [Dialogue(dialogue_id=2, speaker="Melanie", content="prices",
                  timestamp="1:00 pm on 8 June, 2023")]
    )
    weaver.process_remaining()

    assert weaver.stats["profiles_written"] == 1
    assert weaver.stats["profiles_unchanged"] == 1
    profiles = [
        entry for entry in store.get_all_entries()
        if entry.kind == KIND_ENTITY_PROFILE
    ]
    assert len(profiles) == 1
    assert profiles[0].valid_from == "2023-05-01", "kept row keeps its own date"


# ----------------------------------------------------------------------
# P2 - one-hop expansion (design doc section 5)
# ----------------------------------------------------------------------


def _seed_fabric_neighbourhood(store):
    """A supersede chain, a bridge edge, a thread summary and an unrelated fact."""
    store.add_entries(
        [
            MemoryEntry(
                entry_id="old",
                lossless_restatement="Alice drinks coffee every morning",
                thread_id="t1",
                valid_from="2023-05-01",
                valid_until="2023-06-08",
                superseded_by="current",
            ),
            MemoryEntry(
                entry_id="current",
                lossless_restatement="Alice stopped drinking coffee",
                thread_id="t1",
                valid_from="2023-06-08",
                links=["bridge:bridged"],
            ),
            MemoryEntry(
                entry_id="bridged",
                lossless_restatement="Melanie brought pottery mugs to the studio",
                thread_id="t2",
                valid_from="2023-06-01",
                links=["bridge:current"],
            ),
            MemoryEntry(
                entry_id=MemoryEntry.thread_summary_id("t1"),
                lossless_restatement="Alice's coffee habit and how it ended",
                topic="Coffee habit",
                kind=KIND_THREAD_SUMMARY,
                thread_id="t1",
                # A later session, so the anchor is strictly after the closure and
                # "old" really is filtered out of the base pool.
                valid_from="2023-07-05",
            ),
            MemoryEntry(
                entry_id="unrelated",
                lossless_restatement="Nate repaired a tractor",
                thread_id="t3",
                valid_from="2023-05-05",
            ),
        ]
    )


def test_one_hop_expansion_walks_every_edge_type(store):
    _seed_fabric_neighbourhood(store)
    anchor = store.get_by_ids(["current"])

    pool = expand_one_hop(store, anchor)

    by_id = {entry.entry_id: entry for entry in pool.entries}
    assert set(by_id) == {"current", "old", "bridged", "thread::t1"}
    assert "unrelated" not in by_id
    # supersede chain backwards, weave edge, structural membership
    assert pool.provenance["old"].edge == EDGE_SUPERSEDES
    assert pool.provenance["old"].anchor_id == "current"
    assert pool.provenance["bridged"].edge == "bridge"
    assert pool.provenance["thread::t1"].edge == EDGE_THREAD
    assert "current" not in pool.provenance, "anchors carry no provenance"
    assert pool.counts() == {EDGE_SUPERSEDES: 1, "bridge": 1, EDGE_THREAD: 1}


def test_one_hop_expansion_follows_the_chain_forwards(store):
    _seed_fabric_neighbourhood(store)

    pool = expand_one_hop(store, store.get_by_ids(["old"]))

    assert pool.provenance["current"].edge == EDGE_SUPERSEDED_BY
    assert pool.provenance["current"].anchor_id == "old"


def test_expansion_is_one_hop_only(store):
    _seed_fabric_neighbourhood(store)
    store.add_entries(
        [
            MemoryEntry(
                entry_id="two_hops",
                lossless_restatement="A pottery kiln was installed",
                thread_id="t2",
                valid_from="2023-06-02",
                links=["refine:bridged"],
            )
        ]
    )
    store.update_metadata("bridged", {"links": ["bridge:current", "refine:two_hops"]})

    pool = expand_one_hop(store, store.get_by_ids(["current"]))

    # "bridged" arrives (one hop); its own neighbour does not.
    assert "bridged" in pool.provenance
    assert "two_hops" not in {entry.entry_id for entry in pool.entries}


def test_summaries_and_profiles_are_not_expanded_outwards(store):
    _seed_fabric_neighbourhood(store)
    summary = store.get_by_ids([MemoryEntry.thread_summary_id("t1")])

    pool = expand_one_hop(store, summary)

    assert [entry.entry_id for entry in pool.entries] == ["thread::t1"]
    assert pool.provenance == {}


def test_expansion_is_inert_on_baseline_entries(store):
    """No fabric fields means nothing to walk - and no empty-string match."""
    store.add_entries(
        [
            MemoryEntry(entry_id="b1", lossless_restatement="Alice drinks coffee"),
            MemoryEntry(entry_id="b2", lossless_restatement="Melanie paints"),
        ]
    )

    pool = expand_one_hop(store, store.get_by_ids(["b1"]))

    assert [entry.entry_id for entry in pool.entries] == ["b1"]
    assert pool.provenance == {}


def test_find_by_field_drops_empty_values_and_guards_the_field(store):
    _seed_fabric_neighbourhood(store)

    assert [e.entry_id for e in store.find_by_field("superseded_by", ["current"])] == [
        "old"
    ]
    # An empty value is the absence of an edge, not a wildcard.
    assert store.find_by_field("superseded_by", ["", None]) == []
    with pytest.raises(ValueError, match="Unknown metadata field"):
        store.find_by_field("not_a_field", ["x"])
    with pytest.raises(ValueError, match="Invalid lookup field"):
        store.backend.find_by_field("superseded_by = 'x' OR TRUE", ["y"])


def test_scoring_text_prefixes_only_expanded_entries(store):
    _seed_fabric_neighbourhood(store)
    pool = expand_one_hop(store, store.get_by_ids(["current"]))
    by_id = {entry.entry_id: entry for entry in pool.entries}

    anchor_text = scoring_text(by_id["current"], pool.provenance.get("current"))
    bridged_text = scoring_text(by_id["bridged"], pool.provenance.get("bridged"))

    assert anchor_text == "Alice stopped drinking coffee"
    assert bridged_text == (
        "[bridge of: Alice stopped drinking coffee] "
        "Melanie brought pottery mugs to the studio"
    )


# ----------------------------------------------------------------------
# P2 - cross-encoder rerank
# ----------------------------------------------------------------------


class FakeCrossEncoder:
    """Scores by keyword overlap, and records what it was asked to score."""

    def __init__(self, *args, **kwargs):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        scores = []
        for query, document in pairs:
            wanted = set(query.lower().split())
            scores.append(
                len(wanted & set(document.lower().split())) / max(len(wanted), 1)
            )
        return scores


def _fake_reranker(top_k=20):
    encoder = FakeCrossEncoder()
    reranker = CrossEncoderReranker(
        top_k=top_k, model_factory=lambda name: encoder
    )
    return reranker, encoder


def test_reranker_orders_by_score_and_applies_top_k():
    reranker, encoder = _fake_reranker(top_k=2)
    documents = ["nothing relevant here", "pottery class signup", "pottery mugs"]

    ranked, reranked = reranker.rerank(
        "pottery mugs", documents, to_text=lambda text: text
    )

    assert reranked is True
    assert ranked == ["pottery mugs", "pottery class signup"]
    assert len(encoder.pairs) == 3, "flat scoring: every candidate is scored once"


def test_reranker_falls_back_to_retrieval_order_when_unavailable():
    def explode(name):
        raise RuntimeError("no model here")

    reranker = CrossEncoderReranker(top_k=2, model_factory=explode)
    documents = ["first", "second", "third"]

    ranked, reranked = reranker.rerank("query", documents, to_text=lambda t: t)

    assert reranked is False
    assert ranked == ["first", "second"], "top_k still applies"
    # The failure is recorded once, not per query.
    reranker.rerank("query", documents, to_text=lambda t: t)
    assert reranker._unavailable is True
    assert reranker.available is False


def test_reranker_scores_expanded_entries_with_their_provenance(store):
    _seed_fabric_neighbourhood(store)
    reranker, encoder = _fake_reranker()
    retriever = _retriever(
        store, semantic_top_k=10, enable_expand_rerank=True, reranker=reranker
    )

    results = retriever.retrieve("Does Alice drink coffee?")

    scored = [document for _, document in encoder.pairs]
    # "old" is the entry the as-of filter removed and the chain edge brought back,
    # so it is the one carrying a provenance prefix here.
    assert (
        "[supersedes of: Alice stopped drinking coffee] "
        "Alice drinks coffee every morning"
    ) in scored
    # Entries the base retrieval already found are scored on their bare text.
    assert "Alice stopped drinking coffee" in scored
    assert "Melanie brought pottery mugs to the studio" in scored
    # Flat rerank: every candidate in the pool is scored exactly once.
    assert len(scored) == len(set(scored)) == len(results)


def test_expand_rerank_disabled_returns_the_base_pool(store):
    _seed_fabric_neighbourhood(store)
    reranker, encoder = _fake_reranker()
    retriever = _retriever(
        store, semantic_top_k=10, enable_expand_rerank=False, reranker=reranker
    )

    results = retriever.retrieve("Does Alice drink coffee?")

    assert encoder.pairs == []
    assert "old" not in {entry.entry_id for entry in results}


def test_state_question_recovers_history_through_expansion(store):
    """P2 changes the P0 property on purpose: history returns, annotated."""
    _seed_fabric_neighbourhood(store)
    reranker, _ = _fake_reranker()
    retriever = _retriever(
        store, semantic_top_k=10, enable_expand_rerank=True, reranker=reranker
    )

    results = retriever.retrieve("Does Alice drink coffee?")
    ids = {entry.entry_id for entry in results}

    # The as-of filter kept "old" out of the base pool; the chain edge brought it
    # back, which is what the answer-side annotation is for.
    assert "current" in ids and "old" in ids


# ----------------------------------------------------------------------
# P2 - supersede chain annotation in the answer context
# ----------------------------------------------------------------------


def _generator(annotate=True):
    return AnswerGenerator(llm_client=ScriptedLLM(), annotate_chains=annotate)


def _chain_contexts():
    return [
        MemoryEntry(
            entry_id="unrelated",
            lossless_restatement="Nate repaired a tractor",
            valid_from="2023-05-05",
        ),
        MemoryEntry(
            entry_id="current",
            lossless_restatement="Alice stopped drinking coffee",
            valid_from="2023-06-08",
        ),
        MemoryEntry(
            entry_id="old",
            lossless_restatement="Alice drinks coffee every morning",
            valid_from="2023-05-01",
            valid_until="2023-06-08",
            superseded_by="current",
        ),
    ]


def test_chain_members_are_placed_adjacently_oldest_first():
    ordered = AnswerGenerator._order_supersede_chains(_chain_contexts())

    assert [entry.entry_id for entry in ordered] == ["unrelated", "old", "current"]


def test_superseded_context_is_annotated_with_its_successor_position():
    formatted = _generator()._format_contexts(_chain_contexts())

    assert "[SUPERSEDED on 2023-06-08 by Context 3]" in formatted
    # The annotation points at the position the successor actually occupies.
    assert formatted.index("[Context 2]") < formatted.index("[SUPERSEDED")
    assert "Alice stopped drinking coffee" in formatted.split("[Context 3]")[1]


def test_closed_fact_without_its_successor_is_still_marked_stale():
    contexts = [
        MemoryEntry(
            entry_id="old",
            lossless_restatement="Alice drinks coffee every morning",
            valid_from="2023-05-01",
            valid_until="2023-06-08",
            superseded_by="absent",
        )
    ]

    formatted = _generator()._format_contexts(contexts)

    assert "[NO LONGER TRUE as of 2023-06-08]" in formatted


def test_annotation_disabled_keeps_the_native_context_format():
    formatted = _generator(annotate=False)._format_contexts(_chain_contexts())

    assert "SUPERSEDED" not in formatted
    assert "NO LONGER TRUE" not in formatted
    # Original order preserved.
    assert formatted.index("Nate repaired") < formatted.index("Alice stopped")


def test_annotation_is_inert_without_fabric_fields():
    contexts = [
        MemoryEntry(entry_id="b1", lossless_restatement="Alice drinks coffee"),
        MemoryEntry(entry_id="b2", lossless_restatement="Melanie paints"),
    ]

    formatted = _generator()._format_contexts(contexts)

    assert "SUPERSEDED" not in formatted
    assert AnswerGenerator._order_supersede_chains(contexts) == contexts


def test_converging_chains_stay_contiguous():
    """The sweep can close several facts with the same successor."""
    contexts = [
        MemoryEntry(entry_id="a", lossless_restatement="Alice drank filter coffee",
                    valid_from="2023-05-01", valid_until="2023-06-08",
                    superseded_by="now"),
        MemoryEntry(entry_id="filler", lossless_restatement="Nate repaired a tractor",
                    valid_from="2023-05-05"),
        MemoryEntry(entry_id="now", lossless_restatement="Alice stopped drinking coffee",
                    valid_from="2023-06-08"),
        MemoryEntry(entry_id="b", lossless_restatement="Alice drank espresso at work",
                    valid_from="2023-05-03", valid_until="2023-06-08",
                    superseded_by="now"),
    ]

    ordered = AnswerGenerator._order_supersede_chains(contexts)
    positions = {entry.entry_id: index for index, entry in enumerate(ordered)}

    assert len(ordered) == len(contexts)
    # Both predecessors sit immediately before their shared successor.
    block = sorted(positions[key] for key in ("a", "b", "now"))
    assert block == list(range(block[0], block[0] + 3))
    assert positions["now"] == block[-1], "successor last in its block"

    formatted = _generator()._format_contexts(contexts)
    successor_context = positions["now"] + 1
    assert formatted.count(f"[SUPERSEDED on 2023-06-08 by Context {successor_context}]") == 2


def test_cyclic_links_do_not_hang_the_layout():
    contexts = [
        MemoryEntry(entry_id="x", lossless_restatement="one",
                    valid_until="2023-06-01", superseded_by="y"),
        MemoryEntry(entry_id="y", lossless_restatement="two",
                    valid_until="2023-06-02", superseded_by="x"),
    ]

    ordered = AnswerGenerator._order_supersede_chains(contexts)

    assert [entry.entry_id for entry in ordered] == ["x", "y"]


# ----------------------------------------------------------------------
# P2 - expansion and rerank switch independently
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "expansion, rerank, expect_expanded, expect_scored",
    [
        (True, True, True, True),      # full P2 stage
        (True, False, True, False),    # C3 only: expansion without the reranker
        (False, True, False, True),    # parity config: reranker without expansion
        (False, False, False, False),  # stage off
    ],
)
def test_expansion_and_rerank_are_independently_switchable(
    store, expansion, rerank, expect_expanded, expect_scored
):
    _seed_fabric_neighbourhood(store)
    reranker, encoder = _fake_reranker()
    retriever = _retriever(
        store,
        semantic_top_k=10,
        enable_expand_rerank=True,
        enable_expansion=expansion,
        enable_rerank=rerank,
        reranker=reranker,
    )

    results = retriever.retrieve("Does Alice drink coffee?")
    ids = {entry.entry_id for entry in results}

    # "old" is only reachable through the supersede edge.
    assert ("old" in ids) is expect_expanded
    assert bool(encoder.pairs) is expect_scored
    assert retriever.enable_expansion is expansion
    assert retriever.enable_rerank is rerank


def test_compound_switch_overrides_both_parts(store):
    _seed_fabric_neighbourhood(store)
    reranker, encoder = _fake_reranker()
    retriever = _retriever(
        store,
        semantic_top_k=10,
        enable_expand_rerank=False,
        enable_expansion=True,
        enable_rerank=True,
        reranker=reranker,
    )

    results = retriever.retrieve("Does Alice drink coffee?")

    assert retriever.enable_expansion is False
    assert retriever.enable_rerank is False
    assert encoder.pairs == []
    assert "old" not in {entry.entry_id for entry in results}


def test_rerank_only_still_applies_the_capacity_constant(store):
    """The parity config keeps top_k, so context size matches the other arm."""
    _seed_fabric_neighbourhood(store)
    reranker, _ = _fake_reranker(top_k=2)
    retriever = _retriever(
        store,
        semantic_top_k=10,
        enable_expand_rerank=True,
        enable_expansion=False,
        enable_rerank=True,
        reranker=reranker,
    )

    assert len(retriever.retrieve("Does Alice drink coffee?")) == 2


def test_expansion_only_keeps_the_capacity_constant(store):
    """Ablating the cross-encoder must not also change the context size.

    Otherwise the --no-rerank row compares "no ranking" against "20 contexts vs
    the whole pool" and attributes the difference to the reranker.
    """
    _seed_fabric_neighbourhood(store)
    reranker, encoder = _fake_reranker(top_k=2)
    retriever = _retriever(
        store,
        semantic_top_k=10,
        enable_expand_rerank=True,
        enable_expansion=True,
        enable_rerank=False,
        reranker=reranker,
    )

    results = retriever.retrieve("Does Alice drink coffee?")

    assert encoder.pairs == [], "no cross-encoder was consulted"
    assert len(results) == 2, "retrieval order, truncated to top_k"


def test_whole_stage_off_keeps_the_native_simplemem_pool(store):
    """--no-expand-rerank is the baseline read path: no top-k stage at all."""
    _seed_fabric_neighbourhood(store)
    reranker, encoder = _fake_reranker(top_k=2)
    retriever = _retriever(
        store,
        semantic_top_k=10,
        enable_expand_rerank=False,
        reranker=reranker,
    )

    results = retriever.retrieve("Does Alice drink coffee?")

    assert encoder.pairs == []
    assert len(results) > 2, "the base pool is handed over untouched"
