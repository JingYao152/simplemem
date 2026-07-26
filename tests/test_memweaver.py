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
from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.memweaver import MemWeaver
from simplemem.core.memweaver.asof import (
    apply_asof,
    asof_filters,
    compute_anchor,
    is_temporal_history_question,
)
from simplemem.core.memweaver.dates import parse_session_datetime, to_day
from simplemem.core.memweaver.fabric import load_fabric
from simplemem.core.models.memory_entry import (
    KIND_FACT,
    KIND_THREAD_SUMMARY,
    Dialogue,
    MemoryEntry,
)
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
