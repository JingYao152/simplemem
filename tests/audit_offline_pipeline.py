"""Offline audit of the MemWeaver P0-P2 pipeline: paths the unit tests do not reach.

    python tests/audit_offline_pipeline.py

Exercises whole-system wiring with fakes only - no API, no embedding model, no
cross-encoder - so every failure here is a defect discoverable without running a
model: baseline-arm end to end, the packaged text system, the full ablation flag
matrix, storage robustness (large id lists, quoted ids, missing rows), recursion
and concurrency limits, and the harness metric plumbing.

Kept as a script rather than pytest cases because it drives SimpleMemSystem with
module-level monkeypatching, which does not compose with the unit-test fixtures.
"""
import json
import os
import sys
import threading
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name):
    def decorator(fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as error:
            FAILURES.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}: {type(error).__name__}: {error}")
    return decorator


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeEmbedder:
    dimension = 8
    model_name = "fake-embedder"
    model_type = "fake"

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def encode_documents(self, texts):
        self.calls += 1
        return np.stack([self._encode(t) for t in texts])

    def encode_single(self, text, is_query=False):
        return self._encode(text)

    @classmethod
    def _encode(cls, text):
        vec = np.zeros(cls.dimension, dtype=np.float32)
        for index, token in enumerate(str(text).lower().split()):
            vec[(len(token) + index) % cls.dimension] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


class FakeLLM:
    """Answers every prompt type the baseline + MemWeaver pipelines can emit."""

    def __init__(self, *args, **kwargs):
        self.prompts = []
        from simplemem.core.utils.llm_client import LLMClient
        self._extract = LLMClient.extract_json.__get__(self)
        self._clean_json_string = LLMClient._clean_json_string.__get__(self)
        self._extract_balanced_json = LLMClient._extract_balanced_json.__get__(self)

    def extract_json(self, text):
        return self._extract(text)

    def chat_completion(self, messages, temperature=0.2, response_format=None, max_retries=3):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)

        if "Assign every turn of this session" in prompt:
            turns = prompt.split("[Session turns]")[1]
            import re
            numbers = [int(m) for m in re.findall(r"^(\d+)\. \[", turns, re.M)]
            return json.dumps({"assignments": [{"turns": numbers, "thread": "new:Topic"}]})
        if "[Pairs]" in prompt:
            return json.dumps({"judgements": []})
        if "[Thread]" in prompt:
            return json.dumps({
                "facts": [{
                    "lossless_restatement": "Alice stopped drinking coffee on 2023-06-08.",
                    "keywords": ["Alice", "coffee"], "timestamp": None, "location": None,
                    "persons": ["Alice"], "entities": [], "topic": "coffee",
                    "weave": {"op": "none", "target": None},
                }],
                "summary": "Alice's coffee habit changed.",
                "summary_impact": "major",
                "outdated_facts": [],
            })
        # Baseline extraction (MemoryBuilder)
        if "extract all valuable information" in prompt:
            return json.dumps([{
                "lossless_restatement": "Alice drinks coffee every morning.",
                "keywords": ["Alice", "coffee"], "timestamp": None, "location": None,
                "persons": ["Alice"], "entities": [], "topic": "coffee",
            }])
        # Retrieval planning
        if "determine what specific information is required" in prompt:
            return json.dumps({"question_type": "factual", "key_entities": ["coffee"],
                               "required_info": [{"info_type": "f", "description": "d",
                                                  "priority": "high"}],
                               "relationships": [], "minimal_queries_needed": 1})
        if "information requirements analysis" in prompt:
            return json.dumps({"reasoning": "one query", "queries": ["coffee"]})
        if "extract key information" in prompt:
            return json.dumps({"keywords": ["coffee"], "persons": ["Alice"],
                               "time_expression": None, "location": None, "entities": []})
        if "sufficient information to answer" in prompt or "completely answer" in prompt:
            return json.dumps({"assessment": "complete", "reasoning": "ok",
                               "missing_info_types": [], "coverage_percentage": 100})
        # Answer generation
        if "Answer the user's question" in prompt:
            return json.dumps({"reasoning": "from context", "answer": "No coffee"})
        raise AssertionError(f"unhandled prompt: {prompt[:100]}")


class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        self.pairs = []
        self.lock = threading.Lock()

    def predict(self, pairs):
        with self.lock:
            self.pairs.extend(pairs)
        return [float(len(d) % 7) for _, d in pairs]


def patched_system(tmp, **kwargs):
    """SimpleMemSystem with fakes injected at the module level."""
    import main
    main.EmbeddingModel = FakeEmbedder
    main.LLMClient = FakeLLM
    from simplemem.core.reranker import CrossEncoderReranker
    system = main.SimpleMemSystem(db_path=tmp, table_name="entries", clear_db=True, **kwargs)
    system.hybrid_retriever.reranker = CrossEncoderReranker(
        top_k=20, model_factory=lambda name: FakeCrossEncoder()
    )
    return system


import tempfile
BASE = tempfile.mkdtemp(prefix="memweaver-audit-")


# ----------------------------------------------------------------------
print("\n=== E. baseline arm end to end (MemoryBuilder + P2 parity read) ===")


@check("baseline arm writes, retrieves and answers with P2 on")
def _():
    from simplemem.core.models.memory_entry import Dialogue
    system = patched_system(f"{BASE}/e1", enable_memweaver=False)
    assert type(system.writer).__name__ == "MemoryBuilder"
    system.add_dialogues([
        Dialogue(dialogue_id=i, speaker="Alice", content=f"turn {i}",
                 timestamp="1:00 pm on 1 May, 2023")
        for i in range(1, 4)
    ])
    system.finalize()
    entries = system.get_all_memories()
    assert entries, "baseline must have written entries"
    # Baseline entries carry empty fabric fields.
    assert all(e.kind == "fact" and not e.thread_id and not e.valid_from for e in entries)
    contexts = system.hybrid_retriever.retrieve("Does Alice drink coffee?")
    assert contexts
    answer = system.answer_generator.generate_answer("Does Alice drink coffee?", contexts)
    assert answer == "No coffee", answer


@check("baseline arm: as-of anchor is empty and expansion is inert")
def _():
    from simplemem.core.models.memory_entry import Dialogue
    system = patched_system(f"{BASE}/e2", enable_memweaver=False)
    system.add_dialogues([Dialogue(dialogue_id=1, speaker="Alice", content="hi",
                                   timestamp="1:00 pm on 1 May, 2023")])
    system.finalize()
    assert system.hybrid_retriever._resolve_as_of("What now?") == ""
    assert system.hybrid_retriever._current_anchor() == ""
    from simplemem.core.memweaver.expansion import expand_one_hop
    pool = expand_one_hop(system.vector_store, system.get_all_memories())
    assert pool.provenance == {}, pool.provenance


@check("baseline arm answer prompt has no fabric annotations")
def _():
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    generator = AnswerGenerator(llm_client=FakeLLM(), annotate_chains=True)
    text = generator._format_contexts([
        MemoryEntry(entry_id="a", lossless_restatement="Alice drinks coffee"),
        MemoryEntry(entry_id="b", lossless_restatement="Bob paints"),
    ])
    assert "SUPERSEDED" not in text and "NO LONGER TRUE" not in text


# ----------------------------------------------------------------------
print("\n=== F. packaged text system (simplemem/text/system.py) ===")


@check("simplemem.text.system routes writes and finalize like main.py")
def _():
    import simplemem.text.system as ts
    ts.EmbeddingModel = FakeEmbedder
    ts.LLMClient = FakeLLM
    system = ts.SimpleMemSystem(db_path=f"{BASE}/f1", table_name="entries",
                                clear_db=True, enable_memweaver=True,
                                enable_reflection=False)
    assert type(system.writer).__name__ == "MemWeaver"
    assert system.memweaver is not None
    system.add_dialogue("Alice", "I drink coffee", "1:00 pm on 1 May, 2023")
    system.add_dialogue("Alice", "not anymore", "2:00 pm on 8 June, 2023")
    system.finalize()
    facts = [e for e in system.get_all_memories() if e.kind == "fact"]
    assert facts and all(e.thread_id for e in facts), "MemWeaver must own the write path"
    assert system.memweaver.stats["sessions"] == 2


@check("simplemem.text.system baseline path still works")
def _():
    import simplemem.text.system as ts
    ts.EmbeddingModel = FakeEmbedder
    ts.LLMClient = FakeLLM
    system = ts.SimpleMemSystem(db_path=f"{BASE}/f2", table_name="entries",
                                clear_db=True, enable_memweaver=False)
    assert type(system.writer).__name__ == "MemoryBuilder"
    assert system.memweaver is None
    system.add_dialogue("Alice", "hello", "1:00 pm on 1 May, 2023")
    system.finalize()
    assert system.get_all_memories()


# ----------------------------------------------------------------------
print("\n=== G. ablation flag propagation through the real system ===")


@check("every ablation flag reaches the component that implements it")
def _():
    matrix = [
        (dict(enable_memweaver=True), lambda s: s.memweaver is not None),
        (dict(enable_memweaver=False), lambda s: s.memweaver is None),
        (dict(enable_memweaver=True, enable_weaving=False),
         lambda s: s.memweaver.enable_weaving is False),
        (dict(enable_memweaver=True, enable_sweep=False),
         lambda s: s.memweaver.enable_sweep is False),
        (dict(enable_memweaver=True, enable_recontext=False),
         lambda s: s.memweaver.enable_recontext is False),
        (dict(enable_memweaver=True, enable_entity_profiles=False),
         lambda s: s.memweaver.enable_entity_profiles is False),
        (dict(enable_memweaver=True, enable_expansion=False),
         lambda s: s.hybrid_retriever.enable_expansion is False
                   and s.hybrid_retriever.enable_rerank is True
                   and s.answer_generator.annotate_chains is False),
        (dict(enable_memweaver=True, enable_rerank=False),
         lambda s: s.hybrid_retriever.enable_rerank is False
                   and s.hybrid_retriever.enable_expansion is True
                   and s.answer_generator.annotate_chains is True),
        (dict(enable_memweaver=True, enable_expand_rerank=False),
         lambda s: not s.hybrid_retriever.enable_expansion
                   and not s.hybrid_retriever.enable_rerank
                   and s.answer_generator.annotate_chains is False),
        (dict(enable_memweaver=False), lambda s: s.hybrid_retriever.enable_rerank is True),
    ]
    for index, (flags, predicate) in enumerate(matrix):
        system = patched_system(f"{BASE}/g{index}", enable_reflection=False, **flags)
        assert predicate(system), f"flags {flags} did not propagate"


@check("degenerate MemWeaver config (every mechanism off) still writes facts")
def _():
    from simplemem.core.models.memory_entry import Dialogue
    system = patched_system(
        f"{BASE}/g_off", enable_memweaver=True, enable_weaving=False,
        enable_sweep=False, enable_recontext=False, enable_entity_profiles=False,
        enable_expand_rerank=False, enable_reflection=False,
    )
    system.add_dialogues([Dialogue(dialogue_id=1, speaker="Alice", content="hi",
                                   timestamp="1:00 pm on 1 May, 2023")])
    system.finalize()
    facts = [e for e in system.get_all_memories() if e.kind == "fact"]
    assert facts, "threads-only config must still produce facts"
    assert all(e.context_digest == "" for e in facts), "recontext off = no digest"
    assert not [e for e in system.get_all_memories() if e.kind == "entity_profile"]


# ----------------------------------------------------------------------
print("\n=== H-J. storage robustness ===")


@check("print_memories renders fabric entries without crashing")
def _():
    import io
    import contextlib
    from simplemem.core.models.memory_entry import Dialogue
    system = patched_system(f"{BASE}/h1", enable_memweaver=True, enable_reflection=False)
    system.add_dialogues([
        Dialogue(dialogue_id=1, speaker="Alice", content="coffee",
                 timestamp="1:00 pm on 1 May, 2023"),
        Dialogue(dialogue_id=2, speaker="Alice", content="no more",
                 timestamp="2:00 pm on 8 June, 2023"),
    ])
    system.finalize()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        system.print_memories()
    assert "All Memory Entries" in buffer.getvalue()


@check("get_by_ids handles a large id list")
def _():
    from simplemem.core.database.vector_store import VectorStore
    from simplemem.core.models.memory_entry import MemoryEntry
    store = VectorStore(db_path=f"{BASE}/j1", table_name="entries",
                        embedding_model=FakeEmbedder())
    store.clear()
    ids = [f"e{i}" for i in range(2000)]
    store.add_entries([MemoryEntry(entry_id=i, lossless_restatement=f"fact {i}")
                       for i in ids])
    fetched = store.get_by_ids(ids)
    assert len(fetched) == 2000, len(fetched)
    assert [e.entry_id for e in fetched] == ids, "order must follow the request"


@check("ids and values containing quotes cannot break the SQL")
def _():
    from simplemem.core.database.vector_store import VectorStore
    from simplemem.core.models.memory_entry import MemoryEntry
    store = VectorStore(db_path=f"{BASE}/j2", table_name="entries",
                        embedding_model=FakeEmbedder())
    store.clear()
    nasty = "id' OR 1=1 --"
    store.add_entries([
        MemoryEntry(entry_id=nasty, lossless_restatement="quoted id"),
        MemoryEntry(entry_id="plain", lossless_restatement="plain id",
                    superseded_by=nasty),
    ])
    assert [e.entry_id for e in store.get_by_ids([nasty])] == [nasty]
    assert [e.entry_id for e in store.find_by_field("superseded_by", [nasty])] == ["plain"]
    store.update_metadata(nasty, {"valid_until": "2023-01-01"})
    assert store.get_by_ids([nasty])[0].valid_until == "2023-01-01"
    store.delete_by_ids([nasty])
    assert [e.entry_id for e in store.get_all_entries()] == ["plain"]


@check("reembed / update_vector on a missing id is a silent no-op")
def _():
    from simplemem.core.database.vector_store import VectorStore
    from simplemem.core.models.memory_entry import MemoryEntry
    store = VectorStore(db_path=f"{BASE}/j3", table_name="entries",
                        embedding_model=FakeEmbedder())
    store.clear()
    store.add_entries([MemoryEntry(entry_id="real", lossless_restatement="here")])
    ghost = MemoryEntry(entry_id="ghost", lossless_restatement="gone")
    assert store.reembed_entries([ghost], ["[ctx] gone"], "d1") == 1
    assert len(store.get_all_entries()) == 1


# ----------------------------------------------------------------------
print("\n=== K-N. algorithmic edge cases ===")


@check("expand_one_hop does not mutate the caller's list")
def _():
    from simplemem.core.database.vector_store import VectorStore
    from simplemem.core.memweaver.expansion import expand_one_hop
    from simplemem.core.models.memory_entry import MemoryEntry
    store = VectorStore(db_path=f"{BASE}/k1", table_name="entries",
                        embedding_model=FakeEmbedder())
    store.clear()
    store.add_entries([
        MemoryEntry(entry_id="old", lossless_restatement="was true", thread_id="t1",
                    valid_from="2023-05-01", valid_until="2023-06-01",
                    superseded_by="new"),
        MemoryEntry(entry_id="new", lossless_restatement="is true", thread_id="t1",
                    valid_from="2023-06-01"),
    ])
    candidates = store.get_by_ids(["new"])
    before = list(candidates)
    pool = expand_one_hop(store, candidates)
    assert candidates == before, "input list was mutated"
    assert len(pool.entries) == 2


@check("a long supersede chain does not blow the recursion limit")
def _():
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    length = 400
    contexts = [
        MemoryEntry(entry_id=f"c{i}", lossless_restatement=f"state {i}",
                    valid_from="2023-05-01",
                    valid_until="2023-06-01" if i < length - 1 else "",
                    superseded_by=f"c{i + 1}" if i < length - 1 else "")
        for i in range(length)
    ]
    ordered = AnswerGenerator._order_supersede_chains(contexts)
    assert [e.entry_id for e in ordered] == [f"c{i}" for i in range(length)]


@check("chain layout is stable when a middle member is missing")
def _():
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    contexts = [
        MemoryEntry(entry_id="a", lossless_restatement="first",
                    valid_until="2023-06-01", superseded_by="b_missing"),
        MemoryEntry(entry_id="c", lossless_restatement="third", valid_from="2023-07-01"),
    ]
    ordered = AnswerGenerator._order_supersede_chains(contexts)
    assert [e.entry_id for e in ordered] == ["a", "c"]
    text = AnswerGenerator(llm_client=FakeLLM(), annotate_chains=True)._format_contexts(contexts)
    assert "[NO LONGER TRUE as of 2023-06-01]" in text


@check("concurrent retrieve() on one retriever is safe (harness uses 16 workers)")
def _():
    import concurrent.futures
    from simplemem.core.models.memory_entry import Dialogue
    system = patched_system(f"{BASE}/n1", enable_memweaver=True, enable_reflection=False)
    system.add_dialogues([
        Dialogue(dialogue_id=1, speaker="Alice", content="coffee",
                 timestamp="1:00 pm on 1 May, 2023"),
        Dialogue(dialogue_id=2, speaker="Alice", content="no more",
                 timestamp="2:00 pm on 8 June, 2023"),
    ])
    system.finalize()

    def ask(index):
        return system.hybrid_retriever.retrieve(f"question {index} about coffee")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(ask, range(32)))
    assert all(results), "some concurrent retrievals returned nothing"
    lengths = {len(r) for r in results}
    assert len(lengths) == 1, f"concurrent retrievals disagreed on size: {lengths}"
    # the anchor cache must not be corrupted by concurrent access
    assert system.hybrid_retriever._current_anchor() == "2023-06-08"


@check("MemWeaver handles undated and single-turn sessions")
def _():
    from simplemem.core.models.memory_entry import Dialogue
    system = patched_system(f"{BASE}/n2", enable_memweaver=True, enable_reflection=False)
    system.add_dialogues([Dialogue(dialogue_id=1, speaker="Alice", content="no date",
                                   timestamp=None)])
    system.finalize()
    facts = [e for e in system.get_all_memories() if e.kind == "fact"]
    assert facts, "an undated session must still be written"
    assert facts[0].valid_from == "", "no date parses to no validity start"


@check("empty input does not crash finalize")
def _():
    system = patched_system(f"{BASE}/n3", enable_memweaver=True, enable_reflection=False)
    system.add_dialogues([])
    system.finalize()
    assert system.get_all_memories() == []
    assert system.memweaver.stats["sessions"] == 0


# ----------------------------------------------------------------------
print("\n=== O. harness metric plumbing with fabric entries ===")


@check("EvidenceScorer tolerates summaries and profiles in the contexts")
def _():
    from locomo_evidence import EvidenceScorer
    scorer = EvidenceScorer([
        (1, "D1:1", "I went to a LGBTQ support group yesterday and it was powerful"),
    ])

    class Entry:
        def __init__(self, entry_id, text, keywords=()):
            self.entry_id = entry_id
            self.lossless_restatement = text
            self.keywords = list(keywords)

    metrics = scorer.score(["D1:1"], [
        Entry("thread::t1", "Caroline - ongoing threads: LGBTQ advocacy"),
        Entry("profile::Caroline", "Caroline - ongoing threads across the conversation"),
        Entry("f1", "Caroline attended an LGBTQ support group", ["LGBTQ"]),
    ])
    assert set(metrics) and metrics["retrieval_evidence_count"] == 1.0
    assert metrics["retrieval_hit_any"] in (0.0, 1.0)


print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S)")
    for name, tb in FAILURES:
        print(f"\n--- {name} ---\n{tb}")
    sys.exit(1)
print("ALL AUDIT CHECKS PASSED")
