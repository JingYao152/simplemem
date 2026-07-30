"""Unit tests for materialized set-view maintenance and retrieval routing.

These tests use an in-memory fake vector store so they run without LanceDB.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pytest

from simplemem.core.models.memory_entry import (
    KIND_FACT,
    KIND_SET_VIEW,
    MemoryEntry,
)
from simplemem.core.memweaver.set_views import (
    maintain_sets,
    _build_set_text,
    _set_label,
    SET_TEXT_BUDGET,
)


# ---------------------------------------------------------------------------
# Fake vector store — just enough to test set-view logic
# ---------------------------------------------------------------------------

class FakeVectorStore:
    """Minimal in-memory store that supports the operations set_views uses."""

    def __init__(self):
        self._entries: Dict[str, MemoryEntry] = {}
        self.revision = 0

    def add_entries(self, entries, embed_texts=None):
        for e in entries:
            self._entries[e.entry_id] = e
        self.revision += 1

    def delete_by_ids(self, ids):
        for i in ids:
            self._entries.pop(i, None)
        self.revision += 1

    def get_by_ids(self, ids):
        return [self._entries[i] for i in ids if i in self._entries]

    def get_all_entries(self):
        return list(self._entries.values())

    def clear(self):
        self._entries.clear()
        self.revision += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fact(entry_id, text, set_key="", valid_from="2024-01-01", persons=None):
    return MemoryEntry(
        entry_id=entry_id,
        lossless_restatement=text,
        kind=KIND_FACT,
        set_key=set_key,
        valid_from=valid_from,
        persons=persons or [],
    )


# ---------------------------------------------------------------------------
# _set_label
# ---------------------------------------------------------------------------

class TestSetLabel:
    def test_colon_format(self):
        assert _set_label("Melanie:camping") == "Melanie's camping activities"

    def test_no_colon(self):
        assert _set_label("something") == "something"


# ---------------------------------------------------------------------------
# _build_set_text
# ---------------------------------------------------------------------------

class TestBuildSetText:
    def test_single_member(self):
        facts = [make_fact("f1", "Melanie went camping at Yosemite")]
        text = _build_set_text(facts)
        assert "- Melanie went camping at Yosemite" in text
        assert "truncated" not in text

    def test_multiple_members(self):
        facts = [
            make_fact("f1", "Melanie went camping at Yosemite", valid_from="2024-01-01"),
            make_fact("f2", "Melanie camped at Yellowstone", valid_from="2024-03-15"),
        ]
        text = _build_set_text(facts)
        assert "Yosemite" in text
        assert "Yellowstone" in text
        assert "truncated" not in text

    def test_truncation(self):
        """When the enumeration exceeds the budget, it is truncated."""
        long_text = "x" * (SET_TEXT_BUDGET // 2 + 100)
        facts = [
            make_fact("f1", long_text, valid_from="2024-01-01"),
            make_fact("f2", long_text, valid_from="2024-02-01"),
        ]
        text = _build_set_text(facts)
        assert "[truncated:" in text
        assert "f2" in text  # omitted ID is listed


# ---------------------------------------------------------------------------
# maintain_sets
# ---------------------------------------------------------------------------

class TestMaintainSets:
    def test_create_new_set(self):
        vs = FakeVectorStore()
        # The fact must already be in the store so _collect_member_facts finds it.
        fact = make_fact("f1", "Melanie went camping at Yosemite", set_key="Melanie:camping")
        vs.add_entries([fact])

        stats = maintain_sets(vs, [fact])
        assert stats["sets_created"] == 1
        assert stats["sets_updated"] == 0

        set_entry = vs.get_by_ids(["set_view::Melanie:camping"])[0]
        assert set_entry.kind == KIND_SET_VIEW
        assert set_entry.set_key == "Melanie:camping"
        assert "f1" in set_entry.set_member_ids
        assert "Yosemite" in set_entry.lossless_restatement
        assert set_entry.topic == "Melanie's camping activities"

    def test_incremental_update(self):
        vs = FakeVectorStore()
        fact1 = make_fact("f1", "Melanie went camping at Yosemite", set_key="Melanie:camping", valid_from="2024-01-01")
        vs.add_entries([fact1])
        maintain_sets(vs, [fact1])

        # Second session adds another camping fact.
        fact2 = make_fact("f2", "Melanie camped at Yellowstone", set_key="Melanie:camping", valid_from="2024-03-15")
        vs.add_entries([fact2])
        stats = maintain_sets(vs, [fact2])
        assert stats["sets_updated"] == 1
        assert stats["sets_created"] == 0

        set_entry = vs.get_by_ids(["set_view::Melanie:camping"])[0]
        assert "f1" in set_entry.set_member_ids
        assert "f2" in set_entry.set_member_ids
        assert "Yosemite" in set_entry.lossless_restatement
        assert "Yellowstone" in set_entry.lossless_restatement

    def test_no_set_key_facts_skipped(self):
        vs = FakeVectorStore()
        fact = make_fact("f1", "Some isolated event", set_key="")
        vs.add_entries([fact])
        stats = maintain_sets(vs, [fact])
        assert stats["sets_created"] == 0
        assert stats["sets_updated"] == 0

    def test_superseded_fact_remains(self):
        """A superseded fact should stay in the set."""
        vs = FakeVectorStore()
        fact1 = make_fact("f1", "Melanie went camping at Yosemite", set_key="Melanie:camping", valid_from="2024-01-01")
        fact1.valid_until = "2024-06-01"  # superseded
        fact1.superseded_by = "f2"
        fact2 = make_fact("f2", "Melanie went camping at Yellowstone", set_key="Melanie:camping", valid_from="2024-06-01")
        vs.add_entries([fact1, fact2])

        maintain_sets(vs, [fact2])
        set_entry = vs.get_by_ids(["set_view::Melanie:camping"])[0]
        # Both the superseded and the new fact should be in the set.
        assert "f1" in set_entry.set_member_ids
        assert "f2" in set_entry.set_member_ids
        assert "Yosemite" in set_entry.lossless_restatement
        assert "Yellowstone" in set_entry.lossless_restatement

    def test_multiple_distinct_sets(self):
        vs = FakeVectorStore()
        f1 = make_fact("f1", "Melanie went camping", set_key="Melanie:camping")
        f2 = make_fact("f2", "Bob started a project", set_key="Bob:work_project")
        vs.add_entries([f1, f2])

        stats = maintain_sets(vs, [f1, f2])
        assert stats["sets_created"] == 2

        s1 = vs.get_by_ids(["set_view::Melanie:camping"])[0]
        s2 = vs.get_by_ids(["set_view::Bob:work_project"])[0]
        assert "f1" in s1.set_member_ids
        assert "f2" in s2.set_member_ids
