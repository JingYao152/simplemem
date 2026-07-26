"""Fabric state: threads, facts and the write-side weaving execution.

The fabric lives entirely in the flat vector table. Thread state is therefore
re-derived from storage rather than cached in the writer, which keeps the writer
correct across ``vector_store.clear()`` (the LoCoMo harness clears per sample)
and makes every organization decision auditable from the stored rows alone.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional

from simplemem.core.database.vector_store import VectorStore
from simplemem.core.models.memory_entry import (
    KIND_FACT,
    KIND_THREAD_SUMMARY,
    WEAVE_BRIDGE,
    WEAVE_REFINE,
    MemoryEntry,
)


_THREAD_ID_PATTERN = re.compile(r"^t(\d+)$")


@dataclass
class ThreadState:
    """A living thread: its title and current summary entry."""

    thread_id: str
    title: str
    summary: str
    summary_entry_id: str = ""

    def one_line(self) -> str:
        """One-line rendering used in the Call A thread catalogue."""
        summary = " ".join((self.summary or "").split())
        if len(summary) > 240:
            summary = summary[:237] + "..."
        title = self.title or "(untitled)"
        return f"[{self.thread_id}] {title}" + (f" - {summary}" if summary else "")


@dataclass
class FabricSnapshot:
    """Read-only view of the fabric at the start of a session."""

    threads: Dict[str, ThreadState] = field(default_factory=dict)
    facts_by_thread: Dict[str, List[MemoryEntry]] = field(default_factory=dict)
    entries: List[MemoryEntry] = field(default_factory=list)

    def facts(self, thread_id: str) -> List[MemoryEntry]:
        return self.facts_by_thread.get(thread_id, [])

    def next_thread_id(self, taken: Optional[Iterable[str]] = None) -> str:
        """Allocate the next ``t<N>`` id, skipping ids already handed out."""
        used = set(self.threads) | set(taken or ())
        highest = 0
        for thread_id in used:
            match = _THREAD_ID_PATTERN.match(thread_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"t{highest + 1}"


def load_fabric(vector_store: VectorStore) -> FabricSnapshot:
    """Rebuild the fabric snapshot from stored entries."""
    entries = vector_store.get_all_entries()

    threads: Dict[str, ThreadState] = {}
    facts_by_thread: Dict[str, List[MemoryEntry]] = {}

    for entry in entries:
        if entry.kind == KIND_THREAD_SUMMARY and entry.thread_id:
            threads[entry.thread_id] = ThreadState(
                thread_id=entry.thread_id,
                title=entry.topic or "",
                summary=entry.lossless_restatement or "",
                summary_entry_id=entry.entry_id,
            )
        elif entry.kind == KIND_FACT and entry.thread_id:
            facts_by_thread.setdefault(entry.thread_id, []).append(entry)

    # Facts of a thread are numbered for the LLM in chronological order so the
    # weave candidate list reads as the thread's history.
    for facts in facts_by_thread.values():
        facts.sort(key=lambda fact: (fact.valid_from or "", fact.entry_id))

    # A thread can exist through its facts even if its summary write failed.
    for thread_id in facts_by_thread:
        threads.setdefault(
            thread_id, ThreadState(thread_id=thread_id, title="", summary="")
        )

    return FabricSnapshot(
        threads=threads, facts_by_thread=facts_by_thread, entries=entries
    )


def build_thread_summary_entry(
    thread_id: str,
    title: str,
    summary: str,
    session_date: str,
    session_datetime: str,
    persons: Optional[List[str]] = None,
) -> MemoryEntry:
    """Build the living summary entry for a thread (fixed id convention)."""
    return MemoryEntry(
        entry_id=MemoryEntry.thread_summary_id(thread_id),
        lossless_restatement=summary,
        keywords=[],
        timestamp=session_datetime or None,
        location=None,
        persons=sorted(set(persons or [])),
        entities=[],
        topic=title,
        kind=KIND_THREAD_SUMMARY,
        thread_id=thread_id,
        valid_from=session_date,
        valid_until="",
        superseded_by="",
        links=[],
        context_digest="",
    )


def execute_supersede(
    vector_store: VectorStore,
    old_entry: MemoryEntry,
    new_entry: MemoryEntry,
    session_date: str,
) -> bool:
    """Close ``old_entry`` at ``session_date`` and point it at its successor.

    Deterministic code, not an LLM decision (design doc section 3). Returns
    False when the old entry is already closed, so a later supersede never
    silently rewrites an existing chain link.
    """
    if not old_entry.is_open or old_entry.entry_id == new_entry.entry_id:
        return False

    vector_store.update_metadata(
        old_entry.entry_id,
        {
            "valid_until": session_date,
            "superseded_by": new_entry.entry_id,
        },
    )
    old_entry.valid_until = session_date
    old_entry.superseded_by = new_entry.entry_id
    return True


def execute_link(
    vector_store: VectorStore,
    edge_type: str,
    left: MemoryEntry,
    right: MemoryEntry,
) -> bool:
    """Write a refine/bridge edge on both endpoints (bidirectional)."""
    if edge_type not in (WEAVE_REFINE, WEAVE_BRIDGE):
        return False
    if left.entry_id == right.entry_id:
        return False

    wrote = False
    for source, target in ((left, right), (right, left)):
        edge = f"{edge_type}:{target.entry_id}"
        if edge in source.links:
            continue
        source.links = list(source.links) + [edge]
        vector_store.update_metadata(source.entry_id, {"links": source.links})
        wrote = True
    return wrote
