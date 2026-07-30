"""Materialized set-view maintenance for cross-session aggregation.

After each session's facts are written, the writer calls :func:`maintain_sets`
to create or update set-view entries for every distinct ``set_key`` produced
by the session.  A set-view entry is a single :class:`MemoryEntry` with
``kind=KIND_SET_VIEW`` whose ``lossless_restatement`` is a deterministic
verbatim enumeration of all member facts' texts.
"""

from typing import Dict, List, Optional, Sequence

from simplemem.core.database.vector_store import VectorStore
from simplemem.core.models.memory_entry import (
    KIND_FACT,
    KIND_SET_VIEW,
    MemoryEntry,
)


#: Maximum character budget for a set-view's ``lossless_restatement``.
#: When the enumeration exceeds this, it is truncated and a provenance line
#: listing omitted fact IDs is appended.
SET_TEXT_BUDGET = 4000


def _set_label(set_key: str) -> str:
    """Human-readable label used as the set's embedding text.

    ``"Melanie:camping"`` -> ``"Melanie's camping activities"``
    """
    if ":" in set_key:
        entity, predicate = set_key.split(":", 1)
        return f"{entity}'s {predicate} activities"
    return set_key


def _build_set_text(member_facts: Sequence[MemoryEntry]) -> str:
    """Assemble the verbatim enumeration of member fact texts.

    If the total exceeds :data:`SET_TEXT_BUDGET`, truncate and append a
    provenance line listing the omitted fact IDs.
    """
    lines: List[str] = []
    total = 0
    omitted: List[str] = []
    for fact in member_facts:
        line = f"- {fact.lossless_restatement}"
        if total + len(line) + 1 > SET_TEXT_BUDGET:
            omitted.append(fact.entry_id)
            continue
        lines.append(line)
        total += len(line) + 1

    text = "\n".join(lines)
    if omitted:
        text += f"\n[truncated: {len(omitted)} fact(s) omitted — IDs: {', '.join(omitted)}]"
    return text


def _load_existing_set(
    vector_store: VectorStore,
    set_key: str,
) -> Optional[MemoryEntry]:
    """Load the existing set-view entry by its fixed ID, if any."""
    set_id = MemoryEntry.set_view_id(set_key)
    entries = vector_store.get_by_ids([set_id])
    if not entries:
        return None
    entry = entries[0]
    if entry.kind != KIND_SET_VIEW:
        return None
    return entry


def _collect_member_facts(
    vector_store: VectorStore,
    set_key: str,
) -> List[MemoryEntry]:
    """Return all facts with the given ``set_key``, in chronological order."""
    all_entries = vector_store.get_all_entries()
    members = [
        entry
        for entry in all_entries
        if entry.kind == KIND_FACT and entry.set_key == set_key
    ]
    members.sort(key=lambda f: (f.valid_from or "", f.entry_id))
    return members


def maintain_sets(
    vector_store: VectorStore,
    new_facts: Sequence[MemoryEntry],
) -> Dict[str, int]:
    """Create or update set-view entries for every distinct set_key in *new_facts*.

    Returns a stats dict with counts: ``sets_created``, ``sets_updated``,
    ``members_added``, ``text_truncated``.
    """
    stats = {
        "sets_created": 0,
        "sets_updated": 0,
        "members_added": 0,
        "text_truncated": 0,
    }

    set_keys = {
        fact.set_key
        for fact in new_facts
        if fact.set_key and fact.kind == KIND_FACT
    }
    if not set_keys:
        return stats

    for set_key in set_keys:
        existing = _load_existing_set(vector_store, set_key)
        member_facts = _collect_member_facts(vector_store, set_key)
        if not member_facts:
            continue

        member_ids = [f.entry_id for f in member_facts]
        set_text = _build_set_text(member_facts)
        truncated = "[truncated:" in set_text

        entry = MemoryEntry(
            entry_id=MemoryEntry.set_view_id(set_key),
            lossless_restatement=set_text,
            keywords=[],
            timestamp=None,
            location=None,
            persons=member_facts[0].persons if member_facts else [],
            entities=[],
            topic=_set_label(set_key),
            kind=KIND_SET_VIEW,
            thread_id="",
            valid_from=member_facts[0].valid_from if member_facts else "",
            valid_until="",
            superseded_by="",
            links=[],
            context_digest="",
            set_key=set_key,
            set_member_ids=member_ids,
        )

        # Upsert: delete old row (if any) then insert the new one.
        if existing is not None:
            vector_store.delete_by_ids([existing.entry_id])
            stats["sets_updated"] += 1
        else:
            stats["sets_created"] += 1

        # Embed with the short label, not the full enumeration.
        vector_store.add_entries([entry], embed_texts=[_set_label(set_key)])

        new_member_count = len(member_ids) - (
            len(existing.set_member_ids) if existing else 0
        )
        stats["members_added"] += max(0, new_member_count)
        if truncated:
            stats["text_truncated"] += 1

    return stats
