"""One-hop evidence expansion (design doc section 5, contribution C3).

The retrieval base returns a candidate set. Each candidate that is a *fact* is
then used as an anchor and walked outwards by exactly one hop along the edges the
write side created:

1. **supersede chain, both directions** - ``superseded_by`` forwards and the
   entries pointing at the anchor backwards, so a fact's history and its
   successor come along (temporal questions ask "what was it before").
2. **weave edges, both directions** - ``links`` already holds refine/bridge edges
   on both endpoints, so bridged facts across threads are reachable. A bridged
   fact has low direct similarity to the query - that is exactly why retrieval
   missed it.
3. **structural membership** - ``thread_id`` to the thread's living summary, for
   topic-level context.

Summaries and profiles are never expanded outwards: they are already aggregates.
One hop only, deduplicated into the pool, and every added entry records its
provenance (which anchor, which edge) so the reranker can score it in context.

This primitive exists only because of the fabric: no ``valid_until`` means no
chain, no weave edge means nothing to bridge. Evidence degree is structurally low
(1.42 evidence turns per question on LoCoMo), so there is no truncation
parameter.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from simplemem.core.database.vector_store import VectorStore
from simplemem.core.models.memory_entry import (
    KIND_FACT,
    WEAVE_BRIDGE,
    WEAVE_REFINE,
    MemoryEntry,
)


#: Edge labels used in provenance (and in the reranker's scoring prefix).
EDGE_SUPERSEDED_BY = "superseded by"
EDGE_SUPERSEDES = "supersedes"
EDGE_REFINE = WEAVE_REFINE
EDGE_BRIDGE = WEAVE_BRIDGE
EDGE_THREAD = "thread context"


@dataclass(frozen=True)
class Provenance:
    """Why an entry joined the pool: the anchor it hung off, and via which edge."""

    edge: str
    anchor_id: str
    anchor_text: str

    def prefix(self) -> str:
        """Short scoring prefix, e.g. ``[bridge of: Melanie bought brushes.]``."""
        anchor = " ".join((self.anchor_text or "").split())
        if len(anchor) > 160:
            anchor = anchor[:157] + "..."
        return f"[{self.edge} of: {anchor}]" if anchor else f"[{self.edge}]"


@dataclass
class ExpandedPool:
    """The candidate pool after one-hop expansion."""

    entries: List[MemoryEntry]
    provenance: Dict[str, Provenance]

    def counts(self) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        for record in self.provenance.values():
            tally[record.edge] = tally.get(record.edge, 0) + 1
        return tally


def expand_one_hop(
    vector_store: VectorStore,
    candidates: Sequence[MemoryEntry],
) -> ExpandedPool:
    """Add the one-hop fabric neighbourhood of every fact in ``candidates``."""
    entries: List[MemoryEntry] = list(candidates)
    if not entries:
        return ExpandedPool(entries=entries, provenance={})

    seen = {entry.entry_id for entry in entries}
    anchors = [entry for entry in entries if entry.kind == KIND_FACT]
    if not anchors:
        return ExpandedPool(entries=entries, provenance={})

    provenance: Dict[str, Provenance] = {}

    def admit(entry: MemoryEntry, edge: str, anchor: MemoryEntry) -> None:
        if entry.entry_id in seen:
            return
        seen.add(entry.entry_id)
        entries.append(entry)
        provenance[entry.entry_id] = Provenance(
            edge=edge,
            anchor_id=anchor.entry_id,
            anchor_text=anchor.lossless_restatement,
        )

    # (1) supersede chain, forwards: the successor of a closed fact.
    successor_ids = [anchor.superseded_by for anchor in anchors if anchor.superseded_by]
    for entry in _fetch_by_ids(vector_store, successor_ids):
        for anchor in anchors:
            if anchor.superseded_by == entry.entry_id:
                admit(entry, EDGE_SUPERSEDED_BY, anchor)
                break

    # (1) supersede chain, backwards: what this fact replaced.
    anchor_ids = [anchor.entry_id for anchor in anchors]
    for entry in _find_by_field(vector_store, "superseded_by", anchor_ids):
        for anchor in anchors:
            if entry.superseded_by == anchor.entry_id:
                admit(entry, EDGE_SUPERSEDES, anchor)
                break

    # (2) weave edges: written on both endpoints at weave time.
    weave_targets: Dict[str, List[Tuple[str, MemoryEntry]]] = {}
    for anchor in anchors:
        for edge_type in (WEAVE_REFINE, WEAVE_BRIDGE):
            for target_id in anchor.weave_targets(edge_type):
                weave_targets.setdefault(target_id, []).append((edge_type, anchor))
    for entry in _fetch_by_ids(vector_store, list(weave_targets)):
        edge_type, anchor = weave_targets[entry.entry_id][0]
        admit(entry, edge_type, anchor)

    # (3) structural membership: the thread's living summary (fixed id).
    summary_owners: Dict[str, MemoryEntry] = {}
    for anchor in anchors:
        if not anchor.thread_id:
            continue
        summary_owners.setdefault(
            MemoryEntry.thread_summary_id(anchor.thread_id), anchor
        )
    for entry in _fetch_by_ids(vector_store, list(summary_owners)):
        admit(entry, EDGE_THREAD, summary_owners[entry.entry_id])

    return ExpandedPool(entries=entries, provenance=provenance)


def scoring_text(entry: MemoryEntry, provenance: Optional[Provenance]) -> str:
    """Text the reranker scores: anchors bare, expanded entries with provenance.

    A bridged fact scored on its bare text would be ranked out again for the same
    reason retrieval missed it, so the prefix tells the cross-encoder why the
    entry is here.
    """
    if provenance is None:
        return entry.lossless_restatement
    return f"{provenance.prefix()} {entry.lossless_restatement}"


def _fetch_by_ids(
    vector_store: VectorStore,
    entry_ids: Sequence[str],
) -> List[MemoryEntry]:
    wanted = [entry_id for entry_id in dict.fromkeys(entry_ids) if entry_id]
    if not wanted:
        return []
    try:
        return vector_store.get_by_ids(wanted)
    except Exception as error:
        print(f"[Expand] id lookup failed: {error}")
        return []


def _find_by_field(
    vector_store: VectorStore,
    field: str,
    values: Sequence[str],
) -> List[MemoryEntry]:
    try:
        return vector_store.find_by_field(field, list(values))
    except Exception as error:
        print(f"[Expand] reverse lookup on {field} failed: {error}")
        return []
