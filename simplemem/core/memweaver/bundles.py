"""Requirement-preserving evidence selection for the fixed answer budget."""

from typing import Callable, Dict, List, Mapping, Optional, Sequence

from simplemem.core.memweaver.expansion import (
    EDGE_BRIDGE,
    EDGE_REFINE,
    Provenance,
)
from simplemem.core.models.memory_entry import KIND_FACT, MemoryEntry
from simplemem.core.reranker import CrossEncoderReranker


def select_requirement_bundles(
    required_info: Sequence[Mapping],
    ranked_entries: Sequence[MemoryEntry],
    reranker: CrossEncoderReranker,
    to_text: Callable[[MemoryEntry], str],
    provenance: Dict[str, Provenance],
    top_k: int,
) -> List[MemoryEntry]:
    """Reserve evidence for high-priority requirements within ``top_k``.

    The incoming entries are already in global query-rerank order. For every
    high-priority requirement, a local cross-encoder score chooses one fact.
    A bridge or refine fact keeps its expansion anchor beside it. Remaining
    capacity follows the incoming order, preserving the fixed answer budget.
    """
    if top_k <= 0:
        return []

    ordered = list(ranked_entries)
    candidates = [entry for entry in ordered if entry.kind == KIND_FACT]
    if not candidates:
        return ordered[:top_k]

    entries_by_id = {entry.entry_id: entry for entry in ordered}
    selected: List[MemoryEntry] = []
    selected_ids = set()

    def add(entry: Optional[MemoryEntry]) -> None:
        if entry is None or entry.entry_id in selected_ids or len(selected) >= top_k:
            return
        selected.append(entry)
        selected_ids.add(entry.entry_id)

    for requirement in required_info:
        if requirement.get("priority") != "high":
            continue
        text = " ".join(
            part for part in (
                str(requirement.get("info_type") or ""),
                str(requirement.get("description") or ""),
            ) if part
        )
        if not text:
            continue

        scores = reranker.score(text, [to_text(entry) for entry in candidates])
        if scores is None or len(scores) != len(candidates):
            return ordered[:top_k]
        best_index = max(range(len(candidates)), key=lambda index: scores[index])
        primary = candidates[best_index]
        add(primary)

        record = provenance.get(primary.entry_id)
        if record is not None and record.edge in (EDGE_BRIDGE, EDGE_REFINE):
            add(entries_by_id.get(record.anchor_id))

    for entry in ordered:
        add(entry)
    return selected
