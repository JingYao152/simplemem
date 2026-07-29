"""Constraint-based evidence selection for multi-requirement questions."""

from typing import Callable, Dict, List, Mapping, Optional, Sequence

from simplemem.core.memweaver.expansion import (
    EDGE_BRIDGE,
    EDGE_REFINE,
    Provenance,
)
from simplemem.core.models.memory_entry import KIND_FACT, MemoryEntry
from simplemem.core.reranker import CrossEncoderReranker


def _requirement_text(requirement: Mapping) -> str:
    return " ".join(
        part
        for part in (
            str(requirement.get("info_type") or ""),
            str(requirement.get("description") or ""),
        )
        if part
    )


def _eligible_requirements(required_info: Sequence[Mapping]) -> List[Mapping]:
    return [
        requirement
        for requirement in required_info
        if str(requirement.get("priority") or "").lower() in {"high", "medium"}
        and _requirement_text(requirement)
    ]


def plan_requires_evidence_weave(information_plan: Mapping) -> bool:
    """Whether a plan contains enough independent needs for proof selection."""
    try:
        minimal_queries_needed = int(information_plan.get("minimal_queries_needed", 1))
    except (TypeError, ValueError):
        minimal_queries_needed = 1

    required_info = information_plan.get("required_info") or []
    return (
        minimal_queries_needed >= 2
        and len(_eligible_requirements(required_info)) >= 2
    )


def select_constraint_evidence_weave(
    required_info: Sequence[Mapping],
    ranked_entries: Sequence[MemoryEntry],
    reranker: CrossEncoderReranker,
    to_text: Callable[[MemoryEntry], str],
    provenance: Dict[str, Provenance],
    top_k: int,
) -> List[MemoryEntry]:
    """Keep distinct requirement evidence and typed anchors within ``top_k``.

    The incoming entries have already been globally reranked. This selector only
    changes which entries consume the fixed answer budget: each eligible
    requirement receives a distinct fact when possible, bridge/refine anchors
    remain beside their selected fact, and residual capacity prefers facts from
    source turns not already represented in the packet.
    """
    if top_k <= 0:
        return []

    ordered = list(ranked_entries)
    candidates = [entry for entry in ordered if entry.kind == KIND_FACT]
    requirements = _eligible_requirements(required_info)
    if len(candidates) == 0 or len(requirements) < 2:
        return ordered[:top_k]

    documents = [to_text(entry) for entry in candidates]
    requirement_scores: List[List[float]] = []
    for requirement in requirements:
        scores = reranker.score(_requirement_text(requirement), documents)
        if scores is None or len(scores) != len(candidates):
            return ordered[:top_k]
        requirement_scores.append(scores)

    entries_by_id = {entry.entry_id: entry for entry in ordered}
    selected: List[MemoryEntry] = []
    selected_ids = set()
    primary_ids = set()

    def add(entry: Optional[MemoryEntry]) -> None:
        if entry is None or entry.entry_id in selected_ids or len(selected) >= top_k:
            return
        selected.append(entry)
        selected_ids.add(entry.entry_id)

    for scores in requirement_scores:
        available = [
            index
            for index, entry in enumerate(candidates)
            if entry.entry_id not in primary_ids
        ]
        if not available:
            break

        best_index = max(available, key=lambda index: (scores[index], -index))
        primary = candidates[best_index]
        primary_ids.add(primary.entry_id)
        add(primary)

        record = provenance.get(primary.entry_id)
        if record is not None and record.edge in (EDGE_BRIDGE, EDGE_REFINE):
            add(entries_by_id.get(record.anchor_id))

    represented_turn_ids = {
        turn_id
        for entry in selected
        if entry.kind == KIND_FACT
        for turn_id in entry.source_turn_ids
    }
    for entry in ordered:
        if entry.entry_id in selected_ids or entry.kind != KIND_FACT:
            continue
        if not entry.source_turn_ids:
            continue
        if represented_turn_ids.intersection(entry.source_turn_ids):
            continue
        add(entry)
        represented_turn_ids.update(entry.source_turn_ids)

    for entry in ordered:
        add(entry)

    return selected
