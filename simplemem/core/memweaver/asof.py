"""As-of retrieval primitives (design doc section 5, contribution C3).

The time anchor is derived from the fabric, not configured: LoCoMo QA carries no
``question_date``, so the anchor is the largest session date in the memory
(structural rule). Non-"when" questions only see entries that are still valid at
that anchor; "when"-type questions skip the filter because asking about history
needs the closed facts.

This module exists only because the write side produced ``valid_until``: with no
validity interval there is no as-of retrieval.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

from simplemem.core.database.vector_store_backend import AnyOf, FieldPredicate
from simplemem.core.models.memory_entry import (
    KIND_ENTITY_PROFILE,
    KIND_THREAD_SUMMARY,
    MemoryEntry,
)


#: Kinds the write pipeline stamps with the session date they were written in.
_SESSION_DATED_KINDS = frozenset({KIND_THREAD_SUMMARY, KIND_ENTITY_PROFILE})


# One deterministic structural rule, the only question-type judgement on the
# read side (design doc section 10.2). "how long"/"how many weeks" questions are
# temporal-history questions too: they ask about elapsed time between facts.
_TEMPORAL_HISTORY_PATTERN = re.compile(
    r"\bwhen\b|\bhow long\b|\bhow many (?:days|weeks|months|years)\b",
    re.IGNORECASE,
)


def is_temporal_history_question(query: str) -> bool:
    """True when the question asks about history and must see closed facts."""
    if not query:
        return False
    return bool(_TEMPORAL_HISTORY_PATTERN.search(query))


def compute_anchor(entries: Iterable[MemoryEntry]) -> str:
    """Largest session date present in the memory, or ``""`` when unknown.

    Session dates are recovered from the entries the write pipeline stamps with
    them: living thread summaries and entity profiles (rewritten at the session
    date) and closed facts (``valid_until`` = the superseding session's date).
    Facts' own ``valid_from`` can be a future date mentioned in dialogue, so it
    is only a last-resort fallback.
    """
    session_dates: List[str] = []
    fallback: List[str] = []

    for entry in entries:
        if entry.kind in _SESSION_DATED_KINDS and entry.valid_from:
            session_dates.append(entry.valid_from)
        if entry.valid_until:
            session_dates.append(entry.valid_until)
        if entry.valid_from:
            fallback.append(entry.valid_from)

    if session_dates:
        return max(session_dates)
    return max(fallback) if fallback else ""


def asof_filters(anchor: str) -> Optional[Dict[str, Any]]:
    """Backend prefilter for "still valid at ``anchor``"."""
    if not anchor:
        return None
    return {
        "valid_until": AnyOf(
            [FieldPredicate("=", ""), FieldPredicate(">=", anchor)]
        )
    }


def is_visible_at(entry: MemoryEntry, anchor: str) -> bool:
    """Same predicate as :func:`asof_filters`, for post-filtering result lists."""
    if not anchor:
        return True
    return not entry.valid_until or entry.valid_until >= anchor


def apply_asof(entries: List[MemoryEntry], anchor: str) -> List[MemoryEntry]:
    """Drop entries closed before the anchor (used on the non-prefilterable paths)."""
    if not anchor:
        return entries
    return [entry for entry in entries if is_visible_at(entry, anchor)]
