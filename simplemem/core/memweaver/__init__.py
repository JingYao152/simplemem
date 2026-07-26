"""MemWeaver - write-time self-organizing memory fabric.

Implements P0 of docs/memweaver-design.md section 8: the fabric data model, the
Call A / Call B write pipeline (thread assignment, extraction, typed weaving,
living summaries), supersede execution, the finalize cross-thread sweep, and the
as-of retrieval primitives the write side makes possible.
"""

from simplemem.core.memweaver.asof import (
    apply_asof,
    asof_filters,
    compute_anchor,
    is_temporal_history_question,
    is_visible_at,
)
from simplemem.core.memweaver.dates import parse_session_datetime, to_day
from simplemem.core.memweaver.fabric import (
    FabricSnapshot,
    ThreadState,
    build_thread_summary_entry,
    execute_link,
    execute_supersede,
    load_fabric,
)
from simplemem.core.memweaver.writer import MemWeaver, ThreadAssignment, ThreadUpdate

__all__ = [
    "MemWeaver",
    "ThreadAssignment",
    "ThreadUpdate",
    "FabricSnapshot",
    "ThreadState",
    "load_fabric",
    "build_thread_summary_entry",
    "execute_supersede",
    "execute_link",
    "compute_anchor",
    "asof_filters",
    "apply_asof",
    "is_visible_at",
    "is_temporal_history_question",
    "parse_session_datetime",
    "to_day",
]
