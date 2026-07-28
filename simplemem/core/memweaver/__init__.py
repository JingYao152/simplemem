"""MemWeaver - write-time self-organizing memory fabric.

Implements P0 and P1 of docs/memweaver-design.md section 8:

* **P0** - the fabric data model, the Call A / Call B write pipeline (thread
  assignment, extraction, typed weaving, living summaries), supersede execution,
  the finalize cross-thread sweep, and the as-of retrieval primitives the write
  side makes possible.
* **P1** - context-inheriting embeddings (the vector layer carries the living
  thread summary as a prefix), semantically triggered re-embedding of the facts
  Call B names as outdated, and person-level entity profiles in the pool.
* **P2** - one-hop evidence expansion along the fabric's own edges (supersede
  chains both ways, weave edges, thread membership) with provenance, feeding a
  flat cross-encoder rerank (``simplemem.core.reranker``, an inherited generic
  component) and the supersede-chain annotation in the answer context.
"""

from simplemem.core.memweaver.asof import (
    apply_asof,
    asof_filters,
    compute_anchor,
    is_temporal_history_question,
    is_visible_at,
)
from simplemem.core.memweaver.context import (
    context_digest,
    context_prefix,
    contextual_embed_text,
)
from simplemem.core.memweaver.dates import parse_session_datetime, to_day
from simplemem.core.memweaver.dual_view import (
    STATE_ANCHOR_TABLE_SUFFIX,
    STATE_ANCHOR_VERSION,
    state_anchor_digest,
    state_anchor_table_name,
    state_anchor_text,
)
from simplemem.core.memweaver.expansion import (
    ExpandedPool,
    Provenance,
    expand_one_hop,
    scoring_text,
)
from simplemem.core.memweaver.fabric import (
    FabricSnapshot,
    ThreadState,
    build_entity_profile_entry,
    build_thread_summary_entry,
    execute_link,
    execute_supersede,
    load_fabric,
    speaker_threads,
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
    "build_entity_profile_entry",
    "speaker_threads",
    "execute_supersede",
    "execute_link",
    "compute_anchor",
    "asof_filters",
    "apply_asof",
    "is_visible_at",
    "is_temporal_history_question",
    "context_prefix",
    "contextual_embed_text",
    "context_digest",
    "expand_one_hop",
    "scoring_text",
    "ExpandedPool",
    "Provenance",
    "parse_session_datetime",
    "to_day",
    "STATE_ANCHOR_TABLE_SUFFIX",
    "STATE_ANCHOR_VERSION",
    "state_anchor_digest",
    "state_anchor_table_name",
    "state_anchor_text",
]
