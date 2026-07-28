"""
Core Data Structure - MemoryEntry (Memory Unit)

Section 3.1: Semantic Structured Compression
Each MemoryEntry represents a compact, context-independent memory unit
with multi-view indexing (Semantic, Lexical, Symbolic layers)

MemWeaver (docs/memweaver-design.md section 2) extends the same unit with
fabric fields (kind / thread_id / validity interval / weave links) so a single
flat table can carry facts, living thread summaries and entity profiles.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


# MemWeaver entry kinds (design doc section 2)
KIND_FACT = "fact"
KIND_THREAD_SUMMARY = "thread_summary"
KIND_ENTITY_PROFILE = "entity_profile"

# Typed weave operations (design doc section 3)
WEAVE_NONE = "none"
WEAVE_SUPERSEDE = "supersede"
WEAVE_REFINE = "refine"
WEAVE_BRIDGE = "bridge"


class MemoryEntry(BaseModel):
    """
    Memory Unit - Self-contained entry indexed via multi-view indexing (Section 3.1)

    Indexed via: I(m_k) = {s_k (Semantic), l_k (Lexical), r_k (Symbolic)}
    """
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # [Semantic Layer] - Dense embedding base (v_k = E_dense(S_k))
    lossless_restatement: str = Field(
        ...,
        description="Self-contained fact with Φ_coref (no pronouns) and Φ_time (absolute timestamps)"
    )

    # [Lexical Layer] - Sparse keyword vectors (h_k = Sparse(S_k))
    keywords: List[str] = Field(
        default_factory=list,
        description="Core keywords for BM25-style exact matching"
    )

    # [Symbolic Layer] - Metadata constraints (R_k = {(key, val)})
    timestamp: Optional[str] = Field(
        None,
        description="Standardized time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)"
    )
    location: Optional[str] = Field(
        None,
        description="Natural language location description"
    )
    persons: List[str] = Field(
        default_factory=list,
        description="List of extracted persons"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="List of extracted entities (companies, products, etc.)"
    )
    topic: Optional[str] = Field(
        None,
        description="Topic phrase summarized by LLM"
    )

    # [Fabric Layer] - MemWeaver organization state (design doc section 2).
    # Empty-string defaults keep pure-SimpleMem entries valid and let the flat
    # storage schema stay non-nullable.
    kind: str = Field(
        default=KIND_FACT,
        description='"fact" | "thread_summary" | "entity_profile"'
    )
    thread_id: str = Field(
        default="",
        description="Id of the fabric thread this entry belongs to"
    )
    valid_from: str = Field(
        default="",
        description="Day-resolution date the entry becomes effective "
                    "(fact's own timestamp when present, else the session date)"
    )
    valid_until: str = Field(
        default="",
        description="Empty means still valid; closed to the superseding entry's "
                    "session date when superseded"
    )
    superseded_by: str = Field(
        default="",
        description="entry_id of the successor, forming the fact evolution chain"
    )
    links: List[str] = Field(
        default_factory=list,
        description='Typed weave edges, e.g. ["refine:<id>", "bridge:<id>"] '
                    "(written on both endpoints)"
    )
    context_digest: str = Field(
        default="",
        description="Fingerprint of the thread summary used at embedding time "
                    "(P1 re-contextualization)"
    )
    source_turn_ids: List[int] = Field(
        default_factory=list,
        description="Dialogue ids in the source session that support this fact"
    )

    @staticmethod
    def thread_summary_id(thread_id: str) -> str:
        """Fixed id convention for a living thread summary entry."""
        return f"thread::{thread_id}"

    @staticmethod
    def entity_profile_id(name: str) -> str:
        """Fixed id convention for an entity profile entry."""
        return f"profile::{name}"

    @property
    def is_open(self) -> bool:
        """True while the entry has not been closed by a supersede."""
        return not self.valid_until

    def weave_targets(self, edge_type: str) -> List[str]:
        """Return the ids linked to this entry by the given weave edge type."""
        prefix = f"{edge_type}:"
        return [link[len(prefix):] for link in self.links if link.startswith(prefix)]

    class Config:
        json_schema_extra = {
            "example": {
                "entry_id": "550e8400-e29b-41d4-a716-446655440000",
                "lossless_restatement": "Alice discussed the marketing strategy for new product XYZ with Bob at Starbucks in Shanghai on November 15, 2025 at 14:30.",
                "keywords": ["Alice", "Bob", "product XYZ", "marketing strategy", "discussion"],
                "timestamp": "2025-11-15T14:30:00",
                "location": "Starbucks, Shanghai",
                "persons": ["Alice", "Bob"],
                "entities": ["product XYZ"],
                "topic": "Product marketing strategy discussion",
                "kind": "fact",
                "thread_id": "t3",
                "valid_from": "2025-11-15",
                "valid_until": "",
                "superseded_by": "",
                "links": ["refine:8f14e45f-ceea-467a-9f0b-4a1b1c6bb1a1"],
                "context_digest": ""
            }
        }


class Dialogue(BaseModel):
    """
    Original dialogue entry
    """
    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None  # ISO 8601 format

    def __str__(self) -> str:
        time_str = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{time_str}{self.speaker}: {self.content}"
