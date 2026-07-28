"""Typed state-anchor representation for MemWeaver's dual-view retrieval."""

import hashlib

from simplemem.core.models.memory_entry import MemoryEntry


STATE_ANCHOR_VERSION = "state-anchor-v1"
STATE_ANCHOR_TABLE_SUFFIX = "__state_anchors"


def state_anchor_table_name(primary_table_name: str) -> str:
    """Return the deterministic companion-table name for a memory table."""
    return f"{primary_table_name}{STATE_ANCHOR_TABLE_SUFFIX}"


def state_anchor_text(entry: MemoryEntry) -> str:
    """Render the independently embedded state view of one fact.

    The representation uses only persisted fact fields. It therefore remains
    reproducible when a thread summary later changes and retains the factual
    sentence as the final state clause for relation-level retrieval.
    """
    fields = ["State anchor"]
    if entry.thread_id:
        fields.append(f"thread: {entry.thread_id}")
    if entry.topic:
        fields.append(f"topic: {entry.topic}")
    if entry.persons:
        fields.append(f"persons: {', '.join(entry.persons)}")
    if entry.entities:
        fields.append(f"entities: {', '.join(entry.entities)}")
    if entry.location:
        fields.append(f"location: {entry.location}")
    if entry.valid_from:
        fields.append(f"valid from: {entry.valid_from}")
    if entry.valid_until:
        fields.append(f"valid until: {entry.valid_until}")
    fields.append(f"fact: {entry.lossless_restatement}")
    return " | ".join(fields)


def state_anchor_digest(entry: MemoryEntry) -> str:
    """Return a versioned fingerprint for the materialized anchor text."""
    digest = hashlib.sha1(state_anchor_text(entry).encode("utf-8")).hexdigest()[:16]
    return f"{STATE_ANCHOR_VERSION}:{digest}"
