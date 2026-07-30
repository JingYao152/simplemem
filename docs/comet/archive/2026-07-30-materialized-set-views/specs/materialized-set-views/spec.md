# Materialized Set Views

## Purpose

Materialized Set Views is an opt-in extension of the SimpleMem memory system
that pre-materializes cross-session entity-predicate aggregation sets. It
addresses the completeness gap in aggregation queries (LoCoMo categories C3
open-domain and C1 multi-hop): top-k retrieval returns a partial subset, so
the answer generator cannot enumerate every relevant fact. By maintaining a
complete, incrementally updated set per `(entity, predicate)` key, the system
returns the full enumeration as a single retrievable entry, coexisting with
the existing atomic-fact layer.

## Configuration

The project-local capability flag `ENABLE_SET_VIEWS` SHALL be disabled by
default. When enabled, the MemWeaver write pipeline assigns each newly
extracted fact a `set_key`, maintains materialized set entries in the same
LanceDB-backed flat table, and the retrieval path classifies queries to route
set-type queries to the materialized layer. Existing SimpleMem and MemWeaver
entry points SHALL retain their current behavior when the flag is disabled.

## Data model

A new entry kind `KIND_SET_VIEW = "set_view"` SHALL be added to the
`MemoryEntry` model. A set-view entry uses the existing fields with the
following semantics:

- `entry_id`: fixed convention `set_view::<set_key>` (e.g.
  `set_view::Melanie:camping`), enabling idempotent upserts.
- `lossless_restatement`: deterministic verbatim enumeration of all member
  facts' `lossless_restatement` texts, one per line, assembled by code. If
  the enumeration exceeds a configurable character budget, it is truncated and
  a provenance line listing omitted fact IDs is appended.
- `kind`: `KIND_SET_VIEW`.
- `thread_id`: empty (sets are cross-thread by design).
- `topic`: a human-readable label derived from `set_key` (e.g., "Melanie's
  camping activities"), used as the embedding text for semantic search.
- `persons`: the entity portion of `set_key`, when it is a person name.
- `valid_from`: the earliest `valid_from` among member facts.
- `valid_until`: empty (sets are living entries, updated in place).
- `links`: unused for set-view entries.
- `source_turn_ids`: unused; member fact IDs are tracked in a new
  `set_member_ids` field.

The `MemoryEntry` model SHALL gain a `set_key: str` field (default `""`) and a
`set_member_ids: List[str]` field (default `[]`). For atomic facts, `set_key`
holds the key assigned by Call B (e.g., `"Melanie:camping"`); for set-view
entries, `set_key` holds the same key and `set_member_ids` lists the `entry_id`
values of all member facts in chronological order.

## Write path

### Call B extension

The Call B extraction prompt SHALL add a `set_key` field to each fact in its
output schema. The prompt SHALL instruct the LLM to assign a `set_key` of the
form `"<entity>:<predicate>"` (e.g., `"Melanie:camping"`,
`"Bob:work_project"`) when the fact belongs to a potentially multi-member
aggregation group, or an empty string when the fact is an isolated event with
no natural aggregation group. The LLM MAY assign the same `set_key` to facts
in different sessions and threads.

The deterministic fallback (plain SimpleMem extraction) SHALL leave `set_key`
empty. A missing or unparseable `set_key` in Call B output SHALL default to
empty.

### Incremental set maintenance

After `_apply_session` completes (facts written, weaves applied, summaries
updated), the writer SHALL perform set maintenance for every distinct
`set_key` produced by the current session:

1. Collect all newly written facts with the same `set_key`.
2. Load the existing set-view entry by its fixed ID
   (`set_view::<set_key>`). If none exists, create one.
3. Merge the new fact IDs into `set_member_ids`, maintaining chronological
   order by `valid_from`.
4. Rebuild `lossless_restatement` as the verbatim enumeration of all member
   facts' `lossless_restatement` texts, applying the character budget
   truncation rule.
5. Update `valid_from` to the earliest member's `valid_from`.
6. Upsert the set-view entry into the vector store (delete old row by fixed
   ID, insert new row). Re-embed using the `topic` label text.

### Supersession handling

When a fact is superseded via `execute_supersede`, the superseded fact
remains in the vector store with a closed `valid_until`. The set-view entry
SHALL retain the superseded fact in `set_member_ids` and in the enumeration
text, because aggregation queries ask for the complete history of activities.
The superseded fact's `lossless_restatement` text is unchanged by supersession
(only `valid_until` and `superseded_by` change), so the enumeration text needs
no update.

When a fact is superseded, the writer SHALL NOT remove it from any set-view
entry. The superseded fact's `entry_id` stays in `set_member_ids`.

## Retrieval path

### Query classification

The `_analyze_information_requirements` LLM call in `HybridRetriever` SHALL
add a `query_type` field to its output with values `"set"` or `"point"`. The
prompt SHALL instruct the LLM to classify as `"set"` when the query asks for
a complete enumeration of an entity's activities, preferences, or attributes
(e.g., "What camping has Melanie done?", "List all of Bob's hobbies"), and as
`"point"` when the query asks for a specific fact, time, or location (e.g.,
"Where did Melanie camp last summer?").

A missing or unparseable `query_type` SHALL default to `"point"`, preserving
the existing retrieval path.

### Set-type query retrieval

When `query_type` is `"set"`, the retriever SHALL:

1. Semantic-search the query against set-view entries in the vector store,
   returning the top-k matching sets (using the set's `topic` label as
   embedding text).
2. Collect the `set_member_ids` of all returned sets into a `covered_ids`
   set.
3. Run the existing retrieval path (semantic + keyword + structured,
   expansion, rerank) to obtain atomic fact candidates.
4. Filter out atomic facts whose `entry_id` is in `covered_ids`.
5. Return the set-view entries first, followed by the uncovered atomic facts,
   within the same context budget.

If no set-view entries match (semantic search returns zero results), the
retriever SHALL fall back entirely to the existing retrieval path.

### Point query retrieval

When `query_type` is `"point"`, the retriever SHALL use the existing
retrieval path unchanged. Set-view entries SHALL NOT appear in point query
results (they are filtered by kind).

## Observability

The writer SHALL expose counts for: facts with a non-empty `set_key`, sets
created, sets updated, set member additions, and set text truncations. The
retriever SHALL expose counts for: queries classified as set-type, set-type
queries that matched at least one set, set-type queries that fell back to
atomic-only, and atomic facts deduplicated out by set coverage.

Evaluation outputs SHALL separately report F1, BLEU, evidence coverage, and
context size, broken down by LoCoMo category, with and without the
`ENABLE_SET_VIEWS` flag.

## Implementation plan

1. Add `KIND_SET_VIEW`, `set_key`, and `set_member_ids` to `MemoryEntry`.
2. Extend the Call B prompt and parser to output and parse `set_key` per
   fact; update the deterministic fallback to leave `set_key` empty.
3. Implement incremental set maintenance in `MemWeaver._apply_session` (or a
   dedicated method called after `_apply_session`): create/update/rebuild
   set-view entries, upsert into the vector store.
4. Extend `_analyze_information_requirements` prompt and parser to output
   `query_type`.
5. Implement set-type retrieval routing in `HybridRetriever.retrieve`:
   semantic search for sets, deduplication of covered atomic facts, fallback
   to existing path.
6. Add `ENABLE_SET_VIEWS` flag to settings, wire it through `MemWeaver` and
   `HybridRetriever` constructors.
7. Add deterministic unit tests, disabled-flag regressions, and a bounded
   benchmark command with set-view provenance fields.
