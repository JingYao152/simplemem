# Outcome

Add a materialized-set-view layer to the SimpleMem memory system so that
aggregation queries (e.g., "What are all of Melanie's camping-related
activities?") receive a complete, pre-enumerated answer instead of relying on
lossy top-k recall of atomic facts that the answer generator must piece
together.

The materialized sets coexist with atomic facts in the same storage. At
retrieval time the system selects which layer to return based on query type:
set-type queries receive the complete set; point queries continue to receive
individual facts.

# Scope

Modify the SimpleMem write path to incrementally maintain entity-predicate set
entries that aggregate cross-session facts. Modify the retrieval path to detect
set-type queries and route them to the materialized set layer.

The change touches the `simplemem` source tree directly (unlike
`temporal-memory-fabric`, which is a standalone project). The materialized
sets live in the same LanceDB-backed flat table as existing facts, thread
summaries, and entity profiles, using a new entry kind.

# Non-goals

- This change does not create a standalone project or external adapters.
- This change does not replace the existing MemWeaver thread structure, weave
  relations, or entity profiles.
- This change does not modify the answer generator; the generator receives the
  same normalized context format, only with potentially different source
  entries.
- This change does not claim a benchmark improvement before a completed
  evaluation.
- This change does not modify benchmark datasets.

# Acceptance examples

1. A set-type query ("What camping has Melanie done?") receives a single
   materialized set entry containing all cross-session camping facts for
   Melanie, not a partial top-k subset.
2. A point query ("Where did Melanie camp last summer?") continues to receive
   individual atomic facts via the existing retrieval path.
3. When a new fact is written that belongs to an existing set, the set is
   updated incrementally without rebuilding from scratch.
4. When a fact in a set is superseded, the set reflects the update without
   losing historical members.
5. With the feature flag disabled, existing MemWeaver and SimpleMem tests
   retain their behavior.
6. A set-type query returns matching sets plus atomic facts not covered by any
   returned set, with no duplicate entries.

# Constraints and invariants

- All new behavior must be opt-in through a project-local configuration flag
  whose default is disabled.
- Each materialized set must retain traceable identifiers of its member facts.
- Set membership must be maintainable incrementally; full rebuilds are allowed
  only at initialization or after data loss.
- The answer context budget must be respected: if a set is too large, it is
  truncated or summarized with a provenance note.
- The materialized set layer must not use benchmark labels or reference
  answers.

# Decisions

The change modifies the `simplemem` source tree directly, adding a new entry
kind and write/retrieval logic alongside the existing MemWeaver code.

**Set-key assignment (Q1, resolved 2026-07-31):** Call B assigns each
newly-extracted fact a `set_key` string (e.g., `"Melanie:camping"`) during the
existing extraction call. This reuses Call B's semantic understanding at
near-zero marginal LLM cost and is consistent with MemWeaver's principle that
all organization decisions are LLM-driven. The LLM may assign the same
`set_key` to facts across different sessions and threads, forming the
cross-session aggregation group.

**Retrieval routing (Q2, resolved 2026-07-31):** The existing
`_analyze_information_requirements` LLM call in `HybridRetriever` adds a
`query_type: "set" | "point"` classification. Set-type queries trigger a
search for matching materialized set entries; point queries use the existing
retrieval path unchanged. This reuses the existing LLM call at near-zero
marginal cost and handles paraphrased set queries that rule-based detection
would miss.

**Set text content (Q3, resolved 2026-07-31):** The materialized set entry's
`lossless_restatement` is a deterministic verbatim enumeration of all member
facts' `lossless_restatement` texts, assembled by code (not LLM). If the set
exceeds the context budget, it is truncated with a provenance note listing
omitted fact IDs. The set entry is embedded using a short label derived from
`set_key` (e.g., "Melanie's camping activities"), not the full enumeration, so
semantic search matches on intent rather than member text.

**Set + uncovered atomic facts (Q4, resolved 2026-07-31):** For set-type
queries, the system returns matching sets plus atomic facts from the existing
retrieval path that are not members of any returned set. Member facts already
in a returned set are deduplicated out of the atomic results to avoid
redundancy. This ensures facts without a `set_key` are not lost while
respecting the context budget.

**Confirmation (2026-07-31):** The user confirmed the goal, scope, four key
decisions, acceptance criteria, and non-goals as reflecting a shared
understanding.

# Open questions

No unresolved scope decisions remain.

# Verification expectations

Add offline unit tests for set creation, incremental update, supersession
handling, set-type query detection, set-plus-uncovered-fact deduplication, and
disabled-flag compatibility. Run focused regression tests for MemWeaver and
the new materialized-set layer. Perform a bounded LoCoMo evaluation after the
implementation is complete, reporting F1, BLEU, evidence coverage, and context
size.
