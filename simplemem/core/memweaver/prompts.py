"""Prompts for the MemWeaver write pipeline (design doc section 3).

Call A assigns a session's turns to threads. Call B processes one thread: it
extracts facts, decides their weave relation to the thread's existing facts, and
rewrites the thread's living summary. The finalize sweep asks one batched
judgement about cross-thread supersede pairs the assignment may have split.

All three replace numeric thresholds with LLM semantic judgement, and each has a
deterministic fallback in the writer when parsing fails.
"""

from typing import List, Sequence

from simplemem.core.models.memory_entry import MemoryEntry


EXTRACTION_SYSTEM_PROMPT = (
    "You are a professional information extraction assistant, skilled at "
    "extracting structured, unambiguous information from conversations. "
    "You must output valid JSON format."
)

ASSIGNMENT_SYSTEM_PROMPT = (
    "You organize a conversation memory into topical threads. "
    "You must output valid JSON format."
)

SWEEP_SYSTEM_PROMPT = (
    "You judge whether two remembered facts are versions of the same evolving "
    "fact. You must output valid JSON format."
)


def format_turns(turns: Sequence) -> str:
    """Number a session's turns 1..n for the LLM to reference."""
    return "\n".join(
        f"{index}. [{turn.speaker}] {turn.content}"
        for index, turn in enumerate(turns, 1)
    )


def format_source_turns(turns: Sequence) -> str:
    """Render turns with their stable ``Dialogue.dialogue_id`` values."""
    return "\n".join(
        f"{turn.dialogue_id}. [{turn.speaker}] {turn.content}"
        for turn in turns
    )


def format_fact_candidates(facts: Sequence[MemoryEntry]) -> str:
    """Number a thread's existing facts 1..m as weave candidates."""
    if not facts:
        return "(none - this thread has no facts yet)"

    lines = []
    for index, fact in enumerate(facts, 1):
        state = f"valid from {fact.valid_from}" if fact.valid_from else "no date"
        if fact.valid_until:
            state += f", already superseded on {fact.valid_until}"
        lines.append(f"{index}. {fact.lossless_restatement} ({state})")
    return "\n".join(lines)


def build_thread_assignment_prompt(
    turns: Sequence,
    thread_catalogue: List[str],
    session_date: str,
) -> str:
    """Call A: assign the session's turns to existing or new threads."""
    catalogue = "\n".join(thread_catalogue) if thread_catalogue else "(no threads yet)"

    return f"""You maintain a long-running memory organized as topical threads.
A new conversation session took place on {session_date or "an unknown date"}.
Assign every turn of this session to a thread.

[Existing threads]
{catalogue}

[Session turns]
{format_turns(turns)}

[Rules]
1. Cover EVERY turn number exactly once; do not skip or repeat turn numbers.
2. Attach turns to an existing thread whenever they continue that thread's
   topic - threads live across sessions and should be reused, not duplicated.
3. Create a new thread only for a topic none of the existing threads covers.
4. Group turns that belong to the same topic together; a session usually
   touches only a handful of threads.
5. Use "existing:<thread id>" exactly as listed (e.g. "existing:t3"), or
   "new:<short thread title>" for a new thread.

[Output Format]
```json
{{
  "assignments": [
    {{"turns": [1, 2, 3], "thread": "existing:t3"}},
    {{"turns": [4, 5], "thread": "new:Pottery class"}}
  ]
}}
```

Return ONLY the JSON, no other explanations.
"""


def build_thread_update_prompt(
    thread_title: str,
    thread_summary: str,
    candidates: Sequence[MemoryEntry],
    turns: Sequence,
    session_date: str,
) -> str:
    """Call B: extract facts, weave them, and rewrite the living summary."""
    return f"""You maintain one thread of a long-running conversation memory.
New turns from a session dated {session_date or "an unknown date"} belong to this
thread. Extract the facts they carry, decide how each new fact relates to the
facts this thread already holds, and rewrite the thread summary.

[Thread]
Title: {thread_title or "(new thread)"}
Current summary: {thread_summary or "(no summary yet - this thread is new)"}

[Existing facts of this thread - weave candidates, referenced by number]
{format_fact_candidates(candidates)}

[New turns assigned to this thread]
{format_source_turns(turns)}

[Fact extraction requirements]
1. **Complete Coverage**: Generate enough facts to ensure ALL information in the
   turns is captured.
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it,
   they, this, that) and relative time (yesterday, today, last week, tomorrow).
   The session date is {session_date or "unknown"}; resolve relative time
   against it.
3. **Lossless Information**: Each restatement must be a complete, independent,
   understandable sentence. The thread summary above is context for resolving
   references - never copy it into a fact.
4. **Precise Extraction**: keywords (names, places, entities, topic words),
   timestamp (ISO 8601, only when the dialogue states a time), location,
   persons, entities, topic.
5. **Source Coverage**: Every fact must list the exact turn numbers above that
   support it. Every turn that contains only greeting, acknowledgement, or
   repeated confirmation must appear in exempt_turn_ids. Do not exempt a turn
   that introduces a preference, event, plan, state, or other factual detail.
6. **Set Key**: Assign each fact a ``set_key`` of the form
   ``"<entity>:<predicate>"`` (e.g., ``"Melanie:camping"``,
   ``"Bob:work_project"``) when the fact belongs to a potentially multi-member
   aggregation group — i.e., the entity may have multiple facts about the same
   predicate across sessions. Use an empty string for isolated events with no
   natural aggregation group. Facts that share the same ``set_key`` across
   different sessions form one cross-session aggregation set.

[Weaving - how each new fact relates to an existing candidate]
- "supersede": the new fact replaces a candidate that is no longer true
  (the state changed: moved, quit, finished, changed mind, new preference).
- "refine": the new fact adds detail to a candidate that remains true.
- "bridge": the new fact connects to a candidate that is neither replaced nor
  refined but is needed to understand it.
- "none": no relation to any candidate (default - use it when unsure).
Set "target" to the candidate number, or null when op is "none". Never
supersede a candidate that is already superseded.

[Summary]
Rewrite the thread summary so it reflects the thread including the new facts:
a compact paragraph capturing what is currently true and how it evolved.
Report summary_impact: "major" if the thread's storyline changed, "minor" for
small additions, "none" if nothing changed (then keep the old summary text).
List in outdated_facts the numbers of existing candidates whose wording no
longer matches the rewritten summary.

[Output Format]
```json
{{
  "facts": [
    {{
      "lossless_restatement": "Complete unambiguous restatement",
      "keywords": ["keyword1", "keyword2"],
      "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
      "location": "location name or null",
      "persons": ["name1"],
      "entities": ["entity1"],
      "topic": "topic phrase",
      "source_turn_ids": [12],
      "weave": {{"op": "none|supersede|refine|bridge", "target": null}},
      "set_key": "Melanie:camping"
    }}
  ],
  "summary": "rewritten thread summary",
  "summary_impact": "none|minor|major",
  "outdated_facts": [],
  "exempt_turn_ids": []
}}
```

Return ONLY the JSON, no other explanations.
"""


def build_coverage_repair_prompt(
    source_turns: Sequence,
    nearby_turns: Sequence,
    session_date: str,
    thread_title: str,
    thread_summary: str,
    related_facts: Sequence[MemoryEntry],
) -> str:
    """Build a small repair request for one unresolved coverage debt."""
    facts = format_fact_candidates(related_facts)
    nearby = format_source_turns(nearby_turns) if nearby_turns else "(none)"
    return f"""Repair one unresolved gap in a conversation memory.

[Coverage debt]
Session date: {session_date or "an unknown date"}
Thread title: {thread_title or "(unknown thread)"}
Thread summary: {thread_summary or "(no summary)"}

[Uncovered source turns]
{format_source_turns(source_turns)}

[Nearby context]
{nearby}

[Related facts already stored]
{facts}

[Rules]
1. Extract only facts supported by the uncovered source turns.
2. Each fact must be self-contained, resolve names and relative time, and list
   the supporting source_turn_ids from the uncovered source turns.
3. Use exempt_turn_ids only for greetings, acknowledgements, or repeated
   confirmations that contain no factual information.
4. Do not rewrite the thread summary and do not propose weave relations.

[Output Format]
```json
{{
  "facts": [
    {{
      "lossless_restatement": "Complete unambiguous restatement",
      "keywords": ["keyword1"],
      "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
      "location": "location name or null",
      "persons": ["name1"],
      "entities": ["entity1"],
      "topic": "topic phrase",
      "source_turn_ids": [12]
    }}
  ],
  "exempt_turn_ids": []
}}
```

Return ONLY the JSON, no other explanations.
"""


def build_sweep_prompt(pairs: Sequence[dict]) -> str:
    """Finalize sweep: batched judgement of cross-thread supersede pairs.

    Batching is prompt plumbing only - every candidate pair is judged, so the
    batch size does not change any decision.
    """
    blocks = []
    for pair in pairs:
        blocks.append(
            f"""Pair {pair['index']}:
  Earlier ({pair['earlier_date'] or 'no date'}): {pair['earlier_text']}
  Later   ({pair['later_date'] or 'no date'}): {pair['later_text']}"""
        )
    rendered = "\n".join(blocks)

    return f"""Each pair below holds two facts remembered from the same
conversation, listed earlier-first by date. For each pair decide whether the
later fact is an UPDATE that makes the earlier fact no longer true - the same
underlying fact changed state (moved, quit, replaced, finished, changed
preference), so the earlier version should be retired.

Answer false when the two facts merely share a topic, are both still true, are
about different subjects, or describe separate events that can coexist.

[Pairs]
{rendered}

[Output Format]
```json
{{
  "judgements": [
    {{"index": 1, "supersedes": true, "reason": "brief"}},
    {{"index": 2, "supersedes": false, "reason": "brief"}}
  ]
}}
```

Return ONLY the JSON, no other explanations.
"""
