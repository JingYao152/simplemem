"""
Answer Generator - Final synthesis from retrieved contexts

Section 3.3: Intent-Aware Retrieval Planning
Generates answers from the merged context C_q after multi-view retrieval
"""
from typing import Dict, List, Optional
from simplemem.core.answer_canonicalizer import canonicalize_answer
from simplemem.core.models.memory_entry import (
    KIND_FACT,
    KIND_THREAD_SUMMARY,
    MemoryEntry,
)
from simplemem.core.utils.llm_client import LLMClient
from simplemem.core.settings import settings as config


class AnswerGenerator:
    """
    Answer Generator - Synthesis from retrieved memory units (Section 3.3)

    Generates answers from C_q = R_sem ∪ R_lex ∪ R_sym

    The prompt itself stays SimpleMem-native (no question-type formatting). With
    ``annotate_chains`` on, members of a supersede chain are laid out next to each
    other and marked "[SUPERSEDED on <date> by Context N]" - temporal questions
    routinely ask what a fact used to be, and that reading requires knowing which
    context replaced which (design doc section 5, contribution C3).
    """
    def __init__(
        self,
        llm_client: LLMClient,
        annotate_chains: Optional[bool] = None,
        enable_answer_canonicalization: Optional[bool] = None,
        enable_thread_evidence_crosscheck: Optional[bool] = None,
    ):
        self.llm_client = llm_client
        self.annotate_chains = (
            annotate_chains
            if annotate_chains is not None
            else getattr(config, 'ENABLE_EXPAND_RERANK', False)
        )
        self.enable_answer_canonicalization = (
            enable_answer_canonicalization
            if enable_answer_canonicalization is not None
            else getattr(config, 'ENABLE_ANSWER_CANONICALIZATION', False)
        )
        self.enable_thread_evidence_crosscheck = (
            enable_thread_evidence_crosscheck
            if enable_thread_evidence_crosscheck is not None
            else getattr(config, 'ENABLE_THREAD_EVIDENCE_CROSSCHECK', False)
        )

    def generate_answer(self, query: str, contexts: List[MemoryEntry]) -> str:
        """
        Generate answer

        Args:
        - query: User question
        - contexts: List of retrieved relevant MemoryEntry

        Returns:
        - Generated answer (concise phrase)
        """
        if not contexts:
            return "No relevant information found"

        # Build context string
        context_str = self._format_contexts(contexts)

        # Build prompt
        prompt = self._build_answer_prompt(query, context_str)

        # Call LLM to generate answer
        messages = [
            {
                "role": "system",
                "content": "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                # Parse JSON response
                result = self.llm_client.extract_json(response)
                # Return the answer from JSON
                return self._finalize_answer(
                    result.get("answer", response.strip()), query
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to parse JSON response after {max_retries} attempts: {e}")
                    # Fallback to raw response
                    if 'response' in locals():
                        return self._finalize_answer(response.strip(), query)
                    else:
                        return "Failed to generate answer"

    def _finalize_answer(self, answer: str, query: str) -> str:
        """Apply the optional deterministic answer canonicalization step."""
        if not self.enable_answer_canonicalization:
            return answer
        return canonicalize_answer(answer, query)

    @staticmethod
    def _order_supersede_chains(contexts: List[MemoryEntry]) -> List[MemoryEntry]:
        """Lay each supersede chain out as a contiguous block, oldest first.

        Chains converge: the cross-thread sweep can close several facts with the
        same successor, so a successor may have more than one predecessor present.
        All present predecessors are emitted (recursively) before it, which keeps
        the whole chain contiguous and lets the annotation point forwards.

        Only entries already in the context list are moved; nothing is added or
        dropped. Without fabric validity fields this is the identity.
        """
        by_id = {entry.entry_id: entry for entry in contexts}
        predecessors: Dict[str, List[MemoryEntry]] = {}
        for entry in contexts:
            if entry.superseded_by and entry.superseded_by in by_id:
                predecessors.setdefault(entry.superseded_by, []).append(entry)
        has_successor_present = {
            entry.entry_id
            for group in predecessors.values()
            for entry in group
        }

        ordered: List[MemoryEntry] = []
        placed = set()

        def emit(entry: MemoryEntry) -> None:
            if entry.entry_id in placed:
                return
            placed.add(entry.entry_id)  # set first: guards against cyclic links
            for predecessor in predecessors.get(entry.entry_id, []):
                emit(predecessor)
            ordered.append(entry)

        for entry in contexts:
            # Enter each chain at its newest present member; predecessors follow
            # from it, so every chain is emitted exactly once.
            if entry.entry_id not in has_successor_present:
                emit(entry)

        # Cyclic links (never produced by the write side) keep their order.
        ordered.extend(entry for entry in contexts if entry.entry_id not in placed)
        return ordered

    @staticmethod
    def _chain_annotations(contexts: List[MemoryEntry]) -> Dict[str, str]:
        """Map entry id -> "[SUPERSEDED on <date> by Context N]"."""
        positions = {entry.entry_id: index for index, entry in enumerate(contexts, 1)}
        annotations: Dict[str, str] = {}
        for entry in contexts:
            if not entry.valid_until:
                continue
            successor = positions.get(entry.superseded_by)
            if successor is not None:
                annotations[entry.entry_id] = (
                    f"[SUPERSEDED on {entry.valid_until} by Context {successor}]"
                )
            else:
                annotations[entry.entry_id] = (
                    f"[NO LONGER TRUE as of {entry.valid_until}]"
                )
        return annotations

    def _prepare_contexts(
        self, contexts: List[MemoryEntry]
    ) -> tuple[List[MemoryEntry], Dict[str, str]]:
        """Preserve chain-aware ordering before rendering the answer context."""
        annotations: Dict[str, str] = {}
        if self.annotate_chains and contexts:
            contexts = self._order_supersede_chains(contexts)
            annotations = self._chain_annotations(contexts)
        return contexts, annotations

    @staticmethod
    def _entry_details(
        entry: MemoryEntry, content: str, content_label: str = "Content"
    ) -> List[str]:
        """Render shared entry metadata without changing the evidence set."""
        parts = [f"{content_label}: {content}"]

        if entry.timestamp:
            parts.append(f"Time: {entry.timestamp}")

        if entry.location:
            parts.append(f"Location: {entry.location}")

        if entry.persons:
            parts.append(f"Persons: {', '.join(entry.persons)}")

        if entry.entities:
            parts.append(f"Related Entities: {', '.join(entry.entities)}")

        if entry.topic:
            parts.append(f"Topic: {entry.topic}")

        return parts

    def _render_entry(
        self,
        entry: MemoryEntry,
        heading: str,
        annotations: Dict[str, str],
        content_label: str = "Content",
    ) -> str:
        """Render one retrieved entry while retaining chain annotations."""
        content = entry.lossless_restatement
        annotation = annotations.get(entry.entry_id)
        if annotation:
            content = f"{content} {annotation}"
        return "\n".join(
            [heading, *self._entry_details(entry, content, content_label)]
        )

    def _format_thread_evidence_contexts(
        self, contexts: List[MemoryEntry], annotations: Dict[str, str]
    ) -> str:
        """Group retrieved thread summaries beside their retrieved fact evidence.

        The renderer only states the persisted shared ``thread_id`` relation. It
        keeps every retrieved entry and makes no claim that an unretrieved fact
        supports a summary.
        """
        summaries = [entry for entry in contexts if entry.kind == KIND_THREAD_SUMMARY]
        summary_thread_ids = {
            entry.thread_id for entry in summaries if entry.thread_id
        }
        facts = [entry for entry in contexts if entry.kind == KIND_FACT]

        current_supporting = [
            entry
            for entry in facts
            if entry.thread_id in summary_thread_ids and entry.is_open
        ]
        historical_supporting = [
            entry
            for entry in facts
            if entry.thread_id in summary_thread_ids and not entry.is_open
        ]
        other_facts = [
            entry for entry in facts if entry.thread_id not in summary_thread_ids
        ]
        other_entries = [
            entry
            for entry in contexts
            if entry.kind not in {KIND_THREAD_SUMMARY, KIND_FACT}
        ]

        sections: List[str] = []
        if summaries:
            summary_blocks = []
            for index, entry in enumerate(summaries, 1):
                thread_label = entry.thread_id or f"summary-{index}"
                summary_blocks.append(
                    self._render_entry(
                        entry,
                        f"[Thread {thread_label}]",
                        annotations,
                        content_label="Summary",
                    )
                )
            sections.append("[Thread Summaries]\n" + "\n\n".join(summary_blocks))

        def render_facts(
            facts_to_render: List[MemoryEntry],
            section_name: str,
            historical: bool,
            supports_summary: bool,
        ) -> None:
            if not facts_to_render:
                return
            fact_blocks = []
            for index, entry in enumerate(facts_to_render, 1):
                state = "current"
                if historical:
                    state = f"historical, superseded on {entry.valid_until}"
                relation = ""
                if supports_summary:
                    relation = f" | supports Thread {entry.thread_id}"
                fact_blocks.append(
                    self._render_entry(
                        entry,
                        f"[Fact {index} | thread={entry.thread_id or 'unassigned'} | "
                        f"{state}{relation}]",
                        annotations,
                    )
                )
            sections.append(f"[{section_name}]\n" + "\n\n".join(fact_blocks))

        render_facts(
            current_supporting,
            "Supporting Facts",
            historical=False,
            supports_summary=True,
        )
        render_facts(
            historical_supporting,
            "Historical or Linked Facts",
            historical=True,
            supports_summary=True,
        )
        render_facts(
            other_facts,
            "Other Retrieved Facts",
            historical=False,
            supports_summary=False,
        )

        if other_entries:
            entry_blocks = [
                self._render_entry(entry, f"[Context {index}]", annotations)
                for index, entry in enumerate(other_entries, 1)
            ]
            sections.append("[Other Retrieved Context]\n" + "\n\n".join(entry_blocks))

        return "\n\n".join(sections)

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """Format contexts for answer generation without changing retrieval."""
        contexts, annotations = self._prepare_contexts(contexts)
        if self.enable_thread_evidence_crosscheck:
            return self._format_thread_evidence_contexts(contexts, annotations)

        formatted = []
        for i, entry in enumerate(contexts, 1):
            formatted.append(
                self._render_entry(entry, f"[Context {i}]", annotations)
            )

        return "\n\n".join(formatted)

    def _build_answer_prompt(self, query: str, context_str: str) -> str:
        """
        Build answer generation prompt
        """
        crosscheck_requirements = ""
        if self.enable_thread_evidence_crosscheck:
            crosscheck_requirements = """
6. Thread summaries are compressed orientation only.
7. A \"supports Thread <id>\" label identifies facts from the same memory thread.
8. Verify every state claimed by a thread summary against its supporting facts.
9. If a summary conflicts with any supporting fact, use the fact.
10. For historical questions, respect the fact validity interval and supersede markers.
11. Do not infer details that are absent from the supporting facts.
"""
        return f"""
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
{context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format
{crosscheck_requirements}

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
```json
{{
  "reasoning": "The context explicitly states the meeting time as 2025-11-16T14:00:00",
  "answer": "16 November 2025 at 2:00 PM"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
"""
