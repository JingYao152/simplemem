"""MemWeaver write pipeline - Call A / Call B (design doc section 3).

The session is the atomic write unit: its boundary comes from the data
(``Dialogue.timestamp`` changing), so there is no flush-size parameter. For each
session:

* **Call A** (1 LLM call) assigns the session's turns to threads. Every existing
  thread's one-line summary goes into the prompt - the thread count is
  structurally bounded, so no pre-filtering parameter is needed.
* **Call B** (1 LLM call per touched thread, threads in parallel) extracts facts
  from the turns assigned to that thread, decides each fact's weave relation
  against *all* of the thread's existing facts, and rewrites the thread's living
  summary.
* **Weaving execution** is deterministic code: supersede closes the old fact's
  validity interval, refine/bridge write bidirectional edges.

``finalize()`` additionally runs the cross-thread sweep, which catches
knowledge-update pairs Call A split across threads.

Every LLM decision point has a deterministic fallback and each fallback's rate
is recorded in :attr:`MemWeaver.stats` as a system health indicator.
"""

import concurrent.futures
from dataclasses import dataclass, field
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from simplemem.core.database.vector_store import VectorStore
from simplemem.core.memory_builder import MemoryBuilder
from simplemem.core.memweaver.context import (
    context_digest,
    context_prefix,
    contextual_embed_text,
)
from simplemem.core.memweaver.dates import parse_session_datetime, to_day
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
from simplemem.core.memweaver.prompts import (
    ASSIGNMENT_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    SWEEP_SYSTEM_PROMPT,
    build_sweep_prompt,
    build_thread_assignment_prompt,
    build_thread_update_prompt,
)
from simplemem.core.models.memory_entry import (
    KIND_FACT,
    WEAVE_BRIDGE,
    WEAVE_NONE,
    WEAVE_REFINE,
    WEAVE_SUPERSEDE,
    Dialogue,
    MemoryEntry,
)
from simplemem.core.settings import settings as config
from simplemem.core.utils.llm_client import LLMClient


#: Cross-thread neighbours examined per open fact in the finalize sweep
#: (design doc section 3: "top-3 跨线程近邻"). Structural, not tuned.
SWEEP_NEIGHBOURS = 3

#: Rows pulled from the vector index before cross-thread filtering, so that
#: SWEEP_NEIGHBOURS cross-thread neighbours can actually be found.
SWEEP_SCAN_DEPTH = 12

#: Pairs per sweep judgement call. Prompt plumbing only - every candidate pair is
#: judged either way, so this changes no decision.
SWEEP_BATCH_SIZE = 20

_VALID_WEAVE_OPS = (WEAVE_NONE, WEAVE_SUPERSEDE, WEAVE_REFINE, WEAVE_BRIDGE)
_VALID_IMPACTS = ("none", "minor", "major")


@dataclass
class ThreadAssignment:
    """Turns of one session routed to one thread."""

    thread_id: str
    title: str
    is_new: bool
    turns: List[Dialogue] = field(default_factory=list)


@dataclass
class ThreadUpdate:
    """Parsed Call B output for one thread."""

    facts: List[MemoryEntry] = field(default_factory=list)
    weaves: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    summary: str = ""
    summary_impact: str = "none"
    outdated_facts: List[int] = field(default_factory=list)
    used_fallback: bool = False


class MemWeaver:
    """Write-time self-organizing memory fabric.

    Drop-in replacement for :class:`~simplemem.core.memory_builder.MemoryBuilder`
    on the write path: same ``add_dialogue`` / ``add_dialogues`` /
    ``process_remaining`` surface, plus ``finalize()`` for the sweep.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        enable_weaving: Optional[bool] = None,
        enable_sweep: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        temperature: Optional[float] = None,
        fallback_extractor: Optional[MemoryBuilder] = None,
        enable_recontext: Optional[bool] = None,
        enable_entity_profiles: Optional[bool] = None,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.enable_weaving = (
            enable_weaving
            if enable_weaving is not None
            else getattr(config, "ENABLE_WEAVING", True)
        )
        self.enable_sweep = (
            enable_sweep
            if enable_sweep is not None
            else getattr(config, "ENABLE_SWEEP", True)
        )
        # P1: context-inheriting embeddings + semantically triggered re-embedding.
        # Off = facts are embedded as bare sentences, i.e. pure SimpleMem.
        self.enable_recontext = (
            enable_recontext
            if enable_recontext is not None
            else getattr(config, "ENABLE_RECONTEXT", True)
        )
        # P1: person-level living profiles in the retrieval pool.
        self.enable_entity_profiles = (
            enable_entity_profiles
            if enable_entity_profiles is not None
            else getattr(config, "ENABLE_ENTITY_PROFILES", True)
        )
        self.max_parallel_workers = (
            max_parallel_workers
            if max_parallel_workers is not None
            else getattr(config, "MAX_PARALLEL_WORKERS", 4)
        )
        # One global temperature for every MemWeaver decision (design doc
        # section 6). Decision variance is a reported property of the method,
        # not something tuned away.
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(config, "LLM_TEMPERATURE", 0.7)
        )

        # Deterministic fallback when Call B cannot be parsed: plain SimpleMem
        # extraction of the same turns, written with op=none.
        self._fallback_extractor = fallback_extractor or MemoryBuilder(
            llm_client=llm_client,
            vector_store=vector_store,
            enable_parallel_processing=False,
        )

        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0
        # Health indicators; Call B runs threads in parallel, so counting is locked.
        self._stats_lock = threading.Lock()
        self.stats: Dict[str, int] = {
            "sessions": 0,
            "threads_created": 0,
            "facts_written": 0,
            "call_a_fallback": 0,
            "call_a_partial_fallback": 0,
            "call_b_fallback": 0,
            "weave_supersede": 0,
            "weave_refine": 0,
            "weave_bridge": 0,
            "weave_none": 0,
            "weave_invalid_target": 0,
            "summary_rewrites": 0,
            "summary_skipped": 0,
            "outdated_signals": 0,
            "recontext_reembedded": 0,
            "recontext_up_to_date": 0,
            "profiles_written": 0,
            "sweep_pairs_judged": 0,
            "sweep_supersedes": 0,
            "sweep_unordered_pairs": 0,
            "sweep_parse_failures": 0,
        }

    # ------------------------------------------------------------------
    # Ingestion surface (mirrors MemoryBuilder)
    # ------------------------------------------------------------------

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True) -> None:
        self.dialogue_buffer.append(dialogue)
        if auto_process:
            self._process_complete_sessions()

    def add_dialogues(
        self,
        dialogues: List[Dialogue],
        auto_process: bool = True,
    ) -> None:
        self.dialogue_buffer.extend(dialogues)
        if auto_process:
            self._process_complete_sessions()

    def process_remaining(self) -> None:
        """Flush the trailing session still held in the buffer."""
        if not self.dialogue_buffer:
            return
        pending = self.dialogue_buffer
        self.dialogue_buffer = []
        for session in self._split_sessions(pending):
            self._process_session(session)

    def finalize(self) -> None:
        """Flush the buffer, run the cross-thread sweep, optimize the store."""
        self.process_remaining()

        if self.enable_sweep:
            self._cross_thread_sweep()
        else:
            print("[MemWeaver] Cross-thread sweep disabled (ENABLE_SWEEP=False)")

        try:
            self.vector_store.optimize()
        except Exception as error:  # optimization is best-effort
            print(f"[MemWeaver] optimize skipped: {error}")

        self.print_stats()

    def _bump(self, name: str, amount: int = 1) -> None:
        """Increment a health indicator (Call B threads share the counters)."""
        with self._stats_lock:
            self.stats[name] = self.stats.get(name, 0) + amount

    def print_stats(self) -> None:
        """Report fallback rates and weave counts as health indicators."""
        print("\n[MemWeaver] Write-side health indicators")
        for name, value in self.stats.items():
            print(f"  {name:26s}: {value}")

    # ------------------------------------------------------------------
    # Session handling
    # ------------------------------------------------------------------

    def _process_complete_sessions(self) -> None:
        """Process every session whose boundary is already known.

        The trailing group stays buffered: more turns of the same session may
        still arrive. ``process_remaining()`` flushes it.
        """
        sessions = self._split_sessions(self.dialogue_buffer)
        if len(sessions) <= 1:
            return

        for session in sessions[:-1]:
            self._process_session(session)
        self.dialogue_buffer = sessions[-1]

    @staticmethod
    def _split_sessions(dialogues: Sequence[Dialogue]) -> List[List[Dialogue]]:
        """Split a dialogue run into sessions on every timestamp change."""
        sessions: List[List[Dialogue]] = []
        current_stamp = object()

        for dialogue in dialogues:
            stamp = dialogue.timestamp
            if not sessions or stamp != current_stamp:
                sessions.append([dialogue])
                current_stamp = stamp
            else:
                sessions[-1].append(dialogue)
        return sessions

    def _process_session(self, turns: List[Dialogue]) -> None:
        if not turns:
            return

        session_date, session_datetime = parse_session_datetime(turns[0].timestamp)
        self._bump("sessions")
        print(
            f"\n[MemWeaver] Session {self.stats['sessions']} "
            f"({session_date or 'undated'}): {len(turns)} turns"
        )

        snapshot = load_fabric(self.vector_store)
        assignments = self._assign_threads(turns, snapshot, session_date)
        print(
            f"[MemWeaver] Call A -> {len(assignments)} thread(s): "
            + ", ".join(
                f"{a.thread_id}{'(new)' if a.is_new else ''}:{len(a.turns)}turns"
                for a in assignments
            )
        )

        updates = self._run_thread_updates(assignments, snapshot, session_date)
        self._apply_session(
            assignments, updates, snapshot, session_date, session_datetime
        )
        self.processed_count += len(turns)

    # ------------------------------------------------------------------
    # Call A - thread assignment
    # ------------------------------------------------------------------

    def _assign_threads(
        self,
        turns: List[Dialogue],
        snapshot: FabricSnapshot,
        session_date: str,
    ) -> List[ThreadAssignment]:
        catalogue = [
            snapshot.threads[thread_id].one_line()
            for thread_id in sorted(snapshot.threads)
        ]
        prompt = build_thread_assignment_prompt(turns, catalogue, session_date)

        groups: Optional[List[Tuple[str, List[int]]]] = None
        try:
            response = self._chat(prompt, ASSIGNMENT_SYSTEM_PROMPT)
            groups = self._parse_assignment(response, len(turns), snapshot)
        except Exception as error:
            print(f"[MemWeaver] Call A failed: {error}")

        if groups is None:
            # Deterministic degradation: the whole session becomes one new thread.
            self._bump("call_a_fallback")
            print("[MemWeaver] Call A fallback: whole session -> one new thread")
            groups = [(f"new:{self._session_thread_title(session_date)}", list(range(1, len(turns) + 1)))]

        return self._materialize_assignments(groups, turns, snapshot, session_date)

    def _parse_assignment(
        self,
        response: str,
        turn_count: int,
        snapshot: FabricSnapshot,
    ) -> Optional[List[Tuple[str, List[int]]]]:
        """Parse Call A output; return None to request the whole-session fallback."""
        data = self.llm_client.extract_json(response)
        if isinstance(data, dict):
            data = data.get("assignments")
        if not isinstance(data, list) or not data:
            return None

        seen: set = set()
        groups: List[Tuple[str, List[int]]] = []

        for item in data:
            if not isinstance(item, dict):
                return None
            thread_ref = item.get("thread")
            if not isinstance(thread_ref, str) or not thread_ref.strip():
                return None
            thread_ref = thread_ref.strip()

            if thread_ref.startswith("existing:"):
                thread_id = thread_ref[len("existing:"):].strip()
                if thread_id not in snapshot.threads:
                    # Referencing a thread that does not exist invalidates the
                    # whole assignment (design doc section 3).
                    print(
                        f"[MemWeaver] Call A referenced unknown thread {thread_id!r}"
                    )
                    return None
                thread_ref = f"existing:{thread_id}"
            elif thread_ref.startswith("new:"):
                title = thread_ref[len("new:"):].strip()
                thread_ref = f"new:{title}" if title else "new:"
            else:
                return None

            raw_turns = item.get("turns")
            if not isinstance(raw_turns, list):
                return None

            indices = []
            for value in raw_turns:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    return None
                if not 1 <= index <= turn_count or index in seen:
                    continue  # out of range or already claimed: first claim wins
                seen.add(index)
                indices.append(index)

            if indices:
                groups.append((thread_ref, indices))

        if not groups:
            return None

        missing = [index for index in range(1, turn_count + 1) if index not in seen]
        if missing:
            # Partial coverage: keep what was assigned, park the rest in a new
            # thread rather than dropping information.
            self._bump("call_a_partial_fallback")
            print(
                f"[MemWeaver] Call A left {len(missing)} turn(s) unassigned "
                "-> parked in a new thread"
            )
            groups.append(("new:", missing))

        return groups

    def _materialize_assignments(
        self,
        groups: List[Tuple[str, List[int]]],
        turns: List[Dialogue],
        snapshot: FabricSnapshot,
        session_date: str,
    ) -> List[ThreadAssignment]:
        """Resolve thread references to concrete ids, merging repeated targets."""
        new_ids: Dict[str, str] = {}
        assignments: Dict[str, ThreadAssignment] = {}
        order: List[str] = []

        for thread_ref, indices in groups:
            if thread_ref.startswith("existing:"):
                thread_id = thread_ref[len("existing:"):]
                title = snapshot.threads[thread_id].title
                is_new = False
            else:
                title = thread_ref[len("new:"):].strip() or self._session_thread_title(
                    session_date
                )
                key = title.lower()
                if key in new_ids:
                    thread_id = new_ids[key]
                else:
                    thread_id = snapshot.next_thread_id(new_ids.values())
                    new_ids[key] = thread_id
                    self._bump("threads_created")
                is_new = thread_id not in snapshot.threads

            if thread_id not in assignments:
                assignments[thread_id] = ThreadAssignment(
                    thread_id=thread_id, title=title, is_new=is_new
                )
                order.append(thread_id)
            assignments[thread_id].turns.extend(turns[index - 1] for index in indices)

        for thread_id in order:
            assignments[thread_id].turns.sort(key=lambda turn: turn.dialogue_id)

        return [assignments[thread_id] for thread_id in order]

    @staticmethod
    def _session_thread_title(session_date: str) -> str:
        return f"Session {session_date}" if session_date else "Unsorted session"

    # ------------------------------------------------------------------
    # Call B - per-thread extraction, weaving and summary rewrite
    # ------------------------------------------------------------------

    def _run_thread_updates(
        self,
        assignments: List[ThreadAssignment],
        snapshot: FabricSnapshot,
        session_date: str,
    ) -> List[ThreadUpdate]:
        """Run Call B for each touched thread (threads in parallel)."""
        if not assignments:
            return []
        if len(assignments) == 1 or self.max_parallel_workers <= 1:
            return [
                self._update_thread(assignment, snapshot, session_date)
                for assignment in assignments
            ]

        workers = min(self.max_parallel_workers, len(assignments))
        updates: List[Optional[ThreadUpdate]] = [None] * len(assignments)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._update_thread, assignment, snapshot, session_date
                ): position
                for position, assignment in enumerate(assignments)
            }
            for future in concurrent.futures.as_completed(futures):
                position = futures[future]
                try:
                    updates[position] = future.result()
                except Exception as error:
                    print(
                        f"[MemWeaver] Call B for "
                        f"{assignments[position].thread_id} failed: {error}"
                    )
                    self._bump("call_b_fallback")
                    updates[position] = ThreadUpdate(used_fallback=True)

        return [update or ThreadUpdate(used_fallback=True) for update in updates]

    def _update_thread(
        self,
        assignment: ThreadAssignment,
        snapshot: FabricSnapshot,
        session_date: str,
    ) -> ThreadUpdate:
        state = snapshot.threads.get(
            assignment.thread_id,
            ThreadState(
                thread_id=assignment.thread_id, title=assignment.title, summary=""
            ),
        )
        candidates = snapshot.facts(assignment.thread_id)
        prompt = build_thread_update_prompt(
            thread_title=state.title or assignment.title,
            thread_summary=state.summary,
            candidates=candidates,
            turns=assignment.turns,
            session_date=session_date,
        )

        for attempt in range(3):
            try:
                response = self._chat(prompt, EXTRACTION_SYSTEM_PROMPT)
                return self._parse_thread_update(
                    response, assignment, candidates, session_date
                )
            except Exception as error:
                if attempt < 2:
                    print(
                        f"[MemWeaver] Call B parse attempt {attempt + 1}/3 for "
                        f"{assignment.thread_id} failed: {error}. Retrying..."
                    )
                else:
                    print(
                        f"[MemWeaver] Call B failed for {assignment.thread_id}: "
                        f"{error}. Falling back to plain extraction."
                    )

        return self._fallback_thread_update(assignment, session_date)

    def _parse_thread_update(
        self,
        response: str,
        assignment: ThreadAssignment,
        candidates: Sequence[MemoryEntry],
        session_date: str,
    ) -> ThreadUpdate:
        data = self.llm_client.extract_json(response)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object but got {type(data)}")

        raw_facts = data.get("facts")
        if not isinstance(raw_facts, list):
            raise ValueError("Call B output has no 'facts' array")

        update = ThreadUpdate()
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            restatement = item.get("lossless_restatement")
            if not isinstance(restatement, str) or not restatement.strip():
                continue

            timestamp = item.get("timestamp")
            timestamp = timestamp if isinstance(timestamp, str) and timestamp else None
            update.facts.append(
                MemoryEntry(
                    lossless_restatement=restatement.strip(),
                    keywords=_string_list(item.get("keywords")),
                    timestamp=timestamp,
                    location=item.get("location") if isinstance(item.get("location"), str) else None,
                    persons=_string_list(item.get("persons")),
                    entities=_string_list(item.get("entities")),
                    topic=item.get("topic") if isinstance(item.get("topic"), str) else None,
                    kind=KIND_FACT,
                    thread_id=assignment.thread_id,
                    # Fact's own timestamp when it has one, else the session date.
                    valid_from=to_day(timestamp) or session_date,
                )
            )
            update.weaves.append(self._parse_weave(item.get("weave"), candidates))

        summary = data.get("summary")
        update.summary = summary.strip() if isinstance(summary, str) else ""
        impact = data.get("summary_impact")
        update.summary_impact = (
            impact if isinstance(impact, str) and impact in _VALID_IMPACTS else "none"
        )
        update.outdated_facts = [
            index
            for index in _int_list(data.get("outdated_facts"))
            if 1 <= index <= len(candidates)
        ]
        return update

    def _parse_weave(
        self,
        raw: Any,
        candidates: Sequence[MemoryEntry],
    ) -> Tuple[str, Optional[int]]:
        """Parse one weave decision; anything unusable degrades to op=none."""
        if not isinstance(raw, dict):
            return WEAVE_NONE, None

        op = raw.get("op")
        if not isinstance(op, str) or op not in _VALID_WEAVE_OPS:
            return WEAVE_NONE, None
        if op == WEAVE_NONE:
            return WEAVE_NONE, None

        target = raw.get("target")
        try:
            index = int(target)
        except (TypeError, ValueError):
            self._bump("weave_invalid_target")
            return WEAVE_NONE, None

        if not 1 <= index <= len(candidates):
            self._bump("weave_invalid_target")
            return WEAVE_NONE, None
        return op, index

    def _fallback_thread_update(
        self,
        assignment: ThreadAssignment,
        session_date: str,
    ) -> ThreadUpdate:
        """Plain SimpleMem extraction, attached to the thread with op=none."""
        self._bump("call_b_fallback")
        try:
            facts = self._fallback_extractor._generate_memory_entries(assignment.turns)
        except Exception as error:
            print(f"[MemWeaver] Fallback extraction failed: {error}")
            facts = []

        update = ThreadUpdate(used_fallback=True)
        for fact in facts or []:
            fact.kind = KIND_FACT
            fact.thread_id = assignment.thread_id
            fact.valid_from = to_day(fact.timestamp) or session_date
            update.facts.append(fact)
            update.weaves.append((WEAVE_NONE, None))
        return update

    # ------------------------------------------------------------------
    # Weaving execution (deterministic)
    # ------------------------------------------------------------------

    def _apply_session(
        self,
        assignments: List[ThreadAssignment],
        updates: List[ThreadUpdate],
        snapshot: FabricSnapshot,
        session_date: str,
        session_datetime: str,
    ) -> None:
        # The thread context a fact inherits is the summary as it stands *after*
        # this session, so prefix and stored summary always agree (P1).
        prefixes = {
            assignment.thread_id: self._thread_prefix(assignment, update, snapshot)
            for assignment, update in zip(assignments, updates)
        }

        new_facts: List[MemoryEntry] = []
        embed_texts: List[str] = []
        for assignment, update in zip(assignments, updates):
            prefix = prefixes[assignment.thread_id]
            digest = context_digest(prefix)
            for fact in update.facts:
                fact.context_digest = digest
                new_facts.append(fact)
                embed_texts.append(
                    contextual_embed_text(prefix, fact.lossless_restatement)
                )

        if new_facts:
            # The prefix reaches the embedder only; every stored field keeps the
            # fact's own text.
            self.vector_store.add_entries(new_facts, embed_texts=embed_texts)
            self._bump("facts_written", len(new_facts))

        summary_entries: List[MemoryEntry] = []
        stale_summary_ids: List[str] = []

        for assignment, update in zip(assignments, updates):
            self._apply_weaves(assignment, update, snapshot, session_date)
            self._bump("outdated_signals", len(update.outdated_facts))

            entry, stale_id = self._plan_summary_write(
                assignment, update, snapshot, session_date, session_datetime
            )
            if entry is not None:
                summary_entries.append(entry)
            if stale_id:
                stale_summary_ids.append(stale_id)

        # Summary rewrite = delete the old row, insert the new one (fixed id).
        if stale_summary_ids:
            self.vector_store.delete_by_ids(stale_summary_ids)
        if summary_entries:
            self.vector_store.add_entries(summary_entries)

        # P1: the facts Call B named as outdated are re-embedded under the
        # rewritten context (local compute, zero API cost).
        for assignment, update in zip(assignments, updates):
            self._recontextualize(
                assignment, update, snapshot, prefixes[assignment.thread_id]
            )

        # P1: refresh the living profile of every speaker this session involved.
        self._update_profiles(
            assignments, updates, snapshot, session_date, session_datetime
        )

    def _apply_weaves(
        self,
        assignment: ThreadAssignment,
        update: ThreadUpdate,
        snapshot: FabricSnapshot,
        session_date: str,
    ) -> None:
        candidates = snapshot.facts(assignment.thread_id)

        for fact, (op, target) in zip(update.facts, update.weaves):
            if op == WEAVE_NONE or target is None:
                self._bump("weave_none")
                continue
            if not self.enable_weaving:
                self._bump("weave_none")
                continue

            candidate = candidates[target - 1]
            if op == WEAVE_SUPERSEDE:
                if execute_supersede(
                    self.vector_store, candidate, fact, session_date
                ):
                    self._bump("weave_supersede")
                    print(
                        f"[MemWeaver] supersede: {candidate.entry_id} closed on "
                        f"{session_date} by {fact.entry_id}"
                    )
                else:
                    self._bump("weave_none")
            elif op in (WEAVE_REFINE, WEAVE_BRIDGE):
                if execute_link(self.vector_store, op, fact, candidate):
                    self._bump(f"weave_{op}")
                else:
                    self._bump("weave_none")

    def _resolve_summary(
        self,
        assignment: ThreadAssignment,
        update: ThreadUpdate,
        snapshot: FabricSnapshot,
    ) -> Tuple[str, str, str, bool]:
        """The thread's summary after this session.

        Returns ``(summary_text, title, existing_summary_id, rewrite)``. Single
        source of truth for both the stored summary row and the P1 context
        prefix, so a fact's ``context_digest`` always matches the summary that
        is actually stored.
        """
        state = snapshot.threads.get(assignment.thread_id)
        existing_id = state.summary_entry_id if state else ""
        title = (state.title if state and state.title else assignment.title) or ""
        current = state.summary if state else ""

        if existing_id and (update.summary_impact == "none" or not update.summary):
            # Deterministic fallback for the summary decision: do not rewrite.
            return current, title, existing_id, False

        text = update.summary or (
            update.facts[0].lossless_restatement if update.facts else title
        )
        if not text:
            return current, title, existing_id, False
        return text, title, existing_id, True

    def _thread_prefix(
        self,
        assignment: ThreadAssignment,
        update: ThreadUpdate,
        snapshot: FabricSnapshot,
    ) -> str:
        """One-line thread context used as this thread's embedding prefix (P1)."""
        if not self.enable_recontext:
            return ""
        summary, title, _, _ = self._resolve_summary(assignment, update, snapshot)
        return context_prefix(summary, title)

    def _plan_summary_write(
        self,
        assignment: ThreadAssignment,
        update: ThreadUpdate,
        snapshot: FabricSnapshot,
        session_date: str,
        session_datetime: str,
    ) -> Tuple[Optional[MemoryEntry], str]:
        """Decide whether the living summary is rewritten this session."""
        summary_text, title, existing_id, rewrite = self._resolve_summary(
            assignment, update, snapshot
        )
        if not rewrite:
            self._bump("summary_skipped")
            return None, ""

        self._bump("summary_rewrites")
        entry = build_thread_summary_entry(
            thread_id=assignment.thread_id,
            title=title,
            summary=summary_text,
            session_date=session_date,
            session_datetime=session_datetime,
            persons=[person for fact in update.facts for person in fact.persons],
        )
        return entry, existing_id

    # ------------------------------------------------------------------
    # P1 - representation co-evolution
    # ------------------------------------------------------------------

    def _recontextualize(
        self,
        assignment: ThreadAssignment,
        update: ThreadUpdate,
        snapshot: FabricSnapshot,
        prefix: str,
    ) -> None:
        """Re-embed the facts Call B named as outdated under the new context.

        The semantic trigger is Call B's ``outdated_facts``; the digest makes the
        operation idempotent, so a fact already embedded under this context is
        left alone. Fact text is never touched.
        """
        if not self.enable_recontext or not update.outdated_facts:
            return

        candidates = snapshot.facts(assignment.thread_id)
        digest = context_digest(prefix)
        targets: List[MemoryEntry] = []
        texts: List[str] = []

        for index in update.outdated_facts:
            fact = candidates[index - 1]
            if fact.context_digest == digest:
                self._bump("recontext_up_to_date")
                continue
            targets.append(fact)
            texts.append(contextual_embed_text(prefix, fact.lossless_restatement))

        if not targets:
            return

        self.vector_store.reembed_entries(targets, texts, digest)
        self._bump("recontext_reembedded", len(targets))
        print(
            f"[MemWeaver] re-contextualized {len(targets)} fact(s) of "
            f"{assignment.thread_id} under the rewritten summary"
        )

    def _update_profiles(
        self,
        assignments: List[ThreadAssignment],
        updates: List[ThreadUpdate],
        snapshot: FabricSnapshot,
        session_date: str,
        session_datetime: str,
    ) -> None:
        """Rebuild the living profile of each speaker this session involved."""
        if not self.enable_entity_profiles:
            return

        spoke_in: Dict[str, set] = {}
        for assignment in assignments:
            for turn in assignment.turns:
                if turn.speaker:
                    spoke_in.setdefault(turn.speaker, set()).add(assignment.thread_id)
        if not spoke_in:
            return

        # Fabric view after this session: summaries as resolved above, plus the
        # facts just written. No extra table scan.
        threads_after = dict(snapshot.threads)
        facts_after = {
            thread_id: list(facts)
            for thread_id, facts in snapshot.facts_by_thread.items()
        }
        for assignment, update in zip(assignments, updates):
            summary, title, _, rewrite = self._resolve_summary(
                assignment, update, snapshot
            )
            previous = snapshot.threads.get(assignment.thread_id)
            threads_after[assignment.thread_id] = ThreadState(
                thread_id=assignment.thread_id,
                title=title,
                summary=summary,
                updated_on=session_date if rewrite
                else (previous.updated_on if previous else session_date),
            )
            facts_after.setdefault(assignment.thread_id, []).extend(update.facts)

        stale_ids: List[str] = []
        profile_entries: List[MemoryEntry] = []
        for name in sorted(spoke_in):
            threads = speaker_threads(
                threads_after, facts_after, name, spoke_in[name]
            )
            if not threads:
                continue
            existing = snapshot.profiles.get(name)
            if existing is not None:
                stale_ids.append(existing.entry_id)
            profile_entries.append(
                build_entity_profile_entry(
                    name, threads, session_date, session_datetime
                )
            )

        if stale_ids:
            self.vector_store.delete_by_ids(stale_ids)
        if profile_entries:
            self.vector_store.add_entries(profile_entries)
            self._bump("profiles_written", len(profile_entries))

    # ------------------------------------------------------------------
    # finalize() - cross-thread supersede sweep
    # ------------------------------------------------------------------

    def _cross_thread_sweep(self) -> None:
        """Catch supersede pairs Call A split across threads.

        A necessary component rather than a safety net: knowledge-update facts
        legitimately land in different threads. Neighbour search is local
        embedding work (zero LLM cost); only the high-similarity pairs are sent
        to the LLM, batched.
        """
        snapshot = load_fabric(self.vector_store)
        open_facts = [
            entry
            for entry in snapshot.entries
            if entry.kind == KIND_FACT and entry.is_open
        ]
        if len(open_facts) < 2:
            return

        print(f"\n[MemWeaver] Cross-thread sweep over {len(open_facts)} open facts")
        pairs = self._collect_sweep_pairs(open_facts)
        if not pairs:
            print("[MemWeaver] Sweep found no cross-thread candidate pairs")
            return

        print(f"[MemWeaver] Sweep judging {len(pairs)} candidate pair(s)")
        verdicts = self._judge_sweep_pairs(pairs)

        closed: set = set()
        for pair in pairs:
            if not verdicts.get(pair["index"]):
                continue
            earlier, later = pair["earlier"], pair["later"]
            if earlier.entry_id in closed:
                continue
            if execute_supersede(
                self.vector_store, earlier, later, later.valid_from
            ):
                closed.add(earlier.entry_id)
                self._bump("sweep_supersedes")
                print(
                    f"[MemWeaver] sweep supersede: {earlier.entry_id} "
                    f"({earlier.thread_id}) closed by {later.entry_id} "
                    f"({later.thread_id}) on {later.valid_from}"
                )

    def _collect_sweep_pairs(self, open_facts: List[MemoryEntry]) -> List[dict]:
        """Find cross-thread nearest neighbours with local embeddings only."""
        vectors = self.vector_store.embedding_model.encode_documents(
            [fact.lossless_restatement for fact in open_facts]
        )
        by_key: Dict[Tuple[str, str], dict] = {}
        unorderable: set = set()

        for fact, vector in zip(open_facts, vectors):
            neighbours = self.vector_store.semantic_search_by_vector(
                vector.tolist(), top_k=SWEEP_SCAN_DEPTH
            )
            picked = 0
            for neighbour in neighbours:
                if picked >= SWEEP_NEIGHBOURS:
                    break
                if (
                    neighbour.entry_id == fact.entry_id
                    or neighbour.kind != KIND_FACT
                    or not neighbour.is_open
                    or neighbour.thread_id == fact.thread_id
                ):
                    continue
                picked += 1

                key = tuple(sorted((fact.entry_id, neighbour.entry_id)))
                if key in by_key or key in unorderable:
                    continue
                ordered = self._order_by_date(fact, neighbour)
                if ordered is None:
                    # Same-day facts (typically the same session) carry no
                    # deterministic direction, so they are left alone. Counted
                    # once per distinct pair, not once per encounter.
                    unorderable.add(key)
                    self._bump("sweep_unordered_pairs")
                    continue
                earlier, later = ordered
                by_key[key] = {
                    "earlier": earlier,
                    "later": later,
                    "earlier_text": earlier.lossless_restatement,
                    "later_text": later.lossless_restatement,
                    "earlier_date": earlier.valid_from,
                    "later_date": later.valid_from,
                }

        pairs = list(by_key.values())
        for index, pair in enumerate(pairs, 1):
            pair["index"] = index
        return pairs

    @staticmethod
    def _order_by_date(
        left: MemoryEntry,
        right: MemoryEntry,
    ) -> Optional[Tuple[MemoryEntry, MemoryEntry]]:
        """Order a pair earlier-first; None when dates cannot order them.

        Direction is decided by dates in code, never by the LLM - the LLM only
        judges whether the pair is the same evolving fact.
        """
        left_date, right_date = left.valid_from, right.valid_from
        if not left_date or not right_date or left_date == right_date:
            return None
        return (left, right) if left_date < right_date else (right, left)

    def _judge_sweep_pairs(self, pairs: List[dict]) -> Dict[int, bool]:
        batches = [
            pairs[start:start + SWEEP_BATCH_SIZE]
            for start in range(0, len(pairs), SWEEP_BATCH_SIZE)
        ]
        verdicts: Dict[int, bool] = {}

        def judge(batch: List[dict]) -> Dict[int, bool]:
            try:
                response = self._chat(build_sweep_prompt(batch), SWEEP_SYSTEM_PROMPT)
                data = self.llm_client.extract_json(response)
            except Exception as error:
                print(f"[MemWeaver] Sweep judgement failed: {error}")
                self._bump("sweep_parse_failures")
                return {}

            if isinstance(data, dict):
                data = data.get("judgements")
            if not isinstance(data, list):
                self._bump("sweep_parse_failures")
                return {}

            allowed = {pair["index"] for pair in batch}
            parsed: Dict[int, bool] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if index in allowed:
                    parsed[index] = bool(item.get("supersedes"))
            return parsed

        workers = min(self.max_parallel_workers, len(batches))
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                for result in executor.map(judge, batches):
                    verdicts.update(result)
        else:
            for batch in batches:
                verdicts.update(judge(batch))

        self._bump("sweep_pairs_judged", len(verdicts))
        return verdicts

    # ------------------------------------------------------------------

    def _chat(self, prompt: str, system_prompt: str) -> str:
        response_format = None
        if getattr(config, "USE_JSON_FORMAT", False):
            response_format = {"type": "json_object"}

        return self.llm_client.chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format=response_format,
        )


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _int_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    numbers = []
    for item in value:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    return numbers
