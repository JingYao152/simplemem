"""Retrieval hit rate measured directly from LoCoMo's QA ``evidence`` field.

Design doc section 8: "另用 QA 的 evidence 字段直接量检索命中率（session/dia_id
级），不必等端到端分数." This scores the contexts a retriever returned, so a write
pipeline change can be evaluated per question without waiting for end-to-end
answer quality.

Stdlib only, so the measurement is importable (and testable) without the answer
quality metric stack.
"""

from collections import defaultdict
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


#: An evidence turn counts as retrieved when some retrieved memory entry reaches
#: this IDF-weighted coverage of the turn's distinctive content. Evaluation-side
#: constant: it measures the system, it is not a system parameter.
#:
#: Calibrated on LoCoMo's own derived statements (the ``observation`` field, whose
#: entries cite the turns they came from) as a proxy for what a memory entry looks
#: like: at 0.3, a statement that does cite a turn reaches the threshold 49% of
#: the time, while 25 randomly drawn non-citing statements reach it 1.6% of the
#: time. The hit rate is therefore a relative measure between arms - a perfect
#: retriever does not score 1.0, because a faithful restatement keeps the turn's
#: fact and drops its chatter. The threshold-free mean coverage is reported next
#: to it.
EVIDENCE_HIT_THRESHOLD = 0.3

#: Metric keys produced by :meth:`EvidenceScorer.score`.
RETRIEVAL_METRIC_KEYS = (
    "retrieval_hit_any",
    "retrieval_hit_all",
    "retrieval_coverage",
    "retrieval_session_hit_all",
    "retrieval_evidence_count",
)

_WORD_PATTERN = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those there here so as of to in on at by for with
    from about into over after before between out up down off again further once
    i me my we our you your he him his she her it its they them their who whom which what when where why how
    is am are was were be been being do does did doing have has had having will would shall should can could may might must
    not no nor only own same too very s t just don now
    """.split()
)


def content_tokens(text: str) -> Set[str]:
    """Content words of a text, lowercased and stopword-filtered."""
    if not text:
        return set()
    return {
        token
        for token in _WORD_PATTERN.findall(str(text).lower())
        if token not in _STOPWORDS and len(token) > 1
    }


class EvidenceScorer:
    """Measures retrieval hit rate against one conversation's evidence turns.

    Memory entries are LLM restatements, so they carry no turn ids and cannot be
    matched to evidence by identity. Coverage is therefore measured lexically and
    weighted by IDF over the conversation's own turns: an evidence turn counts as
    retrieved when some retrieved entry carries at least
    ``EVIDENCE_HIT_THRESHOLD`` of the turn's distinctive content. IDF weighting is
    what makes the measure discriminative - unweighted overlap is dominated by
    conversational filler that any entry shares with any turn.

    Two granularities are reported:

    * dia_id level - per evidence turn (``hit_any`` / ``hit_all`` / ``coverage``)
    * session level - each retrieved entry is attributed to the session holding
      the turn it best matches, and ``session_hit_all`` asks whether every
      evidence session was reached

    The identical measurement runs on every arm of an A/B, so the delta is
    meaningful even though the absolute level depends on the threshold.
    """

    def __init__(self, turns: Iterable[Tuple[int, str, str]]):
        """``turns`` yields ``(session_id, dia_id, text)`` in any order."""
        self.turns: Dict[str, Tuple[int, Set[str]]] = {}
        self.session_turns: List[Tuple[int, Set[str]]] = []
        self._idf: Dict[str, float] = {}
        # Entry text is immutable within a conversation, so per-entry session
        # attribution is memoized across that conversation's questions.
        self._session_cache: Dict[str, Optional[int]] = {}

        frequencies: Dict[str, int] = defaultdict(int)
        documents = 0
        for session_id, dia_id, text in turns:
            tokens = content_tokens(text)
            self.turns[dia_id] = (session_id, tokens)
            if not tokens:
                continue
            documents += 1
            self.session_turns.append((session_id, tokens))
            for token in tokens:
                frequencies[token] += 1

        self._default_idf = math.log(documents + 1) if documents else 1.0
        for token, frequency in frequencies.items():
            self._idf[token] = math.log((documents + 1) / (1 + frequency))

    @classmethod
    def from_sample(cls, sample) -> "EvidenceScorer":
        """Build from a parsed LoCoMo sample (``conversation.sessions``)."""
        return cls(
            (session_id, turn.dia_id, turn.text)
            for session_id, session in sample.conversation.sessions.items()
            for turn in session.turns
        )

    def _weight(self, token: str) -> float:
        # Tokens unseen in this conversation are maximally distinctive.
        return self._idf.get(token, self._default_idf)

    def _mass(self, tokens: Set[str]) -> float:
        return sum(self._weight(token) for token in tokens)

    def _coverage(self, target: Set[str], candidate: Set[str], total: float) -> float:
        if total <= 0:
            return 0.0
        return self._mass(target & candidate) / total

    def _best_coverage(self, target: Set[str], candidates: Sequence[Set[str]]) -> float:
        total = self._mass(target)
        if total <= 0:
            return 0.0
        return max(
            (self._coverage(target, candidate, total) for candidate in candidates),
            default=0.0,
        )

    def _attributed_session(self, entry_id: str, tokens: Set[str]) -> Optional[int]:
        """Session holding the turn this entry most plausibly came from."""
        if entry_id in self._session_cache:
            return self._session_cache[entry_id]

        best_session, best_score = None, 0.0
        for session_id, turn_tokens in self.session_turns:
            score = self._coverage(turn_tokens, tokens, self._mass(turn_tokens))
            if score > best_score:
                best_session, best_score = session_id, score

        attributed = best_session if best_score >= EVIDENCE_HIT_THRESHOLD else None
        self._session_cache[entry_id] = attributed
        return attributed

    def score(self, evidence: Sequence[str], contexts: Sequence) -> Dict[str, float]:
        """Retrieval metrics for one question; empty when evidence is unusable.

        ``contexts`` are retrieved memory entries (anything exposing
        ``entry_id``, ``lossless_restatement`` and ``keywords``).
        """
        wanted = [
            self.turns[dia_id]
            for dia_id in evidence or []
            if isinstance(dia_id, str) and dia_id in self.turns
        ]
        wanted = [(session_id, tokens) for session_id, tokens in wanted if tokens]
        if not wanted:
            return {}

        entries = [
            (
                entry.entry_id,
                content_tokens(
                    f"{entry.lossless_restatement} {' '.join(entry.keywords or [])}"
                ),
            )
            for entry in contexts
        ]
        candidates = [tokens for _, tokens in entries]

        coverages = [self._best_coverage(tokens, candidates) for _, tokens in wanted]
        hits = [coverage >= EVIDENCE_HIT_THRESHOLD for coverage in coverages]

        retrieved_sessions = {
            session
            for entry_id, tokens in entries
            for session in [self._attributed_session(entry_id, tokens)]
            if session is not None
        }
        wanted_sessions = {session_id for session_id, _ in wanted}

        return {
            "retrieval_evidence_count": float(len(wanted)),
            "retrieval_coverage": sum(coverages) / len(coverages),
            "retrieval_hit_any": float(any(hits)),
            "retrieval_hit_all": float(all(hits)),
            "retrieval_session_hit_all": float(
                wanted_sessions.issubset(retrieved_sessions)
            ),
        }
