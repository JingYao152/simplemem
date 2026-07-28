"""Persistent coverage debt tracking for MemWeaver write-time extraction.

Coverage debts retain dialogue turns that contained information but were not
mapped to any stored fact.  They are deliberately kept outside ``VectorStore``:
a debt is an audit record, not evidence that answer generation may retrieve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from typing import Dict, Iterable, List, Sequence

from simplemem.core.models.memory_entry import Dialogue, MemoryEntry


_PENDING = "pending"
_REPAIRED = "repaired"
_EXEMPT = "exempt"
_VALID_STATUSES = {_PENDING, _REPAIRED, _EXEMPT}


def audit_turn_coverage(
    turns: Sequence[Dialogue],
    facts: Sequence[MemoryEntry],
    exempt_turn_ids: Iterable[int],
) -> List[Dialogue]:
    """Return turns that have neither fact support nor an explicit exemption."""
    allowed_ids = {turn.dialogue_id for turn in turns}
    covered_ids = {
        turn_id
        for fact in facts
        for turn_id in fact.source_turn_ids
        if turn_id in allowed_ids
    }
    exempt_ids = {turn_id for turn_id in exempt_turn_ids if turn_id in allowed_ids}
    return [
        turn
        for turn in turns
        if turn.dialogue_id not in covered_ids and turn.dialogue_id not in exempt_ids
    ]


def _serialize_turn(turn: Dialogue) -> Dict[str, object]:
    return {
        "dialogue_id": turn.dialogue_id,
        "speaker": turn.speaker,
        "content": turn.content,
        "timestamp": turn.timestamp,
    }


def _deserialize_turn(data: Dict[str, object]) -> Dialogue:
    return Dialogue(
        dialogue_id=int(data["dialogue_id"]),
        speaker=str(data["speaker"]),
        content=str(data["content"]),
        timestamp=str(data["timestamp"]) if data.get("timestamp") else None,
    )


@dataclass
class CoverageDebt:
    """One unresolved set of source turns from a MemWeaver session."""

    debt_id: str
    session_id: str
    thread_id: str
    source_turns: List[Dialogue]
    nearby_turns: List[Dialogue]
    entities: List[str]
    topic: str
    status: str = _PENDING
    attempt_count: int = 0
    repair_fact_ids: List[str] = field(default_factory=list)
    created_at: str = ""
    resolved_at: str = ""

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unknown coverage debt status: {self.status!r}")
        if not self.created_at:
            self.created_at = _utc_now()

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["source_turns"] = [_serialize_turn(turn) for turn in self.source_turns]
        data["nearby_turns"] = [_serialize_turn(turn) for turn in self.nearby_turns]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CoverageDebt":
        return cls(
            debt_id=str(data["debt_id"]),
            session_id=str(data["session_id"]),
            thread_id=str(data["thread_id"]),
            source_turns=[
                _deserialize_turn(turn) for turn in data.get("source_turns", [])
            ],
            nearby_turns=[
                _deserialize_turn(turn) for turn in data.get("nearby_turns", [])
            ],
            entities=[str(entity) for entity in data.get("entities", [])],
            topic=str(data.get("topic") or ""),
            status=str(data.get("status") or _PENDING),
            attempt_count=int(data.get("attempt_count") or 0),
            repair_fact_ids=[
                str(fact_id) for fact_id in data.get("repair_fact_ids", [])
            ],
            created_at=str(data.get("created_at") or ""),
            resolved_at=str(data.get("resolved_at") or ""),
        )


class CoverageDebtStore:
    """Atomically persist coverage debts in a JSON document beside LanceDB."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._debts = self._load()

    def record(
        self,
        *,
        session_id: str,
        thread_id: str,
        source_turns: Sequence[Dialogue],
        nearby_turns: Sequence[Dialogue],
        entities: Sequence[str],
        topic: str,
    ) -> CoverageDebt:
        """Create or refresh the stable debt for a source-turn set."""
        if not source_turns:
            raise ValueError("Coverage debt requires at least one source turn")

        debt_id = self._debt_id(session_id, thread_id, source_turns)
        with self._lock:
            existing = self._debts.get(debt_id)
            if existing is not None:
                existing.nearby_turns = list(nearby_turns)
                existing.entities = _unique_strings(entities)
                existing.topic = topic
                self._write()
                return existing

            debt = CoverageDebt(
                debt_id=debt_id,
                session_id=session_id,
                thread_id=thread_id,
                source_turns=list(source_turns),
                nearby_turns=list(nearby_turns),
                entities=_unique_strings(entities),
                topic=topic,
            )
            self._debts[debt_id] = debt
            self._write()
            return debt

    def pending(self) -> List[CoverageDebt]:
        with self._lock:
            return [
                debt
                for debt in self._debts.values()
                if debt.status == _PENDING
            ]

    def pending_related(
        self,
        thread_ids: Iterable[str],
        entities: Iterable[str],
    ) -> List[CoverageDebt]:
        """Return pending debts touching a thread or a normalized entity."""
        thread_set = {thread_id for thread_id in thread_ids if thread_id}
        entity_set = {entity.casefold() for entity in entities if entity}
        with self._lock:
            related = []
            for debt in self.pending():
                debt_entities = {entity.casefold() for entity in debt.entities}
                if debt.thread_id in thread_set or debt_entities & entity_set:
                    related.append(debt)
            return related

    def mark_repaired(
        self,
        debt_id: str,
        repair_fact_ids: Sequence[str],
    ) -> CoverageDebt:
        with self._lock:
            debt = self._require(debt_id)
            debt.status = _REPAIRED
            debt.repair_fact_ids = list(repair_fact_ids)
            debt.resolved_at = _utc_now()
            self._write()
            return debt

    def increment_attempt(self, debt_id: str) -> CoverageDebt:
        with self._lock:
            debt = self._require(debt_id)
            debt.attempt_count += 1
            self._write()
            return debt

    def mark_exempt(self, debt_id: str) -> CoverageDebt:
        with self._lock:
            debt = self._require(debt_id)
            debt.status = _EXEMPT
            debt.resolved_at = _utc_now()
            self._write()
            return debt

    def _load(self) -> Dict[str, CoverageDebt]:
        if not self.path.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as handle:
            raw_debts = json.load(handle)
        return {
            debt.debt_id: debt
            for debt in (CoverageDebt.from_dict(raw) for raw in raw_debts)
        }

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_name(f".{self.path.name}.tmp")
        payload = [debt.to_dict() for debt in self._debts.values()]
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary_path.replace(self.path)

    @staticmethod
    def _debt_id(
        session_id: str,
        thread_id: str,
        source_turns: Sequence[Dialogue],
    ) -> str:
        turn_ids = ",".join(str(turn.dialogue_id) for turn in source_turns)
        raw = f"{session_id}\n{thread_id}\n{turn_ids}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _require(self, debt_id: str) -> CoverageDebt:
        try:
            return self._debts[debt_id]
        except KeyError as error:
            raise KeyError(f"Unknown coverage debt: {debt_id}") from error


def _unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        normalized = value.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
