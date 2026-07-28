"""Vector store backend contracts and the default LanceDB implementation."""

from dataclasses import dataclass
from enum import Enum
import os
import re
import threading
from typing import Any, Dict, List, Optional, Protocol, Sequence

import lancedb
import pyarrow as pa


class ScoreOrder(str, Enum):
    """Ordering semantics for scores returned by a retrieval path."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


#: Comparison operators a filter predicate may use.
FILTER_OPERATORS = frozenset({"=", "!=", ">", ">=", "<", "<="})


@dataclass(frozen=True)
class FieldPredicate:
    """A single comparison against one metadata field."""

    op: str
    value: Any

    def __post_init__(self) -> None:
        if self.op not in FILTER_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {self.op!r}")


@dataclass(frozen=True)
class AnyOf:
    """Disjunction of predicates over one metadata field.

    MemWeaver's as-of filter needs it: an entry is visible at the anchor when
    ``valid_until = ''`` (still open) OR ``valid_until >= anchor``.
    """

    predicates: Sequence[FieldPredicate]

    def __post_init__(self) -> None:
        if not self.predicates:
            raise ValueError("AnyOf requires at least one predicate")


@dataclass(frozen=True)
class VectorStoreRecord:
    """A vector and its backend-neutral memory metadata."""

    entry_id: str
    vector: Sequence[float]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class VectorStoreSearchResult:
    """A backend-neutral retrieval result."""

    entry_id: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class VectorStoreBackend(Protocol):
    """Storage contract for semantic, lexical, and structured retrieval."""

    semantic_score_order: ScoreOrder
    keyword_score_order: ScoreOrder

    def insert(self, records: Sequence[VectorStoreRecord]) -> None:
        """Insert memory records and their dense vectors."""
        ...

    def semantic_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        """Return dense-vector results ranked from best to worst."""
        ...

    def keyword_search(
        self,
        keywords: Sequence[str],
        top_k: int,
    ) -> List[VectorStoreSearchResult]:
        """Return full-text results ranked from best to worst."""
        ...

    def structured_search(
        self,
        persons: Optional[Sequence[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorStoreSearchResult]:
        """Return records matching structured metadata constraints."""
        ...

    def count(self) -> int:
        """Return the number of stored records."""
        ...

    def get_all(self) -> List[VectorStoreSearchResult]:
        """Return every stored record."""
        ...

    def get_by_ids(
        self,
        entry_ids: Sequence[str],
    ) -> List[VectorStoreSearchResult]:
        """Return the records with the given ids (missing ids are skipped)."""
        ...

    def find_by_field(
        self,
        field: str,
        values: Sequence[str],
    ) -> List[VectorStoreSearchResult]:
        """Return every record whose ``field`` equals one of ``values``.

        Needed to walk an edge backwards: MemWeaver's one-hop expansion has to
        find the entries that point *at* a candidate (``superseded_by``).
        """
        ...

    def update_metadata(self, entry_id: str, fields: Dict[str, Any]) -> None:
        """Update metadata fields of one stored record in place."""
        ...

    def update_vector(
        self,
        entry_id: str,
        vector: Sequence[float],
        fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Replace one record's dense vector (and optionally metadata) in place."""
        ...

    def delete_by_ids(self, entry_ids: Sequence[str]) -> None:
        """Delete the records with the given ids."""
        ...

    def optimize(self) -> None:
        """Optimize backend indexes after bulk insertion."""
        ...

    def clear(self) -> None:
        """Remove all stored records and recreate backend state."""
        ...


class LanceDBVectorStoreBackend:
    """Default vector store backend using LanceDB and Tantivy."""

    semantic_score_order = ScoreOrder.ASCENDING
    keyword_score_order = ScoreOrder.DESCENDING
    _field_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        db_path: str,
        table_name: str,
        vector_dimension: int,
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.vector_dimension = vector_dimension
        # The full-text index is not maintained incrementally by LanceDB, so any
        # mutation marks it stale and the next lexical query rebuilds it. This
        # matters for MemWeaver, which writes once per session instead of once
        # per run.
        self._fts_dirty = True
        self._write_lock = threading.Lock()
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        if self._is_cloud_storage:
            self.db = lancedb.connect(
                self.db_path,
                storage_options=storage_options,
            )
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self) -> None:
        schema = pa.schema(
            [
                pa.field("entry_id", pa.string()),
                pa.field("lossless_restatement", pa.string()),
                pa.field("keywords", pa.list_(pa.string())),
                pa.field("timestamp", pa.string()),
                pa.field("location", pa.string()),
                pa.field("persons", pa.list_(pa.string())),
                pa.field("entities", pa.list_(pa.string())),
                pa.field("topic", pa.string()),
                # MemWeaver fabric fields (design doc section 2)
                pa.field("kind", pa.string()),
                pa.field("thread_id", pa.string()),
                pa.field("valid_from", pa.string()),
                pa.field("valid_until", pa.string()),
                pa.field("superseded_by", pa.string()),
                pa.field("links", pa.list_(pa.string())),
                pa.field("context_digest", pa.string()),
                pa.field("source_turn_ids", pa.list_(pa.int64())),
                pa.field(
                    "vector",
                    pa.list_(pa.float32(), self.vector_dimension),
                ),
            ]
        )

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")
            self._migrate_table(schema)

    def _migrate_table(self, schema: pa.Schema) -> None:
        """Add fabric columns to a table created before MemWeaver."""
        existing = set(self.table.schema.names)
        missing = [field for field in schema if field.name not in existing]
        if not missing:
            return

        for field in missing:
            try:
                if pa.types.is_string(field.type):
                    # SQL literal default keeps the column non-null for old rows.
                    self.table.add_columns({field.name: "''"})
                else:
                    self.table.add_columns(field)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to migrate table {self.table_name!r}: cannot add "
                    f"column {field.name!r} ({error}). Clear the table to "
                    "recreate it with the current schema."
                ) from error
        print(
            f"Migrated table {self.table_name}: added "
            f"{', '.join(field.name for field in missing)}"
        )

    def _ensure_fts_index(self) -> None:
        """(Re)build the full-text index when it is stale."""
        with self._write_lock:
            if not self._fts_dirty:
                return
            if self.table.count_rows() == 0:
                return

            try:
                if self._is_cloud_storage:
                    self.table.create_fts_index(
                        "lossless_restatement",
                        use_tantivy=False,
                        replace=True,
                    )
                    print("FTS index created (native mode for cloud storage)")
                else:
                    self.table.create_fts_index(
                        "lossless_restatement",
                        use_tantivy=True,
                        tokenizer_name="en_stem",
                        replace=True,
                    )
                    print("FTS index created (Tantivy mode)")
            except Exception as error:
                # Clearing the flag on failure too bounds the cost to one attempt
                # per mutation instead of one per lexical query; the next write
                # marks the index stale again, so a transient failure recovers.
                print(f"FTS index creation skipped: {error}")
            finally:
                self._fts_dirty = False

    def insert(self, records: Sequence[VectorStoreRecord]) -> None:
        if not records:
            return

        rows = [
            {
                "entry_id": record.entry_id,
                **record.metadata,
                "vector": list(record.vector),
            }
            for record in records
        ]
        with self._write_lock:
            self.table.add(rows)
            self._fts_dirty = True

    def semantic_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreSearchResult]:
        if self.count() == 0:
            return []

        query = self.table.search(list(query_vector))
        if filters:
            query = query.where(self._build_filter_expression(filters), prefilter=True)

        results = self._rows_to_results(
            query.limit(top_k).to_list(),
            score_field="_distance",
        )
        results.sort(key=lambda result: result.score)
        return results

    def keyword_search(
        self,
        keywords: Sequence[str],
        top_k: int,
    ) -> List[VectorStoreSearchResult]:
        if not keywords or self.count() == 0:
            return []

        self._ensure_fts_index()
        query = " ".join(keywords)
        results = self._rows_to_results(
            self.table.search(query).limit(top_k).to_list(),
            score_field="_score",
        )
        results.sort(key=lambda result: result.score, reverse=True)
        return results

    def structured_search(
        self,
        persons: Optional[Sequence[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[VectorStoreSearchResult]:
        if self.count() == 0:
            return []
        if not any([persons, timestamp_range, location, entities]):
            return []

        conditions = []
        if persons:
            values = ", ".join(self._quote(value) for value in persons)
            conditions.append(f"array_has_any(persons, make_array({values}))")
        if location:
            conditions.append(f"location LIKE '%{self._escape(location)}%'")
        if entities:
            values = ", ".join(self._quote(value) for value in entities)
            conditions.append(f"array_has_any(entities, make_array({values}))")
        if timestamp_range:
            start_time, end_time = timestamp_range
            conditions.append(
                f"timestamp >= {self._quote(start_time)} "
                f"AND timestamp <= {self._quote(end_time)}"
            )

        query = self.table.search().where(" AND ".join(conditions), prefilter=True)
        if top_k:
            query = query.limit(top_k)
        return self._rows_to_results(query.to_list())

    def count(self) -> int:
        return self.table.count_rows()

    def get_all(self) -> List[VectorStoreSearchResult]:
        return self._rows_to_results(self.table.to_arrow().to_pylist())

    def get_by_ids(
        self,
        entry_ids: Sequence[str],
    ) -> List[VectorStoreSearchResult]:
        if not entry_ids or self.count() == 0:
            return []

        values = ", ".join(self._quote(entry_id) for entry_id in entry_ids)
        rows = self.table.search().where(f"entry_id IN ({values})").to_list()
        results = self._rows_to_results(rows)

        # Preserve the caller's id order; drop ids that no longer exist.
        by_id = {result.entry_id: result for result in results}
        return [by_id[entry_id] for entry_id in entry_ids if entry_id in by_id]

    def find_by_field(
        self,
        field: str,
        values: Sequence[str],
    ) -> List[VectorStoreSearchResult]:
        if not values or self.count() == 0:
            return []
        if not self._field_pattern.fullmatch(field):
            raise ValueError(f"Invalid lookup field: {field!r}")

        literals = ", ".join(self._quote(value) for value in values)
        rows = self.table.search().where(f"{field} IN ({literals})").to_list()
        return self._rows_to_results(rows)

    def update_metadata(self, entry_id: str, fields: Dict[str, Any]) -> None:
        if not fields:
            return

        for field in fields:
            if not self._field_pattern.fullmatch(field):
                raise ValueError(f"Invalid metadata field: {field!r}")
        if "entry_id" in fields or "vector" in fields:
            raise ValueError("entry_id and vector cannot be updated in place")

        with self._write_lock:
            self.table.update(
                where=f"entry_id = {self._quote(entry_id)}",
                values=dict(fields),
            )
            self._fts_dirty = True

    def update_vector(
        self,
        entry_id: str,
        vector: Sequence[float],
        fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        values: Dict[str, Any] = {"vector": list(vector)}
        for field, value in (fields or {}).items():
            if not self._field_pattern.fullmatch(field):
                raise ValueError(f"Invalid metadata field: {field!r}")
            if field in ("entry_id", "vector"):
                raise ValueError(f"{field} cannot be updated through fields")
            values[field] = value

        with self._write_lock:
            self.table.update(
                where=f"entry_id = {self._quote(entry_id)}",
                values=values,
            )
            # The fact's text is unchanged by a re-embed, so the full-text index
            # stays valid; only the dense vector moved.

    def delete_by_ids(self, entry_ids: Sequence[str]) -> None:
        if not entry_ids:
            return

        values = ", ".join(self._quote(entry_id) for entry_id in entry_ids)
        with self._write_lock:
            self.table.delete(f"entry_id IN ({values})")
            self._fts_dirty = True

    def optimize(self) -> None:
        self.table.optimize()

    def clear(self) -> None:
        self.db.drop_table(self.table_name)
        self._fts_dirty = True
        self._init_table()

    @classmethod
    def _build_filter_expression(cls, filters: Dict[str, Any]) -> str:
        conditions = []
        for field, value in filters.items():
            if not cls._field_pattern.fullmatch(field):
                raise ValueError(f"Invalid semantic filter field: {field!r}")
            conditions.append(cls._build_field_condition(field, value))
        return " AND ".join(conditions)

    @classmethod
    def _build_field_condition(cls, field: str, value: Any) -> str:
        if isinstance(value, AnyOf):
            clauses = [
                cls._build_field_condition(field, predicate)
                for predicate in value.predicates
            ]
            return "(" + " OR ".join(clauses) + ")"
        if isinstance(value, FieldPredicate):
            return f"{field} {value.op} {cls._format_filter_value(value.value)}"
        return f"{field} = {cls._format_filter_value(value)}"

    @classmethod
    def _format_filter_value(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return cls._quote(value)
        raise TypeError(
            "Semantic filters support only string, boolean, integer, and float values"
        )

    @classmethod
    def _quote(cls, value: Any) -> str:
        return "'" + cls._escape(value) + "'"

    @staticmethod
    def _escape(value: Any) -> str:
        return str(value).replace("'", "''")

    @staticmethod
    def _rows_to_results(
        rows: Sequence[Dict[str, Any]],
        score_field: Optional[str] = None,
    ) -> List[VectorStoreSearchResult]:
        results = []
        for row in rows:
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"entry_id", "vector", "_distance", "_score"}
            }
            score = None
            if score_field is not None and row.get(score_field) is not None:
                score = float(row[score_field])
            results.append(
                VectorStoreSearchResult(
                    entry_id=row["entry_id"],
                    metadata=metadata,
                    score=score,
                )
            )
        return results
