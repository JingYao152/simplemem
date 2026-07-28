"""Provider-neutral facade for SimpleMem's three retrieval paths."""

from typing import Any, Callable, Dict, List, Optional

from simplemem.core.database.vector_store_backend import (
    LanceDBVectorStoreBackend,
    VectorStoreBackend,
    VectorStoreRecord,
    VectorStoreSearchResult,
)
from simplemem.core.models.memory_entry import KIND_FACT, MemoryEntry
from simplemem.core.settings import settings as config
from simplemem.core.utils.embedding import EmbeddingModel


BackendFactory = Callable[[int], VectorStoreBackend]


class VectorStore:
    """Coordinate embeddings with a pluggable multi-view storage backend."""

    #: Metadata fields a stored entry carries (MemWeaver fabric fields included).
    METADATA_FIELDS = frozenset(
        {
            "lossless_restatement",
            "keywords",
            "timestamp",
            "location",
            "persons",
            "entities",
            "topic",
            "kind",
            "thread_id",
            "valid_from",
            "valid_until",
            "superseded_by",
            "links",
            "context_digest",
            "source_turn_ids",
        }
    )

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        storage_options: Optional[Dict[str, Any]] = None,
        backend_factory: Optional[BackendFactory] = None,
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.storage_options = storage_options
        # Bumped on every mutation so readers can cache derived state (such as
        # MemWeaver's as-of anchor) without rescanning the table.
        self._revision = 0

        if backend_factory is None:
            self.backend = LanceDBVectorStoreBackend(
                db_path=self.db_path,
                table_name=self.table_name,
                vector_dimension=self.embedding_model.dimension,
                storage_options=self.storage_options,
            )
        else:
            self.backend = backend_factory(self.embedding_model.dimension)

    @property
    def revision(self) -> int:
        """Monotonic counter of mutations applied through this facade."""
        return self._revision

    @property
    def db(self) -> Any:
        """Expose the default backend's database handle for compatibility."""
        return getattr(self.backend, "db", None)

    @property
    def table(self) -> Any:
        """Expose the default backend's table handle for compatibility."""
        return getattr(self.backend, "table", None)

    def add_entries(
        self,
        entries: List[MemoryEntry],
        embed_texts: Optional[List[str]] = None,
    ) -> None:
        """Embed and insert a batch of memory entries.

        ``embed_texts`` overrides what is handed to the embedder, position by
        position, while the stored text stays the entry's own restatement. This
        is how MemWeaver's context-inheriting embeddings keep the thread-context
        prefix out of every stored field (design doc section 4).
        """
        if not entries:
            return

        restatements = [entry.lossless_restatement for entry in entries]
        if embed_texts is None:
            embed_texts = restatements
        elif len(embed_texts) != len(entries):
            raise ValueError(
                f"embed_texts has {len(embed_texts)} items for {len(entries)} entries"
            )

        vectors = self.embedding_model.encode_documents(embed_texts)
        records = [
            VectorStoreRecord(
                entry_id=entry.entry_id,
                vector=vector.tolist(),
                metadata=self._entry_to_metadata(entry),
            )
            for entry, vector in zip(entries, vectors)
        ]

        self.backend.insert(records)
        self._revision += 1
        print(f"Added {len(entries)} memory entries")

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryEntry]:
        """Search the semantic retrieval path.

        ``filters`` is applied as a backend prefilter so the top-k budget is
        spent on visible entries only (MemWeaver's as-of retrieval).
        """
        try:
            if self.backend.count() == 0:
                return []

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            results = self.backend.semantic_search(
                query_vector.tolist(),
                top_k=top_k,
                filters=filters,
            )
            return self._results_to_entries(results)
        except Exception as error:
            print(f"Error during semantic search: {error}")
            return []

    def semantic_search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryEntry]:
        """Search the semantic path with an already-computed vector.

        Lets callers batch their embedding work (MemWeaver's finalize sweep
        embeds every open fact in one pass).
        """
        try:
            if self.backend.count() == 0:
                return []
            results = self.backend.semantic_search(
                list(query_vector),
                top_k=top_k,
                filters=filters,
            )
            return self._results_to_entries(results)
        except Exception as error:
            print(f"Error during semantic search: {error}")
            return []

    def keyword_search(
        self,
        keywords: List[str],
        top_k: int = 3,
    ) -> List[MemoryEntry]:
        """Search the lexical full-text retrieval path."""
        try:
            if not keywords or self.backend.count() == 0:
                return []
            return self._results_to_entries(
                self.backend.keyword_search(keywords, top_k=top_k)
            )
        except Exception as error:
            print(f"Error during keyword search: {error}")
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Search the structured metadata retrieval path."""
        try:
            if self.backend.count() == 0:
                return []
            if not any([persons, timestamp_range, location, entities]):
                return []

            return self._results_to_entries(
                self.backend.structured_search(
                    persons=persons,
                    timestamp_range=timestamp_range,
                    location=location,
                    entities=entities,
                    top_k=top_k,
                )
            )
        except Exception as error:
            print(f"Error during structured search: {error}")
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        return self._results_to_entries(self.backend.get_all())

    def get_by_ids(self, entry_ids: List[str]) -> List[MemoryEntry]:
        """Fetch entries by id, skipping ids that are not stored."""
        if not entry_ids:
            return []
        return self._results_to_entries(self.backend.get_by_ids(entry_ids))

    def find_by_field(self, field: str, values: List[str]) -> List[MemoryEntry]:
        """Fetch entries whose ``field`` equals one of ``values``.

        Empty values are dropped: an empty string is the *absence* of an edge, so
        looking it up would match every entry that has no such edge.
        """
        if field not in self.METADATA_FIELDS:
            raise ValueError(f"Unknown metadata field: {field!r}")

        wanted = [value for value in dict.fromkeys(values or []) if value]
        if not wanted:
            return []
        return self._results_to_entries(self.backend.find_by_field(field, wanted))

    def update_metadata(self, entry_id: str, fields: Dict[str, Any]) -> None:
        """Update metadata fields of a stored entry in place.

        Used by MemWeaver's weaving execution (closing ``valid_until``, setting
        ``superseded_by``, appending weave ``links``).
        """
        if not fields:
            return

        unknown = set(fields) - self.METADATA_FIELDS
        if unknown:
            raise ValueError(f"Unknown metadata fields: {sorted(unknown)}")

        self.backend.update_metadata(entry_id, fields)
        self._revision += 1

    def reembed_entries(
        self,
        entries: List[MemoryEntry],
        embed_texts: List[str],
        context_digest: str = "",
    ) -> int:
        """Recompute stored vectors from new embedding texts, in place.

        Used by MemWeaver's re-contextualization: a rewritten thread summary
        changes the context a fact should be embedded under, but not the fact
        itself. Embedding is local, so this costs no API calls.
        """
        if not entries:
            return 0
        if len(embed_texts) != len(entries):
            raise ValueError(
                f"embed_texts has {len(embed_texts)} items for {len(entries)} entries"
            )

        vectors = self.embedding_model.encode_documents(embed_texts)
        fields = {"context_digest": context_digest}
        for entry, vector in zip(entries, vectors):
            self.backend.update_vector(entry.entry_id, vector.tolist(), fields)
            entry.context_digest = context_digest

        self._revision += 1
        return len(entries)

    def delete_by_ids(self, entry_ids: List[str]) -> None:
        """Delete entries by id (summary/profile rewrite = delete + insert)."""
        if not entry_ids:
            return
        self.backend.delete_by_ids(entry_ids)
        self._revision += 1

    def optimize(self) -> None:
        """Optimize backend indexes after bulk insertions."""
        self.backend.optimize()
        print("Table optimized")

    def clear(self) -> None:
        """Clear all backend data."""
        self.backend.clear()
        self._revision += 1
        print("Database cleared")

    @staticmethod
    def _entry_to_metadata(entry: MemoryEntry) -> Dict[str, Any]:
        """Flatten an entry into backend-neutral metadata."""
        return {
            "lossless_restatement": entry.lossless_restatement,
            "keywords": entry.keywords,
            "timestamp": entry.timestamp or "",
            "location": entry.location or "",
            "persons": entry.persons,
            "entities": entry.entities,
            "topic": entry.topic or "",
            "kind": entry.kind or KIND_FACT,
            "thread_id": entry.thread_id or "",
            "valid_from": entry.valid_from or "",
            "valid_until": entry.valid_until or "",
            "superseded_by": entry.superseded_by or "",
            "links": entry.links,
            "context_digest": entry.context_digest or "",
            "source_turn_ids": entry.source_turn_ids,
        }

    @staticmethod
    def _results_to_entries(
        results: List[VectorStoreSearchResult],
    ) -> List[MemoryEntry]:
        entries = []
        for result in results:
            try:
                metadata = result.metadata
                entries.append(
                    MemoryEntry(
                        entry_id=result.entry_id,
                        lossless_restatement=metadata["lossless_restatement"],
                        keywords=list(metadata.get("keywords") or []),
                        timestamp=metadata.get("timestamp") or None,
                        location=metadata.get("location") or None,
                        persons=list(metadata.get("persons") or []),
                        entities=list(metadata.get("entities") or []),
                        topic=metadata.get("topic") or None,
                        kind=metadata.get("kind") or KIND_FACT,
                        thread_id=metadata.get("thread_id") or "",
                        valid_from=metadata.get("valid_from") or "",
                        valid_until=metadata.get("valid_until") or "",
                        superseded_by=metadata.get("superseded_by") or "",
                        links=list(metadata.get("links") or []),
                        context_digest=metadata.get("context_digest") or "",
                        source_turn_ids=list(metadata.get("source_turn_ids") or []),
                    )
                )
            except Exception as error:
                print(f"Warning: Failed to parse result: {error}")
        return entries
