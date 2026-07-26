"""Cross-encoder reranker - the one generic read-side component.

Design doc section 5 and 10: this is an *inherited* component, not part of
MemWeaver's contribution claims, and the baseline parity discipline says every
compared system is fitted with the same reranker before comparison. It therefore
lives outside ``memweaver/`` and knows nothing about the fabric.

It scores every candidate in the pool flatly (no tree, no cascade) and keeps the
top ``RERANK_TOP_K``. Inference is local; when the model cannot be loaded the
reranker degrades deterministically to the pool's incoming order, so a run in an
environment without the model still completes and says so once.
"""

import threading
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

from simplemem.core.settings import settings as config


#: Pairs per cross-encoder forward pass. Plumbing: batching changes no score.
RERANK_BATCH_SIZE = 32

T = TypeVar("T")


class CrossEncoderReranker:
    """Local cross-encoder scoring with a deterministic no-model fallback."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        model_factory: Optional[Callable[[str], object]] = None,
    ):
        self.model_name = model_name or getattr(
            config, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.top_k = top_k or getattr(config, "RERANK_TOP_K", 20)
        self._model_factory = model_factory
        self._model = None
        self._unavailable = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True once a model is loaded; False after a load failure."""
        return self._model is not None

    def _load(self):
        """Load the cross-encoder once, or mark it unavailable once."""
        if self._model is not None or self._unavailable:
            return self._model

        with self._lock:
            if self._model is None and not self._unavailable:
                try:
                    if self._model_factory is not None:
                        self._model = self._model_factory(self.model_name)
                    else:
                        from sentence_transformers import CrossEncoder

                        print(f"Loading reranker: {self.model_name}")
                        self._model = CrossEncoder(self.model_name)
                except Exception as error:
                    self._unavailable = True
                    print(
                        f"Reranker unavailable ({error}); keeping retrieval order. "
                        "Install sentence-transformers and make the model "
                        "reachable to enable reranking."
                    )
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> Optional[List[float]]:
        """Relevance scores for each document, or None when unavailable."""
        if not documents:
            return []

        model = self._load()
        if model is None:
            return None

        try:
            scores: List[float] = []
            for start in range(0, len(documents), RERANK_BATCH_SIZE):
                batch = [
                    (query, document)
                    for document in documents[start:start + RERANK_BATCH_SIZE]
                ]
                predicted = model.predict(batch)
                scores.extend(float(value) for value in predicted)
            return scores
        except Exception as error:
            print(f"Reranking failed ({error}); keeping retrieval order")
            return None

    def rerank(
        self,
        query: str,
        items: Sequence[T],
        to_text: Callable[[T], str],
        top_k: Optional[int] = None,
    ) -> Tuple[List[T], bool]:
        """Return ``(top_k items best-first, reranked?)``.

        Falls back to the incoming order truncated to ``top_k`` when the model is
        unavailable, so the capacity constant applies either way.
        """
        limit = top_k or self.top_k
        if not items:
            return [], False

        scores = self.score(query, [to_text(item) for item in items])
        if scores is None or len(scores) != len(items):
            return list(items)[:limit], False

        ordered = sorted(
            zip(items, scores), key=lambda pair: pair[1], reverse=True
        )
        return [item for item, _ in ordered[:limit]], True
