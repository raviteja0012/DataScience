"""Vector similarity search for RAG retrieval.

Implements an in-memory vector store using FAISS for efficient nearest-neighbor
search. In production, this would be replaced by Snowflake Cortex Search
service, but the interface remains identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """A single vector search result.

    Attributes:
        chunk_id: Identifier of the matched chunk.
        text: Text content of the matched chunk.
        score: Similarity score (higher is more relevant).
        metadata: Chunk metadata (source, section, etc.).
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore:
    """In-memory vector store backed by FAISS for similarity search.

    Stores document chunk embeddings and supports efficient k-nearest-neighbor
    retrieval. Maintains a parallel metadata store for chunk text and metadata.

    For production deployment, this is swapped for Snowflake Cortex Search
    via the same interface.

    Attributes:
        dimension: Embedding vector dimension.
        metric: Distance metric ('cosine' or 'l2').
    """

    def __init__(self, dimension: int = 384, metric: str = "cosine") -> None:
        self.dimension = dimension
        self.metric = metric
        self._index: Any = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadata: list[dict[str, Any]] = []
        self._initialized = False

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return len(self._chunk_ids)

    def _ensure_index(self) -> None:
        """Initialize the FAISS index if not already created."""
        if self._index is not None:
            return

        try:
            import faiss

            if self.metric == "cosine":
                # Inner product on normalized vectors = cosine similarity
                self._index = faiss.IndexFlatIP(self.dimension)
            else:
                self._index = faiss.IndexFlatL2(self.dimension)

            self._initialized = True
            logger.info("faiss_index_created", dimension=self.dimension, metric=self.metric)

        except ImportError:
            logger.warning("faiss_not_available", fallback="numpy_brute_force")
            self._index = None
            self._initialized = True

    def add(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        texts: list[str],
        metadata_list: list[dict[str, Any]],
    ) -> None:
        """Add vectors with associated metadata to the store.

        Args:
            embeddings: Embedding vectors, shape (n, dimension).
            chunk_ids: Unique identifiers for each chunk.
            texts: Text content for each chunk.
            metadata_list: Metadata dictionaries for each chunk.

        Raises:
            ValueError: If input arrays have mismatched lengths.
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunk_ids)} chunk_ids"
            )

        self._ensure_index()

        # Normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        if self._index is not None:
            self._index.add(embeddings)
        else:
            # Store raw embeddings for brute-force fallback
            if not hasattr(self, "_raw_embeddings"):
                self._raw_embeddings = embeddings
            else:
                self._raw_embeddings = np.vstack([self._raw_embeddings, embeddings])

        self._chunk_ids.extend(chunk_ids)
        self._chunk_texts.extend(texts)
        self._chunk_metadata.extend(metadata_list)

        logger.info("vectors_added", count=len(chunk_ids), total=self.size)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search for the most similar vectors to a query.

        Args:
            query_embedding: Query vector, shape (dimension,).
            top_k: Number of results to return.
            score_threshold: Minimum similarity score to include.

        Returns:
            List of SearchResult objects, ordered by descending similarity.
        """
        if self.size == 0:
            return []

        self._ensure_index()

        # Normalize query vector
        query = query_embedding.astype(np.float32).reshape(1, -1)
        if self.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        top_k = min(top_k, self.size)

        if self._index is not None:
            scores, indices = self._index.search(query, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute-force fallback
            scores, indices = self._brute_force_search(query[0], top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= self.size:
                continue
            if score < score_threshold:
                continue

            results.append(SearchResult(
                chunk_id=self._chunk_ids[idx],
                text=self._chunk_texts[idx],
                score=float(score),
                metadata=self._chunk_metadata[idx],
            ))

        logger.debug(
            "vector_search_complete",
            top_k=top_k,
            results_returned=len(results),
        )

        return results

    def _brute_force_search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback brute-force similarity search using numpy."""
        if not hasattr(self, "_raw_embeddings"):
            return np.array([]), np.array([])

        if self.metric == "cosine":
            similarities = self._raw_embeddings @ query
        else:
            distances = np.linalg.norm(self._raw_embeddings - query, axis=1)
            similarities = -distances  # Negate so higher = closer

        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        return top_scores, top_indices

    def clear(self) -> None:
        """Remove all vectors from the store."""
        self._index = None
        self._chunk_ids.clear()
        self._chunk_texts.clear()
        self._chunk_metadata.clear()
        if hasattr(self, "_raw_embeddings"):
            del self._raw_embeddings
        self._initialized = False
        logger.info("vector_store_cleared")
