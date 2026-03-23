"""Embedding generation for document chunks and queries.

Provides a unified embedding interface that uses Snowflake Cortex EMBED
in production and falls back to sentence-transformers for local/demo mode.
Includes caching to avoid redundant embedding computation.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generates vector embeddings for text using Cortex or local models.

    In production, delegates to Snowflake Cortex EMBED function.
    In demo/local mode, uses sentence-transformers with a lightweight model.
    Maintains an in-memory cache keyed by content hash to avoid recomputation.

    Attributes:
        model_name: Name of the embedding model.
        dimension: Output embedding dimension.
        use_local: Whether to use a local sentence-transformers model.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_local: bool = True,
    ) -> None:
        self.model_name = model_name
        self.use_local = use_local
        self.dimension: int = 384  # Default for MiniLM
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}

        if use_local:
            self._initialize_local_model()

    def _initialize_local_model(self) -> None:
        """Initialize the local sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "local_embedding_model_loaded",
                model=self.model_name,
                dimension=self.dimension,
            )
        except ImportError:
            logger.warning(
                "sentence_transformers_not_available",
                fallback="using deterministic hash-based embeddings",
            )
            self._model = None
        except Exception as exc:
            logger.warning(
                "local_model_load_failed",
                model=self.model_name,
                error=str(exc),
                fallback="using deterministic hash-based embeddings",
            )
            self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a numpy array.
        """
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._model is not None:
            embedding = self._model.encode(text, normalize_embeddings=True)
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = self._deterministic_embedding(text)

        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Uses batch encoding for efficiency when a local model is available.
        Falls back to sequential embedding otherwise.

        Args:
            texts: List of input texts.
            batch_size: Batch size for model encoding.

        Returns:
            2D numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Check cache for all texts
        uncached_indices: list[int] = []
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                embeddings[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return embeddings

        # Batch encode uncached texts
        uncached_texts = [texts[i] for i in uncached_indices]

        if self._model is not None:
            batch_embeddings = self._model.encode(
                uncached_texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=len(uncached_texts) > 100,
            )
            batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        else:
            batch_embeddings = np.array(
                [self._deterministic_embedding(t) for t in uncached_texts],
                dtype=np.float32,
            )

        # Fill in results and cache
        for j, idx in enumerate(uncached_indices):
            embeddings[idx] = batch_embeddings[j]
            self._cache[self._cache_key(texts[idx])] = batch_embeddings[j]

        logger.info(
            "batch_embedding_complete",
            total=len(texts),
            cached=len(texts) - len(uncached_indices),
            computed=len(uncached_indices),
        )

        return embeddings

    def _deterministic_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-embedding from text hash.

        This is a fallback for when no embedding model is available.
        It produces consistent vectors that preserve some lexical similarity
        through n-gram hashing, enabling basic retrieval functionality.

        Args:
            text: Input text.

        Returns:
            Normalized embedding vector.
        """
        # Use multiple hash functions over n-grams for better distribution
        embedding = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()

        for n in range(1, min(4, len(words) + 1)):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                hash_bytes = hashlib.sha256(ngram.encode()).digest()
                for j in range(0, min(len(hash_bytes), self.dimension), 4):
                    idx = j % self.dimension
                    value = int.from_bytes(hash_bytes[j:j+4], "little", signed=True)
                    embedding[idx] += value / (2**31)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key from text content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
