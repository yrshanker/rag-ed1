"""Deterministic embeddings for testing.

The :class:`PassThroughEmbeddings` class implements a lightweight bag-of-words
embedding that hashes tokens into a fixed-size vector. This avoids any network
calls to external language model APIs and is suitable for offline tests.
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings


class PassThroughEmbeddings(Embeddings):
    """Simple hash-based embeddings that require no external service."""

    def __init__(self, dimension: int = 128) -> None:
        """Create a new :class:`PassThroughEmbeddings` instance.

        Parameters
        ----------
        dimension:
            Length of the generated embedding vectors.
        """
        self._dim = dimension

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        for token in text.lower().split():
            idx = hash(token) % self._dim
            vec[idx] += 1.0
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._embed(text)
