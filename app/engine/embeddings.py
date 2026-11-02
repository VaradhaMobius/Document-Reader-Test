"""Embedding backends for retrieval."""

from __future__ import annotations

import hashlib
import math
import os
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class EmbeddingBackend(ABC):
    """Produces L2-normalized vectors for text."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return shape (n, d) float32 matrix of unit vectors."""


class HashEmbeddingBackend(EmbeddingBackend):
    """Offline deterministic bag-of-tokens embedding (no API required)."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _tokenize(text)
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI text-embedding models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.client = OpenAI(api_key=key)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # OpenAI accepts batches; keep them moderate
        batch_size = 64
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return matrix / norms


def build_embedding_backend(
    mode: str = "auto",
    *,
    api_key: str | None = None,
) -> EmbeddingBackend:
    """
    mode:
      - openai: force OpenAI
      - local: force hash embeddings
      - auto: OpenAI when key present, else local
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if mode == "local":
        return HashEmbeddingBackend()
    if mode == "openai":
        return OpenAIEmbeddingBackend(api_key=key)
    if key:
        return OpenAIEmbeddingBackend(api_key=key)
    return HashEmbeddingBackend()


def _tokenize(text: str) -> list[str]:
    raw = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in raw.split() if token]


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity for unit vectors (dot product)."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query.reshape(-1)
    q_norm = math.sqrt(float(np.dot(q, q))) or 1.0
    return (matrix @ q) / q_norm