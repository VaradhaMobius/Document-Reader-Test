"""In-memory vector index over document chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import TextChunk
from .embeddings import EmbeddingBackend, cosine_similarity


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float


class VectorStore:
    """Simple numpy-backed similarity search."""

    def __init__(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        self.chunks = chunks
        self.embeddings = embeddings

    @classmethod
    def from_chunks(
        cls,
        chunks: list[TextChunk],
        backend: EmbeddingBackend,
    ) -> "VectorStore":
        texts = [chunk.text for chunk in chunks]
        embeddings = backend.embed(texts) if texts else np.zeros((0, 0), dtype=np.float32)
        return cls(chunks=chunks, embeddings=embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        scores = cosine_similarity(query_embedding, self.embeddings)
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(chunk=self.chunks[int(i)], score=float(scores[int(i)]))
            for i in top_indices
        ]


def retrieve_for_query(
    store: VectorStore,
    backend: EmbeddingBackend,
    query: str,
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    query_vec = backend.embed([query])[0]
    return store.search(query_vec, top_k=top_k)