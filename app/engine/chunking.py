"""Split document text into overlapping retrieval chunks."""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import DocumentRecord


@dataclass(frozen=True)
class TextChunk:
    """A retrieval unit with provenance."""

    chunk_id: str
    source_path: str
    text: str
    start_char: int
    end_char: int


def chunk_documents(
    documents: list[DocumentRecord],
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[TextChunk]:
    """Chunk documents with character windows and overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: list[TextChunk] = []
    for doc_index, document in enumerate(documents):
        text = document.text
        start = 0
        part = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Prefer breaking on whitespace near the window end
            if end < len(text):
                soft = text.rfind(" ", start + chunk_size // 2, end)
                if soft > start:
                    end = soft
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    TextChunk(
                        chunk_id=f"doc{doc_index}-c{part}",
                        source_path=document.source_path,
                        text=piece,
                        start_char=start,
                        end_char=end,
                    )
                )
                part += 1
            if end >= len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks