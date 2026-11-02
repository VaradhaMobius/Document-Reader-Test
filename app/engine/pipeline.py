"""End-to-end document reader pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, BinaryIO

from .chunking import TextChunk, chunk_documents
from .embeddings import EmbeddingBackend, build_embedding_backend
from .extractor import (
    ExtractionField,
    ExtractionResult,
    Extractor,
    build_extractor,
    parse_prompt_set,
)
from .ingest import DocumentRecord, ingest_zip
from .retriever import RetrievedChunk, VectorStore, retrieve_for_query


@dataclass
class PipelineConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    mode: str = "auto"  # auto | openai | local
    api_key: str | None = None


@dataclass
class PipelineResult:
    documents: list[dict[str, Any]]
    chunk_count: int
    fields: list[dict[str, str]]
    extraction: dict[str, Any]
    evidence: dict[str, list[dict[str, Any]]]
    mode: str
    retrieved: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_json(self, *, indent: int = 2) -> str:
        payload = {
            "mode": self.mode,
            "documents": self.documents,
            "chunk_count": self.chunk_count,
            "fields": self.fields,
            "extraction": self.extraction,
            "evidence": self.evidence,
            "retrieved": self.retrieved,
        }
        return json.dumps(payload, indent=indent, ensure_ascii=False)


class DocumentReaderPipeline:
    """
    Upload zip → parse docs → chunk → embed → retrieve per field → extract JSON.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.embedding_backend = embedding_backend or build_embedding_backend(
            self.config.mode, api_key=self.config.api_key
        )
        self.extractor = extractor or build_extractor(
            self.config.mode, api_key=self.config.api_key
        )

    def run(
        self,
        zip_file: BinaryIO | bytes,
        prompt_set: str | list[ExtractionField],
    ) -> PipelineResult:
        documents = ingest_zip(zip_file)
        if not documents:
            raise ValueError(
                "No supported documents found in zip. "
                "Supported: .txt, .md, .csv, .pdf, .docx"
            )

        fields = (
            prompt_set
            if isinstance(prompt_set, list)
            else parse_prompt_set(prompt_set)
        )
        if not fields:
            raise ValueError(
                "No extraction fields found. Provide lines like "
                "`invoice_total | number | Extract the invoice total amount`"
            )

        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        store = VectorStore.from_chunks(chunks, self.embedding_backend)

        retrieved_by_field: dict[str, list[RetrievedChunk]] = {}
        for field_spec in fields:
            query = f"{field_spec.name}: {field_spec.prompt}"
            retrieved_by_field[field_spec.name] = retrieve_for_query(
                store,
                self.embedding_backend,
                query,
                top_k=self.config.top_k,
            )

        extraction: ExtractionResult = self.extractor.extract(
            fields, retrieved_by_field
        )

        return PipelineResult(
            documents=[
                {
                    "source_path": doc.source_path,
                    "media_type": doc.media_type,
                    "chars": len(doc.text),
                }
                for doc in documents
            ],
            chunk_count=len(chunks),
            fields=[asdict(f) for f in fields],
            extraction=extraction.values,
            evidence=extraction.evidence,
            mode=extraction.mode,
            retrieved={
                name: [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "source_path": hit.chunk.source_path,
                        "score": round(hit.score, 4),
                        "text": hit.chunk.text,
                    }
                    for hit in hits
                ]
                for name, hits in retrieved_by_field.items()
            },
        )


# Re-export helpers used by the UI / tests
__all__ = [
    "DocumentReaderPipeline",
    "PipelineConfig",
    "PipelineResult",
    "parse_prompt_set",
    "DocumentRecord",
    "TextChunk",
]