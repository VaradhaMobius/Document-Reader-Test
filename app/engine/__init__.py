"""Document reader engine: ingest, embed, retrieve, extract."""

from .pipeline import DocumentReaderPipeline, PipelineResult

__all__ = ["DocumentReaderPipeline", "PipelineResult"]