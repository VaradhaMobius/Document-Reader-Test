"""Ingest documents from a zip archive into plain text records."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx"}


@dataclass(frozen=True)
class DocumentRecord:
    """A single ingested document."""

    source_path: str
    text: str
    media_type: str


def _read_txt(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def _read_docx(data: bytes) -> str:
    document = Document(io.BytesIO(data))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _read_csv(data: bytes) -> str:
    text = _read_txt(data)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    lines = [" | ".join(header), "-" * max(len(" | ".join(header)), 8)]
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append(" | ".join(padded[: len(header)]))
    return "\n".join(lines)


def _extract_text(name: str, data: bytes) -> DocumentRecord | None:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return None
    if suffix in {".txt", ".md"}:
        text = _read_txt(data)
        media = "text/plain" if suffix == ".txt" else "text/markdown"
    elif suffix == ".csv":
        text = _read_csv(data)
        media = "text/csv"
    elif suffix == ".pdf":
        text = _read_pdf(data)
        media = "application/pdf"
    elif suffix == ".docx":
        text = _read_docx(data)
        media = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    else:
        return None

    text = text.strip()
    if not text:
        return None
    return DocumentRecord(source_path=name, text=text, media_type=media)


def ingest_zip(file_obj: BinaryIO | bytes) -> list[DocumentRecord]:
    """Extract supported documents from a zip file-like object or bytes."""
    if isinstance(file_obj, (bytes, bytearray)):
        file_obj = io.BytesIO(file_obj)

    records: list[DocumentRecord] = []
    with zipfile.ZipFile(file_obj) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            # Skip macOS / hidden junk
            name = info.filename.replace("\\", "/")
            base = Path(name).name
            if base.startswith(".") or name.startswith("__MACOSX/"):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            data = archive.read(info)
            record = _extract_text(name, data)
            if record:
                records.append(record)
    return records


def ingest_paths(paths: Iterable[Path]) -> list[DocumentRecord]:
    """Ingest loose files (useful for tests / samples)."""
    records: list[DocumentRecord] = []
    for path in paths:
        if not path.is_file():
            continue
        data = path.read_bytes()
        record = _extract_text(str(path.name), data)
        if record:
            records.append(record)
    return records