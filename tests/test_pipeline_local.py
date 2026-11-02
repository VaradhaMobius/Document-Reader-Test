"""Smoke tests for offline local-mode pipeline."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from app.engine.extractor import parse_prompt_set
from app.engine.pipeline import DocumentReaderPipeline, PipelineConfig

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _zip_samples() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in SAMPLES.glob("*"):
            if path.is_file():
                archive.writestr(path.name, path.read_bytes())
    return buffer.getvalue()


def test_parse_prompt_set_pipe_format() -> None:
    fields = parse_prompt_set(
        "vendor_name | string | Extract the vendor\n"
        "total_amount | currency | Extract total\n"
    )
    assert len(fields) == 2
    assert fields[0].name == "vendor_name"
    assert fields[1].type_hint == "currency"


def test_local_pipeline_extracts_json() -> None:
    prompts = """
    vendor_name | string | Extract the vendor or supplier company name
    invoice_number | string | Extract the invoice number
    total_amount | currency | Extract the total amount due
    due_date | date | Extract the payment due date
    """
    pipeline = DocumentReaderPipeline(PipelineConfig(mode="local", top_k=4))
    result = pipeline.run(_zip_samples(), prompts)
    payload = json.loads(result.to_json())

    assert payload["mode"] == "local"
    assert payload["chunk_count"] > 0
    assert len(payload["documents"]) >= 2
    extraction = payload["extraction"]
    assert extraction["invoice_number"] is not None
    assert "INV-2024-88421" in str(extraction["invoice_number"])
    assert extraction["total_amount"] is not None
    assert extraction["due_date"] is not None