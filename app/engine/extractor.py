"""LLM-backed and offline value extraction into JSON."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .retriever import RetrievedChunk


@dataclass(frozen=True)
class ExtractionField:
    """One field the user wants extracted."""

    name: str
    prompt: str
    type_hint: str = "string"


@dataclass(frozen=True)
class ExtractionResult:
    values: dict[str, Any]
    evidence: dict[str, list[dict[str, Any]]]
    mode: str


class Extractor(ABC):
    @abstractmethod
    def extract(
        self,
        fields: list[ExtractionField],
        retrieved_by_field: dict[str, list[RetrievedChunk]],
    ) -> ExtractionResult:
        raise NotImplementedError


class LocalHeuristicExtractor(Extractor):
    """
    Offline extractor: pulls likely values from retrieved text using
    simple patterns keyed off the field name / prompt keywords.
    Useful for demos without an API key.
    """

    def extract(
        self,
        fields: list[ExtractionField],
        retrieved_by_field: dict[str, list[RetrievedChunk]],
    ) -> ExtractionResult:
        values: dict[str, Any] = {}
        evidence: dict[str, list[dict[str, Any]]] = {}

        for field in fields:
            hits = retrieved_by_field.get(field.name, [])
            evidence[field.name] = [
                {
                    "source_path": hit.chunk.source_path,
                    "score": round(hit.score, 4),
                    "excerpt": hit.chunk.text[:400],
                }
                for hit in hits
            ]
            corpus = "\n".join(hit.chunk.text for hit in hits)
            values[field.name] = _heuristic_value(field, corpus)

        return ExtractionResult(values=values, evidence=evidence, mode="local")


class OpenAIExtractor(Extractor):
    """Ask a chat model to fill a JSON object from retrieved context."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI extraction")
        self.client = OpenAI(api_key=key)
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def extract(
        self,
        fields: list[ExtractionField],
        retrieved_by_field: dict[str, list[RetrievedChunk]],
    ) -> ExtractionResult:
        evidence: dict[str, list[dict[str, Any]]] = {}
        context_blocks: list[str] = []

        for field in fields:
            hits = retrieved_by_field.get(field.name, [])
            evidence[field.name] = [
                {
                    "source_path": hit.chunk.source_path,
                    "score": round(hit.score, 4),
                    "excerpt": hit.chunk.text[:400],
                }
                for hit in hits
            ]
            snippets = []
            for hit in hits:
                snippets.append(
                    f"[{hit.chunk.source_path} | score={hit.score:.3f}]\n{hit.chunk.text}"
                )
            context_blocks.append(
                f"### Field `{field.name}` ({field.type_hint})\n"
                f"Instruction: {field.prompt}\n"
                f"Context:\n" + ("\n---\n".join(snippets) if snippets else "(no context)")
            )

        schema_hint = {
            field.name: f"<{field.type_hint}> — {field.prompt}" for field in fields
        }
        system = (
            "You are a precise document information extraction engine. "
            "Use only the provided context. If a value is missing, return null. "
            "Respond with a single JSON object whose keys match the requested fields. "
            "Do not include markdown fences or commentary."
        )
        user = (
            "Extract the following fields into JSON:\n"
            f"{json.dumps(schema_hint, indent=2)}\n\n"
            "Retrieved context by field:\n\n"
            + "\n\n".join(context_blocks)
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"_raw": raw, "_error": "Model returned non-JSON content"}

        # Keep only requested keys (plus any parse error markers)
        values: dict[str, Any] = {}
        for field in fields:
            values[field.name] = parsed.get(field.name)
        for extra in ("_raw", "_error"):
            if extra in parsed:
                values[extra] = parsed[extra]

        return ExtractionResult(values=values, evidence=evidence, mode="openai")


def build_extractor(
    mode: str = "auto",
    *,
    api_key: str | None = None,
) -> Extractor:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if mode == "local":
        return LocalHeuristicExtractor()
    if mode == "openai":
        return OpenAIExtractor(api_key=key)
    if key:
        return OpenAIExtractor(api_key=key)
    return LocalHeuristicExtractor()


def parse_prompt_set(text: str) -> list[ExtractionField]:
    """
    Parse a multiline prompt set.

    Formats accepted (one field per line / block):
      field_name: extraction instruction
      field_name | type | extraction instruction
      JSON array: [{"name": "...", "prompt": "...", "type": "string"}]
    """
    stripped = text.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        data = json.loads(stripped)
        fields: list[ExtractionField] = []
        for item in data:
            fields.append(
                ExtractionField(
                    name=str(item["name"]).strip(),
                    prompt=str(item.get("prompt") or item.get("instruction") or "").strip(),
                    type_hint=str(item.get("type") or item.get("type_hint") or "string").strip(),
                )
            )
        return [f for f in fields if f.name and f.prompt]

    fields = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            parts = [p.strip() for p in line.split("|", 2)]
            if len(parts) == 3:
                name, type_hint, prompt = parts
            elif len(parts) == 2:
                name, prompt = parts
                type_hint = "string"
            else:
                continue
        elif ":" in line:
            name, prompt = line.split(":", 1)
            name, prompt, type_hint = name.strip(), prompt.strip(), "string"
        else:
            continue
        if name and prompt:
            fields.append(
                ExtractionField(name=name, prompt=prompt, type_hint=type_hint or "string")
            )
    return fields


def _heuristic_value(field: ExtractionField, corpus: str) -> Any:
    if not corpus.strip():
        return None

    type_hint = field.type_hint.lower()
    lowered = corpus.lower()
    keywords = _keywords(field.name + " " + field.prompt)

    # Prefer lines that mention a keyword
    lines = [ln.strip() for ln in corpus.splitlines() if ln.strip()]
    ranked = sorted(
        lines,
        key=lambda ln: sum(1 for kw in keywords if kw in ln.lower()),
        reverse=True,
    )
    focus = "\n".join(ranked[:8]) if ranked else corpus

    if type_hint in {"number", "float", "int", "integer", "currency", "amount"}:
        money = re.findall(r"\$?\s?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?", focus)
        if money:
            raw = money[0].replace(",", "").replace("$", "").strip()
            try:
                return float(raw) if "." in raw or type_hint in {"float", "number", "currency", "amount"} else int(raw)
            except ValueError:
                return raw

    if type_hint in {"date", "datetime"}:
        date = re.search(
            r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
            focus,
            flags=re.I,
        )
        if date:
            return date.group(0)

    if type_hint in {"email"}:
        email = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", focus, flags=re.I)
        if email:
            return email.group(0)

    # Key: value patterns
    for kw in keywords:
        pattern = re.compile(
            rf"(?im)\b{re.escape(kw)}\b\s*[:\-–]\s*(.+)$"
        )
        match = pattern.search(focus)
        if match:
            return match.group(1).strip().strip(".,;")

    # Fallback: first short sentence mentioning a keyword
    for line in ranked:
        if any(kw in line.lower() for kw in keywords):
            return line[:240]
    return ranked[0][:240] if ranked else None


def _keywords(text: str) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "find",
        "extract",
        "get",
        "value",
        "from",
        "document",
        "what",
        "is",
        "are",
        "please",
    }
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if t not in stop and len(t) > 1]