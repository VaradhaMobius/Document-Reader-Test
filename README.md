# Document Reader Engine

Upload a **zip of source documents**, define an **extraction prompt set**, then run a retrieval-augmented pipeline that:

1. Ingests `.txt`, `.md`, `.csv`, `.pdf`, and `.docx` files from the zip  
2. Chunks text and builds embeddings  
3. Retrieves the most relevant chunks per field  
4. Extracts values with an LLM (or offline heuristics)  
5. Returns structured **JSON** in a Streamlit UI  

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: enable OpenAI embeddings + chat extraction
cp .env.example .env
# edit .env and set OPENAI_API_KEY

streamlit run app/streamlit_app.py
```

Open the app, upload `samples/documents.zip` (or your own zip), adjust the prompt set, and click **Run extraction**.

## Prompt set format

One field per line:

```text
vendor_name | string | Extract the vendor or supplier company name
invoice_number | string | Extract the invoice or reference number
total_amount | currency | Extract the total amount due
```

Or JSON:

```json
[
  {"name": "vendor_name", "type": "string", "prompt": "Extract the vendor name"},
  {"name": "total_amount", "type": "currency", "prompt": "Extract the total due"}
]
```

## Engine modes

| Mode | Behavior |
|------|----------|
| `auto` | Uses OpenAI when `OPENAI_API_KEY` is set; otherwise local |
| `openai` | OpenAI embeddings (`text-embedding-3-small`) + chat JSON extraction |
| `local` | Hash embeddings + heuristic extractor (no API key) |

## Project layout

```text
app/
  streamlit_app.py      # Streamlit UI
  engine/
    ingest.py           # Zip → document text
    chunking.py         # Overlapping chunks
    embeddings.py       # OpenAI / local vectors
    retriever.py        # Cosine top-k search
    extractor.py        # Prompt parsing + LLM / local extract
    pipeline.py         # Orchestration
samples/                # Example documents + zip
tests/                  # Offline smoke tests
```

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Sample zip

```bash
cd samples && zip -r documents.zip invoice_acme.txt contract_notes.md && cd ..
```
