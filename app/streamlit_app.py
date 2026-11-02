"""Streamlit UI for the document reader engine."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Allow `streamlit run app/streamlit_app.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.engine.pipeline import DocumentReaderPipeline, PipelineConfig  # noqa: E402
from app.engine.extractor import parse_prompt_set  # noqa: E402

DEFAULT_PROMPTS = """# One field per line: name | type | instruction
# Or: name: instruction
vendor_name | string | Extract the vendor or supplier company name
invoice_number | string | Extract the invoice or reference number
invoice_date | date | Extract the invoice date
total_amount | currency | Extract the total amount due
due_date | date | Extract the payment due date
"""


def _init_state() -> None:
    defaults = {
        "result_json": None,
        "error": None,
        "running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="Document Reader Engine",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
          :root {
            --ink: #0b1c2c;
            --muted: #4a6072;
            --line: #b9c7d4;
            --panel: rgba(255,255,255,0.72);
          }
          html, body, [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
            color: var(--ink);
          }
          .stApp {
            background:
              radial-gradient(900px 420px at 0% 0%, rgba(34, 163, 184, 0.22), transparent 55%),
              radial-gradient(800px 380px at 100% 10%, rgba(26, 86, 168, 0.18), transparent 50%),
              linear-gradient(165deg, #e8eef5 0%, #d5e2ec 48%, #c9d8e6 100%);
            background-attachment: fixed;
          }
          h1, h2, h3 {
            font-family: Syne, sans-serif !important;
            letter-spacing: -0.03em;
          }
          div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1c2c 0%, #12344a 100%);
          }
          div[data-testid="stSidebar"] * {
            color: #e8f1f7 !important;
          }
          .hero {
            padding: 0.25rem 0 1.1rem 0;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.15rem;
          }
          .hero .brand {
            font-family: Syne, sans-serif;
            font-size: clamp(2rem, 4vw, 2.75rem);
            font-weight: 800;
            margin: 0;
            color: var(--ink);
            letter-spacing: -0.04em;
          }
          .hero .tag {
            margin: 0.4rem 0 0 0;
            font-size: 1.05rem;
            color: var(--muted);
            max-width: 46rem;
            line-height: 1.45;
          }
          .metric-strip {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 0.35rem 0 1rem 0;
          }
          .metric-pill {
            background: var(--panel);
            border: 1px solid var(--line);
            padding: 0.55rem 0.85rem;
            border-radius: 0.35rem;
            min-width: 6.5rem;
            backdrop-filter: blur(6px);
          }
          .metric-pill .label {
            font-size: 0.72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
          }
          .metric-pill .value {
            font-family: Syne, sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <p class="brand">Document Reader</p>
          <p class="tag">Upload a zip of source documents, define extraction prompts, then retrieve and extract structured JSON with embeddings + an LLM.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Parameters")
        mode = st.selectbox(
            "Engine mode",
            options=["auto", "openai", "local"],
            index=0,
            help="auto uses OpenAI when OPENAI_API_KEY is set; otherwise local offline heuristics.",
        )
        api_key = st.text_input(
            "OpenAI API key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Optional if already set in the environment / .env",
        )
        chunk_size = st.slider("Chunk size", 300, 2000, 900, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
        top_k = st.slider("Top-K chunks per field", 1, 12, 5)
        st.caption("Supported files inside the zip: .txt .md .csv .pdf .docx")

    col_left, col_right = st.columns([1.05, 1], gap="large")

    with col_left:
        st.subheader("Document source set")
        upload = st.file_uploader(
            "Upload a .zip of documents",
            type=["zip"],
            accept_multiple_files=False,
        )

        st.subheader("Extraction prompt set")
        prompt_text = st.text_area(
            "Fields to extract",
            value=DEFAULT_PROMPTS,
            height=260,
            help="Use `name | type | instruction` lines, or a JSON array of objects.",
        )

        preview_fields = []
        parse_error = None
        try:
            preview_fields = parse_prompt_set(prompt_text)
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)

        if parse_error:
            st.error(f"Prompt set parse error: {parse_error}")
        else:
            st.caption(f"{len(preview_fields)} field(s) ready")
            if preview_fields:
                st.dataframe(
                    [
                        {
                            "name": f.name,
                            "type": f.type_hint,
                            "prompt": f.prompt,
                        }
                        for f in preview_fields
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        run = st.button(
            "Run extraction",
            type="primary",
            use_container_width=True,
            disabled=upload is None or not preview_fields,
        )

    with col_right:
        st.subheader("JSON output")
        if run:
            if upload is None:
                st.warning("Upload a zip archive first.")
            elif not preview_fields:
                st.warning("Add at least one extraction field.")
            else:
                with st.spinner("Ingesting, embedding, retrieving, extracting…"):
                    try:
                        config = PipelineConfig(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k,
                            mode=mode,
                            api_key=api_key or None,
                        )
                        pipeline = DocumentReaderPipeline(config)
                        result = pipeline.run(upload.getvalue(), prompt_text)
                        st.session_state.result_json = result.to_json()
                        st.session_state.error = None
                        st.session_state._result_obj = result
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.error = str(exc)
                        st.session_state.result_json = None

        if st.session_state.error:
            st.error(st.session_state.error)

        result_obj = st.session_state.get("_result_obj")
        if result_obj is not None and st.session_state.result_json:
            st.markdown(
                f"""
                <div class="metric-strip">
                  <div class="metric-pill"><div class="label">Mode</div><div class="value">{result_obj.mode}</div></div>
                  <div class="metric-pill"><div class="label">Documents</div><div class="value">{len(result_obj.documents)}</div></div>
                  <div class="metric-pill"><div class="label">Chunks</div><div class="value">{result_obj.chunk_count}</div></div>
                  <div class="metric-pill"><div class="label">Fields</div><div class="value">{len(result_obj.fields)}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tabs = st.tabs(["Extraction", "Full JSON", "Evidence", "Documents"])
            with tabs[0]:
                st.json(result_obj.extraction)
            with tabs[1]:
                st.code(st.session_state.result_json, language="json")
                st.download_button(
                    "Download JSON",
                    data=st.session_state.result_json,
                    file_name="extraction_result.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with tabs[2]:
                st.json(result_obj.evidence)
            with tabs[3]:
                st.json(result_obj.documents)
        else:
            st.info(
                "Upload a document zip, refine the prompt set, then run extraction. "
                "Results appear here as structured JSON."
            )


if __name__ == "__main__":
    main()