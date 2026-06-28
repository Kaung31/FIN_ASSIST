# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this fully before writing code.

## What we're building

**FinAssist** — a multimodal financial-report assistant. A user uploads earnings PDFs
(narrative text, financial-statement tables, and charts/figures). The system:

1. **Parses** the document with **Azure AI Document Intelligence** (`prebuilt-layout`) → markdown text + tables + figure regions.
2. **Enriches** charts/figures by describing them with a vision-capable model (Azure OpenAI).
3. **Chunks** content (section/table/figure), **embeds** with **Azure OpenAI** (`text-embedding-3-small`, 1536-d), and indexes it in **Azure AI Search** (hybrid BM25 + vector + semantic ranker).
4. Produces three things on top of that index:
   - **Structured summaries** (revenue, net income, EPS, guidance, segment performance, YoY deltas) as typed JSON.
   - **Anomaly detection** (period-over-period numeric outliers, with grounded explanations).
   - **Grounded Q&A** (RAG chat with inline citations back to the source pages).

This is a **document-analysis** tool, not an investment-advice tool. See "Non-negotiable principles".

## Mock mode (default, mandatory)

`MOCK_MODE=true` (the default) mocks **all three Azure services** — Document Intelligence, Azure
OpenAI, and Azure AI Search — with canned data, so the pipeline and tests run at zero cost.
Unlike a local vector store, AI Search + Document Intelligence have **no free local equivalent**,
so real retrieval can only be tested against the cloud. Canned outputs live in
`src/finassist/mocks/fixtures.py`. `tests/conftest.py` forces mock mode, so `pytest` is hermetic
even when `.env` has `MOCK_MODE=false` for a live run.

## Tech stack (decisions are made — do not substitute)

| Concern | Choice | Notes |
|---|---|---|
| Parsing | `azure-ai-documentintelligence` | `prebuilt-layout`, markdown output. NOT `azure-ai-formrecognizer` (legacy). Mocked in mock mode. |
| Embeddings | Azure OpenAI `text-embedding-3-small` (1536-d) | Via the LLM client. Dim is config-driven (`EMBEDDING_DIMENSIONS`) and MUST equal the index vector dim. |
| Vector store + retrieval | `azure-search-documents` | Hybrid (BM25 + vector, RRF) + **semantic ranker** (replaces a separate reranker). Standard tier+. Mocked in mock mode. |
| LLM / VLM | `openai` (`AzureOpenAI`, v1 surface) | Chat + vision. Mocked in mock mode. Read deployment names from config; pass them as `model`. |
| Numbers | deterministic Python | `analysis/math_guardrail.py` — never the LLM. |
| Eval / tracing | `ragas`, `arize-phoenix` (`.[eval]`) | Lazy-imported. Phoenix works in the main venv; Ragas may need its own venv (heavy langchain tree). |
| API / Config / Schemas / UI / Tests | fastapi+uvicorn / pydantic-settings / pydantic v2 / streamlit / pytest | All secrets via `config/settings.py`; mock Azure in unit tests. |

> **Azure OpenAI client rule (prevents a doubled-URL 404):** build with `azure_endpoint=` ONLY —
> never set `base_url`, and never append `/openai`, `/openai/v1`, or `/deployments` to the endpoint
> (the SDK adds the path). `AZURE_OPENAI_ENDPOINT` ends at `.openai.azure.com/`. Deployment names
> must match the Azure deployments exactly.

## Repository map

```
config/            settings + prompt templates (config/prompts/*.txt)
src/finassist/
  mocks/           canned Document Intelligence / AI Search / Azure OpenAI outputs for MOCK_MODE
  ingestion/       parser.py (Document Intelligence + mock), table_extractor.py, figure_extractor.py
  multimodal/      chart_describer.py (vision description of figures)
  chunking/        chunker.py (flat section/table/figure chunks)
  embeddings/      embedder.py (Azure OpenAI embeddings, batched)
  search/          clients.py, index_schema.py (vector + semantic config), indexer.py (write), retriever.py (hybrid + semantic + filter)
  analysis/        models.py, math_guardrail.py (deterministic numbers), summarizer.py, anomaly_detector.py
  qa/              rag_chat.py (grounded answers + citations)
  llm/             client.py (Azure OpenAI wrapper + MOCK_MODE short-circuit; azure_endpoint only)
  eval/            golden_set.jsonl, run_ragas.py, phoenix_trace.py (requires .[eval])
  pipeline.py      end-to-end ingest (shared by API route + CLI script)
  api/             FastAPI app + deps (shared LLM) + routes (ingest, summarize, anomalies, chat)
ui/                Streamlit demo (thin; calls the API)
scripts/           create_index.py, ingest_local.py
```

## Data flow (the contract between modules)

```
PDF bytes
  -> ingestion.parser.parse()                  -> ParsedDocument {raw_text, tables, figures}
  -> ingestion.figure_extractor.crop_figures() -> figures gain image_path (no-op in mock mode)
  -> multimodal.chart_describer.describe()      -> figures gain text descriptions
  -> chunking.chunker.chunk()                   -> list[Chunk] (table / figure / narrative)
  -> search.indexer.upload()                    -> embeds (Azure OpenAI) + uploads to AI Search
query
  -> search.retriever.search()   -> hybrid (BM25 + vector) + semantic ranker + OData filter
  -> analysis.summarizer / analysis.anomaly_detector / qa.rag_chat
  -> typed pydantic result (+ citations)
```

`pipeline.ingest(source, name, llm)` runs the ingest half; the API + `ingest_local.py` both call
it. Keep these boundaries stable; if you change a signature, update callers and tests.

## Non-negotiable principles

1. **Never invent numbers.** Metric values and anomaly magnitudes are computed in Python
   (`analysis/math_guardrail.py`) from parsed table cells. The LLM only selects, organizes, and
   explains. If a number isn't in the source, the field is `null`.
2. **Always cite.** Summaries, anomalies, and answers carry `Citation` objects (chunk_id, doc id,
   page, section). No citation → the claim shouldn't be made.
3. **Not investment advice.** Outputs describe what the filing says; the system prompts enforce
   this — don't remove that framing.
4. **Ground before you generate.** Retrieve first, prompt with retrieved context only. If retrieval
   returns nothing, decline rather than answer from general knowledge.
5. **Secrets only via settings.** No keys in code, logs, or commits. `.env` is gitignored.

## How to run (local dev)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                         # MOCK_MODE=true by default
python scripts/ingest_local.py data/samples/sample.pdf   # zero cost, no keys
pytest -q                                    # hermetic mock mode, no network
```

See `AZURE_SETUP.md` to provision the three Azure resources and flip `MOCK_MODE=false`.

## Conventions

- **Python 3.11+**, full type hints, `ruff` for lint/format, Google-style docstrings.
- Keep cloud client construction in `llm/` and `search/`. CLI scripts may `print`; modules use `logging`.
- Tests mock all Azure services and run hermetically (mock mode forced in `conftest.py`) — no
  network in `pytest`. Real-cloud tests go behind `@pytest.mark.integration` (`pytest --run-integration`).
- The repo-root `config` package is importable in tests via `pythonpath = ["src", "."]`; scripts add
  the repo root to `sys.path` so `python scripts/x.py` works.
