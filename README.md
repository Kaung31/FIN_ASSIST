# FinAssist

Upload an earnings PDF and ask it questions — get structured financial summaries, YoY anomaly flags, and cited Q&A answers, with every number pulled straight from the source tables instead of guessed by an LLM.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-TODO-lightgrey)

<!-- TODO: record a short GIF/screenshot of the Streamlit demo — upload a 10-Q, then the Summary tab and one Q&A answer with its citation showing. -->

## Overview

Reading a 40-page 10-Q to answer "what was Services revenue this quarter, and is that normal?" is slow, and handing the whole PDF to an LLM risks a confidently wrong number. FinAssist parses the filing with Azure Document Intelligence, indexes the text/tables/charts in Azure AI Search, and answers three kinds of requests against that index: a structured summary, a period-over-period anomaly report, and grounded Q&A with citations. Every dollar figure is computed in Python from the parsed tables — the LLM only writes the surrounding prose. Built for analysts and finance folks who need facts out of a filing, not investment advice (it explicitly won't give you any).

## Features

- Parses earnings PDFs into text, financial tables, and chart regions using Azure Document Intelligence (`prebuilt-layout`)
- Describes charts and figures with a vision model so they're searchable and citable, not just decorative images
- Hybrid search (keyword + vector, fused with RRF) re-ranked by Azure AI Search's semantic ranker, scoped by company and fiscal period
- Structured earnings summaries (revenue, net income, EPS, segments, guidance) where every number traces back to a source table cell
- Deterministic anomaly detection on YoY swings — flags what changed, then asks the model to explain why, never to invent the number
- Grounded Q&A chat that cites the exact page/section it answered from, and refuses instead of guessing when retrieval comes up empty or low-confidence
- Runs entirely in a mock mode with canned data — no Azure keys, no cost, full pipeline and test suite still exercised

## Tech Stack

| Technology | Used for |
|---|---|
| Azure AI Document Intelligence | PDF → markdown text, tables, figure regions |
| Azure OpenAI (chat + vision + embeddings) | Summaries, chart descriptions, Q&A, `text-embedding-3-small` vectors |
| Azure AI Search | Hybrid BM25 + vector index with semantic re-ranking |
| FastAPI + Uvicorn | REST API (`/ingest`, `/summarize`, `/anomalies`, `/chat`) |
| Streamlit | Thin demo UI that calls the API |
| Pydantic / pydantic-settings | Typed data models and `.env`-based config |
| PyMuPDF | Crops chart/figure bounding boxes out of the source PDF |
| Tenacity | Retries transient Azure OpenAI errors with backoff |
| Ragas + Arize Phoenix | RAG evaluation and local tracing (optional extras) |

## Getting Started

### Prerequisites
- Python 3.11+ (developed on 3.12)
- No Azure account needed to run in mock mode — the default

### Installation
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

### Environment variables

All variables live in [`.env.example`](.env.example). In mock mode (the default) you can leave everything but `MOCK_MODE` blank.

| Variable | Purpose | Where to get it |
|---|---|---|
| `MOCK_MODE` | `true` mocks all three Azure services with canned data; `false` calls real Azure | — (default `true`) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource URL, must end in `.openai.azure.com/` | Azure Portal → your Azure OpenAI resource |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key | Azure Portal → your Azure OpenAI resource → Keys and Endpoint |
| `AZURE_OPENAI_API_VERSION` | API version pinned in code (`2024-10-21`) | Leave as-is unless you know why to change it |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Deployment name of a vision-capable chat model | Azure AI Foundry → Deployments (you create this) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment name of the embedding model | Azure AI Foundry → Deployments (you create this) |
| `EMBEDDING_DIMENSIONS` | Vector dimension, must match the deployment and the search index | `1536` for `text-embedding-3-small`, `3072` for `3-large` |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search service URL | Azure Portal → your Search resource |
| `AZURE_SEARCH_KEY` | Azure AI Search admin key | Azure Portal → your Search resource → Keys |
| `AZURE_SEARCH_INDEX_NAME` | Index name to create/query | Your choice, default `earnings-index` |
| `AZURE_SEARCH_USE_SEMANTIC_RANKER` | Enables the semantic ranker | Needs Search **Standard tier or above** |
| `AZURE_DOCINTEL_ENDPOINT` | Document Intelligence resource URL | Azure Portal → your Document Intelligence resource |
| `AZURE_DOCINTEL_KEY` | Document Intelligence API key | Azure Portal → your Document Intelligence resource → Keys |
| `ENABLE_PHOENIX` | Turns on local Arize Phoenix tracing | — |
| `USE_AAD_AUTH` | Use Managed Identity / `az login` instead of the keys above | — |
| `LOG_LEVEL` | Python logging level | — |

Provisioning the three Azure resources for live mode is covered in [`AZURE_SETUP.md`](AZURE_SETUP.md).

### Run it
```bash
python scripts/ingest_local.py data/samples/sample.pdf   # full pipeline, mock mode, zero cost
```
For the API + UI:
```bash
uvicorn finassist.api.main:app --reload    # http://localhost:8000/docs
streamlit run ui/streamlit_app.py          # http://localhost:8501
```
Going live with real Azure resources instead of mocks:
```bash
python scripts/create_index.py             # one-time: create the AI Search index
python scripts/ingest_local.py data/samples/your.pdf
```
Azure AI Search runs about $75/month while the service exists — delete it between sessions and recreate the index with `scripts/create_index.py`.

## Usage

**CLI — ingest a PDF and ask one demo question** (any PDF works in mock mode — it's ignored in favor of canned data, see [`data/samples/README.md`](data/samples/README.md)):
```bash
python scripts/ingest_local.py data/samples/sample.pdf
```
```
Ingested:
  doc_id: mock-acme-fy2025-q2
  company: Acme Corp
  fiscal_period: FY2025 Q2
  pages: 4
  tables: 2
  figures: 1
  chunks: 4
  indexed: 4

Demo question: What was total revenue this quarter and how did it change year over year?
  grounded: True
  answer:   According to the filing, Acme Corp reported total revenue of $1,250 million for
            Q2 FY2025, up 25% from $1,000 million in the prior-year quarter, driven primarily
            by the Cloud segment.
  citations: [('Condensed Consolidated Statements of Operations', 'p2'), ('Revenue by Segment', 'p3'), ...]
```

**API — grounded Q&A:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What was total revenue this quarter?", "company": "Acme Corp"}'
```
```json
{
  "answer": "According to the filing, Acme Corp reported total revenue of $1,250 million for Q2 FY2025, up 25% from $1,000 million in the prior-year quarter, driven primarily by the Cloud segment.",
  "citations": [
    {"chunk_id": "...", "page": 2, "section": "Condensed Consolidated Statements of Operations"},
    {"chunk_id": "...", "page": 3, "section": "Revenue by Segment"}
  ],
  "grounded": true
}
```

**API — structured summary:**
```bash
curl "http://localhost:8000/summarize?company=Acme%20Corp&fiscal_period=FY2025%20Q2"
```
Returns a JSON `EarningsSummary` with `key_metrics` (each a number + prior value + YoY % + citation), `segments`, and `guidance` — see [`analysis/models.py`](src/finassist/analysis/models.py).

## Project Structure

```
config/            settings.py (env config) + prompts/ (LLM system prompts)
src/finassist/
  mocks/            canned Document Intelligence / AI Search / Azure OpenAI outputs for MOCK_MODE
  ingestion/        PDF parsing, table extraction, figure cropping
  multimodal/       vision-model chart descriptions
  chunking/         turns a parsed document into retrieval chunks
  embeddings/       Azure OpenAI embedding calls
  search/           AI Search index schema, indexer (write), retriever (read)
  analysis/         data models, deterministic math guardrail, summarizer, anomaly detector
  qa/               grounded Q&A / RAG chat
  llm/              Azure OpenAI client wrapper (chat + vision + embeddings)
  eval/             golden Q&A set, Ragas runner, Phoenix tracing
  api/              FastAPI app, routes, shared dependencies
  pipeline.py       end-to-end ingest, shared by the API and the CLI script
ui/                 Streamlit demo (calls the API only, no logic of its own)
scripts/            create_index.py, ingest_local.py
tests/              one file per module, hermetic (mock mode forced)
```

## How It Works

Ingest: a PDF goes through Document Intelligence to get text + tables + figure regions, figures get cropped and described by a vision model, everything becomes flat chunks (tables and described figures stay whole, narrative text is windowed), each chunk gets embedded, and the batch is uploaded to Azure AI Search. Query: a request runs a hybrid (keyword + vector) search against that index, re-ranked by Azure's semantic ranker and filtered by company/period. From there, three paths diverge — `summarizer` and `anomaly_detector` hand the retrieved tables to Python for arithmetic and only ask the LLM for prose framing, while `rag_chat` asks the LLM to answer directly from retrieved passages. If retrieval comes back empty or below a relevance threshold, the system declines instead of answering from the model's general knowledge.

## Roadmap / Known Limitations

- No authentication or rate limiting on the API — anyone who can reach the port can call every route
- No document deletion or versioning in the index — re-ingesting the same PDF adds duplicate chunks rather than replacing them
- No multi-turn conversation memory — every `/chat` call is a single independent turn
- Anomaly detection is a flat ±20%/50% YoY threshold, not seasonality- or variance-aware
- No Dockerfile or CI pipeline yet — run locally with `uvicorn` today

## License

<!-- TODO: no LICENSE file exists yet. MIT is the common default for a solo portfolio project — add a LICENSE file if you want one. -->

## Contact

<!-- TODO: add your name, GitHub profile link, and LinkedIn -->
- GitHub: [Kaung31](https://github.com/Kaung31)
- Name: TODO
- LinkedIn: TODO
