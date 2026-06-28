# FinAssist — Multimodal Financial Report Assistant

Upload earnings PDFs (text, financial tables, and charts) and get **structured summaries**,
**period-over-period anomaly detection**, and **grounded Q&A with citations** — built entirely on
**Azure**: Document Intelligence, Azure OpenAI, and Azure AI Search.

> ⚠️ This is a document-analysis tool. It summarizes and answers questions about what filings
> say. It does **not** provide investment advice or recommendations.

## Mock-first

The default mode (`MOCK_MODE=true`) mocks **all three Azure services** — Document Intelligence,
Azure OpenAI, and Azure AI Search — with canned data, so the full pipeline and tests run at
**zero cost, no keys**. Unlike a local vector store, AI Search and Document Intelligence have no
free local equivalent, so *real retrieval can only be tested against the cloud*. Add keys and set
`MOCK_MODE=false` to go live — see [`AZURE_SETUP.md`](AZURE_SETUP.md).

## Features

- **Multimodal ingestion** — Document Intelligence (`prebuilt-layout`) extracts narrative text and
  financial tables as markdown; figures are described by a vision model so they're searchable + citable.
- **Hybrid + semantic retrieval** — Azure AI Search combines BM25 + vector (RRF) re-ranked by the
  **semantic ranker** (which replaces a separate reranker), scoped by company/period filters.
- **Structured summaries** — typed JSON: revenue, net income, EPS, margins, guidance, and segment
  performance with YoY deltas. **Numbers are computed in Python**, never by the LLM, each cited.
- **Anomaly detection** — period-over-period deltas computed deterministically; the model only
  *explains* flagged outliers. No invented numbers.
- **Grounded Q&A (RAG)** — answers from retrieved context only, with citations; declines when the
  answer isn't in the documents.

## Stack

| Layer | Service |
|---|---|
| PDF parsing | Azure Document Intelligence (`prebuilt-layout`) |
| Embeddings | Azure OpenAI `text-embedding-3-small` (1536-dim) |
| Vector store + retrieval | Azure AI Search (hybrid BM25 + vector + semantic ranker) |
| LLM / VLM | Azure OpenAI (chat + vision) |
| Numbers | deterministic Python (`analysis/math_guardrail.py`) |
| Eval / tracing | Ragas + Arize Phoenix (`.[eval]` extra) |
| API / UI | FastAPI / Streamlit |

See [`docs/architecture.md`](docs/architecture.md). Three Azure resources are required for live
mode — provisioning steps in [`AZURE_SETUP.md`](AZURE_SETUP.md).

## Quickstart (no keys, mock mode)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                       # defaults to MOCK_MODE=true

python scripts/ingest_local.py data/samples/sample.pdf   # full pipeline, zero cost
pytest -q                                                # mock mode, no network
```

`pytest` always runs hermetically (forces mock mode), so it stays green even when `.env` is set to
live mode. Integration tests (real Azure) are opt-in: `pytest --run-integration`.

## Going live

Provision the three Azure resources (see [`AZURE_SETUP.md`](AZURE_SETUP.md)), fill the keys in
`.env`, set `MOCK_MODE=false`, then:

```bash
python scripts/create_index.py                          # create the AI Search index
python scripts/ingest_local.py data/samples/your.pdf
uvicorn finassist.api.main:app --reload                 # http://localhost:8000/docs
streamlit run ui/streamlit_app.py
```

> 💸 Azure AI Search costs ~$75/month while it exists — delete it between sessions and recreate the
> index with `scripts/create_index.py` when needed.

## License

Add your license here.
