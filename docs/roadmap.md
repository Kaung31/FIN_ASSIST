# Build Roadmap

> **Note:** The project briefly used a LlamaParse + Qdrant + Cohere stack, then switched **back to
> the all-Azure stack** (Document Intelligence + AI Search + Azure OpenAI) described here. That
> build is implemented and runs mock-first with `pytest` green. The phase list below is the
> original Azure plan and matches the current architecture. See [`../README.md`](../README.md),
> [`../CLAUDE.md`](../CLAUDE.md), and [`../AZURE_SETUP.md`](../AZURE_SETUP.md).

Build bottom-up so each layer is verifiable before the next depends on it. Each phase ends with a
green test or a runnable command. Tell Claude Code to do one phase at a time.

## Phase 0 — Foundations (no Azure calls) ✅
- [x] `config/settings.py` loads `.env` (already scaffolded). Add a quick `python -c` check.
- [x] `analysis/models.py` — already defined. Confirm it imports cleanly.
- [x] `pip install -e ".[dev]"`, `ruff check`, `pytest` (skipped tests pass).

## Phase 1 — Clients ✅ (code complete; real-Azure run needs creds in `.env`)
- [x] `llm/client.py` — verify it constructs against your Azure OpenAI resource. Smoke-test
      `chat`, `embed`, `describe_image` behind a `@pytest.mark.integration` test.
      → `tests/test_llm_client.py`; run with `pytest --run-integration`.
- [x] `search/index_schema.py` + `scripts/create_index.py` — create the real index. Confirm in the
      Azure portal that the vector field dims match `EMBEDDING_DIMENSIONS`.
      → `create_index.py` wired to `SearchIndexClient.create_or_update_index` via
      `search/clients.py`; schema pinned by `tests/test_index_schema.py`.

## Phase 2 — Ingestion (the multimodal core)
- [ ] `ingestion/document_parser.py` — run `prebuilt-layout` on a sample PDF; assert tables and
      figure regions come back.
- [ ] `ingestion/table_extractor.py` — reconstruct grids + markdown; unit-test on a known table.
- [ ] `ingestion/figure_extractor.py` — crop figures to PNG; eyeball the output.
- [ ] `multimodal/chart_describer.py` — describe a cropped chart; sanity-check the description.

## Phase 3 — Index build
- [ ] `chunking/chunker.py` — tables/figures become whole chunks; narrative windows correctly.
      Implement `test_chunker.py`.
- [ ] `embeddings/embedder.py` — vectors have the right dimension.
- [ ] `search/indexer.py` — upload chunks; confirm document count in the index.
- [ ] `scripts/ingest_local.py` — full offline pipeline on one PDF, end to end.

## Phase 4 — Retrieval
- [ ] `search/retriever.py` — hybrid + semantic query with company/period filter. Manually verify
      that questions return the right pages. This is where you spend time tuning relevance.

## Phase 5 — Analysis features
- [ ] `analysis/summarizer.py` — structured `EarningsSummary`; verify every number traces to a
      source chunk (null when absent).
- [ ] `analysis/anomaly_detector.py` — implement `detect()` (deterministic) first; pin it with
      `test_anomaly_detector.py`. Then `explain()`.
- [ ] `qa/rag_chat.py` — grounded answers + citations; verify it refuses when context is missing.

## Phase 6 — Serve + demo
- [ ] `api/routes/*` — wire each route to its module; share one `LLMClient` via a dependency.
- [ ] `ui/streamlit_app.py` — upload → summary/anomalies/Q&A tabs calling the API.

## Phase 7 — Hardening (optional, strengthens the demo)
- [ ] Cache parsed documents + their tables so anomalies/summaries don't re-parse.
- [ ] Add an eval set of Q&A pairs; measure retrieval hit-rate and answer groundedness.
- [ ] Switch local keys → Managed Identity (`USE_AAD_AUTH=true`) for a production story.
- [ ] Rate-limit / cost-guard the embedding and chat calls.

## Demo script (for a fintech audience)
1. Upload a real earnings PDF → show parsed tables + a described chart.
2. Show the structured summary (key metrics + YoY, each with a page citation).
3. Show anomalies (e.g. a margin swing) with a grounded, cited explanation.
4. Ask 2–3 nuanced questions → grounded answers with citations; ask one the doc can't answer to
   show it declines instead of hallucinating.
