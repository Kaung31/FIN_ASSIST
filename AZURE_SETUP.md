# AZURE_SETUP — going live

Everything works in **mock mode** with no keys and no cost. This guide is for switching to the
real all-Azure stack. You need **three Azure resources**.

> 💸 **Cost reminder:** Azure AI Search has no free local equivalent — a Standard-tier service
> costs **~$75/month while it exists**. Delete it between sessions to avoid idle charges (you can
> recreate the index in seconds with `scripts/create_index.py`).

## 1. Azure OpenAI

Create the resource, then **create two deployments** (Azure AI Foundry → Deployments → Deploy):
- a **chat model** (vision-capable: `gpt-4o-mini` or `gpt-4.1-mini`)
- an **embedding model**: `text-embedding-3-small` (1536-dim)

In `.env`:
```
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/   # BARE url — see rule below
AZURE_OPENAI_KEY=<key>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=<your chat deployment name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your embedding deployment name>
EMBEDDING_DIMENSIONS=1536
```

> ⚠️ **Endpoint rule (prevents a 404):** `AZURE_OPENAI_ENDPOINT` must end at `.openai.azure.com/`
> with **nothing after it** — never append `/openai`, `/openai/v1`, or `/deployments` (the SDK adds
> the path). The deployment names must match your Azure deployments **exactly** (a mismatch or a
> resource with no deployments gives `DeploymentNotFound`).

## 2. Azure AI Search

Create a **Standard tier (or above)** service — the semantic ranker isn't available on the free
tier. In `.env`:
```
AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net
AZURE_SEARCH_KEY=<admin key>
AZURE_SEARCH_INDEX_NAME=earnings-index
AZURE_SEARCH_USE_SEMANTIC_RANKER=true
```

## 3. Azure Document Intelligence

Create the resource. In `.env`:
```
AZURE_DOCINTEL_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCINTEL_KEY=<key>
```

## Go live

```bash
# 1. fill the keys above in .env, then flip the mode:
MOCK_MODE=false

# 2. create the search index (one-time; recreate after deleting the service):
python scripts/create_index.py

# 3. ingest a real earnings PDF and query it:
python scripts/ingest_local.py data/samples/your_earnings.pdf
uvicorn finassist.api.main:app --reload     # API at :8000/docs
streamlit run ui/streamlit_app.py           # demo UI
```

Bring the three services up one at a time — when something breaks (auth, response shape, deployment
name), only one variable changed. The vector dimension in the index **must equal**
`EMBEDDING_DIMENSIONS` (1536 for `text-embedding-3-small`); if you change the embedding model,
update `EMBEDDING_DIMENSIONS` and recreate the index (`python scripts/create_index.py`).

## Optional: tracing (Phoenix)

```bash
pip install '.[eval]'                         # Arize Phoenix (tracing), no key/tokens
python -m finassist.eval.phoenix_trace --demo # local Phoenix UI at :6006
```

## Optional: RAGAS evaluation (its OWN venv)

RAGAS needs `ragas<0.3` + the `langchain` 0.3 stack (it imports a module removed in
langchain-community ≥0.4). That langchain pin would clash with the core/Phoenix deps, so it lives
in a **dedicated venv**:

```bash
python3.12 -m venv .venv-ragas
.venv-ragas/bin/pip install -e ".[ragas]"
# mock mode: builds the dataset from canned mocks, no billing (skips the LLM judge)
MOCK_MODE=true .venv-ragas/bin/python -m finassist.eval.run_ragas --limit 5
# live mode: scores faithfulness / answer relevancy / context precision / recall (real tokens)
.venv-ragas/bin/python -m finassist.eval.run_ragas
```

Scores write to `eval/results/ragas_latest.json`. The judge is pinned to your Azure chat
deployment for reproducibility. Diagnostic: **low context precision → fix retrieval; low
faithfulness → fix the prompt/guardrails.**
