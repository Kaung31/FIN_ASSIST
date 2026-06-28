"""RAGAS evaluation of the RAG pipeline.

Metrics: faithfulness, answer relevancy, context precision, context recall. The judge is an LLM
(pinned to the Azure OpenAI deployment from settings), so LIVE scoring CONSUMES REAL TOKENS.

RAGAS pulls a version-sensitive langchain tree (it imports langchain_community.chat_models.vertexai,
removed in langchain-community >=0.4), so install it in its OWN venv:
    python3.12 -m venv .venv-ragas
    .venv-ragas/bin/pip install -e ".[ragas]"
    .venv-ragas/bin/python -m finassist.eval.run_ragas [--limit N]

In MOCK_MODE the query path runs against the canned mocks and the dataset is written out, but the
LLM-judge scoring is skipped (it needs live Azure) — so the script is testable without billing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make the repo-root `config` package importable when run as a file.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config.settings import get_settings
from finassist.llm.client import LLMClient
from finassist.qa import rag_chat
from finassist.search import retriever

logger = logging.getLogger(__name__)

GOLDEN_SET = Path(__file__).with_name("golden_set.jsonl")
RESULTS_PATH = Path(__file__).resolve().parents[3] / "eval" / "results" / "ragas_latest.json"


def load_golden(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in GOLDEN_SET.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def _require_ragas():
    """Import the RAGAS surface, or raise a clear error pointing at the dedicated venv."""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - optional, version-sensitive dependency
        raise RuntimeError(
            "RAGAS is unavailable. Install it in its own venv (langchain conflict): "
            "python3.12 -m venv .venv-ragas && .venv-ragas/bin/pip install -e '.[ragas]'. "
            f"(import error: {exc!r})"
        ) from exc
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return EvaluationDataset, evaluate, metrics


def build_samples(limit: int | None = None) -> list[dict]:
    """Run the real query path for each golden question → RAGAS sample dicts (+ eval metadata)."""
    llm = LLMClient()
    samples = []
    for row in load_golden(limit):
        company = row.get("expected_company")
        # Filter by company (the Task 3 metadata fix, P2). Period is left unfiltered for the
        # single-doc baseline — period/version scoping is Task 5 (filing-date awareness).
        hits = retriever.search(row["question"], top=8, company=company)
        response = rag_chat.answer(row["question"], llm, company=company)
        samples.append(
            {
                "user_input": row["question"],
                "response": response.answer,
                "retrieved_contexts": [h.chunk.content for h in hits],
                "reference": row["ground_truth"],
                # eval-only metadata (not passed to RAGAS):
                "_id": row.get("id"),
                "_category": row.get("category"),
                "_grounded": response.grounded,
                "_expected_refusal": row.get("expected_refusal", False),
                "_n_contexts": len(hits),
            }
        )
    return samples


def _azure_judge():
    """Pin the RAGAS judge to the Azure OpenAI deployment from settings (reproducible scores)."""
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    s = get_settings()
    chat = AzureChatOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        azure_deployment=s.azure_openai_chat_deployment,
        api_version=s.azure_openai_api_version,
        api_key=s.azure_openai_key,
        temperature=0,
    )
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=s.azure_openai_endpoint,
        azure_deployment=s.azure_openai_embedding_deployment,
        api_version=s.azure_openai_api_version,
        api_key=s.azure_openai_key,
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(embeddings)


_RAGAS_KEYS = ("user_input", "response", "retrieved_contexts", "reference")


def run(limit: int | None = None):
    """Build the dataset from the query path and (live only) score it with RAGAS."""
    EvaluationDataset, evaluate, metrics = _require_ragas()
    settings = get_settings()
    samples = build_samples(limit)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if settings.mock_mode:
        refusals = sum(1 for s in samples if not s["_grounded"])
        RESULTS_PATH.write_text(
            json.dumps({"mock_mode": True, "n": len(samples), "samples": samples}, indent=2)
        )
        logger.warning("MOCK_MODE: skipping LLM-judge scoring (needs live Azure).")
        print(
            f"MOCK_MODE: built {len(samples)} samples ({refusals} refusals) → {RESULTS_PATH}\n"
            "Run live (MOCK_MODE=false) to compute faithfulness / relevancy / precision / recall."
        )
        return None

    dataset = EvaluationDataset.from_list([{k: s[k] for k in _RAGAS_KEYS} for s in samples])
    judge_llm, judge_embeddings = _azure_judge()
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)

    scores = {k: float(v) for k, v in result.to_pandas().mean(numeric_only=True).to_dict().items()}
    RESULTS_PATH.write_text(
        json.dumps({"mock_mode": False, "n": len(samples), "scores": scores}, indent=2)
    )
    _print_table(scores)
    return result


def _print_table(scores: dict) -> None:
    print("\nRAGAS scores")
    print("-" * 40)
    for name, value in scores.items():
        print(f"  {name:<32} {value:.4f}")
    print(f"\nwrote {RESULTS_PATH}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the golden set.")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    args = parser.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
