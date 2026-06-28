"""Produce a structured EarningsSummary from retrieved context (grounded, no invented numbers).

Numbers (key metrics, segment revenue, YoY deltas) are computed deterministically by
``math_guardrail`` from the retrieved tables. The LLM only writes the prose framing (headline,
guidance restatement, notable items). Every metric carries a citation back to its source chunk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from finassist.analysis import math_guardrail
from finassist.analysis.models import Citation, EarningsSummary, RetrievedChunk
from finassist.llm.client import LLMClient
from finassist.search import retriever

logger = logging.getLogger(__name__)

_SYSTEM = (Path(__file__).parents[3] / "config/prompts/summary.txt").read_text()
_RETRIEVAL_QUERY = (
    "income statement total revenue net income operating income gross profit diluted EPS "
    "margins guidance outlook segment results"
)


def summarize(
    company: str | None,
    fiscal_period: str | None,
    llm: LLMClient,
    *,
    top: int = 12,
) -> EarningsSummary:
    """Retrieve financial statements + MD&A, compute numbers in Python, prose from the LLM."""
    retrieved = retriever.search(
        _RETRIEVAL_QUERY,
        top=top,
        company=company,
        fiscal_period=fiscal_period,
    )
    if not retriever.passes_relevance_gate(retrieved):
        retrieved = []  # low-relevance → no numbers rather than summarizing from noise

    tables = math_guardrail.tables_with_citations_from_retrieved(retrieved)
    key_metrics = math_guardrail.key_metrics_from_tables(tables, period=fiscal_period)
    segments = math_guardrail.segments_from_tables(tables, period=fiscal_period)

    prose = _safe_json(llm.chat(_SYSTEM, _user_prompt(retrieved), json_mode=True))

    return EarningsSummary(
        company=company or prose.get("company"),
        fiscal_period=fiscal_period or prose.get("fiscal_period"),
        headline=prose.get("headline", ""),
        key_metrics=key_metrics,
        segments=segments,
        guidance=prose.get("guidance", []),
        notable_items=prose.get("notable_items", []),
        citations=_dedupe_citations(c for _, c in tables),
    )


def _user_prompt(retrieved: list[RetrievedChunk]) -> str:
    return (
        "Summarise the quarter from the context below. Return JSON for the EarningsSummary "
        "schema; you only need to provide: company, fiscal_period, headline, guidance (list of "
        "company-stated statements), notable_items (list). Numbers are handled separately.\n\n"
        f"Context passages:\n{_format_context(retrieved)}"
    )


def _format_context(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.chunk_id} | p{hit.chunk.page} | {hit.chunk.section}] {hit.chunk.content}"
        for hit in retrieved
    )


def _safe_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        logger.warning("Summary LLM did not return valid JSON; using empty prose.")
        return {}


def _dedupe_citations(citations) -> list[Citation]:  # noqa: ANN001 - iterable of Citation
    seen, out = set(), []
    for citation in citations:
        if citation.chunk_id not in seen:
            seen.add(citation.chunk_id)
            out.append(citation)
    return out
