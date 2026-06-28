"""Detect period-over-period anomalies. Numbers are computed in Python; the LLM only explains.

Pipeline:
  1. ``detect`` computes abs_delta and pct_change deterministically from extracted tables and
     flags rows by threshold + severity (all math lives in ``analysis.math_guardrail``).
  2. ``explain`` asks the LLM to write a grounded explanation per anomaly using the anomaly
     prompt + retrieved context — it must NOT alter any numbers.

This separation is the guardrail against hallucinated financials.
"""

from __future__ import annotations

from pathlib import Path

from finassist.analysis import math_guardrail
from finassist.analysis.models import (
    Anomaly,
    AnomalyReport,
    Citation,
    FinancialTable,
    RetrievedChunk,
)
from finassist.llm.client import LLMClient

_SYSTEM = (Path(__file__).parents[3] / "config/prompts/anomaly.txt").read_text()


def detect(
    tables: list[FinancialTable],
    current_period: str,
    prior_period: str,
    *,
    threshold_pct: float = math_guardrail.DEFAULT_THRESHOLD_PCT,
    citations: list[Citation | None] | None = None,
) -> list[Anomaly]:
    """Compute anomalies deterministically from extracted tables."""
    return math_guardrail.compute_anomalies(
        tables,
        current_period,
        prior_period,
        threshold_pct=threshold_pct,
        citations=citations,
    )


def explain(
    anomalies: list[Anomaly],
    llm: LLMClient,
    *,
    retrieved: list[RetrievedChunk] | None = None,
    company: str | None = None,
) -> AnomalyReport:
    """Attach grounded explanations without changing any numbers."""
    context = _format_context(retrieved or [])
    for anomaly in anomalies:
        user = (
            "Explain this pre-computed anomaly (the numbers are final — do not change them):\n"
            f"- Metric: {anomaly.metric}\n"
            f"- {anomaly.prior_period}: {anomaly.prior_value}\n"
            f"- {anomaly.current_period}: {anomaly.current_value}\n"
            f"- Absolute change: {anomaly.abs_delta}\n"
            f"- Percent change: {anomaly.pct_change}%\n\n"
            f"Context passages:\n{context or '(none provided)'}\n"
        )
        anomaly.explanation = llm.chat(_SYSTEM, user).strip()
    return AnomalyReport(company=company, anomalies=anomalies)


def report_from_retrieved(
    retrieved: list[RetrievedChunk],
    llm: LLMClient,
    *,
    company: str | None = None,
    current_period: str | None = None,
    prior_period: str | None = None,
    threshold_pct: float = math_guardrail.DEFAULT_THRESHOLD_PCT,
) -> AnomalyReport:
    """Reconstruct tables from retrieved chunks, detect anomalies, then explain them.

    Period labels default to the current/prior column headers of the first table.
    """
    pairs = math_guardrail.tables_with_citations_from_retrieved(retrieved)
    tables = [table for table, _ in pairs]
    citations: list[Citation | None] = [citation for _, citation in pairs]

    if (not current_period or not prior_period) and tables:
        header = tables[0].rows[0]
        current_period = current_period or (header[1] if len(header) > 1 else "current period")
        prior_period = prior_period or (header[2] if len(header) > 2 else "prior period")

    anomalies = detect(
        tables,
        current_period or "current period",
        prior_period or "prior period",
        threshold_pct=threshold_pct,
        citations=citations,
    )
    return explain(anomalies, llm, retrieved=retrieved, company=company)


def _format_context(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.chunk_id} | p{hit.chunk.page} | {hit.chunk.section}] {hit.chunk.content}"
        for hit in retrieved
    )
