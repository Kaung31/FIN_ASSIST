"""Deterministic financial math — the anti-hallucination guardrail.

ALL numbers (metric values, YoY deltas, anomaly magnitudes) are computed here in Python from
parsed table cells. The LLM never originates a number; it only selects / organises / explains.
Table convention: column 1 is the current period, column 2 is the prior period.
"""

from __future__ import annotations

from finassist.analysis.models import (
    Anomaly,
    AnomalySeverity,
    Citation,
    FinancialMetric,
    FinancialTable,
    RetrievedChunk,
    SegmentResult,
)
from finassist.ingestion.table_extractor import parse_number, tables_from_markdown

DEFAULT_THRESHOLD_PCT = 20.0
_MEDIUM_PCT = 20.0
_HIGH_PCT = 50.0
DEFAULT_UNIT = "USD millions"

# Income-statement rows surfaced as headline key metrics (lowercased exact match).
_KEY_METRICS = {
    "total revenue",
    "revenue",
    "gross profit",
    "operating income",
    "net income",
    "diluted eps",
    "basic eps",
    "earnings per share",
}


def yoy_pct(current: float | None, prior: float | None) -> float | None:
    """Year-over-year percent change, or None when it can't be computed."""
    if current is None or prior is None or prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 4)


def severity(pct_change: float) -> AnomalySeverity:
    magnitude = abs(pct_change)
    if magnitude >= _HIGH_PCT:
        return AnomalySeverity.high
    if magnitude >= _MEDIUM_PCT:
        return AnomalySeverity.medium
    return AnomalySeverity.low


def _current_prior(row: list[str]) -> tuple[float | None, float | None]:
    current = parse_number(row[1]) if len(row) > 1 else None
    prior = parse_number(row[2]) if len(row) > 2 else None
    return current, prior


def is_segment_table(table: FinancialTable) -> bool:
    return "segment" in (table.caption or "").lower()


def metrics_from_table(
    table: FinancialTable,
    citation: Citation | None = None,
    *,
    period: str | None = None,
    unit: str = DEFAULT_UNIT,
) -> list[FinancialMetric]:
    """Build a FinancialMetric for every numeric data row in a table."""
    metrics = []
    for row in table.rows[1:]:
        if not row:
            continue
        current, prior = _current_prior(row)
        if current is None:
            continue
        metrics.append(
            FinancialMetric(
                name=row[0].strip(),
                value=current,
                unit=unit,
                period=period,
                prior_value=prior,
                yoy_change_pct=yoy_pct(current, prior),
                citation=citation,
            )
        )
    return metrics


def key_metrics_from_tables(
    tables_with_citations: list[tuple[FinancialTable, Citation | None]],
    *,
    period: str | None = None,
) -> list[FinancialMetric]:
    """Curated headline metrics (revenue, operating/net income, EPS) from non-segment tables."""
    metrics = []
    for table, citation in tables_with_citations:
        if is_segment_table(table):
            continue
        for metric in metrics_from_table(table, citation, period=period):
            if metric.name.lower() in _KEY_METRICS:
                metrics.append(metric)
    return metrics


def segments_from_tables(
    tables_with_citations: list[tuple[FinancialTable, Citation | None]],
    *,
    period: str | None = None,
) -> list[SegmentResult]:
    """Per-segment revenue (deterministic) from any segment table."""
    segments = []
    for table, citation in tables_with_citations:
        if not is_segment_table(table):
            continue
        for row in table.rows[1:]:
            if not row:
                continue
            current, prior = _current_prior(row)
            if current is None:
                continue
            revenue = FinancialMetric(
                name=f"{row[0].strip()} revenue",
                value=current,
                unit=DEFAULT_UNIT,
                period=period,
                prior_value=prior,
                yoy_change_pct=yoy_pct(current, prior),
                citation=citation,
            )
            segments.append(SegmentResult(segment=row[0].strip(), revenue=revenue))
    return segments


def compute_anomalies(
    tables: list[FinancialTable],
    current_period: str,
    prior_period: str,
    *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    citations: list[Citation | None] | None = None,
) -> list[Anomaly]:
    """Flag rows whose |YoY %| ≥ threshold. All magnitudes computed here, not by the LLM."""
    anomalies = []
    for index, table in enumerate(tables):
        citation = citations[index] if citations and index < len(citations) else None
        for row in table.rows[1:]:
            if not row:
                continue
            current, prior = _current_prior(row)
            pct = yoy_pct(current, prior)
            if pct is None or abs(pct) < threshold_pct:
                continue
            anomalies.append(
                Anomaly(
                    metric=row[0].strip(),
                    current_value=current,
                    prior_value=prior,
                    abs_delta=round(current - prior, 6),
                    pct_change=pct,
                    current_period=current_period,
                    prior_period=prior_period,
                    severity=severity(pct),
                    citation=citation,
                )
            )
    return anomalies


def tables_with_citations_from_retrieved(
    retrieved: list[RetrievedChunk],
) -> list[tuple[FinancialTable, Citation]]:
    """Reconstruct (FinancialTable, Citation) pairs from retrieved table chunks."""
    pairs = []
    for hit in retrieved:
        if hit.chunk.content_type != "table":
            continue
        citation = Citation(
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            page=hit.chunk.page,
            section=hit.chunk.section,
            filing_date=hit.chunk.filing_date,
        )
        for table in tables_from_markdown(
            hit.chunk.content, page=hit.chunk.page, caption=hit.chunk.section
        ):
            pairs.append((table, citation))
    return pairs
