"""Typed data contract for FinAssist.

These models define (a) the intermediate objects passed between pipeline stages and
(b) the structured outputs returned to callers. The LLM is prompted to emit JSON matching
the *output* models (EarningsSummary, AnomalyReport, QAResponse).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Ingestion / retrieval data contract
# ─────────────────────────────────────────────────────────────────────────────


class FinancialTable(BaseModel):
    """A table extracted by Document Intelligence, preserved in structured form."""

    page: int
    caption: str | None = None
    # rows[r][c] cell text; keep raw so downstream numeric parsing is explicit and auditable.
    rows: list[list[str]]
    markdown: str = Field(description="Markdown rendering of the table for LLM context.")


class Figure(BaseModel):
    """A chart/figure region. `description` is filled by the vision model in multimodal/."""

    page: int
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="(x0, y0, x1, y1) on the page, for cropping."
    )
    image_path: str | None = None
    description: str | None = None


class ParsedDocument(BaseModel):
    """Output of ingestion.parser.parse()."""

    doc_id: str
    source_name: str
    company: str | None = None
    fiscal_period: str | None = None  # e.g. "FY2025 Q2"
    filing_date: str | None = None  # period-end / filing date, for version-aware citations
    page_count: int
    raw_text: str
    tables: list[FinancialTable] = []
    figures: list[Figure] = []


class Chunk(BaseModel):
    """A retrieval unit produced by chunking.chunker.chunk() (flat: table / figure / narrative)."""

    chunk_id: str
    doc_id: str
    content: str
    page: int
    section: str | None = None  # e.g. "MD&A", "Risk Factors", "Income Statement"
    content_type: str = "text"  # one of: text | table | figure (== source_type)
    company: str | None = None
    fiscal_period: str | None = None
    filing_date: str | None = None  # period-end / filing date of the source document
    vector: list[float] | None = None  # filled by embeddings.embedder.embed_chunks()


class RetrievedChunk(BaseModel):
    """A chunk returned from search, with relevance scores."""

    chunk: Chunk
    score: float
    reranker_score: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Citations (shared by all outputs)
# ─────────────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    chunk_id: str
    doc_id: str  # document/version identifier
    page: int
    section: str | None = None
    filing_date: str | None = None  # which filing/version + date the fact came from


# ─────────────────────────────────────────────────────────────────────────────
# Structured summary output
# ─────────────────────────────────────────────────────────────────────────────


class FinancialMetric(BaseModel):
    """A single reported metric. value is null when the filing doesn't state it."""

    name: str  # e.g. "Total revenue"
    value: float | None = None
    unit: str | None = None  # e.g. "USD millions"
    period: str | None = None  # e.g. "FY2025 Q2"
    prior_value: float | None = None
    yoy_change_pct: float | None = None
    citation: Citation | None = None


class SegmentResult(BaseModel):
    segment: str
    revenue: FinancialMetric | None = None
    operating_income: FinancialMetric | None = None


class EarningsSummary(BaseModel):
    """Top-level structured summary. All numbers must come from the source document."""

    company: str | None = None
    fiscal_period: str | None = None
    headline: str = Field(description="One-sentence neutral summary of the quarter.")
    key_metrics: list[FinancialMetric] = []
    segments: list[SegmentResult] = []
    guidance: list[str] = Field(
        default=[], description="Forward guidance AS STATED BY THE COMPANY, each with a citation."
    )
    notable_items: list[str] = []
    citations: list[Citation] = []


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection output
# ─────────────────────────────────────────────────────────────────────────────


class AnomalySeverity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class Anomaly(BaseModel):
    """A flagged period-over-period change. The numbers are computed in Python, not by the LLM."""

    metric: str
    current_value: float
    prior_value: float
    abs_delta: float
    pct_change: float
    current_period: str
    prior_period: str
    severity: AnomalySeverity
    explanation: str | None = None  # filled by the LLM, grounded in context only
    citation: Citation | None = None


class AnomalyReport(BaseModel):
    company: str | None = None
    anomalies: list[Anomaly] = []
    method_note: str = Field(
        default="Deltas computed deterministically from extracted figures; explanations are "
        "grounded in the source document and are not investment advice."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Q&A output
# ─────────────────────────────────────────────────────────────────────────────


class QAResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = []
    grounded: bool = Field(
        description="False when retrieval found nothing relevant and the model declined to answer."
    )
