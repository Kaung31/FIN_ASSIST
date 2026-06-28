"""Canned outputs for MOCK_MODE.

All three Azure services are mocked here (none have a free local equivalent):
  * Document Intelligence -> a fixed ``ParsedDocument`` (two financial tables with current+prior
                             columns, narrative text, one figure reference),
  * Azure AI Search       -> a fake hit list (mock_search_results) built from that document,
  * Azure OpenAI          -> schema-valid canned chat/vision/embedding responses.

The Python math (guardrail) and chunking run for real against these fixtures, so the pipeline
and tests are exercised end-to-end at zero cost — but real retrieval requires the cloud.
"""

from __future__ import annotations

import json

from finassist.analysis.models import Figure, FinancialTable, ParsedDocument

MOCK_DOC_ID = "mock-acme-fy2025-q2"


def _to_markdown(rows: list[list[str]]) -> str:
    header, *body = rows
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


_INCOME_ROWS = [
    ["($ in millions, except per share)", "Q2 FY2025", "Q2 FY2024"],
    ["Total revenue", "1,250", "1,000"],
    ["Cost of revenue", "500", "450"],
    ["Gross profit", "750", "550"],
    ["Operating expenses", "400", "300"],
    ["Operating income", "350", "250"],
    ["Net income", "300", "220"],
    ["Diluted EPS", "1.20", "0.88"],
]

_SEGMENT_ROWS = [
    ["Segment", "Q2 FY2025", "Q2 FY2024"],
    ["Cloud", "800", "600"],
    ["Devices", "450", "400"],
]

_NARRATIVE = (
    "Acme Corp reported total revenue of $1,250 million for the second quarter of fiscal 2025, "
    "an increase of 25% year over year, driven primarily by continued strength in the Cloud "
    "segment. Operating income rose to $350 million from $250 million in the prior-year quarter. "
    "Operating expenses increased to $400 million, reflecting higher investment in research and "
    "development and go-to-market capacity.\n\n"
    "Management reaffirmed full-year fiscal 2025 guidance of $5.0 billion to $5.2 billion in total "
    "revenue and expects continued margin expansion in the Cloud segment. These statements are "
    "forward-looking and reflect the company's current expectations as of the reporting date."
)


def mock_parsed_document(source_name: str = "acme_fy2025_q2.pdf") -> ParsedDocument:
    """Canned LlamaParse output: two financial tables, narrative text, one figure ref."""
    return ParsedDocument(
        doc_id=MOCK_DOC_ID,
        source_name=source_name,
        company="Acme Corp",
        fiscal_period="FY2025 Q2",
        filing_date="2025-03-30",
        page_count=4,
        raw_text=_NARRATIVE,
        tables=[
            FinancialTable(
                page=2,
                caption="Condensed Consolidated Statements of Operations",
                rows=_INCOME_ROWS,
                markdown=_to_markdown(_INCOME_ROWS),
            ),
            FinancialTable(
                page=3,
                caption="Revenue by Segment",
                rows=_SEGMENT_ROWS,
                markdown=_to_markdown(_SEGMENT_ROWS),
            ),
        ],
        figures=[Figure(page=4, bbox=(72.0, 120.0, 520.0, 360.0))],
    )


# ── LLM / VLM canned responses ────────────────────────────────────────────────


def mock_chart_description() -> str:
    return (
        "Bar chart titled 'Quarterly Total Revenue'. Total revenue rises from $1,000 million in "
        "Q2 FY2024 to $1,250 million in Q2 FY2025, with the Cloud segment as the largest "
        "contributor in both periods."
    )


def mock_summary_json() -> str:
    """A schema-valid EarningsSummary JSON.

    The prose fields (headline, guidance, notable_items, segment names) are used as-is; the
    numeric ``key_metrics`` are recomputed deterministically by the summarizer, so numbers
    never originate from the LLM.
    """
    return json.dumps(
        {
            "company": "Acme Corp",
            "fiscal_period": "FY2025 Q2",
            "headline": (
                "Acme Corp grew Q2 FY2025 revenue 25% year over year to $1,250M, with operating "
                "income up 40%, led by the Cloud segment."
            ),
            "key_metrics": [],
            "segments": [{"segment": "Cloud"}, {"segment": "Devices"}],
            "guidance": [
                "Management reaffirmed full-year FY2025 revenue guidance of $5.0B to $5.2B and "
                "expects continued Cloud margin expansion."
            ],
            "notable_items": [
                "Operating expenses rose to $400M, reflecting higher R&D and go-to-market "
                "investment."
            ],
        }
    )


def mock_anomaly_explanation() -> str:
    return (
        "Operating expenses increased year over year, outpacing revenue growth in percentage "
        "terms. The filing attributes higher spending to increased investment in research and "
        "development and go-to-market capacity. This describes the reported figures and is not "
        "investment advice."
    )


def mock_qa_answer() -> str:
    return (
        "According to the filing, Acme Corp reported total revenue of $1,250 million for Q2 "
        "FY2025, up 25% from $1,000 million in the prior-year quarter, driven primarily by the "
        "Cloud segment."
    )


def mock_chat_response(system: str, user: str, *, json_mode: bool = False) -> str:
    """Route a chat call to the right canned response based on the system prompt's intent."""
    s = system.lower()
    if "anomal" in s:
        return mock_anomaly_explanation()
    if "summar" in s or json_mode:
        return mock_summary_json()
    return mock_qa_answer()


# ── Azure AI Search canned result set ─────────────────────────────────────────


def mock_search_results(
    query: str,
    *,
    top: int = 8,
    doc_id: str | None = None,
    company: str | None = None,
    fiscal_period: str | None = None,
):
    """A fake AI Search hit list built from the canned document.

    Tables come first (so the math guardrail finds the numbers), then the figure and narrative.
    A non-matching company filter returns nothing, so grounding/refusal is exercised.
    """
    from finassist.analysis.models import RetrievedChunk
    from finassist.chunking.chunker import chunk

    if company and company.strip().lower() != "acme corp":
        return []

    doc = mock_parsed_document()
    doc.figures[0].description = mock_chart_description()

    results = [
        RetrievedChunk(
            chunk=c,
            score=round(1.0 - index * 0.01, 4),
            reranker_score=round(3.0 - index * 0.05, 4),
        )
        for index, c in enumerate(chunk(doc))
    ]
    return results[:top]
