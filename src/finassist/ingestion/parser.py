"""Parse a PDF into a ParsedDocument with Azure AI Document Intelligence (prebuilt-layout).

prebuilt-layout returns the document as markdown (text + tables) plus figure bounding regions.
We parse the markdown tables into FinancialTable and map figure regions to Figure (cropped later
by figure_extractor). In MOCK_MODE a canned ParsedDocument is returned (zero cost).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from config.settings import get_settings
from finassist.analysis.models import Figure, ParsedDocument
from finassist.ingestion.table_extractor import strip_tables, to_financial_tables

logger = logging.getLogger(__name__)

_PERIOD_RE = re.compile(
    r"(FY\s?20\d{2}\s?Q[1-4]|Q[1-4]\s?FY\s?20\d{2}|Q[1-4]\s?20\d{2})", re.IGNORECASE
)
_REGISTRANT_RE = re.compile(r"Exact name of (?:the )?[Rr]egistrant")
_TAG_RE = re.compile(r"<[^>]+>")
# Period-end / filing date on the cover, e.g. "quarterly period ended June 28, 2025".
_FILING_DATE_RE = re.compile(
    r"(?:period|year)\s+ended\s+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})", re.IGNORECASE
)


def parse(
    source: str | Path | bytes,
    *,
    source_name: str | None = None,
    doc_id: str | None = None,
    company: str | None = None,
    fiscal_period: str | None = None,
) -> ParsedDocument:
    """Parse ``source`` (a PDF path or raw bytes) into a ParsedDocument."""
    name = _resolve_name(source, source_name)

    if get_settings().mock_mode:
        logger.info("MOCK_MODE: returning canned ParsedDocument for %s", name)
        from finassist.mocks import fixtures

        return fixtures.mock_parsed_document(name)

    return _analyze(
        _read_bytes(source), name, doc_id=doc_id, company=company, fiscal_period=fiscal_period
    )


def _analyze(
    pdf_bytes: bytes,
    name: str,
    *,
    doc_id: str | None,
    company: str | None,
    fiscal_period: str | None,
) -> ParsedDocument:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import (
        AnalyzeDocumentRequest,
        DocumentContentFormat,
    )

    settings = get_settings()
    client = DocumentIntelligenceClient(
        endpoint=settings.azure_docintel_endpoint, credential=_credential()
    )
    poller = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        output_content_format=DocumentContentFormat.MARKDOWN,
    )
    result = poller.result()

    markdown = result.content or ""
    tables = to_financial_tables(result.tables)  # structured cells (DI markdown uses HTML tables)
    raw_text = strip_tables(markdown).strip()

    return ParsedDocument(
        doc_id=doc_id or _slug(name),
        source_name=name,
        company=company or _infer_company(raw_text),
        fiscal_period=fiscal_period or _infer_period(raw_text),
        filing_date=_infer_filing_date(raw_text),
        page_count=len(result.pages or []) or 1,
        raw_text=raw_text,
        tables=tables,
        figures=_figures(result),
    )


def _credential():
    settings = get_settings()
    if settings.use_aad_auth:
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()
    from azure.core.credentials import AzureKeyCredential

    return AzureKeyCredential(settings.azure_docintel_key)


def _figures(result) -> list[Figure]:  # noqa: ANN001 - DI AnalyzeResult
    figures = []
    for figure in result.figures or []:
        regions = figure.bounding_regions or []
        if not regions:
            continue
        region = regions[0]
        figures.append(Figure(page=region.page_number, bbox=_polygon_to_bbox(region.polygon)))
    return figures


def _polygon_to_bbox(polygon) -> tuple[float, float, float, float] | None:  # noqa: ANN001
    if not polygon:
        return None
    xs = [float(v) for v in polygon[0::2]]
    ys = [float(v) for v in polygon[1::2]]
    # DI layout coordinates for PDFs are in inches; PyMuPDF cropping expects points (72/inch).
    return (min(xs) * 72, min(ys) * 72, max(xs) * 72, max(ys) * 72)


def _read_bytes(source: str | Path | bytes) -> bytes:
    return source if isinstance(source, bytes) else Path(source).read_bytes()


def _resolve_name(source: str | Path | bytes, source_name: str | None) -> str:
    if source_name:
        return source_name
    if isinstance(source, (str, Path)):
        return Path(source).name
    return "document.pdf"


def _slug(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "document"


def _infer_period(text: str) -> str | None:
    match = _PERIOD_RE.search(text)
    return match.group(0).upper() if match else None


def _infer_company(text: str) -> str | None:
    """Extract the registrant/company name from the filing cover (e.g. 'Apple Inc.').

    Anchored to the '(Exact name of Registrant ...)' line every 10-Q/10-K cover carries. DI may
    wrap the name in layout tags (e.g. ``<figure>Apple Inc.</figure>``), so we strip tags and take
    the last name-like line before that anchor — giving the issuer exactly, which is what makes the
    company metadata filter work.
    """
    match = _REGISTRANT_RE.search(text)
    if not match:
        return None
    before = _TAG_RE.sub(" ", text[: match.start()]).rstrip().rstrip("(").rstrip()
    lines = [line.strip() for line in before.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1].rstrip(",").strip()
    if candidate and len(candidate) <= 80 and any(ch.isalpha() for ch in candidate):
        return candidate
    return None


def _infer_filing_date(text: str) -> str | None:
    """Extract the period-end / filing date from the cover (e.g. 'June 28, 2025')."""
    match = _FILING_DATE_RE.search(text)
    return match.group(1).strip() if match else None
