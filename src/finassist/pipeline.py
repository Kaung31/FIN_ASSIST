"""End-to-end ingestion pipeline, shared by the API route and the CLI script.

parse (Document Intelligence) → crop figures → describe charts → chunk → embed → index (AI Search).
In MOCK_MODE every Azure step is canned; the data still flows through the real chunking/validation.
"""

from __future__ import annotations

import logging

from finassist.chunking import chunker
from finassist.ingestion import figure_extractor, parser
from finassist.llm.client import LLMClient
from finassist.multimodal import chart_describer
from finassist.search import indexer

logger = logging.getLogger(__name__)


def ingest(
    source: str | bytes,
    source_name: str,
    llm: LLMClient,
    *,
    figures_dir: str = ".figures_tmp",
) -> dict:
    """Run the full ingestion pipeline for one document. Returns a summary of counts."""
    pdf_bytes = source if isinstance(source, bytes) else None

    doc = parser.parse(source, source_name=source_name)
    if pdf_bytes is not None:
        figure_extractor.crop_figures(pdf_bytes, doc.figures, figures_dir)
    chart_describer.describe(doc.figures, llm)

    chunks = chunker.chunk(doc)
    indexed = indexer.upload(chunks)

    result = {
        "doc_id": doc.doc_id,
        "company": doc.company,
        "fiscal_period": doc.fiscal_period,
        "pages": doc.page_count,
        "tables": len(doc.tables),
        "figures": len(doc.figures),
        "chunks": len(chunks),
        "indexed": indexed,
    }
    logger.info("Ingested %s: %s", source_name, result)
    return result
