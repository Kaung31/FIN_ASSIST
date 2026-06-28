"""Section/table/figure chunking with metadata (flat — no parent-child).

Tables and described figures each become a single chunk (preserved whole, so the math guardrail
can reconstruct the numbers and figures stay citable). Narrative text is split into section-sized
windows on paragraph boundaries. Each chunk carries doc/company/period metadata plus a citable
page + section. Relevance is handled by the Azure AI Search semantic ranker, so we deliberately
keep chunking flat rather than building a parent-child hierarchy.
"""

from __future__ import annotations

import re
import uuid

from finassist.analysis.models import Chunk, ParsedDocument

_MAX_CHARS = 1000


def chunk(doc: ParsedDocument) -> list[Chunk]:
    """Turn a ParsedDocument into a flat list of retrieval chunks."""
    meta = {
        "doc_id": doc.doc_id,
        "company": doc.company,
        "fiscal_period": doc.fiscal_period,
        "filing_date": doc.filing_date,
    }
    chunks: list[Chunk] = []

    for table in doc.tables:
        chunks.append(
            Chunk(
                chunk_id=_new_id(),
                content=table.markdown,
                page=table.page,
                section=table.caption or "Financial table",
                content_type="table",
                **meta,
            )
        )

    for figure in doc.figures:
        if figure.description:  # figures are chunked once chart_describer has run
            chunks.append(
                Chunk(
                    chunk_id=_new_id(),
                    content=figure.description,
                    page=figure.page,
                    section="Figure",
                    content_type="figure",
                    **meta,
                )
            )

    for piece in _split_text(doc.raw_text):
        chunks.append(
            Chunk(
                chunk_id=_new_id(),
                content=piece,
                page=1,
                section="Narrative",
                content_type="text",
                **meta,
            )
        )

    return chunks


def _split_text(text: str, max_chars: int = _MAX_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if buffer and len(buffer) + len(paragraph) + 2 > max_chars:
            pieces.append(buffer)
            buffer = paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
    if buffer:
        pieces.append(buffer)
    return pieces


def _new_id() -> str:
    return str(uuid.uuid4())
