"""Write embedded chunks into Azure AI Search.

Embeds the chunks (Azure OpenAI) and uploads them as documents matching the index schema.
No-op in MOCK_MODE (there is no free local AI Search), returning the would-be count.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from finassist.analysis.models import Chunk
from finassist.embeddings import embedder
from finassist.search import clients

logger = logging.getLogger(__name__)


def upload(chunks: list[Chunk]) -> int:
    """Embed + upload chunks to the configured index. Returns the count indexed."""
    if not chunks:
        return 0

    if get_settings().mock_mode:
        logger.info("MOCK_MODE: pretending to index %d chunks", len(chunks))
        return len(chunks)

    embedder.embed_chunks(chunks)
    client = clients.get_search_client()
    results = client.upload_documents(documents=[_to_document(c) for c in chunks])
    succeeded = sum(1 for r in results if r.succeeded)
    if succeeded != len(chunks):
        logger.warning("Indexed %d/%d documents", succeeded, len(chunks))
    return succeeded


def _to_document(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "content": chunk.content,
        "page": chunk.page,
        "section": chunk.section,
        "content_type": chunk.content_type,
        "company": chunk.company,
        "fiscal_period": chunk.fiscal_period,
        "filing_date": chunk.filing_date,
        "content_vector": chunk.vector,
    }
