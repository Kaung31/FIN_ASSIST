"""Azure OpenAI text embeddings (batched).

Embeds chunk/query text via the Azure OpenAI embedding deployment (text-embedding-3-small = 1536).
Returns mock vectors of the configured dimension in MOCK_MODE (the LLMClient short-circuits).
The output dimension must equal settings.embedding_dimensions (the AI Search index vector dim).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from finassist.analysis.models import Chunk

logger = logging.getLogger(__name__)

_BATCH = 16  # keep request sizes modest


@lru_cache(maxsize=1)
def _client():
    from finassist.llm.client import LLMClient

    return LLMClient()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed passages."""
    if not texts:
        return []
    client = _client()
    out: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        out.extend(client.embed(texts[start : start + _BATCH]))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return embed_texts([text])[0]


def _embed_text(chunk: Chunk) -> str:
    """Text actually sent to the embedder: a [company | period | statement] context header +
    the chunk content. Anchors retrieval to company/period (fixes wrong-company matches). The
    header is for retrieval only — the stored content and the numbers path are unchanged.
    """
    label = chunk.section or chunk.content_type
    parts = [p for p in (chunk.company, chunk.fiscal_period, label) if p]
    if not parts:
        return chunk.content
    return f"[{' | '.join(parts)}]\n{chunk.content}"


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Fill ``chunk.vector`` for each chunk in place; returns the same list."""
    vectors = embed_texts([_embed_text(chunk) for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.vector = vector
    return chunks
